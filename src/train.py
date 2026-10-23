"""Training loop with AMP, warmup + cosine LR, gradient clipping and EMA."""

import argparse
import contextlib
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from .config import Config, load_config
from .data import build_loaders, load_datasets, seed_everything
from .model import build_model, count_parameters, inception_loss
from .utils import (
    AverageMeter,
    History,
    ModelEMA,
    build_param_groups,
    build_scheduler,
    device_report,
    format_duration,
    get_device,
    make_autocast,
    make_scaler,
    plot_curves,
    save_json,
)


class Trainer:
    def __init__(self, cfg: Config, device=None, verbose=True, write_history=True):
        self.cfg = cfg
        self.device = device or get_device()
        self.verbose = verbose

        seed_everything(cfg.train.seed)

        self.train_set, self.val_set, self.test_set, self.classes, self.data_root = (
            load_datasets(cfg)
        )
        self.train_loader, self.val_loader, self.test_loader = build_loaders(
            self.train_set, self.val_set, self.test_set, cfg
        )

        self.model = build_model(
            num_classes=len(self.classes),
            aux_weight=cfg.model.aux_weight,
            dropout=cfg.model.dropout,
            variant=cfg.model.variant,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            build_param_groups(self.model, cfg.train.weight_decay),
            lr=cfg.train.lr,
            betas=(0.9, 0.999),
        )

        steps_per_epoch = max(1, len(self.train_loader))
        self.scheduler = build_scheduler(
            self.optimizer,
            warmup_steps=cfg.train.warmup_epochs * steps_per_epoch,
            total_steps=cfg.train.epochs * steps_per_epoch,
            min_lr_ratio=cfg.train.min_lr_ratio,
        )

        self.scaler = make_scaler(cfg.train.amp, self.device)
        self.autocast_dtype = torch.float16
        self.ema = (
            ModelEMA(self.model, cfg.train.ema_decay) if cfg.train.use_ema else None
        )

        self.out_dir = Path(cfg.run.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Evaluation-only runs must not truncate the training log.
        self.history = History(self.out_dir / "history.csv") if write_history else None
        self.best_acc = 0.0
        self.best_epoch = -1

    def log(self, message):
        if self.verbose:
            print(message, flush=True)

    def summary(self):
        total, trainable = count_parameters(self.model)
        aux_params = (
            count_parameters(self.model.aux)[0] if self.model.aux is not None else 0
        )
        self.log(device_report(self.device))
        self.log(f"data root     : {self.data_root}")
        self.log(f"train / val / test : {len(self.train_set)} / "
                 f"{len(self.val_set)} / {len(self.test_set)}")
        self.log(f"classes       : {len(self.classes)}")
        self.log(f"variant       : {self.cfg.model.variant}")
        self.log(f"parameters    : {total:,} total, {trainable:,} trainable "
                 f"(aux head {aux_params:,})")
        self.log(f"batch size    : {self.cfg.data.batch_size} "
                 f"({len(self.train_loader)} steps/epoch)")
        self.log(f"epochs        : {self.cfg.train.epochs}  "
                 f"lr {self.cfg.train.lr:g} -> "
                 f"{self.cfg.train.lr * self.cfg.train.min_lr_ratio:g} cosine, "
                 f"{self.cfg.train.warmup_epochs} warmup")
        self.log(f"amp           : {self.scaler is not None}   "
                 f"ema           : {self.ema is not None}")
        self.log("")

    def run_epoch(self, loader, training, model=None):
        model = self.model if model is None else model
        model.train(training)

        loss_m = AverageMeter()
        main_m = AverageMeter()
        aux_m = AverageMeter()
        acc_m = AverageMeter()
        want_aux = model.aux is not None

        ctx = (
            make_autocast(self.device, self.autocast_dtype)
            if training
            else contextlib.nullcontext()
        )

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with ctx:
                outputs = model(images, with_aux=want_aux)
                loss, main, aux = inception_loss(
                    outputs, targets, self.cfg.model.aux_weight,
                    self.cfg.train.label_smoothing,
                )

            logits = outputs[0] if want_aux else outputs
            preds = logits.argmax(1)
            correct = (preds == targets).sum().item()

            loss_m.update(loss.item(), targets.size(0))
            main_m.update(main.item(), targets.size(0))
            aux_m.update(aux.item(), targets.size(0))
            acc_m.update(correct / targets.size(0), targets.size(0))

            if not training:
                continue

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if self.cfg.train.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_norm_(model.parameters(), self.cfg.train.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.train.grad_clip:
                    clip_grad_norm_(model.parameters(), self.cfg.train.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            if self.ema is not None:
                self.ema.update(model)

        return {
            "loss": loss_m.avg,
            "main_loss": main_m.avg,
            "aux_loss": aux_m.avg,
            "acc": acc_m.avg,
        }

    def fit(self):
        self.summary()
        cfg = self.cfg
        bad_epochs = 0

        for epoch in range(1, cfg.train.epochs + 1):
            started = time.time()
            train_stats = self.run_epoch(self.train_loader, training=True)

            eval_model = self.ema.ema if self.ema is not None else self.model
            val_stats = self.evaluate(self.val_loader, model=eval_model)

            elapsed = time.time() - started
            lr = self.optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "train_loss": round(train_stats["loss"], 5),
                "train_main_loss": round(train_stats["main_loss"], 5),
                "train_aux_loss": round(train_stats["aux_loss"], 5),
                "train_acc": round(train_stats["acc"], 5),
                "val_loss": round(val_stats["loss"], 5),
                "val_main_loss": round(val_stats["main_loss"], 5),
                "val_aux_loss": round(val_stats["aux_loss"], 5),
                "val_acc": round(val_stats["acc"], 5),
                "lr": lr,
                "seconds": round(elapsed, 1),
            }
            self.history.add(row)

            improved = val_stats["acc"] > self.best_acc
            if improved:
                self.best_acc = val_stats["acc"]
                self.best_epoch = epoch
                bad_epochs = 0
                self.save_checkpoint(self.out_dir / "best.pt", epoch, val_stats["acc"])
            else:
                bad_epochs += 1

            tag = "*" if improved else " "
            self.log(
                f"epoch {epoch:3d}/{cfg.train.epochs}{tag} "
                f"train loss {train_stats['loss']:.4f} acc {train_stats['acc']:.4f} | "
                f"val loss {val_stats['loss']:.4f} acc {val_stats['acc']:.4f} | "
                f"lr {lr:.2e} | {format_duration(elapsed)}"
            )

            if cfg.train.patience and bad_epochs >= cfg.train.patience:
                self.log(f"\nno improvement for {bad_epochs} epochs, stopping early")
                break

        self.save_checkpoint(self.out_dir / "last.pt", epoch, val_stats["acc"],
                             resumable=True)
        plot_curves(
            self.history.to_dict(),
            self.out_dir / "curves.png",
            title=f"{cfg.run.project_name} ({cfg.model.variant})",
        )
        save_json(cfg.to_dict(), self.out_dir / "config_used.json")

        self.log("")
        self.log(f"best val acc {self.best_acc:.4f} at epoch {self.best_epoch}")
        return self.history

    def evaluate(self, loader, model=None):
        model = self.model if model is None else model
        return self.run_epoch(loader, training=False, model=model)

    def save_checkpoint(self, path, epoch, val_acc, resumable=False):
        payload = {
            "epoch": epoch,
            "val_acc": val_acc,
            "model_state": self.model.state_dict(),
            "classes": self.classes,
            "config": self.cfg.to_dict(),
            "torch_version": torch.__version__,
        }
        if self.ema is not None:
            payload["ema_state"] = self.ema.state_dict()
        # Optimizer state roughly doubles the file, so only keep it where a
        # resumed run would actually need it.
        if resumable:
            payload["optimizer_state"] = self.optimizer.state_dict()
            payload["scheduler_state"] = self.scheduler.state_dict()
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path, use_ema=True):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state = payload.get("ema_state") if (use_ema and "ema_state" in payload) \
            else payload["model_state"]
        self.model.load_state_dict(state)
        self.model.eval()
        return payload


def main():
    parser = argparse.ArgumentParser(description="Train GoogLeNet on sports balls")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override a value, e.g. --set train.epochs=20")
    parser.add_argument("--eval-only", action="store_true",
                        help="skip training and evaluate an existing checkpoint")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    trainer = Trainer(cfg, write_history=not args.eval_only)

    if args.eval_only:
        from .evaluate import full_evaluation

        path = args.checkpoint or Path(cfg.run.output_dir) / "best.pt"
        payload = trainer.load_checkpoint(path)
        # Keep the checkpoint's own validation numbers in the JSON report.
        trainer.best_acc = float(payload.get("val_acc", 0.0))
        trainer.best_epoch = int(payload.get("epoch", -1))
        full_evaluation(trainer, path)
        return

    trainer.fit()

    from .evaluate import full_evaluation

    trainer.load_checkpoint(Path(cfg.run.output_dir) / "best.pt")
    full_evaluation(trainer, Path(cfg.run.output_dir) / "best.pt")


if __name__ == "__main__":
    main()
