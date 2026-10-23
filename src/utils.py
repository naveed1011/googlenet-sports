"""Small helpers: device setup, EMA, meters, the learning rate schedule, plots."""

import contextlib
import copy
import csv
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_report(device):
    if device.type != "cuda":
        return f"device={device} (no CUDA GPU found)"
    name = torch.cuda.get_device_name(device)
    mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return f"device={device} gpu='{name}' memory={mem:.1f}GB torch={torch.__version__}"


def make_autocast(device, dtype=torch.float16):
    if device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast("cuda", dtype=dtype)
    except TypeError:  # torch < 2.0
        return torch.cuda.amp.autocast(dtype=dtype)


def make_scaler(enabled, device):
    if not enabled or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except TypeError:  # torch < 2.3
        return torch.cuda.amp.GradScaler()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(1, self.count)


class ModelEMA:
    """Exponential moving average of the weights, used for evaluation."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = model.state_dict()
        for name, param in self.ema.named_parameters():
            if param.dtype.is_floating_point:
                param.mul_(self.decay).add_(src[name].detach(), alpha=1 - self.decay)
            else:
                param.copy_(src[name])
        # Batch norm running stats are already averages, so copy them straight in.
        for name, buf in self.ema.named_buffers():
            if name in src:
                buf.copy_(src[name])

    def state_dict(self):
        return self.ema.state_dict()


def build_param_groups(model, weight_decay):
    """Weight decay on conv/linear kernels only, never on norms or biases."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    total_steps = max(1, total_steps)

    def factor(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, factor)


class History:
    """Append-only epoch log, written straight to CSV so a crash loses nothing."""

    FIELDS = [
        "epoch", "train_loss", "train_main_loss", "train_aux_loss", "train_acc",
        "val_loss", "val_main_loss", "val_aux_loss", "val_acc", "lr", "seconds",
    ]

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = []
        with open(self.path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.FIELDS).writeheader()

    def add(self, row):
        self.rows.append(row)
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.FIELDS).writerow(
                {k: row.get(k, "") for k in self.FIELDS}
            )

    def to_dict(self):
        out = {k: [] for k in self.FIELDS}
        for row in self.rows:
            for k in self.FIELDS:
                out[k].append(row.get(k))
        return out


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.start
        return False


def format_duration(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def plot_curves(history_dict, path, title="Training curves"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    epochs = history_dict["epoch"]

    axes[0].plot(epochs, history_dict["train_loss"], label="train total")
    axes[0].plot(epochs, history_dict["train_main_loss"], label="train main", ls="--")
    axes[0].plot(epochs, history_dict["val_loss"], label="val total")
    axes[0].plot(epochs, history_dict["val_main_loss"], label="val main", ls="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history_dict["train_acc"], label="train")
    axes[1].plot(epochs, history_dict["val_acc"], label="val")
    axes[1].set_title("Top-1 accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history_dict["lr"])
    axes[2].set_title("Learning rate")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_json(obj, path):
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def read_history_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for row in rows:
        for key, value in row.items():
            out.setdefault(key, []).append(
                float(value) if value not in ("", None) else np.nan
            )
    return out
