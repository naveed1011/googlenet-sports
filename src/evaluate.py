"""Test-set evaluation: metrics, confusion matrix, error analysis, Grad-CAM."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD, pretty
from .utils import save_json


@torch.no_grad()
def predict(model, loader, device, return_paths=False):
    model.eval()
    probs, targets, paths = [], [], []
    for batch in loader:
        images, labels = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        out = model(images)
        logits = out[0] if isinstance(out, tuple) else out
        probs.append(logits.softmax(1).cpu().numpy())
        targets.append(labels.numpy())
        if return_paths:
            paths.extend(batch[2])

    probs = np.concatenate(probs)
    targets = np.concatenate(targets)
    preds = probs.argmax(1)
    if return_paths:
        return probs, targets, preds, paths
    return probs, targets, preds


def sample_paths(dataset):
    from torch.utils.data import Subset

    if isinstance(dataset, Subset):
        return [dataset.dataset.samples[i][0] for i in dataset.indices]
    return [s[0] for s in dataset.samples]


def metrics_table(y_true, y_pred, class_names):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    return pd.DataFrame({
        "class": class_names,
        "precision": precision.round(4),
        "recall": recall.round(4),
        "f1": f1.round(4),
        "support": support,
    })


def top_confusions(y_true, y_pred, class_names, k=10):
    """The most frequent true -> predicted pairs that are actually wrong."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda p: -p[2])
    return pairs[:k]


def plot_confusion(y_true, y_pred, class_names, path, normalize=True, title=None):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    labels = [pretty(c) for c in class_names]

    fig, ax = plt.subplots(figsize=(11, 9))
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float),
                            where=row_sums > 0)
        sns.heatmap(cm_plot, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={"size": 7}, cbar_kws={"label": "row share"})
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={"size": 7})

    ax.set_title(title or "Confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_class(table, path):
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(table))
    width = 0.27
    ax.bar(x - width, table["precision"], width, label="precision")
    ax.bar(x, table["recall"], width, label="recall")
    ax.bar(x + width, table["f1"], width, label="f1")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty(c) for c in table["class"]], rotation=40, ha="right",
                       fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Per-class performance on the held-out test set")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusions(pairs, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    names = [f"{pretty(t)}\n-> {pretty(p)}" for t, p, _ in pairs]
    counts = [c for _, _, c in pairs]
    ax.barh(range(len(pairs))[::-1], counts, color="#c0504d")
    ax.set_yticks(range(len(pairs))[::-1])
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("test images misclassified this way")
    ax.set_title("Most common confusions")
    for i, c in enumerate(counts):
        ax.text(c + 0.3, len(pairs) - 1 - i, str(c), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


class GradCAM:
    """Gradient-weighted activation map on the last conv feature layer."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.layer = target_layer if target_layer is not None \
            else model.grad_cam_layer()
        self.activations = None
        self.gradients = None
        self._fh = self.layer.register_forward_hook(self._on_forward)
        self._bh = self.layer.register_full_backward_hook(self._on_backward)

    def _on_forward(self, module, inputs, output):
        self.activations = output.detach()

    def _on_backward(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, target_class=None):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        logits = out[0] if isinstance(out, tuple) else out
        if target_class is None:
            target_class = logits.argmax(1)

        score = logits.gather(1, target_class.view(-1, 1)).squeeze(1).sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        return F.relu(cam)

    def remove(self):
        self._fh.remove()
        self._bh.remove()


def denormalize(tensor):
    """Inverse of the ImageNet normalisation, for display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def plot_gradcam_grid(model, device, items, class_names, out_path,
                      columns=4, size=224):
    """items is a list of (image_path, target_class_index, caption)."""
    loader = transforms.Compose([
        transforms.Resize(int(round(size * 256 / 224))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    cam = GradCAM(model)
    rows = int(np.ceil(len(items) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (img_path, target, caption) in zip(axes, items):
        raw = Image.open(img_path).convert("RGB")
        x = loader(raw).unsqueeze(0).to(device)

        model.eval()
        heatmap = cam(x, torch.tensor([target], device=device))[0]
        heatmap = F.interpolate(heatmap.unsqueeze(0), size=(size, size),
                                mode="bilinear", align_corners=False)[0, 0]
        heatmap = heatmap.cpu().numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        ax.imshow(denormalize(x[0]))
        ax.imshow(heatmap, cmap="jet", alpha=0.45)
        ax.set_title(caption, fontsize=8)
        ax.axis("off")

    for ax in axes[len(items):]:
        ax.axis("off")

    cam.remove()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def full_evaluation(trainer, checkpoint_path=None, gradcam_samples=8):
    """Evaluate on the untouched test/ split and write every artifact out."""
    cfg = trainer.cfg
    out_dir = Path(cfg.run.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = trainer.device

    model = trainer.model
    probs, y_true, y_pred = predict(model, trainer.test_loader, device)
    class_names = trainer.classes
    accuracy = float((y_true == y_pred).mean())

    table = metrics_table(y_true, y_pred, class_names)
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    confusions = top_confusions(y_true, y_pred, class_names)

    trainer.log("=" * 68)
    trainer.log(f"TEST SET  n={len(y_true)}  accuracy={accuracy:.4f}  "
                f"macro F1={macro[2]:.4f}  weighted F1={weighted[2]:.4f}")
    if checkpoint_path:
        trainer.log(f"checkpoint: {checkpoint_path}")
    trainer.log("=" * 68)
    trainer.log(classification_report(
        y_true, y_pred, target_names=[pretty(c) for c in class_names],
        zero_division=0))
    trainer.log(table.to_string(index=False))
    trainer.log("")
    trainer.log("most common confusions:")
    for true_name, pred_name, count in confusions:
        trainer.log(f"  {pretty(true_name):<20s} -> {pretty(pred_name):<20s} {count}")

    plot_confusion(y_true, y_pred, class_names, out_dir / "confusion_normalized.png",
                   normalize=True, title=f"GoogLeNet ({cfg.model.variant}) - test set")
    plot_confusion(y_true, y_pred, class_names, out_dir / "confusion_counts.png",
                   normalize=False)
    plot_per_class(table, out_dir / "per_class.png")
    plot_confusions(confusions, out_dir / "confusions.png")

    table.to_csv(out_dir / "per_class_metrics.csv", index=False)
    np.save(out_dir / "test_probs.npy", probs)
    np.save(out_dir / "test_true.npy", y_true)
    np.save(out_dir / "test_pred.npy", y_pred)

    results = {
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "variant": cfg.model.variant,
        "test_size": int(len(y_true)),
        "accuracy": accuracy,
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "best_val_acc": trainer.best_acc,
        "best_val_epoch": trainer.best_epoch,
        "top_confusions": [
            {"true": t, "pred": p, "count": c} for t, p, c in confusions
        ],
        "per_class": table.to_dict(orient="records"),
    }
    save_json(results, out_dir / "test_results.json")

    if gradcam_samples:
        items = _pick_gradcam_items(
            trainer, y_true, y_pred, probs, class_names, gradcam_samples)
        if items:
            plot_gradcam_grid(model, device, items, class_names,
                              out_dir / "gradcam.png")
            trainer.log(f"\nwrote Grad-CAM grid for {len(items)} test images")

    trainer.log(f"\nartifacts in {out_dir.resolve()}")
    return results


def _pick_gradcam_items(trainer, y_true, y_pred, probs, class_names, n):
    """Half correct, half wrong, spread over different classes."""
    paths = sample_paths(trainer.test_set)
    correct = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]]
    wrong = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]

    def spread(indices, count):
        chosen, seen = [], set()
        for i in indices:
            key = int(y_true[i])
            if key in seen:
                continue
            seen.add(key)
            chosen.append(i)
            if len(chosen) >= count:
                break
        for i in indices:
            if len(chosen) >= count:
                break
            if i not in chosen:
                chosen.append(i)
        return chosen

    picks = spread(wrong, n // 2) + spread(correct, n - min(n // 2, len(wrong)))
    items = []
    for i in picks[:n]:
        true_name = class_names[int(y_true[i])]
        if y_true[i] == y_pred[i]:
            caption = f"{pretty(true_name)}\ncorrect ({probs[i].max():.2f})"
        else:
            caption = (f"true {pretty(true_name)}\n"
                       f"pred {pretty(class_names[int(y_pred[i])])} "
                       f"({probs[i].max():.2f})")
        items.append((paths[i], int(y_true[i]), caption))
    return items
