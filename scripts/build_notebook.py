"""Generates notebooks/googlenet_sports_torch.ipynb.

Kept as a script so the notebook JSON stays valid while the cells are edited.
Run:  python scripts/build_notebook.py
"""
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent.parent / "notebooks" / "googlenet_sports_torch.ipynb"

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.split("\n")})


def code(source):
    lines = source.split("\n")
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": lines,
        "outputs": [],
        "execution_count": None,
    })


def fix(lines):
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines else [])


md(
"""# GoogLeNet on Sports Balls - PyTorch rebuild

This is a from-scratch reimplementation of a GoogLeNet (Inception-v1 style)
classifier for the Kaggle *sports-balls* dataset, written in PyTorch.

An earlier version of this project was built in Keras and trained with a fixed
learning rate of 1e-5, no augmentation and two auxiliary heads that together
held about 4.2M of the 10.3M parameters. It reached 55.5% on a validation split
carved out of `train/`.

This notebook is the second pass. The changes that matter, in order of how much
they moved the score:

1. **A real training recipe.** AdamW at 1e-3 with a short warmup and a cosine
   decay, instead of a constant 1e-5 that was far too small to converge.
2. **Augmentation.** The v1 model never saw a flipped or recoloured image.
3. **A proper evaluation protocol.** The dataset ships an untouched `test/`
   split; v1 never looked at it. Here it is only touched once, at the end.
4. **Architecture.** The 5x5 branch of every inception block is factored into
   two 3x3 convolutions and batch normalisation is applied in every branch,
   which cuts parameters from 10.3M to 6.5M.
5. **One auxiliary head instead of two**, with a clean weighted loss rather
   than duplicating each label three times.

Everything below runs end to end on a Kaggle GPU in roughly 40 minutes.

> **Note on the two builds.** The metrics in the README come from the executed
> Kaggle notebook (`googlenet_sports_kaggle.ipynb`), which uses an 8.47M-parameter
> version of the model. This walkthrough runs the refactored `src/` build
> (6.50M with the slimmed 0.22M auxiliary head). The backbone is the same; only
> the auxiliary head changed size.
"""
)

md("## Setup")

code(
"""import os, sys, subprocess
from pathlib import Path

import torch
print(torch.__version__, "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")"""
)

md(
"""### Get the code

The model and training loop live in the `src/` package of the GitHub repo so
they can be reused and tested outside the notebook. The cell below finds them
in this order: already importable, sitting next to the notebook, or cloned
fresh from GitHub.

The URL points at the main repo; change it if you are running from a fork."""
)

code(
"""REPO_URL = "https://github.com/naveed1011/googlenet-sports.git"

def get_src():
    try:
        import src
        return Path(src.__file__).parent.parent
    except ImportError:
        pass

    local = Path.cwd() / "googlenet-sports"
    if (local / "src" / "model.py").exists():
        sys.path.insert(0, str(local))
        return local

    target = Path.cwd() / "googlenet-sports-repo"
    if not (target / "src" / "model.py").exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(target)],
                       check=True)
    sys.path.insert(0, str(target))
    return target

repo_root = get_src()
print("using code from:", repo_root)"""
)

code(
"""import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

from src.config import load_config
from src.data import (CLASS_NAMES, build_loaders, class_counts, find_dataset_root,
                      load_datasets, pretty, seed_everything)
from src.model import build_model, count_parameters
from src.train import Trainer
from src.evaluate import full_evaluation
from src.utils import read_history_csv

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)"""
)

md(
"""## The data

15 classes, 9,169 images. `train/` holds 7,328 of them and `test/` holds the
remaining 1,841. The split below takes 10% of `train/` as validation and leaves
`test/` completely alone."""
)

code(
"""cfg = load_config(
    str(repo_root / "configs" / "default.yaml"),
    overrides=[
        "data.root=",
        "run.output_dir=outputs",
    ],
)

data_root = find_dataset_root(cfg.data.root or None)
print("dataset at:", data_root)

train_set, val_set, test_set, classes, root = load_datasets(cfg)
train_loader, val_loader, test_loader = build_loaders(train_set, val_set, test_set, cfg)

print(f"train {len(train_set):5d}   val {len(val_set):5d}   test {len(test_set):5d}")"""
)

code(
"""counts = class_counts(test_set.dataset)
order = np.argsort(-counts)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar([pretty(classes[i]) for i in order], counts[order], color="#4c72b0")
ax.set_title("Held-out test set, images per class")
ax.set_xticklabels([pretty(classes[i]) for i in order], rotation=40, ha="right",
                   fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()"""
)

code(
"""from PIL import Image as PILImage
from torchvision import transforms

show = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224)])
fig, axes = plt.subplots(3, 4, figsize=(12, 9.5))
rng = np.random.default_rng(cfg.train.seed)
paths = [s[0] for s in test_set.dataset.samples]
picks = rng.choice(len(paths), size=12, replace=False)

for ax, i in zip(axes.ravel(), picks):
    idx = int(i)
    label = test_set.dataset.samples[idx][1]
    ax.imshow(show(PILImage.open(paths[idx]).convert("RGB")))
    ax.set_title(pretty(classes[label]), fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()"""
)

md(
"""## The model

Same filter budget as the 2014 paper, but with the 5x5 branch factored into two
3x3 convolutions and batch normalisation in every branch. The parameter table
below also builds the two comparison variants so the difference is visible:

- `modern` - factored 5x5, batch norm (the one being trained here)
- `bn5x5`  - real 5x5, batch norm (isolates the effect of factoring)
- `v1`     - real 5x5, no batch norm (closest to the original notebook)"""
)

code(
"""rows = []
for variant in ("modern", "bn5x5", "v1"):
    m = build_model(num_classes=len(classes), variant=variant, aux_weight=0.0)
    total, _ = count_parameters(m)
    aux_only = build_model(num_classes=len(classes), variant=variant, aux_weight=0.3)
    aux_params = count_parameters(aux_only.aux)[0]
    rows.append((variant, total, aux_params, total + aux_params))

table = pd.DataFrame(rows, columns=["variant", "backbone", "aux head", "total"])
table["v1 keras reference"] = 10_345_501
print(table.to_string(index=False))
print()
print("the Keras version spent 4,196,352 of its 10.3M parameters on two aux heads")"""
)

md("## Train")

code(
"""trainer = Trainer(cfg)
history = trainer.fit()"""
)

md(
"""## Curves

`history.csv` is written every epoch, so an interrupted run still has its
log."""
)

code(
"""hist = read_history_csv(OUT / "history.csv")
display(Image(filename=str(OUT / "curves.png")))

best = int(np.nanargmax(hist["val_acc"])) + 1
print(f"best validation accuracy {max(hist['val_acc']):.4f} at epoch {best}")"""
)

md(
"""## Test set, once

Everything above used `train/` only. This is the single pass over the untouched
`test/` split. It loads the best checkpoint, which stores the EMA weights."""
)

code(
"""trainer.load_checkpoint(OUT / "best.pt")
results = full_evaluation(trainer, OUT / "best.pt")"""
)

code(
"""display(Image(filename=str(OUT / "confusion_normalized.png")))"""
)

code(
"""display(Image(filename=str(OUT / "per_class.png")))
display(Image(filename=str(OUT / "confusions.png")))"""
)

md(
"""## Where the model looks

Grad-CAM on the final convolution layer, half on images it got wrong and half
on images it got right."""
)

code(
"""display(Image(filename=str(OUT / "gradcam.png")))"""
)

md(
"""## Against the first version

Numbers on the left are from the Keras run. The validation number is not
directly comparable because the splits differ, but the parameter count and the
test score are new information that simply did not exist before."""
)

code(
"""comparison = pd.DataFrame([
    ["framework", "TensorFlow / Keras", "PyTorch"],
    ["parameters", "10,345,501", f"{count_parameters(trainer.model)[0]:,}"],
    ["aux heads", "2 (4.2M params)", "1 (0.2M params)"],
    ["augmentation", "none", "crop, flip, rotate, jitter"],
    ["optimiser", "Adam 1e-5 fixed", "AdamW 1e-3 + warmup + cosine"],
    ["epochs", "50", str(cfg.train.epochs)],
    ["val accuracy", "0.5549", f"{results['best_val_acc']:.4f}"],
    ["test accuracy", "never measured", f"{results['accuracy']:.4f}"],
    ["test macro F1", "never measured", f"{results['macro_f1']:.4f}"],
], columns=["", "v1 (Keras)", "v2 (this notebook)"])
print(comparison.to_string(index=False))"""
)

md(
"""## Notes and next things to try

- The auxiliary head is worth about a point of accuracy in the early epochs
  and nothing by the end. Setting `model.aux_weight=0.0` removes it entirely.
- `model.variant=bn5x5` re-runs the same training with a real 5x5 convolution,
  which is the cleanest way to see what the factorisation actually bought.
- The two hockey classes and the two football-ish classes are where most of
  the remaining error lives; the dataset author notes some images are painted
  to look like other balls, so part of that error is labelling noise.
"""
)

for cell in cells:
    cell["source"] = fix(cell["source"])

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} with {len(cells)} cells")
