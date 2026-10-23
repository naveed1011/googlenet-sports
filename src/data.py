"""Dataset loading for the Kaggle sports-balls set.

The dataset ships train/ and test/. The original notebook only used train/ split
80/20, so it never touched the official test images. Here train/ is split into
train and validation and test/ is left completely alone until final evaluation.
"""

import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Alphabetical, which is the order ImageFolder assigns indices in.
CLASS_NAMES = [
    "american_football",
    "baseball",
    "basketball",
    "billiard_ball",
    "bowling_ball",
    "cricket_ball",
    "football",
    "golf_ball",
    "hockey_ball",
    "hockey_puck",
    "rugby_ball",
    "shuttlecock",
    "table_tennis_ball",
    "tennis_ball",
    "volleyball",
]

PRETTY_NAMES = {
    "american_football": "American football",
    "baseball": "Baseball",
    "basketball": "Basketball",
    "billiard_ball": "Billiard ball",
    "bowling_ball": "Bowling ball",
    "cricket_ball": "Cricket ball",
    "football": "Football",
    "golf_ball": "Golf ball",
    "hockey_ball": "Field hockey ball",
    "hockey_puck": "Hockey puck",
    "rugby_ball": "Rugby ball",
    "shuttlecock": "Shuttlecock",
    "table_tennis_ball": "Table tennis ball",
    "tennis_ball": "Tennis ball",
    "volleyball": "Volleyball",
}

DATASET_SLUG = "sports-balls-multiclass-image-classification"

SEARCH_ROOTS = [
    f"/kaggle/input/{DATASET_SLUG}",
    "/kaggle/input",
    "/content",
    "./data",
    "./data/sports-balls",
    "~/data/sports-balls",
]


def pretty(name):
    return PRETTY_NAMES.get(name, name.replace("_", " ").title())


def find_dataset_root(explicit=None):
    """Locate a directory holding train/ and test/ with the 15 class folders."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for root in SEARCH_ROOTS:
        base = Path(root).expanduser()
        candidates.append(base)
        for child in sorted(base.glob(f"*{DATASET_SLUG}*")) if base.exists() else []:
            candidates.append(child)

    for cand in candidates:
        if (cand / "train").is_dir() and (cand / "test").is_dir():
            return cand.resolve()

    tried = ", ".join(str(c) for c in candidates[:6])
    raise FileNotFoundError(
        "Could not find the dataset. Looked in: " + tried +
        "\nOn Kaggle, attach the dataset to the notebook. Locally, pass "
        "--set data.root=/path/to/sports-balls"
    )


def resolve_workers(num_workers):
    if num_workers is not None and num_workers >= 0:
        return num_workers
    # Kaggle gives 4 CPUs; Windows cannot fork workers.
    if platform.system() == "Windows":
        return 0
    return min(4, max(1, os.cpu_count() // 2))


def build_transforms(image_size, augment=True):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, scale=(0.7, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    eval_size = int(round(image_size * 256 / 224))
    eval_tf = transforms.Compose([
        transforms.Resize(eval_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def class_counts(dataset):
    counts = np.zeros(len(CLASS_NAMES), dtype=int)
    for _, label in dataset.samples:
        counts[label] += 1
    return counts


def load_datasets(cfg):
    """Returns train, val, test datasets plus the class-name list."""
    root = find_dataset_root(cfg.data.root or None)
    train_tf, eval_tf = build_transforms(cfg.data.image_size, cfg.data.augment)

    full_train = datasets.ImageFolder(root / "train", transform=train_tf)
    plain_train = datasets.ImageFolder(root / "train")
    test_set = datasets.ImageFolder(root / "test", transform=eval_tf)

    if list(plain_train.class_to_idx) != list(test_set.class_to_idx):
        raise ValueError(
            "train/ and test/ disagree on class ordering; refusing to mix labels"
        )

    labels = np.array([s[1] for s in plain_train.samples])
    idx = np.arange(len(labels))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=cfg.data.val_fraction,
        stratify=labels,
        random_state=cfg.train.seed,
    )

    train_set = Subset(full_train, tr_idx.tolist())
    # Validation must not be augmented, so rebuild it off the eval transform.
    val_full = datasets.ImageFolder(root / "train", transform=eval_tf)
    val_set = Subset(val_full, va_idx.tolist())

    return train_set, val_set, test_set, list(plain_train.classes), root


def build_loaders(train_set, val_set, test_set, cfg, shuffle_train=True):
    workers = resolve_workers(cfg.data.num_workers)
    common = dict(
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    train_loader = DataLoader(
        train_set, batch_size=cfg.data.batch_size, shuffle=shuffle_train,
        drop_last=True, **common,
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.data.batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        test_set, batch_size=cfg.data.batch_size, shuffle=False, **common
    )
    return train_loader, val_loader, test_loader


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic convolutions cost some speed; enable only when reproducing.
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
