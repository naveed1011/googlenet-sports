"""Config objects, loaded from YAML with command line overrides."""

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    root: str = ""
    image_size: int = 224
    batch_size: int = 64
    val_fraction: float = 0.1
    num_workers: int = -1
    augment: bool = True

    # RandomResizedCrop scale range, and colour jitter strengths.
    crop_scale_min: float = 0.7
    jitter: float = 0.2
    rotation: float = 10.0


@dataclass
class ModelConfig:
    variant: str = "modern"
    num_classes: int = 15
    aux_weight: float = 0.3
    dropout: float = 0.4


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    min_lr_ratio: float = 0.01
    warmup_epochs: int = 3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    amp: bool = True
    ema_decay: float = 0.999
    use_ema: bool = True
    seed: int = 123
    log_every: int = 20
    patience: int = 0


@dataclass
class RunConfig:
    output_dir: str = "outputs"
    project_name: str = "googlenet-sports"
    notes: str = ""


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "run": RunConfig,
}


def _coerce(cls, raw):
    if raw is None:
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"{cls.__name__} got unknown keys: {sorted(unknown)}")
    return cls(**raw)


def load_config(path=None, overrides=None):
    """Read a YAML config and apply dot-notation overrides like train.lr=3e-4."""
    cfg = Config()
    if path:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        unknown = set(raw) - set(SECTIONS)
        if unknown:
            raise KeyError(f"unknown config sections: {sorted(unknown)}")
        for section, cls in SECTIONS.items():
            setattr(cfg, section, _coerce(cls, raw.get(section)))

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like section.key=value")
        key, value = item.split("=", 1)
        section, _, name = key.partition(".")
        if section not in SECTIONS or not name:
            raise KeyError(f"unknown override target {key!r}")
        target = getattr(cfg, section)
        if not hasattr(target, name):
            raise KeyError(f"{section} has no field {name!r}")
        current = getattr(target, name)
        setattr(target, name, _parse(value, type(current)))

    return cfg


def _parse(value, kind):
    if kind is bool:
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if kind is str:
        return value
    return kind(value)
