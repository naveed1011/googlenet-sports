"""Quick sanity checks. Run with:  python -m pytest tests/ -q
or without pytest:  python tests/test_model.py
"""
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model import VARIANTS, build_model, count_parameters, inception_loss

NUM_CLASSES = 15


def _build(variant, aux_weight):
    return build_model(num_classes=NUM_CLASSES, aux_weight=aux_weight,
                       variant=variant)


def _forward_shapes(variant, aux_weight, batch=2):
    model = _build(variant, aux_weight)
    x = torch.randn(batch, 3, 224, 224)
    with torch.no_grad():
        model.train()
        train_out = model(x)
        model.eval()
        eval_out = model(x)
    return model, train_out, eval_out


def test_forward_shapes_all_variants():
    for variant in VARIANTS:
        model, train_out, eval_out = _forward_shapes(variant, aux_weight=0.3)
        assert isinstance(train_out, tuple), variant
        assert train_out[0].shape == (2, NUM_CLASSES), variant
        assert train_out[1].shape == (2, NUM_CLASSES), variant
        assert eval_out.shape == (2, NUM_CLASSES), variant
        del model, train_out, eval_out
        gc.collect()


def test_eval_returns_single_tensor():
    _, _, eval_out = _forward_shapes("modern", aux_weight=0.3)
    assert isinstance(eval_out, torch.Tensor)


def test_aux_can_be_disabled():
    model, train_out, _ = _forward_shapes("modern", aux_weight=0.0)
    assert model.aux is None
    assert isinstance(train_out, torch.Tensor)
    del model
    gc.collect()


def test_feature_map_schedule():
    model = _build("modern", aux_weight=0.0)
    model.eval()
    with torch.no_grad():
        features = model.forward_features(torch.randn(1, 3, 224, 224))
    assert tuple(features.shape) == (1, 1024, 7, 7)
    del model, features
    gc.collect()


def test_parameter_counts_are_sane():
    for variant in VARIANTS:
        model = _build(variant, aux_weight=0.3)
        total = count_parameters(model)[0]
        # factoring the 5x5 branch adds a conv per block, so all three land in
        # the same band, far under the Keras version's 10.3M
        assert 6_000_000 < total < 7_000_000, (variant, total)
        del model
        gc.collect()


def test_initial_loss_is_near_uniform():
    torch.manual_seed(0)
    model = _build("modern", aux_weight=0.3)
    model.train()
    target = torch.randint(0, NUM_CLASSES, (8,))
    with torch.no_grad():
        loss, _, _ = inception_loss(model(torch.randn(8, 3, 224, 224)), target, 0.3)
    # ln(15) ~= 2.71; with the aux term the total should land close to that
    assert 2.0 < loss.item() < 4.5, loss.item()
    del model
    gc.collect()


def test_config_overrides():
    cfg = load_config(overrides=["train.epochs=7", "model.variant=bn5x5",
                                 "train.amp=false"])
    assert cfg.train.epochs == 7
    assert cfg.model.variant == "bn5x5"
    assert cfg.train.amp is False


def test_config_rejects_unknown_keys():
    try:
        load_config(overrides=["train.nope=1"])
    except KeyError:
        return
    raise AssertionError("unknown override should raise KeyError")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print("\nall checks passed" if not failures else f"\n{failures} check(s) failed")
    sys.exit(1 if failures else 0)
