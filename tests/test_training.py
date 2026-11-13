"""Training mechanics: loss can fall, early stopping, checkpoints, factories, seeds, configs."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from deepvision.models.cnn import SimpleCNN
from deepvision.models.resnet import ResNet18Classifier
from deepvision.training.checkpoint import load_checkpoint, save_checkpoint
from deepvision.training.finetune import build_finetune_groups
from deepvision.training.optimizer import build_optimizer
from deepvision.training.scheduler import build_scheduler
from deepvision.training.trainer import fit, train_one_epoch
from deepvision.utils.config import config_hash, deep_merge, load_config
from deepvision.utils.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


def tiny_loader(n: int = 32) -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)


def test_model_can_overfit_a_tiny_dataset():
    """The core sanity check: if the loss can't drop here, the pipeline has a bug."""
    set_seed(0)
    loader, model = tiny_loader(), SimpleCNN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    crit = nn.CrossEntropyLoss()
    first, _ = train_one_epoch(model, loader, crit, opt, CPU)
    for _ in range(60):
        last, _ = train_one_epoch(model, loader, crit, opt, CPU)
    assert last < 0.5 * first


def test_fit_saves_checkpoint_and_calls_callback(tmp_path):
    set_seed(0)
    loader, model = tiny_loader(), SimpleCNN()
    calls = []
    history = fit(
        model, loader, loader, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-3), CPU,
        epochs=2, checkpoint_path=tmp_path / "ckpt.pt",
        best_val_acc=-1.0,                       # guarantees the first epoch counts as "best"
        epoch_callback=lambda epoch, row: calls.append(epoch),
    )
    assert (tmp_path / "ckpt.pt").exists()
    assert len(history["val_acc"]) == 2 and calls == [1, 2]


def test_early_stopping_triggers(tmp_path):
    loader, model = tiny_loader(), SimpleCNN()
    history = fit(
        model, loader, loader, nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.0), CPU,
        epochs=10, checkpoint_path=tmp_path / "ckpt.pt", patience=1, best_val_acc=-1.0,
    )
    assert len(history["val_acc"]) == 2          # lr=0: no improvement after epoch 1 -> stop


def test_checkpoint_roundtrip(tmp_path):
    set_seed(0)
    a, b = SimpleCNN().eval(), SimpleCNN().eval()
    save_checkpoint(tmp_path / "c.pt", a, None, epoch=3, best_val_acc=0.5, extra={"seed": 1})
    ckpt = load_checkpoint(tmp_path / "c.pt", b)
    x = torch.randn(2, 3, 32, 32)
    assert torch.equal(a(x), b(x)) and ckpt["epoch"] == 3 and ckpt["extra"]["seed"] == 1


def test_optimizer_and_scheduler_factories():
    model = SimpleCNN()
    for name, cls in [("sgd", torch.optim.SGD), ("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)]:
        assert isinstance(build_optimizer(name, model.parameters(), 1e-3), cls)
    with pytest.raises(ValueError):
        build_optimizer("nope", model.parameters(), 1e-3)

    opt = build_optimizer("sgd", model.parameters(), 0.1)
    sched = build_scheduler("cosine", opt, epochs=10)
    for _ in range(10):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] < 1e-6
    assert build_scheduler("none", opt, 10) is None


def test_discriminative_learning_rates():
    model = ResNet18Classifier(pretrained=False)
    groups = {g["name"]: g["lr"] for g in build_finetune_groups(
        model, ["layer3", "layer4"], head_lr=1e-3, backbone_lr=1e-4, lr_decay=0.5)}
    assert groups == {"fc": 1e-3, "layer4": 1e-4, "layer3": 5e-5}


def test_set_seed_makes_random_numbers_repeatable():
    set_seed(1); a = torch.rand(3)
    set_seed(1); b = torch.rand(3)
    assert torch.equal(a, b)


def test_config_merge_override_and_hash(tmp_path, monkeypatch):
    base = {"a": {"x": 1, "y": 2}, "b": 1}
    merged = deep_merge(base, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1} and base["a"]["y"] == 2     # input not mutated

    monkeypatch.chdir(ROOT)                      # load_config reads configs/base.yaml relative to cwd
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump({"run_name": "t", "strategy": "scratch"}))
    cfg = load_config(str(exp), overrides=["train.lr=0.0003", "finetune.stages=[layer4]"])
    assert cfg["train"]["lr"] == 0.0003 and cfg["finetune"]["stages"] == ["layer4"] and cfg["run_name"] == "t"
    with pytest.raises(ValueError):
        load_config(str(exp), overrides=["no_equals_sign"])
    assert config_hash(cfg) == config_hash(dict(cfg)) and config_hash(cfg) != config_hash({**cfg, "seed": 99})
