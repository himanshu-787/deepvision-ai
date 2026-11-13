"""Fixtures shared by all tests."""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from deepvision.data.dataset import CLASS_NAMES
from deepvision.models.cnn import SimpleCNN
from deepvision.training.checkpoint import save_checkpoint


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """A random-weight SimpleCNN checkpoint in our real checkpoint format."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("model") / "model.pt"
    save_checkpoint(
        path, SimpleCNN(num_classes=10), optimizer=None, epoch=1, best_val_acc=0.0,
        extra={"model_name": "simple_cnn", "model_kwargs": {}, "image_size": 32, "stats": "cifar",
               "class_names": CLASS_NAMES, "seed": 0},
    )
    return path


@pytest.fixture
def client(tiny_checkpoint, monkeypatch):
    """API client with the tiny model loaded (lifespan runs inside the `with`)."""
    monkeypatch.setenv("MODEL_PATH", str(tiny_checkpoint))
    monkeypatch.setenv("MAX_UPLOAD_MB", "5")
    from api.main import app
    with TestClient(app) as c:
        yield c
