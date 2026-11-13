"""Transforms, split integrity and DataLoader shapes, using a fake in-memory CIFAR-10."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from deepvision.data import dataset as dataset_module
from deepvision.data.dataloader import get_dataloaders
from deepvision.data.dataset import get_datasets
from deepvision.data.transforms import CIFAR10_MEAN, CIFAR10_STD, get_eval_transform, get_train_transform


class FakeCIFAR10(Dataset):
    """Stands in for torchvision's CIFAR10: no download, random images."""

    def __init__(self, root=None, train=True, download=False, transform=None):
        n = 100 if train else 20
        rng = np.random.default_rng(0)
        self.data = rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
        self.targets = rng.integers(0, 10, n).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = Image.fromarray(self.data[i])
        return (self.transform(img) if self.transform else img), self.targets[i]


@pytest.fixture
def fake_cifar(monkeypatch):
    monkeypatch.setattr(dataset_module.datasets, "CIFAR10", FakeCIFAR10)


def test_eval_transform_normalizes_correctly():
    black = Image.new("RGB", (32, 32), (0, 0, 0))
    out = get_eval_transform()(black)
    assert out.shape == (3, 32, 32)
    for c in range(3):
        assert out[c, 0, 0].item() == pytest.approx((0 - CIFAR10_MEAN[c]) / CIFAR10_STD[c], rel=1e-4)


@pytest.mark.parametrize("size", [32, 64])
def test_transform_output_size(size):
    img = Image.new("RGB", (32, 32), (120, 30, 200))
    assert get_train_transform(image_size=size)(img).shape == (3, size, size)
    assert get_eval_transform(image_size=size)(img).shape == (3, size, size)


def test_split_sizes_and_no_overlap(fake_cifar):
    train, val, test = get_datasets(val_size=20)
    assert (len(train), len(val), len(test)) == (80, 20, 20)
    assert set(train.indices).isdisjoint(val.indices)


def test_split_is_seeded(fake_cifar):
    a = get_datasets(val_size=20, seed=1)[1].indices
    b = get_datasets(val_size=20, seed=1)[1].indices
    c = get_datasets(val_size=20, seed=2)[1].indices
    assert a == b and a != c


def test_validation_pipeline_is_deterministic(fake_cifar):
    _, val, _ = get_datasets(val_size=20)
    assert torch.equal(val[0][0], val[0][0])


def test_dataloader_batch_shapes(fake_cifar):
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=16, val_size=20)
    xb, yb = next(iter(train_loader))
    assert xb.shape == (16, 3, 32, 32) and yb.shape == (16,)
    assert yb.dtype == torch.long
    assert sum(len(y) for _, y in val_loader) == 20
