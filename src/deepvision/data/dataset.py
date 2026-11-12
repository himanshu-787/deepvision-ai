"""CIFAR-10 datasets with a leak-free train/validation split."""

from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets

from deepvision.data.transforms import get_eval_transform, get_train_transform

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_datasets(
    root: str = "data/raw",
    val_size: int = 5000,
    seed: int = 42,
    augment: bool = True,
    image_size: int = 32,
    stats: str = "cifar",
) -> Tuple[Dataset, Dataset, Dataset]:
    """Return (train, val, test). Same seeded split for any image_size or stats,
    so every model is validated on exactly the same 5,000 images."""
    train_aug = datasets.CIFAR10(root, train=True, download=True,
                                 transform=get_train_transform(augment=augment, image_size=image_size, stats=stats))
    train_plain = datasets.CIFAR10(root, train=True, download=True,
                                   transform=get_eval_transform(image_size=image_size, stats=stats))
    test = datasets.CIFAR10(root, train=False, download=True,
                            transform=get_eval_transform(image_size=image_size, stats=stats))

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_aug), generator=generator).tolist()
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    return Subset(train_aug, train_idx), Subset(train_plain, val_idx), test
