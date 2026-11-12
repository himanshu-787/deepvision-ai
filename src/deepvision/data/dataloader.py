"""DataLoader construction."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from deepvision.data.dataset import get_datasets


def get_dataloaders(
    root: str = "data/raw",
    batch_size: int = 128,
    num_workers: int = 0,
    val_size: int = 5000,
    seed: int = 42,
    augment: bool = True,
    image_size: int = 32,
    stats: str = "cifar",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_ds, val_ds, test_ds = get_datasets(root, val_size, seed, augment, image_size, stats)
    pin = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader
