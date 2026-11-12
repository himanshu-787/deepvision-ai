"""Learning-rate scheduler factory (stepped once per epoch)."""

from typing import Optional

import torch


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    step_size: Optional[int] = None,
    gamma: float = 0.1,
):
    name = name.lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size or max(1, epochs // 3), gamma=gamma)
    raise ValueError(f"Unknown scheduler: {name!r}")
