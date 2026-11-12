"""Optimizer factory."""

from typing import Iterable, Union

import torch


def build_optimizer(
    name: str,
    params: Union[Iterable[torch.nn.Parameter], Iterable[dict]],
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """`params` may be parameters or a list of param-group dicts (used in Phase 8)."""
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name!r}")
