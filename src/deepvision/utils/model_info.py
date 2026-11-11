"""Model inspection helpers."""

from typing import Tuple

import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
