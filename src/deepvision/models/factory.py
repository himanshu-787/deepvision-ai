"""Build models by name, so scripts and (later) configs stay simple."""

import torch.nn as nn

from deepvision.models.cnn import SimpleCNN


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {name!r}")
