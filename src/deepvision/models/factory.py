"""Build models by name."""

import torch.nn as nn

from deepvision.models.cnn import ImprovedCNN, SimpleCNN
from deepvision.models.resnet import ResNet18Classifier


def build_model(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """kwargs are model-specific options (e.g. dropout=0.3 or pretrained=True)."""
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, **kwargs)
    if name == "improved_cnn":
        return ImprovedCNN(num_classes=num_classes, **kwargs)
    if name == "resnet18":
        return ResNet18Classifier(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model name: {name!r}")
