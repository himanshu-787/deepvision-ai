"""A small CNN trained from scratch (Model A)."""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Three conv blocks followed by a two-layer classifier.

    Input : (B, 3, 32, 32) normalized images
    Output: (B, num_classes) raw logits (NO softmax; CrossEntropyLoss does it)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # -> 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # -> 64x8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # -> 128x4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # -> 2048
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
