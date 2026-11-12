"""CNNs trained from scratch."""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Phase 5 baseline. Input (B,3,32,32) -> logits (B,num_classes)."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU -> MaxPool. bias=False because BatchNorm has its own shift."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ImprovedCNN(nn.Module):
    """Same layout as SimpleCNN plus BatchNorm and Dropout.

    Shapes: (B,3,32,32) -> 32x16x16 -> 64x8x8 -> 128x4x4 -> 2048 -> 256 -> logits.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(_conv_block(3, 32), _conv_block(32, 64), _conv_block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
