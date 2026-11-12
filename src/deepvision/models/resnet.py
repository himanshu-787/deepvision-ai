"""ResNet18 for CIFAR-10 transfer learning and fine-tuning."""

from typing import Iterable, List

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Classifier(nn.Module):
    """torchvision ResNet18 with a new `num_classes` head.

    Stages (input -> output): stem, layer1, layer2, layer3, layer4, fc.
    `set_trainable` freezes everything except the listed stages.
    Frozen BatchNorm layers are automatically kept in eval mode (see `train`).
    """

    STAGES = ("stem", "layer1", "layer2", "layer3", "layer4", "fc")

    def __init__(self, num_classes: int = 10, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None   # downloads ~45 MB once
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)   # 512 -> 10, random init

    def _stage_modules(self, stage: str) -> List[nn.Module]:
        b = self.backbone
        mapping = {
            "stem": [b.conv1, b.bn1], "layer1": [b.layer1], "layer2": [b.layer2],
            "layer3": [b.layer3], "layer4": [b.layer4], "fc": [b.fc],
        }
        if stage not in mapping:
            raise ValueError(f"Unknown stage {stage!r}. Choose from {self.STAGES}")
        return mapping[stage]

    def stage_parameters(self, stage: str) -> List[nn.Parameter]:
        return [p for m in self._stage_modules(stage) for p in m.parameters()]

    def set_trainable(self, stages: Iterable[str]) -> None:
        """Freeze all parameters, then unfreeze the given stages."""
        for p in self.parameters():
            p.requires_grad = False
        for stage in stages:
            for p in self.stage_parameters(stage):
                p.requires_grad = True

    def train(self, mode: bool = True) -> "ResNet18Classifier":
        """Keep BatchNorm layers whose parameters are all frozen in eval mode.

        Otherwise their running statistics would keep changing even though their
        weights are frozen, which silently alters the 'frozen' features.
        """
        super().train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                    m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
