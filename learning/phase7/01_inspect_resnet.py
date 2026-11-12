"""Phase 7 - inspect ResNet18: stages, parameters, shapes, freezing and BatchNorm behaviour."""

import torch

from deepvision.models.resnet import ResNet18Classifier
from deepvision.utils.model_info import count_parameters

model = ResNet18Classifier(num_classes=10, pretrained=True)
print("head:", model.backbone.fc)

print("\nParameters per stage:")
for stage in ResNet18Classifier.STAGES:
    print(f"  {stage:7s} {sum(p.numel() for p in model.stage_parameters(stage)):>10,}")
print("total:", f"{count_parameters(model)[0]:,}")

# ---- Shapes through the backbone for different input sizes ----
b = model.backbone.eval()
for size in (32, 64, 128):
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        shapes = [tuple(x.shape)]
        for layer in (b.layer1, b.layer2, b.layer3, b.layer4):
            x = layer(x)
            shapes.append(tuple(x.shape))
    print(f"\ninput {size}x{size} -> after stem {shapes[0]}, layer4 output {shapes[-1]}")

# ---- Freeze for feature extraction ----
model.set_trainable(["fc"])
total, trainable = count_parameters(model)
print(f"\nfrozen backbone: total={total:,} trainable={trainable:,}")   # trainable = 512*10+10 = 5,130

model.train()
print("layer4 BatchNorm in training mode (frozen)? ", model.backbone.layer4[0].bn1.training)  # False

model.set_trainable(["fc", "layer4"])
model.train()
print("layer4 BatchNorm in training mode (unfrozen)?", model.backbone.layer4[0].bn1.training)  # True

# ---- A residual block in miniature: y = F(x) + x ----
class TinyResidual(torch.nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Conv2d(c, c, 3, padding=1), torch.nn.ReLU(),
                                     torch.nn.Conv2d(c, c, 3, padding=1))
    def forward(self, x):
        return torch.relu(self.f(x) + x)      # the shortcut "+ x"

print("\nresidual block keeps shape:", TinyResidual(8)(torch.randn(1, 8, 16, 16)).shape)
