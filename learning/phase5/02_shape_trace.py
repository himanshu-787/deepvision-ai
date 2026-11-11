"""Phase 5.2 - trace tensor shapes through the CNN and verify the parameter count."""

import torch

from deepvision.models.cnn import SimpleCNN
from deepvision.utils.model_info import count_parameters

model = SimpleCNN()
x = torch.randn(8, 3, 32, 32)
print("input:", tuple(x.shape))
for layer in model.features:
    x = layer(x)
    print(f"{layer.__class__.__name__:10s} -> {tuple(x.shape)}")
for layer in model.classifier:
    x = layer(x)
    print(f"{layer.__class__.__name__:10s} -> {tuple(x.shape)}")

total, trainable = count_parameters(model)
print(f"total parameters: {total:,} | trainable: {trainable:,}")   # expect 620,362
