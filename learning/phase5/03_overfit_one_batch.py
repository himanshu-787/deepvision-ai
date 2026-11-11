"""Phase 5.5 - sanity check: a healthy pipeline can memorize one tiny batch."""

import torch
import torch.nn as nn

from deepvision.data.dataloader import get_dataloaders
from deepvision.models.cnn import SimpleCNN
from deepvision.utils.device import get_device
from deepvision.utils.seed import set_seed


def main() -> None:
    set_seed(42)
    device = get_device()
    train_loader, _, _ = get_dataloaders(batch_size=64, augment=False)
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"initial loss should be near ln(10) = 2.303")

    for step in range(1, 101):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if step in (1, 25, 50, 75, 100):
            acc = (logits.argmax(1) == labels).float().mean().item()
            print(f"step {step:3d} | loss {loss.item():.4f} | acc {acc:.3f}")


if __name__ == "__main__":
    main()
