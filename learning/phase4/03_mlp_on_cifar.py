"""Phase 4.3 - an MLP baseline on CIFAR-10 to show why CNNs are needed."""

import torch
import torch.nn as nn

from deepvision.data.dataloader import get_dataloaders
from deepvision.utils.device import get_device
from deepvision.utils.seed import set_seed


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                 # (B,3,32,32) -> (B,3072)
            nn.Linear(3072, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10),           # 10 class scores (logits)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    set_seed(42)
    device = get_device()
    print("device:", device)
    train_loader, val_loader, _ = get_dataloaders(batch_size=128)   # test loader unused on purpose

    model = MLP().to(device)
    print("parameters:", sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()     # softmax + cross-entropy, takes raw logits
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   # optimizers explained in Phase 6

    for epoch in range(1, 4):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                correct += (model(xb).argmax(dim=1) == yb).sum().item()
        print(f"epoch {epoch} | train loss {running / 45000:.3f} | val acc {correct / 5000:.3f}")


if __name__ == "__main__":
    main()
