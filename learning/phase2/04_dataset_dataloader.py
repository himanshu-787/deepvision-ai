"""Phase 2.4 - custom Dataset and DataLoader."""

import torch
from torch.utils.data import DataLoader, Dataset


class PointsDataset(Dataset):
    """Synthetic 2-D points. Label is 1 if x0 + x1 > 0 else 0."""

    def __init__(self, n: int = 100) -> None:
        self.x = torch.randn(n, 2)
        self.y = (self.x.sum(dim=1) > 0).long()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


if __name__ == "__main__":
    ds = PointsDataset(100)
    print("dataset size:", len(ds), "| sample 0:", ds[0])

    loader = DataLoader(ds, batch_size=16, shuffle=True)
    print("iterations per epoch:", len(loader))          # 7
    for i, (xb, yb) in enumerate(loader):
        print(f"batch {i}: x {tuple(xb.shape)}, y {tuple(yb.shape)}")

    drop = DataLoader(ds, batch_size=16, shuffle=True, drop_last=True)
    print("with drop_last=True:", len(drop))              # 6
