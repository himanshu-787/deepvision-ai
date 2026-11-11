"""Inspect CIFAR-10 and validate the data pipeline. Run from project root."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

from deepvision.data.dataloader import get_dataloaders
from deepvision.data.dataset import CLASS_NAMES, get_datasets
from deepvision.data.transforms import get_train_transform

ROOT = "data/raw"
PLOTS = Path("artifacts/plots")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    raw = datasets.CIFAR10(ROOT, train=True, download=True)
    images, targets = raw.data, np.array(raw.targets)

    # 1. Basic facts
    print("images array:", images.shape, images.dtype,
          "| pixel range:", images.min(), "-", images.max())

    # 2. Verify normalization constants
    scaled = images / 255.0
    print("channel mean:", scaled.mean(axis=(0, 1, 2)).round(4))
    print("channel std :", scaled.std(axis=(0, 1, 2)).round(4))

    # 3. Class distribution
    counts = np.bincount(targets)
    for name, c in zip(CLASS_NAMES, counts):
        print(f"{name:12s} {c}")

    # 4. Sample grid: 4 examples per class
    fig, axes = plt.subplots(4, 10, figsize=(15, 6.5))
    for c in range(10):
        idxs = np.where(targets == c)[0][:4]
        for r, i in enumerate(idxs):
            axes[r, c].imshow(images[i])
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(CLASS_NAMES[c], fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS / "cifar10_samples.png", dpi=120)
    plt.close()

    # 5. Augmentation preview: same image, 8 random augmentations
    aug = get_train_transform(normalize=False)
    pil_image = raw[0][0]
    fig, axes = plt.subplots(1, 8, figsize=(14, 2))
    for ax in axes:
        ax.imshow(aug(pil_image).permute(1, 2, 0).numpy())   # (C,H,W) -> (H,W,C)
        ax.axis("off")
    plt.savefig(PLOTS / "augmentation_preview.png", dpi=120)
    plt.close()

    # 6. Split integrity: sizes and zero overlap
    train_ds, val_ds, test_ds = get_datasets(ROOT)
    overlap = set(train_ds.indices) & set(val_ds.indices)
    print(f"train/val/test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")
    print("train/val overlap:", len(overlap))
    assert len(overlap) == 0, "DATA LEAKAGE: train and val overlap!"

    # 7. One batch
    train_loader, _, _ = get_dataloaders(ROOT, batch_size=128)
    xb, yb = next(iter(train_loader))
    print("batch images:", tuple(xb.shape), "| labels:", tuple(yb.shape))
    print(f"normalized batch mean={xb.mean():.3f} std={xb.std():.3f}")


if __name__ == "__main__":
    main()
