"""Save one CIFAR-10 test image per class to demo_images/ for demos and smoke tests."""

from pathlib import Path

from torchvision import datasets

from deepvision.data.dataset import CLASS_NAMES

if __name__ == "__main__":
    test = datasets.CIFAR10("data/raw", train=False, download=True)
    out = Path("demo_images")
    out.mkdir(exist_ok=True)
    seen = set()
    for image, label in test:
        if label not in seen:
            image.save(out / f"{CLASS_NAMES[label]}.png")
            seen.add(label)
        if len(seen) == 10:
            break
    print(f"saved {len(seen)} images to {out}/")
