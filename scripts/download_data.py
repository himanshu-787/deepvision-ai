"""Download CIFAR-10 into data/raw (about 170 MB). Run from project root."""

from torchvision import datasets

if __name__ == "__main__":
    train = datasets.CIFAR10("data/raw", train=True, download=True)
    test = datasets.CIFAR10("data/raw", train=False, download=True)
    print(f"train images: {len(train)} | test images: {len(test)}")
