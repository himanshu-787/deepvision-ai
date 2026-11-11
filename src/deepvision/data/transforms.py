"""Image transforms for CIFAR-10."""

from torchvision import transforms

# Per-channel (R, G, B) statistics of the CIFAR-10 TRAINING set.
# Verify them yourself with scripts/explore_data.py (step 3.4).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_train_transform(normalize: bool = True, augment: bool = True) -> transforms.Compose:
    """Training transform: optional augmentation, then tensor conversion and normalization."""
    steps = []
    if augment:
        steps += [
            transforms.RandomCrop(32, padding=4),   # pad 4px, crop 32x32 at a random spot
            transforms.RandomHorizontalFlip(),      # mirror with probability 0.5
        ]
    steps.append(transforms.ToTensor())             # PIL (H,W,C) uint8 -> tensor (C,H,W) in [0,1]
    if normalize:
        steps.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return transforms.Compose(steps)


def get_eval_transform(normalize: bool = True) -> transforms.Compose:
    """Validation/test transform: NO augmentation, only deterministic steps."""
    steps = [transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return transforms.Compose(steps)
