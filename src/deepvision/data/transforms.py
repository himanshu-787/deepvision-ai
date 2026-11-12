"""Image transforms for CIFAR-10 (optionally resized, with CIFAR or ImageNet normalization)."""

from torchvision import transforms

# Per-channel (R, G, B) statistics of the CIFAR-10 TRAINING set (verified in Phase 3).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
# Statistics the torchvision ImageNet-pretrained weights were trained with.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

NORM_STATS = {
    "cifar": (CIFAR10_MEAN, CIFAR10_STD),
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
}


def get_normalize(stats: str = "cifar") -> transforms.Normalize:
    mean, std = NORM_STATS[stats]
    return transforms.Normalize(mean, std)


def get_train_transform(
    normalize: bool = True, augment: bool = True, image_size: int = 32, stats: str = "cifar"
) -> transforms.Compose:
    """Augment at native 32x32 (cheap), then resize, then tensor + normalize."""
    steps = []
    if augment:
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if image_size != 32:
        steps.append(transforms.Resize(image_size))
    steps.append(transforms.ToTensor())
    if normalize:
        steps.append(get_normalize(stats))
    return transforms.Compose(steps)


def get_eval_transform(
    normalize: bool = True, image_size: int = 32, stats: str = "cifar"
) -> transforms.Compose:
    """Validation/test transform: deterministic only, no augmentation."""
    steps = []
    if image_size != 32:
        steps.append(transforms.Resize(image_size))
    steps.append(transforms.ToTensor())
    if normalize:
        steps.append(get_normalize(stats))
    return transforms.Compose(steps)
