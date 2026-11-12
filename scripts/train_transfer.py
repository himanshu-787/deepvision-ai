"""Feature extraction: pretrained ResNet18, frozen backbone, new trainable head (Model B).

    python scripts/train_transfer.py --run-name r18_head --image-size 64 --epochs 10
"""

import argparse
from pathlib import Path

import torch.nn as nn

from deepvision.data.dataloader import get_dataloaders
from deepvision.data.dataset import CLASS_NAMES
from deepvision.models.factory import build_model
from deepvision.training.optimizer import build_optimizer
from deepvision.training.reporting import save_run_summary
from deepvision.training.scheduler import build_scheduler
from deepvision.training.trainer import fit
from deepvision.utils.device import get_device
from deepvision.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet18 feature extraction on CIFAR-10")
    p.add_argument("--run-name", required=True)
    p.add_argument("--image-size", type=int, default=64, help="CIFAR images are upscaled to this size")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optimizer", default="adam", choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", default="none", choices=["none", "cosine", "step"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"device: {device} | args: {vars(args)}")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
        augment=not args.no_augment, image_size=args.image_size, stats="imagenet",
    )
    model_kwargs = {"pretrained": True}
    model = build_model("resnet18", num_classes=len(CLASS_NAMES), **model_kwargs).to(device)
    model.set_trainable(["fc"])                       # freeze the backbone

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable tensors: {len(trainable_params)} "
          f"({sum(p.numel() for p in trainable_params):,} parameters)")

    optimizer = build_optimizer(args.optimizer, trainable_params, args.lr, args.weight_decay)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)
    extra = {
        "model_name": "resnet18", "model_kwargs": model_kwargs, "image_size": args.image_size,
        "stats": "imagenet", "class_names": CLASS_NAMES, "seed": args.seed,
    }
    history = fit(
        model, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer, device, args.epochs,
        Path("artifacts/checkpoints") / f"{args.run_name}_best.pt",
        scheduler=scheduler, amp=args.amp, patience=args.patience, extra=extra,
    )
    cfg = {**vars(args), "model": "resnet18", "strategy": "head_only"}
    s = save_run_summary(args.run_name, model, history, device, config=cfg)
    print(f"\nparams {s['total_parameters']:,} (trainable {s['trainable_parameters']:,}) | "
          f"best val acc {s['best_val_acc']:.4f} at epoch {s['best_epoch']} | "
          f"train time {s['total_train_seconds']:.0f}s")


if __name__ == "__main__":
    main()
