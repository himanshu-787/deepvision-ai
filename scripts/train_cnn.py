"""Train the baseline CNN from scratch (Model A).

Smoke test : python scripts/train_cnn.py --epochs 1
Full run   : python scripts/train_cnn.py --epochs 15
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from deepvision.data.dataloader import get_dataloaders
from deepvision.data.dataset import CLASS_NAMES
from deepvision.evaluation.visualize import plot_history
from deepvision.models.factory import build_model
from deepvision.training.trainer import fit
from deepvision.utils.device import get_device
from deepvision.utils.model_info import count_parameters
from deepvision.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 from scratch")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-augment", action="store_true", help="disable training augmentation")
    p.add_argument("--run-name", type=str, default="cnn_baseline")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"device: {device} | args: {vars(args)}")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, augment=not args.no_augment,
    )   # the test loader is deliberately ignored until Phase 9

    model = build_model("simple_cnn", num_classes=len(CLASS_NAMES)).to(device)
    total, trainable = count_parameters(model)
    print(f"parameters: total={total:,} trainable={trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = Path("artifacts/checkpoints") / f"{args.run_name}_best.pt"
    history = fit(
        model, train_loader, val_loader, criterion, optimizer, device,
        epochs=args.epochs, checkpoint_path=ckpt_path,
        extra={"model_name": "simple_cnn", "class_names": CLASS_NAMES, "seed": args.seed},
    )

    best_epoch = max(range(args.epochs), key=lambda i: history["val_acc"][i]) + 1
    summary = {
        "run_name": args.run_name,
        "args": vars(args),
        "device": str(device),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "best_epoch": best_epoch,
        "best_val_acc": max(history["val_acc"]),
        "total_train_seconds": sum(history["epoch_seconds"]),
        "history": history,
    }
    metrics_path = Path("artifacts/metrics") / f"{args.run_name}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2))

    plot_history(history, Path("artifacts/plots") / f"{args.run_name}_curves.png", title=args.run_name)
    print(f"\nbest val acc {summary['best_val_acc']:.4f} at epoch {best_epoch}")
    print(f"saved: {ckpt_path}, {metrics_path}")


if __name__ == "__main__":
    main()
