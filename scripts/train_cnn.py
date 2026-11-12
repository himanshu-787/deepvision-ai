"""Train a CNN from scratch with configurable training strategy.

Reproduce the Phase 5 baseline:
    python scripts/train_cnn.py --model simple_cnn --run-name cnn_baseline
(Skip that if you already ran it in Phase 5.)
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
    p = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 from scratch")
    p.add_argument("--run-name", required=True, help="unique name; used for output files")
    p.add_argument("--model", default="simple_cnn", choices=["simple_cnn", "improved_cnn"])
    p.add_argument("--dropout", type=float, default=0.3, help="improved_cnn only")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optimizer", default="adam", choices=["sgd", "adam", "adamw"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD only")
    p.add_argument("--scheduler", default="none", choices=["none", "cosine", "step"])
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--patience", type=int, default=None, help="early stopping patience")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"device: {device} | args: {vars(args)}")

    train_loader, val_loader, _ = get_dataloaders(   # test loader deliberately unused
        batch_size=args.batch_size, num_workers=args.num_workers,
        seed=args.seed, augment=not args.no_augment,
    )
    model_kwargs = {"dropout": args.dropout} if args.model == "improved_cnn" else {}
    model = build_model(args.model, num_classes=len(CLASS_NAMES), **model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay, args.momentum)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)

    extra = {
        "model_name": args.model, "model_kwargs": model_kwargs, "image_size": 32,
        "stats": "cifar", "class_names": CLASS_NAMES, "seed": args.seed,
    }
    history = fit(
        model, train_loader, val_loader, criterion, optimizer, device, args.epochs,
        Path("artifacts/checkpoints") / f"{args.run_name}_best.pt",
        scheduler=scheduler, amp=args.amp, grad_clip=args.grad_clip,
        accum_steps=args.accum_steps, patience=args.patience, extra=extra,
    )
    s = save_run_summary(args.run_name, model, history, device, config=vars(args))
    print(f"\nparams {s['total_parameters']:,} | best val acc {s['best_val_acc']:.4f} at epoch "
          f"{s['best_epoch']} | train time {s['total_train_seconds']:.0f}s")


if __name__ == "__main__":
    main()
