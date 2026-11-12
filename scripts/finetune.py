"""Fine-tune a pretrained ResNet18 (Model C).

    python scripts/finetune.py --run-name r18_ft_l34 --strategy partial
"""

import argparse
from pathlib import Path

import torch.nn as nn

from deepvision.data.dataloader import get_dataloaders
from deepvision.data.dataset import CLASS_NAMES
from deepvision.models.factory import build_model
from deepvision.training.checkpoint import load_checkpoint
from deepvision.training.finetune import two_stage_finetune
from deepvision.training.reporting import save_run_summary
from deepvision.utils.device import get_device
from deepvision.utils.seed import set_seed

FULL_STAGES = ["stem", "layer1", "layer2", "layer3", "layer4"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune ResNet18 on CIFAR-10")
    p.add_argument("--run-name", required=True)
    p.add_argument("--strategy", choices=["partial", "full"], default="partial")
    p.add_argument("--stages", nargs="+", default=["layer3", "layer4"],
                   help="stages to unfreeze for --strategy partial")
    p.add_argument("--head-epochs", type=int, default=3)
    p.add_argument("--ft-epochs", type=int, default=7)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr", type=float, default=1e-4, help="LR of the deepest unfrozen stage")
    p.add_argument("--lr-decay", type=float, default=0.5, help="LR multiplier per stage toward the input")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--init-from", type=str, default=None,
                   help="head-trained checkpoint (e.g. r18_head_best.pt); skips stage 1")
    p.add_argument("--no-pretrained", action="store_true", help="random init ablation (full strategy)")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--accum-steps", type=int, default=1)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--no-augment", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    stages = FULL_STAGES if args.strategy == "full" else args.stages
    head_epochs = args.head_epochs

    if args.no_pretrained:
        stages, head_epochs = FULL_STAGES, 0
        print("Random-init ablation: training all stages from scratch. Consider "
              "--head-lr 1e-3 --backbone-lr 1e-3 --lr-decay 1.0 (tiny LRs suit pretrained weights only).")
    print(f"device: {device} | stages to unfreeze: {stages} | args: {vars(args)}")

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed,
        augment=not args.no_augment, image_size=args.image_size, stats="imagenet",
    )
    model_kwargs = {"pretrained": not args.no_pretrained}
    model = build_model("resnet18", num_classes=len(CLASS_NAMES), **model_kwargs).to(device)

    if args.init_from:
        ckpt = load_checkpoint(Path(args.init_from), model, device=str(device))
        head_epochs = 0
        print(f"Loaded {args.init_from} (val acc {ckpt['best_val_acc']:.4f}); skipping stage 1.")

    extra = {
        "model_name": "resnet18", "model_kwargs": model_kwargs, "image_size": args.image_size,
        "stats": "imagenet", "class_names": CLASS_NAMES, "seed": args.seed,
    }
    history = two_stage_finetune(
        model, train_loader, val_loader, nn.CrossEntropyLoss(), device,
        stages=stages, head_epochs=head_epochs, ft_epochs=args.ft_epochs,
        head_lr=args.head_lr, backbone_lr=args.backbone_lr, lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        checkpoint_path=Path("artifacts/checkpoints") / f"{args.run_name}_best.pt",
        patience=args.patience, amp=args.amp, grad_clip=args.grad_clip,
        accum_steps=args.accum_steps, extra=extra,
    )
    cfg = {**vars(args), "model": "resnet18", "strategy": f"finetune_{args.strategy}"}
    s = save_run_summary(args.run_name, model, history, device, config=cfg)
    print(f"\nbest val acc {s['best_val_acc']:.4f} at epoch {s['best_epoch']} | "
          f"train time {s['total_train_seconds']:.0f}s | trainable at end {s['trainable_parameters']:,}")


if __name__ == "__main__":
    main()
