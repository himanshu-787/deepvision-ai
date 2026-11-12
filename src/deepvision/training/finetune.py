"""Two-stage fine-tuning: (1) train the new head, (2) unfreeze backbone stages with
discriminative learning rates."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch.nn as nn
from torch.utils.data import DataLoader

from deepvision.models.resnet import ResNet18Classifier
from deepvision.training.optimizer import build_optimizer
from deepvision.training.reporting import merge_histories
from deepvision.training.scheduler import build_scheduler
from deepvision.training.trainer import fit


def build_finetune_groups(
    model: ResNet18Classifier,
    stages: Sequence[str],
    head_lr: float,
    backbone_lr: float,
    lr_decay: float,
) -> List[Dict[str, Any]]:
    """Param groups: fc gets head_lr; the deepest listed stage gets backbone_lr; each
    earlier stage gets lr_decay times the LR of the stage after it."""
    groups = [{"params": model.stage_parameters("fc"), "lr": head_lr, "name": "fc"}]
    ordered = [s for s in ResNet18Classifier.STAGES if s in stages and s != "fc"]   # input -> output order
    for i, stage in enumerate(ordered):
        lr = backbone_lr * (lr_decay ** (len(ordered) - 1 - i))
        groups.append({"params": model.stage_parameters(stage), "lr": lr, "name": stage})
    return groups


def two_stage_finetune(
    model: ResNet18Classifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device,
    *,
    stages: Sequence[str],
    head_epochs: int,
    ft_epochs: int,
    head_lr: float,
    backbone_lr: float,
    lr_decay: float,
    weight_decay: float,
    checkpoint_path: Path,
    patience: Optional[int] = None,
    amp: bool = False,
    grad_clip: Optional[float] = None,
    accum_steps: int = 1,
    extra: Optional[Dict[str, Any]] = None,
    epoch_callback: Optional[Callable] = None,
) -> Dict[str, List[float]]:
    """Stage 1: head only (Adam). Stage 2: head + `stages` (AdamW, discriminative LRs, cosine).
    The best-validation checkpoint is tracked across both stages."""
    common = dict(amp=amp, grad_clip=grad_clip, accum_steps=accum_steps, extra=extra, epoch_callback=epoch_callback)
    history, best, done = None, 0.0, 0

    if head_epochs > 0:
        print(f"--- Stage 1: train head only for {head_epochs} epochs ---")
        model.set_trainable(["fc"])
        opt = build_optimizer("adam", [p for p in model.parameters() if p.requires_grad], head_lr)
        history = fit(model, train_loader, val_loader, criterion, opt, device, head_epochs,
                      checkpoint_path, patience=patience, **common)
        best, done = max(history["val_acc"]), len(history["val_acc"])

    if ft_epochs > 0:
        print(f"--- Stage 2: unfreeze {list(stages)} for {ft_epochs} epochs ---")
        model.set_trainable(["fc", *stages])
        groups = build_finetune_groups(model, stages, head_lr, backbone_lr, lr_decay)
        for g in groups:
            print(f"  group {g['name']:7s} lr={g['lr']:.2e}")
        opt = build_optimizer("adamw", groups, lr=backbone_lr, weight_decay=weight_decay)
        sched = build_scheduler("cosine", opt, ft_epochs)
        h2 = fit(model, train_loader, val_loader, criterion, opt, device, ft_epochs, checkpoint_path,
                 scheduler=sched, patience=patience, start_epoch=done, best_val_acc=best, **common)
        history = merge_histories(history, h2) if history else h2

    if history is None:
        raise ValueError("Nothing to train: head_epochs and ft_epochs are both 0.")
    return history
