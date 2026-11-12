"""Training and evaluation loops.

v2 adds: LR scheduler, mixed precision, gradient clipping, gradient accumulation,
early stopping, resumable epoch numbering and an epoch callback (used by Phase 10 tracking).
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepvision.training.checkpoint import save_checkpoint

EpochCallback = Callable[[int, Dict[str, float]], None]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional["torch.amp.GradScaler"] = None,
    use_amp: bool = False,
    grad_clip: Optional[float] = None,
    accum_steps: int = 1,
) -> Tuple[float, float]:
    """One training epoch. Returns (average_loss, accuracy)."""
    model.train()
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=False)   # disabled scaler = pass-through
    total_loss, correct, seen = 0.0, 0, 0
    n_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)

    for i, (images, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        # Divide by accum_steps so the summed gradients equal one big-batch average.
        scaler.scale(loss / accum_steps).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == n_batches:
            if grad_clip is not None:
                scaler.unscale_(optimizer)        # clip real gradients, not scaled ones
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """Evaluate without updating weights. Returns (average_loss, accuracy)."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)
    return total_loss / seen, correct / seen


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
    scheduler: Optional[Any] = None,
    amp: bool = False,
    grad_clip: Optional[float] = None,
    accum_steps: int = 1,
    patience: Optional[int] = None,
    start_epoch: int = 0,
    best_val_acc: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    epoch_callback: Optional[EpochCallback] = None,
) -> Dict[str, List[float]]:
    """Train for up to `epochs`; checkpoint the best validation accuracy.

    start_epoch / best_val_acc let a second training stage (fine-tuning) continue
    the numbering and the 'best so far' bookkeeping of the first stage.
    patience: stop after this many epochs without a new best (None = never).
    The recorded 'lr' is the LR of the first parameter group during that epoch.
    """
    use_amp = amp and device.type == "cuda"
    if amp and not use_amp:
        print("AMP requested but no CUDA GPU found: running in full precision.")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: Dict[str, List[float]] = {
        k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_seconds", "lr")
    }
    stale = 0
    for epoch in range(1, epochs + 1):
        global_epoch = start_epoch + epoch
        lr = optimizer.param_groups[0]["lr"]
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp, grad_clip, accum_steps
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)
        if scheduler is not None:
            scheduler.step()
        seconds = time.perf_counter() - start

        row = {"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss,
               "val_acc": val_acc, "epoch_seconds": seconds, "lr": lr}
        for key, value in row.items():
            history[key].append(value)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc, stale = val_acc, 0
            save_checkpoint(checkpoint_path, model, optimizer, global_epoch, best_val_acc, extra)
            marker = "  <- best, checkpoint saved"
        else:
            stale += 1
        print(
            f"epoch {global_epoch:2d} | lr {lr:.2e} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | {seconds:.0f}s{marker}"
        )
        if epoch_callback is not None:
            epoch_callback(global_epoch, row)
        if patience is not None and stale >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break
    return history
