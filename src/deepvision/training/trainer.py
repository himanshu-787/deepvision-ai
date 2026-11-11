"""Training and evaluation loops."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepvision.training.checkpoint import save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """One pass over the training data. Returns (average_loss, accuracy)."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)   # (B,3,32,32), (B,)
        optimizer.zero_grad()
        logits = model(images)                                   # (B,10)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)   # weight by batch size (last batch is smaller)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += images.size(0)
    return total_loss / seen, correct / seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate without updating weights. Returns (average_loss, accuracy)."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
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
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[float]]:
    """Train for `epochs`, saving the checkpoint with the best validation accuracy."""
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "epoch_seconds": [],
    }
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        seconds = time.perf_counter() - start

        for key, value in zip(history, (train_loss, train_acc, val_loss, val_acc, seconds)):
            history[key].append(value)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_acc, extra)
            marker = "  <- best, checkpoint saved"
        print(
            f"epoch {epoch:2d}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | {seconds:.0f}s{marker}"
        )
    return history
