"""Plots for training and evaluation."""

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _prepare(out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def plot_history(history: Dict[str, List[float]], out_path: Path, title: str = "") -> None:
    """Save loss and accuracy curves (train vs validation)."""
    out_path = _prepare(out_path)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.set_title("Loss"); ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="validation")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_title("Accuracy"); ax2.legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], out_path: Path,
                          normalize: bool = True, title: str = "") -> None:
    out_path = _prepare(out_path)
    cm = np.asarray(cm, dtype=float)
    data = cm / cm.sum(axis=1, keepdims=True) if normalize else cm
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, cmap="Blues", vmin=0)
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(n):
        for j in range(n):
            text = f"{data[i, j]:.2f}" if normalize else f"{int(data[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if data[i, j] > data.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_class_f1(report: Dict, class_names: Sequence[str], out_path: Path, title: str = "") -> None:
    out_path = _prepare(out_path)
    f1 = {c: report[c]["f1-score"] for c in class_names}
    items = sorted(f1.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([k for k, _ in items], [v for _, v in items])
    ax.set_xlim(0, 1); ax.set_xlabel("F1-score"); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_reliability(bin_conf: np.ndarray, bin_acc: np.ndarray, ece: float, out_path: Path, title: str = "") -> None:
    """Reliability diagram: points below the diagonal mean over-confidence."""
    out_path = _prepare(out_path)
    ok = ~np.isnan(bin_conf)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(bin_conf[ok], bin_acc[ok], "o-", label="model")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend()
    ax.set_title(f"{title}  ECE={ece:.3f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_misclassified(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray,
                       class_names: Sequence[str], out_path: Path, cols: int = 4) -> None:
    """images: (N,H,W,3) uint8 raw images."""
    out_path = _prepare(out_path)
    n = len(images)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, img, t, p, c in zip(np.atleast_1d(axes).ravel(), images, y_true, y_pred, conf):
        ax.imshow(img)
        ax.set_title(f"true: {class_names[t]}\npred: {class_names[p]} ({c:.0%})", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
