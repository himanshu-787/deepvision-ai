"""Save run summaries (JSON) and curves (PNG) in one consistent format."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from deepvision.evaluation.visualize import plot_history
from deepvision.utils.model_info import count_parameters


def merge_histories(a: Dict[str, List[float]], b: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Concatenate two training histories (e.g. head stage + fine-tuning stage)."""
    return {k: a.get(k, []) + b.get(k, []) for k in a}


def save_run_summary(
    run_name: str,
    model: nn.Module,
    history: Dict[str, List[float]],
    device: torch.device,
    config: Dict[str, Any],
    extra_fields: Optional[Dict[str, Any]] = None,
    metrics_dir: str = "artifacts/metrics",
    plots_dir: str = "artifacts/plots",
) -> Dict[str, Any]:
    total, trainable = count_parameters(model)
    best = max(range(len(history["val_acc"])), key=lambda i: history["val_acc"][i])
    summary = {
        "run_name": run_name,
        "config": config,
        "device": str(device),
        "total_parameters": total,
        "trainable_parameters": trainable,    # state at the END of training
        "epochs_run": len(history["val_acc"]),
        "best_epoch": best + 1,
        "best_val_acc": history["val_acc"][best],
        "train_acc_at_best": history["train_acc"][best],
        "train_loss_at_best": history["train_loss"][best],
        "val_loss_at_best": history["val_loss"][best],
        "total_train_seconds": sum(history["epoch_seconds"]),
        "history": history,
        **(extra_fields or {}),
    }
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    (Path(metrics_dir) / f"{run_name}.json").write_text(json.dumps(summary, indent=2))
    plot_history(history, Path(plots_dir) / f"{run_name}_curves.png", title=run_name)
    return summary
