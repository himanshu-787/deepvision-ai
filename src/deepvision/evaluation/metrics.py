"""Prediction collection and metric computation."""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (y_true, y_pred, y_prob, avg_loss) in loader order (so use shuffle=False loaders)."""
    model.eval()
    labels_all, probs_all, loss_sum, seen = [], [], 0.0, 0
    for images, labels in loader:
        logits = model(images.to(device)).float().cpu()
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        probs_all.append(torch.softmax(logits, dim=1))
        labels_all.append(labels)
        seen += labels.size(0)
    y_true = torch.cat(labels_all).numpy()
    y_prob = torch.cat(probs_all).numpy().astype(np.float64)
    y_prob /= y_prob.sum(axis=1, keepdims=True)           # exact row sums for roc_auc_score
    return y_true, y_prob.argmax(axis=1), y_prob, loss_sum / seen


def expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (ECE, mean_confidence_per_bin, accuracy_per_bin, count_per_bin); empty bins are NaN."""
    conf = y_prob.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf, bin_acc, bin_count, ece = [], [], [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        n = int(mask.sum())
        bin_count.append(n)
        if n == 0:
            bin_conf.append(np.nan); bin_acc.append(np.nan)
            continue
        bin_conf.append(conf[mask].mean()); bin_acc.append(correct[mask].mean())
        ece += (n / len(conf)) * abs(bin_acc[-1] - bin_conf[-1])
    return float(ece), np.array(bin_conf), np.array(bin_acc), np.array(bin_count)


def top_confusions(cm: np.ndarray, class_names: Sequence[str], k: int = 5) -> List[Tuple[str, str, int]]:
    """Most frequent (true -> predicted) mistakes."""
    n = len(class_names)
    pairs = [(class_names[i], class_names[j], int(cm[i, j])) for i in range(n) for j in range(n) if i != j]
    return sorted(pairs, key=lambda t: -t[2])[:k]


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: Sequence[str]
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    """Return (summary_metrics, per_class_report, confusion_matrix)."""
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    ece = expected_calibration_error(y_true, y_pred, y_prob)[0]
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f_macro),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
        "roc_auc_macro_ovr": float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")),
        "ece": ece,
        "mean_confidence": float(y_prob.max(axis=1).mean()),
    }
    report = classification_report(y_true, y_pred, target_names=list(class_names),
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return summary, report, cm
