"""Evaluation metrics on cases with known answers."""

import numpy as np
import pytest

from deepvision.evaluation.metrics import compute_metrics, expected_calibration_error, top_confusions

NAMES = [f"c{i}" for i in range(10)]


def one_hot(pred, smooth=0.0):
    p = np.full((len(pred), 10), smooth / 9)
    p[np.arange(len(pred)), pred] = 1.0 - smooth
    return p


def test_perfect_predictions():
    y = np.repeat(np.arange(10), 3)
    summary, _, cm = compute_metrics(y, y, one_hot(y, smooth=0.1), NAMES)
    assert summary["accuracy"] == 1.0 and summary["f1_macro"] == 1.0 and summary["roc_auc_macro_ovr"] == 1.0
    assert np.trace(cm) == len(y)


def test_ece_of_confident_wrong_model_is_one():
    y_true, y_pred = np.zeros(20, dtype=int), np.ones(20, dtype=int)
    ece, *_ = expected_calibration_error(y_true, y_pred, one_hot(y_pred))
    assert ece == pytest.approx(1.0)


def test_top_confusions_ordering():
    cm = np.zeros((10, 10), dtype=int)
    cm[3, 5], cm[1, 2] = 7, 4
    assert top_confusions(cm, NAMES, k=2) == [("c3", "c5", 7), ("c1", "c2", 4)]
