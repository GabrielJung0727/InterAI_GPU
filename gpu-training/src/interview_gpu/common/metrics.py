"""Evaluation metrics: classification, regression, multi-output and calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn import metrics as skm


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: confidence/accuracy gap averaged over confidence bins."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score (mean squared error against one-hot labels)."""
    one_hot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def classification_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    """Standard + balanced + calibration metrics for single-label classification.

    Args:
        y_true: Integer class ids, shape ``[N]``.
        probs: Class probabilities, shape ``[N, C]``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = probs.argmax(axis=1)
    labels = np.arange(probs.shape[1])
    result: dict[str, Any] = {
        "accuracy": float(skm.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(skm.f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "precision": float(
            skm.precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "recall": float(skm.recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "ece": expected_calibration_error(y_true, probs),
        "brier": brier_score(y_true, probs),
    }
    # Ranking metrics are undefined when a class is absent from y_true.
    one_hot = np.eye(probs.shape[1])[y_true]
    present = one_hot.sum(axis=0) > 0
    if present.sum() >= 2 and present.all():
        if probs.shape[1] == 2:
            result["auroc"] = float(skm.roc_auc_score(y_true, probs[:, 1]))
        else:
            result["auroc"] = float(skm.roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        result["auprc"] = float(skm.average_precision_score(one_hot, probs, average="macro"))
    else:
        result["auroc"] = None
        result["auprc"] = None
    return result


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """MAE, RMSE, R² and Spearman correlation for one output."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    count = int(y_true.size)
    if count == 0:
        return {"mae": None, "rmse": None, "r2": None, "spearman": None, "count": 0}
    error = y_pred - y_true
    constant = np.ptp(y_true) == 0 or np.ptp(y_pred) == 0
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(skm.r2_score(y_true, y_pred)) if count >= 2 and np.ptp(y_true) > 0 else None,
        "spearman": None if count < 3 or constant else float(stats.spearmanr(y_true, y_pred).statistic),
        "count": count,
    }


def multi_output_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray, names: list[str]
) -> dict[str, Any]:
    """Per-head metrics plus their unweighted mean.

    Args:
        y_true, y_pred: Arrays of shape ``[N, H]`` in original label units.
        mask: Boolean ``[N, H]``; ``False`` marks a missing label.
        names: Head names, length ``H``.
    """
    per_head = {
        name: regression_metrics(y_true[mask[:, i], i], y_pred[mask[:, i], i])
        for i, name in enumerate(names)
    }
    mean: dict[str, Any] = {}
    for key in ("mae", "rmse", "r2", "spearman"):
        values = [head[key] for head in per_head.values() if head[key] is not None]
        mean[key] = float(np.mean(values)) if values else None
    return {"mean": mean, "per_head": per_head}


def error_distribution(y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.9) -> dict[str, Any]:
    """Residual statistics usable as an empirical prediction interval.

    ``interval_low``/``interval_high`` are residual quantiles: adding them to a
    new prediction gives an interval that covered ``coverage`` of this split.
    """
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if residual.size == 0:
        return {"count": 0}
    tail = (1.0 - coverage) / 2.0
    absolute = np.abs(residual)
    return {
        "count": int(residual.size),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "abs_error_p50": float(np.quantile(absolute, 0.5)),
        "abs_error_p90": float(np.quantile(absolute, 0.9)),
        "abs_error_p95": float(np.quantile(absolute, 0.95)),
        "coverage": coverage,
        "interval_low": float(np.quantile(residual, tail)),
        "interval_high": float(np.quantile(residual, 1.0 - tail)),
    }


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested metric dicts: ``{"mean": {"mae": 1}}`` -> ``{"mean_mae": 1}``."""
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f"{name}_"))
        else:
            flat[name] = value
    return flat
