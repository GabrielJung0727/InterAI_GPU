"""Feature standardisation fitted on the training split only (numpy)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def fit_normalizer(train_features: np.ndarray) -> dict[str, Any]:
    """Per-feature mean/std over the last axis, ignoring NaN.

    Features that are entirely missing in the training split get mean 0 /
    std 1, so they become constant zeros after :func:`apply_normalizer`.
    """
    flat = train_features.reshape(-1, train_features.shape[-1]).astype(np.float64)
    observed = np.isfinite(flat)
    flat = np.where(observed, flat, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
        mean = np.where(observed.any(axis=0), np.nanmean(flat, axis=0), 0.0)
        std = np.where(observed.sum(axis=0) > 1, np.nanstd(flat, axis=0), 1.0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist()}


def apply_normalizer(features: np.ndarray, normalizer: dict[str, Any]) -> np.ndarray:
    """Standardise and impute missing values with the training mean (-> 0)."""
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    std = np.asarray(normalizer["std"], dtype=np.float64)
    scaled = (features.astype(np.float64) - mean) / std
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
