"""Leakage-free dataset splitting."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from interview_gpu.common.errors import EmptyDatasetError


def group_split(
    groups: Sequence[str], val_ratio: float, test_ratio: float, seed: int
) -> dict[str, np.ndarray]:
    """Split sample indices into train/val/test so that no group is shared.

    Groups are speakers, candidates or sessions: every sample of one person
    lands in exactly one split, preventing identity leakage.

    Args:
        groups: Group id per sample.
        val_ratio, test_ratio: Fraction of *groups* assigned to each split.
        seed: Shuffling seed (the split is fully deterministic given it).

    Returns:
        ``{"train": idx, "val": idx, "test": idx}`` with sorted index arrays.
    """
    groups = np.asarray(groups).astype(str)
    if groups.size == 0:
        raise EmptyDatasetError("Cannot split an empty dataset.")
    unique = np.unique(groups)
    needed = 1 + int(val_ratio > 0) + int(test_ratio > 0)
    if len(unique) < needed:
        raise EmptyDatasetError(
            f"Need at least {needed} distinct groups (speakers/sessions) for a "
            f"train/val/test split, found {len(unique)}."
        )
    order = np.random.default_rng(seed).permutation(unique)
    n_val = max(1, round(len(order) * val_ratio)) if val_ratio > 0 else 0
    n_test = max(1, round(len(order) * test_ratio)) if test_ratio > 0 else 0
    while n_val + n_test >= len(order):  # always keep at least one training group
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        else:
            n_val -= 1
    assignment = {
        "val": set(order[:n_val]),
        "test": set(order[n_val : n_val + n_test]),
    }
    in_val = np.isin(groups, list(assignment["val"]))
    in_test = np.isin(groups, list(assignment["test"]))
    return {
        "train": np.flatnonzero(~in_val & ~in_test),
        "val": np.flatnonzero(in_val),
        "test": np.flatnonzero(in_test),
    }
