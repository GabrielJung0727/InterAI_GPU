"""Task definition shared by every model: targets, splits, losses and metrics.

All three domains (speech / video / answer) reduce to the same supervised
problem — a table of samples with ``label_*`` columns and a group column — so
target validation, group splitting, target standardisation, loss construction
and metric computation live here once.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import softmax

from interview_gpu.common.config import DataConfig, TrainingConfig
from interview_gpu.common.errors import EmptyDatasetError, InvalidLabelError
from interview_gpu.common.logging import get_logger
from interview_gpu.common.metrics import (
    classification_metrics,
    error_distribution,
    multi_output_regression_metrics,
)
from interview_gpu.datasets.splits import group_split

LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]

LABEL_PREFIX = "label_"


@dataclass
class TaskSpec:
    """What the model predicts and how to map outputs back to label units."""

    kind: str  # "regression" | "classification"
    target_names: list[str]
    class_names: list[str] | None = None
    target_mean: list[float] | None = None
    target_std: list[float] | None = None

    @property
    def output_dim(self) -> int:
        if self.kind == "classification":
            return len(self.class_names or [])
        return len(self.target_names)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        """Convert standardised regression outputs back to label units."""
        if self.kind != "regression" or self.target_mean is None:
            return values
        return values * np.asarray(self.target_std) + np.asarray(self.target_mean)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        return cls(**data)


@dataclass
class Supervision:
    """Validated targets and leakage-free splits for one processed table."""

    frame: pd.DataFrame
    #: regression: float32 ``[N, H]`` standardised, NaN = missing label.
    #: classification: int64 ``[N]``.
    targets: np.ndarray
    task: TaskSpec
    splits: dict[str, np.ndarray]
    class_counts: np.ndarray | None = None

    def target_tensors(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(target, mask)`` tensors for one row."""
        if self.task.kind == "classification":
            return torch.tensor(int(self.targets[row])), torch.tensor(1.0)
        values = self.targets[row]
        mask = np.isfinite(values)
        return (
            torch.from_numpy(np.nan_to_num(values, nan=0.0).astype(np.float32)),
            torch.from_numpy(mask.astype(np.float32)),
        )


def resolve_target_columns(data_cfg: DataConfig) -> dict[str, str]:
    """Normalise ``target_columns`` into ``{output_name: column}``."""
    columns = data_cfg.target_columns
    if isinstance(columns, dict):
        return dict(columns)
    return {column.removeprefix(LABEL_PREFIX): column for column in columns}


def prepare_supervision(frame: pd.DataFrame, data_cfg: DataConfig, seed: int) -> Supervision:
    """Validate labels, split by group and fit target statistics on train only.

    Raises:
        EmptyDatasetError: The table has no rows.
        InvalidLabelError: No targets configured, columns missing, non-numeric
            regression labels, or no labeled rows at all.
    """
    if len(frame) == 0:
        raise EmptyDatasetError("The processed table contains no samples.")
    mapping = resolve_target_columns(data_cfg)
    available = sorted(c for c in frame.columns if c.startswith(LABEL_PREFIX))
    if not mapping:
        raise InvalidLabelError(
            "No target columns configured (data.target_columns is empty).\n"
            f"Label columns available in the processed data: {available or 'none'}.\n"
            "A model cannot be trained without supervised labels; add labels to the dataset "
            "and list them under data.target_columns."
        )
    missing = [column for column in mapping.values() if column not in frame.columns]
    if missing:
        raise InvalidLabelError(
            f"Target column(s) {missing} not found in the processed data.\n"
            f"Available label columns: {available or 'none'}."
        )

    names = list(mapping)
    columns = list(mapping.values())
    class_names: list[str] | None = None

    if data_cfg.task == "classification":
        if len(columns) != 1:
            raise InvalidLabelError("Classification expects exactly one target column.")
        frame = frame[frame[columns[0]].notna()].reset_index(drop=True)
        if len(frame) == 0:
            raise InvalidLabelError(f"No labeled samples in column '{columns[0]}'.")
        labels = frame[columns[0]].astype(str)
        class_names = sorted(labels.unique())
        if len(class_names) < 2:
            raise InvalidLabelError(
                f"Classification needs at least 2 classes; column '{columns[0]}' has {class_names}."
            )
        targets = labels.map({name: i for i, name in enumerate(class_names)}).to_numpy(dtype=np.int64)
    else:
        raw = frame[columns]
        numeric = raw.apply(pd.to_numeric, errors="coerce")
        bad = (numeric.isna() & raw.notna()).sum()
        if bad.any():
            raise InvalidLabelError(
                "Non-numeric regression labels found: "
                + ", ".join(f"{col} ({int(n)} rows)" for col, n in bad.items() if n)
            )
        values = numeric.to_numpy(dtype=np.float64, copy=True)  # pandas may return a read-only view
        values[~np.isfinite(values)] = np.nan
        keep = np.isfinite(values).any(axis=1)
        if not keep.any():
            raise InvalidLabelError(f"No labeled samples: columns {columns} are entirely empty.")
        if not keep.all():
            get_logger().warning("Dropping %d sample(s) without any label.", int((~keep).sum()))
        frame = frame[keep].reset_index(drop=True)
        targets = values[keep]

    group_column = data_cfg.split.group_column
    if group_column in frame.columns:
        groups = frame[group_column].astype(str).to_numpy()
    else:
        get_logger().warning(
            "Group column '%s' not found: falling back to a per-sample split. "
            "The same speaker may leak across train/val/test.",
            group_column,
        )
        groups = np.arange(len(frame)).astype(str)
    splits = group_split(groups, data_cfg.split.val_ratio, data_cfg.split.test_ratio, seed)

    task = TaskSpec(kind=data_cfg.task, target_names=names, class_names=class_names)
    class_counts = None
    if task.kind == "classification":
        class_counts = np.bincount(targets[splits["train"]], minlength=len(class_names or []))
    else:
        train = targets[splits["train"]]
        observed = np.isfinite(train)
        mean = np.where(observed.any(axis=0), np.nanmean(np.where(observed, train, np.nan), axis=0), 0.0)
        std = np.where(observed.sum(axis=0) > 1, np.nanstd(np.where(observed, train, np.nan), axis=0), 1.0)
        std = np.where(std < 1e-6, 1.0, std)
        task.target_mean = [float(v) for v in mean]
        task.target_std = [float(v) for v in std]
        targets = ((targets - mean) / std).astype(np.float32)

    return Supervision(frame=frame, targets=targets, task=task, splits=splits, class_counts=class_counts)


def class_weights(class_counts: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean ~1."""
    counts = np.maximum(class_counts.astype(np.float64), 1.0)
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def sample_weights(labels: np.ndarray, class_counts: np.ndarray) -> torch.Tensor:
    """Per-sample weights for ``WeightedRandomSampler``."""
    return torch.tensor(1.0 / np.maximum(class_counts[labels], 1), dtype=torch.double)


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Multi-class focal loss (Lin et al., 2017)."""
    log_probs = F.log_softmax(logits, dim=-1)
    ce = F.nll_loss(log_probs, targets, weight=weight, reduction="none")
    pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1).exp()
    return ((1.0 - pt) ** gamma * ce).mean()


def masked_regression_loss(
    outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, kind: str = "huber"
) -> torch.Tensor:
    """Element-wise regression loss averaged over observed labels only."""
    fn = {"huber": F.smooth_l1_loss, "mse": F.mse_loss, "l1": F.l1_loss}[kind]
    per_element = fn(outputs, targets, reduction="none")
    mask = mask.to(per_element.dtype)
    return (per_element * mask).sum() / mask.sum().clamp_min(1.0)


def build_loss_fn(
    task: TaskSpec, training_cfg: TrainingConfig, class_counts: np.ndarray | None = None
) -> LossFn:
    """Create ``loss(outputs, targets, mask)`` for the task."""
    if task.kind == "regression":
        return lambda outputs, targets, mask: masked_regression_loss(
            outputs, targets, mask, training_cfg.loss
        )

    imbalance = training_cfg.imbalance
    weight = class_weights(class_counts) if imbalance.class_weights and class_counts is not None else None

    def classification_loss(outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = weight.to(outputs.device) if weight is not None else None
        if imbalance.focal_loss:
            return focal_loss(outputs, targets, imbalance.focal_gamma, w)
        return F.cross_entropy(outputs, targets, weight=w)

    return classification_loss


def build_metric_fn(task: TaskSpec) -> MetricFn:
    """Create ``metrics(outputs, targets, mask)`` operating on numpy arrays."""
    if task.kind == "classification":
        return lambda outputs, targets, mask: classification_metrics(targets, softmax(outputs, axis=1))
    return lambda outputs, targets, mask: multi_output_regression_metrics(
        task.denormalize(targets), task.denormalize(outputs), mask.astype(bool), task.target_names
    )


def summarize_predictions(
    task: TaskSpec, outputs: np.ndarray, targets: np.ndarray, mask: np.ndarray, sample_ids: list[str]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calibration-oriented statistics and a per-sample prediction table."""
    table: dict[str, Any] = {"sample_id": sample_ids}
    if task.kind == "classification":
        probs = softmax(outputs, axis=1)
        names = task.class_names or []
        table["true"] = [names[i] for i in targets]
        table["pred"] = [names[i] for i in probs.argmax(axis=1)]
        for i, name in enumerate(names):
            table[f"prob_{name}"] = probs[:, i]
        # ECE / Brier are already part of classification_metrics.
        return {}, pd.DataFrame(table)

    preds = task.denormalize(outputs)
    truth = np.where(mask.astype(bool), task.denormalize(targets), np.nan)
    calibration = {}
    for i, name in enumerate(task.target_names):
        table[f"pred_{name}"] = preds[:, i]
        table[f"true_{name}"] = truth[:, i]
        observed = np.isfinite(truth[:, i])
        calibration[name] = error_distribution(truth[observed, i], preds[observed, i])
    return {"error_distribution": calibration}, pd.DataFrame(table)
