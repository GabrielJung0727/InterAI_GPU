"""Helpers for the ``data/processed/<task>/`` tables written by preprocessing."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from interview_gpu.common.errors import DatasetNotFoundError
from interview_gpu.common.io import write_json
from interview_gpu.datasets.base import BaseDatasetAdapter

LABEL_PREFIX = "label_"


def safe_name(sample_id: str) -> str:
    """File-system safe version of a sample id."""
    return re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", sample_id)


def label_columns(labels: dict[str, Any]) -> dict[str, Any]:
    """Scalar labels as ``label_<name>`` columns (nested values are dropped)."""
    return {
        f"{LABEL_PREFIX}{key}": value
        for key, value in labels.items()
        if isinstance(value, (str, int, float)) and not isinstance(value, bool)
    }


def coerce_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Make each label column homogeneous: float if fully numeric, else string."""
    for column in [c for c in frame.columns if c.startswith(LABEL_PREFIX)]:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() == frame[column].notna().sum():
            frame[column] = numeric.astype("float64")
        else:
            frame[column] = frame[column].map(lambda v: None if pd.isna(v) else str(v))
    return frame


def write_manifest(
    output_dir: Path, adapter: BaseDatasetAdapter, num_samples: int, **extra: Any
) -> dict[str, Any]:
    """Record provenance and license of a processed dataset next to it."""
    manifest = {
        "dataset_name": adapter.name,
        "usage_scope": adapter.license.usage_scope,
        "license": adapter.license.model_dump(),
        "raw_root": adapter.root.as_posix(),
        "num_samples": num_samples,
        "skipped_records": dict(adapter.skipped),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def require_processed(path: Path, prepare_command: str) -> Path:
    """Return ``path`` or raise a clear error telling how to create it."""
    if not path.exists():
        raise DatasetNotFoundError(
            path.parent,
            f"Processed file '{path.name}' is missing. Place the raw dataset under data/raw/ "
            f"and run:\n  {prepare_command}",
        )
    return path
