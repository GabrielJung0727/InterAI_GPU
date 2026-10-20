"""Small JSON / table helpers that keep outputs portable.

Tables are stored as UTF-8 CSV rather than parquet: pandas reads and writes
CSV without any extra native library, so it also works where pyarrow cannot
be loaded (e.g. Windows application-control policies blocking its DLLs).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Columns holding identifiers or class names; never parsed as numbers.
TEXT_COLUMNS = ("sample_id", "group_id", "true", "pred")


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy types and replace NaN/inf with ``None``."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: str | Path, payload: Any) -> None:
    """Write ``payload`` as UTF-8 JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_table(path: str | Path, frame: pd.DataFrame) -> None:
    """Write ``frame`` as UTF-8 CSV, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def read_table(path: str | Path) -> pd.DataFrame:
    """Inverse of :func:`write_table`.

    Only an empty cell is a missing value (a class named ``NA`` stays a
    string), ids and ``*_path`` columns stay strings even when they look
    numeric, and floats round-trip exactly.
    """
    columns = pd.read_csv(path, nrows=0, encoding="utf-8").columns
    text = {column: str for column in columns if column in TEXT_COLUMNS or column.endswith("_path")}
    return pd.read_csv(
        path, dtype=text, keep_default_na=False, na_values=[""], float_precision="round_trip", encoding="utf-8"
    )
