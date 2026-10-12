"""Small JSON helpers that keep outputs strictly JSON-serializable."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


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
