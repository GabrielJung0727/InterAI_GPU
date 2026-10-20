"""Checkpoint persistence and the on-disk model registry.

Registry layout::

    checkpoints/<task>/<task>_v001/
        model.pt        weights + optimizer/scheduler/scaler state + all metadata
        config.yaml     resolved training config
        metrics.json    validation / test metrics
        metadata.json   human-readable copy of the metadata stored in model.pt

``model.pt`` only contains tensors and plain python containers, so it is loaded
with ``weights_only=True`` (no arbitrary code execution).
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from interview_gpu.common.config import PROJECT_ROOT, save_config
from interview_gpu.common.errors import CheckpointMismatchError
from interview_gpu.common.io import to_jsonable, write_json

MODEL_FILE = "model.pt"


def git_commit_hash() -> str | None:
    """Current git commit, or ``None`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _version_dirs(task_dir: Path, task: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(task)}_v(\d+)$")
    found = []
    if task_dir.is_dir():
        for child in task_dir.iterdir():
            match = pattern.match(child.name)
            if child.is_dir() and match:
                found.append((int(match.group(1)), child))
    return sorted(found)


def next_version_dir(root: str | Path, task: str) -> Path:
    """Create and return the next ``<root>/<task>/<task>_vNNN`` directory."""
    task_dir = Path(root) / task
    versions = _version_dirs(task_dir, task)
    number = versions[-1][0] + 1 if versions else 1
    path = task_dir / f"{task}_v{number:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def latest_version_dir(root: str | Path, task: str) -> Path | None:
    """Most recent version directory that actually contains a model."""
    for _, path in reversed(_version_dirs(Path(root) / task, task)):
        if (path / MODEL_FILE).is_file():
            return path
    return None


def save_checkpoint(
    directory: str | Path,
    *,
    model: torch.nn.Module,
    epoch: int,
    metric_name: str,
    metric_value: float | None,
    config: dict[str, Any],
    task: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    label_mapping: dict[str, Any] | None = None,
    normalization: dict[str, Any] | None = None,
    dataset_metadata: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    filename: str = MODEL_FILE,
) -> Path:
    """Persist a training checkpoint with everything needed to reuse it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = to_jsonable(
        {
            "epoch": epoch,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "task": task,
            "label_mapping": label_mapping or {},
            "normalization": normalization or {},
            "dataset_metadata": dataset_metadata or {},
            "training_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit_hash(),
            "pytorch_version": str(torch.__version__),  # TorchVersion is rejected by weights_only loading
        }
    )
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "config": to_jsonable(config),
        **metadata,
    }
    path = directory / filename
    torch.save(payload, path)
    save_config(config, directory / "config.yaml")
    write_json(directory / "metadata.json", metadata)
    if metrics is not None:
        write_json(directory / "metrics.json", metrics)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint file (or a registry directory containing ``model.pt``)."""
    path = Path(path)
    if path.is_dir():
        path = path / MODEL_FILE
    if not path.is_file():
        raise CheckpointMismatchError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # corrupted / foreign file
        raise CheckpointMismatchError(f"Could not read checkpoint {path}: {exc}") from exc


def load_model_state(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load weights strictly, turning shape/key mismatches into a clear error."""
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise CheckpointMismatchError(
            "Checkpoint does not match the model architecture. The config used to build "
            "the model must be the one stored next to the checkpoint (config.yaml).\n"
            f"Details: {str(exc)[:600]}"
        ) from exc
