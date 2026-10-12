"""Multimodal fusion evaluation entry point (skeleton)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from interview_gpu.common.runner import run_evaluation
from interview_gpu.multimodal.dataset import build_multimodal_data
from interview_gpu.multimodal.model import build_multimodal_model


def evaluate(
    checkpoint_dir: str | Path, split: str = "test", device_override: str | None = None
) -> dict[str, Any]:
    """Evaluate a registered fusion checkpoint on one split."""
    return run_evaluation(checkpoint_dir, build_multimodal_data, build_multimodal_model, split, device_override)
