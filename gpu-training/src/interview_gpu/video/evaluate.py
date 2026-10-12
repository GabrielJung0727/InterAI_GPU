"""Video model evaluation entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from interview_gpu.common.runner import run_evaluation
from interview_gpu.video.dataset import build_video_data
from interview_gpu.video.model import build_video_model


def evaluate(
    checkpoint_dir: str | Path, split: str = "test", device_override: str | None = None
) -> dict[str, Any]:
    """Evaluate a registered video checkpoint on one split."""
    return run_evaluation(checkpoint_dir, build_video_data, build_video_model, split, device_override)
