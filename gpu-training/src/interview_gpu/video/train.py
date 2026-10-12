"""Video model training entry point."""

from __future__ import annotations

from typing import Any

from interview_gpu.common.config import Config
from interview_gpu.common.runner import run_training
from interview_gpu.video.dataset import build_video_data
from interview_gpu.video.model import build_video_model


def train(cfg: Config, device_override: str | None = None) -> dict[str, Any]:
    """Train the temporal video behaviour model described by ``cfg``."""
    return run_training(cfg, build_video_data, build_video_model, device_override)
