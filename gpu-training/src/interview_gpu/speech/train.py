"""Speech model training entry point."""

from __future__ import annotations

from typing import Any

from interview_gpu.common.config import Config
from interview_gpu.common.runner import run_training
from interview_gpu.speech.dataset import build_speech_data
from interview_gpu.speech.model import build_speech_model


def train(cfg: Config, device_override: str | None = None) -> dict[str, Any]:
    """Train the speech behaviour model described by ``cfg``."""
    return run_training(cfg, build_speech_data, build_speech_model, device_override)
