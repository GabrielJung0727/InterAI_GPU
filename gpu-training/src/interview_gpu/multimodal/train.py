"""Multimodal fusion training entry point (skeleton — needs aligned data)."""

from __future__ import annotations

from typing import Any

from interview_gpu.common.config import Config
from interview_gpu.common.runner import run_training
from interview_gpu.multimodal.dataset import build_multimodal_data
from interview_gpu.multimodal.model import build_multimodal_model


def train(cfg: Config, device_override: str | None = None) -> dict[str, Any]:
    """Train the fusion model. Fails with a clear message while no data exists."""
    return run_training(cfg, build_multimodal_data, build_multimodal_model, device_override)
