"""Answer evaluator training entry point."""

from __future__ import annotations

from typing import Any

from interview_gpu.answer.dataset import build_answer_data
from interview_gpu.answer.model import build_answer_model
from interview_gpu.common.config import Config
from interview_gpu.common.runner import run_training


def train(cfg: Config, device_override: str | None = None) -> dict[str, Any]:
    """Train the multi-head answer evaluator described by ``cfg``."""
    return run_training(cfg, build_answer_data, build_answer_model, device_override)
