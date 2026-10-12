"""Speech model evaluation entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from interview_gpu.common.runner import run_evaluation
from interview_gpu.speech.dataset import build_speech_data
from interview_gpu.speech.model import build_speech_model


def evaluate(
    checkpoint_dir: str | Path, split: str = "test", device_override: str | None = None
) -> dict[str, Any]:
    """Evaluate a registered speech checkpoint on one split."""
    return run_evaluation(checkpoint_dir, build_speech_data, build_speech_model, split, device_override)
