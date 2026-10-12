"""Answer evaluator evaluation entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from interview_gpu.answer.dataset import build_answer_data
from interview_gpu.answer.model import build_answer_model
from interview_gpu.common.runner import run_evaluation


def evaluate(
    checkpoint_dir: str | Path, split: str = "test", device_override: str | None = None
) -> dict[str, Any]:
    """Evaluate a registered answer-evaluator checkpoint on one split."""
    return run_evaluation(checkpoint_dir, build_answer_data, build_answer_model, split, device_override)
