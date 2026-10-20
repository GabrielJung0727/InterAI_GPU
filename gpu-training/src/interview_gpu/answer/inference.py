"""Service-facing answer evaluation interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from interview_gpu.answer.dataset import DEFAULT_ENCODER, AnswerCollator
from interview_gpu.answer.model import make_answer_model
from interview_gpu.answer.prompts import format_answer_input
from interview_gpu.common.checkpoint import load_checkpoint, load_model_state
from interview_gpu.common.config import config_from_dict, resolve_path
from interview_gpu.common.device import resolve_precision, select_device
from interview_gpu.common.tasks import TaskSpec
from interview_gpu.schemas.answer import SCORE_MAX, SCORE_MIN


class AnswerEvaluator:
    """Score interview answers on each rubric dimension.

    Returns numeric scores only; natural-language feedback is produced by the
    service / LLM layer from these scores.

    Example::

        evaluator = AnswerEvaluator("checkpoints/answer/answer_v001")
        evaluator.predict(question="...", answer="...", job="Backend Engineer")
        # {"scores": {"relevance": 3.2, "structure": 2.6, ...}, "model_version": "answer_v001"}
    """

    def __init__(self, checkpoint_dir: str | Path, device: str = "auto", tokenizer: Any | None = None) -> None:
        checkpoint_dir = resolve_path(checkpoint_dir)
        self.device = select_device(device)
        self.amp = resolve_precision("auto", self.device)
        payload = load_checkpoint(checkpoint_dir, self.device)
        cfg = config_from_dict(payload["config"])
        self.task = TaskSpec.from_dict(payload["task"])
        self.input_style = payload["normalization"].get("input_style", "e5")
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(cfg.model.params.get("encoder_name", DEFAULT_ENCODER))
        self.collator = AnswerCollator(tokenizer, int(payload["normalization"].get("max_length", 512)))
        # Architecture only; all weights (encoder + heads) come from the checkpoint.
        self.model = make_answer_model(cfg.model.params, self.task.target_names, pretrained=False)
        load_model_state(self.model, payload["model_state"])
        self.model.to(self.device).eval()
        self.model_version = checkpoint_dir.name

    @torch.no_grad()
    def predict_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score several answers at once.

        Args:
            items: Dicts with ``question``, ``answer`` and optional context
                keys (``job``, ``interview_type``, ``company``,
                ``resume_context``, ``company_context``).
        """
        pairs = [
            format_answer_input(item["question"], item["answer"], item, self.input_style) for item in items
        ]
        inputs = self.collator.encode([a for a, _ in pairs], [b for _, b in pairs])
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.autocast(device_type=self.device.type, dtype=self.amp.dtype, enabled=self.amp.enabled):
            outputs = self.model(**inputs)
        scores = np.clip(self.task.denormalize(outputs.float().cpu().numpy()), SCORE_MIN, SCORE_MAX)
        return [
            {
                "scores": {name: round(float(value), 2) for name, value in zip(self.task.target_names, row)},
                "model_version": self.model_version,
            }
            for row in scores
        ]

    def predict(
        self,
        question: str,
        answer: str,
        job: str | None = None,
        interview_type: str | None = None,
        company: str | None = None,
        resume_context: str | None = None,
        company_context: str | None = None,
    ) -> dict[str, Any]:
        """Score one answer. Returns a JSON-serializable dict."""
        return self.predict_batch(
            [
                {
                    "question": question,
                    "answer": answer,
                    "job": job,
                    "interview_type": interview_type,
                    "company": company,
                    "resume_context": resume_context,
                    "company_context": company_context,
                }
            ]
        )[0]
