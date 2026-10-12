"""Answer-evaluator dataset built from ``data/processed/answer/answers.jsonl``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

from interview_gpu.answer.prompts import format_answer_input
from interview_gpu.common.config import Config, resolve_path
from interview_gpu.common.runner import DataBundle, load_manifest
from interview_gpu.common.tasks import Supervision, prepare_supervision
from interview_gpu.datasets.processed import require_processed
from interview_gpu.schemas.answer import ANSWER_DIMENSIONS, AnswerTrainingExample

ANSWERS_FILE = "answers.jsonl"
PREPARE_COMMAND = "python scripts/prepare_answer.py --input <raw dataset dir> --output data/processed/answer"
DEFAULT_ENCODER = "intfloat/multilingual-e5-small"


def load_answer_frame(path: Path) -> pd.DataFrame:
    """Read ``answers.jsonl`` into a flat table with ``label_<dimension>`` columns."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            example = AnswerTrainingExample.model_validate(json.loads(line))
            rows.append(
                {
                    "sample_id": example.sample_id,
                    "group_id": example.group_id,
                    "question": example.question,
                    "answer": example.answer,
                    "context": example.context.model_dump(),
                    **{f"label_{name}": example.scores.get(name) for name in ANSWER_DIMENSIONS},
                }
            )
    columns = ["sample_id", "group_id", "question", "answer", "context"] + [f"label_{n}" for n in ANSWER_DIMENSIONS]
    return pd.DataFrame(rows, columns=columns)


class AnswerDataset(Dataset):
    """Question/answer text pairs with multi-head rubric targets."""

    def __init__(self, supervision: Supervision, rows: Any, input_style: str) -> None:
        self.supervision = supervision
        self.rows = rows
        self.input_style = input_style
        self.records = supervision.frame[["sample_id", "question", "answer", "context"]].to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = int(self.rows[index])
        record = self.records[row]
        first, second = format_answer_input(record["question"], record["answer"], record["context"], self.input_style)
        target, mask = self.supervision.target_tensors(row)
        return {"text_a": first, "text_b": second, "targets": target, "mask": mask, "sample_id": str(record["sample_id"])}


class AnswerCollator:
    """Tokenises text pairs into padded model inputs.

    Args:
        tokenizer: HuggingFace-style tokenizer callable.
        max_length: Truncation length in tokens.
    """

    def __init__(self, tokenizer: Callable[..., Any], max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode(self, first: list[str], second: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            first, second, padding=True, truncation="longest_first", max_length=self.max_length, return_tensors="pt"
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "inputs": self.encode([item["text_a"] for item in batch], [item["text_b"] for item in batch]),
            "targets": torch.stack([item["targets"] for item in batch]),
            "mask": torch.stack([item["mask"] for item in batch]),
            "sample_id": [item["sample_id"] for item in batch],
        }


def build_answer_data(cfg: Config, tokenizer: Callable[..., Any] | None = None) -> DataBundle:
    """Load processed answers and build train/val/test datasets.

    Args:
        tokenizer: Injected tokenizer (tests); defaults to the tokenizer of
            ``model.params.encoder_name``.
    """
    processed_dir = resolve_path(cfg.data.processed_dir)
    frame = load_answer_frame(require_processed(processed_dir / ANSWERS_FILE, PREPARE_COMMAND))
    supervision = prepare_supervision(frame, cfg.data, cfg.experiment.seed)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.params.get("encoder_name", DEFAULT_ENCODER))
    input_style = cfg.data.params.get("input_style", "e5")
    max_length = int(cfg.data.params.get("max_length", 512))
    make = lambda split: AnswerDataset(supervision, supervision.splits[split], input_style)  # noqa: E731
    return DataBundle(
        train=make("train"), val=make("val"), test=make("test"),
        task=supervision.task,
        collate_fn=AnswerCollator(tokenizer, max_length),
        normalization={"input_style": input_style, "max_length": max_length},
        dataset_metadata=load_manifest(processed_dir),
    )
