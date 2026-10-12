"""Build answer-evaluator examples from a dataset (CPU only).

Usage:
    python scripts/prepare_answer.py --input /path/to/raw --output data/processed/answer

    Stage A - export prompts for an external LLM judge, then merge its scores:
    python scripts/prepare_answer.py --input ... --export-prompts outputs/labeling_prompts.jsonl
    python scripts/prepare_answer.py --input ... --labels path/to/labels.jsonl
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from interview_gpu.common.cli import run_cli
from interview_gpu.common.config import resolve_path
from interview_gpu.common.logging import setup_logging
from interview_gpu.datasets import ADAPTERS, get_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="raw dataset root (default: data/raw/<dataset dir>)")
    parser.add_argument("--output", default="data/processed/answer")
    parser.add_argument("--dataset", default="aihub_interview", choices=sorted(ADAPTERS))
    parser.add_argument("--labels", default=None, help="JSONL of {sample_id, scores} to merge")
    parser.add_argument("--export-prompts", default=None, help="write labeling prompts for unlabeled examples")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    from interview_gpu.answer.preprocess import prepare_answer_dataset

    prepare_answer_dataset(
        get_adapter(args.dataset, args.input),
        resolve_path(args.output),
        labels_path=args.labels,
        prompts_path=resolve_path(args.export_prompts) if args.export_prompts else None,
        limit=args.limit,
    )


if __name__ == "__main__":
    run_cli(main)
