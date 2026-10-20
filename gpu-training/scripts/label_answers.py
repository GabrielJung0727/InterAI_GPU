"""Stage A: score interview answers with a local LLM judge (GPU, no external API).

Usage:
    1) 200-answer dry run, written to <output dir>/dry_run/ for human review:
    python scripts/label_answers.py --dry-run

    2) 2,000-answer validation labeling after the rubric is settled:
    python scripts/label_answers.py --limit 2000 --consistency-runs 2

    3) Everything (resumes from the validation labels; safe to interrupt and re-run):
    python scripts/label_answers.py --full --consistency-runs 2

    Merge the labels into the training set:
    python scripts/prepare_answer.py --labels data/labeled/qwen3_answers.jsonl

Each run also writes <output>.review.csv (question, answer, scores, evidence),
<output>.stats.json and <output>.failed_samples.jsonl.
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from interview_gpu.common.cli import run_cli
from interview_gpu.common.config import resolve_path
from interview_gpu.common.errors import ConfigError
from interview_gpu.common.logging import setup_logging


def _batch_size(value: str) -> int | str:
    return value if value == "auto" else int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/labeling.yaml")
    parser.add_argument("--model", default=None, help="e.g. Qwen/Qwen3-8B (default) or Qwen/Qwen3-14B")
    parser.add_argument("--backend", default=None, choices=["transformers", "vllm"], help="vllm needs Linux or WSL2")
    parser.add_argument("--quantization", default=None, choices=["auto", "none", "8bit", "4bit"], help="transformers only")
    parser.add_argument("--vllm-model", default=None, help="model served by vLLM, e.g. a pre-quantized AWQ build")
    parser.add_argument("--input", default=None, help="answers.jsonl from prepare_answer.py")
    parser.add_argument("--output", default=None, help="label JSONL (also the resume checkpoint)")
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:N | cpu")
    parser.add_argument("--batch-size", default=None, type=_batch_size, help="integer or 'auto'")
    parser.add_argument("--consistency-runs", default=None, type=int, help="score each answer N times")
    parser.add_argument("--temperature", default=None, type=float)
    parser.add_argument("--limit", default=None, type=int, help="total number of answers to label")
    parser.add_argument("--dry-run", action="store_true", help="label a small sample into <output dir>/dry_run/")
    parser.add_argument("--full", action="store_true", help="label every answer (required to run without --limit)")
    args = parser.parse_args()

    setup_logging()
    from interview_gpu.answer.labeling import create_backend, load_labeling_config, run_labeling

    config = load_labeling_config(
        resolve_path(args.config),
        {
            "backend": args.backend, "model": args.model, "quantization": args.quantization, "input": args.input,
            "output": args.output, "vllm_model": args.vllm_model,
            "batch_size": args.batch_size, "consistency_runs": args.consistency_runs, "temperature": args.temperature,
        },
    )
    if args.full and (args.limit is not None or args.dry_run):
        raise ConfigError("--full cannot be combined with --limit or --dry-run.")
    if not (args.full or args.dry_run or args.limit is not None):
        raise ConfigError(
            "Refusing to label the whole dataset by accident. Pass --dry-run, --limit N, or --full for everything."
        )
    output = resolve_path(config.output)
    limit = args.limit
    if args.dry_run:
        output = output.parent / "dry_run" / output.name
        limit = limit or config.dry_run_limit

    backend = create_backend(config, args.device)
    stats = run_labeling(config, backend, input_path=resolve_path(config.input), output_path=output, limit=limit)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nLabels: {output}\nReview table: {output.with_suffix('.review.csv')}")


if __name__ == "__main__":
    run_cli(main)
