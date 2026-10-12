"""Evaluate the latest registered checkpoint of every task.

Usage:
    python scripts/evaluate_all.py [--split test] [--device cuda] [--checkpoint-dir checkpoints]
"""

from __future__ import annotations

import argparse
import importlib
import json

import _bootstrap  # noqa: F401

from interview_gpu.common.cli import run_cli
from interview_gpu.common.config import resolve_path
from interview_gpu.common.errors import InterviewGpuError

TASKS = ("speech", "video", "answer", "multimodal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    from interview_gpu.common.checkpoint import latest_version_dir

    summary = {}
    for task in TASKS:
        checkpoint = latest_version_dir(resolve_path(args.checkpoint_dir), task)
        if checkpoint is None:
            summary[task] = {"status": "skipped", "reason": "no trained checkpoint"}
            continue
        try:
            evaluate = importlib.import_module(f"interview_gpu.{task}.evaluate").evaluate
            report = evaluate(checkpoint, args.split, args.device)
            summary[task] = {"status": "ok", "checkpoint": checkpoint.name, "metrics": report["metrics"]}
        except InterviewGpuError as exc:
            summary[task] = {"status": "failed", "checkpoint": checkpoint.name, "reason": str(exc)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_cli(main)
