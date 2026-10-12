"""Inspect a raw dataset and save a JSON report.

Usage:
    python scripts/inspect_dataset.py --dataset aihub_interview --path /path/to/data
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from interview_gpu.common.cli import run_cli
from interview_gpu.common.config import resolve_path
from interview_gpu.common.io import to_jsonable, write_json
from interview_gpu.datasets import ADAPTERS, get_adapter
from interview_gpu.datasets.inspection import inspect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--path", default=None, help="dataset root (default: data/raw/<dataset dir>)")
    parser.add_argument("--limit", type=int, default=None, help="inspect only the first N samples")
    parser.add_argument("--no-media-check", action="store_true", help="skip opening audio/video files")
    parser.add_argument("--output", default=None, help="report path (default: outputs/inspection/<dataset>.json)")
    args = parser.parse_args()

    report = inspect_dataset(get_adapter(args.dataset, args.path), args.limit, not args.no_media_check)
    output = resolve_path(args.output or f"outputs/inspection/{args.dataset}.json")
    write_json(output, report)
    print(json.dumps(to_jsonable(report), ensure_ascii=False, indent=2))
    print(f"\nReport saved to {output}")


if __name__ == "__main__":
    run_cli(main)
