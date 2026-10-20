"""Compare two labelings of the same answers, e.g. rubric V1 vs V2 (CPU only).

Usage:
    python scripts/compare_labels.py \
        --old data/labeled/dry_run_rubric_v1/qwen3_answers.jsonl --old-scale 10 \
        --new data/labeled/dry_run/qwen3_answers.jsonl

Scores are divided by their scale maximum before comparing. Prints per-dimension
means, null rates and rank agreement, and the halo effect (mean correlation
between different dimensions) of each labeling. The full report, including the
Pearson / Spearman matrices, is written next to --new as <name>.compare.json.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from interview_gpu.common.cli import run_cli
from interview_gpu.common.config import resolve_path
from interview_gpu.common.errors import EmptyDatasetError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", required=True, help="baseline label JSONL")
    parser.add_argument("--new", required=True, help="label JSONL to compare against the baseline")
    parser.add_argument("--old-scale", type=float, default=4.0, help="maximum score of --old (10 for rubric V1)")
    parser.add_argument("--new-scale", type=float, default=4.0, help="maximum score of --new")
    args = parser.parse_args()

    from interview_gpu.answer.labeling import compare_label_sets, read_jsonl
    from interview_gpu.common.io import write_json
    from interview_gpu.schemas.answer import ANSWER_DIMENSIONS

    old_path, new_path = resolve_path(args.old), resolve_path(args.new)
    old, new = read_jsonl(old_path), read_jsonl(new_path)
    if not old or not new:
        raise EmptyDatasetError(f"No labels found in {old_path if not old else new_path}.")
    report = compare_label_sets(old, new, args.old_scale, args.new_scale)
    output = new_path.with_suffix(".compare.json")
    write_json(output, report)

    def cell(value: float | None) -> str:
        return "   n/a" if value is None or value != value else f"{value:6.2f}"

    print(f"{report['num_common']} answers in both files. Scores are on a 0-1 scale (score / scale maximum).\n")
    print(f"{'dimension':<20} old_mean new_mean change old_null new_null rank_agree")
    for name in ANSWER_DIMENSIONS:
        row = report["dimensions"][name]
        print(
            f"{name:<20} {cell(row['old_mean'])}   {cell(row['new_mean'])} {cell(row['change'])}"
            f"   {cell(row['old_null_rate'])}   {cell(row['new_null_rate'])}     {cell(row['rank_agreement_spearman'])}"
        )
    print("\nHalo effect (mean correlation between different dimensions; lower = more independent):")
    for side in ("old", "new"):
        halo = report["halo"][side]
        print(f"  {side}: pearson {cell(halo['pearson_mean_offdiag'])}   spearman {cell(halo['spearman_mean_offdiag'])}")
    print(f"\nFull report: {output}")


if __name__ == "__main__":
    run_cli(main)
