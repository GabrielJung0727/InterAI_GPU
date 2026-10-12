"""Print the GPU / PyTorch environment and the recommended precision.

Usage:
    python scripts/check_gpu.py [--json]
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from interview_gpu.common.cli import run_cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args()

    from interview_gpu.common.device import (
        environment_report,
        format_environment_report,
        resolve_precision,
        select_device,
    )

    device = select_device("auto")
    report = environment_report(device, resolve_precision("auto", device))
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(format_environment_report(report))
    if not report["cuda_available"]:
        print(f"\nNo CUDA device detected: training will run on {report['selected_device']}.")


if __name__ == "__main__":
    run_cli(main)
