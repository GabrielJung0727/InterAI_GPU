"""Helpers shared by the command-line scripts."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from interview_gpu.common.errors import InterviewGpuError

TORCH_BLOCKED_HINT = (
    "PyTorch is installed but its native libraries could not be loaded.\n"
    "On Windows, 'WinError 4551' means an application-control policy (e.g. Smart App "
    "Control) blocked PyTorch's unsigned DLLs. See README.md > Troubleshooting."
)


def run_cli(main: Callable[[], Any]) -> None:
    """Run ``main`` and print expected errors without a stack trace."""
    try:
        main()
    except InterviewGpuError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            print(f"\nERROR: {exc}\n\n{TORCH_BLOCKED_HINT}", file=sys.stderr)
            sys.exit(3)
        raise
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


def train_cli(description: str, default_config: str, task_module: str) -> None:
    """Standard ``--config / --device / --set`` training entry point.

    Args:
        description: CLI help text.
        default_config: Config path used when ``--config`` is omitted.
        task_module: Module exposing ``train(cfg, device_override)``,
            imported lazily so ``--help`` works without PyTorch.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config, help="YAML config file")
    parser.add_argument("--device", default=None, help="auto | cuda | cuda:N | mps | cpu (overrides config)")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Override a config value, e.g. --set training.batch_size=8 (repeatable)",
    )
    args = parser.parse_args()

    def main() -> None:
        import importlib

        from interview_gpu.common.config import load_config

        cfg = load_config(args.config, args.overrides)
        summary = importlib.import_module(task_module).train(cfg, device_override=args.device)
        print(json.dumps({k: summary[k] for k in ("checkpoint_dir", "output_dir", "best_epoch",
                                                    "best_metric_name", "best_metric_value")}, indent=2))

    run_cli(main)
