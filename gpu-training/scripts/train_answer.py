"""Train the multi-head interview answer evaluator.

Usage:
    python scripts/train_answer.py --config configs/answer.yaml [--device cuda] [--set training.batch_size=8]
"""

import _bootstrap  # noqa: F401

from interview_gpu.common.cli import train_cli

if __name__ == "__main__":
    train_cli(__doc__, "configs/answer.yaml", "interview_gpu.answer.train")
