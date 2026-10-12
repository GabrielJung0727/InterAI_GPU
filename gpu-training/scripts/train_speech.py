"""Train the speech behaviour model.

Usage:
    python scripts/train_speech.py --config configs/speech.yaml [--device cuda] [--set training.batch_size=8]
"""

import _bootstrap  # noqa: F401

from interview_gpu.common.cli import train_cli

if __name__ == "__main__":
    train_cli(__doc__, "configs/speech.yaml", "interview_gpu.speech.train")
