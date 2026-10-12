"""Train the temporal video behaviour model.

Usage:
    python scripts/train_video.py --config configs/video.yaml [--device cuda] [--set model.params.backbone=gru]
"""

import _bootstrap  # noqa: F401

from interview_gpu.common.cli import train_cli

if __name__ == "__main__":
    train_cli(__doc__, "configs/video.yaml", "interview_gpu.video.train")
