"""Train the multimodal fusion model (skeleton; needs aligned multimodal data).

Usage:
    python scripts/train_multimodal.py --config configs/multimodal.yaml [--device cuda]
"""

import _bootstrap  # noqa: F401

from interview_gpu.common.cli import train_cli

if __name__ == "__main__":
    train_cli(__doc__, "configs/multimodal.yaml", "interview_gpu.multimodal.train")
