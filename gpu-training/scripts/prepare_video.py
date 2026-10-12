"""Extract MediaPipe landmark sequences from videos (CPU preprocessing).

Usage:
    python scripts/prepare_video.py --input /path/to/video --output data/processed/video
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from interview_gpu.common.cli import run_cli
from interview_gpu.common.config import resolve_path
from interview_gpu.common.logging import setup_logging
from interview_gpu.datasets import ADAPTERS, get_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="raw dataset root (default: data/raw/<dataset dir>)")
    parser.add_argument("--output", default="data/processed/video")
    parser.add_argument("--dataset", default="aihub_multimodal", choices=sorted(ADAPTERS))
    parser.add_argument("--fps", type=float, default=5.0, help="frame sampling rate")
    parser.add_argument("--max-frames", type=int, default=None, help="maximum sampled frames per video")
    parser.add_argument("--face-model", default=None, help="path to face_landmarker.task")
    parser.add_argument("--pose-model", default=None, help="path to pose_landmarker.task")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    from interview_gpu.video.preprocess import prepare_video_dataset

    prepare_video_dataset(
        get_adapter(args.dataset, args.input),
        resolve_path(args.output),
        target_fps=args.fps,
        max_frames=args.max_frames,
        face_model=args.face_model,
        pose_model=args.pose_model,
        limit=args.limit,
    )


if __name__ == "__main__":
    run_cli(main)
