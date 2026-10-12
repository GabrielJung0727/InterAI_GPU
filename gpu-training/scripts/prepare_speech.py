"""Extract speech features (CPU) and optional encoder embeddings (GPU).

Usage:
    python scripts/prepare_speech.py --input /path/to/raw --output data/processed/speech
    python scripts/prepare_speech.py --input ... --output ... --encoder microsoft/wavlm-base-plus
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
    parser.add_argument("--output", default="data/processed/speech")
    parser.add_argument("--dataset", default="aihub_interview", choices=sorted(ADAPTERS))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--encoder", default=None, help="HuggingFace speech encoder for Stage-2 embeddings")
    parser.add_argument("--device", default="auto", help="device for the encoder")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    from interview_gpu.speech.preprocess import prepare_speech_dataset

    prepare_speech_dataset(
        get_adapter(args.dataset, args.input),
        resolve_path(args.output),
        sample_rate=args.sample_rate,
        encoder_name=args.encoder,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    run_cli(main)
