"""Speech preprocessing: raw dataset -> ``features.csv`` (+ embeddings).

Handcrafted features run on CPU. Encoder embeddings (optional, Stage 2) are
the only part that benefits from a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from interview_gpu.common.errors import CorruptedMediaError, EmptyDatasetError
from interview_gpu.common.io import write_table
from interview_gpu.common.logging import get_logger
from interview_gpu.datasets.base import BaseDatasetAdapter
from interview_gpu.datasets.processed import coerce_label_columns, label_columns, safe_name, write_manifest
from interview_gpu.speech.features import SpeechFeatureConfig, extract_speech_features, load_audio


def prepare_speech_dataset(
    adapter: BaseDatasetAdapter,
    output_dir: str | Path,
    *,
    sample_rate: int = 16000,
    encoder_name: str | None = None,
    device: str = "auto",
    limit: int | None = None,
) -> dict[str, Any]:
    """Extract speech features for every sample that has audio.

    Corrupted or missing audio files are skipped and counted, not fatal.

    Returns:
        The manifest written to ``<output_dir>/manifest.json``.
    """
    logger = get_logger()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SpeechFeatureConfig(sample_rate=sample_rate)

    encoder = None
    if encoder_name:
        from interview_gpu.common.device import select_device
        from interview_gpu.speech.encoder import SpeechEncoder

        encoder = SpeechEncoder.from_pretrained(encoder_name).to(select_device(device)).eval()
        (output_dir / "embeddings").mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    for sample in adapter.iter_samples(limit):
        if not sample.audio_path:
            adapter.skipped["no_audio"] += 1
            continue
        try:
            wave = load_audio(sample.audio_path, sample_rate)
            features = extract_speech_features(wave, sample.transcript, config)
        except CorruptedMediaError as exc:
            adapter.skipped["corrupted_audio"] += 1
            logger.warning("Skipping %s: %s", sample.sample_id, exc)
            continue
        row: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "group_id": sample.group_id,
            "audio_path": sample.audio_path,
            "has_transcript": bool(sample.transcript),
            **features,
            **label_columns(sample.labels),
        }
        if encoder is not None:
            relative = f"embeddings/{safe_name(sample.sample_id)}.npy"
            np.save(output_dir / relative, encoder.embed(wave, sample_rate=sample_rate))
            row["embedding_path"] = relative
        rows.append(row)
        if len(rows) % 500 == 0:
            logger.info("Processed %d audio files", len(rows))

    if not rows:
        raise EmptyDatasetError(
            f"No audio could be processed from {adapter.root.as_posix()} "
            f"(skipped: {dict(adapter.skipped)})."
        )
    frame = coerce_label_columns(pd.DataFrame(rows))
    write_table(output_dir / "features.csv", frame)
    manifest = write_manifest(
        output_dir,
        adapter,
        num_samples=len(frame),
        sample_rate=sample_rate,
        encoder_name=encoder_name,
        label_columns=sorted(c for c in frame.columns if c.startswith("label_")),
    )
    logger.info("Wrote %d rows to %s", len(frame), output_dir / "features.csv")
    return manifest
