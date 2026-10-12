"""Video preprocessing: raw videos -> landmark sequences (CPU, MediaPipe)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from interview_gpu.common.errors import CorruptedMediaError, EmptyDatasetError
from interview_gpu.common.logging import get_logger
from interview_gpu.datasets.base import BaseDatasetAdapter
from interview_gpu.datasets.processed import coerce_label_columns, label_columns, safe_name, write_manifest
from interview_gpu.schemas.video import VIDEO_FEATURE_COLUMNS
from interview_gpu.video.landmarks import LandmarkExtractor
from interview_gpu.video.temporal_features import build_feature_table, save_feature_table, summarize_table


def prepare_video_dataset(
    adapter: BaseDatasetAdapter,
    output_dir: str | Path,
    *,
    target_fps: float = 5.0,
    max_frames: int | None = None,
    face_model: str | Path | None = None,
    pose_model: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Extract a landmark feature sequence for every sample that has video.

    Corrupted videos are skipped and counted. Only landmark-derived numbers
    are written to disk — no frames or face crops are stored.

    Returns:
        The manifest written to ``<output_dir>/manifest.json``.
    """
    logger = get_logger()
    output_dir = Path(output_dir)
    adapter.ensure_exists()  # report missing data before missing model files
    extractor = LandmarkExtractor(face_model, pose_model)
    (output_dir / "sequences").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for sample in adapter.iter_samples(limit):
        if not sample.video_path:
            adapter.skipped["no_video"] += 1
            continue
        try:
            table = build_feature_table(extractor.process_video(sample.video_path, target_fps, max_frames))
        except CorruptedMediaError as exc:
            adapter.skipped["corrupted_video"] += 1
            logger.warning("Skipping %s: %s", sample.sample_id, exc)
            continue
        relative = f"sequences/{safe_name(sample.sample_id)}.npy"
        save_feature_table(table, output_dir / relative)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "group_id": sample.group_id,
                "video_path": sample.video_path,
                "sequence_path": relative,
                "num_frames": len(table),
                **{f"metric_{key}": value for key, value in summarize_table(table).items()},
                **label_columns(sample.labels),
            }
        )
        if len(rows) % 50 == 0:
            logger.info("Processed %d videos", len(rows))

    if not rows:
        raise EmptyDatasetError(
            f"No video could be processed from {adapter.root.as_posix()} (skipped: {dict(adapter.skipped)})."
        )
    frame = coerce_label_columns(pd.DataFrame(rows))
    frame.to_parquet(output_dir / "index.parquet", index=False)
    manifest = write_manifest(
        output_dir,
        adapter,
        num_samples=len(frame),
        target_fps=target_fps,
        feature_columns=list(VIDEO_FEATURE_COLUMNS),
        label_columns=sorted(c for c in frame.columns if c.startswith("label_")),
    )
    logger.info("Wrote %d sequences to %s", len(frame), output_dir)
    return manifest
