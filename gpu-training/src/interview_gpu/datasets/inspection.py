"""Dataset inspection report (CPU only)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from interview_gpu.datasets.base import BaseDatasetAdapter

MAX_LISTED_FILES = 50
MAX_LABEL_VALUES = 30


def audio_duration(path: str | Path) -> float | None:
    """Duration in seconds from the file header; ``None`` if unreadable."""
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception:  # soundfile raises several backend-specific errors
        return None
    return float(info.duration) if info.frames > 0 and info.samplerate > 0 else None


def video_duration(path: str | Path) -> float | None:
    """Duration in seconds from container metadata; ``None`` if unreadable."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        return float(frames / fps) if fps > 0 and frames > 0 else None
    finally:
        capture.release()


def _directory_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def inspect_dataset(
    adapter: BaseDatasetAdapter, limit: int | None = None, check_media: bool = True
) -> dict[str, Any]:
    """Summarise a dataset: counts, labels, speakers, durations, corruption.

    Args:
        adapter: Dataset adapter (raises if the data is missing).
        limit: Inspect only the first ``limit`` samples.
        check_media: Open every audio/video header to measure duration and
            detect corrupted files (slower on large datasets).
    """
    counts: Counter[str] = Counter()
    speakers: set[str] = set()
    categorical: dict[str, Counter[str]] = defaultdict(Counter)
    numeric: dict[str, list[float]] = defaultdict(list)
    audio_durations: list[float] = []
    video_durations: list[float] = []
    missing: list[str] = []
    corrupted: list[str] = []

    for sample in adapter.iter_samples(limit):
        counts["samples"] += 1
        if sample.answer or sample.transcript:
            counts["text"] += 1
        if sample.speaker_id:
            speakers.add(sample.speaker_id)
        for key, value in sample.labels.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric[key].append(float(value))
            else:
                categorical[key][str(value)] += 1
        for kind, path, probe, durations in (
            ("audio", sample.audio_path, audio_duration, audio_durations),
            ("video", sample.video_path, video_duration, video_durations),
        ):
            if not path:
                continue
            counts[kind] += 1
            if not Path(path).is_file():
                missing.append(path)
            elif check_media:
                duration = probe(path)
                if duration is None:
                    corrupted.append(path)
                else:
                    durations.append(duration)

    label_distribution: dict[str, Any] = {
        key: dict(counter.most_common(MAX_LABEL_VALUES)) for key, counter in categorical.items()
    }
    for key, values in numeric.items():
        label_distribution[key] = {
            "count": len(values),
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }
    disk_bytes = _directory_size(adapter.root)
    return {
        "dataset": adapter.name,
        "root": adapter.root.as_posix(),
        "license": adapter.license.model_dump(),
        "inspected_limit": limit,
        "number_of_samples": counts["samples"],
        "audio_files": counts["audio"],
        "video_files": counts["video"],
        "text_samples": counts["text"],
        "missing_files": len(missing),
        "missing_file_examples": missing[:MAX_LISTED_FILES],
        "label_distribution": label_distribution,
        "speaker_count": len(speakers),
        "average_audio_duration_sec": float(np.mean(audio_durations)) if audio_durations else None,
        "average_video_duration_sec": float(np.mean(video_durations)) if video_durations else None,
        "corrupted_files": len(corrupted),
        "corrupted_file_examples": corrupted[:MAX_LISTED_FILES],
        "skipped_records": dict(adapter.skipped),
        "disk_size_bytes": disk_bytes,
        "disk_size_gb": round(disk_bytes / 1024**3, 3),
    }
