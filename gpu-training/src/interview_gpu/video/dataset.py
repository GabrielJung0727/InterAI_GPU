"""Video training dataset built from ``data/processed/video``.

Layout written by ``prepare_video.py``::

    index.parquet            sample_id, group_id, sequence_path, num_frames, metric_*, label_*
    sequences/<id>.npy       float32 [T, len(VIDEO_FEATURE_COLUMNS)]
    manifest.json
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from interview_gpu.common.config import Config, resolve_path
from interview_gpu.common.errors import EmptyDatasetError
from interview_gpu.common.normalization import apply_normalizer, fit_normalizer
from interview_gpu.common.runner import DataBundle, load_manifest
from interview_gpu.common.tasks import Supervision, prepare_supervision
from interview_gpu.datasets.processed import require_processed
from interview_gpu.schemas.video import VIDEO_MODEL_FEATURES

INDEX_FILE = "index.parquet"
PREPARE_COMMAND = "python scripts/prepare_video.py --input <raw dataset dir> --output data/processed/video"


class VideoSequenceDataset(Dataset):
    """Normalised landmark sequences, cropped to ``max_frames``."""

    def __init__(
        self,
        sequences: list[np.ndarray],
        supervision: Supervision,
        rows: np.ndarray,
        max_frames: int,
        training: bool,
    ) -> None:
        self.sequences = sequences
        self.supervision = supervision
        self.rows = rows
        self.max_frames = max_frames
        self.training = training
        self.sample_ids = supervision.frame["sample_id"].astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = int(self.rows[index])
        sequence = self.sequences[row]
        if len(sequence) > self.max_frames:
            # Random window while training, centre window for evaluation.
            spare = len(sequence) - self.max_frames
            start = np.random.randint(0, spare + 1) if self.training else spare // 2
            sequence = sequence[start : start + self.max_frames]
        target, mask = self.supervision.target_tensors(row)
        return {
            "inputs": {"sequence": torch.from_numpy(sequence)},
            "targets": target,
            "mask": mask,
            "sample_id": str(self.sample_ids[row]),
        }


def collate_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Zero-pad sequences to the longest in the batch and record lengths."""
    lengths = torch.tensor([len(item["inputs"]["sequence"]) for item in batch], dtype=torch.long)
    features = batch[0]["inputs"]["sequence"].shape[1]
    sequence = torch.zeros(len(batch), int(lengths.max()), features)
    for i, item in enumerate(batch):
        sequence[i, : lengths[i]] = item["inputs"]["sequence"]
    return {
        "inputs": {"sequence": sequence, "lengths": lengths},
        "targets": torch.stack([item["targets"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
    }


def build_video_data(cfg: Config) -> DataBundle:
    """Load processed landmark sequences and build train/val/test datasets."""
    processed_dir = resolve_path(cfg.data.processed_dir)
    frame = pd.read_parquet(require_processed(processed_dir / INDEX_FILE, PREPARE_COMMAND))
    supervision = prepare_supervision(frame, cfg.data, cfg.experiment.seed)
    splits = supervision.splits

    # Sequences are small (T x 16 floats), so they are kept in memory.
    # Column 0 is the timestamp, which is not a model input.
    raw = [np.load(processed_dir / path)[:, 1:] for path in supervision.frame["sequence_path"]]
    if any(len(sequence) == 0 for sequence in raw):
        raise EmptyDatasetError("A processed video sequence has zero frames; re-run prepare_video.py.")
    normalizer = fit_normalizer(np.concatenate([raw[i] for i in splits["train"]]))
    sequences = [apply_normalizer(sequence, normalizer) for sequence in raw]

    max_frames = int(cfg.data.params.get("max_frames", 600))
    make = lambda split: VideoSequenceDataset(  # noqa: E731
        sequences, supervision, splits[split], max_frames, training=split == "train"
    )
    return DataBundle(
        train=make("train"), val=make("val"), test=make("test"),
        task=supervision.task,
        collate_fn=collate_sequences,
        normalization={**normalizer, "feature_names": list(VIDEO_MODEL_FEATURES), "max_frames": max_frames},
        dataset_metadata=load_manifest(processed_dir),
        label_mapping={"class_names": supervision.task.class_names},
        class_counts=supervision.class_counts,
        train_labels=supervision.targets[splits["train"]] if supervision.task.kind == "classification" else None,
        info={"input_dim": len(VIDEO_MODEL_FEATURES)},
    )
