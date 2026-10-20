"""Speech training datasets built from ``data/processed/speech``.

``data.params.input`` selects the model input:

* ``handcrafted``  Stage-1 feature vector
* ``embedding``    frozen-encoder embedding (requires ``prepare_speech.py --encoder``)
* ``both``         concatenation of the two
* ``waveform``     raw audio for encoder fine-tuning (Stage 3)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from interview_gpu.common.config import Config, resolve_path
from interview_gpu.common.errors import ConfigError, InvalidLabelError
from interview_gpu.common.io import read_table
from interview_gpu.common.normalization import apply_normalizer, fit_normalizer
from interview_gpu.common.runner import DataBundle, load_manifest
from interview_gpu.common.tasks import Supervision, prepare_supervision
from interview_gpu.datasets.processed import require_processed
from interview_gpu.schemas.speech import SPEECH_FEATURE_NAMES
from interview_gpu.speech.features import load_audio

FEATURES_FILE = "features.csv"
PREPARE_COMMAND = "python scripts/prepare_speech.py --input <raw dataset dir> --output data/processed/speech"
INPUT_MODES = ("handcrafted", "embedding", "both", "waveform")


def build_feature_matrix(frame: pd.DataFrame, processed_dir: Path, mode: str) -> np.ndarray:
    """Assemble the un-normalised ``[N, D]`` input matrix for ``mode``."""
    parts = []
    if mode in ("handcrafted", "both"):
        parts.append(frame[list(SPEECH_FEATURE_NAMES)].to_numpy(dtype=np.float32))
    if mode in ("embedding", "both"):
        if "embedding_path" not in frame.columns or frame["embedding_path"].isna().any():
            raise ConfigError(
                f"data.params.input='{mode}' needs encoder embeddings. Re-run preprocessing with "
                "--encoder, e.g. scripts/prepare_speech.py ... --encoder microsoft/wavlm-base-plus"
            )
        parts.append(np.stack([np.load(processed_dir / path) for path in frame["embedding_path"]]).astype(np.float32))
    return np.concatenate(parts, axis=1)


class SpeechFeatureDataset(Dataset):
    """Pre-computed, normalised feature vectors with targets."""

    def __init__(self, features: np.ndarray, supervision: Supervision, rows: np.ndarray) -> None:
        self.features = torch.from_numpy(features)
        self.supervision = supervision
        self.rows = rows
        self.sample_ids = supervision.frame["sample_id"].astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = int(self.rows[index])
        target, mask = self.supervision.target_tensors(row)
        return {
            "inputs": {"features": self.features[row]},
            "targets": target,
            "mask": mask,
            "sample_id": str(self.sample_ids[row]),
        }


class SpeechWaveformDataset(Dataset):
    """Raw waveforms for encoder fine-tuning (random crop while training)."""

    def __init__(
        self,
        supervision: Supervision,
        rows: np.ndarray,
        sample_rate: int,
        max_seconds: float,
        training: bool,
    ) -> None:
        self.supervision = supervision
        self.rows = rows
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.training = training
        self.paths = supervision.frame["audio_path"].to_numpy()
        self.sample_ids = supervision.frame["sample_id"].astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = int(self.rows[index])
        wave = load_audio(self.paths[row], self.sample_rate)
        if len(wave) > self.max_samples:
            start = np.random.randint(0, len(wave) - self.max_samples + 1) if self.training else 0
            wave = wave[start : start + self.max_samples]
        target, mask = self.supervision.target_tensors(row)
        return {
            "inputs": {"waveform": torch.from_numpy(wave)},
            "targets": target,
            "mask": mask,
            "sample_id": str(self.sample_ids[row]),
        }


def collate_waveforms(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Zero-pad waveforms to the longest in the batch and build the mask."""
    lengths = [len(item["inputs"]["waveform"]) for item in batch]
    waveform = torch.zeros(len(batch), max(lengths))
    attention_mask = torch.zeros(len(batch), max(lengths), dtype=torch.long)
    for i, item in enumerate(batch):
        waveform[i, : lengths[i]] = item["inputs"]["waveform"]
        attention_mask[i, : lengths[i]] = 1
    return {
        "inputs": {"waveform": waveform, "attention_mask": attention_mask},
        "targets": torch.stack([item["targets"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
    }


def build_speech_data(cfg: Config) -> DataBundle:
    """Load processed speech data and build train/val/test datasets."""
    processed_dir = resolve_path(cfg.data.processed_dir)
    frame = read_table(require_processed(processed_dir / FEATURES_FILE, PREPARE_COMMAND))
    manifest = load_manifest(processed_dir)
    mode = cfg.data.params.get("input", "handcrafted")
    if mode not in INPUT_MODES:
        raise ConfigError(f"data.params.input must be one of {INPUT_MODES}, got '{mode}'.")

    supervision = prepare_supervision(frame, cfg.data, cfg.experiment.seed)
    splits = supervision.splits
    common = {
        "task": supervision.task,
        "dataset_metadata": manifest,
        "label_mapping": {"class_names": supervision.task.class_names},
        "class_counts": supervision.class_counts,
        "train_labels": supervision.targets[splits["train"]] if supervision.task.kind == "classification" else None,
    }

    if mode == "waveform":
        if supervision.frame["audio_path"].isna().any():
            raise InvalidLabelError("Some samples have no audio_path; waveform training needs audio for every row.")
        sample_rate = int(manifest.get("sample_rate", 16000))
        max_seconds = float(cfg.data.params.get("max_seconds", 20.0))
        make = lambda split: SpeechWaveformDataset(  # noqa: E731
            supervision, splits[split], sample_rate, max_seconds, training=split == "train"
        )
        return DataBundle(
            train=make("train"), val=make("val"), test=make("test"),
            collate_fn=collate_waveforms,
            normalization={"input": mode, "sample_rate": sample_rate, "max_seconds": max_seconds},
            **common,
        )

    raw = build_feature_matrix(supervision.frame, processed_dir, mode)
    normalizer = fit_normalizer(raw[splits["train"]])
    features = apply_normalizer(raw, normalizer)
    make = lambda split: SpeechFeatureDataset(features, supervision, splits[split])  # noqa: E731
    return DataBundle(
        train=make("train"), val=make("val"), test=make("test"),
        normalization={
            "input": mode,
            **normalizer,
            "feature_names": list(SPEECH_FEATURE_NAMES),
            "encoder_name": manifest.get("encoder_name"),
            "sample_rate": int(manifest.get("sample_rate", 16000)),
        },
        info={"input_dim": int(features.shape[1])},
        **common,
    )
