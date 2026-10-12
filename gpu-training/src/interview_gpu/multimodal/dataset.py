"""Multimodal embedding dataset (skeleton).

Expected layout — to be produced by a future alignment step once aligned
audio + video + text interview data exists::

    data/processed/multimodal/
        index.parquet    sample_id, group_id, speech_path, video_path, text_path, label_*
        <modality>/<id>.npy     pooled embedding per modality (path may be null)
        manifest.json           must contain "embedding_dims": {"speech": .., "video": .., "text": ..}

No preprocessing script generates this yet; without the files, training stops
with a clear "dataset not found" message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from interview_gpu.common.config import Config, resolve_path
from interview_gpu.common.errors import ConfigError
from interview_gpu.common.runner import DataBundle, load_manifest
from interview_gpu.common.tasks import Supervision, prepare_supervision
from interview_gpu.datasets.processed import require_processed
from interview_gpu.multimodal.model import MODALITIES

INDEX_FILE = "index.parquet"
PREPARE_HINT = "(no preprocessing script yet: aligned multimodal interview data is required first)"


class MultimodalEmbeddingDataset(Dataset):
    """Per-sample modality embeddings with a presence mask."""

    def __init__(
        self, supervision: Supervision, rows: np.ndarray, processed_dir: Path, input_dims: dict[str, int]
    ) -> None:
        self.supervision = supervision
        self.rows = rows
        self.processed_dir = processed_dir
        self.input_dims = input_dims
        self.records = supervision.frame.to_dict("records")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = int(self.rows[index])
        record = self.records[row]
        inputs: dict[str, torch.Tensor] = {}
        present = []
        for name in MODALITIES:
            path = record.get(f"{name}_path")
            has = isinstance(path, str) and bool(path)
            vector = np.load(self.processed_dir / path) if has else np.zeros(self.input_dims[name])
            inputs[name] = torch.from_numpy(np.asarray(vector, dtype=np.float32))
            present.append(float(has))
        inputs["modality_mask"] = torch.tensor(present)
        target, mask = self.supervision.target_tensors(row)
        return {"inputs": inputs, "targets": target, "mask": mask, "sample_id": str(record["sample_id"])}


def build_multimodal_data(cfg: Config) -> DataBundle:
    """Load processed multimodal embeddings and build train/val/test datasets."""
    processed_dir = resolve_path(cfg.data.processed_dir)
    frame = pd.read_parquet(require_processed(processed_dir / INDEX_FILE, PREPARE_HINT))
    manifest = load_manifest(processed_dir)
    input_dims = manifest.get("embedding_dims")
    if not isinstance(input_dims, dict) or set(input_dims) != set(MODALITIES):
        raise ConfigError(f"manifest.json must define embedding_dims for {MODALITIES}.")
    supervision = prepare_supervision(frame, cfg.data, cfg.experiment.seed)
    make = lambda split: MultimodalEmbeddingDataset(  # noqa: E731
        supervision, supervision.splits[split], processed_dir, input_dims
    )
    return DataBundle(
        train=make("train"), val=make("val"), test=make("test"),
        task=supervision.task,
        dataset_metadata=manifest,
        info={"input_dims": {name: int(input_dims[name]) for name in MODALITIES}},
    )
