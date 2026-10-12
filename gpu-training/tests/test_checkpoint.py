"""Checkpoint save/load and the model registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import load_torch

torch = load_torch()

from interview_gpu.common.checkpoint import (  # noqa: E402
    latest_version_dir,
    load_checkpoint,
    load_model_state,
    next_version_dir,
    save_checkpoint,
)
from interview_gpu.common.errors import CheckpointMismatchError  # noqa: E402


def _save(directory: Path, model: torch.nn.Module, **overrides) -> Path:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    arguments = dict(
        model=model, optimizer=optimizer, scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0),
        scaler=torch.amp.GradScaler("cpu", enabled=False), epoch=3, metric_name="val_loss", metric_value=0.25,
        config={"model": {"name": "speech_mlp", "params": {}}, "data": {"processed_dir": "x"}},
        task={"kind": "regression", "target_names": ["pace_score"], "class_names": None,
              "target_mean": [1.0], "target_std": [2.0]},
        label_mapping={"class_names": None},
        normalization={"mean": [0.0, 1.0], "std": [1.0, 2.0]},
        dataset_metadata={"dataset_name": "unit-test", "usage_scope": "research"},
        metrics={"validation": {"mean": {"mae": 0.1}}},
    )
    arguments.update(overrides)
    return save_checkpoint(directory, **arguments)


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    path = _save(tmp_path / "speech_v001", model)
    assert path.name == "model.pt"
    assert {p.name for p in path.parent.iterdir()} == {"model.pt", "config.yaml", "metrics.json", "metadata.json"}

    payload = load_checkpoint(path.parent)  # a registry directory is accepted
    for key in ("model_state", "optimizer_state", "scheduler_state", "scaler_state", "epoch", "metric_value",
                "config", "label_mapping", "normalization", "dataset_metadata", "training_timestamp",
                "git_commit", "task"):
        assert key in payload
    assert payload["epoch"] == 3 and payload["metric_value"] == 0.25
    assert payload["normalization"] == {"mean": [0.0, 1.0], "std": [1.0, 2.0]}

    restored = torch.nn.Linear(2, 1)
    load_model_state(restored, payload["model_state"])
    assert torch.equal(restored.weight, model.weight)
    optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    optimizer.load_state_dict(payload["optimizer_state"])  # resumable

    metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_metadata"]["dataset_name"] == "unit-test"
    assert "model_state" not in metadata


def test_architecture_mismatch_is_reported(tmp_path: Path) -> None:
    payload = load_checkpoint(_save(tmp_path / "ckpt", torch.nn.Linear(2, 1)))
    with pytest.raises(CheckpointMismatchError, match="does not match the model architecture"):
        load_model_state(torch.nn.Linear(3, 1), payload["model_state"])


def test_missing_or_corrupted_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(CheckpointMismatchError, match="not found"):
        load_checkpoint(tmp_path / "nope")
    broken = tmp_path / "model.pt"
    broken.write_bytes(b"garbage")
    with pytest.raises(CheckpointMismatchError, match="Could not read"):
        load_checkpoint(broken)


def test_registry_versions(tmp_path: Path) -> None:
    assert latest_version_dir(tmp_path, "speech") is None
    first = next_version_dir(tmp_path, "speech")
    second = next_version_dir(tmp_path, "speech")
    assert (first.name, second.name) == ("speech_v001", "speech_v002")
    assert first.parent == tmp_path / "speech"
    assert latest_version_dir(tmp_path, "speech") is None  # directories without a model are ignored
    _save(first, torch.nn.Linear(2, 1))
    assert latest_version_dir(tmp_path, "speech") == first
    assert next_version_dir(tmp_path, "video").name == "video_v001"
