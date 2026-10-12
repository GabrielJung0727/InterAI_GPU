"""Tiny end-to-end training loops on throw-away dummy data.

These only prove the pipeline wiring (data -> model -> loss -> checkpoint ->
evaluation -> inference). The data is random, so no metric value here means
anything about model quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import base_config, load_torch, needs_parquet

torch = load_torch()
pytestmark = needs_parquet

from interview_gpu.answer.dataset import build_answer_data  # noqa: E402
from interview_gpu.answer.model import AnswerEvaluatorModel  # noqa: E402
from interview_gpu.common.checkpoint import latest_version_dir, load_checkpoint  # noqa: E402
from interview_gpu.common.config import config_from_dict  # noqa: E402
from interview_gpu.common.errors import (  # noqa: E402
    DatasetNotFoundError,
    InvalidLabelError,
    NaNLossError,
    UsageScopeError,
)
from interview_gpu.common.io import write_json  # noqa: E402
from interview_gpu.common.runner import run_evaluation, run_training  # noqa: E402
from interview_gpu.schemas.answer import ANSWER_DIMENSIONS  # noqa: E402
from interview_gpu.schemas.speech import SPEECH_FEATURE_NAMES  # noqa: E402
from interview_gpu.schemas.video import VIDEO_FEATURE_COLUMNS  # noqa: E402
from interview_gpu.speech import evaluate as speech_evaluate  # noqa: E402
from interview_gpu.speech import train as speech_train  # noqa: E402
from interview_gpu.speech.inference import SpeechAnalyzer  # noqa: E402
from interview_gpu.video import train as video_train  # noqa: E402
from interview_gpu.video.inference import VideoBehaviorModel  # noqa: E402
from interview_gpu.video.temporal_features import load_feature_table  # noqa: E402

SPEAKERS, PER_SPEAKER = 10, 4


def make_speech_data(directory: Path, usage_scope: str = "research") -> Path:
    rng = np.random.default_rng(0)
    rows = SPEAKERS * PER_SPEAKER
    frame = pd.DataFrame(rng.normal(size=(rows, len(SPEECH_FEATURE_NAMES))), columns=list(SPEECH_FEATURE_NAMES))
    frame["filler_count"] = np.nan  # a feature that is missing everywhere must not break training
    frame.insert(0, "sample_id", [f"s{i}" for i in range(rows)])
    frame.insert(1, "group_id", np.repeat([f"spk{i}" for i in range(SPEAKERS)], PER_SPEAKER))
    frame["label_pace"] = rng.uniform(1, 7, rows)
    frame["label_pause"] = rng.uniform(1, 7, rows)
    frame.loc[::5, "label_pause"] = np.nan  # partially labeled head
    frame["label_band"] = np.where(frame["label_pace"] > 5.5, "fast", "normal")
    directory.mkdir(parents=True)
    frame.to_parquet(directory / "features.parquet", index=False)
    write_json(directory / "manifest.json", {"dataset_name": "dummy", "usage_scope": usage_scope, "sample_rate": 16000})
    return directory


def speech_config(tmp_path: Path, processed: Path, **kwargs) -> dict:
    data = {"target_columns": {"pace_score": "label_pace", "pause_score": "label_pause"},
            "split": {"val_ratio": 0.2, "test_ratio": 0.2}}
    data.update(kwargs.pop("data", {}))
    return base_config(tmp_path, "speech", processed, {"name": "speech_mlp", "params": {"hidden_dims": [16]}},
                       data=data, **kwargs)


def test_speech_training_end_to_end(tmp_path: Path) -> None:
    processed = make_speech_data(tmp_path / "processed")
    cfg = config_from_dict(speech_config(tmp_path, processed, training={"gradient_accumulation_steps": 2},
                                         checkpoint={"metric": "val_mean_mae", "mode": "min"}))
    summary = speech_train.train(cfg)

    assert summary["epochs_run"] == 2 and summary["best_metric_name"] == "val_mean_mae"
    assert set(summary["metrics"]) == {"validation", "test"}
    assert set(summary["metrics"]["test"]["per_head"]) == {"pace_score", "pause_score"}
    assert "error_distribution" in summary["metrics"]["test"]  # calibration statistics
    sizes = summary["split_sizes"]
    assert sizes == {"train": 24, "val": 8, "test": 8}

    output_dir, checkpoint_dir = Path(summary["output_dir"]), Path(summary["checkpoint_dir"])
    for name in ("train.log", "metrics.json", "history.csv", "config.yaml", "loss_curve.png", "predictions.parquet"):
        assert (output_dir / name).is_file(), name
    assert checkpoint_dir.name == "speech_v001"
    for name in ("model.pt", "config.yaml", "metrics.json", "metadata.json"):
        assert (checkpoint_dir / name).is_file(), name
    history = pd.read_csv(output_dir / "history.csv")
    assert {"train_loss", "val_loss", "val_mean_mae", "lr", "gpu_max_allocated_mb", "gpu_max_reserved_mb"} <= set(history)
    assert np.isfinite(history["train_loss"]).all()
    predictions = pd.read_parquet(output_dir / "predictions.parquet")
    assert len(predictions) == 8 and {"sample_id", "pred_pace_score", "true_pace_score"} <= set(predictions)

    payload = load_checkpoint(checkpoint_dir)
    assert payload["task"]["target_names"] == ["pace_score", "pause_score"]
    assert len(payload["normalization"]["mean"]) == len(SPEECH_FEATURE_NAMES)
    assert payload["dataset_metadata"]["dataset_name"] == "dummy"

    # A second run registers a new version instead of overwriting.
    assert Path(speech_train.train(cfg)["checkpoint_dir"]).name == "speech_v002"
    assert latest_version_dir(tmp_path / "checkpoints", "speech").name == "speech_v002"

    report = speech_evaluate.evaluate(checkpoint_dir, split="test", device_override="cpu")
    assert report["num_samples"] == 8
    assert report["metrics"]["mean"]["mae"] == pytest.approx(summary["metrics"]["test"]["mean"]["mae"], rel=1e-4)
    assert (checkpoint_dir / "eval_test" / "predictions.parquet").is_file()


def test_speech_analyzer_interface(tmp_path: Path) -> None:
    from conftest import silence, tone

    wave = np.concatenate([tone(1.0), silence(0.5), tone(1.0)])
    metrics_only = SpeechAnalyzer(device="cpu").predict(wave, transcript="음 안녕하세요 저는 지원자입니다")
    assert metrics_only["scores"] == {} and metrics_only["model_version"] is None
    assert metrics_only["metrics"]["pause_count"] == 1.0 and metrics_only["metrics"]["filler_count"] == 1.0

    processed = make_speech_data(tmp_path / "processed")
    summary = speech_train.train(config_from_dict(speech_config(tmp_path, processed)))
    result = SpeechAnalyzer(summary["checkpoint_dir"], device="cpu").predict(wave)
    assert set(result["scores"]) == {"pace_score", "pause_score"}
    assert result["model_version"] == "speech_v001"
    assert result["metrics"]["speech_rate_wpm"] is None  # no transcript -> null, not a made-up value
    json.dumps(result)  # JSON serializable


def test_classification_with_imbalance_options(tmp_path: Path) -> None:
    processed = make_speech_data(tmp_path / "processed")
    cfg = config_from_dict(speech_config(
        tmp_path, processed,
        data={"task": "classification", "target_columns": ["label_band"]},
        training={"imbalance": {"class_weights": True, "weighted_sampler": True, "focal_loss": True}},
        checkpoint={"metric": "val_balanced_accuracy", "mode": "max"},
    ))
    summary = speech_train.train(cfg)
    test_metrics = summary["metrics"]["test"]
    assert {"accuracy", "balanced_accuracy", "macro_f1", "precision", "recall", "ece", "brier"} <= set(test_metrics)
    assert (Path(summary["output_dir"]) / "confusion_matrix.png").is_file()
    assert load_checkpoint(summary["checkpoint_dir"])["task"]["class_names"] == ["fast", "normal"]


def test_training_fails_clearly_without_data_or_labels(tmp_path: Path) -> None:
    missing = config_from_dict(speech_config(tmp_path, tmp_path / "nothing_here"))
    with pytest.raises(DatasetNotFoundError) as error:
        speech_train.train(missing)
    assert "Dataset not found." in str(error.value) and "prepare_speech.py" in str(error.value)

    processed = make_speech_data(tmp_path / "processed")
    wrong = config_from_dict(speech_config(tmp_path, processed, data={"target_columns": ["label_confidence"]}))
    with pytest.raises(InvalidLabelError, match="label_pace"):  # lists what is available
        speech_train.train(wrong)
    unlabeled = config_from_dict(speech_config(tmp_path, processed, data={"target_columns": []}))
    with pytest.raises(InvalidLabelError, match="No target columns configured"):
        speech_train.train(unlabeled)


def test_research_data_cannot_train_production_model(tmp_path: Path) -> None:
    processed = make_speech_data(tmp_path / "processed", usage_scope="research")
    cfg = config_from_dict(speech_config(tmp_path, processed, data={"usage_scope": "production"}))
    with pytest.raises(UsageScopeError, match="never mixed into production"):
        speech_train.train(cfg)


def test_nan_loss_stops_training(tmp_path: Path) -> None:
    processed = make_speech_data(tmp_path / "processed")
    cfg = config_from_dict(speech_config(tmp_path, processed, training={"lr": 1e30, "grad_clip_norm": None,
                                                                         "scheduler": "none", "epochs": 30}))
    with pytest.raises(NaNLossError):
        speech_train.train(cfg)


def make_video_data(directory: Path) -> Path:
    rng = np.random.default_rng(1)
    (directory / "sequences").mkdir(parents=True)
    rows = []
    for index in range(SPEAKERS * 2):
        frames = int(rng.integers(8, 30))
        table = rng.normal(size=(frames, len(VIDEO_FEATURE_COLUMNS))).astype(np.float32)
        table[:, 0] = np.arange(frames) * 0.2
        table[rng.random(frames) < 0.2, 1:4] = np.nan  # frames without a detected face
        np.save(directory / "sequences" / f"v{index}.npy", table)
        rows.append({"sample_id": f"v{index}", "group_id": f"spk{index // 2}",
                     "sequence_path": f"sequences/v{index}.npy", "num_frames": frames,
                     "label_engagement": float(rng.uniform(1, 7))})
    pd.DataFrame(rows).to_parquet(directory / "index.parquet", index=False)
    write_json(directory / "manifest.json", {"dataset_name": "dummy", "usage_scope": "research", "target_fps": 5.0})
    return directory


@pytest.mark.parametrize("backbone", ["gru", "transformer"])
def test_video_training_and_inference(tmp_path: Path, backbone: str) -> None:
    processed = make_video_data(tmp_path / "processed")
    cfg = config_from_dict(base_config(
        tmp_path, "video", processed,
        {"name": "video_temporal", "params": {"backbone": backbone, "hidden_dim": 16, "num_heads": 2, "num_layers": 1}},
        data={"target_columns": {"camera_engagement": "label_engagement"},
              "split": {"val_ratio": 0.2, "test_ratio": 0.2}, "params": {"max_frames": 16}},
    ))
    summary = video_train.train(cfg)
    assert Path(summary["checkpoint_dir"]).name == "video_v001"
    assert set(summary["metrics"]["test"]["per_head"]) == {"camera_engagement"}

    model = VideoBehaviorModel(summary["checkpoint_dir"], device="cpu")
    table = load_feature_table(processed / "sequences" / "v0.npy")
    result = model.predict(table)  # pre-extracted landmarks: no MediaPipe needed
    assert set(result["scores"]) == {"camera_engagement"} and result["model_version"] == "video_v001"
    assert "face_visible_ratio" in result["metrics"]
    json.dumps(result)


class FakeTokenizer:
    """Deterministic whitespace tokenizer standing in for a HuggingFace tokenizer."""

    def __call__(self, first, second, padding=True, truncation=True, max_length=32, return_tensors="pt"):
        encoded = []
        for a, b in zip(first, second):
            tokens = [3 + sum(map(ord, word)) % 90 for word in f"{a} {b}".split()][: max_length - 2]
            encoded.append([1] + tokens + [2])
        width = max(len(ids) for ids in encoded)
        input_ids = torch.zeros(len(encoded), width, dtype=torch.long)
        attention_mask = torch.zeros(len(encoded), width, dtype=torch.long)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            attention_mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_answer_data(directory: Path) -> Path:
    rng = np.random.default_rng(2)
    directory.mkdir(parents=True)
    with (directory / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(30):
            scores = {name: float(rng.integers(0, 11)) for name in ANSWER_DIMENSIONS if rng.random() > 0.2}
            handle.write(json.dumps({
                "sample_id": f"a{index}", "group_id": f"cand{index // 3}",
                "question": "프로젝트 경험을 설명해주세요.", "answer": f"저는 서비스 {index}의 성능을 개선했습니다.",
                "context": {"job": "Backend Engineer", "interview_type": "technical"},
                "scores": scores if index else {},  # one fully unlabeled example is dropped
            }, ensure_ascii=False) + "\n")
    write_json(directory / "manifest.json", {"dataset_name": "dummy", "usage_scope": "research"})
    return directory


def test_answer_evaluator_training(tmp_path: Path) -> None:
    from transformers import BertConfig, BertModel

    processed = make_answer_data(tmp_path / "processed")
    cfg = config_from_dict(base_config(
        tmp_path, "answer", processed, {"name": "answer_evaluator", "params": {}},
        data={"target_columns": [f"label_{name}" for name in ANSWER_DIMENSIONS],
              "split": {"val_ratio": 0.2, "test_ratio": 0.2}, "params": {"max_length": 32}},
        checkpoint={"metric": "val_mean_mae", "mode": "min"},
    ))

    def build_model(cfg, bundle):
        encoder = BertModel(BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1,
                                       num_attention_heads=2, intermediate_size=64, max_position_embeddings=64))
        return AnswerEvaluatorModel(encoder, 32, bundle.task.target_names, head_hidden_dim=16)

    build_data = lambda cfg: build_answer_data(cfg, tokenizer=FakeTokenizer())  # noqa: E731
    summary = run_training(cfg, build_data, build_model)
    assert sum(summary["split_sizes"].values()) == 29
    per_head = summary["metrics"]["test"]["per_head"]
    assert set(per_head) == set(ANSWER_DIMENSIONS)  # one metric block per rubric head
    assert summary["metrics"]["test"]["mean"]["mae"] is not None

    report = run_evaluation(summary["checkpoint_dir"], build_data, build_model, split="val", device_override="cpu")
    assert report["split"] == "val" and set(report["metrics"]["per_head"]) == set(ANSWER_DIMENSIONS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_speech_training_on_cuda_with_mixed_precision(tmp_path: Path) -> None:
    processed = make_speech_data(tmp_path / "processed")
    raw = speech_config(tmp_path, processed)
    raw["device"] = {"type": "cuda"}
    summary = speech_train.train(config_from_dict(raw))
    assert summary["environment"]["selected_device"].startswith("cuda")
    assert summary["environment"]["mixed_precision"] in ("bf16", "fp16")
    history = pd.read_csv(Path(summary["output_dir"]) / "history.csv")
    assert (history["gpu_max_allocated_mb"] > 0).all()
