"""Dataset adapters, splitting, inspection and CPU preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import needs_parquet
from interview_gpu.answer.preprocess import prepare_answer_dataset
from interview_gpu.common.errors import (
    ConfigError,
    DatasetNotFoundError,
    EmptyDatasetError,
    InvalidLabelError,
    UsageScopeError,
)
from interview_gpu.datasets import ADAPTERS, ensure_scope_allowed, get_adapter
from interview_gpu.datasets.inspection import inspect_dataset
from interview_gpu.datasets.internal import load_before_after
from interview_gpu.datasets.splits import group_split
from interview_gpu.schemas.interview import BeforeAfterSample, InterviewSample
from interview_gpu.speech.preprocess import prepare_speech_dataset


def test_registry_covers_all_planned_datasets() -> None:
    assert {"aihub_interview", "aihub_speech", "aihub_multimodal", "aihub_robot_interaction",
            "aihub_korean_face", "mit_interview", "chalearn", "internal"} == set(ADAPTERS)
    with pytest.raises(ConfigError, match="Unknown dataset"):
        get_adapter("nope")


def test_missing_dataset_gives_actionable_error(tmp_path: Path) -> None:
    adapter = get_adapter("aihub_interview", tmp_path / "aihub-interview")
    with pytest.raises(DatasetNotFoundError) as error:
        adapter.load()
    message = str(error.value)
    assert "Dataset not found." in message
    assert "Expected path:" in message
    assert "aihub-interview/" in message
    assert "never downloaded automatically" in message


def test_default_root_is_under_data_raw() -> None:
    assert get_adapter("aihub_interview").root.as_posix().endswith("data/raw/aihub-interview")


def test_unrecognised_layout_is_reported_as_empty(tmp_path: Path) -> None:
    root = tmp_path / "weird"
    root.mkdir()
    (root / "notes.txt").write_text("not a dataset")
    with pytest.raises(EmptyDatasetError, match="No usable samples"):
        get_adapter("aihub_interview", root).load()


def test_aihub_interview_adapter(aihub_interview_dir: Path) -> None:
    adapter = get_adapter("aihub_interview", aihub_interview_dir)
    samples = adapter.load()
    assert len(samples) == 2  # the record with an empty answer is skipped
    assert adapter.skipped["missing_transcript"] == 1
    assert adapter.skipped["missing_audio"] == 1
    first = samples[0]
    assert isinstance(first, InterviewSample)
    assert first.question == "자기소개를 해주세요."
    assert first.transcript == first.answer
    assert first.job_family == "ICT"
    assert first.labels == {"answer_intent": "experience", "question_intent": "background"}
    assert first.audio_path and Path(first.audio_path).is_file()
    assert samples[1].audio_path is None
    assert first.source_dataset == "aihub_interview" and first.usage_scope == "research"
    assert first.group_id == first.sample_id  # no speaker metadata -> per-sample group


def test_mit_adapter_exposes_only_behaviour_labels(mit_interview_dir: Path) -> None:
    samples = get_adapter("mit_interview", mit_interview_dir).load()
    assert len(samples) == 12
    by_id = {s.sample_id: s for s in samples}
    assert by_id["P1"].speaker_id == by_id["PP1"].speaker_id == "mit_1"
    labels = by_id["P1"].labels
    assert labels["SpeakingRate"] == pytest.approx(3.4)  # aggregated row, not the single worker
    assert "RecommendHiring" not in labels and "Overall" not in labels


def test_internal_adapter_requires_consent(internal_dir: Path) -> None:
    adapter = get_adapter("internal", internal_dir)
    samples = adapter.load()
    assert adapter.skipped["no_consent"] == 1
    assert len(samples) == 6 + 12  # 6 single answers + 6 pairs x (before, after)
    assert all(s.consent is not None and s.consent.opt_in for s in samples)
    assert all(s.usage_scope == "production" for s in samples)
    before = next(s for s in samples if s.sample_id == "pair_1_before")
    after = next(s for s in samples if s.sample_id == "pair_1_after")
    assert before.group_id == after.group_id  # a pair never straddles train/test
    assert before.metadata["deltas"]["delta_specificity"] == 4.0


def test_before_after_deltas(internal_dir: Path) -> None:
    pairs = load_before_after(internal_dir)
    assert len(pairs) == 6
    deltas = pairs[0].deltas()
    assert deltas["delta_relevance"] == 2.0 and deltas["delta_specificity"] == 4.0
    assert deltas["delta_structure"] is None  # not scored -> not fabricated
    assert deltas["overall_delta"] == 3.0
    empty = BeforeAfterSample(pair_id="p", question="q", before_answer="a", after_answer="b")
    assert empty.deltas()["overall_delta"] is None


def test_research_data_is_blocked_from_production() -> None:
    ensure_scope_allowed(ADAPTERS["internal"].license, "production")
    ensure_scope_allowed(ADAPTERS["aihub_interview"].license, "research")
    for name in ("aihub_interview", "mit_interview", "chalearn"):
        with pytest.raises(UsageScopeError):
            ensure_scope_allowed(ADAPTERS[name].license, "production")


def test_group_split_has_no_leakage() -> None:
    groups = np.repeat([f"speaker_{i}" for i in range(20)], 5)
    splits = group_split(groups, 0.2, 0.2, seed=1)
    members = {name: set(groups[idx]) for name, idx in splits.items()}
    assert not members["train"] & members["val"]
    assert not members["train"] & members["test"]
    assert not members["val"] & members["test"]
    assert sum(len(idx) for idx in splits.values()) == len(groups)
    assert len(members["val"]) == 4 and len(members["test"]) == 4
    again = group_split(groups, 0.2, 0.2, seed=1)
    assert all(np.array_equal(splits[k], again[k]) for k in splits)  # deterministic


def test_group_split_rejects_too_few_groups() -> None:
    with pytest.raises(EmptyDatasetError, match="distinct groups"):
        group_split(["a", "a", "b"], 0.2, 0.2, seed=0)
    with pytest.raises(EmptyDatasetError):
        group_split([], 0.2, 0.2, seed=0)


def test_inspect_dataset_report(aihub_interview_dir: Path) -> None:
    corrupt = aihub_interview_dir / "audio" / "ckmk_d_ict_f_e_00000.wav"
    corrupt.write_bytes(b"this is not audio")
    report = inspect_dataset(get_adapter("aihub_interview", aihub_interview_dir))
    assert report["number_of_samples"] == 2
    assert report["audio_files"] == 1 and report["text_samples"] == 2 and report["video_files"] == 0
    assert report["corrupted_files"] == 1
    assert report["label_distribution"]["answer_intent"] == {"experience": 2}
    assert report["skipped_records"]["missing_transcript"] == 1
    assert report["disk_size_bytes"] > 0
    json.dumps(report)  # JSON serializable


def test_inspect_dataset_durations(mit_interview_dir: Path) -> None:
    report = inspect_dataset(get_adapter("mit_interview", mit_interview_dir))
    assert report["speaker_count"] == 6 and report["audio_files"] == 12
    assert 1.0 < report["average_audio_duration_sec"] < 2.0
    assert report["label_distribution"]["EyeContact"]["count"] == 12


@needs_parquet
def test_prepare_speech_dataset(mit_interview_dir: Path, tmp_path: Path) -> None:
    (mit_interview_dir / "Audio" / "P1.wav").write_bytes(b"broken")
    adapter = get_adapter("mit_interview", mit_interview_dir)
    manifest = prepare_speech_dataset(adapter, tmp_path / "speech")
    frame = pd.read_parquet(tmp_path / "speech" / "features.parquet")
    assert len(frame) == 11 and manifest["num_samples"] == 11
    assert manifest["skipped_records"]["corrupted_audio"] == 1
    assert manifest["usage_scope"] == "research" and manifest["license"]["research_only"] is True
    assert "label_SpeakingRate" in frame.columns and "label_RecommendHiring" not in frame.columns
    assert frame["pause_count"].min() >= 1
    assert frame["group_id"].nunique() == 6


def test_prepare_answer_dataset(internal_dir: Path, tmp_path: Path) -> None:
    prompts = tmp_path / "prompts.jsonl"
    manifest = prepare_answer_dataset(get_adapter("internal", internal_dir), tmp_path / "answer", prompts_path=prompts)
    assert manifest["num_samples"] == 18 and manifest["num_labeled"] == 18
    assert manifest["usage_scope"] == "production"
    lines = (tmp_path / "answer" / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert set(first) == {"sample_id", "group_id", "question", "answer", "context", "scores", "metadata"}
    assert first["context"]["resume_context"] is None  # resumes are never attached automatically
    assert prompts.read_text(encoding="utf-8") == ""  # everything already labeled


def test_prepare_answer_merges_and_validates_external_labels(aihub_interview_dir: Path, tmp_path: Path) -> None:
    adapter = get_adapter("aihub_interview", aihub_interview_dir)
    prompts = tmp_path / "prompts.jsonl"
    manifest = prepare_answer_dataset(adapter, tmp_path / "a1", prompts_path=prompts)
    assert manifest["num_labeled"] == 0
    exported = [json.loads(line) for line in prompts.read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 2 and "relevance" in exported[0]["prompt"]

    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps({"sample_id": exported[0]["sample_id"], "scores": {"relevance": 8}}) + "\n")
    assert prepare_answer_dataset(adapter, tmp_path / "a2", labels_path=labels)["num_labeled"] == 1

    labels.write_text(json.dumps({"sample_id": "x", "scores": {"relevance": 42}}) + "\n")
    with pytest.raises(InvalidLabelError, match="outside"):
        prepare_answer_dataset(adapter, tmp_path / "a3", labels_path=labels)
