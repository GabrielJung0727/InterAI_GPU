"""Stage-A local judge: parsing, applicability, consistency check, resume and statistics (no model needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from interview_gpu.answer.labeling import (
    FAILED_SUFFIX,
    LabelingConfig,
    build_judge_messages,
    compare_label_sets,
    create_backend,
    merge_runs,
    parse_judge_response,
    read_jsonl,
    run_labeling,
    select_examples,
)
from interview_gpu.answer.preprocess import load_external_labels
from interview_gpu.answer.prompts import QUESTION_TYPES
from interview_gpu.common.errors import ConfigError, InvalidLabelError
from interview_gpu.schemas.answer import ANSWER_DIMENSIONS

CONDITIONAL = ("technical_depth", "ownership")


def response(score: int = 2, *, applicable: tuple[str, ...] = (), confidence: str = "HIGH", **overrides) -> str:
    """A judge response; ``technical_depth`` / ``ownership`` are null unless listed in ``applicable``."""
    scores = {name: score if name not in CONDITIONAL or name in applicable else None for name in ANSWER_DIMENSIONS}
    scores.update(overrides)
    payload = {
        "question_type": "PERSONALITY",
        "applicability": {name: value is not None for name, value in scores.items()},
        "score_evidence": {"relevance": ["근거"], "ownership": ["제가 했습니다"]},
        "scores": scores,
        "missing_points": ["결과"],
        "judge_confidence": confidence,
        "review_reason": "답변이 잘림" if confidence == "LOW" else None,
    }
    return json.dumps(payload, ensure_ascii=False)


class FakeBackend:
    """Returns scripted responses per call; records how many prompts it saw."""

    batch_size = 4

    def __init__(self, reply, description: str = "fake:model") -> None:
        self.reply = reply
        self.description = description
        self.calls = 0
        self.prompts = 0

    def generate(self, messages, temperature, seed):
        self.calls += 1
        self.prompts += len(messages)
        return [self.reply(self.calls, message[1]["content"]) for message in messages]


def write_answers(path: Path, count: int) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            row = {"sample_id": f"s{index:03d}", "question": f"질문 {index}", "answer": f"답변 {index}", "context": {"job": "BM"}}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def label(tmp_path: Path, backend: FakeBackend, limit: int | None, **config) -> dict:
    return run_labeling(
        LabelingConfig(**config), backend, input_path=tmp_path / "answers.jsonl",
        output_path=tmp_path / "labels.jsonl", limit=limit,
    )


def test_prompt_states_the_rubric_v2_rules() -> None:
    prompt = build_judge_messages("질문", "답변", {"job": "BM"})[1]["content"]
    for needle in (*ANSWER_DIMENSIONS, *QUESTION_TYPES, "null", "독립적으로", "0~4점", "HIGH|MEDIUM|LOW", "직무: BM"):
        assert needle in prompt


def test_parse_judge_response() -> None:
    parsed = parse_judge_response("설명: " + response(3) + " 끝", "s1")
    assert parsed["question_type"] == "PERSONALITY" and parsed["judge_confidence"] == "HIGH"
    assert parsed["scores"]["relevance"] == 3 and isinstance(parsed["scores"]["relevance"], int)
    # Not applicable is null, never 0, and carries no evidence.
    assert parsed["scores"]["technical_depth"] is None and parsed["applicability"]["technical_depth"] is False
    assert parsed["score_evidence"] == {"relevance": ["근거"]} and parsed["review_reason"] is None

    scored = parse_judge_response(response(1, applicable=("ownership",)), "s1")
    assert scored["scores"]["ownership"] == 1 and scored["applicability"]["ownership"] is True
    # A score given for a dimension declared not applicable is discarded.
    payload = json.loads(response(2, technical_depth=0))
    payload["applicability"]["technical_depth"] = False
    assert parse_judge_response(json.dumps(payload), "s1")["scores"]["technical_depth"] is None

    for bad in (
        "no json here",
        response(2, relevance=None),  # always-applicable dimension without a score
        response(5),  # out of the 0-4 range
        response(2, structure=2.5),  # half points are not allowed
        response(2).replace("PERSONALITY", "SMALL_TALK"),
        response(2).replace('"HIGH"', '"0.9"'),
        response(2)[:-20],  # truncated
    ):
        with pytest.raises(InvalidLabelError):
            parse_judge_response(bad, "s1")


def test_merge_runs_flags_disagreement_and_low_confidence() -> None:
    first, close, far = (parse_judge_response(response(1, structure=s)) for s in (1, 2, 3))
    stable = merge_runs("s1", [first, close], review_threshold=2, retries=0)
    # The label is the first run: integers only, no averaged half points.
    assert stable["needs_review"] is False and stable["scores"]["structure"] == 1 and stable["review_reason"] is None
    unstable = merge_runs("s1", [first, far], review_threshold=2, retries=0)
    assert unstable["needs_review"] is True and "structure (1-3)" in unstable["review_reason"]

    with_ownership = parse_judge_response(response(1, applicable=("ownership",)))
    split = merge_runs("s1", [first, with_ownership], review_threshold=2, retries=0)
    assert split["needs_review"] is True and "applicability disagreement: ownership" in split["review_reason"]

    low = merge_runs("s1", [parse_judge_response(response(1, confidence="LOW"))], review_threshold=2, retries=0)
    assert low["needs_review"] is True and "답변이 잘림" in low["review_reason"] and "run_scores" not in low
    medium = merge_runs("s1", [first, parse_judge_response(response(1, confidence="MEDIUM"))], 2, 0)
    assert medium["judge_confidence"] == "MEDIUM" and medium["needs_review"] is False


def test_select_examples_is_a_stable_prefix() -> None:
    examples = [{"sample_id": f"s{i}"} for i in range(50)]
    small, large = select_examples(examples, 5, 42), select_examples(examples[::-1], 20, 42)
    assert small == large[:5] and len(select_examples(examples, None, 42)) == 50


def test_labeling_resumes_and_reports(tmp_path: Path) -> None:
    write_answers(tmp_path / "answers.jsonl", 20)
    backend = FakeBackend(lambda call, prompt: response(3))
    stats = label(tmp_path, backend, limit=8)
    assert stats["num_labeled"] == 8 and stats["parse_failure_rate"] == 0 and backend.prompts == 8
    assert stats["dimensions"]["relevance"] == {
        "applicable_rate": 1.0, "null_rate": 0.0, "mean": 3.0, "median": 3.0, "std": 0.0, "distribution": {"3": 8},
    }
    assert stats["dimensions"]["technical_depth"]["null_rate"] == 1.0
    assert stats["question_type"] == {"PERSONALITY": 8} and stats["judge_confidence"] == {"HIGH": 8, "MEDIUM": 0, "LOW": 0}
    assert stats["needs_review_rate"] == 0 and set(stats["correlation_spearman"]) == set(ANSWER_DIMENSIONS)

    # A larger limit only labels the new samples; the first 8 are kept.
    stats = label(tmp_path, backend, limit=20)
    assert stats["num_labeled"] == 20 and backend.prompts == 20
    records = read_jsonl(tmp_path / "labels.jsonl")
    assert len({record["sample_id"] for record in records}) == 20
    assert set(records[0]) == {
        "sample_id", "question_type", "applicability", "scores", "score_evidence", "missing_points",
        "judge_confidence", "needs_review", "review_reason", "parse_retries",
    }
    review = (tmp_path / "labels.review.csv").read_text(encoding="utf-8-sig")
    assert "question_type" in review.splitlines()[0] and "답변 3" in review
    # Not-applicable dimensions are left unlabeled when merged into the training set.
    merged = load_external_labels(tmp_path / "labels.jsonl")
    assert len(merged) == 20 and "technical_depth" not in merged["s000"] and merged["s000"]["relevance"] == 3.0


def test_labeling_retries_then_records_failures(tmp_path: Path) -> None:
    write_answers(tmp_path / "answers.jsonl", 4)

    def reply(call: int, prompt: str) -> str:
        if "답변 0" in prompt:
            return "not json"  # never parses
        return response(2) if call > 1 else "{broken"  # everything else parses on the first retry

    stats = label(tmp_path, FakeBackend(reply), limit=None)
    assert stats["num_labeled"] == 3 and stats["num_failed"] == 1
    assert stats["parse_failure_rate"] == 0.25 and stats["first_attempt_failure_rate"] == 1.0
    failed = read_jsonl(tmp_path / f"labels{FAILED_SUFFIX}")
    assert [record["sample_id"] for record in failed] == ["s000"]

    # The failed sample is retried on the next run and leaves the failure file once it succeeds.
    stats = label(tmp_path, FakeBackend(lambda call, prompt: response(2)), limit=None)
    assert stats["num_labeled"] == 4 and stats["num_failed"] == 0
    assert read_jsonl(tmp_path / f"labels{FAILED_SUFFIX}") == []


def test_consistency_runs_mark_needs_review(tmp_path: Path) -> None:
    write_answers(tmp_path / "answers.jsonl", 4)
    # Second run disagrees by 2 points on one dimension for one answer.
    backend = FakeBackend(lambda call, prompt: response(1, evidence=3 if call == 2 and "답변 1" in prompt else 1))
    stats = label(tmp_path, backend, limit=None, consistency_runs=2)
    assert stats["needs_review_rate"] == 0.25 and backend.prompts == 8
    flagged = [record for record in read_jsonl(tmp_path / "labels.jsonl") if record["needs_review"]]
    assert [record["sample_id"] for record in flagged] == ["s001"] and len(flagged[0]["run_scores"]) == 2
    assert flagged[0]["scores"]["evidence"] == 1
    assert len(load_external_labels(tmp_path / "labels.jsonl")) == 3  # flagged labels are not merged


def test_changed_settings_refuse_to_append(tmp_path: Path) -> None:
    write_answers(tmp_path / "answers.jsonl", 4)
    label(tmp_path, FakeBackend(lambda call, prompt: response(2)), limit=2)
    with pytest.raises(ConfigError, match="different settings"):
        label(tmp_path, FakeBackend(lambda call, prompt: response(2), "fake:other-model"), limit=4)


def test_compare_label_sets_across_scales() -> None:
    # V1: 0-10, every dimension scored and moving together. V2: 0-4 with nulls.
    old = [{"sample_id": f"s{i}", "scores": dict.fromkeys(ANSWER_DIMENSIONS, float(i))} for i in range(2, 9)]
    new = [
        {"sample_id": f"s{i}", "scores": {**dict.fromkeys(ANSWER_DIMENSIONS, i % 5), "structure": (i * 3) % 5,
                                         "technical_depth": None}}
        for i in range(2, 9)
    ]
    report = compare_label_sets(old, new, old_scale=10, new_scale=4)
    assert report["num_common"] == 7 and report["halo"]["old"]["pearson_mean_offdiag"] == pytest.approx(1.0)
    assert report["halo"]["new"]["pearson_mean_offdiag"] < 1.0
    depth = report["dimensions"]["technical_depth"]
    assert depth["new_null_rate"] == 1.0 and depth["old_null_rate"] == 0.0 and depth["change"] is None
    assert report["dimensions"]["relevance"]["old_mean"] == pytest.approx(0.5)


def test_vllm_is_not_forced_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    import interview_gpu.answer.labeling as labeling

    built = []
    monkeypatch.setattr(labeling, "TransformersBackend", lambda config, device: built.append("transformers") or "t")
    monkeypatch.setattr(labeling, "VLLMBackend", lambda config: built.append("vllm") or "v")
    monkeypatch.setattr(labeling.sys, "platform", "win32")
    assert create_backend(LabelingConfig(backend="vllm")) == "t"
    monkeypatch.setattr(labeling.sys, "platform", "linux")
    assert create_backend(LabelingConfig(backend="vllm")) == "v"
    assert create_backend(LabelingConfig()) == "t" and built == ["transformers", "vllm", "transformers"]
