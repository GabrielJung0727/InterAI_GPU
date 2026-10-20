"""Metrics, calibration, normalisation and answer schema helpers (no torch)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from interview_gpu.answer.prompts import build_labeling_prompt, format_answer_input, parse_labeling_response
from interview_gpu.common.errors import InvalidLabelError
from interview_gpu.common.io import to_jsonable
from interview_gpu.common.metrics import (
    brier_score,
    classification_metrics,
    error_distribution,
    expected_calibration_error,
    flatten_metrics,
    multi_output_regression_metrics,
    regression_metrics,
)
from interview_gpu.common.normalization import apply_normalizer, fit_normalizer
from interview_gpu.schemas.answer import ANSWER_DIMENSIONS, validate_scores


def test_regression_metrics() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(y_true, y_true + np.array([0.5, -0.5, 0.5, -0.5]))
    assert metrics["mae"] == pytest.approx(0.5) and metrics["rmse"] == pytest.approx(0.5)
    assert metrics["r2"] == pytest.approx(0.8)
    assert metrics["spearman"] == pytest.approx(2 / np.sqrt(5))  # predictions tie pairwise
    assert regression_metrics(y_true, y_true)["r2"] == pytest.approx(1.0)
    constant = regression_metrics(y_true, np.full(4, 2.5))
    assert constant["spearman"] is None  # undefined, not NaN
    assert regression_metrics(np.array([]), np.array([]))["mae"] is None


def test_multi_output_metrics_respect_mask() -> None:
    y_true = np.array([[1.0, 10.0], [2.0, 0.0], [3.0, 30.0], [4.0, 40.0]])
    y_pred = np.array([[1.0, 11.0], [2.0, 99.0], [3.0, 31.0], [4.0, 41.0]])
    mask = np.array([[1, 1], [1, 0], [1, 1], [1, 1]], dtype=bool)
    metrics = multi_output_regression_metrics(y_true, y_pred, mask, ["a", "b"])
    assert metrics["per_head"]["a"]["mae"] == 0.0
    assert metrics["per_head"]["b"]["mae"] == pytest.approx(1.0)  # masked 99 ignored
    assert metrics["per_head"]["b"]["count"] == 3
    assert metrics["mean"]["mae"] == pytest.approx(0.5)
    flat = flatten_metrics(metrics)
    assert flat["mean_mae"] == pytest.approx(0.5) and "per_head_b_rmse" in flat


def test_classification_metrics_on_imbalanced_data() -> None:
    y_true = np.array([0] * 8 + [1] * 2)
    probs = np.tile([0.9, 0.1], (10, 1))  # always predicts the majority class
    metrics = classification_metrics(y_true, probs)
    assert metrics["accuracy"] == pytest.approx(0.8)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)  # exposes the imbalance
    assert metrics["macro_f1"] < 0.5
    assert metrics["auroc"] == pytest.approx(0.5)
    assert 0.0 <= metrics["auprc"] <= 1.0

    perfect = classification_metrics(np.array([0, 1, 2, 0, 1, 2]), np.eye(3)[[0, 1, 2, 0, 1, 2]])
    assert perfect["accuracy"] == perfect["macro_f1"] == perfect["auroc"] == 1.0
    assert perfect["ece"] == pytest.approx(0.0) and perfect["brier"] == pytest.approx(0.0)

    missing_class = classification_metrics(np.array([0, 0, 1]), np.eye(3)[[0, 0, 1]])
    assert missing_class["auroc"] is None  # class 2 absent -> undefined


def test_calibration_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    overconfident = np.array([[0.99, 0.01]] * 4)
    assert expected_calibration_error(y_true, overconfident) == pytest.approx(0.49)
    assert brier_score(y_true, overconfident) > brier_score(y_true, np.array([[0.5, 0.5]] * 4))


def test_error_distribution_interval() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=5000)
    y_pred = y_true + rng.normal(scale=1.0, size=5000)
    stats = error_distribution(y_true, y_pred, coverage=0.9)
    assert stats["interval_low"] == pytest.approx(-1.645, abs=0.1)
    assert stats["interval_high"] == pytest.approx(1.645, abs=0.1)
    inside = (y_true - y_pred >= stats["interval_low"]) & (y_true - y_pred <= stats["interval_high"])
    assert inside.mean() == pytest.approx(0.9, abs=0.01)
    assert error_distribution(np.array([]), np.array([])) == {"count": 0}


def test_normalizer_uses_train_statistics_and_imputes() -> None:
    train = np.array([[1.0, np.nan, 5.0], [3.0, np.nan, 5.0]])
    normalizer = fit_normalizer(train)
    assert normalizer["mean"] == [2.0, 0.0, 5.0]
    assert normalizer["std"] == [1.0, 1.0, 1.0]  # all-missing and constant features -> std 1
    scaled = apply_normalizer(np.array([[4.0, np.nan, np.nan]]), normalizer)
    assert scaled.tolist() == [[2.0, 0.0, 0.0]] and scaled.dtype == np.float32
    sequences = fit_normalizer(np.zeros((2, 7, 3)))  # [N, T, F] also supported
    assert len(sequences["mean"]) == 3


def test_to_jsonable_handles_numpy_and_nan() -> None:
    payload = to_jsonable({"a": np.float32(1.5), "b": float("nan"), "c": np.arange(2), "d": (np.int64(3),)})
    assert payload == {"a": 1.5, "b": None, "c": [0, 1], "d": [3]}
    json.dumps(payload)


def test_answer_score_validation() -> None:
    assert validate_scores({"relevance": 3, "structure": "2.5"}) == {"relevance": 3.0, "structure": 2.5}
    for bad in ({"relevance": 5}, {"relevance": -1}, {"charisma": 5}, {"relevance": "high"}):
        with pytest.raises(InvalidLabelError):
            validate_scores(bad)


def test_answer_prompts() -> None:
    first, second = format_answer_input("질문?", "답변.", {"job": "Backend Engineer", "interview_type": "technical"})
    assert first == "query: [직무: Backend Engineer | 면접 유형: technical] 질문?"
    assert second == "passage: 답변."
    assert format_answer_input("Q", "A", None, style="plain") == ("Q", "A")

    prompt = build_labeling_prompt("질문?", "답변.")
    assert all(dimension in prompt for dimension in ANSWER_DIMENSIONS)
    response = 'Here you go:\n{"relevance": 3, "structure": 2}'
    assert parse_labeling_response(response) == {"relevance": 3.0, "structure": 2.0}
    with pytest.raises(InvalidLabelError):
        parse_labeling_response("no json here")
    with pytest.raises(InvalidLabelError):
        parse_labeling_response('{"relevance": 80}')
