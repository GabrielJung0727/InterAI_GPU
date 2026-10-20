"""Interview-answer evaluation schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from interview_gpu.common.errors import InvalidLabelError

#: Rubric dimensions; each gets its own regression head.
ANSWER_DIMENSIONS: tuple[str, ...] = (
    "relevance",
    "structure",
    "specificity",
    "evidence",
    "technical_depth",
    "ownership",
    "logical_consistency",
)

#: Ordinal rubric scale: 0 매우 부족 · 1 부족 · 2 보통 · 3 좋음 · 4 매우 좋음.
#: The service converts to a 100-point scale if it needs one.
SCORE_MIN, SCORE_MAX = 0.0, 4.0


class AnswerContext(BaseModel):
    """Optional conditioning context for an answer."""

    model_config = ConfigDict(extra="allow")

    job: str | None = None
    interview_type: str | None = None
    company: str | None = None
    #: Short, already-redacted excerpts only — never a raw resume.
    resume_context: str | None = None
    company_context: str | None = None


class AnswerTrainingExample(BaseModel):
    """One supervised example for the answer evaluator."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    group_id: str
    question: str
    answer: str
    context: AnswerContext = Field(default_factory=AnswerContext)
    #: Subset of ``ANSWER_DIMENSIONS`` -> score in [0, 4]. A dimension that does not
    #: apply to the question is absent. May be empty (unlabeled).
    scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_scores(scores: dict[str, Any], sample_id: str = "?") -> dict[str, float]:
    """Check rubric scores: known dimensions, numeric, within ``[0, 4]``.

    Raises:
        InvalidLabelError: On an unknown dimension or an out-of-range value.
    """
    clean: dict[str, float] = {}
    for name, value in scores.items():
        if name not in ANSWER_DIMENSIONS:
            raise InvalidLabelError(f"[{sample_id}] unknown score dimension '{name}'.")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidLabelError(f"[{sample_id}] score '{name}' is not numeric: {value!r}") from exc
        if not SCORE_MIN <= number <= SCORE_MAX:
            raise InvalidLabelError(
                f"[{sample_id}] score '{name}'={number} outside [{SCORE_MIN}, {SCORE_MAX}]."
            )
        clean[name] = number
    return clean
