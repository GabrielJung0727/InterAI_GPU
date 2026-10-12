"""Text formatting for the answer evaluator and Stage-A labeling prompts.

Stage A (creating rubric labels with rules + an external LLM judge) is only
*structured* here: this module builds the prompt and validates the response.
It never calls an LLM API — that belongs to the service / data team.
"""

from __future__ import annotations

import json
import re
from typing import Any

from interview_gpu.common.errors import InvalidLabelError
from interview_gpu.schemas.answer import ANSWER_DIMENSIONS, AnswerContext, validate_scores

#: What each rubric dimension measures (0 = absent, 10 = excellent).
RUBRIC: dict[str, str] = {
    "relevance": "질문의 의도에 직접적으로 답하고 있는가",
    "structure": "답변이 논리적 순서(예: 상황-과제-행동-결과)로 구성되어 있는가",
    "specificity": "추상적 표현이 아닌 구체적인 상황·행동·수치를 제시하는가",
    "evidence": "주장을 뒷받침하는 근거, 사례, 결과가 있는가",
    "technical_depth": "직무 관련 지식과 기술적 판단의 깊이가 드러나는가",
    "ownership": "본인의 역할과 기여, 의사결정이 명확히 드러나는가",
    "logical_consistency": "답변 내 주장들이 서로 모순되지 않고 일관적인가",
}

_CONTEXT_LABELS = (
    ("job", "직무"),
    ("interview_type", "면접 유형"),
    ("company", "기업"),
    ("company_context", "기업 정보"),
    ("resume_context", "이력 요약"),
)


def build_context_text(context: AnswerContext | dict[str, Any] | None) -> str:
    """Render the available context fields as one compact line."""
    if context is None:
        return ""
    data = context.model_dump() if isinstance(context, AnswerContext) else context
    return " | ".join(f"{label}: {data[key]}" for key, label in _CONTEXT_LABELS if data.get(key))


def format_answer_input(
    question: str, answer: str, context: AnswerContext | dict[str, Any] | None = None, style: str = "e5"
) -> tuple[str, str]:
    """Build the encoder's sentence pair ``(context + question, answer)``.

    Args:
        style: ``"e5"`` adds the ``query:`` / ``passage:`` prefixes expected by
            E5 models; ``"plain"`` adds none (BGE-M3, generic encoders).
    """
    context_text = build_context_text(context)
    first = f"[{context_text}] {question}" if context_text else question
    if style == "e5":
        return f"query: {first}", f"passage: {answer}"
    return first, answer


def build_labeling_prompt(question: str, answer: str, context: AnswerContext | dict[str, Any] | None = None) -> str:
    """Prompt asking an external LLM judge for rubric scores as strict JSON."""
    rubric = "\n".join(f"- {name}: {description}" for name, description in RUBRIC.items())
    keys = ", ".join(f'"{name}": <0-10>' for name in ANSWER_DIMENSIONS)
    return (
        "당신은 채용 면접 답변을 평가하는 평가자입니다. 아래 기준 각각에 대해 0~10점 정수로 채점하세요.\n"
        "답변의 내용만 평가하고, 지원자의 성격·심리 상태·채용 적합성은 추정하지 마세요.\n\n"
        f"[평가 기준]\n{rubric}\n\n"
        f"[맥락]\n{build_context_text(context) or '없음'}\n\n"
        f"[질문]\n{question}\n\n[답변]\n{answer}\n\n"
        f"다음 JSON 형식으로만 응답하세요:\n{{{keys}}}"
    )


def parse_labeling_response(text: str, sample_id: str = "?") -> dict[str, float]:
    """Extract and validate rubric scores from a judge response.

    Raises:
        InvalidLabelError: No JSON object, unknown dimensions or out-of-range scores.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise InvalidLabelError(f"[{sample_id}] no JSON object in labeling response.")
    try:
        scores = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise InvalidLabelError(f"[{sample_id}] labeling response is not valid JSON.") from exc
    if not isinstance(scores, dict):
        raise InvalidLabelError(f"[{sample_id}] labeling response is not a JSON object.")
    return validate_scores(scores, sample_id)
