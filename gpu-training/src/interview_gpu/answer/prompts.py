"""Text formatting for the answer evaluator and Stage-A labeling prompts.

Stage A (creating rubric labels with an LLM judge) is only *structured* here:
this module builds the prompt and validates the response. It never calls an
LLM; the local judge lives in :mod:`interview_gpu.answer.labeling`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from interview_gpu.common.errors import InvalidLabelError
from interview_gpu.schemas.answer import ANSWER_DIMENSIONS, AnswerContext, validate_scores

RUBRIC_VERSION = "v2"

#: Rubric V2: what each dimension measures on the 0-4 scale, and what it must
#: *not* penalize. V1 scored every dimension on every question and demanded job
#: relevance, numbers and technical depth even where the question asked for none.
RUBRIC: dict[str, str] = {
    "relevance": (
        "질문에 직접 답했는가만 본다. 질문이 요구하지 않은 직무 연관성이 없다는 이유로 감점하지 않는다. "
        "(0 질문과 무관 · 2 일부만 답하거나 옆길로 샘 · 4 질문의 모든 부분에 직접 답함)"
    ),
    "structure": (
        "결론이 있는가, 이유·설명이 자연스럽게 이어지는가, 반복하거나 장황하게 새지 않는가. "
        "상황-과제-행동-결과 흐름은 경험을 묻는 질문에서만 기대한다. "
        "(0 두서없음 · 2 흐름은 있으나 반복·장황 · 4 결론과 전개가 정돈됨)"
    ),
    "specificity": (
        "추상적 표현 대신 질문에 맞는 구체적인 경험·행동·대상·과정·상황이 있는가. 수치가 없어도 높게 줄 수 있다. "
        "(0 추상적 표현뿐 · 2 일부만 구체적 · 4 충분히 구체적)"
    ),
    "evidence": (
        "주장을 뒷받침하는 근거가 있는가. 실제 경험, 사례, 이유, 과정, 결과, 사실, 수치를 모두 근거로 인정한다. "
        "(0 주장만 있음 · 2 근거가 약하거나 일부 주장에만 있음 · 4 주장마다 타당한 근거)"
    ),
    "technical_depth": (
        "기술·직무 전문지식·도메인 지식을 실제로 요구하는 질문에서만 평가한다. "
        "성격, 취미, 지원 동기, 갈등 해결, 장단점, 근무 조건 같은 질문에서는 평가 대상이 아니다. "
        "(0 전문 내용이 없거나 틀림 · 2 기본 개념 수준 · 4 정확하고 깊은 이해)"
    ),
    "ownership": (
        "프로젝트, 협업, 문제 해결, 성과, 갈등, 실패, 리더십처럼 본인이 무엇을 했는지 묻는 경험형 질문에서만 평가한다. "
        "의견·지식·계획을 묻는 질문에서는 평가 대상이 아니다. "
        "(0 본인 역할이 드러나지 않음 · 2 역할은 있으나 행동·판단이 흐림 · 4 본인의 행동과 판단이 분명함)"
    ),
    "logical_consistency": (
        "자기모순이 없는가, 원인과 결과가 이어지는가, 주장이 처음부터 끝까지 일관적인가만 본다. "
        "직무 연관성이나 기술적 깊이를 섞지 않는다. "
        "(0 모순되거나 논리가 무너짐 · 2 대체로 이어지나 비약이 있음 · 4 모순 없이 일관됨)"
    ),
}

#: Dimensions scored on every question; the rest may be "not applicable" (null).
ALWAYS_APPLICABLE: tuple[str, ...] = ("relevance", "structure", "specificity", "evidence", "logical_consistency")

QUESTION_TYPES: dict[str, str] = {
    "BEHAVIORAL_EXPERIENCE": "과거 경험·행동을 묻는 질문 (갈등, 실패, 협업, 리더십 등)",
    "TECHNICAL_KNOWLEDGE": "기술·직무 전문지식을 묻는 질문",
    "PROJECT_EXPERIENCE": "수행한 프로젝트·업무와 본인 역할을 묻는 질문",
    "SITUATIONAL": "가정된 상황에서 어떻게 할지 묻는 질문",
    "MOTIVATION": "지원 동기·직무 선택 이유를 묻는 질문",
    "CULTURE_FIT": "근무 조건·조직 문화·가치관 적합성을 묻는 질문",
    "PERSONALITY": "성격, 장단점, 취미, 일상을 묻는 질문",
    "OPINION_STRATEGY": "사회·산업 이슈에 대한 의견이나 전략을 묻는 질문",
    "GENERAL_KNOWLEDGE": "일반 상식·시사 지식을 묻는 질문",
    "CAREER_GOAL": "입사 후 포부·경력 목표를 묻는 질문",
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
    keys = ", ".join(f'"{name}": <0-4>' for name in ANSWER_DIMENSIONS)
    return (
        "당신은 채용 면접 답변을 평가하는 평가자입니다. 아래 기준 각각에 대해 0~4점 정수로 채점하세요.\n"
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
