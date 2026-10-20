"""Stage A — rubric labels from a *local* open-weight LLM judge (no external API).

``answers.jsonl`` -> ``<output>.jsonl`` with one record per answer::

    {"sample_id": ..., "question_type": "BEHAVIORAL_EXPERIENCE",
     "applicability": {<7 dimensions>: true|false},
     "scores": {<7 dimensions>: 0-4 | null},
     "score_evidence": {<dimension>: ["quote"]}, "missing_points": [...],
     "judge_confidence": "HIGH|MEDIUM|LOW", "needs_review": false, "review_reason": null}

``null`` means "this dimension does not apply to the question" and is masked
out of the training loss; it is not a score of 0.

The output is append-only and doubles as the checkpoint: a restarted run skips
every ``sample_id`` already present. Samples are taken in a seeded random order,
so ``--limit 200`` is a prefix of ``--limit 2000`` which is a prefix of the full
run. The goal is consistent labels, not fluent text, so no feedback is generated.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from interview_gpu.answer.prompts import ALWAYS_APPLICABLE, QUESTION_TYPES, RUBRIC, RUBRIC_VERSION, build_context_text
from interview_gpu.common.errors import ConfigError, EmptyDatasetError, GpuOutOfMemoryError, InvalidLabelError, require
from interview_gpu.common.io import read_json, write_json
from interview_gpu.common.logging import get_logger
from interview_gpu.schemas.answer import ANSWER_DIMENSIONS, SCORE_MAX, validate_scores

#: ``<output>.failed_samples.jsonl`` holds answers whose response never parsed.
FAILED_SUFFIX = ".failed_samples.jsonl"
#: Sampling temperature for the extra consistency runs and for parse retries.
#: Greedy decoding would reproduce the first output exactly.
RESAMPLE_TEMPERATURE = 0.2
RETRY_TEMPERATURE = 0.3
MAX_MISSING_POINTS = 3
CONFIDENCE_LEVELS = ("LOW", "MEDIUM", "HIGH")  # ascending

SYSTEM_PROMPT = (
    "당신은 채용 면접 답변을 평가하는 평가자입니다. 답변의 내용만 평가하고, "
    "지원자의 성격·심리 상태·채용 적합성은 추정하지 마세요. 설명 없이 JSON 객체 하나만 출력하세요."
)


class LabelingConfig(BaseModel):
    """Settings of the local judge (``configs/labeling.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    #: ``vllm`` needs Linux / WSL2; on native Windows it falls back to ``transformers``.
    backend: Literal["transformers", "vllm"] = "transformers"
    #: HuggingFace model id. ``Qwen/Qwen3-14B`` is the better judge on 24 GB+ GPUs.
    model: str = "Qwen/Qwen3-8B"
    #: Transformers only. ``auto`` loads full precision when the weights fit in free VRAM, else 4-bit.
    quantization: Literal["auto", "none", "8bit", "4bit"] = "auto"
    #: vLLM only: model to serve instead of ``model``, e.g. a pre-quantized AWQ build.
    vllm_model: str | None = None
    vllm_gpu_memory_utilization: float = Field(0.90, gt=0.0, le=1.0)
    vllm_max_model_len: int = Field(4096, ge=1024)
    input: str = "data/processed/answer/answers.jsonl"
    output: str = "data/labeled/qwen3_answers.jsonl"
    batch_size: int | Literal["auto"] = "auto"
    max_new_tokens: int = Field(448, ge=32)
    temperature: float = Field(0.0, ge=0.0, le=0.2)
    #: Score every answer this many times; disagreement sets ``needs_review``.
    consistency_runs: int = Field(1, ge=1, le=5)
    #: A dimension whose runs differ by this many points (0-4 scale) or more needs review.
    review_threshold: float = Field(2.0, gt=0)
    max_retries: int = Field(2, ge=0)
    #: Longer answers are cut before judging to bound prompt length.
    max_answer_chars: int = Field(4000, ge=200)
    seed: int = 42
    dry_run_limit: int = Field(200, ge=1)


def load_labeling_config(path: str | Path, overrides: dict[str, Any] | None = None) -> LabelingConfig:
    """Read the YAML config and apply non-``None`` CLI overrides."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data.update({key: value for key, value in (overrides or {}).items() if value is not None})
    try:
        return LabelingConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid labeling configuration:\n{exc}") from exc


def build_judge_messages(
    question: str, answer: str, context: dict[str, Any] | None = None, max_answer_chars: int = 4000
) -> list[dict[str, str]]:
    """Chat messages asking the judge for question type, applicability and scores as strict JSON."""
    conditional = [name for name in ANSWER_DIMENSIONS if name not in ALWAYS_APPLICABLE]
    rubric = "\n".join(f"- {name}: {description}" for name, description in RUBRIC.items())
    types = "\n".join(f"- {name}: {description}" for name, description in QUESTION_TYPES.items())
    applicability = ", ".join(f'"{name}": <true|false>' for name in ANSWER_DIMENSIONS)
    scores = ", ".join(f'"{name}": <0-4 정수 또는 null>' for name in ANSWER_DIMENSIONS)
    user = (
        "다음 순서로 평가하세요.\n"
        "1. 질문을 아래 [질문 유형] 중 하나로 분류한다.\n"
        "2. 평가 항목마다 이 질문에서 평가 대상인지(applicability) 정한다. "
        f"{', '.join(conditional)}는 [평가 기준]의 조건에 맞을 때만 true이고, 나머지 항목은 항상 true다.\n"
        "3. 평가 대상인 항목만 0~4점 정수로 채점한다 (0 매우 부족, 1 부족, 2 보통, 3 좋음, 4 매우 좋음). "
        "평가 대상이 아닌 항목의 점수는 0이 아니라 null이다.\n\n"
        "[채점 원칙]\n"
        "- 각 항목을 독립적으로 판단한다. 전체 인상이 좋다고 모든 항목에 비슷한 점수를 주지 않는다. "
        "같은 답변에서 relevance 4, structure 2, evidence 1처럼 항목마다 점수가 크게 달라도 된다.\n"
        "- 답변은 음성을 받아쓴 글이다. 간투사(어, 음)나 받아쓰기 오류 자체는 감점하지 않는다.\n"
        "- missing_points에는 이 질문에 답하는 데 실제로 필요했던 개선점만 쓴다. 질문이 요구하지 않은 "
        "직무 연관성, 기술적 깊이, 수치, 개인 기여도, 성과를 기계적으로 요구하지 않는다. 없으면 빈 배열로 둔다.\n"
        "- judge_confidence: HIGH = 답변이 온전하고 기준 적용이 분명함, MEDIUM = 일부 애매함(받아쓰기 오류, 맥락 부족), "
        "LOW = 질문과 답변이 맞지 않거나, 답변이 잘렸거나, 뜻을 알 수 없거나, 기준 적용이 어려움. "
        "LOW일 때만 review_reason에 사유를 쓰고 그 외에는 null로 둔다.\n\n"
        f"[질문 유형]\n{types}\n\n"
        f"[평가 기준]\n{rubric}\n\n"
        f"[맥락]\n{build_context_text(context) or '없음'}\n\n"
        f"[질문]\n{question}\n\n[답변]\n{answer[:max_answer_chars]}\n\n"
        "다음 JSON 형식으로만, 줄바꿈과 들여쓰기 없이 한 줄로 응답하세요:\n"
        f'{{"question_type": "<유형>", "applicability": {{{applicability}}}, '
        '"score_evidence": {<평가 대상 항목 이름>: [<그 점수의 근거가 된 답변 속 구절을 25자 이내로 그대로 인용, 항목당 1개>]}, '
        f'"scores": {{{scores}}}, '
        '"missing_points": [<짧은 명사구, 최대 3개>], '
        '"judge_confidence": "<HIGH|MEDIUM|LOW>", "review_reason": <사유 문자열 또는 null>}'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def prompt_fingerprint() -> str:
    """Hash of the judge prompt; labels made with different rubrics must not be mixed."""
    messages = build_judge_messages("{question}", "{answer}", None)
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _quotes(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in items if item is not None and str(item).strip()][:2]


def parse_judge_response(text: str, sample_id: str = "?") -> dict[str, Any]:
    """Extract and validate one judge response.

    Raises:
        InvalidLabelError: No JSON object, an unknown question type, a missing
            or non-integer score for an applicable dimension, an out-of-range
            score or a malformed field.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise InvalidLabelError(f"[{sample_id}] no JSON object in judge response.")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise InvalidLabelError(f"[{sample_id}] judge response is not valid JSON.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("scores"), dict):
        raise InvalidLabelError(f"[{sample_id}] judge response has no 'scores' object.")

    question_type = str(payload.get("question_type", "")).strip().upper()
    if question_type not in QUESTION_TYPES:
        raise InvalidLabelError(f"[{sample_id}] unknown question_type '{question_type}'.")
    confidence = str(payload.get("judge_confidence", "")).strip().upper()
    if confidence not in CONFIDENCE_LEVELS:
        raise InvalidLabelError(f"[{sample_id}] judge_confidence must be one of {CONFIDENCE_LEVELS}.")

    raw_scores = payload["scores"]
    declared = payload.get("applicability") if isinstance(payload.get("applicability"), dict) else {}
    numeric = validate_scores({name: value for name, value in raw_scores.items() if value is not None}, sample_id)
    applicability: dict[str, bool] = {}
    scores: dict[str, int | None] = {}
    for name in ANSWER_DIMENSIONS:
        applicable = name in ALWAYS_APPLICABLE or bool(declared.get(name, name in numeric))
        applicability[name] = applicable
        if not applicable:
            scores[name] = None  # "not applicable" is never a score of 0
            continue
        if name not in numeric:
            raise InvalidLabelError(f"[{sample_id}] applicable dimension '{name}' has no score.")
        if numeric[name] != int(numeric[name]):
            raise InvalidLabelError(f"[{sample_id}] score '{name}'={numeric[name]} is not an integer.")
        scores[name] = int(numeric[name])

    evidence = payload.get("score_evidence") if isinstance(payload.get("score_evidence"), dict) else {}
    missing = payload.get("missing_points", [])
    if not isinstance(missing, list):
        raise InvalidLabelError(f"[{sample_id}] 'missing_points' must be a list.")
    reason = payload.get("review_reason")
    return {
        "question_type": question_type,
        "applicability": applicability,
        "scores": scores,
        "score_evidence": {
            name: _quotes(evidence[name]) for name in ANSWER_DIMENSIONS
            if applicability[name] and name in evidence and _quotes(evidence[name])
        },
        "missing_points": [str(item).strip() for item in missing if str(item).strip()][:MAX_MISSING_POINTS],
        "judge_confidence": confidence,
        "review_reason": str(reason).strip() if isinstance(reason, str) and reason.strip() else None,
    }


def merge_runs(sample_id: str, runs: list[dict[str, Any]], review_threshold: float, retries: int) -> dict[str, Any]:
    """Combine the consistency runs of one sample into its output record.

    The label is the first (greedy) run, so scores stay integers; the other
    runs only test its stability. ``needs_review`` is set when any run reports
    LOW confidence, when runs disagree on whether a dimension applies, or when
    a dimension differs by ``review_threshold`` or more between runs.
    """
    first = runs[0]
    confidence = min((run["judge_confidence"] for run in runs), key=CONFIDENCE_LEVELS.index)
    reasons: list[str] = []
    if confidence == "LOW":
        stated = next((run["review_reason"] for run in runs if run["review_reason"]), None)
        reasons.append(f"judge_confidence LOW: {stated or '사유 없음'}")
    for name in ANSWER_DIMENSIONS:
        values = [run["scores"][name] for run in runs]
        scored = [value for value in values if value is not None]
        if scored and len(scored) < len(values):
            reasons.append(f"applicability disagreement: {name}")
        elif scored and max(scored) - min(scored) >= review_threshold:
            reasons.append(f"score disagreement: {name} ({min(scored)}-{max(scored)})")
    record = {
        "sample_id": sample_id,
        "question_type": first["question_type"],
        "applicability": first["applicability"],
        "scores": first["scores"],
        "score_evidence": first["score_evidence"],
        "missing_points": first["missing_points"],
        "judge_confidence": confidence,
        "needs_review": bool(reasons),
        "review_reason": "; ".join(reasons) or None,
        "parse_retries": retries,
    }
    if len(runs) > 1:
        record["run_scores"] = [run["scores"] for run in runs]
    return record


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file, ignoring blank lines and a torn final line."""
    path = Path(path)
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # interrupted mid-write; the sample is simply labeled again
    return records


def select_examples(examples: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    """Seeded shuffle, then the first ``limit``: smaller limits are prefixes of larger ones."""
    order = sorted(examples, key=lambda example: example["sample_id"])
    random.Random(seed).shuffle(order)
    return order if limit is None else order[:limit]


class LabelerBackend(Protocol):
    """Anything that turns chat prompts into raw text responses."""

    #: Samples scored and checkpointed together.
    batch_size: int
    #: Recorded in ``<output>.meta.json``; a different value refuses to resume.
    description: str

    def generate(self, messages: list[list[dict[str, str]]], temperature: float, seed: int) -> list[str]: ...


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Apply the chat template with thinking off and prefill the opening brace."""
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:  # template without a thinking switch
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + "{"  # the response must start a JSON object


class TransformersBackend:
    """Batched local generation with HuggingFace Transformers (works on Windows and Linux)."""

    def __init__(self, config: LabelingConfig, device: str = "auto") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from interview_gpu.common.device import select_device

        self._torch = torch
        self._logger = get_logger()
        self.device = select_device(device)
        self.max_new_tokens = config.max_new_tokens
        quantization = self._resolve_quantization(config)
        self.description = f"transformers:{config.model}:{quantization}"

        kwargs: dict[str, Any] = {"dtype": torch.bfloat16 if self.device.type == "cuda" else torch.float32}
        if quantization != "none":
            require("bitsandbytes", "uv sync --extra labeling")
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = (
                BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                if quantization == "4bit"
                else BitsAndBytesConfig(load_in_8bit=True)
            )
            kwargs["device_map"] = {"": self.device.index or 0}
        self._logger.info("Loading judge %s (quantization=%s) on %s", config.model, quantization, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
        if quantization == "none":
            self.model.to(self.device)
        self.model.eval()
        self.batch_size = self._resolve_batch_size(config.batch_size)
        self._logger.info("Judge ready. batch_size=%d", self.batch_size)

    def _free_vram_gb(self) -> float:
        free, _ = self._torch.cuda.mem_get_info(self.device)
        return free / 2**30

    def _resolve_quantization(self, config: LabelingConfig) -> str:
        if config.quantization != "auto":
            return config.quantization
        if self.device.type != "cuda":
            return "none"
        try:
            from huggingface_hub import model_info

            files = model_info(config.model, files_metadata=True).siblings
            weights_gb = sum(f.size or 0 for f in files if f.rfilename.endswith(".safetensors")) / 2**30
        except Exception as exc:  # offline or unknown repo layout
            self._logger.warning("Could not read the size of %s (%s); using 4-bit.", config.model, exc)
            return "4bit"
        # Leave room for the KV cache and activations of a batch.
        return "none" if weights_gb + 4.0 <= self._free_vram_gb() else "4bit"

    def _resolve_batch_size(self, requested: int | str) -> int:
        if requested != "auto":
            return int(requested)
        if self.device.type != "cuda":
            return 1
        # ~0.6 GB per sequence covers the KV cache of a ~2k-token prompt on an 8-14B model.
        return int(min(32, max(1, self._free_vram_gb() // 0.6)))

    def generate(self, messages: list[list[dict[str, str]]], temperature: float, seed: int) -> list[str]:
        torch = self._torch
        prompts = [render_prompt(self.tokenizer, item) for item in messages]
        outputs: list[str] = []
        start = 0
        while start < len(prompts):
            chunk = prompts[start : start + self.batch_size]
            encoded = self.tokenizer(chunk, return_tensors="pt", padding=True).to(self.model.device)
            sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
            torch.manual_seed(seed + start)
            try:
                with torch.inference_mode():
                    generated = self.model.generate(
                        **encoded, max_new_tokens=self.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id, top_p=None, top_k=None, **sampling,
                    )
            except torch.OutOfMemoryError as exc:
                del encoded
                torch.cuda.empty_cache()
                if self.batch_size == 1:
                    raise GpuOutOfMemoryError(
                        "The judge ran out of GPU memory at batch size 1. Use --quantization 4bit, "
                        "a smaller --model or a lower max_answer_chars."
                    ) from exc
                self.batch_size = max(1, self.batch_size // 2)
                self._logger.warning("GPU out of memory; batch size reduced to %d.", self.batch_size)
                continue
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            outputs.extend("{" + text for text in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
            start += len(chunk)
        return outputs


class VLLMBackend:
    """Offline batched generation with vLLM (Linux / WSL2 only) for the full labeling run.

    vLLM schedules the whole batch itself (continuous batching), so the batch
    size here is only the checkpoint interval.
    """

    def __init__(self, config: LabelingConfig) -> None:
        vllm = require("vllm", "pip install vllm  (Linux or WSL2; not available on native Windows)")
        model = config.vllm_model or config.model
        self._sampling_params = vllm.SamplingParams
        self.max_new_tokens = config.max_new_tokens
        self.description = f"vllm:{model}"
        get_logger().info("Loading judge %s with vLLM", model)
        self.llm = vllm.LLM(
            model=model, seed=config.seed, max_model_len=config.vllm_max_model_len,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.batch_size = 256 if config.batch_size == "auto" else int(config.batch_size)

    def generate(self, messages: list[list[dict[str, str]]], temperature: float, seed: int) -> list[str]:
        prompts = [render_prompt(self.tokenizer, item) for item in messages]
        params = self._sampling_params(temperature=temperature, max_tokens=self.max_new_tokens, seed=seed)
        return ["{" + output.outputs[0].text for output in self.llm.generate(prompts, params, use_tqdm=False)]


def create_backend(config: LabelingConfig, device: str = "auto") -> LabelerBackend:
    """Build the configured backend; vLLM is never forced on native Windows."""
    if config.backend == "vllm":
        if sys.platform != "win32":
            return VLLMBackend(config)
        get_logger().warning("vLLM does not run on native Windows; using the Transformers backend instead.")
    return TransformersBackend(config, device)


def _judge_run(
    backend: LabelerBackend, batch: list[dict[str, Any]], config: LabelingConfig, temperature: float, seed: int
) -> tuple[dict[str, dict[str, Any]], dict[str, int], dict[str, str]]:
    """One scoring pass over ``batch`` with parse retries.

    Returns ``(parsed by sample_id, retries used by sample_id, last raw text of failures)``.
    """
    parsed: dict[str, dict[str, Any]] = {}
    retries = {example["sample_id"]: 0 for example in batch}
    raw: dict[str, str] = {}
    pending = batch
    for attempt in range(config.max_retries + 1):
        if not pending:
            break
        messages = [
            build_judge_messages(e["question"], e["answer"], e.get("context"), config.max_answer_chars) for e in pending
        ]
        texts = backend.generate(
            messages, temperature if attempt == 0 else max(temperature, RETRY_TEMPERATURE), seed + 1000 * attempt
        )
        failed = []
        for example, text in zip(pending, texts, strict=True):
            sample_id = example["sample_id"]
            try:
                parsed[sample_id] = parse_judge_response(text, sample_id)
                retries[sample_id] = attempt
            except InvalidLabelError:
                raw[sample_id] = text
                failed.append(example)
        pending = failed
    return parsed, retries, {example["sample_id"]: raw[example["sample_id"]] for example in pending}


def _check_meta(output: Path, meta: dict[str, Any]) -> None:
    """Refuse to append to labels made with a different model, rubric or protocol."""
    meta_path = output.with_suffix(".meta.json")
    if output.is_file() and meta_path.is_file():
        previous = read_json(meta_path)
        changed = [key for key in meta if previous.get(key) != meta[key]]
        if changed:
            raise ConfigError(
                f"{output} was labeled with different settings ({', '.join(changed)}): "
                f"{ {key: previous.get(key) for key in changed} } vs { {key: meta[key] for key in changed} }.\n"
                "Mixing them would corrupt the labels. Use another --output, or move the old file away."
            )
    write_json(meta_path, meta)


def run_labeling(
    config: LabelingConfig,
    backend: LabelerBackend,
    *,
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None,
) -> dict[str, Any]:
    """Label the selected answers, resuming from ``output_path``.

    Args:
        limit: Number of answers to label in total (``None`` = all of them).

    Returns:
        The statistics written to ``<output>.stats.json``.
    """
    logger = get_logger()
    input_path, output_path = Path(input_path), Path(output_path)
    examples = read_jsonl(input_path)
    if not examples:
        raise EmptyDatasetError(f"No answers found in {input_path}. Run scripts/prepare_answer.py first.")
    selected = select_examples(examples, limit, config.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _check_meta(
        output_path,
        {
            "backend": backend.description, "rubric": RUBRIC_VERSION, "prompt": prompt_fingerprint(),
            "consistency_runs": config.consistency_runs, "review_threshold": config.review_threshold,
            "temperature": config.temperature, "seed": config.seed,
        },
    )
    failed_path = output_path.with_suffix(FAILED_SUFFIX)
    done = {record["sample_id"] for record in read_jsonl(output_path)}
    pending = [example for example in selected if example["sample_id"] not in done]
    # Longest first: an out-of-memory batch shows up immediately, not hours in.
    pending.sort(key=lambda example: len(example["answer"]) + len(example["question"]), reverse=True)
    logger.info("%d selected, %d already labeled, %d to do.", len(selected), len(selected) - len(pending), len(pending))

    batch_size = backend.batch_size
    started = time.time()
    labeled = 0
    with output_path.open("a", encoding="utf-8") as out, failed_path.open("a", encoding="utf-8") as failed_out:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            runs: dict[str, list[dict[str, Any]]] = {example["sample_id"]: [] for example in batch}
            retries = dict.fromkeys(runs, 0)
            failures: dict[str, str] = {}
            for run in range(config.consistency_runs):
                alive = [example for example in batch if example["sample_id"] not in failures]
                temperature = config.temperature if run == 0 else max(config.temperature, RESAMPLE_TEMPERATURE)
                parsed, used, failed = _judge_run(backend, alive, config, temperature, config.seed + start + 7919 * run)
                failures.update(failed)
                for sample_id, result in parsed.items():
                    runs[sample_id].append(result)
                    retries[sample_id] += used[sample_id]
            for example in batch:
                sample_id = example["sample_id"]
                if sample_id in failures:
                    failed_out.write(json.dumps({"sample_id": sample_id, "response": failures[sample_id]}, ensure_ascii=False) + "\n")
                else:
                    record = merge_runs(sample_id, runs[sample_id], config.review_threshold, retries[sample_id])
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    labeled += 1
            out.flush()
            failed_out.flush()
            finished = min(start + batch_size, len(pending))
            rate = finished / max(time.time() - started, 1e-6)
            logger.info(
                "%d/%d (%.2f samples/s, ~%.0f min left, %d failed)",
                finished, len(pending), rate, (len(pending) - finished) / rate / 60, finished - labeled,
            )

    records = read_jsonl(output_path)
    labeled_ids = {record["sample_id"] for record in records}
    # A sample that failed earlier and succeeded on resume is no longer a failure.
    still_failed = {r["sample_id"]: r for r in read_jsonl(failed_path) if r["sample_id"] not in labeled_ids}
    with failed_path.open("w", encoding="utf-8") as handle:
        for record in still_failed.values():
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = compute_label_stats(records, len(still_failed))
    stats["samples_per_sec"] = labeled / max(time.time() - started, 1e-6) if labeled else None
    write_json(output_path.with_suffix(".stats.json"), stats)
    export_review_table(records, examples, output_path.with_suffix(".review.csv"))
    return stats


def score_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Scores as a float table indexed by ``sample_id``; not-applicable dimensions are NaN."""
    frame = pd.DataFrame(
        [{name: record["scores"].get(name) for name in ANSWER_DIMENSIONS} for record in records],
        index=[record["sample_id"] for record in records], columns=list(ANSWER_DIMENSIONS),
    )
    return frame.astype(float)


def _halo(frame: pd.DataFrame, method: str) -> float | None:
    """Mean off-diagonal correlation: how much the dimensions move as one overall impression."""
    corr = frame.corr(method=method).to_numpy()
    values = corr[~np.eye(len(corr), dtype=bool)]
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def compute_label_stats(records: list[dict[str, Any]], num_failed: int) -> dict[str, Any]:
    """Score distributions, applicability, question types, correlations, review and failure rates."""
    total = len(records) + num_failed
    stats: dict[str, Any] = {
        "num_labeled": len(records),
        "num_failed": num_failed,
        "parse_failure_rate": num_failed / total if total else 0.0,
        #: Share of samples whose first response could not be parsed (retried or failed).
        "first_attempt_failure_rate": (
            (num_failed + sum(1 for r in records if r.get("parse_retries", 0) > 0)) / total if total else 0.0
        ),
    }
    if not records:
        return stats
    scores = score_frame(records)
    confidence = Counter(record.get("judge_confidence") for record in records)
    stats.update(
        needs_review_rate=float(np.mean([bool(record.get("needs_review")) for record in records])),
        judge_confidence={level: confidence.get(level, 0) for level in reversed(CONFIDENCE_LEVELS)},
        question_type=dict(Counter(record.get("question_type") for record in records).most_common()),
        dimensions={
            name: {
                "applicable_rate": float(scores[name].notna().mean()),
                "null_rate": float(scores[name].isna().mean()),
                "mean": float(scores[name].mean()),
                "median": float(scores[name].median()),
                "std": float(scores[name].std(ddof=0)),
                "distribution": {
                    str(score): int(count)
                    for score, count in scores[name].dropna().round().astype(int).value_counts().sort_index().items()
                },
            }
            for name in ANSWER_DIMENSIONS
        },
        # Pairwise over samples where both dimensions apply. NaN becomes null in the JSON report.
        halo={"pearson_mean_offdiag": _halo(scores, "pearson"), "spearman_mean_offdiag": _halo(scores, "spearman")},
        correlation_pearson=scores.corr(method="pearson").to_dict(),
        correlation_spearman=scores.corr(method="spearman").to_dict(),
    )
    return stats


def compare_label_sets(
    old: list[dict[str, Any]], new: list[dict[str, Any]], old_scale: float, new_scale: float = SCORE_MAX
) -> dict[str, Any]:
    """Compare two labelings of the same answers (e.g. rubric V1 vs V2).

    Scores are divided by their scale maximum so a 0-10 and a 0-4 rubric can be
    compared; ``change`` is new minus old on that 0-1 scale.
    """
    old_frame, new_frame = score_frame(old) / old_scale, score_frame(new) / new_scale
    common = old_frame.index.intersection(new_frame.index)
    old_frame, new_frame = old_frame.loc[common], new_frame.loc[common]
    dimensions = {}
    for name in ANSWER_DIMENSIONS:
        both = old_frame[name].notna() & new_frame[name].notna()
        pair = pd.DataFrame({"old": old_frame.loc[both, name], "new": new_frame.loc[both, name]})
        dimensions[name] = {
            "old_mean": float(old_frame[name].mean()),
            "new_mean": float(new_frame[name].mean()),
            "change": float((pair["new"] - pair["old"]).mean()) if len(pair) else None,
            "old_std": float(old_frame[name].std(ddof=0)),
            "new_std": float(new_frame[name].std(ddof=0)),
            "old_null_rate": float(old_frame[name].isna().mean()),
            "new_null_rate": float(new_frame[name].isna().mean()),
            #: Do the two rubrics rank the answers the same way on this dimension?
            "rank_agreement_spearman": float(pair["old"].corr(pair["new"], method="spearman")) if len(pair) > 2 else None,
        }
    return {
        "num_common": len(common),
        "dimensions": dimensions,
        "halo": {
            side: {"pearson_mean_offdiag": _halo(frame, "pearson"), "spearman_mean_offdiag": _halo(frame, "spearman")}
            for side, frame in (("old", old_frame), ("new", new_frame))
        },
        "correlation_pearson": {"old": old_frame.corr().to_dict(), "new": new_frame.corr().to_dict()},
        "correlation_spearman": {
            "old": old_frame.corr(method="spearman").to_dict(), "new": new_frame.corr(method="spearman").to_dict(),
        },
    }


def export_review_table(records: list[dict[str, Any]], examples: list[dict[str, Any]], path: str | Path) -> None:
    """Write question, answer, scores and evidence side by side for human review."""
    by_id = {example["sample_id"]: example for example in examples}
    rows = []
    for record in records:
        example = by_id.get(record["sample_id"], {})
        rows.append(
            {
                "sample_id": record["sample_id"],
                "job": (example.get("context") or {}).get("job"),
                "question_type": record["question_type"],
                "question": example.get("question"),
                "answer": example.get("answer"),
                **record["scores"],  # not applicable -> empty cell
                "judge_confidence": record["judge_confidence"],
                "needs_review": record["needs_review"],
                "review_reason": record["review_reason"],
                "missing_points": " | ".join(record["missing_points"]),
                "score_evidence": " | ".join(
                    f"{name}: {' / '.join(quotes)}" for name, quotes in record["score_evidence"].items()
                ),
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig: Excel only detects UTF-8 (Korean text) with a BOM.
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
