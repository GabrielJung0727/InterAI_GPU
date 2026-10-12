"""Answer preprocessing: InterviewSample -> ``answers.jsonl`` (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from interview_gpu.answer.prompts import build_labeling_prompt
from interview_gpu.common.errors import EmptyDatasetError, InvalidLabelError
from interview_gpu.common.logging import get_logger
from interview_gpu.datasets.base import BaseDatasetAdapter
from interview_gpu.datasets.processed import write_manifest
from interview_gpu.schemas.answer import ANSWER_DIMENSIONS, AnswerContext, AnswerTrainingExample, validate_scores

ANSWERS_FILE = "answers.jsonl"


def load_external_labels(path: str | Path) -> dict[str, dict[str, float]]:
    """Read Stage-A labels: JSONL lines of ``{"sample_id": ..., "scores": {...}}``.

    Raises:
        InvalidLabelError: A line is malformed or holds invalid scores.
    """
    labels: dict[str, dict[str, float]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sample_id = str(record["sample_id"])
                scores = record["scores"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise InvalidLabelError(f"{path}:{number} must be {{'sample_id': ..., 'scores': {{...}}}}") from exc
            labels[sample_id] = validate_scores(scores, sample_id)
    return labels


def prepare_answer_dataset(
    adapter: BaseDatasetAdapter,
    output_dir: str | Path,
    *,
    labels_path: str | Path | None = None,
    prompts_path: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert a dataset into answer-evaluator examples.

    Scores come from the sample's own labels (service data) and/or from an
    external label file (Stage A). Samples without scores are still written,
    so they can be labeled later; training only uses labeled ones.

    Args:
        labels_path: Optional Stage-A label JSONL merged by ``sample_id``.
        prompts_path: If set, also export one labeling prompt per *unlabeled*
            example to this JSONL file (for an external LLM judge).

    Returns:
        The manifest written to ``<output_dir>/manifest.json``.
    """
    logger = get_logger()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    external = load_external_labels(labels_path) if labels_path else {}

    examples: list[AnswerTrainingExample] = []
    for sample in adapter.iter_samples(limit):
        answer = sample.answer or sample.transcript
        if not sample.question or not answer:
            adapter.skipped["missing_question_or_answer"] += 1
            continue
        raw_scores = {k: v for k, v in sample.labels.items() if k in ANSWER_DIMENSIONS}
        raw_scores.update(external.get(sample.sample_id, {}))
        try:
            scores = validate_scores(raw_scores, sample.sample_id)
        except InvalidLabelError as exc:
            adapter.skipped["invalid_labels"] += 1
            logger.warning("Dropping labels: %s", exc)
            scores = {}
        examples.append(
            AnswerTrainingExample(
                sample_id=sample.sample_id,
                group_id=sample.group_id,
                question=sample.question,
                answer=answer,
                # Resume text is never attached automatically (privacy).
                context=AnswerContext(
                    job=sample.position or sample.job_family,
                    interview_type=sample.metadata.get("interview_type"),
                    company=sample.company,
                ),
                scores=scores,
                metadata={k: sample.metadata[k] for k in ("phase", "pair_id") if k in sample.metadata},
            )
        )

    if not examples:
        raise EmptyDatasetError(
            f"No question/answer pairs found in {adapter.root.as_posix()} (skipped: {dict(adapter.skipped)})."
        )
    with (output_dir / ANSWERS_FILE).open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(example.model_dump_json() + "\n")

    labeled = sum(1 for example in examples if example.scores)
    if prompts_path:
        with Path(prompts_path).open("w", encoding="utf-8") as handle:
            for example in examples:
                if not example.scores:
                    prompt = build_labeling_prompt(example.question, example.answer, example.context)
                    handle.write(json.dumps({"sample_id": example.sample_id, "prompt": prompt}, ensure_ascii=False) + "\n")
        logger.info("Exported %d labeling prompts to %s", len(examples) - labeled, prompts_path)

    manifest = write_manifest(
        output_dir, adapter, num_samples=len(examples), num_labeled=labeled, dimensions=list(ANSWER_DIMENSIONS)
    )
    logger.info("Wrote %d examples (%d labeled) to %s", len(examples), labeled, output_dir / ANSWERS_FILE)
    if labeled == 0:
        logger.warning(
            "No example has rubric scores yet. The answer evaluator cannot be trained until labels "
            "are added (Stage A): use --export-prompts, score them, then re-run with --labels."
        )
    return manifest
