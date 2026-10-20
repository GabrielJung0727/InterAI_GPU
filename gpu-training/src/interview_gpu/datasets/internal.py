"""Dataset 08 — the service's own data (most important long-term).

Expected layout::

    data/raw/internal/
        *.jsonl      one JSON object per line

Two record shapes are accepted.

Single answer::

    {"sample_id": "...", "candidate_id": "...", "question": "...", "answer": "...",
     "company": "...", "position": "...", "interview_type": "technical",
     "audio_path": "...", "video_path": "...",
     "scores": {"relevance": 3, ...},
     "speech_metrics": {...}, "video_metrics": {...}, "answer_metrics": {...},
     "consent": {"opt_in": true, "consent_version": "v1"}}

Before -> Feedback -> After pair::

    {"pair_id": "...", "candidate_id": "...", "question": "...",
     "before_answer": "...", "feedback": "...", "after_answer": "...",
     "before_scores": {...}, "after_scores": {...},
     "consent": {"opt_in": true}}

Privacy: records without ``consent.opt_in == true`` are always skipped. Raw
resumes are never read; only short context fields present in the record are
passed through.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import ValidationError

from interview_gpu.datasets.base import BaseDatasetAdapter
from interview_gpu.schemas.interview import (
    BeforeAfterSample,
    ConsentInfo,
    DatasetLicense,
    InterviewSample,
)

PAIR_FIELDS = (
    "pair_id", "question", "before_answer", "feedback", "after_answer", "before_scores",
    "after_scores", "company", "position", "interview_type", "candidate_id", "metadata",
)


def _numeric(scores: Any) -> dict[str, float]:
    if not isinstance(scores, dict):
        return {}
    return {str(k): float(v) for k, v in scores.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


class InternalServiceAdapter(BaseDatasetAdapter):
    """Opt-in service data, including Before/After improvement pairs."""

    name = "internal"
    default_subdir = "internal"
    license = DatasetLicense(
        dataset_name="Service internal interview data",
        source="First-party service data (explicit user opt-in required)",
        license="Proprietary",
        commercial_use=True,
        research_only=False,
        redistribution_allowed=False,
        usage_scope="production",
        notes="Only records with consent.opt_in == true are loaded.",
    )

    def _iter_records(self) -> Iterator[dict[str, Any]]:
        for path in sorted(self.root.rglob("*.jsonl")):
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        self.skipped["invalid_json"] += 1
                        continue
                    consent = record.get("consent") if isinstance(record, dict) else None
                    if not (isinstance(consent, dict) and consent.get("opt_in") is True):
                        self.skipped["no_consent"] += 1
                        continue
                    yield record

    def iter_pairs(self) -> Iterator[BeforeAfterSample]:
        """Yield consented Before -> Feedback -> After records."""
        self.ensure_exists()
        self.skipped.clear()
        for record in self._iter_records():
            if "before_answer" not in record or "after_answer" not in record:
                continue
            try:
                yield BeforeAfterSample(
                    **{key: record[key] for key in PAIR_FIELDS if record.get(key) is not None},
                    consent=ConsentInfo(**record["consent"]),
                )
            except ValidationError:
                self.skipped["invalid_pair"] += 1

    def _pair_samples(self, pair: BeforeAfterSample) -> Iterator[InterviewSample]:
        deltas = pair.deltas()
        for phase, answer, scores in (
            ("before", pair.before_answer, pair.before_scores),
            ("after", pair.after_answer, pair.after_scores),
        ):
            yield self._make_sample(
                sample_id=f"{pair.pair_id}_{phase}",
                question=pair.question,
                answer=answer,
                transcript=answer,
                company=pair.company,
                position=pair.position,
                # before/after of one pair must never be split across train/test.
                speaker_id=pair.candidate_id,
                session_id=pair.pair_id,
                labels=dict(scores),
                metadata={
                    "phase": phase,
                    "pair_id": pair.pair_id,
                    "feedback": pair.feedback,
                    "interview_type": pair.interview_type,
                    "deltas": deltas,
                },
                consent=pair.consent,
            )

    def _iter_samples(self) -> Iterator[InterviewSample]:
        for record in self._iter_records():
            consent = ConsentInfo(**record["consent"])
            if "before_answer" in record and "after_answer" in record:
                try:
                    pair = BeforeAfterSample(
                        **{key: record[key] for key in PAIR_FIELDS if record.get(key) is not None},
                        consent=consent,
                    )
                except ValidationError:
                    self.skipped["invalid_pair"] += 1
                    continue
                yield from self._pair_samples(pair)
                continue
            answer = record.get("answer")
            if not record.get("sample_id") or not isinstance(answer, str) or not answer.strip():
                self.skipped["missing_transcript"] += 1
                continue
            yield self._make_sample(
                sample_id=str(record["sample_id"]),
                question=record.get("question"),
                answer=answer,
                transcript=record.get("transcript") or answer,
                audio_path=record.get("audio_path"),
                video_path=record.get("video_path"),
                company=record.get("company"),
                position=record.get("position"),
                job_family=record.get("job_family"),
                speaker_id=str(record["candidate_id"]) if record.get("candidate_id") else None,
                session_id=str(record["session_id"]) if record.get("session_id") else None,
                labels=_numeric(record.get("scores")),
                metadata={
                    "interview_type": record.get("interview_type"),
                    "speech_metrics": record.get("speech_metrics"),
                    "video_metrics": record.get("video_metrics"),
                    "answer_metrics": record.get("answer_metrics"),
                    "before_score": record.get("before_score"),
                    "after_score": record.get("after_score"),
                },
                consent=consent,
            )


def load_before_after(root: str | Path | None = None) -> list[BeforeAfterSample]:
    """Convenience loader for all consented Before/After pairs."""
    return list(InternalServiceAdapter(root).iter_pairs())
