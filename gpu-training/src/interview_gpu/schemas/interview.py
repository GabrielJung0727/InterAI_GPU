"""Dataset-independent contracts every adapter converts into."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

UsageScope = Literal["research", "production"]

#: Score dimensions tracked across the Before -> Feedback -> After loop.
DELTA_DIMENSIONS = ("relevance", "structure", "specificity", "evidence", "delivery")


class DatasetLicense(BaseModel):
    """License / provenance metadata attached to every dataset adapter."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    source: str
    license: str
    commercial_use: bool
    research_only: bool
    redistribution_allowed: bool
    #: ``research`` datasets are never mixed into production training.
    usage_scope: UsageScope
    notes: str = ""


class ConsentInfo(BaseModel):
    """Consent record for user-generated (service) data."""

    model_config = ConfigDict(extra="allow")

    opt_in: bool = False
    source: str | None = None
    consent_version: str | None = None
    consented_at: str | None = None


class InterviewSample(BaseModel):
    """One interview turn in the common schema.

    Adapters map raw dataset records to this structure; model-specific
    preprocessing only ever reads ``InterviewSample`` objects.
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    question: str | None = None
    answer: str | None = None
    audio_path: str | None = None
    video_path: str | None = None
    transcript: str | None = None
    company: str | None = None
    position: str | None = None
    job_family: str | None = None
    labels: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    #: Identity fields used for leakage-free group splits.
    speaker_id: str | None = None
    session_id: str | None = None
    source_dataset: str = "unknown"
    usage_scope: UsageScope = "research"
    consent: ConsentInfo | None = None

    @property
    def group_id(self) -> str:
        """Speaker > session > sample id: the key for group splitting."""
        return self.speaker_id or self.session_id or self.sample_id


class BeforeAfterSample(BaseModel):
    """A Before -> Feedback -> After record from the service's own data."""

    model_config = ConfigDict(extra="forbid")

    pair_id: str
    question: str
    before_answer: str
    feedback: str | None = None
    after_answer: str
    before_scores: dict[str, float] = Field(default_factory=dict)
    after_scores: dict[str, float] = Field(default_factory=dict)
    company: str | None = None
    position: str | None = None
    interview_type: str | None = None
    candidate_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    consent: ConsentInfo | None = None

    def deltas(self) -> dict[str, float | None]:
        """Per-dimension improvement and their mean (``overall_delta``).

        A dimension missing on either side yields ``None`` rather than a
        fabricated zero.
        """
        result: dict[str, float | None] = {}
        dimensions = list(DELTA_DIMENSIONS) + sorted(
            (set(self.before_scores) & set(self.after_scores)) - set(DELTA_DIMENSIONS)
        )
        for name in dimensions:
            before, after = self.before_scores.get(name), self.after_scores.get(name)
            result[f"delta_{name}"] = None if before is None or after is None else float(after - before)
        observed = [value for value in result.values() if value is not None]
        result["overall_delta"] = sum(observed) / len(observed) if observed else None
        return result
