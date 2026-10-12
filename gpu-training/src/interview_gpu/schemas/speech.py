"""Speech feature and prediction schemas.

Only *observable* properties of speech are modelled. Nothing here represents
an inferred psychological state (confidence, anxiety, ...).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

#: Handcrafted features, in the order they are fed to the model (Stage 1).
SPEECH_FEATURE_NAMES: tuple[str, ...] = (
    "utterance_duration",
    "response_latency",
    "pause_count",
    "pause_total_duration",
    "pause_mean_duration",
    "pause_max_duration",
    "long_pause_count",
    "pause_rate_per_min",
    "speech_ratio",
    "pitch_mean",
    "pitch_std",
    "pitch_cv",
    "energy_mean_db",
    "energy_std_db",
    "voice_stability_proxy",
    # The following need a transcript and are NaN without one.
    "speech_rate_syllables_per_sec",
    "articulation_rate_syllables_per_sec",
    "speech_rate_wpm",
    "filler_count",
    "filler_per_min",
)

#: Example behaviour scores a trained speech model may output.
SPEECH_TARGET_EXAMPLES: tuple[str, ...] = (
    "pace_score",
    "pause_score",
    "delivery_stability",
    "speech_clarity_proxy",
)


class SpeechPrediction(BaseModel):
    """Result of ``SpeechAnalyzer.predict`` (JSON-serializable)."""

    #: Raw measurements (always present): the evidence behind any score.
    metrics: dict[str, float | None]
    #: Model scores in label units; empty when no checkpoint is loaded.
    scores: dict[str, float] = Field(default_factory=dict)
    model_version: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
