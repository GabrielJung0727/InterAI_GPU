"""Dataset 06 — MIT Interview Dataset (Naim et al.).

Expected layout::

    data/raw/mit-interview/
        Audio/P1.wav, PP1.wav, ...
        Videos/P1.avi, ...                         (optional)
        Labels/turker_scores_full_interview.csv    (optional)

Only *behavioural* rating columns are exposed as labels. Outcome-style columns
such as ``RecommendHiring`` or ``Overall`` are intentionally excluded: this
project does not train models that predict hiring suitability.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from interview_gpu.datasets.base import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    BaseDatasetAdapter,
    MediaIndex,
)
from interview_gpu.schemas.interview import DatasetLicense, InterviewSample

#: Observable delivery behaviours rated by annotators.
BEHAVIOR_LABELS = (
    "EyeContact",
    "SpeakingRate",
    "NoFillers",
    "Paused",
    "EngagingTone",
    "StructuredAnswers",
    "Focused",
)
SCORE_FILE = "turker_scores_full_interview.csv"


class MITInterviewAdapter(BaseDatasetAdapter):
    """Mock job interviews with crowd-rated delivery behaviours."""

    name = "mit_interview"
    default_subdir = "mit-interview"
    license = DatasetLicense(
        dataset_name="MIT Interview Dataset",
        source="https://roc-hci.com/past-projects/automated-prediction-of-job-interview-performances/",
        license="Research use (request from the authors)",
        commercial_use=False,
        research_only=True,
        redistribution_allowed=False,
        usage_scope="research",
        notes="Small (≈138 interviews, 69 participants); English. Benchmark / research only.",
    )

    def _load_scores(self) -> dict[str, dict[str, float]]:
        matches = list(self.root.rglob(SCORE_FILE))
        if not matches:
            return {}
        table = pd.read_csv(matches[0])
        if "Worker" in table.columns:  # keep the aggregated row per interview
            aggregated = table[table["Worker"].astype(str).str.upper() == "AGGR"]
            table = aggregated if len(aggregated) else table.groupby("Participant", as_index=False).mean(numeric_only=True)
        scores = {}
        for row in table.to_dict("records"):
            labels = {
                name: float(row[name]) for name in BEHAVIOR_LABELS if name in row and pd.notna(row[name])
            }
            scores[str(row["Participant"]).lower()] = labels
        return scores

    def _iter_samples(self) -> Iterator[InterviewSample]:
        scores = self._load_scores()
        video_index = MediaIndex(self.root, VIDEO_EXTENSIONS)
        media = dict(MediaIndex(self.root, AUDIO_EXTENSIONS).items())
        for stem in sorted(set(media) | {s for s, _ in video_index.items()}):
            audio, video = media.get(stem), video_index.find(stem)
            key = stem.lower()
            # "P12" (pre) and "PP12" (post) are the same participant.
            participant = key.lstrip("p")
            if key not in scores:
                self.skipped["missing_labels"] += 1
            yield self._make_sample(
                sample_id=stem,
                audio_path=audio.as_posix() if audio else None,
                video_path=video.as_posix() if video else None,
                speaker_id=f"mit_{participant}",
                labels=scores.get(key, {}),
                metadata={"session": "post" if key.startswith("pp") else "pre", "language": "en"},
            )
