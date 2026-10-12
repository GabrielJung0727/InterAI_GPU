"""Dataset 07 — ChaLearn First Impressions V2.

Expected layout::

    data/raw/chalearn-first-impressions/
        **/*.mp4          clips named "<youtube_id>.<clip>.mp4"

Used purely as a video+audio representation benchmark. The dataset's
annotations are apparent-personality traits and an "invite to interview"
score — both are outside what this project predicts, so the adapter exposes
NO labels. (The annotation files are pickles; they are not loaded.)
"""

from __future__ import annotations

from typing import Iterator

from interview_gpu.datasets.base import VIDEO_EXTENSIONS, BaseDatasetAdapter, MediaIndex
from interview_gpu.schemas.interview import DatasetLicense, InterviewSample


class ChaLearnFirstImpressionsAdapter(BaseDatasetAdapter):
    """Short talking-to-camera clips for multimodal representation research."""

    name = "chalearn"
    default_subdir = "chalearn-first-impressions"
    license = DatasetLicense(
        dataset_name="ChaLearn First Impressions V2",
        source="https://chalearnlap.cvc.uab.cat/dataset/24/description/",
        license="Research / academic use (ChaLearn LAP terms)",
        commercial_use=False,
        research_only=True,
        redistribution_allowed=False,
        usage_scope="research",
        notes="Research benchmark only. Personality / interview-invite annotations are not used.",
    )

    def _iter_samples(self) -> Iterator[InterviewSample]:
        for stem, video in MediaIndex(self.root, VIDEO_EXTENSIONS).items():
            # Clips cut from the same YouTube video share the prefix before the last dot.
            source_video = stem.rsplit(".", 1)[0] if "." in stem else stem
            yield self._make_sample(
                sample_id=stem,
                video_path=video.as_posix(),
                speaker_id=f"chalearn_{source_video}",
                metadata={"language": "en"},
            )
