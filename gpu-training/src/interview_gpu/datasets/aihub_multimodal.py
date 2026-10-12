"""Datasets 03–05 — AI Hub multimodal / interaction / face video datasets.

* 03 멀티모달 영상                       video + audio + text representation
* 04 안내 로봇 인터랙션 멀티모달          face, gaze, speech, interaction behaviour
* 05 한국인 감정인식을 위한 복합 영상     Korean face representation research

Their annotation schemas differ and are not parsed here: each video becomes a
sample and the path of a same-named JSON annotation (if any) is recorded in
``metadata["annotation_path"]`` for dataset-specific follow-up work. Emotion
annotations are never turned into behaviour-model targets.
"""

from __future__ import annotations

from typing import Iterator

from interview_gpu.datasets.base import (
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    BaseDatasetAdapter,
    MediaIndex,
)
from interview_gpu.schemas.interview import DatasetLicense, InterviewSample


def _aihub_license(dataset_name: str, notes: str) -> DatasetLicense:
    return DatasetLicense(
        dataset_name=dataset_name,
        source=f"https://aihub.or.kr ({dataset_name})",
        license="AI Hub 이용약관 (승인 후 다운로드)",
        commercial_use=False,
        research_only=True,
        redistribution_allowed=False,
        usage_scope="research",
        notes=notes,
    )


class _VideoFolderAdapter(BaseDatasetAdapter):
    """Generic adapter: one sample per video file found under the root."""

    def _iter_samples(self) -> Iterator[InterviewSample]:
        audio_index = MediaIndex(self.root, AUDIO_EXTENSIONS)
        annotations = {path.stem: path for path in self.root.rglob("*.json")}
        for stem, video in MediaIndex(self.root, VIDEO_EXTENSIONS).items():
            audio = audio_index.find(stem)
            annotation = annotations.get(stem)
            folder = video.parent.relative_to(self.root).as_posix()
            yield self._make_sample(
                sample_id=stem,
                video_path=video.as_posix(),
                audio_path=audio.as_posix() if audio else None,
                # Without verified speaker metadata, group by parent folder
                # (recordings of one person/session usually share a folder).
                session_id=None if folder == "." else folder,
                metadata={"annotation_path": annotation.as_posix() if annotation else None},
            )


class AIHubMultimodalVideoAdapter(_VideoFolderAdapter):
    name = "aihub_multimodal"
    default_subdir = "aihub-multimodal-video"
    license = _aihub_license("AI Hub 멀티모달 영상", "Video/audio/text multimodal representation research.")


class AIHubRobotInteractionAdapter(_VideoFolderAdapter):
    name = "aihub_robot_interaction"
    default_subdir = "aihub-robot-interaction"
    license = _aihub_license(
        "AI Hub 안내 로봇 인터랙션 멀티모달", "Face / gaze / speech interaction behaviour research."
    )


class AIHubKoreanFaceVideoAdapter(_VideoFolderAdapter):
    name = "aihub_korean_face"
    default_subdir = "aihub-korean-emotion-video"
    license = _aihub_license(
        "AI Hub 한국인 감정인식을 위한 복합 영상",
        "Korean face representation research only; emotion labels are not used as targets.",
    )
