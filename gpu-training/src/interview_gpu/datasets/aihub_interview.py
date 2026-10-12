"""Dataset 01 — AI Hub 채용면접 인터뷰 데이터 (core dataset).

Expected layout (any nesting depth is accepted)::

    data/raw/aihub-interview/
        **/*.json     one label file per question/answer turn
        **/*.wav      audio, matched to the label by file name / stem

IMPORTANT: the JSON key paths below follow the publicly documented structure
of the dataset but have NOT been verified against a real download in this
repository. After placing the data, run ``scripts/inspect_dataset.py`` and
adjust ``FIELD_PATHS`` if fields come back empty.
"""

from __future__ import annotations

from typing import Iterator

from interview_gpu.datasets.base import (
    AUDIO_EXTENSIONS,
    BaseDatasetAdapter,
    MediaIndex,
    dig,
    read_json_file,
)
from interview_gpu.schemas.interview import DatasetLicense, InterviewSample

#: Candidate dotted key paths per field, tried in order.
FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "question": ("dataSet.question.raw.text", "question.raw.text", "question.text", "question"),
    "answer": ("dataSet.answer.raw.text", "answer.raw.text", "answer.text", "answer"),
    "summary": ("dataSet.answer.summary.text", "answer.summary.text", "summary"),
    "question_intent": ("dataSet.question.intent", "question.intent"),
    "answer_intent": ("dataSet.answer.intent", "answer.intent"),
    "occupation": ("dataSet.info.occupation", "info.occupation", "occupation"),
    "experience": ("dataSet.info.experience", "info.experience"),
    "channel": ("dataSet.info.channel", "info.channel"),
    "place": ("dataSet.info.place", "info.place"),
    "gender": ("dataSet.info.gender", "info.gender"),
    "age_range": ("dataSet.info.ageRange", "info.ageRange"),
    "audio_file": (
        "rawDataInfo.answer.fileName",
        "dataSet.rawDataInfo.answer.fileName",
        "rawDataInfo.fileName",
        "audio.fileName",
    ),
    "speaker_id": ("dataSet.info.speakerId", "info.speakerId", "speaker_id", "speakerId"),
    "session_id": ("dataSet.info.sessionId", "info.sessionId", "session_id"),
}


def _first_category(value: object) -> str | None:
    """Reduce an intent annotation (str / dict / list of dicts) to one label."""
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, dict):
        value = value.get("category") or value.get("text") or value.get("expression")
    return str(value) if isinstance(value, (str, int)) and str(value) else None


class AIHubInterviewAdapter(BaseDatasetAdapter):
    """Korean interview questions, answers (STT text), intent, summary, job, audio."""

    name = "aihub_interview"
    default_subdir = "aihub-interview"
    license = DatasetLicense(
        dataset_name="AI Hub 채용면접 인터뷰 데이터",
        source="https://aihub.or.kr (채용면접 인터뷰 데이터)",
        license="AI Hub 이용약관 (승인 후 다운로드)",
        commercial_use=False,
        research_only=True,
        redistribution_allowed=False,
        usage_scope="research",
        notes=(
            "Conservative default. AI Hub terms differ per dataset and applicant; confirm the "
            "approved purpose before switching usage_scope to production. Never redistribute."
        ),
    )

    def _iter_samples(self) -> Iterator[InterviewSample]:
        audio_index = MediaIndex(self.root, AUDIO_EXTENSIONS)
        for label_path in sorted(self.root.rglob("*.json")):
            record = read_json_file(label_path)
            if not isinstance(record, dict):
                self.skipped["unreadable_json"] += 1
                continue
            values = {field: dig(record, *paths) for field, paths in FIELD_PATHS.items()}
            answer = values["answer"] if isinstance(values["answer"], str) else None
            if not answer or not answer.strip():
                self.skipped["missing_transcript"] += 1
                continue
            audio = audio_index.find(values["audio_file"]) or audio_index.find(label_path.name)
            if audio is None:
                self.skipped["missing_audio"] += 1  # text is still usable for the answer model

            labels = {}
            for key in ("answer_intent", "question_intent"):
                category = _first_category(values[key])
                if category:
                    labels[key] = category
            occupation = values["occupation"]
            yield self._make_sample(
                sample_id=label_path.stem,
                question=values["question"] if isinstance(values["question"], str) else None,
                answer=answer.strip(),
                transcript=answer.strip(),
                audio_path=audio.as_posix() if audio else None,
                job_family=str(occupation) if occupation else None,
                speaker_id=str(values["speaker_id"]) if values["speaker_id"] else None,
                session_id=str(values["session_id"]) if values["session_id"] else None,
                labels=labels,
                metadata={
                    "answer_summary": values["summary"] if isinstance(values["summary"], str) else None,
                    "experience": values["experience"],
                    "channel": values["channel"],
                    "place": values["place"],
                    "gender": values["gender"],
                    "age_range": values["age_range"],
                    "label_path": label_path.as_posix(),
                },
            )
