"""Dataset 02 — AI Hub 감정 분류를 위한 대화 음성 데이터셋.

Used for Korean speech *representation* work (prosody, pitch, energy), not for
emotion judgement: emotion annotations are kept in ``metadata`` only and are
deliberately not exposed as training labels.

Expected layout::

    data/raw/aihub-emotion-speech/
        **/*.wav
        **/*.csv      optional; rows keyed by a wav id column with the utterance text

The CSV column names are matched heuristically (unverified against a real
download) — check with ``scripts/inspect_dataset.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from interview_gpu.datasets.base import AUDIO_EXTENSIONS, BaseDatasetAdapter, MediaIndex
from interview_gpu.schemas.interview import DatasetLicense, InterviewSample

ID_COLUMNS = ("wav_id", "wavid", "file_name", "filename", "id")
TEXT_COLUMNS = ("발화문", "text", "transcript", "utterance")
SPEAKER_COLUMNS = ("speaker_id", "speaker", "화자")


def _read_csv(path: Path) -> pd.DataFrame | None:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str)
        except (UnicodeDecodeError, pd.errors.ParserError, OSError):
            continue
    return None


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {column.strip().lower(): column for column in columns}
    return next((lowered[c] for c in candidates if c in lowered), None)


class AIHubEmotionSpeechAdapter(BaseDatasetAdapter):
    """Korean conversational speech clips with optional transcripts."""

    name = "aihub_speech"
    default_subdir = "aihub-emotion-speech"
    license = DatasetLicense(
        dataset_name="AI Hub 감정 분류를 위한 대화 음성 데이터셋",
        source="https://aihub.or.kr (감정 분류를 위한 대화 음성 데이터셋)",
        license="AI Hub 이용약관 (승인 후 다운로드)",
        commercial_use=False,
        research_only=True,
        redistribution_allowed=False,
        usage_scope="research",
        notes="Speech representation / prosody research. Emotion labels are not used as targets.",
    )

    def _load_annotations(self) -> dict[str, dict[str, str | None]]:
        annotations: dict[str, dict[str, str | None]] = {}
        for csv_path in sorted(self.root.rglob("*.csv")):
            table = _read_csv(csv_path)
            if table is None:
                self.skipped["unreadable_csv"] += 1
                continue
            columns = list(table.columns)
            id_column = _pick(columns, ID_COLUMNS)
            if id_column is None:
                continue
            text_column = _pick(columns, TEXT_COLUMNS)
            speaker_column = _pick(columns, SPEAKER_COLUMNS)
            for row in table.to_dict("records"):
                stem = Path(str(row[id_column])).stem
                annotations[stem] = {
                    "transcript": row.get(text_column) if text_column else None,
                    "speaker_id": row.get(speaker_column) if speaker_column else None,
                    "annotation_file": csv_path.name,
                }
        return annotations

    def _iter_samples(self) -> Iterator[InterviewSample]:
        annotations = self._load_annotations()
        for stem, audio in MediaIndex(self.root, AUDIO_EXTENSIONS).items():
            info = annotations.get(stem, {})
            transcript = info.get("transcript")
            transcript = transcript.strip() if isinstance(transcript, str) and transcript.strip() else None
            if transcript is None:
                self.skipped["missing_transcript"] += 1  # audio-only features still work
            speaker = info.get("speaker_id")
            yield self._make_sample(
                sample_id=stem,
                audio_path=audio.as_posix(),
                transcript=transcript,
                speaker_id=str(speaker) if isinstance(speaker, str) and speaker else None,
                metadata={"annotation_file": info.get("annotation_file")},
            )
