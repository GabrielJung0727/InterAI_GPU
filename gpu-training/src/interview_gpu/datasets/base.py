"""Dataset adapter interface.

    raw dataset  ->  Adapter  ->  InterviewSample  ->  model-specific Dataset

Adapters never download anything. If the data is not on disk they raise
:class:`DatasetNotFoundError` naming the directory to populate.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar, Iterator

from interview_gpu.common.config import resolve_path
from interview_gpu.common.errors import DatasetNotFoundError, EmptyDatasetError, UsageScopeError
from interview_gpu.schemas.interview import DatasetLicense, InterviewSample

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
RAW_DATA_DIR = "data/raw"


def dig(record: Any, *paths: str) -> Any:
    """Return the first non-empty value found under dotted ``paths``.

    Example: ``dig(obj, "dataSet.answer.raw.text", "answer")``.
    """
    for path in paths:
        node = record
        for key in path.split("."):
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if node not in (None, "", [], {}):
            return node
    return None


def read_json_file(path: Path) -> Any | None:
    """Parse a JSON file (UTF-8 with or without BOM); ``None`` if unreadable."""
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


class MediaIndex:
    """One-pass index ``file stem -> path`` for media files under a root."""

    def __init__(self, root: Path, extensions: tuple[str, ...]) -> None:
        self._by_stem: dict[str, Path] = {}
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() in extensions and path.is_file():
                self._by_stem.setdefault(path.stem, path)

    def __len__(self) -> int:
        return len(self._by_stem)

    def find(self, name: str | None) -> Path | None:
        """Look up by file name or stem."""
        return self._by_stem.get(Path(name).stem) if name else None

    def items(self) -> Iterator[tuple[str, Path]]:
        return iter(self._by_stem.items())


class BaseDatasetAdapter(ABC):
    """Converts one raw dataset into :class:`InterviewSample` objects.

    Subclasses define ``name``, ``default_subdir``, ``license`` and implement
    :meth:`_iter_samples`. Records that cannot be used are counted in
    ``self.skipped`` (reason -> count) instead of aborting the whole run.
    """

    name: ClassVar[str]
    default_subdir: ClassVar[str]
    license: ClassVar[DatasetLicense]

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else resolve_path(RAW_DATA_DIR) / self.default_subdir
        self.skipped: Counter[str] = Counter()

    def ensure_exists(self) -> None:
        """Raise :class:`DatasetNotFoundError` when the directory is missing or empty."""
        if not self.root.is_dir() or not any(self.root.iterdir()):
            raise DatasetNotFoundError(
                self.root,
                "Please place the downloaded dataset in this directory.\n"
                f"Dataset: {self.license.dataset_name}\nSource:  {self.license.source}\n"
                "Datasets are never downloaded automatically.",
            )

    @abstractmethod
    def _iter_samples(self) -> Iterator[InterviewSample]:
        """Yield samples from ``self.root`` (which is guaranteed to exist)."""

    def _make_sample(self, **fields: Any) -> InterviewSample:
        """Build a sample stamped with this dataset's name and usage scope."""
        return InterviewSample(
            source_dataset=self.name, usage_scope=self.license.usage_scope, **fields
        )

    def iter_samples(self, limit: int | None = None) -> Iterator[InterviewSample]:
        """Iterate samples, raising if the dataset is missing or yields nothing."""
        self.ensure_exists()
        self.skipped.clear()
        count = 0
        for sample in self._iter_samples():
            yield sample
            count += 1
            if limit is not None and count >= limit:
                return
        if count == 0:
            raise EmptyDatasetError(
                f"No usable samples found in {self.root.as_posix()} for dataset '{self.name}'. "
                f"Skipped records: {dict(self.skipped) or 'none'}. "
                "The directory layout may differ from what the adapter expects — "
                "run scripts/inspect_dataset.py and adjust the adapter."
            )

    def load(self, limit: int | None = None) -> list[InterviewSample]:
        """Materialise all samples."""
        return list(self.iter_samples(limit))


def ensure_scope_allowed(license_info: DatasetLicense, requested_scope: str) -> None:
    """Block research-only datasets from production pipelines."""
    if requested_scope == "production" and license_info.usage_scope != "production":
        raise UsageScopeError(
            f"Dataset '{license_info.dataset_name}' is licensed for '{license_info.usage_scope}' use only "
            f"({license_info.license}) and cannot be used with usage_scope=production."
        )
