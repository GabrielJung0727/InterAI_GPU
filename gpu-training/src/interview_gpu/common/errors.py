"""Project-specific exceptions with actionable messages.

Every error a user can realistically hit (missing data, OOM, corrupted media,
checkpoint mismatch, ...) has its own type so CLIs can print a clean message
instead of a stack trace.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType


class InterviewGpuError(Exception):
    """Base class for all expected, user-facing errors."""


class ConfigError(InterviewGpuError):
    """Invalid or inconsistent configuration."""


class DatasetNotFoundError(InterviewGpuError):
    """The dataset directory (raw or processed) does not exist or is empty."""

    def __init__(self, expected_path: str | Path, hint: str | None = None) -> None:
        self.expected_path = Path(expected_path)
        message = (
            "Dataset not found.\n\n"
            f"Expected path:\n{self.expected_path.as_posix()}/\n\n"
            + (hint or "Please place the downloaded dataset in this directory.")
        )
        super().__init__(message)


class EmptyDatasetError(InterviewGpuError):
    """The dataset exists but yields no usable samples."""


class CorruptedMediaError(InterviewGpuError):
    """An audio or video file could not be decoded."""


class InvalidLabelError(InterviewGpuError):
    """Labels are missing, non-numeric or out of the expected range."""


class UsageScopeError(InterviewGpuError):
    """A research-only dataset was requested for production training."""


class MissingDependencyError(InterviewGpuError):
    """An optional dependency required by the requested feature is missing."""


class CheckpointMismatchError(InterviewGpuError):
    """Checkpoint weights do not match the model architecture."""


class NaNLossError(InterviewGpuError):
    """The training loss became NaN or infinite."""


class GpuOutOfMemoryError(InterviewGpuError):
    """CUDA ran out of memory. Never silently falls back to CPU."""


def require(module_name: str, install_hint: str) -> ModuleType:
    """Import an optional dependency or raise a clear error.

    Args:
        module_name: Importable module name, e.g. ``"mediapipe"``.
        install_hint: Command the user should run to install it.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise MissingDependencyError(
            f"Missing dependency '{module_name}'. Install it with: {install_hint}"
        ) from exc
