"""Dataset adapters and their registry."""

from __future__ import annotations

from pathlib import Path

from interview_gpu.common.errors import ConfigError
from interview_gpu.datasets.aihub_interview import AIHubInterviewAdapter
from interview_gpu.datasets.aihub_multimodal import (
    AIHubKoreanFaceVideoAdapter,
    AIHubMultimodalVideoAdapter,
    AIHubRobotInteractionAdapter,
)
from interview_gpu.datasets.aihub_speech import AIHubEmotionSpeechAdapter
from interview_gpu.datasets.base import BaseDatasetAdapter, ensure_scope_allowed
from interview_gpu.datasets.chalearn import ChaLearnFirstImpressionsAdapter
from interview_gpu.datasets.internal import InternalServiceAdapter
from interview_gpu.datasets.mit_interview import MITInterviewAdapter

ADAPTERS: dict[str, type[BaseDatasetAdapter]] = {
    adapter.name: adapter
    for adapter in (
        AIHubInterviewAdapter,
        AIHubEmotionSpeechAdapter,
        AIHubMultimodalVideoAdapter,
        AIHubRobotInteractionAdapter,
        AIHubKoreanFaceVideoAdapter,
        MITInterviewAdapter,
        ChaLearnFirstImpressionsAdapter,
        InternalServiceAdapter,
    )
}


def get_adapter(name: str, root: str | Path | None = None) -> BaseDatasetAdapter:
    """Instantiate the adapter registered under ``name``."""
    if name not in ADAPTERS:
        raise ConfigError(f"Unknown dataset '{name}'. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[name](root)


__all__ = ["ADAPTERS", "BaseDatasetAdapter", "ensure_scope_allowed", "get_adapter"]
