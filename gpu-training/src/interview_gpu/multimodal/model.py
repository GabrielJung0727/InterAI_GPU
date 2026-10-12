"""Multimodal fusion model skeleton (P4).

    speech / video / text embeddings -> projection layers -> fusion transformer
    -> multi-task heads (answer_quality, delivery_quality, communication_metrics)

Inputs are *pooled embeddings* produced by the unimodal models
(``SpeechEncoder``, ``TemporalBehaviorModel.encode``,
``AnswerEvaluatorModel.encode``). This is an extensible skeleton: it is
covered by a smoke test only and has not been trained, because no aligned
multimodal interview data exists yet.
"""

from __future__ import annotations

import torch
from torch import nn

from interview_gpu.common.config import Config
from interview_gpu.common.errors import ConfigError
from interview_gpu.common.runner import DataBundle
from interview_gpu.multimodal.fusion import FusionTransformer

MODALITIES: tuple[str, ...] = ("speech", "video", "text")
DEFAULT_HEADS: dict[str, int] = {"answer_quality": 1, "delivery_quality": 1, "communication_metrics": 4}


class MultimodalInterviewModel(nn.Module):
    """Late-fusion model with one head per task.

    Args:
        input_dims: Embedding width per modality, e.g. ``{"speech": 768, ...}``.
        heads: Output size per task head; outputs are concatenated in this order.
        fusion_dim: Shared token width.
        num_layers, num_heads, dropout: Fusion transformer settings.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        heads: dict[str, int] | None = None,
        fusion_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.head_sizes = dict(heads or DEFAULT_HEADS)
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(nn.LayerNorm(input_dims[name]), nn.Linear(input_dims[name], fusion_dim), nn.GELU())
                for name in MODALITIES
            }
        )
        self.fusion = FusionTransformer(fusion_dim, len(MODALITIES), num_layers, num_heads, dropout)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Dropout(dropout), nn.Linear(fusion_dim, size))
                for name, size in self.head_sizes.items()
            }
        )

    @property
    def output_dim(self) -> int:
        return sum(self.head_sizes.values())

    def split_outputs(self, outputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Split concatenated outputs back into named task heads."""
        return dict(zip(self.head_sizes, torch.split(outputs, list(self.head_sizes.values()), dim=1)))

    def forward(
        self, speech: torch.Tensor, video: torch.Tensor, text: torch.Tensor, modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """Fuse embeddings and predict all tasks.

        Args:
            speech, video, text: ``[B, input_dims[m]]`` (zeros when missing).
            modality_mask: ``[B, 3]`` in ``MODALITIES`` order, 1 = present.

        Returns:
            ``[B, output_dim]`` — head outputs concatenated.
        """
        embeddings = {"speech": speech, "video": video, "text": text}
        tokens = torch.stack([self.projections[name](embeddings[name]) for name in MODALITIES], dim=1)
        fused = self.fusion(tokens, modality_mask)
        return torch.cat([head(fused) for head in self.heads.values()], dim=1)


def build_multimodal_model(cfg: Config, bundle: DataBundle) -> nn.Module:
    """Instantiate the fusion model; head sizes must add up to the targets."""
    if cfg.model.name != "multimodal_fusion":
        raise ConfigError(f"Unknown multimodal model '{cfg.model.name}'. Use multimodal_fusion.")
    params = dict(cfg.model.params)
    heads = params.pop("heads", None) or {"outputs": bundle.task.output_dim}
    if sum(heads.values()) != bundle.task.output_dim:
        raise ConfigError(
            f"model.params.heads sizes sum to {sum(heads.values())} but data.target_columns "
            f"defines {bundle.task.output_dim} target(s)."
        )
    return MultimodalInterviewModel(bundle.info["input_dims"], heads, **params)
