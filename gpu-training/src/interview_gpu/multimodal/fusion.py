"""Fusion transformer over one token per modality."""

from __future__ import annotations

import torch
from torch import nn


class FusionTransformer(nn.Module):
    """Fuses modality tokens with self-attention and returns a [CLS] summary.

    Missing modalities are masked out of attention, so the same model handles
    audio-only, text-only or full audio+video+text samples.

    Args:
        dim: Token width.
        num_modalities: Number of modality slots.
        num_layers: Transformer encoder layers.
        num_heads: Attention heads.
        dropout: Dropout probability.
    """

    def __init__(
        self, dim: int = 256, num_modalities: int = 3, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.modality_embedding = nn.Parameter(torch.zeros(1, num_modalities, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.modality_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            dim, num_heads, dim_feedforward=dim * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        """Fuse modality tokens.

        Args:
            tokens: ``[B, M, dim]`` one projected token per modality.
            modality_mask: ``[B, M]``, non-zero where the modality is present.

        Returns:
            ``[B, dim]`` fused representation.
        """
        batch = tokens.shape[0]
        tokens = tokens + self.modality_embedding.to(tokens.dtype)
        sequence = torch.cat([self.cls_token.to(tokens.dtype).expand(batch, -1, -1), tokens], dim=1)
        present = torch.cat([modality_mask.new_ones(batch, 1), modality_mask], dim=1) > 0
        return self.norm(self.encoder(sequence, src_key_padding_mask=~present)[:, 0])
