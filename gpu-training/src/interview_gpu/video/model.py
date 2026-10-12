"""Temporal models over landmark feature sequences ``[B, T, F]``.

Backbones: ``gru`` | ``lstm`` | ``tcn`` (1D temporal CNN) | ``transformer``.
All share masked mean pooling and an MLP head, so they are interchangeable
via ``model.params.backbone``.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from interview_gpu.common.config import Config
from interview_gpu.common.errors import ConfigError
from interview_gpu.common.runner import DataBundle

BACKBONES = ("gru", "lstm", "tcn", "transformer")


def length_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """Boolean ``[B, T]`` mask, True for real (non-padded) steps."""
    return torch.arange(max_length, device=lengths.device)[None, :] < lengths[:, None]


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, padding=dilation * (kernel_size - 1) // 2, dilation=dilation
        )
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (x + self.dropout(torch.nn.functional.gelu(self.norm(self.conv(x))))) * mask


class TemporalBehaviorModel(nn.Module):
    """Sequence encoder + masked mean pooling + regression/classification head.

    Args:
        input_dim: Number of per-frame features ``F``.
        output_dim: Number of outputs.
        backbone: One of ``BACKBONES``.
        hidden_dim: Model width.
        num_layers: Recurrent / convolutional / transformer layers.
        num_heads: Attention heads (transformer only).
        kernel_size: Odd convolution kernel size (tcn only).
        bidirectional: Bidirectional recurrence (gru / lstm only).
        dropout: Dropout probability.
        max_length: Maximum sequence length for positional encoding (transformer only).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        backbone: str = "transformer",
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        kernel_size: int = 5,
        bidirectional: bool = True,
        dropout: float = 0.1,
        max_length: int = 4096,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ConfigError(f"Unknown video backbone '{backbone}'. Choose from {BACKBONES}.")
        if backbone == "tcn" and kernel_size % 2 == 0:
            raise ConfigError("tcn kernel_size must be odd.")
        self.backbone = backbone
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoded_dim = hidden_dim

        if backbone in ("gru", "lstm"):
            rnn = nn.GRU if backbone == "gru" else nn.LSTM
            self.encoder = rnn(
                hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0,
            )
            encoded_dim = hidden_dim * (2 if bidirectional else 1)
        elif backbone == "tcn":
            self.encoder = nn.ModuleList(
                _ResidualConvBlock(hidden_dim, kernel_size, 2**i, dropout) for i in range(num_layers)
            )
        else:
            layer = nn.TransformerEncoderLayer(
                hidden_dim, num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
            position = torch.arange(max_length).unsqueeze(1)
            frequency = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
            encoding = torch.zeros(max_length, hidden_dim)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency)[:, : hidden_dim // 2]
            self.register_buffer("positional_encoding", encoding, persistent=False)

        self.head = nn.Sequential(
            nn.LayerNorm(encoded_dim), nn.Dropout(dropout),
            nn.Linear(encoded_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim),
        )

    def encode(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Pooled sequence representation ``[B, D]`` (reused by the fusion model)."""
        steps = sequence.shape[1]
        mask = length_mask(lengths, steps)
        x = self.input_proj(sequence)
        if self.backbone in ("gru", "lstm"):
            # cuDNN RNNs are run in fp32: bf16 autocast is not reliably supported for them.
            with torch.autocast(device_type=x.device.type, enabled=False):
                packed = pack_padded_sequence(x.float(), lengths.cpu(), batch_first=True, enforce_sorted=False)
                x, _ = pad_packed_sequence(self.encoder(packed)[0], batch_first=True, total_length=steps)
        elif self.backbone == "tcn":
            channel_mask = mask.unsqueeze(1).to(x.dtype)
            x = x.transpose(1, 2) * channel_mask
            for block in self.encoder:
                x = block(x, channel_mask)
            x = x.transpose(1, 2)
        else:
            if steps > self.positional_encoding.shape[0]:
                raise ConfigError(
                    f"Sequence length {steps} exceeds model max_length {self.positional_encoding.shape[0]}."
                )
            x = x + self.positional_encoding[:steps].to(x.dtype)
            x = self.encoder(x, src_key_padding_mask=~mask)
        weights = mask.unsqueeze(-1).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """``[B, T, F]`` + ``[B]`` lengths -> ``[B, output_dim]``."""
        return self.head(self.encode(sequence, lengths))


def build_video_model(cfg: Config, bundle: DataBundle) -> nn.Module:
    """Instantiate the temporal video model from ``cfg.model``."""
    if cfg.model.name != "video_temporal":
        raise ConfigError(f"Unknown video model '{cfg.model.name}'. Use video_temporal.")
    return TemporalBehaviorModel(
        input_dim=bundle.info["input_dim"], output_dim=bundle.task.output_dim, **cfg.model.params
    )
