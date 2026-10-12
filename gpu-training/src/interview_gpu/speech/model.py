"""Speech behaviour models.

* ``speech_mlp``               features and/or frozen-encoder embeddings -> MLP   (Stage 1 / 2)
* ``speech_encoder_finetune``  raw waveform -> partially unfrozen encoder -> MLP  (Stage 3)
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from interview_gpu.common.config import Config
from interview_gpu.common.errors import ConfigError
from interview_gpu.common.runner import DataBundle
from interview_gpu.speech.encoder import DEFAULT_ENCODER, SpeechEncoder


def mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    """LayerNorm-MLP used as prediction head across the project."""
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers += [nn.Linear(previous, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class SpeechBehaviorMLP(nn.Module):
    """MLP over a fixed-size speech feature vector."""

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: list[int] | None = None, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.net = mlp(input_dim, hidden_dims or [128, 64], output_dim, dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``[B, input_dim]`` -> ``[B, output_dim]``."""
        return self.net(features)


class SpeechEncoderRegressor(nn.Module):
    """Pretrained encoder with a small head, for partial fine-tuning."""

    def __init__(
        self,
        encoder: SpeechEncoder,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        encoder_lr_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = mlp(encoder.hidden_size, hidden_dims or [256], output_dim, dropout)
        self.encoder_lr_scale = encoder_lr_scale

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``[B, samples]`` -> ``[B, output_dim]``."""
        return self.head(self.encoder(waveform, attention_mask))

    def parameter_groups(self, lr: float, weight_decay: float) -> list[dict[str, Any]]:
        """Lower learning rate for pretrained layers than for the new head."""
        encoder = [p for p in self.encoder.parameters() if p.requires_grad]
        groups = [{"params": list(self.head.parameters()), "lr": lr, "weight_decay": weight_decay}]
        if encoder:
            groups.append({"params": encoder, "lr": lr * self.encoder_lr_scale, "weight_decay": weight_decay})
        return groups


def build_speech_model(cfg: Config, bundle: DataBundle) -> nn.Module:
    """Instantiate the model named in ``cfg.model``."""
    params = dict(cfg.model.params)
    output_dim = bundle.task.output_dim
    if cfg.model.name == "speech_mlp":
        return SpeechBehaviorMLP(
            input_dim=bundle.info["input_dim"],
            output_dim=output_dim,
            hidden_dims=params.get("hidden_dims"),
            dropout=params.get("dropout", 0.1),
        )
    if cfg.model.name == "speech_encoder_finetune":
        if cfg.data.params.get("input") != "waveform":
            raise ConfigError("speech_encoder_finetune requires data.params.input: waveform")
        encoder = SpeechEncoder.from_pretrained(
            params.get("encoder_name", DEFAULT_ENCODER), trainable_layers=params.get("trainable_layers", 2)
        )
        return SpeechEncoderRegressor(
            encoder,
            output_dim,
            hidden_dims=params.get("hidden_dims"),
            dropout=params.get("dropout", 0.1),
            encoder_lr_scale=params.get("encoder_lr_scale", 0.1),
        )
    raise ConfigError(f"Unknown speech model '{cfg.model.name}'. Use speech_mlp or speech_encoder_finetune.")
