"""Multi-head interview answer evaluator.

    shared text encoder -> pooled representation -> one regression head per
    rubric dimension (relevance, structure, specificity, evidence, ...)

Training stages supported by configuration:

* Stage B  small pretrained encoder (multilingual-e5, BGE-M3, Korean encoders),
           fully fine-tuned, partially unfrozen or frozen.
* Stage C  LoRA adapters on the encoder (``model.params.lora.enabled``).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from interview_gpu.common.config import Config
from interview_gpu.common.errors import ConfigError, require
from interview_gpu.common.runner import DataBundle

DEFAULT_ENCODER = "intfloat/multilingual-e5-small"


def freeze_encoder(encoder: nn.Module, trainable_layers: int = 0) -> None:
    """Freeze the encoder, optionally leaving the top ``trainable_layers`` trainable."""
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    if trainable_layers > 0:
        for layer in encoder.encoder.layer[-trainable_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True


def apply_lora(encoder: nn.Module, settings: dict[str, Any]) -> nn.Module:
    """Wrap the encoder with LoRA adapters (requires the ``lora`` extra)."""
    peft = require("peft", "uv sync --extra lora")
    config = peft.LoraConfig(
        r=settings.get("r", 8),
        lora_alpha=settings.get("alpha", 16),
        lora_dropout=settings.get("dropout", 0.05),
        target_modules=settings.get("target_modules", ["query", "value"]),
        bias="none",
    )
    return peft.get_peft_model(encoder, config)


class AnswerEvaluatorModel(nn.Module):
    """Shared encoder with one regression head per rubric dimension.

    Args:
        encoder: Module returning ``last_hidden_state`` ``[B, L, H]`` for
            ``input_ids`` / ``attention_mask``.
        hidden_size: Encoder output width ``H``.
        dimensions: Head names, in output order.
        head_hidden_dim: Width of each head's hidden layer.
        dropout: Dropout on the pooled representation.
        encoder_lr_scale: Encoder learning rate relative to the heads.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        dimensions: list[str],
        head_hidden_dim: int = 256,
        dropout: float = 0.1,
        encoder_lr_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dimensions = list(dimensions)
        self.encoder_lr_scale = encoder_lr_scale
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(hidden_size, head_hidden_dim), nn.GELU(), nn.Linear(head_hidden_dim, 1))
                for name in self.dimensions
            }
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pooled answer representation ``[B, H]`` (reused by the fusion model)."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Token ids -> ``[B, len(dimensions)]`` scores (standardised units)."""
        pooled = self.dropout(self.encode(input_ids, attention_mask))
        return torch.cat([self.heads[name](pooled) for name in self.dimensions], dim=1)

    def parameter_groups(self, lr: float, weight_decay: float) -> list[dict[str, Any]]:
        """Separate learning rates for the pretrained encoder and the new heads."""
        encoder = [p for p in self.encoder.parameters() if p.requires_grad]
        groups = [{"params": list(self.heads.parameters()), "lr": lr, "weight_decay": weight_decay}]
        if encoder:
            groups.append({"params": encoder, "lr": lr * self.encoder_lr_scale, "weight_decay": weight_decay})
        return groups


def build_answer_encoder(params: dict[str, Any], pretrained: bool = True) -> nn.Module:
    """Create the text encoder and apply freezing / LoRA settings.

    Args:
        params: ``cfg.model.params``.
        pretrained: Load hub weights (training). ``False`` builds the bare
            architecture for inference, where weights come from a checkpoint.
    """
    from transformers import AutoConfig, AutoModel

    name = params.get("encoder_name", DEFAULT_ENCODER)
    encoder = AutoModel.from_pretrained(name) if pretrained else AutoModel.from_config(AutoConfig.from_pretrained(name))
    if params.get("gradient_checkpointing"):
        encoder.gradient_checkpointing_enable()
    lora = params.get("lora") or {}
    if lora.get("enabled"):
        return apply_lora(encoder, lora)  # peft freezes the base weights itself
    if params.get("freeze_encoder"):
        freeze_encoder(encoder, 0)
    elif params.get("trainable_layers") is not None:
        freeze_encoder(encoder, int(params["trainable_layers"]))
    return encoder


def make_answer_model(params: dict[str, Any], dimensions: list[str], pretrained: bool = True) -> AnswerEvaluatorModel:
    """Assemble encoder and heads from model parameters."""
    encoder = build_answer_encoder(params, pretrained)
    return AnswerEvaluatorModel(
        encoder,
        hidden_size=int(encoder.config.hidden_size),
        dimensions=dimensions,
        head_hidden_dim=params.get("head_hidden_dim", 256),
        dropout=params.get("dropout", 0.1),
        encoder_lr_scale=params.get("encoder_lr_scale", 1.0),
    )


def build_answer_model(cfg: Config, bundle: DataBundle) -> nn.Module:
    """Instantiate the answer evaluator from ``cfg.model``."""
    if cfg.model.name != "answer_evaluator":
        raise ConfigError(f"Unknown answer model '{cfg.model.name}'. Use answer_evaluator.")
    if bundle.task.kind != "regression":
        raise ConfigError("The answer evaluator is a multi-head regression model (data.task: regression).")
    return make_answer_model(cfg.model.params, bundle.task.target_names)
