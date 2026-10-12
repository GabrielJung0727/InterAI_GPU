"""Stage 2 / 3 — pretrained speech encoders (WavLM, Wav2Vec2, HuBERT).

Stage 2 keeps the encoder frozen and only extracts utterance embeddings.
Stage 3 unfreezes the last ``trainable_layers`` transformer layers.
The convolutional feature extractor always stays frozen.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

DEFAULT_ENCODER = "microsoft/wavlm-base-plus"


class SpeechEncoder(nn.Module):
    """Wraps a HuggingFace speech backbone and mean-pools its frame outputs.

    Args:
        backbone: Model returning ``last_hidden_state`` of shape ``[B, frames, H]``
            for a raw waveform input ``[B, samples]``.
        trainable_layers: Number of top transformer layers left trainable
            (0 = fully frozen).
    """

    def __init__(self, backbone: nn.Module, trainable_layers: int = 0) -> None:
        super().__init__()
        self.backbone = backbone
        self.hidden_size = int(backbone.config.hidden_size)
        self.trainable_layers = trainable_layers
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        if trainable_layers > 0:
            for layer in backbone.encoder.layers[-trainable_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    @classmethod
    def from_pretrained(cls, name: str = DEFAULT_ENCODER, trainable_layers: int = 0) -> "SpeechEncoder":
        """Load pretrained weights from the HuggingFace hub (or a local path)."""
        from transformers import AutoModel

        return cls(AutoModel.from_pretrained(name), trainable_layers)

    def train(self, mode: bool = True) -> "SpeechEncoder":
        """Keep a fully frozen backbone in eval mode (no dropout / masking)."""
        super().train(mode)
        if self.trainable_layers == 0:
            self.backbone.eval()
        return self

    def forward(self, waveform: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode waveforms into utterance embeddings.

        Args:
            waveform: ``[B, samples]`` mono audio at 16 kHz.
            attention_mask: ``[B, samples]`` with 1 for real samples, 0 for padding.

        Returns:
            ``[B, hidden_size]`` mean of the frame representations over
            non-padded frames.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(waveform, dtype=torch.long)
        valid = attention_mask.to(waveform.dtype)
        lengths = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        # Per-utterance zero-mean / unit-variance, as these encoders expect.
        mean = (waveform * valid).sum(dim=1, keepdim=True) / lengths
        variance = (((waveform - mean) * valid) ** 2).sum(dim=1, keepdim=True) / lengths
        waveform = (waveform - mean) / torch.sqrt(variance + 1e-7) * valid

        uses_mask = getattr(self.backbone.config, "feat_extract_norm", "group") == "layer"
        hidden = self.backbone(waveform, attention_mask=attention_mask if uses_mask else None).last_hidden_state

        frames = hidden.shape[1]
        valid_frames = torch.ceil(lengths.squeeze(1) / waveform.shape[1] * frames).clamp(1, frames)
        frame_mask = (torch.arange(frames, device=hidden.device)[None, :] < valid_frames[:, None]).to(hidden.dtype)
        return (hidden * frame_mask.unsqueeze(-1)).sum(dim=1) / frame_mask.sum(dim=1, keepdim=True)

    @torch.no_grad()
    def embed(self, wave: np.ndarray, max_seconds: float = 30.0, sample_rate: int = 16000) -> np.ndarray:
        """Embed one utterance (truncated to ``max_seconds``) as a numpy vector."""
        self.eval()
        device = next(self.parameters()).device
        tensor = torch.from_numpy(wave[: int(max_seconds * sample_rate)]).float().unsqueeze(0).to(device)
        return self.forward(tensor).squeeze(0).float().cpu().numpy()
