"""Service-facing speech inference interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from interview_gpu.common.checkpoint import load_checkpoint, load_model_state
from interview_gpu.common.config import config_from_dict, resolve_path
from interview_gpu.common.device import select_device
from interview_gpu.common.errors import CheckpointMismatchError
from interview_gpu.common.io import to_jsonable
from interview_gpu.common.normalization import apply_normalizer
from interview_gpu.common.tasks import TaskSpec
from interview_gpu.schemas.speech import SpeechPrediction
from interview_gpu.speech.encoder import DEFAULT_ENCODER, SpeechEncoder
from interview_gpu.speech.features import (
    SpeechFeatureConfig,
    extract_speech_features,
    features_to_vector,
    load_audio,
)
from interview_gpu.speech.model import SpeechBehaviorMLP, SpeechEncoderRegressor


class SpeechAnalyzer:
    """Analyse one spoken answer.

    Always returns the raw, observable measurements (``metrics``). When a
    trained checkpoint is supplied, model ``scores`` are added. The output
    contains numbers only; wording for end users belongs to the service layer.

    Example::

        analyzer = SpeechAnalyzer("checkpoints/speech/speech_v001")
        analyzer.predict("answer.wav", transcript="...")
        # {"metrics": {"speech_rate_wpm": 196.0, "long_pause_count": 2.0, ...},
        #  "scores": {"pace_score": 0.82, ...}, "model_version": "speech_v001"}
    """

    def __init__(self, checkpoint_dir: str | Path | None = None, device: str = "auto") -> None:
        self.device = select_device(device)
        self.model: torch.nn.Module | None = None
        self.encoder: SpeechEncoder | None = None
        self.task: TaskSpec | None = None
        self.normalization: dict[str, Any] = {}
        self.model_version: str | None = None
        self.sample_rate = 16000
        if checkpoint_dir is not None:
            self._load(resolve_path(checkpoint_dir))
        self.feature_config = SpeechFeatureConfig(sample_rate=self.sample_rate)

    def _load(self, checkpoint_dir: Path) -> None:
        payload = load_checkpoint(checkpoint_dir, self.device)
        cfg = config_from_dict(payload["config"])
        self.task = TaskSpec.from_dict(payload["task"])
        self.normalization = payload["normalization"]
        self.sample_rate = int(self.normalization.get("sample_rate", 16000))
        self.model_version = checkpoint_dir.name
        params = cfg.model.params
        mode = self.normalization.get("input")
        if mode == "waveform":
            from transformers import AutoConfig, AutoModel

            # Architecture only; the fine-tuned weights come from the checkpoint.
            backbone = AutoModel.from_config(AutoConfig.from_pretrained(params.get("encoder_name", DEFAULT_ENCODER)))
            self.model = SpeechEncoderRegressor(
                SpeechEncoder(backbone, params.get("trainable_layers", 2)),
                self.task.output_dim,
                hidden_dims=params.get("hidden_dims"),
                dropout=params.get("dropout", 0.1),
            )
        elif mode in ("handcrafted", "embedding", "both"):
            self.model = SpeechBehaviorMLP(
                input_dim=len(self.normalization["mean"]),
                output_dim=self.task.output_dim,
                hidden_dims=params.get("hidden_dims"),
                dropout=params.get("dropout", 0.1),
            )
            if mode != "handcrafted":
                encoder_name = self.normalization.get("encoder_name")
                if not encoder_name:
                    raise CheckpointMismatchError("Checkpoint uses embeddings but records no encoder_name.")
                self.encoder = SpeechEncoder.from_pretrained(encoder_name).to(self.device).eval()
        else:
            raise CheckpointMismatchError(f"Checkpoint has an unknown speech input mode: {mode!r}")
        load_model_state(self.model, payload["model_state"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _scores(self, wave: np.ndarray, metrics: dict[str, float]) -> dict[str, float]:
        assert self.model is not None and self.task is not None
        mode = self.normalization["input"]
        if mode == "waveform":
            wave = wave[: int(self.normalization.get("max_seconds", 20.0) * self.sample_rate)]
            tensor = torch.from_numpy(wave).unsqueeze(0).to(self.device)
            outputs = self.model(tensor, torch.ones_like(tensor, dtype=torch.long))
        else:
            parts = []
            if mode in ("handcrafted", "both"):
                parts.append(features_to_vector(metrics))
            if mode in ("embedding", "both"):
                assert self.encoder is not None
                parts.append(self.encoder.embed(wave, sample_rate=self.sample_rate))
            vector = apply_normalizer(np.concatenate(parts)[None, :], self.normalization)
            outputs = self.model(torch.from_numpy(vector).to(self.device))
        outputs = outputs.float().cpu().numpy()
        if self.task.kind == "classification":
            probs = torch.softmax(torch.from_numpy(outputs), dim=1).numpy()[0]
            return {name: float(p) for name, p in zip(self.task.class_names or [], probs)}
        values = self.task.denormalize(outputs)[0]
        return {name: float(value) for name, value in zip(self.task.target_names, values)}

    def predict(
        self,
        audio: str | Path | np.ndarray,
        transcript: str | None = None,
        sample_rate: int | None = None,
        response_latency: float | None = None,
    ) -> dict[str, Any]:
        """Analyse an utterance.

        Args:
            audio: Path to an audio file, or a mono waveform array.
            transcript: STT text of the answer (enables rate / filler metrics).
            sample_rate: Sample rate of ``audio`` when an array is passed.
            response_latency: Seconds from question end to answer start, if known.

        Returns:
            JSON-serializable dict: ``metrics``, ``scores``, ``model_version``.
        """
        if isinstance(audio, (str, Path)):
            wave = load_audio(audio, self.sample_rate)
        else:
            wave = np.asarray(audio, dtype=np.float32)
            if sample_rate and sample_rate != self.sample_rate:
                import librosa

                wave = librosa.resample(wave, orig_sr=sample_rate, target_sr=self.sample_rate)
        metrics = extract_speech_features(wave, transcript, self.feature_config, response_latency)
        scores = self._scores(wave, metrics) if self.model is not None else {}
        prediction = SpeechPrediction(
            metrics=to_jsonable(metrics), scores=scores, model_version=self.model_version
        )
        return prediction.to_json_dict()
