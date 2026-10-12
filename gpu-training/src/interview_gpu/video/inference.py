"""Service-facing video behaviour inference interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from interview_gpu.common.checkpoint import load_checkpoint, load_model_state
from interview_gpu.common.config import config_from_dict, resolve_path
from interview_gpu.common.device import select_device
from interview_gpu.common.errors import EmptyDatasetError
from interview_gpu.common.io import to_jsonable
from interview_gpu.common.normalization import apply_normalizer
from interview_gpu.common.tasks import TaskSpec
from interview_gpu.schemas.video import VIDEO_MODEL_FEATURES, VideoPrediction
from interview_gpu.video.landmarks import LandmarkExtractor
from interview_gpu.video.model import TemporalBehaviorModel
from interview_gpu.video.temporal_features import build_feature_table, summarize_table, table_to_sequence


class VideoBehaviorModel:
    """Analyse the visual delivery of one answer.

    Always returns raw landmark-derived measurements (``metrics``); model
    ``scores`` are added when a trained checkpoint is supplied. Neither
    describes personality, honesty, mental state or hiring suitability.

    Example::

        model = VideoBehaviorModel("checkpoints/video/video_v001")
        model.predict("answer.mp4")
        # {"metrics": {"gaze_center_ratio": 0.61, "head_yaw_std": 4.2, ...},
        #  "scores": {"visual_delivery": 0.81, ...}, "model_version": "video_v001"}
    """

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        device: str = "auto",
        face_model: str | Path | None = None,
        pose_model: str | Path | None = None,
        target_fps: float = 5.0,
    ) -> None:
        self.device = select_device(device)
        self.target_fps = target_fps
        self._model_paths = (face_model, pose_model)
        self._extractor: LandmarkExtractor | None = None
        self.model: TemporalBehaviorModel | None = None
        self.task: TaskSpec | None = None
        self.normalization: dict[str, Any] = {}
        self.model_version: str | None = None
        if checkpoint_dir is not None:
            checkpoint_dir = resolve_path(checkpoint_dir)
            payload = load_checkpoint(checkpoint_dir, self.device)
            cfg = config_from_dict(payload["config"])
            self.task = TaskSpec.from_dict(payload["task"])
            self.normalization = payload["normalization"]
            self.target_fps = float(payload["dataset_metadata"].get("target_fps", target_fps))
            self.model = TemporalBehaviorModel(
                input_dim=len(VIDEO_MODEL_FEATURES), output_dim=self.task.output_dim, **cfg.model.params
            )
            load_model_state(self.model, payload["model_state"])
            self.model.to(self.device).eval()
            self.model_version = checkpoint_dir.name

    def extract(self, video_path: str | Path) -> pd.DataFrame:
        """Run MediaPipe on a video and return its per-frame feature table."""
        if self._extractor is None:  # created lazily: needs the MediaPipe model files
            self._extractor = LandmarkExtractor(*self._model_paths)
        return build_feature_table(self._extractor.process_video(video_path, self.target_fps))

    @torch.no_grad()
    def _scores(self, table: pd.DataFrame) -> dict[str, float]:
        assert self.model is not None and self.task is not None
        sequence = apply_normalizer(table_to_sequence(table), self.normalization)
        max_frames = int(self.normalization.get("max_frames", len(sequence)))
        start = max(0, (len(sequence) - max_frames) // 2)
        tensor = torch.from_numpy(sequence[start : start + max_frames]).unsqueeze(0).to(self.device)
        lengths = torch.tensor([tensor.shape[1]], device=self.device)
        outputs = self.model(tensor, lengths).float().cpu()
        if self.task.kind == "classification":
            probs = torch.softmax(outputs, dim=1).numpy()[0]
            return {name: float(p) for name, p in zip(self.task.class_names or [], probs)}
        values = self.task.denormalize(outputs.numpy())[0]
        return {name: float(value) for name, value in zip(self.task.target_names, values)}

    def predict(self, video: str | Path | pd.DataFrame) -> dict[str, Any]:
        """Analyse a video.

        Args:
            video: Path to a video file, or an already extracted feature
                table (columns ``VIDEO_FEATURE_COLUMNS``) — e.g. when the
                client computes landmarks in the browser.

        Returns:
            JSON-serializable dict: ``metrics``, ``scores``, ``model_version``.
        """
        table = video if isinstance(video, pd.DataFrame) else self.extract(video)
        if len(table) == 0:
            raise EmptyDatasetError("The video produced no frames to analyse.")
        scores = self._scores(table) if self.model is not None else {}
        prediction = VideoPrediction(
            metrics=to_jsonable(summarize_table(table)), scores=scores, model_version=self.model_version
        )
        return prediction.to_json_dict()
