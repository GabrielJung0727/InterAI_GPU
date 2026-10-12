"""Video landmark-feature and prediction schemas.

The video model consumes per-frame behaviour features derived from MediaPipe
landmarks, never raw RGB frames. Targets are behaviour-based (camera
engagement, movement stability, ...) — never personality, honesty, mental
health or hiring suitability.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

#: Columns of the per-frame feature table saved by ``prepare_video.py``.
VIDEO_FEATURE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "head_yaw",
    "head_pitch",
    "head_roll",
    "face_visible",
    "face_center_x",
    "face_center_y",
    "gaze_proxy_x",
    "gaze_proxy_y",
    "left_shoulder_x",
    "left_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_y",
    "shoulder_angle",
    "upper_body_movement",
    "gesture_motion",
    "landmark_confidence",
)

#: Model inputs: everything except the timestamp.
VIDEO_MODEL_FEATURES: tuple[str, ...] = VIDEO_FEATURE_COLUMNS[1:]

#: Example behaviour scores a trained video model may output.
VIDEO_TARGET_EXAMPLES: tuple[str, ...] = (
    "visual_delivery",
    "gaze_stability",
    "movement_stability",
    "camera_engagement",
    "head_movement_consistency",
)


class VideoPrediction(BaseModel):
    """Result of ``VideoBehaviorModel.predict`` (JSON-serializable)."""

    #: Raw measurements (always present), e.g. ``gaze_center_ratio``.
    metrics: dict[str, float | None]
    #: Model scores in label units; empty when no checkpoint is loaded.
    scores: dict[str, float] = Field(default_factory=dict)
    model_version: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
