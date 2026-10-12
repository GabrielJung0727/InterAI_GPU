"""Temporal landmark features: ``T x F`` sequences and raw summary metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from interview_gpu.schemas.video import VIDEO_FEATURE_COLUMNS, VIDEO_MODEL_FEATURES
from interview_gpu.video.landmarks import (
    LEFT_ELBOW,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ELBOW,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    FrameLandmarks,
    VideoLandmarks,
    static_frame_features,
)

_BODY_POINTS = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW)
_HAND_POINTS = (LEFT_WRIST, RIGHT_WRIST)

#: |gaze proxy| below these bounds counts as "looking towards the camera".
GAZE_CENTER_X, GAZE_CENTER_Y = 0.35, 0.5
#: Wrist speed (shoulder widths / second) above which a gesture is counted.
GESTURE_SPEED_THRESHOLD = 0.5


def _motion(previous: FrameLandmarks, current: FrameLandmarks, points: tuple[int, ...]) -> float:
    """Mean landmark speed in shoulder widths per second (scale invariant)."""
    if previous.pose is None or current.pose is None:
        return float("nan")
    dt = current.timestamp - previous.timestamp
    shoulder_width = float(np.linalg.norm(current.pose[LEFT_SHOULDER, :2] - current.pose[RIGHT_SHOULDER, :2]))
    if dt <= 0 or shoulder_width < 1e-4:
        return float("nan")
    displacement = np.linalg.norm(current.pose[list(points), :2] - previous.pose[list(points), :2], axis=1)
    return float(displacement.mean() / shoulder_width / dt)


def build_feature_table(video: VideoLandmarks) -> pd.DataFrame:
    """Per-frame feature table with exactly ``VIDEO_FEATURE_COLUMNS``."""
    rows = []
    previous: FrameLandmarks | None = None
    for frame in video.frames:
        row = static_frame_features(frame, video.width, video.height)
        row["upper_body_movement"] = _motion(previous, frame, _BODY_POINTS) if previous else float("nan")
        row["gesture_motion"] = _motion(previous, frame, _HAND_POINTS) if previous else float("nan")
        rows.append(row)
        previous = frame
    return pd.DataFrame(rows, columns=list(VIDEO_FEATURE_COLUMNS)).astype(np.float32)


def table_to_sequence(table: pd.DataFrame) -> np.ndarray:
    """Model input ``[T, F]`` (timestamp dropped; NaN kept for later imputation)."""
    return table[list(VIDEO_MODEL_FEATURES)].to_numpy(dtype=np.float32)


def _nan_stat(values: pd.Series, fn: str) -> float | None:
    values = values.dropna()
    if values.empty:
        return None
    return float(getattr(values, fn)())


def summarize_table(table: pd.DataFrame) -> dict[str, float | None]:
    """Raw, interpretable behaviour measurements for one video.

    These are the evidence the service layer explains to the user (e.g.
    ``gaze_center_ratio = 0.61``); none of them is a psychological judgement.
    """
    frames = len(table)
    if frames == 0:
        return {"frame_count": 0.0}
    duration = float(table["timestamp"].iloc[-1] - table["timestamp"].iloc[0]) if frames > 1 else 0.0
    visible = table["face_visible"] > 0.5
    gaze = table.loc[visible, ["gaze_proxy_x", "gaze_proxy_y"]].dropna()
    centered = (gaze["gaze_proxy_x"].abs() < GAZE_CENTER_X) & (gaze["gaze_proxy_y"].abs() < GAZE_CENTER_Y)
    active = (table["gesture_motion"] > GESTURE_SPEED_THRESHOLD).to_numpy()
    gesture_events = int(np.sum(active[1:] & ~active[:-1]) + (1 if active[0] else 0))
    return {
        "frame_count": float(frames),
        "duration_sec": duration,
        "face_visible_ratio": float(visible.mean()),
        "gaze_center_ratio": float(centered.mean()) if len(gaze) else None,
        "head_yaw_std": _nan_stat(table["head_yaw"], "std"),
        "head_pitch_std": _nan_stat(table["head_pitch"], "std"),
        "head_roll_std": _nan_stat(table["head_roll"], "std"),
        "head_yaw_abs_mean": _nan_stat(table["head_yaw"].abs(), "mean"),
        "shoulder_angle_abs_mean": _nan_stat(table["shoulder_angle"].abs(), "mean"),
        "upper_body_movement_mean": _nan_stat(table["upper_body_movement"], "mean"),
        "gesture_motion_mean": _nan_stat(table["gesture_motion"], "mean"),
        "gesture_events_per_min": gesture_events / duration * 60.0 if duration > 0 else None,
        "landmark_confidence_mean": _nan_stat(table["landmark_confidence"], "mean"),
    }


def save_feature_table(table: pd.DataFrame, path: str | Path) -> None:
    """Save a feature table as ``.npy`` (``[T, len(VIDEO_FEATURE_COLUMNS)]``) or ``.parquet``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        table.to_parquet(path, index=False)
    else:
        np.save(path, table[list(VIDEO_FEATURE_COLUMNS)].to_numpy(dtype=np.float32))


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """Inverse of :func:`save_feature_table`."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.DataFrame(np.load(path), columns=list(VIDEO_FEATURE_COLUMNS))
