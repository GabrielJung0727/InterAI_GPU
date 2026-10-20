"""Landmark geometry, feature schema and video decoding (CPU, no MediaPipe models needed)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from interview_gpu.common.errors import CorruptedMediaError, MissingDependencyError
from interview_gpu.schemas.video import VIDEO_FEATURE_COLUMNS, VIDEO_MODEL_FEATURES
from interview_gpu.video import landmarks as lm
from interview_gpu.video.temporal_features import (
    build_feature_table,
    load_feature_table,
    save_feature_table,
    summarize_table,
    table_to_sequence,
)

WIDTH, HEIGHT = 640, 480


def synthetic_face(yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0, iris_shift: float = 0.0) -> np.ndarray:
    """Project the generic 3D face model with a known rotation into landmark space."""
    import cv2

    rotation_y, _ = cv2.Rodrigues(np.array([0.0, math.radians(yaw), 0.0]))
    rotation_x, _ = cv2.Rodrigues(np.array([math.radians(pitch), 0.0, 0.0]))
    rotation_z, _ = cv2.Rodrigues(np.array([0.0, 0.0, math.radians(roll)]))
    rvec, _ = cv2.Rodrigues(rotation_z @ rotation_y @ rotation_x)
    points, _ = cv2.projectPoints(
        lm._FACE_MODEL_POINTS, rvec, np.array([0.0, 0.0, 1500.0]), lm.camera_matrix(WIDTH, HEIGHT), None
    )
    face = np.full((lm.IRIS_LANDMARK_COUNT, 3), 0.5, dtype=np.float32)
    face[list(lm._FACE_MODEL_INDICES), :2] = points.reshape(-1, 2) / np.array([WIDTH, HEIGHT])
    # Eyes: corners 0.1 apart, lids 0.04 apart, iris centred (plus optional horizontal shift).
    for (iris, corner_a, corner_b, upper, lower), centre_x in zip(lm.EYES, (0.4, 0.6)):
        if corner_a not in lm._FACE_MODEL_INDICES:
            face[corner_a, :2] = (centre_x - 0.05, 0.4)
        if corner_b not in lm._FACE_MODEL_INDICES:
            face[corner_b, :2] = (centre_x + 0.05, 0.4)
        middle = (face[corner_a, 0] + face[corner_b, 0]) / 2
        half_width = (face[corner_b, 0] - face[corner_a, 0]) / 2
        face[upper, :2] = (middle, 0.38)
        face[lower, :2] = (middle, 0.42)
        face[iris, :2] = (middle + iris_shift * half_width, 0.40)
    return face


def synthetic_pose(shift: float = 0.0, wrist_shift: float = 0.0) -> np.ndarray:
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[:, 3] = 0.9
    pose[lm.LEFT_SHOULDER, :2] = (0.65 + shift, 0.60)
    pose[lm.RIGHT_SHOULDER, :2] = (0.35 + shift, 0.60)
    pose[lm.LEFT_ELBOW, :2] = (0.70 + shift, 0.80)
    pose[lm.RIGHT_ELBOW, :2] = (0.30 + shift, 0.80)
    pose[lm.LEFT_WRIST, :2] = (0.68 + wrist_shift, 0.95)
    pose[lm.RIGHT_WRIST, :2] = (0.32 + wrist_shift, 0.95)
    return pose


@pytest.mark.parametrize("yaw,pitch,roll", [(0, 0, 0), (20, 0, 0), (-15, 0, 0), (0, 12, 0), (0, 0, 10), (15, -8, 5)])
def test_head_pose_recovers_known_rotation(yaw: float, pitch: float, roll: float) -> None:
    estimate = lm.head_pose(synthetic_face(yaw, pitch, roll), WIDTH, HEIGHT)
    assert estimate == pytest.approx((yaw, pitch, roll), abs=1.5)


def test_gaze_proxy() -> None:
    assert lm.gaze_proxy(synthetic_face()) == pytest.approx((0.0, 0.0), abs=1e-4)
    x, y = lm.gaze_proxy(synthetic_face(iris_shift=0.5))
    assert x == pytest.approx(0.5, abs=1e-3) and y == pytest.approx(0.0, abs=1e-4)
    no_iris = synthetic_face()[:468]
    assert all(math.isnan(v) for v in lm.gaze_proxy(no_iris))


def test_static_features_without_detections() -> None:
    features = lm.static_frame_features(lm.FrameLandmarks(0.0, None, None), WIDTH, HEIGHT)
    assert features["face_visible"] == 0.0 and features["landmark_confidence"] == 0.0
    assert math.isnan(features["head_yaw"]) and math.isnan(features["left_shoulder_x"])


def _video(frames: list[lm.FrameLandmarks]) -> lm.VideoLandmarks:
    return lm.VideoLandmarks(frames, WIDTH, HEIGHT, 30.0)


def test_feature_table_schema_and_motion() -> None:
    frames = [
        lm.FrameLandmarks(0.0, synthetic_face(), synthetic_pose()),
        lm.FrameLandmarks(0.2, synthetic_face(yaw=10), synthetic_pose(shift=0.03)),
        lm.FrameLandmarks(0.4, None, synthetic_pose(shift=0.03, wrist_shift=0.06)),
        lm.FrameLandmarks(0.6, synthetic_face(), None),
    ]
    table = build_feature_table(_video(frames))
    assert tuple(table.columns) == VIDEO_FEATURE_COLUMNS
    assert len(table) == 4 and str(table.dtypes.iloc[0]) == "float32"
    assert table["face_visible"].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert table["shoulder_angle"].iloc[0] == pytest.approx(0.0, abs=1e-3)
    assert table["landmark_confidence"].iloc[0] == pytest.approx(0.9)
    # Whole upper body moved 0.03 in 0.2 s with shoulder width 0.3 -> 0.5 widths/s.
    assert math.isnan(table["upper_body_movement"].iloc[0])
    assert table["upper_body_movement"].iloc[1] == pytest.approx(0.5, rel=0.02)
    assert table["upper_body_movement"].iloc[2] == pytest.approx(0.0, abs=1e-4)
    assert table["gesture_motion"].iloc[2] > 0.5  # only the wrists moved
    assert math.isnan(table["gesture_motion"].iloc[3])  # pose lost

    sequence = table_to_sequence(table)
    assert sequence.shape == (4, len(VIDEO_MODEL_FEATURES)) and sequence.dtype == np.float32


def test_summary_metrics_are_raw_measurements() -> None:
    frames = [lm.FrameLandmarks(i * 0.2, synthetic_face(iris_shift=0.0 if i < 6 else 0.8), synthetic_pose()) for i in range(10)]
    frames[3] = lm.FrameLandmarks(0.6, None, synthetic_pose())
    summary = summarize_table(build_feature_table(_video(frames)))
    assert summary["frame_count"] == 10
    assert summary["duration_sec"] == pytest.approx(1.8)
    assert summary["face_visible_ratio"] == pytest.approx(0.9)
    assert summary["gaze_center_ratio"] == pytest.approx(5 / 9)
    assert summary["head_yaw_std"] == pytest.approx(0.0, abs=0.5)
    assert summary["upper_body_movement_mean"] == pytest.approx(0.0, abs=1e-4)
    assert summary["gesture_events_per_min"] == 0.0
    assert summarize_table(build_feature_table(_video([]))) == {"frame_count": 0.0}


@pytest.mark.parametrize("suffix", [".npy", ".csv"])
def test_feature_table_roundtrip(tmp_path: Path, suffix: str) -> None:
    table = build_feature_table(_video([lm.FrameLandmarks(0.0, synthetic_face(), synthetic_pose())] * 3))
    path = tmp_path / f"sequence{suffix}"
    save_feature_table(table, path)
    loaded = load_feature_table(path)
    assert tuple(loaded.columns) == VIDEO_FEATURE_COLUMNS
    np.testing.assert_allclose(loaded.to_numpy(), table.to_numpy(), equal_nan=True)


def test_frame_sampling_from_real_file(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot write MJPG video")
    for index in range(30):
        writer.write(np.full((48, 64, 3), index * 8, dtype=np.uint8))
    writer.release()

    sampled = list(lm.sample_frames(path, target_fps=5.0))
    assert len(sampled) == 5  # every 6th of 30 frames
    timestamps = [timestamp for timestamp, _, _ in sampled]
    assert timestamps == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert sampled[0][1].shape == (48, 64, 3)
    assert len(list(lm.sample_frames(path, target_fps=5.0, max_frames=2))) == 2


def test_corrupted_video_errors(tmp_path: Path) -> None:
    with pytest.raises(CorruptedMediaError, match="not found"):
        list(lm.sample_frames(tmp_path / "missing.mp4"))
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(CorruptedMediaError):
        list(lm.sample_frames(broken))


def test_missing_mediapipe_model_is_explained(tmp_path: Path) -> None:
    with pytest.raises(MissingDependencyError) as error:
        lm.LandmarkExtractor(tmp_path / "face_landmarker.task", tmp_path / "pose_landmarker.task")
    assert "not downloaded automatically" in str(error.value)
    assert "https://storage.googleapis.com/mediapipe-models/" in str(error.value)
