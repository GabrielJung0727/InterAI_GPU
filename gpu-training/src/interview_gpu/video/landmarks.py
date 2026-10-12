"""MediaPipe landmark extraction and per-frame geometry (CPU preprocessing).

    Raw video -> frame sampling -> MediaPipe Face/Pose Landmarker -> FrameLandmarks

MediaPipe runs here as a *preprocessing* step; the GPU only ever sees the
resulting landmark features, never raw frames.

The MediaPipe Tasks API needs two model files that are not bundled with the
package. They are never downloaded automatically — place them under
``models/mediapipe/`` (see ``MODEL_URLS``) or pass explicit paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from interview_gpu.common.config import resolve_path
from interview_gpu.common.errors import CorruptedMediaError, MissingDependencyError, require

DEFAULT_MODEL_DIR = "models/mediapipe"
FACE_MODEL_FILE = "face_landmarker.task"
POSE_MODEL_FILE = "pose_landmarker.task"
MODEL_URLS = {
    FACE_MODEL_FILE: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    POSE_MODEL_FILE: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

# --- Face mesh landmark indices (MediaPipe canonical face model, 478 points) ---
NOSE_TIP, CHIN = 1, 152
EYE_OUTER_IMAGE_LEFT, EYE_OUTER_IMAGE_RIGHT = 33, 263
MOUTH_IMAGE_LEFT, MOUTH_IMAGE_RIGHT = 61, 291
#: (iris centre, corner a, corner b, upper lid, lower lid) per eye.
EYES = ((468, 33, 133, 159, 145), (473, 362, 263, 386, 374))
IRIS_LANDMARK_COUNT = 478

# --- Pose landmark indices (33 points) ---
POSE_NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
UPPER_BODY = (POSE_NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)

#: Generic 3D face model (arbitrary units, x right, y down, z away from camera)
#: for the six landmarks above, used to estimate head pose with solvePnP.
_FACE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # nose tip
        (0.0, 330.0, 65.0),  # chin
        (-225.0, -170.0, 135.0),  # eye outer corner, image left
        (225.0, -170.0, 135.0),  # eye outer corner, image right
        (-150.0, 150.0, 125.0),  # mouth corner, image left
        (150.0, 150.0, 125.0),  # mouth corner, image right
    ],
    dtype=np.float64,
)
_FACE_MODEL_INDICES = (NOSE_TIP, CHIN, EYE_OUTER_IMAGE_LEFT, EYE_OUTER_IMAGE_RIGHT, MOUTH_IMAGE_LEFT, MOUTH_IMAGE_RIGHT)


@dataclass
class FrameLandmarks:
    """Landmarks of one sampled frame (normalised image coordinates)."""

    timestamp: float
    #: ``[478, 3]`` (x, y, z) or ``None`` when no face was detected.
    face: np.ndarray | None
    #: ``[33, 4]`` (x, y, z, visibility) or ``None`` when no body was detected.
    pose: np.ndarray | None


@dataclass
class VideoLandmarks:
    """All sampled frames of one video."""

    frames: list[FrameLandmarks]
    width: int
    height: int
    source_fps: float


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics assuming focal length ≈ image width."""
    return np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)


def head_pose(face: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    """Approximate head (yaw, pitch, roll) in degrees from face landmarks.

    Solves a PnP problem between six landmarks and a generic 3D face model.
    A frontal face gives roughly (0, 0, 0); yaw rotates about the vertical
    axis, pitch about the horizontal axis, roll in the image plane. These are
    geometric proxies, not calibrated measurements.
    """
    import cv2

    points = face[list(_FACE_MODEL_INDICES), :2] * np.array([width, height], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(
        _FACE_MODEL_POINTS, points, camera_matrix(width, height), None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return (math.nan, math.nan, math.nan)
    rotation, _ = cv2.Rodrigues(rvec)
    sy = math.hypot(rotation[0, 0], rotation[1, 0])
    yaw = math.degrees(math.atan2(-rotation[2, 0], sy))
    pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return (yaw, pitch, roll)


def gaze_proxy(face: np.ndarray) -> tuple[float, float]:
    """Iris position inside the eye opening, averaged over both eyes.

    Returns ``(x, y)`` in roughly ``[-1, 1]``; (0, 0) means the iris is
    centred. This is a proxy for gaze direction relative to the head, not a
    calibrated gaze estimate. NaN when iris landmarks are unavailable.
    """
    if face.shape[0] < IRIS_LANDMARK_COUNT:
        return (math.nan, math.nan)
    xs, ys = [], []
    for iris, corner_a, corner_b, upper, lower in EYES:
        width = face[corner_b, 0] - face[corner_a, 0]
        height = face[lower, 1] - face[upper, 1]
        if abs(width) > 1e-6:
            xs.append(((face[iris, 0] - face[corner_a, 0]) / width - 0.5) * 2.0)
        if abs(height) > 1e-6:
            ys.append(((face[iris, 1] - face[upper, 1]) / height - 0.5) * 2.0)
    return (float(np.mean(xs)) if xs else math.nan, float(np.mean(ys)) if ys else math.nan)


def static_frame_features(frame: FrameLandmarks, width: int, height: int) -> dict[str, float]:
    """Features computable from a single frame (no motion)."""
    nan = math.nan
    features = {
        "timestamp": frame.timestamp,
        "head_yaw": nan, "head_pitch": nan, "head_roll": nan,
        "face_visible": 0.0, "face_center_x": nan, "face_center_y": nan,
        "gaze_proxy_x": nan, "gaze_proxy_y": nan,
        "left_shoulder_x": nan, "left_shoulder_y": nan,
        "right_shoulder_x": nan, "right_shoulder_y": nan,
        "shoulder_angle": nan,
        "landmark_confidence": 0.0,
    }
    if frame.face is not None:
        yaw, pitch, roll = head_pose(frame.face, width, height)
        gaze_x, gaze_y = gaze_proxy(frame.face)
        features.update(
            head_yaw=yaw, head_pitch=pitch, head_roll=roll, face_visible=1.0,
            face_center_x=float(frame.face[:, 0].mean()), face_center_y=float(frame.face[:, 1].mean()),
            gaze_proxy_x=gaze_x, gaze_proxy_y=gaze_y, landmark_confidence=1.0,
        )
    if frame.pose is not None:
        left, right = frame.pose[LEFT_SHOULDER], frame.pose[RIGHT_SHOULDER]
        features.update(
            left_shoulder_x=float(left[0]), left_shoulder_y=float(left[1]),
            right_shoulder_x=float(right[0]), right_shoulder_y=float(right[1]),
            # 0° = level shoulders (the subject's left shoulder appears on the image right).
            shoulder_angle=math.degrees(math.atan2((left[1] - right[1]) * height, (left[0] - right[0]) * width)),
            landmark_confidence=float(frame.pose[list(UPPER_BODY), 3].mean()),
        )
    return features


def sample_frames(
    path: str | Path, target_fps: float = 5.0, max_frames: int | None = None
) -> Iterator[tuple[float, np.ndarray, float]]:
    """Yield ``(timestamp_sec, rgb_frame, source_fps)`` at roughly ``target_fps``.

    Raises:
        CorruptedMediaError: The file is missing, cannot be opened or has no frames.
    """
    import cv2

    path = Path(path)
    if not path.is_file():
        raise CorruptedMediaError(f"Video file not found: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise CorruptedMediaError(f"Could not open video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and 1.0 <= fps <= 240.0 else 30.0
        step = max(1, round(fps / target_fps))
        index = produced = 0
        while max_frames is None or produced < max_frames:
            if index % step:
                if not capture.grab():
                    break
            else:
                ok, frame = capture.read()
                if not ok:
                    break
                yield index / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), fps
                produced += 1
            index += 1
        if produced == 0:
            raise CorruptedMediaError(f"Video contains no decodable frames: {path}")
    finally:
        capture.release()


class LandmarkExtractor:
    """Runs MediaPipe Face + Pose Landmarker over sampled video frames.

    Args:
        face_model: Path to ``face_landmarker.task`` (default: ``models/mediapipe/``).
        pose_model: Path to ``pose_landmarker.task`` (default: ``models/mediapipe/``).
    """

    def __init__(self, face_model: str | Path | None = None, pose_model: str | Path | None = None) -> None:
        self._mp = require("mediapipe", "uv sync")
        model_dir = resolve_path(DEFAULT_MODEL_DIR)
        self.face_model = Path(face_model) if face_model else model_dir / FACE_MODEL_FILE
        self.pose_model = Path(pose_model) if pose_model else model_dir / POSE_MODEL_FILE
        for path, key in ((self.face_model, FACE_MODEL_FILE), (self.pose_model, POSE_MODEL_FILE)):
            if not path.is_file():
                raise MissingDependencyError(
                    f"MediaPipe model file not found: {path.as_posix()}\n"
                    "Model files are not downloaded automatically. Download it once from\n"
                    f"  {MODEL_URLS[key]}\n"
                    f"and save it as {path.as_posix()} (or pass an explicit path)."
                )

    def process_video(
        self, path: str | Path, target_fps: float = 5.0, max_frames: int | None = None
    ) -> VideoLandmarks:
        """Extract landmarks for one video.

        Raises:
            CorruptedMediaError: The video cannot be decoded.
        """
        mp = self._mp
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        face_options = vision.FaceLandmarkerOptions(
            base_options=base(model_asset_path=str(self.face_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base(model_asset_path=str(self.pose_model)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        frames: list[FrameLandmarks] = []
        width = height = 0
        source_fps = 0.0
        # VIDEO mode requires strictly increasing timestamps per instance, so
        # landmarkers are created per video.
        face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        try:
            for timestamp, rgb, source_fps in sample_frames(path, target_fps, max_frames):
                height, width = rgb.shape[:2]
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                millis = int(round(timestamp * 1000))
                face_result = face_landmarker.detect_for_video(image, millis)
                pose_result = pose_landmarker.detect_for_video(image, millis)
                face = pose = None
                if face_result.face_landmarks:
                    face = np.array([(p.x, p.y, p.z) for p in face_result.face_landmarks[0]], dtype=np.float32)
                if pose_result.pose_landmarks:
                    pose = np.array(
                        [(p.x, p.y, p.z, p.visibility or 0.0) for p in pose_result.pose_landmarks[0]],
                        dtype=np.float32,
                    )
                frames.append(FrameLandmarks(timestamp, face, pose))
        finally:
            face_landmarker.close()
            pose_landmarker.close()
        return VideoLandmarks(frames, width, height, source_fps)
