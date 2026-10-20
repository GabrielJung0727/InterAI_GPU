"""Stage 1 — handcrafted, observable speech features (CPU only).

Everything here is a direct measurement of the signal or the transcript:
pauses, pitch, energy, speaking rate, filler words. No psychological state is
inferred.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from interview_gpu.common.errors import CorruptedMediaError
from interview_gpu.schemas.speech import SPEECH_FEATURE_NAMES

FRAME_LENGTH = 2048
HOP_LENGTH = 512
#: Silence detection uses a 25 ms window / 10 ms hop (at 16 kHz). The 2048-sample
#: analysis window smears speech ~64 ms into each side of a pause, which made
#: every pause read ~0.13 s short and dropped pauses just above ``min_pause_sec``.
SILENCE_FRAME_LENGTH = 400
SILENCE_HOP_LENGTH = 160

_HANGUL = re.compile(r"[가-힣]")
_LATIN_WORD = re.compile(r"[A-Za-z0-9]+")
_STRIP = re.compile(r"^[\W_]+|[\W_]+$")
#: Hesitation sounds, optionally elongated ("음음", "어어어").
_FILLER_SOUND = re.compile(r"^(음+|어+|아+|에+|으+|um+|uh+|er+|hmm+)$", re.IGNORECASE)
#: Filler words. Several are also legitimate words, so counts are a proxy.
FILLER_WORDS = frozenset({"그", "저", "뭐", "막", "그니까", "그러니까", "저기", "뭐랄까", "이제", "약간"})


@dataclass(frozen=True)
class SpeechFeatureConfig:
    """Thresholds for the handcrafted feature extractor."""

    sample_rate: int = 16000
    #: Frames quieter than ``top_db`` below the peak count as silence.
    top_db: float = 35.0
    min_pause_sec: float = 0.25
    long_pause_sec: float = 1.0
    pitch_fmin: float = 65.0
    pitch_fmax: float = 400.0
    min_duration_sec: float = 0.2


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Load an audio file as mono float32 at ``sample_rate``.

    Raises:
        CorruptedMediaError: Missing, undecodable, empty or non-finite audio.
    """
    import soundfile as sf

    path = Path(path)
    if not path.is_file():
        raise CorruptedMediaError(f"Audio file not found: {path}")
    try:
        data, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise CorruptedMediaError(f"Could not decode audio {path}: {exc}") from exc
    wave = data.mean(axis=1)
    if wave.size == 0 or not np.isfinite(wave).all():
        raise CorruptedMediaError(f"Audio is empty or contains NaN/inf: {path}")
    if source_rate != sample_rate:
        import librosa

        wave = librosa.resample(wave, orig_sr=source_rate, target_sr=sample_rate)
    return np.ascontiguousarray(wave, dtype=np.float32)


def count_syllables(text: str) -> int:
    """Hangul syllable blocks plus one unit per Latin/numeric token."""
    return len(_HANGUL.findall(text)) + len(_LATIN_WORD.findall(text))


def count_fillers(text: str) -> int:
    """Number of whitespace tokens that are hesitation sounds or filler words."""
    count = 0
    for token in text.split():
        token = _STRIP.sub("", token)
        if token and (_FILLER_SOUND.match(token) or token in FILLER_WORDS):
            count += 1
    return count


def transcript_features(transcript: str | None, duration: float, speaking_time: float) -> dict[str, float]:
    """Rate and filler features; NaN when no transcript is available."""
    names = (
        "speech_rate_syllables_per_sec",
        "articulation_rate_syllables_per_sec",
        "speech_rate_wpm",
        "filler_count",
        "filler_per_min",
    )
    if not transcript or not transcript.strip() or duration <= 0:
        return dict.fromkeys(names, float("nan"))
    syllables = count_syllables(transcript)
    fillers = count_fillers(transcript)
    return {
        "speech_rate_syllables_per_sec": syllables / duration,
        "articulation_rate_syllables_per_sec": syllables / speaking_time if speaking_time > 0 else float("nan"),
        "speech_rate_wpm": len(transcript.split()) / duration * 60.0,
        "filler_count": float(fillers),
        "filler_per_min": fillers / duration * 60.0,
    }


def acoustic_features(wave: np.ndarray, config: SpeechFeatureConfig) -> dict[str, float]:
    """Pause, pitch, energy and stability measurements from the waveform."""
    import librosa

    sr = config.sample_rate
    duration = len(wave) / sr
    if duration < config.min_duration_sec:
        raise CorruptedMediaError(f"Audio too short for analysis ({duration:.2f}s).")

    features = dict.fromkeys(SPEECH_FEATURE_NAMES[:15], 0.0)
    features["utterance_duration"] = duration
    for name in ("pitch_mean", "pitch_std", "pitch_cv", "energy_mean_db", "energy_std_db", "voice_stability_proxy"):
        features[name] = float("nan")

    peak = float(np.max(np.abs(wave)))
    if peak < 1e-5:  # digital silence
        features["response_latency"] = duration
        return features

    intervals = librosa.effects.split(
        wave, top_db=config.top_db, frame_length=SILENCE_FRAME_LENGTH, hop_length=SILENCE_HOP_LENGTH
    )
    if len(intervals) == 0:
        features["response_latency"] = duration
        return features

    gaps = (intervals[1:, 0] - intervals[:-1, 1]) / sr
    pauses = gaps[gaps >= config.min_pause_sec]
    speaking_time = float((intervals[:, 1] - intervals[:, 0]).sum() / sr)
    features.update(
        response_latency=float(intervals[0, 0] / sr),
        pause_count=float(len(pauses)),
        pause_total_duration=float(pauses.sum()),
        pause_mean_duration=float(pauses.mean()) if len(pauses) else 0.0,
        pause_max_duration=float(pauses.max()) if len(pauses) else 0.0,
        long_pause_count=float((pauses >= config.long_pause_sec).sum()),
        pause_rate_per_min=float(len(pauses) / duration * 60.0),
        speech_ratio=speaking_time / duration,
    )

    rms = librosa.feature.rms(y=wave, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    active = rms > rms.max() * 10 ** (-config.top_db / 20)
    if active.any():
        energy_db = 20 * np.log10(np.maximum(rms[active], 1e-10))
        features["energy_mean_db"] = float(energy_db.mean())
        features["energy_std_db"] = float(energy_db.std())

    f0 = librosa.yin(
        wave, fmin=config.pitch_fmin, fmax=config.pitch_fmax, sr=sr,
        frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
    )
    frames = min(len(f0), len(active))
    f0, voiced = f0[:frames], active[:frames].copy()
    # YIN saturates at the search bounds on unvoiced frames; drop those.
    voiced &= (f0 > config.pitch_fmin * 1.05) & (f0 < config.pitch_fmax * 0.95)
    if voiced.sum() >= 3:
        pitch = f0[voiced]
        features["pitch_mean"] = float(pitch.mean())
        features["pitch_std"] = float(pitch.std())
        features["pitch_cv"] = float(pitch.std() / pitch.mean())
        consecutive = voiced[1:] & voiced[:-1]
        if consecutive.any():
            jitter = np.abs(np.diff(f0))[consecutive] / pitch.mean()
            # 1.0 = perfectly steady pitch track; decreases with frame-to-frame jitter.
            features["voice_stability_proxy"] = float(1.0 / (1.0 + 10.0 * jitter.mean()))
    return features


def extract_speech_features(
    wave: np.ndarray,
    transcript: str | None = None,
    config: SpeechFeatureConfig | None = None,
    response_latency: float | None = None,
) -> dict[str, float]:
    """All Stage-1 features for one utterance, keyed by ``SPEECH_FEATURE_NAMES``.

    Args:
        wave: Mono waveform sampled at ``config.sample_rate``.
        transcript: Optional transcript (enables rate / filler features).
        config: Extraction thresholds.
        response_latency: Seconds between the end of the question and the
            start of the answer, if the service measured it. Defaults to the
            leading silence of the recording.
    """
    config = config or SpeechFeatureConfig()
    features = acoustic_features(np.asarray(wave, dtype=np.float32), config)
    speaking_time = features["speech_ratio"] * features["utterance_duration"]
    features.update(transcript_features(transcript, features["utterance_duration"], speaking_time))
    if response_latency is not None:
        features["response_latency"] = float(response_latency)
    return {name: float(features[name]) for name in SPEECH_FEATURE_NAMES}


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Order a feature dict into the model input vector."""
    return np.array([features[name] for name in SPEECH_FEATURE_NAMES], dtype=np.float32)
