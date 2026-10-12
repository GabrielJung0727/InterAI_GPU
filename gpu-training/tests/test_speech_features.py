"""Handcrafted speech feature extraction (CPU)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import SAMPLE_RATE, silence, tone, write_wav
from interview_gpu.common.errors import CorruptedMediaError
from interview_gpu.schemas.speech import SPEECH_FEATURE_NAMES
from interview_gpu.speech.features import (
    count_fillers,
    count_syllables,
    extract_speech_features,
    features_to_vector,
    load_audio,
)


def test_pause_pitch_and_energy_measurements() -> None:
    wave = np.concatenate([tone(1.0, 220.0), silence(0.6), tone(1.0, 220.0)])
    features = extract_speech_features(wave)
    assert tuple(features) == SPEECH_FEATURE_NAMES
    assert features["utterance_duration"] == pytest.approx(2.6, abs=0.01)
    assert features["pause_count"] == 1
    assert features["pause_total_duration"] == pytest.approx(0.6, abs=0.15)
    assert features["long_pause_count"] == 0
    assert features["speech_ratio"] == pytest.approx(2.0 / 2.6, abs=0.1)
    assert features["response_latency"] < 0.1
    assert features["pitch_mean"] == pytest.approx(220.0, abs=10.0)
    assert features["voice_stability_proxy"] > 0.8  # a pure tone is a steady pitch track
    assert math.isfinite(features["energy_mean_db"])
    # Transcript-dependent features are NaN (not zero) without a transcript.
    assert math.isnan(features["speech_rate_wpm"]) and math.isnan(features["filler_count"])


def test_long_pause_and_leading_silence() -> None:
    wave = np.concatenate([silence(0.5), tone(0.8), silence(1.4), tone(0.8)])
    features = extract_speech_features(wave)
    assert features["long_pause_count"] == 1
    assert features["response_latency"] == pytest.approx(0.5, abs=0.1)
    assert extract_speech_features(wave, response_latency=2.5)["response_latency"] == 2.5


def test_transcript_features() -> None:
    wave = np.concatenate([tone(2.0), silence(0.5), tone(2.5)])  # 5 seconds
    transcript = "음 저는 백엔드 개발자로 일했습니다 어 그리고 API 설계를 했습니다"
    features = extract_speech_features(wave, transcript)
    assert features["filler_count"] == 2
    assert features["filler_per_min"] == pytest.approx(24.0, rel=0.01)
    assert features["speech_rate_wpm"] == pytest.approx(len(transcript.split()) / 5.0 * 60.0, rel=0.01)
    assert features["speech_rate_syllables_per_sec"] == pytest.approx(count_syllables(transcript) / 5.0, rel=0.01)
    assert features["articulation_rate_syllables_per_sec"] > features["speech_rate_syllables_per_sec"]
    vector = features_to_vector(features)
    assert vector.shape == (len(SPEECH_FEATURE_NAMES),) and vector.dtype == np.float32


def test_text_counters() -> None:
    assert count_syllables("안녕하세요 API 3개") == 5 + 1 + 1 + 1
    assert count_fillers("음... 어, 그러니까 저는 um 개발자입니다") == 4
    assert count_fillers("음식과 어머니") == 0  # substrings of real words are not fillers


def test_silent_and_too_short_audio() -> None:
    features = extract_speech_features(silence(1.0))
    assert features["speech_ratio"] == 0 and features["pause_count"] == 0
    assert math.isnan(features["pitch_mean"])
    with pytest.raises(CorruptedMediaError, match="too short"):
        extract_speech_features(tone(0.05))


def test_load_audio_resamples_and_downmixes(tmp_path: Path) -> None:
    import soundfile as sf

    stereo = np.stack([tone(1.0), tone(1.0)], axis=1)[::2]  # 8 kHz stereo
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, 8000)
    wave = load_audio(path, SAMPLE_RATE)
    assert wave.ndim == 1 and wave.dtype == np.float32
    assert len(wave) == pytest.approx(SAMPLE_RATE, abs=2)


def test_load_audio_errors(tmp_path: Path) -> None:
    with pytest.raises(CorruptedMediaError, match="not found"):
        load_audio(tmp_path / "missing.wav")
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"definitely not a wav file")
    with pytest.raises(CorruptedMediaError, match="Could not decode"):
        load_audio(broken)
    empty = write_wav(tmp_path / "empty.wav", np.zeros(0, dtype=np.float32))
    with pytest.raises(CorruptedMediaError, match="empty"):
        load_audio(empty)
