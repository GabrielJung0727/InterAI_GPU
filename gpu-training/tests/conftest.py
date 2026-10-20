"""Shared test helpers and fixtures.

Tests never touch real datasets: they build tiny temporary fixtures. Any
"training" here only checks that the pipeline runs end to end — it says
nothing about model quality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SAMPLE_RATE = 16000


def load_torch() -> Any:
    """Import torch or skip the calling test module.

    PyTorch may be installed but unloadable (e.g. Windows application-control
    policies blocking its DLLs raise ``OSError``), so both cases are handled.
    """
    try:
        import torch
    except (ImportError, OSError) as exc:
        pytest.skip(f"PyTorch cannot be loaded in this environment: {exc}", allow_module_level=True)
    return torch


def tone(seconds: float, frequency: float = 220.0, amplitude: float = 0.5) -> np.ndarray:
    """A sine tone standing in for voiced speech."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def write_wav(path: Path, wave: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave, sample_rate)
    return path


def base_config(tmp_path: Path, task: str, processed_dir: Path, model: dict, data: dict | None = None,
                training: dict | None = None, checkpoint: dict | None = None) -> dict:
    """Minimal CPU config writing everything under ``tmp_path``."""
    return {
        "experiment": {
            "name": f"test_{task}", "task": task, "seed": 0,
            "output_dir": str(tmp_path / "outputs"), "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
        "data": {"processed_dir": str(processed_dir), **(data or {})},
        "model": model,
        "training": {
            "batch_size": 4, "epochs": 2, "lr": 1e-3, "num_workers": 0,
            "early_stopping_patience": 0, **(training or {}),
        },
        "device": {"type": "cpu"},
        "checkpoint": checkpoint or {"metric": "val_loss", "mode": "min"},
    }


@pytest.fixture
def aihub_interview_dir(tmp_path: Path) -> Path:
    """Three records in the assumed AI Hub interview layout (one without audio, one empty)."""
    root = tmp_path / "aihub-interview"
    for index, (answer, with_audio) in enumerate(
        [("저는 백엔드 개발자로 3년간 일했습니다.", True), ("팀 프로젝트에서 API 설계를 맡았습니다.", False), ("", True)]
    ):
        stem = f"ckmk_d_ict_f_e_{index:05d}"
        record = {
            "dataSet": {
                "info": {"occupation": "ICT", "experience": "EXPERIENCED", "gender": "FEMALE"},
                "question": {"raw": {"text": "자기소개를 해주세요."}, "intent": [{"category": "background"}]},
                "answer": {
                    "raw": {"text": answer},
                    "intent": [{"category": "experience"}],
                    "summary": {"text": "경력 요약"},
                },
            }
        }
        (root / "labels").mkdir(parents=True, exist_ok=True)
        (root / "labels" / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        if with_audio:
            write_wav(root / "audio" / f"{stem}.wav", np.concatenate([tone(0.6), silence(0.4), tone(0.6)]))
    return root


@pytest.fixture
def mit_interview_dir(tmp_path: Path) -> Path:
    """Six participants with pre/post interviews and aggregated behaviour ratings."""
    root = tmp_path / "mit-interview"
    rows = ["Participant,Worker,Overall,RecommendHiring,EyeContact,SpeakingRate,NoFillers,Paused,EngagingTone,StructuredAnswers,Focused"]
    for participant in range(1, 7):
        for prefix in ("P", "PP"):
            name = f"{prefix}{participant}"
            wave = np.concatenate([tone(0.5, 180 + 10 * participant), silence(0.3 + 0.05 * participant), tone(0.5)])
            write_wav(root / "Audio" / f"{name}.wav", wave)
            score = 3 + participant * 0.3
            rows.append(f"{name.lower()},AGGR,{score},{score},{score},{score + 0.1},{score},{score - 0.2},{score},{score},{score}")
            rows.append(f"{name.lower()},W1,1,1,1,1,1,1,1,1,1")
    (root / "Labels").mkdir(parents=True)
    (root / "Labels" / "turker_scores_full_interview.csv").write_text("\n".join(rows), encoding="utf-8")
    return root


@pytest.fixture
def internal_dir(tmp_path: Path) -> Path:
    """Service data: consented single answers, one non-consented record, before/after pairs."""
    root = tmp_path / "internal"
    root.mkdir()
    records: list[dict[str, Any]] = [
        {
            "sample_id": "no_consent", "candidate_id": "c0", "question": "Q?", "answer": "A.",
            "scores": {"relevance": 2}, "consent": {"opt_in": False},
        }
    ]
    for i in range(1, 7):
        records.append(
            {
                "sample_id": f"single_{i}", "candidate_id": f"c{i}", "question": "프로젝트 경험을 말해주세요.",
                "answer": f"결제 시스템 지연을 {i * 10}% 줄였습니다.", "position": "Backend Engineer",
                "interview_type": "technical", "scores": {"relevance": 1 + i * 0.5, "structure": 0.5 + i * 0.5},
                "consent": {"opt_in": True, "consent_version": "v1"},
            }
        )
        records.append(
            {
                "pair_id": f"pair_{i}", "candidate_id": f"c{i}", "question": "갈등 해결 경험은?",
                "before_answer": "잘 해결했습니다.", "feedback": "구체적 사례를 추가하세요.",
                "after_answer": f"스프린트 {i}에서 일정 충돌을 조율했습니다.",
                "before_scores": {"relevance": 2, "specificity": 1}, "after_scores": {"relevance": 3, "specificity": 3},
                "consent": {"opt_in": True},
            }
        )
    with (root / "export.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return root
