"""Config loading, inheritance, overrides and validation."""

from __future__ import annotations

import pytest

from interview_gpu.common.config import PROJECT_ROOT, load_config, resolve_path, save_config
from interview_gpu.common.errors import ConfigError


@pytest.mark.parametrize("name", ["speech", "video", "answer", "multimodal"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(f"configs/{name}.yaml")
    assert cfg.experiment.task == name
    assert cfg.training.mixed_precision == "auto"  # inherited from base.yaml
    assert cfg.training.compile is False
    assert cfg.device.type == "auto"
    assert cfg.data.target_columns


def test_base_values_are_overridden_by_task_config() -> None:
    cfg = load_config("configs/answer.yaml")
    assert cfg.training.gradient_accumulation_steps == 2
    assert cfg.training.weight_decay == 0.01  # only defined in base.yaml
    assert cfg.checkpoint.metric == "val_mean_mae"


def test_cli_overrides() -> None:
    cfg = load_config("configs/speech.yaml", ["training.batch_size=8", "model.params.dropout=0.3", "device.type=cpu"])
    assert cfg.training.batch_size == 8
    assert cfg.model.params["dropout"] == 0.3
    assert cfg.device.type == "cpu"


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_config("configs/speech.yaml", ["training.batch_sizee=8"])


def test_invalid_value_is_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config("configs/speech.yaml", ["training.mixed_precision=fp8"])


def test_missing_file_and_bad_override() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config("configs/does_not_exist.yaml")
    with pytest.raises(ConfigError, match="section.key=value"):
        load_config("configs/speech.yaml", ["no_equals_sign"])


def test_roundtrip_and_path_resolution(tmp_path) -> None:
    cfg = load_config("configs/video.yaml")
    save_config(cfg, tmp_path / "config.yaml")
    assert load_config(tmp_path / "config.yaml").to_dict() == cfg.to_dict()
    assert resolve_path("data/raw") == PROJECT_ROOT / "data/raw"
    assert resolve_path(tmp_path) == tmp_path
