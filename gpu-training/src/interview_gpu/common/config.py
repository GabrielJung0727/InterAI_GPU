"""YAML-driven, validated experiment configuration.

A config file may declare ``base: base.yaml`` to inherit from another file in
the same directory; values are deep-merged. Unknown keys are rejected so typos
fail loudly instead of being silently ignored.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from interview_gpu.common.errors import ConfigError

#: ``gpu-training/`` directory. Relative paths in configs resolve against it.
PROJECT_ROOT = Path(os.environ.get("INTERVIEW_GPU_ROOT", Path(__file__).resolve().parents[3]))


def resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` against the project root unless it is absolute."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentConfig(_Section):
    name: str = "experiment"
    #: Registry sub-folder: speech | video | answer | multimodal.
    task: str = "speech"
    seed: int = 42
    deterministic: bool = False
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"


class SplitConfig(_Section):
    val_ratio: float = Field(0.15, ge=0.0, lt=1.0)
    test_ratio: float = Field(0.15, ge=0.0, lt=1.0)
    #: Column used for group splitting (speaker / candidate / session).
    group_column: str = "group_id"


class DataConfig(_Section):
    processed_dir: str
    #: ``production`` refuses datasets whose license is research-only.
    usage_scope: Literal["research", "production"] = "research"
    task: Literal["regression", "classification"] = "regression"
    #: Either a list of label columns or a mapping ``output_name -> column``.
    target_columns: list[str] | dict[str, str] = Field(default_factory=list)
    split: SplitConfig = Field(default_factory=SplitConfig)
    #: Task-specific options (max sequence length, input mode, ...).
    params: dict[str, Any] = Field(default_factory=dict)


class ModelConfig(_Section):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ImbalanceConfig(_Section):
    """Class-imbalance handling for classification. All off by default."""

    class_weights: bool = False
    weighted_sampler: bool = False
    focal_loss: bool = False
    focal_gamma: float = 2.0


class TrainingConfig(_Section):
    batch_size: int = Field(32, ge=1)
    epochs: int = Field(20, ge=1)
    lr: float = Field(1e-4, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    gradient_accumulation_steps: int = Field(1, ge=1)
    mixed_precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    grad_clip_norm: float | None = 1.0
    scheduler: Literal["none", "cosine", "linear"] = "cosine"
    warmup_ratio: float = Field(0.0, ge=0.0, lt=1.0)
    loss: Literal["huber", "mse", "l1"] = "huber"
    early_stopping_patience: int = Field(5, ge=0)
    num_workers: int = Field(4, ge=0)
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = 2
    compile: bool = False
    log_every: int = Field(50, ge=1)
    imbalance: ImbalanceConfig = Field(default_factory=ImbalanceConfig)


class DeviceConfig(_Section):
    #: auto | cuda | cuda:N | mps | cpu
    type: str = "auto"


class CheckpointConfig(_Section):
    save_best_only: bool = True
    #: ``val_loss`` or any flattened validation metric, e.g. ``val_mean_mae``.
    metric: str = "val_loss"
    mode: Literal["min", "max"] = "min"


class LoggingConfig(_Section):
    wandb: bool = False
    wandb_project: str = "interview-gpu-training"


class Config(_Section):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain-python representation (safe to dump to YAML / checkpoints)."""
        return self.model_dump(mode="json")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    base_name = data.pop("base", None)
    if base_name:
        data = _deep_merge(_read_yaml(path.parent / base_name), data)
    return data


def _apply_override(data: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ConfigError(f"Override must look like 'section.key=value', got: {override}")
    dotted, raw = override.split("=", 1)
    keys = dotted.strip().split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigError(f"Cannot override '{dotted}': '{key}' is not a section")
    node[keys[-1]] = yaml.safe_load(raw)


def config_from_dict(data: dict[str, Any]) -> Config:
    """Validate a raw dictionary into a :class:`Config`."""
    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration:\n{exc}") from exc


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config, apply ``base`` inheritance and CLI overrides.

    Args:
        path: YAML file (relative paths resolve against the project root).
        overrides: Items such as ``"training.batch_size=8"``.
    """
    data = _read_yaml(resolve_path(path))
    for override in overrides or []:
        _apply_override(data, override)
    return config_from_dict(data)


def save_config(config: Config | dict[str, Any], path: str | Path) -> None:
    """Write a config to YAML."""
    payload = config.to_dict() if isinstance(config, Config) else config
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
