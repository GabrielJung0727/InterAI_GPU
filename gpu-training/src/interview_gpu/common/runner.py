"""End-to-end experiment orchestration shared by all tasks.

Each task only supplies two callables:

* ``build_data(cfg) -> DataBundle``   (datasets, task spec, normalisation)
* ``build_model(cfg, bundle) -> nn.Module``

Everything else — logging, seeding, device/precision selection, loaders,
training, test evaluation, registry and experiment outputs — is common.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from interview_gpu.common.checkpoint import load_checkpoint, load_model_state, next_version_dir
from interview_gpu.common.config import Config, config_from_dict, resolve_path, save_config
from interview_gpu.common.device import (
    estimate_training_memory_mb,
    log_environment,
    resolve_precision,
    select_device,
)
from interview_gpu.common.errors import CheckpointMismatchError, EmptyDatasetError, UsageScopeError
from interview_gpu.common.io import read_json, write_json, write_table
from interview_gpu.common.logging import setup_logging
from interview_gpu.common.seed import make_generator, seed_everything, seed_worker
from interview_gpu.common.tasks import (
    TaskSpec,
    build_loss_fn,
    build_metric_fn,
    sample_weights,
    summarize_predictions,
)
from interview_gpu.common.trainer import EvalResult, Trainer, oom_guard


@dataclass
class DataBundle:
    """Everything a task hands to the shared runner."""

    train: Dataset
    val: Dataset
    test: Dataset | None
    task: TaskSpec
    collate_fn: Callable | None = None
    #: Feature normalisation parameters (stored in the checkpoint).
    normalization: dict[str, Any] = field(default_factory=dict)
    #: Manifest of the processed dataset (source, license, usage_scope, ...).
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    label_mapping: dict[str, Any] = field(default_factory=dict)
    class_counts: np.ndarray | None = None
    #: Integer labels of the training split (classification only, for sampling).
    train_labels: np.ndarray | None = None
    #: Free-form values the model builder needs (input_dim, ...).
    info: dict[str, Any] = field(default_factory=dict)


BuildData = Callable[[Config], DataBundle]
BuildModel = Callable[[Config, DataBundle], torch.nn.Module]


def load_manifest(processed_dir: Path) -> dict[str, Any]:
    """Read ``manifest.json`` written by the preprocessing scripts (if any)."""
    path = processed_dir / "manifest.json"
    return read_json(path) if path.is_file() else {}


def check_usage_scope(dataset_metadata: dict[str, Any], requested_scope: str) -> None:
    """Refuse to train a production model on research-only data."""
    data_scope = dataset_metadata.get("usage_scope", "research")
    if requested_scope == "production" and data_scope != "production":
        raise UsageScopeError(
            f"data.usage_scope is 'production' but the processed dataset "
            f"'{dataset_metadata.get('dataset_name', 'unknown')}' is marked '{data_scope}'.\n"
            "Research-only datasets are never mixed into production training. "
            "Use a dataset whose license allows it, or set data.usage_scope: research."
        )


def make_loader(
    dataset: Dataset,
    cfg: Config,
    device: torch.device,
    *,
    shuffle: bool,
    collate_fn: Callable | None = None,
    sampler: torch.utils.data.Sampler | None = None,
) -> DataLoader:
    """Build a DataLoader from the performance options in the config."""
    training = cfg.training
    kwargs: dict[str, Any] = {
        "batch_size": training.batch_size,
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": training.num_workers,
        "pin_memory": training.pin_memory and device.type == "cuda",
        "collate_fn": collate_fn,
        "worker_init_fn": seed_worker,
        "generator": make_generator(cfg.experiment.seed),
    }
    if training.num_workers > 0:
        kwargs["persistent_workers"] = training.persistent_workers
        if training.prefetch_factor is not None:
            kwargs["prefetch_factor"] = training.prefetch_factor
    return DataLoader(dataset, **kwargs)


def _write_eval_outputs(directory: Path, task: TaskSpec, result: EvalResult) -> dict[str, Any]:
    """Save predictions (+ confusion matrix) and return metrics incl. calibration."""
    directory.mkdir(parents=True, exist_ok=True)
    calibration, predictions = summarize_predictions(
        task, result.outputs, result.targets, result.mask, result.sample_ids
    )
    write_table(directory / "predictions.csv", predictions)
    if task.kind == "classification":
        _plot_confusion_matrix(directory / "confusion_matrix.png", task, predictions)
    return {"loss": result.loss, **result.metrics, **calibration}


def _plot_confusion_matrix(path: Path, task: TaskSpec, predictions: Any) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        return
    display = ConfusionMatrixDisplay.from_predictions(
        predictions["true"], predictions["pred"], labels=task.class_names, xticks_rotation=45
    )
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=120)
    plt.close(display.figure_)


def run_training(
    cfg: Config, build_data: BuildData, build_model: BuildModel, device_override: str | None = None
) -> dict[str, Any]:
    """Train, evaluate on the held-out test split and register the model.

    Returns:
        JSON-serializable summary (paths, best epoch, validation/test metrics).
    """
    output_dir = resolve_path(cfg.experiment.output_dir) / cfg.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir / "train.log")
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)

    device = select_device(device_override or cfg.device.type)
    amp = resolve_precision(cfg.training.mixed_precision, device)
    environment = log_environment(logger, device, amp)

    bundle = build_data(cfg)
    check_usage_scope(bundle.dataset_metadata, cfg.data.usage_scope)
    if len(bundle.train) == 0 or len(bundle.val) == 0:
        raise EmptyDatasetError("Train or validation split is empty; add more data or adjust data.split.")
    logger.info(
        "Samples | train %d | val %d | test %d",
        len(bundle.train), len(bundle.val), len(bundle.test) if bundle.test is not None else 0,
    )

    with oom_guard(cfg.training.batch_size):
        model = build_model(cfg, bundle).to(device)
        memory = estimate_training_memory_mb(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            "Model %s | trainable params %s | weights %.1f MB | estimated minimum training VRAM %.1f MB "
            "(excludes activations)",
            cfg.model.name, f"{trainable:,}", memory["weights_mb"], memory["estimated_min_training_mb"],
        )

        sampler = None
        if (
            bundle.task.kind == "classification"
            and cfg.training.imbalance.weighted_sampler
            and bundle.train_labels is not None
            and bundle.class_counts is not None
        ):
            weights = sample_weights(bundle.train_labels, bundle.class_counts)
            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True, generator=make_generator(cfg.experiment.seed)
            )
        train_loader = make_loader(
            bundle.train, cfg, device, shuffle=True, collate_fn=bundle.collate_fn, sampler=sampler
        )
        val_loader = make_loader(bundle.val, cfg, device, shuffle=False, collate_fn=bundle.collate_fn)

        checkpoint_dir = next_version_dir(resolve_path(cfg.experiment.checkpoint_dir), cfg.experiment.task)
        save_config(cfg, output_dir / "config.yaml")
        trainer = Trainer(
            model=model,
            cfg=cfg,
            device=device,
            amp=amp,
            loss_fn=build_loss_fn(bundle.task, cfg.training, bundle.class_counts),
            metric_fn=build_metric_fn(bundle.task),
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_extras={
                "task": bundle.task.to_dict(),
                "label_mapping": bundle.label_mapping,
                "normalization": bundle.normalization,
                "dataset_metadata": bundle.dataset_metadata,
            },
        )
        fit_summary = trainer.fit(train_loader, val_loader)

        metrics: dict[str, Any] = {"validation": trainer.evaluate(val_loader).metrics}
        if bundle.test is not None and len(bundle.test) > 0:
            test_loader = make_loader(bundle.test, cfg, device, shuffle=False, collate_fn=bundle.collate_fn)
            metrics["test"] = _write_eval_outputs(output_dir, bundle.task, trainer.evaluate(test_loader))
        else:
            logger.warning("No test split: reporting validation metrics only.")

    summary = {
        "experiment": cfg.experiment.name,
        "task": cfg.experiment.task,
        "checkpoint_dir": checkpoint_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "split_sizes": {
            "train": len(bundle.train),
            "val": len(bundle.val),
            "test": len(bundle.test) if bundle.test is not None else 0,
        },
        "environment": environment,
        **fit_summary,
        "metrics": metrics,
    }
    write_json(output_dir / "metrics.json", summary)
    write_json(checkpoint_dir / "metrics.json", summary)
    logger.info("Best %s = %s (epoch %d)", fit_summary["best_metric_name"],
                fit_summary["best_metric_value"], fit_summary["best_epoch"])
    logger.info("Checkpoint: %s", checkpoint_dir)
    logger.info("Outputs:    %s", output_dir)
    return summary


def run_evaluation(
    checkpoint_dir: str | Path,
    build_data: BuildData,
    build_model: BuildModel,
    split: str = "test",
    device_override: str | None = None,
) -> dict[str, Any]:
    """Re-evaluate a registered checkpoint on one split of its dataset.

    The config stored next to the checkpoint is reused, so the split is
    reproduced deterministically from the same seed and group column.
    """
    checkpoint_dir = resolve_path(checkpoint_dir)
    payload = load_checkpoint(checkpoint_dir)
    cfg = config_from_dict(payload["config"])
    logger = setup_logging()
    seed_everything(cfg.experiment.seed, cfg.experiment.deterministic)
    device = select_device(device_override or cfg.device.type)
    amp = resolve_precision(cfg.training.mixed_precision, device)
    log_environment(logger, device, amp)

    bundle = build_data(cfg)
    saved_task = TaskSpec.from_dict(payload["task"])
    if (saved_task.target_names, saved_task.class_names) != (bundle.task.target_names, bundle.task.class_names):
        raise CheckpointMismatchError(
            "The processed dataset no longer matches the checkpoint (different targets/classes)."
        )
    if saved_task.to_dict() != bundle.task.to_dict():
        logger.warning("Target statistics differ from training time: the processed data has changed.")
    dataset = {"train": bundle.train, "val": bundle.val, "test": bundle.test}.get(split)
    if dataset is None or len(dataset) == 0:
        raise EmptyDatasetError(f"Split '{split}' is empty.")

    with oom_guard(cfg.training.batch_size):
        model = build_model(cfg, bundle).to(device)
        load_model_state(model, payload["model_state"])
        trainer = Trainer(
            model=model,
            cfg=cfg,
            device=device,
            amp=amp,
            loss_fn=build_loss_fn(saved_task, cfg.training, bundle.class_counts),
            metric_fn=build_metric_fn(saved_task),
            output_dir=checkpoint_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_extras={},
        )
        loader = make_loader(dataset, cfg, device, shuffle=False, collate_fn=bundle.collate_fn)
        metrics = _write_eval_outputs(checkpoint_dir / f"eval_{split}", saved_task, trainer.evaluate(loader))

    report = {"checkpoint_dir": checkpoint_dir.as_posix(), "split": split, "num_samples": len(dataset), "metrics": metrics}
    write_json(checkpoint_dir / f"eval_{split}" / "metrics.json", report)
    logger.info("Evaluation written to %s", checkpoint_dir / f"eval_{split}")
    return report
