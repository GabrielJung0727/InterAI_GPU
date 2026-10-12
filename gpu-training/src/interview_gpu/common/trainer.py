"""The single training loop shared by every model.

Features: mixed precision (bf16 / fp16 + GradScaler), gradient accumulation,
gradient clipping, warmup + cosine/linear schedule, early stopping, best-model
checkpointing, NaN-loss detection, per-epoch VRAM statistics, history.csv,
loss curve, optional ``torch.compile`` and optional Weights & Biases.

Batches are dictionaries::

    {"inputs": {name: tensor, ...},   # forwarded as model(**inputs)
     "targets": tensor,
     "mask": tensor,                  # 1 = label observed
     "sample_id": list[str]}
"""

from __future__ import annotations

import csv
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from interview_gpu.common.checkpoint import MODEL_FILE, load_checkpoint, load_model_state, save_checkpoint
from interview_gpu.common.config import Config
from interview_gpu.common.device import AmpSettings, memory_stats, reset_peak_memory
from interview_gpu.common.errors import ConfigError, GpuOutOfMemoryError, NaNLossError
from interview_gpu.common.io import write_json
from interview_gpu.common.logging import get_logger
from interview_gpu.common.metrics import flatten_metrics
from interview_gpu.common.tasks import LossFn, MetricFn

_OOM_ERROR = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


@contextmanager
def oom_guard(batch_size: int | None = None) -> Iterator[None]:
    """Convert CUDA OOM into an actionable error. Never falls back to CPU."""
    try:
        yield
    except _OOM_ERROR as exc:
        torch.cuda.empty_cache()
        current = f" (current batch_size={batch_size})" if batch_size else ""
        raise GpuOutOfMemoryError(
            f"CUDA out of memory{current}.\n"
            "Reduce memory usage and retry:\n"
            "  1. lower training.batch_size (e.g. --set training.batch_size=<half>)\n"
            "  2. raise training.gradient_accumulation_steps to keep the effective batch size\n"
            "  3. shorten sequences (data.params.max_length / max_frames)\n"
            "  4. freeze more of the encoder or use a smaller model\n"
            "Training was stopped; it does NOT silently continue on CPU.\n"
            f"Original error: {str(exc).splitlines()[0]}"
        ) from exc


@dataclass
class EvalResult:
    """Outputs of one full pass over a loader (numpy, on CPU)."""

    loss: float
    metrics: dict[str, Any]
    outputs: np.ndarray
    targets: np.ndarray
    mask: np.ndarray
    sample_ids: list[str]


def build_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW over trainable parameters.

    Models may expose ``parameter_groups(lr, weight_decay)`` to use different
    learning rates for a pretrained encoder and freshly initialised heads.
    """
    training = cfg.training
    if hasattr(model, "parameter_groups"):
        groups = model.parameter_groups(training.lr, training.weight_decay)
    else:
        groups = [p for p in model.parameters() if p.requires_grad]
    if not groups:
        raise ConfigError("The model has no trainable parameters.")
    return torch.optim.AdamW(groups, lr=training.lr, weight_decay=training.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Linear warmup followed by cosine or linear decay (per optimizer step)."""
    training = cfg.training
    if training.scheduler == "none":
        return None
    warmup = int(total_steps * training.warmup_ratio)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        if training.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0 - progress

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class Trainer:
    """Config-driven training / evaluation loop.

    Args:
        model: Module already moved to ``device``.
        cfg: Full experiment config.
        device: Target device.
        amp: Resolved mixed-precision settings.
        loss_fn: ``loss(outputs, targets, mask)``.
        metric_fn: ``metrics(outputs, targets, mask)`` on numpy arrays.
        output_dir: Experiment output directory (history, plots, metrics).
        checkpoint_dir: Registry directory receiving ``model.pt``.
        checkpoint_extras: Extra fields for :func:`save_checkpoint`
            (``task``, ``label_mapping``, ``normalization``, ``dataset_metadata``).
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        cfg: Config,
        device: torch.device,
        amp: AmpSettings,
        loss_fn: LossFn,
        metric_fn: MetricFn,
        output_dir: str | Path,
        checkpoint_dir: str | Path,
        checkpoint_extras: dict[str, Any],
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device
        self.amp = amp
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_extras = checkpoint_extras
        self.logger = get_logger()
        # torch.compile wraps the module; weights are always saved from self.model.
        self._forward = torch.compile(model) if cfg.training.compile else model
        self.optimizer = build_optimizer(model, cfg)
        self.scaler = torch.amp.GradScaler(device.type, enabled=amp.use_grad_scaler)
        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ steps
    def _run_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = {
            key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch["inputs"].items()
        }
        targets = batch["targets"].to(self.device, non_blocking=True)
        mask = batch["mask"].to(self.device, non_blocking=True)
        with torch.autocast(device_type=self.device.type, dtype=self.amp.dtype, enabled=self.amp.enabled):
            outputs = self._forward(**inputs)
        # Loss in fp32 for numerical stability under bf16/fp16.
        outputs = outputs.float()
        loss = self.loss_fn(outputs, targets, mask)
        return loss, outputs, targets, mask

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        accumulation = self.cfg.training.gradient_accumulation_steps
        clip = self.cfg.training.grad_clip_norm
        total_loss, total_samples = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            loss, _, targets, _ = self._run_batch(batch)
            if not torch.isfinite(loss):
                raise NaNLossError(
                    f"Loss became {loss.item()} at epoch {epoch}, step {step}. "
                    "Check labels and feature normalisation for NaN/inf, lower training.lr, "
                    "or switch training.mixed_precision to fp32."
                )
            self.scaler.scale(loss / accumulation).backward()
            if step % accumulation == 0 or step == len(loader):
                if clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            if step % self.cfg.training.log_every == 0:
                self.logger.info("epoch %d step %d/%d loss %.4f", epoch, step, len(loader), loss.item())
        return total_loss / max(1, total_samples)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EvalResult:
        """Run inference over ``loader`` and compute loss and metrics."""
        self.model.eval()
        total_loss, total_samples = 0.0, 0
        outputs_all, targets_all, mask_all, ids = [], [], [], []
        for batch in loader:
            loss, outputs, targets, mask = self._run_batch(batch)
            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            outputs_all.append(outputs.cpu().numpy())
            targets_all.append(targets.cpu().numpy())
            mask_all.append(mask.cpu().numpy())
            ids.extend(str(sample_id) for sample_id in batch["sample_id"])
        outputs_np = np.concatenate(outputs_all)
        targets_np = np.concatenate(targets_all)
        mask_np = np.concatenate(mask_all)
        return EvalResult(
            loss=total_loss / max(1, total_samples),
            metrics=self.metric_fn(outputs_np, targets_np, mask_np),
            outputs=outputs_np,
            targets=targets_np,
            mask=mask_np,
            sample_ids=ids,
        )

    # -------------------------------------------------------------------- fit
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict[str, Any]:
        """Train with early stopping; restores the best weights before returning."""
        training, ckpt = self.cfg.training, self.cfg.checkpoint
        steps_per_epoch = math.ceil(len(train_loader) / training.gradient_accumulation_steps)
        self.scheduler = build_scheduler(self.optimizer, self.cfg, steps_per_epoch * training.epochs)
        sign = 1.0 if ckpt.mode == "min" else -1.0
        best_value, best_epoch, stale = math.inf, 0, 0
        wandb_run = self._init_wandb()

        for epoch in range(1, training.epochs + 1):
            started = time.perf_counter()
            reset_peak_memory(self.device)
            train_loss = self._train_epoch(train_loader, epoch)
            result = self.evaluate(val_loader)
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": result.loss,
                **{f"val_{k}": v for k, v in flatten_metrics(result.metrics).items()},
                "lr": self.optimizer.param_groups[0]["lr"],
                **memory_stats(self.device),
                "epoch_seconds": round(time.perf_counter() - started, 2),
            }
            self.history.append(row)
            self._write_history()
            if ckpt.metric not in row:
                raise ConfigError(
                    f"checkpoint.metric '{ckpt.metric}' is not produced by this task. "
                    f"Available: {sorted(k for k in row if k.startswith('val_'))}"
                )
            monitored = row[ckpt.metric]
            self.logger.info(
                "epoch %d/%d | train_loss %.4f | val_loss %.4f | %s %s | max_alloc %.0f MB | max_reserved %.0f MB",
                epoch, training.epochs, train_loss, result.loss, ckpt.metric, monitored,
                row["gpu_max_allocated_mb"], row["gpu_max_reserved_mb"],
            )
            if wandb_run is not None:
                wandb_run.log({k: v for k, v in row.items() if isinstance(v, (int, float))})

            improved = monitored is not None and sign * monitored < best_value
            if improved:
                best_value, best_epoch, stale = sign * monitored, epoch, 0
                self._save(epoch, monitored, result.metrics, MODEL_FILE)
            else:
                stale += 1
                if monitored is None:
                    self.logger.warning("Metric %s is undefined this epoch; not counted as improvement.", ckpt.metric)
            if not ckpt.save_best_only:
                self._save(epoch, monitored, result.metrics, f"epoch_{epoch:03d}.pt")
            if training.early_stopping_patience and stale >= training.early_stopping_patience:
                self.logger.info("Early stopping: no improvement for %d epoch(s).", stale)
                break

        if wandb_run is not None:
            wandb_run.finish()
        if best_epoch == 0:
            raise NaNLossError(f"No valid value of '{ckpt.metric}' was ever produced; nothing was saved.")
        # Restore the best weights so the caller evaluates the saved model.
        load_model_state(self.model, load_checkpoint(self.checkpoint_dir, self.device)["model_state"])
        self._plot_loss_curve()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "best_epoch": best_epoch,
            "best_metric_name": ckpt.metric,
            "best_metric_value": sign * best_value,
            "epochs_run": len(self.history),
        }

    # ---------------------------------------------------------------- helpers
    def _save(self, epoch: int, value: float | None, metrics: dict[str, Any], filename: str) -> None:
        save_checkpoint(
            self.checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            metric_name=self.cfg.checkpoint.metric,
            metric_value=value,
            config=self.cfg.to_dict(),
            metrics={"validation": metrics},
            filename=filename,
            **self.checkpoint_extras,
        )

    def _write_history(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fields = list(dict.fromkeys(key for row in self.history for key in row))
        with (self.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.history)
        write_json(self.output_dir / "history.json", self.history)

    def _plot_loss_curve(self) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        epochs = [row["epoch"] for row in self.history]
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(epochs, [row["train_loss"] for row in self.history], label="train")
        axis.plot(epochs, [row["val_loss"] for row in self.history], label="validation")
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.output_dir / "loss_curve.png", dpi=120)
        plt.close(figure)

    def _init_wandb(self) -> Any | None:
        if not self.cfg.logging.wandb:
            return None
        try:
            import wandb
        except ImportError:
            self.logger.warning("logging.wandb is true but wandb is not installed; continuing without it.")
            return None
        return wandb.init(
            project=self.cfg.logging.wandb_project, name=self.cfg.experiment.name, config=self.cfg.to_dict()
        )
