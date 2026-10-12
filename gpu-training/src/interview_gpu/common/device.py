"""Device selection, mixed-precision resolution and GPU memory reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from interview_gpu.common.errors import ConfigError


@dataclass(frozen=True)
class AmpSettings:
    """Resolved automatic-mixed-precision settings."""

    name: str  # "bf16" | "fp16" | "fp32"
    enabled: bool
    dtype: torch.dtype | None
    use_grad_scaler: bool


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def select_device(requested: str = "auto") -> torch.device:
    """Pick a device. Priority for ``auto``: CUDA > MPS > CPU.

    An explicit request for an unavailable accelerator raises instead of
    silently falling back to CPU.
    """
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ConfigError(
                "Device 'cuda' was requested but CUDA is not available. "
                "Run `python scripts/check_gpu.py` to diagnose, or use --device cpu."
            )
        device = torch.device(requested)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ConfigError(
                f"Device '{requested}' requested but only {torch.cuda.device_count()} GPU(s) found."
            )
        return device
    if requested == "mps":
        if not _mps_available():
            raise ConfigError("Device 'mps' was requested but MPS is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ConfigError(f"Unknown device '{requested}'. Use auto | cuda | cuda:N | mps | cpu.")


def bf16_supported() -> bool:
    """True when the GPU natively supports bfloat16 autocast (Ampere or newer)."""
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # older PyTorch without the keyword
        return bool(torch.cuda.is_bf16_supported())


def fp16_supported() -> bool:
    """True when CUDA float16 autocast is usable."""
    return torch.cuda.is_available()


def recommended_precision(device: torch.device) -> str:
    """bf16 on Ampere+, fp16 on older CUDA GPUs, fp32 elsewhere."""
    if device.type != "cuda":
        return "fp32"
    return "bf16" if bf16_supported() else "fp16"


def resolve_precision(mode: str, device: torch.device) -> AmpSettings:
    """Turn a ``mixed_precision`` config value into concrete AMP settings."""
    name = recommended_precision(device) if mode == "auto" else mode
    if name not in {"bf16", "fp16", "fp32"}:
        raise ConfigError(f"Unknown mixed_precision '{mode}'.")
    if device.type != "cuda" and name != "fp32":
        raise ConfigError(
            f"mixed_precision='{name}' requires CUDA but the selected device is '{device.type}'. "
            "Use mixed_precision: auto or fp32."
        )
    if name == "bf16" and not bf16_supported():
        raise ConfigError("bf16 is not natively supported on this GPU. Use fp16 or auto.")
    if name == "bf16":
        return AmpSettings("bf16", True, torch.bfloat16, use_grad_scaler=False)
    if name == "fp16":
        return AmpSettings("fp16", True, torch.float16, use_grad_scaler=True)
    return AmpSettings("fp32", False, None, use_grad_scaler=False)


def gpu_inventory() -> list[dict[str, Any]]:
    """Name and total VRAM for every visible CUDA device."""
    gpus = []
    for index in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": props.name,
                "vram_gb": round(props.total_memory / 1024**3, 2),
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return gpus


def environment_report(
    device: torch.device | None = None, amp: AmpSettings | None = None
) -> dict[str, Any]:
    """Collect everything that must be printed at the start of training."""
    cuda = torch.cuda.is_available()
    report: dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda if cuda else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        "gpu_count": torch.cuda.device_count() if cuda else 0,
        "gpus": gpu_inventory(),
        "mps_available": _mps_available(),
        "bf16_support": bf16_supported(),
        "fp16_support": fp16_supported(),
    }
    if device is not None:
        report["selected_device"] = str(device)
        report["recommended_precision"] = recommended_precision(device)
    if amp is not None:
        report["mixed_precision"] = amp.name if amp.enabled else "off (fp32)"
    return report


def format_environment_report(report: dict[str, Any]) -> str:
    """Human-readable multi-line rendering of :func:`environment_report`."""
    lines = [
        f"PyTorch: {report['pytorch_version']}",
        f"CUDA Available: {report['cuda_available']}",
        f"CUDA Version: {report['cuda_version']}",
        f"GPU Count: {report['gpu_count']}",
    ]
    for gpu in report["gpus"]:
        lines += [f"GPU {gpu['index']}:", f"  {gpu['name']}", f"  VRAM: {gpu['vram_gb']} GB"]
    lines += [
        f"BF16 Support: {report['bf16_support']}",
        f"FP16 Support: {report['fp16_support']}",
    ]
    if "selected_device" in report:
        lines.append(f"Selected Device: {report['selected_device']}")
        lines.append(f"Recommended Precision: {report['recommended_precision']}")
    if "mixed_precision" in report:
        lines.append(f"Mixed Precision: {report['mixed_precision']}")
    return "\n".join(lines)


def log_environment(logger: logging.Logger, device: torch.device, amp: AmpSettings) -> dict[str, Any]:
    """Log the environment report and return it."""
    report = environment_report(device, amp)
    for line in format_environment_report(report).splitlines():
        logger.info(line)
    return report


def reset_peak_memory(device: torch.device) -> None:
    """Reset CUDA peak-memory counters (no-op on other devices)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_stats(device: torch.device) -> dict[str, float]:
    """Peak allocated / reserved VRAM in MB since the last reset."""
    if device.type != "cuda":
        return {"gpu_max_allocated_mb": 0.0, "gpu_max_reserved_mb": 0.0}
    return {
        "gpu_max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1024**2, 1),
        "gpu_max_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1024**2, 1),
    }


def estimate_training_memory_mb(model: torch.nn.Module) -> dict[str, float]:
    """Rough lower bound of VRAM needed for training with AdamW.

    Counts weights, gradients and the two Adam moment buffers for trainable
    parameters. Activation memory depends on batch size / sequence length and
    is *not* included, so treat this as a floor, not a prediction.
    """
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    trainable = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    return {
        "weights_mb": round(total / 1024**2, 1),
        "estimated_min_training_mb": round((total + 3 * trainable) / 1024**2, 1),
    }
