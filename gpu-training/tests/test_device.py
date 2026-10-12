"""Device selection and mixed-precision resolution."""

from __future__ import annotations

import pytest

from conftest import load_torch

torch = load_torch()

from interview_gpu.common.device import (  # noqa: E402
    environment_report,
    estimate_training_memory_mb,
    format_environment_report,
    memory_stats,
    recommended_precision,
    resolve_precision,
    select_device,
)
from interview_gpu.common.errors import ConfigError  # noqa: E402
from interview_gpu.common.seed import seed_everything  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_auto_selects_best_available_device() -> None:
    device = select_device("auto")
    expected = "cuda" if torch.cuda.is_available() else device.type
    assert device.type == expected
    assert select_device("cpu").type == "cpu"


def test_unknown_or_unavailable_device_raises() -> None:
    with pytest.raises(ConfigError, match="Unknown device"):
        select_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(ConfigError, match="CUDA is not available"):
            select_device("cuda")  # never silently falls back to CPU


def test_cpu_precision_is_fp32() -> None:
    cpu = torch.device("cpu")
    amp = resolve_precision("auto", cpu)
    assert (amp.name, amp.enabled, amp.dtype, amp.use_grad_scaler) == ("fp32", False, None, False)
    assert recommended_precision(cpu) == "fp32"
    with pytest.raises(ConfigError, match="requires CUDA"):
        resolve_precision("fp16", cpu)


def test_environment_report_contains_required_fields() -> None:
    device = select_device("auto")
    report = environment_report(device, resolve_precision("auto", device))
    for key in ("pytorch_version", "cuda_available", "cuda_version", "gpu_count", "gpus",
                "selected_device", "mixed_precision", "bf16_support", "fp16_support"):
        assert key in report
    text = format_environment_report(report)
    assert "PyTorch:" in text and "CUDA Available:" in text and "Recommended Precision:" in text
    assert memory_stats(torch.device("cpu")) == {"gpu_max_allocated_mb": 0.0, "gpu_max_reserved_mb": 0.0}


def test_memory_estimate_counts_only_trainable_extras() -> None:
    model = torch.nn.Linear(1024, 1024)
    full = estimate_training_memory_mb(model)
    assert full["estimated_min_training_mb"] == pytest.approx(4 * full["weights_mb"], rel=0.05)
    for parameter in model.parameters():
        parameter.requires_grad = False
    assert estimate_training_memory_mb(model)["estimated_min_training_mb"] == full["weights_mb"]


def test_seed_everything_is_reproducible() -> None:
    seed_everything(123)
    first = torch.rand(3)
    seed_everything(123)
    assert torch.equal(first, torch.rand(3))


@requires_cuda
def test_cuda_precision_and_autocast() -> None:
    device = select_device("cuda")
    amp = resolve_precision("auto", device)
    assert amp.name in ("bf16", "fp16") and amp.enabled
    assert amp.use_grad_scaler == (amp.name == "fp16")
    model = torch.nn.Linear(8, 2).to(device)
    with torch.autocast(device_type="cuda", dtype=amp.dtype):
        output = model(torch.randn(4, 8, device=device))
    assert output.dtype == amp.dtype
    report = environment_report(device, amp)
    assert report["gpu_count"] >= 1 and report["gpus"][0]["vram_gb"] > 0
