"""Model forward passes, losses and the multimodal smoke test."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import load_torch

torch = load_torch()

from interview_gpu.answer.model import AnswerEvaluatorModel, freeze_encoder  # noqa: E402
from interview_gpu.common.config import TrainingConfig  # noqa: E402
from interview_gpu.common.errors import ConfigError  # noqa: E402
from interview_gpu.common.tasks import (  # noqa: E402
    TaskSpec,
    build_loss_fn,
    build_metric_fn,
    class_weights,
    focal_loss,
    masked_regression_loss,
)
from interview_gpu.multimodal.model import DEFAULT_HEADS, MultimodalInterviewModel  # noqa: E402
from interview_gpu.schemas.answer import ANSWER_DIMENSIONS  # noqa: E402
from interview_gpu.schemas.speech import SPEECH_FEATURE_NAMES  # noqa: E402
from interview_gpu.schemas.video import VIDEO_MODEL_FEATURES  # noqa: E402
from interview_gpu.speech.encoder import SpeechEncoder  # noqa: E402
from interview_gpu.speech.model import SpeechBehaviorMLP, SpeechEncoderRegressor  # noqa: E402
from interview_gpu.video.model import BACKBONES, TemporalBehaviorModel  # noqa: E402


def tiny_text_encoder() -> torch.nn.Module:
    from transformers import BertConfig, BertModel

    return BertModel(
        BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                   intermediate_size=64, max_position_embeddings=64)
    )


def tiny_speech_backbone() -> torch.nn.Module:
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    return Wav2Vec2Model(
        Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                       conv_dim=(16, 16), conv_stride=(5, 2), conv_kernel=(10, 3),
                       num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2, mask_time_prob=0.0)
    )


def test_speech_mlp_forward() -> None:
    model = SpeechBehaviorMLP(input_dim=len(SPEECH_FEATURE_NAMES), output_dim=4)
    assert model(torch.randn(5, len(SPEECH_FEATURE_NAMES))).shape == (5, 4)


def test_speech_encoder_handles_padding() -> None:
    encoder = SpeechEncoder(tiny_speech_backbone()).eval()
    assert not any(p.requires_grad for p in encoder.parameters())  # Stage 2: fully frozen
    wave = torch.randn(1, 4000)
    padded = torch.cat([wave, torch.zeros(1, 4000)], dim=1)
    mask = torch.cat([torch.ones(1, 4000), torch.zeros(1, 4000)], dim=1).long()
    with torch.no_grad():
        alone, with_padding = encoder(wave), encoder(padded, mask)
    assert alone.shape == with_padding.shape == (1, 32)
    assert torch.isfinite(alone).all() and torch.isfinite(with_padding).all()
    assert encoder.embed(np.random.randn(4000).astype(np.float32)).shape == (32,)


def test_speech_partial_finetuning_parameter_groups() -> None:
    model = SpeechEncoderRegressor(SpeechEncoder(tiny_speech_backbone(), trainable_layers=1), output_dim=3)
    trainable = [name for name, p in model.encoder.named_parameters() if p.requires_grad]
    assert trainable and all("encoder.layers.1." in name for name in trainable)
    groups = model.parameter_groups(lr=1e-3, weight_decay=0.0)
    assert [group["lr"] for group in groups] == pytest.approx([1e-3, 1e-4])
    waveform = torch.randn(2, 4000)
    assert model(waveform, torch.ones(2, 4000, dtype=torch.long)).shape == (2, 3)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_temporal_backbones_forward_and_mask(backbone: str) -> None:
    features = len(VIDEO_MODEL_FEATURES)
    model = TemporalBehaviorModel(features, output_dim=3, backbone=backbone, hidden_dim=32, num_heads=2).eval()
    sequence = torch.randn(2, 20, features)
    lengths = torch.tensor([20, 12])
    with torch.no_grad():
        output = model(sequence, lengths)
        changed = sequence.clone()
        changed[1, 12:] = 100.0  # padding content must not influence the result
        output_changed = model(changed, lengths)
    assert output.shape == (2, 3)
    assert torch.allclose(output, output_changed, atol=1e-4)


def test_temporal_model_rejects_bad_settings() -> None:
    with pytest.raises(ConfigError, match="Unknown video backbone"):
        TemporalBehaviorModel(16, 1, backbone="resnet")
    with pytest.raises(ConfigError, match="odd"):
        TemporalBehaviorModel(16, 1, backbone="tcn", kernel_size=4)


def test_answer_model_has_one_head_per_dimension() -> None:
    model = AnswerEvaluatorModel(tiny_text_encoder(), hidden_size=32, dimensions=list(ANSWER_DIMENSIONS),
                                 encoder_lr_scale=0.5)
    assert set(model.heads) == set(ANSWER_DIMENSIONS)
    input_ids = torch.randint(3, 100, (3, 12))
    attention_mask = torch.ones(3, 12, dtype=torch.long)
    attention_mask[2, 6:] = 0
    assert model(input_ids, attention_mask).shape == (3, len(ANSWER_DIMENSIONS))
    assert model.encode(input_ids, attention_mask).shape == (3, 32)
    groups = model.parameter_groups(lr=1e-4, weight_decay=0.01)
    assert [group["lr"] for group in groups] == pytest.approx([1e-4, 5e-5])

    freeze_encoder(model.encoder, trainable_layers=1)
    trainable = [name for name, p in model.encoder.named_parameters() if p.requires_grad]
    assert trainable and all(name.startswith("encoder.layer.1.") for name in trainable)
    freeze_encoder(model.encoder, 0)
    assert len(model.parameter_groups(1e-4, 0.0)) == 1  # heads only


def test_multimodal_smoke() -> None:
    """Skeleton check only: shapes, missing modalities and gradient flow."""
    dims = {"speech": 24, "video": 16, "text": 32}
    model = MultimodalInterviewModel(dims, fusion_dim=32, num_heads=2)
    batch = 4
    inputs = {name: torch.randn(batch, size) for name, size in dims.items()}
    mask = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 1], [1, 1, 0]], dtype=torch.float32)
    output = model(**inputs, modality_mask=mask)
    assert output.shape == (batch, sum(DEFAULT_HEADS.values())) and torch.isfinite(output).all()
    heads = model.split_outputs(output)
    assert {name: tensor.shape[1] for name, tensor in heads.items()} == DEFAULT_HEADS
    output.sum().backward()
    assert all(p.grad is not None for p in model.fusion.parameters())

    model.eval()
    with torch.no_grad():
        reference = model(**inputs, modality_mask=mask)
        inputs["video"][1] = 50.0  # sample 1 has no video: its content must be ignored
        assert torch.allclose(reference[1], model(**inputs, modality_mask=mask)[1], atol=1e-4)


def test_masked_regression_loss_ignores_missing_labels() -> None:
    outputs = torch.tensor([[1.0, 5.0], [2.0, 7.0]])
    targets = torch.tensor([[1.0, 0.0], [4.0, 7.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    assert masked_regression_loss(outputs, targets, mask, "mse").item() == pytest.approx(4.0 / 3.0)
    assert masked_regression_loss(outputs, targets, mask, "l1").item() == pytest.approx(2.0 / 3.0)
    assert masked_regression_loss(outputs, targets, torch.zeros(2, 2), "huber").item() == 0.0


def test_classification_losses_and_class_weights() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    cross_entropy = torch.nn.functional.cross_entropy(logits, targets)
    assert focal_loss(logits, targets, gamma=0.0).item() == pytest.approx(cross_entropy.item())
    assert focal_loss(logits, targets, gamma=2.0).item() < cross_entropy.item()  # easy examples down-weighted

    counts = np.array([90, 10])
    weights = class_weights(counts)
    assert weights[1] / weights[0] == pytest.approx(9.0)

    task = TaskSpec(kind="classification", target_names=["y"], class_names=["a", "b"])
    plain = build_loss_fn(task, TrainingConfig(), counts)(logits, targets, torch.ones(3))
    weighted_cfg = TrainingConfig(imbalance={"class_weights": True})
    weighted = build_loss_fn(task, weighted_cfg, counts)(logits, targets, torch.ones(3))
    focal_cfg = TrainingConfig(imbalance={"focal_loss": True, "class_weights": True})
    focal = build_loss_fn(task, focal_cfg, counts)(logits, targets, torch.ones(3))
    assert plain.item() == pytest.approx(cross_entropy.item())
    assert weighted.item() != pytest.approx(plain.item()) and torch.isfinite(focal)


def test_metric_fn_reports_in_label_units() -> None:
    task = TaskSpec(kind="regression", target_names=["a", "b"], target_mean=[5.0, 50.0], target_std=[2.0, 10.0])
    targets = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    outputs = targets + 0.5  # 0.5 standardised units off -> 1.0 and 5.0 in label units
    metrics = build_metric_fn(task)(outputs, targets, np.ones_like(targets))
    assert metrics["per_head"]["a"]["mae"] == pytest.approx(1.0)
    assert metrics["per_head"]["b"]["mae"] == pytest.approx(5.0)
    assert metrics["mean"]["mae"] == pytest.approx(3.0)
