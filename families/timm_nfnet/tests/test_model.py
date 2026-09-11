# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for timm NFNet builds."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file


try:
    import tensorrt  # noqa: F401
except ModuleNotFoundError:
    sys.modules["tensorrt"] = SimpleNamespace()

from families.timm_nfnet import model  # noqa: E402
from families.timm_nfnet.checkpoint import Checkpoint  # noqa: E402
from families.timm_nfnet.support import describe  # noqa: E402
from tensorrt_model_connect.model_support import ModelMetadata  # noqa: E402


def _random(*shape: int) -> np.ndarray:
    return np.random.RandomState(17).randn(*shape).astype(np.float32)


def _conv(tensors: dict[str, np.ndarray], prefix: str, out_ch: int, in_ch: int, k: int) -> None:
    tensors[f"{prefix}.weight"] = _random(out_ch, in_ch, k, k)
    tensors[f"{prefix}.gain"] = _random(out_ch, 1, 1, 1)
    tensors[f"{prefix}.bias"] = _random(out_ch)


def _checkpoint(tmp_path: Path, depths: tuple[int, ...] = (1, 2)) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architecture": "dm_nfnet_f0",
                "num_classes": 5,
                "pretrained_cfg": {
                    "input_size": [3, 192, 192],
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                    "crop_pct": 0.9,
                    "interpolation": "bicubic",
                },
            }
        ),
        encoding="utf-8",
    )
    tensors: dict[str, np.ndarray] = {}
    for position in range(1, 5):
        _conv(tensors, f"stem.conv{position}", 8, 3 if position == 1 else 8, 3)
    for stage, depth in enumerate(depths):
        for index in range(depth):
            prefix = f"stages.{stage}.{index}"
            _conv(tensors, f"{prefix}.conv1", 8, 8, 1)
            _conv(tensors, f"{prefix}.conv2", 8, 8, 3)
            _conv(tensors, f"{prefix}.conv2b", 8, 8, 3)
            _conv(tensors, f"{prefix}.conv3", 8, 8, 1)
            tensors[f"{prefix}.attn_last.fc1.weight"] = _random(4, 8, 1, 1)
            tensors[f"{prefix}.attn_last.fc1.bias"] = _random(4)
            tensors[f"{prefix}.attn_last.fc2.weight"] = _random(8, 4, 1, 1)
            tensors[f"{prefix}.attn_last.fc2.bias"] = _random(8)
            tensors[f"{prefix}.skipinit_gain"] = np.array(0.5, dtype=np.float32)
            if index == 0:
                _conv(tensors, f"{prefix}.downsample.conv", 8, 8, 1)
    _conv(tensors, "final_conv", 8, 8, 1)
    tensors["head.fc.weight"] = _random(5, 8)
    tensors["head.fc.bias"] = _random(5)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def _metadata(model_type: str) -> ModelMetadata:
    return ModelMetadata(config={"model_type": model_type}, model_index={})


def test_support_claims_only_its_own_identity() -> None:
    assert describe(_metadata("dm_nfnet_f0")) is not None
    assert describe(_metadata("dm_nfnet_f6")) is not None
    assert describe(_metadata("regnety_040")) is None


def test_read_config_rejects_the_variants_with_another_activation(tmp_path: Path) -> None:
    """nfnet_l0 and eca_nfnet_l0 use SiLU and a different gate."""
    for architecture in ("nfnet_l0", "eca_nfnet_l0"):
        (tmp_path / "config.json").write_text(
            json.dumps({"architecture": architecture}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="unsupported timm NFNet model identity"):
            model._read_config(tmp_path)


def test_standardisation_centres_and_scales_each_filter(tmp_path: Path) -> None:
    """Each output filter ends up with the variance its gain and fan-in imply."""
    weight = _random(4, 3, 3, 3)
    gain = np.ones((4, 1, 1, 1), dtype=np.float32)
    out = model.standardise(weight, gain, np.float32)
    flat = out.reshape(4, -1).astype(np.float64)
    fan_in = flat.shape[1]
    np.testing.assert_allclose(flat.mean(axis=1), np.zeros(4), atol=1e-6)
    # Unit gain and this fan-in give each filter a norm of one.
    np.testing.assert_allclose(flat.std(axis=1) * np.sqrt(fan_in), np.ones(4), rtol=1e-4)


def test_standardisation_uses_the_epsilon_timm_overrides_to() -> None:
    """timm builds every NFNet convolution with 1e-5, not the 1e-6 default."""
    assert model._WEIGHT_STANDARDISATION_EPSILON == 1e-5


def test_the_activation_gain_is_the_documented_gelu_constant() -> None:
    """Without a norm, the activation itself restores unit variance."""
    assert model._GELU_GAMMA == 1.7015043497085571


def test_same_padding_puts_the_extra_pixel_last() -> None:
    """An odd amount of padding goes on the bottom and the right.

    Checked against timm's own get_same_padding over every size, kernel and
    stride this family produces; these are representative cases. An even input
    with stride 2 is where the asymmetry actually appears.
    """
    assert model.same_padding(192, 3, 2) == (0, 0, 1, 1)
    assert model.same_padding(192, 5, 2) == (1, 1, 2, 2)
    assert model.same_padding(8, 3, 2) == (0, 0, 1, 1)
    # Stride 1 always needs an even amount, so it stays symmetric.
    assert model.same_padding(192, 3, 1) == (1, 1, 1, 1)
    # An odd input happens to need an even amount, so symmetric again.
    assert model.same_padding(7, 3, 2) == (1, 1, 1, 1)
    # A stride that already divides the input needs nothing.
    assert model.same_padding(8, 1, 1) == (0, 0, 0, 0)


def test_residual_scale_restarts_at_every_stage(tmp_path: Path) -> None:
    """beta is a schedule, not a stored weight, and resets at each stage head.

    The branch gain grows with depth so the signal does not, and the reset
    happens after the first block of a stage rather than before it.
    """
    blocks = model._layout(Checkpoint.open(_checkpoint(tmp_path, depths=(1, 2))))
    betas = [round(float(block["beta"]), 6) for block in blocks]
    assert betas == [1.0, round((1.04) ** -0.5, 6), round((1.04) ** -0.5, 6)]


def test_layout_reads_the_optional_leaves_from_the_checkpoint(tmp_path: Path) -> None:
    blocks = model._layout(Checkpoint.open(_checkpoint(tmp_path)))
    assert all(block["has_second_conv"] for block in blocks)
    assert all(block["has_gate"] for block in blocks)
    assert [bool(block["has_downsample"]) for block in blocks] == [True, True, False]


def test_layout_strides_only_after_the_first_stage(tmp_path: Path) -> None:
    blocks = model._layout(Checkpoint.open(_checkpoint(tmp_path, depths=(1, 2))))
    assert [block["stride"] for block in blocks] == [1, 2, 1]


def test_layout_rejects_stage_numbering_that_does_not_start_at_zero(tmp_path: Path) -> None:
    tensors: dict[str, np.ndarray] = {}
    _conv(tensors, "stages.1.0.conv1", 4, 4, 1)
    _conv(tensors, "stages.1.0.conv2", 4, 4, 3)
    _conv(tensors, "stages.1.0.conv3", 4, 4, 1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="not contiguous from 0"):
        model._layout(Checkpoint.open(tmp_path))


def test_fp32_builds_switch_off_the_reduced_precision_path() -> None:
    import tensorrt as trt

    class _Config:
        def __init__(self) -> None:
            self.cleared: list[object] = []

        def clear_flag(self, flag: object) -> None:
            self.cleared.append(flag)

    config = _Config()
    model._configure_precision(config, "fp32")
    assert config.cleared == [trt.BuilderFlag.TF32]
    config = _Config()
    model._configure_precision(config, "fp16")
    assert config.cleared == []


def test_preprocess_config_rejects_a_degenerate_std() -> None:
    raw = {
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 192, 192],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.0, 0.225],
            "crop_pct": 0.9,
            "interpolation": "bicubic",
        },
    }
    with pytest.raises(ValueError, match="preprocessing or classifier config is invalid"):
        model._preprocess_config(raw)


def test_preprocess_config_carries_the_crop_mode() -> None:
    """dm_nfnet asks for "squash", which resizes both axes to the same length.

    Defaulting to "center" instead keeps the aspect ratio and crops, which
    feeds the engine different pixels and changes the predicted class on an
    image where the top logits are close.
    """
    raw = {
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 192, 192],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.9,
            "interpolation": "bicubic",
            "crop_mode": "squash",
        },
    }
    assert model._preprocess_config(raw)["crop_mode"] == "squash"


def test_preprocess_config_defaults_the_crop_mode_to_centre() -> None:
    raw = {
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 192, 192],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.9,
            "interpolation": "bicubic",
        },
    }
    assert model._preprocess_config(raw)["crop_mode"] == "center"


def test_preprocess_config_rejects_an_unknown_crop_mode() -> None:
    raw = {
        "num_classes": 5,
        "pretrained_cfg": {
            "input_size": [3, 192, 192],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.9,
            "interpolation": "bicubic",
            "crop_mode": "border",
        },
    }
    with pytest.raises(ValueError, match="preprocessing or classifier config is invalid"):
        model._preprocess_config(raw)
