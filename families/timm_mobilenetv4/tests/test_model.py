# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm MobileNetV4 image-classification family model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from families.timm_mobilenetv4 import model as model_module
    from families.timm_mobilenetv4.config import ModelConfig
    from families.timm_mobilenetv4.model import _TimmMobilenetv4Model, build as build_family
    from families.timm_mobilenetv4.support import describe
    from tensorrt_model_connect import BuildRequest
    from tensorrt_model_connect.model_support import ModelMetadata
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


model = _TimmMobilenetv4Model()


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _bn(tensors: dict, prefix: str, channels: int) -> None:
    tensors[f"{prefix}.weight"] = _rand(channels)
    tensors[f"{prefix}.bias"] = _rand(channels)
    tensors[f"{prefix}.running_mean"] = _rand(channels)
    tensors[f"{prefix}.running_var"] = np.abs(_rand(channels)) + 1.0


def _conv_norm(tensors: dict, prefix: str, out_ch: int, in_ch: int, kernel: int) -> None:
    tensors[f"{prefix}.conv.weight"] = _rand(out_ch, in_ch, kernel, kernel)
    _bn(tensors, f"{prefix}.bn", out_ch)


def _write_tiny_mnv4(
    tmp_path: Path,
    *,
    stages: int = 5,
    architecture: str = "mobilenetv4_conv_small",
    extra: dict | None = None,
) -> dict[str, np.ndarray]:
    """Write a narrow MobileNetV4 carrying all three block kinds.

    Stage 0 holds plain convolutions, stage 1 an edge residual, stages 2 and 3
    universal inverted bottlenecks, and stage 4 the final plain convolution.
    """
    classes = 5
    config = {
        "architecture": architecture,
        "num_classes": classes,
        "num_features": 16,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.95,
            "crop_mode": "center",
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    tensors: dict[str, np.ndarray] = {}
    ch = 8
    tensors["conv_stem.weight"] = _rand(ch, 3, 3, 3)
    _bn(tensors, "bn1", ch)

    layout = {
        0: [("conv_bn_act", 3), ("conv_bn_act", 1)],
        1: [("edge_residual", 3)],
        2: [("universal_inverted", 5), ("universal_inverted", 5)],
        3: [("universal_inverted", 3)],
        4: [("conv_bn_act", 1)],
    }
    for stage in range(stages):
        for index, (kind, kernel) in enumerate(layout[stage]):
            prefix = f"blocks.{stage}.{index}"
            if kind == "conv_bn_act":
                out = 16 if stage == 4 else ch
                tensors[f"{prefix}.conv.weight"] = _rand(out, ch, kernel, kernel)
                _bn(tensors, f"{prefix}.bn1", out)
            elif kind == "edge_residual":
                mid = ch * 2
                tensors[f"{prefix}.conv_exp.weight"] = _rand(mid, ch, kernel, kernel)
                _bn(tensors, f"{prefix}.bn1", mid)
                tensors[f"{prefix}.conv_pwl.weight"] = _rand(ch, mid, 1, 1)
                _bn(tensors, f"{prefix}.bn2", ch)
            else:
                mid = ch * 2
                _conv_norm(tensors, f"{prefix}.dw_start", ch, 1, kernel)
                _conv_norm(tensors, f"{prefix}.pw_exp", mid, ch, 1)
                _conv_norm(tensors, f"{prefix}.dw_mid", mid, 1, kernel)
                _conv_norm(tensors, f"{prefix}.pw_proj", ch, mid, 1)

    tensors["conv_head.weight"] = _rand(16, 16, 1, 1)
    _bn(tensors, "norm_head", 16)
    tensors["classifier.weight"] = _rand(classes, 16)
    tensors["classifier.bias"] = _rand(classes)
    if extra:
        tensors.update(extra)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tensors


def test_support_claims_only_the_convolution_widths() -> None:
    """The hybrid, aa and blur widths are deliberately not claimed."""
    for arch in (
        "mobilenetv4_conv_small_050",
        "mobilenetv4_conv_small",
        "mobilenetv4_conv_medium",
        "mobilenetv4_conv_large",
    ):
        assert describe(ModelMetadata(config={"architecture": arch}, model_index={})) is not None
    for arch in (
        "mobilenetv4_hybrid_medium",
        "mobilenetv4_conv_aa_large",
        "mobilenetv4_conv_blur_medium",
        "mobilenetv3_large_100",
    ):
        assert describe(ModelMetadata(config={"architecture": arch}, model_index={})) is None


def test_a_plain_convolution_never_adds_its_input_back() -> None:
    """Shape is not the rule, and two published widths prove it.

    `conv_small` and `conv_small_050` each hold a 32-to-32 stride-1 ConvBnAct
    whose input is not added. Using the shape-only rule there still builds a
    working engine: measured against the reference it drops to 0.58
    correlation and picks class 904 where the reference picks 111.
    """
    assert model_module._skips_input("conv_bn_act", 1, 32, 32) is False
    assert model_module._skips_input("universal_inverted", 1, 32, 32) is True
    assert model_module._skips_input("edge_residual", 1, 32, 32) is True
    # Every kind still needs the shape to allow it.
    assert model_module._skips_input("universal_inverted", 2, 32, 32) is False
    assert model_module._skips_input("universal_inverted", 1, 16, 32) is False


def test_the_first_block_of_the_first_four_stages_halves_the_resolution() -> None:
    """Stride is not in a safetensors checkpoint, so it is stated here.

    Stem 2 times four stage strides is the factor of 32 that the published
    224 and 256 input sizes divide by.
    """
    assert model_module._STAGE_COUNT == 5
    assert model_module._STRIDED_STAGES == frozenset({0, 1, 2, 3})
    assert 2 * 2 ** len(model_module._STRIDED_STAGES) == 32


def test_only_some_normalisations_are_followed_by_an_activation() -> None:
    """MobileNetV4 is ReLU throughout, but not after every norm.

    The depthwise convolution opening a universal inverted bottleneck and the
    pointwise projection closing it are both linear, as is the second norm of
    an edge residual. Read off all four widths.
    """
    assert model_module._ACTIVATED[("universal_inverted", "dw_start")] is False
    assert model_module._ACTIVATED[("universal_inverted", "pw_exp")] is True
    assert model_module._ACTIVATED[("universal_inverted", "dw_mid")] is True
    assert model_module._ACTIVATED[("universal_inverted", "pw_proj")] is False
    assert model_module._ACTIVATED[("edge_residual", "bn1")] is True
    assert model_module._ACTIVATED[("edge_residual", "bn2")] is False
    assert model_module._ACTIVATED[("conv_bn_act", "bn1")] is True


def test_block_kind_is_read_from_leaf_names() -> None:
    """The three kinds carry disjoint leaves, so no table is needed."""
    assert model_module._block_kind({"pw_exp", "pw_proj"}) == "universal_inverted"
    assert model_module._block_kind({"conv_exp", "conv_pwl", "bn1", "bn2"}) == "edge_residual"
    assert model_module._block_kind({"conv", "bn1"}) == "conv_bn_act"
    with pytest.raises(ValueError, match="Unsupported MobileNetV4 block keys"):
        model_module._block_kind({"attn"})


def test_layout_classifies_every_block_in_a_tiny_checkpoint(tmp_path: Path) -> None:
    _write_tiny_mnv4(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    model.load_weights(str(tmp_path), config, precision="fp32")
    blocks = config.raw["_timm_mobilenetv4_config"]["blocks"]
    assert [block["kind"] for block in blocks] == [
        "conv_bn_act",
        "conv_bn_act",
        "edge_residual",
        "universal_inverted",
        "universal_inverted",
        "universal_inverted",
        "conv_bn_act",
    ]
    # Stride 2 only on the first block of the first four stages.
    assert [block["stride"] for block in blocks] == [2, 1, 2, 2, 1, 2, 1]


def test_layout_rejects_a_stage_count_that_is_not_five(tmp_path: Path) -> None:
    _write_tiny_mnv4(tmp_path, stages=4)
    config = ModelConfig.from_dir(tmp_path)
    with pytest.raises(ValueError, match="exactly 5 contiguous stages"):
        model.load_weights(str(tmp_path), config, precision="fp32")


def test_layout_refuses_a_block_carrying_squeeze_excite(tmp_path: Path) -> None:
    """Other MobileNetV4 variants gate their blocks; these widths do not.

    Ignoring the gate would build quietly without it, so it is refused.
    """
    extra = {
        "blocks.2.0.se.conv_reduce.weight": _rand(4, 16, 1, 1),
        "blocks.2.0.se.conv_reduce.bias": _rand(4),
    }
    _write_tiny_mnv4(tmp_path, extra=extra)
    config = ModelConfig.from_dir(tmp_path)
    with pytest.raises(ValueError, match="unsupported leaves"):
        model.load_weights(str(tmp_path), config, precision="fp32")


def test_layout_refuses_a_block_missing_its_required_leaves(tmp_path: Path) -> None:
    """A bottleneck that expands and never projects is refused, not guessed."""
    tensors = _write_tiny_mnv4(tmp_path)
    for key in list(tensors):
        if key.startswith("blocks.3.0.pw_proj"):
            del tensors[key]
    save_file(tensors, str(tmp_path / "model.safetensors"))
    config = ModelConfig.from_dir(tmp_path)
    with pytest.raises(ValueError, match="is missing"):
        model.load_weights(str(tmp_path), config, precision="fp32")


def test_load_weights_keeps_norm_statistics_in_float32(tmp_path: Path) -> None:
    """The fold computes 1/sqrt(var + eps) on the host, so fp16 would lose it."""
    _write_tiny_mnv4(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    weights = model.load_weights(str(tmp_path), config, precision="fp16")
    assert weights["bn1.running_var"].dtype == np.float32
    assert weights["conv_stem.weight"].dtype == np.float16
    # The head convolution carries no bias; MobileNetV4 normalises after it.
    assert "conv_head.bias" not in weights
    assert weights["norm_head.running_mean"].dtype == np.float32


def test_bundle_config_reads_nested_pretrained_cfg(tmp_path: Path) -> None:
    _write_tiny_mnv4(tmp_path)
    config = ModelConfig.from_dir(tmp_path)
    model.load_weights(str(tmp_path), config, precision="fp32")
    overrides = model.get_bundle_config_overrides(config)
    assert overrides["input_image_h"] == 224
    assert overrides["crop_pct"] == pytest.approx(0.95)
    assert overrides["interpolation"] == "bicubic"
    assert overrides["num_classes"] == 5


def test_bundle_config_rejects_missing_pretrained_contract(tmp_path: Path) -> None:
    _write_tiny_mnv4(tmp_path)
    raw = json.loads((tmp_path / "config.json").read_text())
    del raw["pretrained_cfg"]["crop_pct"]
    (tmp_path / "config.json").write_text(json.dumps(raw))
    config = ModelConfig.from_dir(tmp_path)
    with pytest.raises(ValueError, match="missing required fields"):
        model.load_weights(str(tmp_path), config, precision="fp32")


def test_build_rejects_an_unclaimed_architecture(tmp_path: Path) -> None:
    _write_tiny_mnv4(tmp_path, architecture="mobilenetv4_hybrid_medium")
    with pytest.raises(ValueError, match="does not support architecture"):
        build_family(
            BuildRequest(
                model_dir=tmp_path,
                output_path=tmp_path / "out.bundle",
                family="timm_mobilenetv4",
                task="classification",
                precision="fp16",
            ),
            writer=None,
        )


def test_build_rejects_quantization(tmp_path: Path) -> None:
    _write_tiny_mnv4(tmp_path)
    with pytest.raises(NotImplementedError, match="quantization"):
        build_family(
            BuildRequest(
                model_dir=tmp_path,
                output_path=tmp_path / "out.bundle",
                family="timm_mobilenetv4",
                task="classification",
                precision="fp16",
                quantization="fp8",
            ),
            writer=None,
        )
