# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm MobileNetV3 image-classification family model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from families.timm_mobilenetv3.config import ModelConfig
    from families.timm_mobilenetv3.model import _TimmMobilenetv3Model, build as build_family
    from tensorrt_model_connect import BuildRequest
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


model = _TimmMobilenetv3Model()


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _bn(t: dict, prefix: str, ch: int) -> None:
    t[f"{prefix}.weight"] = _rand(ch)
    t[f"{prefix}.bias"] = _rand(ch)
    t[f"{prefix}.running_mean"] = _rand(ch)
    t[f"{prefix}.running_var"] = np.abs(_rand(ch)) + 1.0


def _write_tiny_mnv3(tmp_path: Path, *, stages: int = 7) -> dict[str, np.ndarray]:
    """Write a narrow MobileNetV3 with the canonical Large block counts.

    Stage 0 is depthwise-separable, the last stage is a plain conv-bn-act, and
    the middle stages are inverted residuals; one of them carries an SE gate.
    """
    classes = 5
    config = {
        "architecture": "mobilenetv3_large_100",
        "num_classes": classes,
        "num_features": 16,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.875,
            "crop_mode": "center",
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    t: dict[str, np.ndarray] = {}
    ch = 8
    t["conv_stem.weight"] = _rand(ch, 3, 3, 3)
    _bn(t, "bn1", ch)

    counts = (1, 2, 3, 4, 2, 3, 1)
    for stage, count in enumerate(counts[:stages]):
        for index in range(count):
            p = f"blocks.{stage}.{index}"
            if stage == 0:
                t[f"{p}.conv_dw.weight"] = _rand(ch, 1, 3, 3)
                _bn(t, f"{p}.bn1", ch)
                t[f"{p}.conv_pw.weight"] = _rand(ch, ch, 1, 1)
                _bn(t, f"{p}.bn2", ch)
            elif stage == 6:
                t[f"{p}.conv.weight"] = _rand(16, ch, 1, 1)
                _bn(t, f"{p}.bn1", 16)
            else:
                mid = ch * 2
                t[f"{p}.conv_pw.weight"] = _rand(mid, ch, 1, 1)
                _bn(t, f"{p}.bn1", mid)
                t[f"{p}.conv_dw.weight"] = _rand(mid, 1, 3, 3)
                _bn(t, f"{p}.bn2", mid)
                if (stage, index) in {
                    (2, 0),
                    (2, 1),
                    (2, 2),
                    (4, 0),
                    (4, 1),
                    (5, 0),
                    (5, 1),
                    (5, 2),
                }:
                    t[f"{p}.se.conv_reduce.weight"] = _rand(ch, mid, 1, 1)
                    t[f"{p}.se.conv_reduce.bias"] = _rand(ch)
                    t[f"{p}.se.conv_expand.weight"] = _rand(mid, ch, 1, 1)
                    t[f"{p}.se.conv_expand.bias"] = _rand(mid)
                t[f"{p}.conv_pwl.weight"] = _rand(ch, mid, 1, 1)
                _bn(t, f"{p}.bn3", ch)

    t["conv_head.weight"] = _rand(16, 16, 1, 1)
    t["conv_head.bias"] = _rand(16)
    t["classifier.weight"] = _rand(classes, 16)
    t["classifier.bias"] = _rand(classes)
    save_file(t, str(tmp_path / "model.safetensors"))
    return t


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_mnv3(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.architecture == "mobilenetv3_large_100"


def test_bundle_config_reads_nested_pretrained_cfg(tmp_path: Path):
    _write_tiny_mnv3(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    model.load_weights(str(tmp_path), cfg, precision="fp32")

    bundle_config = model.get_bundle_config_overrides(cfg)

    assert bundle_config["num_classes"] == 5
    assert bundle_config["interpolation"] == "bicubic"
    assert bundle_config["crop_pct"] == pytest.approx(0.875)


def test_bundle_config_rejects_missing_pretrained_contract(tmp_path: Path):
    _write_tiny_mnv3(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("pretrained_cfg")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="requires pretrained_cfg"):
        model.load_weights(str(tmp_path), ModelConfig.from_dir(tmp_path), precision="fp32")


def test_layout_classifies_each_block_kind(tmp_path: Path):
    _write_tiny_mnv3(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    model.load_weights(str(tmp_path), cfg, precision="fp32")
    blocks = cfg.raw["_timm_mobilenetv3_config"]["blocks"]

    kinds = [block["kind"] for block in blocks]
    assert kinds[0] == "depthwise_separable"
    assert kinds[-1] == "conv_bn_act"
    assert kinds[1:-1] == ["inverted_residual"] * 14
    assert [b["has_se"] for b in blocks].count(True) == 8


def test_layout_applies_the_large_stride_and_activation_schedule(tmp_path: Path):
    """Strides and activations are not in the checkpoint; they come from a table."""
    _write_tiny_mnv3(tmp_path, stages=7)
    cfg = ModelConfig.from_dir(tmp_path)

    model.load_weights(str(tmp_path), cfg, precision="fp32")
    blocks = cfg.raw["_timm_mobilenetv3_config"]["blocks"]

    assert [b["stride"] for b in blocks] == [
        1,
        2,
        1,
        2,
        1,
        1,
        2,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        1,
        1,
    ]
    assert [b["activation"] for b in blocks] == ["relu"] * 6 + ["hard_swish"] * 10


def test_layout_rejects_a_stage_count_without_a_schedule(tmp_path: Path):
    _write_tiny_mnv3(tmp_path, stages=4)
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="exactly seven contiguous stages"):
        model.load_weights(str(tmp_path), cfg, precision="fp32")


def test_load_weights_reads_the_exact_inventory_and_precision(tmp_path: Path):
    expected = _write_tiny_mnv3(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    weights = model.load_weights(str(tmp_path), cfg, precision="fp16")

    assert set(weights) == set(expected)
    for name, value in weights.items():
        assert value.shape == expected[name].shape
        expected_dtype = np.float32 if ".bn" in name or name.startswith("bn1.") else np.float16
        assert value.dtype == expected_dtype


def test_load_weights_rejects_an_unknown_block_shape(tmp_path: Path):
    _write_tiny_mnv3(tmp_path)
    from safetensors.numpy import load_file

    tensors = dict(load_file(str(tmp_path / "model.safetensors")))
    tensors["blocks.1.0.unknown.weight"] = _rand(1)
    save_file(tensors, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="Unexpected MobileNetV3 block keys"):
        model.load_weights(str(tmp_path), cfg, precision="fp32")


def test_load_weights_rejects_wrong_se_topology(tmp_path: Path):
    _write_tiny_mnv3(tmp_path)
    from safetensors.numpy import load_file

    tensors = dict(load_file(str(tmp_path / "model.safetensors")))
    for suffix in ("weight", "bias"):
        tensors.pop(f"blocks.2.1.se.conv_reduce.{suffix}")
        tensors.pop(f"blocks.2.1.se.conv_expand.{suffix}")
    save_file(tensors, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="SE topology mismatch"):
        model.load_weights(str(tmp_path), cfg, precision="fp32")


def test_load_weights_rejects_missing_classifier(tmp_path: Path):
    _write_tiny_mnv3(tmp_path)
    from safetensors.numpy import load_file

    tensors = dict(load_file(str(tmp_path / "model.safetensors")))
    tensors.pop("classifier.bias")
    save_file(tensors, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(KeyError, match="classifier.bias"):
        model.load_weights(str(tmp_path), cfg, precision="fp32")


def test_build_rejects_unqualified_small_variant(tmp_path: Path):
    _write_tiny_mnv3(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architecture"] = "mobilenetv3_small_100"
    config["model_type"] = "timm_mobilenetv3"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "unused.bundle",
        family="timm_mobilenetv3",
        task="classification",
        precision="fp16",
        max_sequence_length=1,
    )

    with pytest.raises(ValueError, match="architecture='mobilenetv3_small_100'"):
        build_family(request, object())


def test_build_rejects_quantization(tmp_path: Path):
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "unused.bundle",
        family="timm_mobilenetv3",
        task="classification",
        precision="fp16",
        quantization="fp8",
    )

    with pytest.raises(NotImplementedError, match="quantization"):
        build_family(request, object())
