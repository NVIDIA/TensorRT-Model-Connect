# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm MNASNet image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from families.timm_mnasnet.config import ModelConfig
    from families.timm_mnasnet.model import _TimmMnasnetModel, build as build_family
    from tensorrt_model_connect import BuildRequest
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


model = _TimmMnasnetModel()


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _bn(t: dict, prefix: str, ch: int) -> None:
    t[f"{prefix}.weight"] = _rand(ch)
    t[f"{prefix}.bias"] = _rand(ch)
    t[f"{prefix}.running_mean"] = _rand(ch)
    t[f"{prefix}.running_var"] = np.abs(_rand(ch)) + 1.0


def _write_tiny_mnasnet(tmp_path: Path, *, stages: int = 7, with_se: bool = False) -> None:
    """Write a miniature MNASNet with one block per stage."""
    classes = 5
    config = {
        "architecture": "mnasnet_100",
        "num_classes": classes,
        "num_features": 16,
        "pretrained_cfg": {
            "input_size": [3, 224, 224],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "crop_pct": 0.875,
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    t: dict[str, np.ndarray] = {}
    ch = 8
    t["conv_stem.weight"] = _rand(ch, 3, 3, 3)
    _bn(t, "bn1", ch)

    for stage in range(stages):
        p = f"blocks.{stage}.0"
        if stage == 0:
            t[f"{p}.conv_dw.weight"] = _rand(ch, 1, 3, 3)
            _bn(t, f"{p}.bn1", ch)
            t[f"{p}.conv_pw.weight"] = _rand(ch, ch, 1, 1)
            _bn(t, f"{p}.bn2", ch)
        else:
            mid = ch * 2
            t[f"{p}.conv_pw.weight"] = _rand(mid, ch, 1, 1)
            _bn(t, f"{p}.bn1", mid)
            t[f"{p}.conv_dw.weight"] = _rand(mid, 1, 3, 3)
            _bn(t, f"{p}.bn2", mid)
            if with_se and stage == 1:
                t[f"{p}.se.conv_reduce.weight"] = _rand(4, mid, 1, 1)
                t[f"{p}.se.conv_reduce.bias"] = _rand(4)
                t[f"{p}.se.conv_expand.weight"] = _rand(mid, 4, 1, 1)
                t[f"{p}.se.conv_expand.bias"] = _rand(mid)
            t[f"{p}.conv_pwl.weight"] = _rand(ch, mid, 1, 1)
            _bn(t, f"{p}.bn3", ch)

    t["conv_head.weight"] = _rand(16, ch, 1, 1)
    _bn(t, "bn2", 16)
    t["classifier.weight"] = _rand(classes, 16)
    t["classifier.bias"] = _rand(classes)
    save_file(t, str(tmp_path / "model.safetensors"))


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_mnasnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.architecture == "mnasnet_100"


def test_model_config_requires_the_exact_config_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        ModelConfig.from_dir(tmp_path)


def test_bundle_config_reads_nested_pretrained_cfg(tmp_path: Path):
    _write_tiny_mnasnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    model.load_weights(str(tmp_path), cfg, precision="fp32")
    bundle_config = model.get_bundle_config_overrides(cfg)

    assert bundle_config["input_image_h"] == 224
    assert bundle_config["num_classes"] == 5
    assert bundle_config["crop_pct"] == pytest.approx(0.875)


def test_bundle_config_rejects_missing_pretrained_contract(tmp_path: Path):
    _write_tiny_mnasnet(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("pretrained_cfg")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="requires pretrained_cfg"):
        model.load_weights(str(tmp_path), ModelConfig.from_dir(tmp_path), precision="fp32")


def test_layout_classifies_blocks_and_applies_the_stride_schedule(tmp_path: Path):
    _write_tiny_mnasnet(tmp_path, stages=7)
    cfg = ModelConfig.from_dir(tmp_path)

    model.load_weights(str(tmp_path), cfg, precision="fp32")
    blocks = cfg.raw["_timm_mnasnet_config"]["blocks"]

    assert blocks[0]["kind"] == "depthwise_separable"
    assert all(b["kind"] == "inverted_residual" for b in blocks[1:])
    assert [b["stride"] for b in blocks] == [1, 2, 2, 2, 1, 2, 1]


def test_layout_rejects_a_squeeze_excite_gate_it_cannot_build(tmp_path: Path):
    """MNASNet has no SE gate; a checkpoint carrying one must not be built silently."""
    _write_tiny_mnasnet(tmp_path, with_se=True)
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="squeeze-excite"):
        model.load_weights(str(tmp_path), cfg, precision="fp32")


def test_layout_rejects_a_stage_count_without_a_schedule(tmp_path: Path):
    _write_tiny_mnasnet(tmp_path, stages=4)
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="schedule for 4 stages"):
        model.load_weights(str(tmp_path), cfg, precision="fp32")


def test_layout_rejects_an_unconsumed_block_tensor(tmp_path: Path):
    _write_tiny_mnasnet(tmp_path)
    from safetensors.numpy import load_file

    tensors = dict(load_file(str(tmp_path / "model.safetensors")))
    tensors["blocks.1.0.unused.weight"] = _rand(1)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    with pytest.raises(ValueError, match="Unexpected MNASNet block keys"):
        model.load_weights(str(tmp_path), ModelConfig.from_dir(tmp_path), precision="fp32")


def test_load_weights_reads_the_exact_inventory_and_precision(tmp_path: Path):
    _write_tiny_mnasnet(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    from safetensors.numpy import load_file

    expected = load_file(str(tmp_path / "model.safetensors"))
    weights = model.load_weights(str(tmp_path), cfg, precision="fp16")

    assert set(weights) == set(expected)
    for name, value in weights.items():
        expected_dtype = np.float32 if ".bn" in name or name.startswith("bn") else np.float16
        assert value.dtype == expected_dtype


def test_build_rejects_unqualified_semnasnet_variant(tmp_path: Path):
    _write_tiny_mnasnet(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architecture"] = "semnasnet_100"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "unused.bundle",
        family="timm_mnasnet",
        task="classification",
        precision="fp16",
        max_sequence_length=1,
    )

    with pytest.raises(ValueError, match="architecture='semnasnet_100'"):
        build_family(request, object())


def test_build_rejects_quantization(tmp_path: Path):
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "unused.bundle",
        family="timm_mnasnet",
        task="classification",
        precision="fp16",
        quantization="fp8",
    )

    with pytest.raises(NotImplementedError, match="quantization"):
        build_family(request, object())
