# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm Inception image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from families.timm_inception.config import ModelConfig
    from families.timm_inception.model import _TimmInceptionModel, build as build_family
    from tensorrt_model_connect import BuildRequest
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


model = _TimmInceptionModel()


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _conv_bn(t: dict, prefix: str, out_ch: int, in_ch: int, kh: int, kw: int) -> None:
    t[f"{prefix}.conv.weight"] = _rand(out_ch, in_ch, kh, kw)
    t[f"{prefix}.bn.weight"] = _rand(out_ch)
    t[f"{prefix}.bn.bias"] = _rand(out_ch)
    t[f"{prefix}.bn.running_mean"] = _rand(out_ch)
    t[f"{prefix}.bn.running_var"] = np.abs(_rand(out_ch)) + 1.0


# The branch names that identify each topology, with a representative kernel.
_TOPOLOGIES = {
    "inception_a": [
        ("branch1x1", 1, 1),
        ("branch5x5_1", 1, 1),
        ("branch5x5_2", 5, 5),
        ("branch3x3dbl_1", 1, 1),
        ("branch3x3dbl_2", 3, 3),
        ("branch3x3dbl_3", 3, 3),
        ("branch_pool", 1, 1),
    ],
    "inception_b": [
        ("branch3x3", 3, 3),
        ("branch3x3dbl_1", 1, 1),
        ("branch3x3dbl_2", 3, 3),
        ("branch3x3dbl_3", 3, 3),
    ],
    "inception_c": [
        ("branch1x1", 1, 1),
        ("branch7x7_1", 1, 1),
        ("branch7x7_2", 1, 7),
        ("branch7x7_3", 7, 1),
        ("branch7x7dbl_1", 1, 1),
        ("branch7x7dbl_2", 7, 1),
        ("branch7x7dbl_3", 1, 7),
        ("branch7x7dbl_4", 7, 1),
        ("branch7x7dbl_5", 1, 7),
        ("branch_pool", 1, 1),
    ],
    "inception_d": [
        ("branch3x3_1", 1, 1),
        ("branch3x3_2", 3, 3),
        ("branch7x7x3_1", 1, 1),
        ("branch7x7x3_2", 1, 7),
        ("branch7x7x3_3", 7, 1),
        ("branch7x7x3_4", 3, 3),
    ],
    "inception_e": [
        ("branch1x1", 1, 1),
        ("branch3x3_1", 1, 1),
        ("branch3x3_2a", 1, 3),
        ("branch3x3_2b", 3, 1),
        ("branch3x3dbl_1", 1, 1),
        ("branch3x3dbl_2", 3, 3),
        ("branch3x3dbl_3a", 1, 3),
        ("branch3x3dbl_3b", 3, 1),
        ("branch_pool", 1, 1),
    ],
}


def _write_tiny_inception(tmp_path: Path, *, kinds: tuple[str, ...] = ("inception_a",)) -> None:
    """Write a miniature Inception whose Mixed blocks cover the given topologies."""
    classes = 5
    config = {
        "architecture": "inception_v3",
        "num_classes": classes,
        "num_features": 16,
        "pretrained_cfg": {
            "input_size": [3, 299, 299],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "crop_pct": 0.875,
            "interpolation": "bicubic",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    t: dict[str, np.ndarray] = {}
    for name, in_ch, out_ch, k in (
        ("Conv2d_1a_3x3", 3, 8, 3),
        ("Conv2d_2a_3x3", 8, 8, 3),
        ("Conv2d_2b_3x3", 8, 8, 3),
        ("Conv2d_3b_1x1", 8, 8, 1),
        ("Conv2d_4a_3x3", 8, 8, 3),
    ):
        _conv_bn(t, name, out_ch, in_ch, k, k)

    # Mixed_5b, Mixed_5c, ... in the order the caller asked for.
    for index, kind in enumerate(kinds):
        top = f"Mixed_{5 + index}{'bcde'[index % 4]}"
        for branch, kh, kw in _TOPOLOGIES[kind]:
            _conv_bn(t, f"{top}.{branch}", 8, 8, kh, kw)

    t["fc.weight"] = _rand(classes, 16)
    t["fc.bias"] = _rand(classes)
    save_file(t, str(tmp_path / "model.safetensors"))


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write_tiny_inception(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.architecture == "inception_v3"


def test_model_config_requires_the_exact_config_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        ModelConfig.from_dir(tmp_path)


def test_bundle_config_uses_the_inception_input_size_and_normalisation(tmp_path: Path):
    _write_tiny_inception(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)

    model.load_weights(str(tmp_path), cfg, precision="fp32")
    bundle_config = model.get_bundle_config_overrides(cfg)

    # Inception is a 299x299 model normalised to [-1, 1], unlike the 224x224 families.
    assert bundle_config["input_image_h"] == 299
    assert bundle_config["image_mean"] == [0.5, 0.5, 0.5]
    assert bundle_config["image_std"] == [0.5, 0.5, 0.5]


def test_bundle_config_rejects_missing_pretrained_contract(tmp_path: Path):
    _write_tiny_inception(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("pretrained_cfg")
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="requires pretrained_cfg"):
        model.load_weights(str(tmp_path), ModelConfig.from_dir(tmp_path), precision="fp32")


@pytest.mark.parametrize(
    "kind", ["inception_a", "inception_b", "inception_c", "inception_d", "inception_e"]
)
def test_each_block_topology_is_identified_from_its_branch_names(tmp_path: Path, kind: str):
    _write_tiny_inception(tmp_path, kinds=(kind,))
    cfg = ModelConfig.from_dir(tmp_path)

    model.load_weights(str(tmp_path), cfg, precision="fp32")
    blocks = cfg.raw["_timm_inception_config"]["blocks"]

    assert [b["kind"] for b in blocks] == [kind]


def test_blocks_are_ordered_by_name(tmp_path: Path):
    _write_tiny_inception(tmp_path, kinds=("inception_a", "inception_c", "inception_e"))
    cfg = ModelConfig.from_dir(tmp_path)

    model.load_weights(str(tmp_path), cfg, precision="fp32")
    blocks = cfg.raw["_timm_inception_config"]["blocks"]

    assert [b["prefix"] for b in blocks] == ["Mixed_5b", "Mixed_6c", "Mixed_7d"]


def test_load_weights_rejects_an_unrecognised_block(tmp_path: Path):
    _write_tiny_inception(tmp_path)
    from safetensors.numpy import load_file

    tensors = load_file(str(tmp_path / "model.safetensors"))
    kept = {k: v for k, v in tensors.items() if "Mixed_5b" not in k}
    # a Mixed block whose branches match no known topology
    kept["Mixed_5b.branch_unknown.conv.weight"] = _rand(8, 8, 1, 1)
    save_file(kept, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="unrecognised Inception block"):
        model.load_weights(str(tmp_path), cfg, precision="fp32")


def test_load_weights_rejects_an_unconsumed_branch(tmp_path: Path):
    _write_tiny_inception(tmp_path)
    from safetensors.numpy import load_file

    tensors = dict(load_file(str(tmp_path / "model.safetensors")))
    tensors["Mixed_5b.branch_unused.conv.weight"] = _rand(8, 8, 1, 1)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    with pytest.raises(ValueError, match="branch inventory mismatch"):
        model.load_weights(str(tmp_path), ModelConfig.from_dir(tmp_path), precision="fp32")


def test_load_weights_rejects_a_checkpoint_without_mixed_blocks(tmp_path: Path):
    _write_tiny_inception(tmp_path)
    from safetensors.numpy import load_file

    kept = {
        k: v
        for k, v in load_file(str(tmp_path / "model.safetensors")).items()
        if not k.startswith("Mixed_")
    }
    save_file(kept, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="no Mixed_"):
        model.load_weights(str(tmp_path), cfg, precision="fp32")


def test_load_weights_reads_the_exact_inventory_and_precision(tmp_path: Path):
    _write_tiny_inception(tmp_path)
    cfg = ModelConfig.from_dir(tmp_path)
    from safetensors.numpy import load_file

    expected = load_file(str(tmp_path / "model.safetensors"))
    weights = model.load_weights(str(tmp_path), cfg, precision="fp16")

    assert set(weights) == set(expected)
    for name, value in weights.items():
        expected_dtype = np.float32 if ".bn." in name else np.float16
        assert value.dtype == expected_dtype


def test_build_rejects_inception_v4(tmp_path: Path):
    _write_tiny_inception(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architecture"] = "inception_v4"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "unused.bundle",
        family="timm_inception",
        task="classification",
        precision="fp16",
        max_sequence_length=1,
    )

    with pytest.raises(ValueError, match="architecture='inception_v4'"):
        build_family(request, object())


def test_build_rejects_quantization(tmp_path: Path):
    request = BuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "unused.bundle",
        family="timm_inception",
        task="classification",
        precision="fp16",
        quantization="fp8",
    )

    with pytest.raises(NotImplementedError, match="quantization"):
        build_family(request, object())
