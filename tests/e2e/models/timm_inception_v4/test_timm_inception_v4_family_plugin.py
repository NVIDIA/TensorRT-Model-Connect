# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the timm Inception-v4 image-classification family plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


try:
    from safetensors.numpy import save_file
    from tensorrt_model_connect.config import ModelConfig
    from tensorrt_model_connect.families.timm_inception_v4 import plugin
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires TensorRT", allow_module_level=True)


def _rand(*shape: int) -> np.ndarray:
    return np.random.RandomState(7).randn(*shape).astype(np.float32)


def _conv(t: dict, prefix: str, out_ch: int, in_ch: int, kh: int, kw: int) -> None:
    t[f"{prefix}.conv.weight"] = _rand(out_ch, in_ch, kh, kw)
    t[f"{prefix}.bn.weight"] = _rand(out_ch)
    t[f"{prefix}.bn.bias"] = _rand(out_ch)
    t[f"{prefix}.bn.running_mean"] = _rand(out_ch)
    t[f"{prefix}.bn.running_var"] = np.abs(_rand(out_ch)) + 1.0


_STEM_BLOCKS = (0, 1, 2)


def _stem(t: dict) -> None:
    _conv(t, "features.0", 8, 3, 3, 3)
    _conv(t, "features.1", 8, 8, 3, 3)
    _conv(t, "features.2", 8, 8, 3, 3)


def _write(tmp_path: Path, kinds: list[str]) -> None:
    """Write a miniature Inception-v4 carrying the requested block shapes.

    The blocks are laid out contiguously after the three stem convolutions,
    because the loader requires the features indices to have no gaps.
    """
    classes = 5
    config = {
        "architecture": "inception_v4",
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
    _stem(t)
    for offset, kind in enumerate(kinds):
        index = len(_STEM_BLOCKS) + offset
        f = f"features.{index}"
        if kind in ("mixed3a", "mixed5a"):
            _conv(t, f"{f}.conv", 8, 8, 3, 3)
        elif kind == "mixed4a":
            _conv(t, f"{f}.branch0.0", 8, 8, 1, 1)
            _conv(t, f"{f}.branch0.1", 8, 8, 3, 3)
            _conv(t, f"{f}.branch1.0", 8, 8, 1, 1)
            _conv(t, f"{f}.branch1.1", 8, 8, 1, 7)
            _conv(t, f"{f}.branch1.2", 8, 8, 7, 1)
            _conv(t, f"{f}.branch1.3", 8, 8, 3, 3)
        elif kind == "reduction_a":
            # the distinguishing feature: branch0 is a single convolution
            _conv(t, f"{f}.branch0", 8, 8, 3, 3)
            _conv(t, f"{f}.branch1.0", 8, 8, 1, 1)
            _conv(t, f"{f}.branch1.1", 8, 8, 3, 3)
            _conv(t, f"{f}.branch1.2", 8, 8, 3, 3)
        elif kind == "reduction_b":
            _conv(t, f"{f}.branch0.0", 8, 8, 1, 1)
            _conv(t, f"{f}.branch0.1", 8, 8, 3, 3)
            _conv(t, f"{f}.branch1.0", 8, 8, 1, 1)
            _conv(t, f"{f}.branch1.1", 8, 8, 1, 7)
            _conv(t, f"{f}.branch1.2", 8, 8, 7, 1)
            _conv(t, f"{f}.branch1.3", 8, 8, 3, 3)
        elif kind == "inception_ab":
            _conv(t, f"{f}.branch0", 8, 8, 1, 1)
            _conv(t, f"{f}.branch1.0", 8, 8, 1, 1)
            _conv(t, f"{f}.branch1.1", 8, 8, 3, 3)
            _conv(t, f"{f}.branch2.0", 8, 8, 1, 1)
            _conv(t, f"{f}.branch2.1", 8, 8, 3, 3)
            _conv(t, f"{f}.branch2.2", 8, 8, 3, 3)
            _conv(t, f"{f}.branch3.1", 8, 8, 1, 1)
        elif kind == "inception_c":
            _conv(t, f"{f}.branch0", 8, 8, 1, 1)
            _conv(t, f"{f}.branch1_0", 8, 8, 1, 1)
            _conv(t, f"{f}.branch1_1a", 8, 8, 1, 3)
            _conv(t, f"{f}.branch1_1b", 8, 8, 3, 1)
            _conv(t, f"{f}.branch2_0", 8, 8, 1, 1)
            _conv(t, f"{f}.branch2_1", 8, 8, 3, 1)
            _conv(t, f"{f}.branch2_2", 8, 8, 1, 3)
            _conv(t, f"{f}.branch2_3a", 8, 8, 1, 3)
            _conv(t, f"{f}.branch2_3b", 8, 8, 3, 1)
            _conv(t, f"{f}.branch3.1", 8, 8, 1, 1)
    t["last_linear.weight"] = _rand(classes, 16)
    t["last_linear.bias"] = _rand(classes)
    save_file(t, str(tmp_path / "model.safetensors"))


def test_model_config_uses_timm_architecture_when_model_type_absent(tmp_path: Path):
    _write(tmp_path, ["mixed3a"])
    cfg = ModelConfig.from_dir(tmp_path)

    assert cfg.model_type == "inception_v4"
    assert plugin.matches(cfg.model_type)


@pytest.mark.parametrize("model_type", ["inception_v4", "timm_inception_v4"])
def test_plugin_matches_inception_v4(model_type: str):
    assert plugin.matches(model_type)


@pytest.mark.parametrize(
    "model_type", ["inception_v3", "inception_resnet_v2", "resnet50", "xception41", ""]
)
def test_plugin_rejects_unrelated_model_types(model_type: str):
    """v3 belongs to timm_inception; the two families must not overlap."""
    assert not plugin.matches(model_type)


def test_bundle_config_uses_the_inception_input_size_and_normalisation(tmp_path: Path):
    _write(tmp_path, ["mixed3a"])
    cfg = ModelConfig.from_dir(tmp_path)

    bundle_config = plugin.get_bundle_config_overrides(cfg)

    assert bundle_config["runtime_strategy"] == "timm_inception_v4_image_classification"
    assert bundle_config["input_image_h"] == 299
    assert bundle_config["image_mean"] == [0.5, 0.5, 0.5]


@pytest.mark.parametrize(
    "kind", ["mixed3a", "mixed4a", "inception_ab", "inception_c"],
)
def test_each_block_topology_is_identified(tmp_path: Path, kind: str):
    _write(tmp_path, [kind])
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    blocks = cfg.raw["_timm_inception_v4_config"]["blocks"]

    assert [b["kind"] for b in blocks[3:]] == [kind]


def test_the_two_reductions_are_told_apart_by_their_first_branch(tmp_path: Path):
    """Reduction-A's first branch is a single convolution, so it is
    distinguishable from the chained pair. Mixed4a and Reduction-B are not
    distinguishable by shape at all, so the first chained pair is taken as
    Mixed4a and any later one as Reduction-B."""
    _write(tmp_path, ["reduction_a", "mixed4a", "reduction_b"])
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    found = [b["kind"] for b in cfg.raw["_timm_inception_v4_config"]["blocks"][3:]]

    assert found == ["reduction_a", "mixed4a", "reduction_b"]


def test_stem_blocks_are_taken_by_index_not_by_branch_names(tmp_path: Path):
    _write(tmp_path, ["mixed3a"])
    cfg = ModelConfig.from_dir(tmp_path)

    plugin.load_weights(str(tmp_path), cfg)
    blocks = cfg.raw["_timm_inception_v4_config"]["blocks"]

    assert [b["kind"] for b in blocks[:3]] == ["stem", "stem", "stem"]


def test_load_weights_rejects_an_unrecognised_block(tmp_path: Path):
    _write(tmp_path, ["inception_ab"])
    from safetensors.numpy import load_file

    tensors = load_file(str(tmp_path / "model.safetensors"))
    kept = {k: v for k, v in tensors.items() if not k.startswith("features.3.")}
    kept["features.3.branch_unknown.conv.weight"] = _rand(8, 8, 1, 1)
    save_file(kept, str(tmp_path / "model.safetensors"))
    cfg = ModelConfig.from_dir(tmp_path)

    with pytest.raises(ValueError, match="unrecognised Inception-v4 block"):
        plugin.load_weights(str(tmp_path), cfg)


def test_build_engine_rejects_quantized_context(tmp_path: Path):
    _write(tmp_path, ["mixed3a"])
    cfg = ModelConfig.from_dir(tmp_path)
    weights = plugin.load_weights(str(tmp_path), cfg)

    with pytest.raises(NotImplementedError, match="quantized"):
        plugin.build_engine(cfg, weights, 0, quant_ctx=object())
