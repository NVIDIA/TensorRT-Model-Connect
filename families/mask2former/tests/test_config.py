# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint configuration: malformed or unsupported values are rejected before any engine is built."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from families.mask2former import config as config_module

# The released facebook/mask2former-swin-large-ade-semantic structure, with three of its 150 classes.
_CONFIG = {
    "model_type": "mask2former",
    "architectures": ["Mask2FormerForUniversalSegmentation"],
    "backbone_config": {
        "model_type": "swin",
        "depths": [2, 2, 18, 2],
        "num_heads": [6, 12, 24, 48],
        "embed_dim": 192,
        "patch_size": 4,
        "window_size": 12,
        "mlp_ratio": 4.0,
        "layer_norm_eps": 1e-05,
        "out_features": ["stage1", "stage2", "stage3", "stage4"],
        "hidden_act": "gelu",
        "qkv_bias": True,
        "use_absolute_embeddings": False,
        "num_channels": 3,
    },
    "pre_norm": False,
    "activation_function": "relu",
    "enforce_input_projection": False,
    "feature_size": 256,
    "hidden_dim": 256,
    "mask_feature_size": 256,
    "num_attention_heads": 8,
    "feature_strides": [4, 8, 16, 32],
    "common_stride": 4,
    "encoder_layers": 6,
    "encoder_feedforward_dim": 1024,
    "decoder_layers": 10,
    "dim_feedforward": 2048,
    "num_queries": 100,
    "use_timm_backbone": None,
    "backbone": None,
    "id2label": {"0": "wall", "1": "building", "2": "sky"},
}
_PROCESSOR = {
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.48500001430511475, 0.4560000002384186, 0.4059999883174896],
    "image_std": [0.2290000021457672, 0.2239999920129776, 0.22499999403953552],
    "resample": 2,
    "rescale_factor": 0.00392156862745098,
    "size": {"height": 384, "width": 384},
    "size_divisor": 32,
}


def _load(directory: Path, config: dict | str = _CONFIG, processor: dict | str = _PROCESSOR):
    for name, value in (("config.json", config), ("preprocessor_config.json", processor)):
        text = value if isinstance(value, str) else json.dumps(value)
        (directory / name).write_text(text, encoding="utf-8")
    return config_module.load(directory)


def _with(base: dict, path: tuple[str, ...], value) -> dict:
    result = copy.deepcopy(base)
    target = result
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return result


def test_the_released_configuration_loads(tmp_path: Path) -> None:
    loaded = _load(tmp_path)
    assert loaded.labels == ("wall", "building", "sky")
    assert (loaded.image_height, loaded.image_width) == (384, 384)
    assert (loaded.backbone_eps, loaded.mlp_ratio, loaded.num_queries) == (1e-05, 4.0, 100)


def test_repeated_json_keys_are_rejected(tmp_path: Path) -> None:
    config = json.dumps(_CONFIG).replace(
        '"num_queries": 100', '"num_queries": 100, "num_queries": 50'
    )
    with pytest.raises(ValueError, match="config.json .*repeats the key 'num_queries'"):
        _load(tmp_path, config=config)
    processor = json.dumps(_PROCESSOR).replace(
        '"rescale_factor": ', '"rescale_factor": 1.0, "rescale_factor": '
    )
    with pytest.raises(
        ValueError, match="preprocessor_config.json .*repeats the key 'rescale_factor'"
    ):
        _load(tmp_path, processor=processor)


@pytest.mark.parametrize("key", ["00", "+1", " 1", "1_0"])
def test_class_indices_must_be_canonical(tmp_path: Path, key: str) -> None:
    # int() reads each of these keys as an index another key already holds, or as a gap.
    config = _with(_CONFIG, ("id2label",), {"0": "wall", "1": "building", key: "sky"})
    with pytest.raises(ValueError, match="canonical class indices"):
        _load(tmp_path, config=config)


@pytest.mark.parametrize(
    "value", ["1e-05", True, -1e-05, 0.0, float("nan"), float("inf"), 1e-50, None]
)
def test_the_backbone_norm_epsilon_must_be_a_positive_float32(tmp_path: Path, value) -> None:
    config = _with(_CONFIG, ("backbone_config", "layer_norm_eps"), value)
    with pytest.raises(ValueError, match="layer_norm_eps"):
        _load(tmp_path, config=config)


@pytest.mark.parametrize(
    "value", [True, -4.0, float("nan"), "4.0", None, pytest.param(10**400, id="huge-integer")]
)
def test_the_mlp_ratio_must_be_a_positive_number(tmp_path: Path, value) -> None:
    config = _with(_CONFIG, ("backbone_config", "mlp_ratio"), value)
    with pytest.raises(ValueError, match="mlp_ratio must be a positive number"):
        _load(tmp_path, config=config)


@pytest.mark.parametrize(
    ("section", "path", "value"),
    [
        ("processor", ("resample",), 2.0),
        ("processor", ("resample",), None),
        ("processor", ("size_divisor",), True),
        ("config", ("backbone_config", "num_channels"), 3.0),
    ],
)
def test_integer_fields_accept_only_json_integers(
    tmp_path: Path, section: str, path, value
) -> None:
    if section == "config":
        arguments = {"config": _with(_CONFIG, path, value)}
    else:
        arguments = {"processor": _with(_PROCESSOR, path, value)}
    with pytest.raises(ValueError, match="must be an integer"):
        _load(tmp_path, **arguments)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        # Finite and nonzero, but 1 / 1e-40 overflows float32.
        ("image_std", [0.2290000021457672, 0.2239999920129776, 1e-40]),
        # Finite as float64, infinite as the float32 the runtime stores.
        ("image_mean", [1e39, 0.4560000002384186, 0.4059999883174896]),
        # Infinite as float32, where it would turn every value into a finite zero.
        ("image_std", [1e39, 0.2239999920129776, 0.22499999403953552]),
        # 255 * 1e300 does not fit float32.
        ("rescale_factor", 1e300),
    ],
)
def test_normalization_must_stay_finite_in_float32(tmp_path: Path, field: str, value) -> None:
    with pytest.raises(ValueError, match="float32"):
        _load(tmp_path, processor=_with(_PROCESSOR, (field,), value))
