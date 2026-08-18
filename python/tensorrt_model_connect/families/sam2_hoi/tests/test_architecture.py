# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import numpy as np
import pytest

from tensorrt_model_connect.families.sam2_hoi.architecture import (
    ARCHITECTURE,
    validate_architecture,
)
from tensorrt_model_connect.families.sam2_hoi.interaction_builder import (
    _add_constant,
    interaction_mlp_numpy,
)


def _raw_config() -> dict:
    return {
        "image_size": 1024,
        "sam2_hoi": {
            "variant": "sam2.1_hiera_small_hoi_c4",
            "hiera_embed_dim": 96,
            "hiera_stages": [1, 2, 11, 2],
            "hiera_global_attention_blocks": [7, 10, 13],
            "fpn_hidden_size": 256,
            "hoi_num_queries": 1500,
            "hoi_num_classes": 4,
            "hoi_num_feature_levels": 3,
            "hoi_encoder_layers": 6,
            "hoi_decoder_layers": 6,
            "memory_attention_layers": 4,
            "memory_channels": 64,
            "num_mask_memory_frames": 7,
            "score_threshold": 0.35,
            "class_nms_threshold": 0.5,
            "global_nms_threshold": 0.75,
            "hand_nms_threshold": 0.25,
            "interaction_threshold": 0.5,
            "mask_logit_threshold": 0.01,
        },
    }


def test_architecture_accepts_only_reviewed_contract() -> None:
    assert validate_architecture(_raw_config()) is ARCHITECTURE
    changed = copy.deepcopy(_raw_config())
    changed["sam2_hoi"]["hoi_num_queries"] = 900
    with pytest.raises(RuntimeError, match="expected 1500, got 900"):
        validate_architecture(changed)


def test_architecture_exposes_fixed_stage_shapes() -> None:
    assert ARCHITECTURE.tracker_feature_shapes == (
        (1, 32, 256, 256),
        (1, 64, 128, 128),
        (1, 256, 64, 64),
    )
    assert ARCHITECTURE.detector_feature_shapes[-1] == (1, 256, 32, 32)
    config = ARCHITECTURE.bundle_config()
    assert config["sam2_hoi_object_batch"] == 2
    assert config["image_mean"] == [0.485, 0.456, 0.406]
    assert config["image_interpolation"] == "bicubic"


def test_interaction_numpy_oracle_matches_zero_weight_softmax() -> None:
    weights = {
        "image_encoder.hoi_head.query_head.interaction_head.mlp.0.weight": np.zeros(
            (256, 512), dtype=np.float32
        ),
        "image_encoder.hoi_head.query_head.interaction_head.mlp.0.bias": np.zeros(
            256, dtype=np.float32
        ),
        "image_encoder.hoi_head.query_head.interaction_head.mlp.2.weight": np.zeros(
            (256, 256), dtype=np.float32
        ),
        "image_encoder.hoi_head.query_head.interaction_head.mlp.2.bias": np.zeros(
            256, dtype=np.float32
        ),
        "image_encoder.hoi_head.query_head.interaction_head.mlp.4.weight": np.zeros(
            (2, 256), dtype=np.float32
        ),
        "image_encoder.hoi_head.query_head.interaction_head.mlp.4.bias": np.array(
            [-1.0, 1.0], dtype=np.float32
        ),
    }
    result = interaction_mlp_numpy(np.ones((3, 512), dtype=np.float32), weights)
    expected = np.exp(np.array([-1.0, 1.0], dtype=np.float32))
    expected /= expected.sum()
    np.testing.assert_allclose(result, np.tile(expected, (3, 1)), rtol=1e-6, atol=1e-6)


def test_interaction_numpy_oracle_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="shape .pairs, 512."):
        interaction_mlp_numpy(np.zeros((1, 256), dtype=np.float32), {})


def test_interaction_constants_use_contiguous_tensorrt_weights() -> None:
    captured: dict[str, object] = {}
    output = object()

    class FakeWeights:
        def __init__(self, values: np.ndarray) -> None:
            assert values.dtype == np.float32
            assert values.flags.c_contiguous
            captured["values"] = values.copy()

    class FakeLayer:
        @staticmethod
        def get_output(index: int):
            assert index == 0
            return output

    class FakeNetwork:
        @staticmethod
        def add_constant(shape, weights):
            captured["shape"] = tuple(shape)
            captured["weights"] = weights
            return FakeLayer()

    class FakeTrt:
        Weights = FakeWeights

    source = np.arange(12, dtype=np.float64).reshape(3, 4).T
    assert not source.flags.c_contiguous
    assert _add_constant(FakeNetwork(), FakeTrt, source) is output
    assert captured["shape"] == (4, 3)
    assert isinstance(captured["weights"], FakeWeights)
    np.testing.assert_array_equal(captured["values"], source.astype(np.float32))
