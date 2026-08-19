# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic tests for MagpieTTS tensor-parallel decoder sharding."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")


pytest.importorskip(
    "tensorrt_model_connect.config",
    reason="tensorrt_model_connect requires tensorrt",
)

from tensorrt_model_connect.parallel_config import ParallelConfig


_LAYERS = 2
_HIDDEN = 16
_HEADS = 4
_FFN = 64
_MAX_SRC = 8
_XA_HEADS = 1
_XA_D_HEAD = 4
_NUM_CODEBOOKS = 2
_CODEBOOK_SIZE = 8


def _magpie_tp_builder_module():
    return pytest.importorskip(
        "tensorrt_model_connect.models.magpie_tts.decoder_tp_builder",
        reason="TensorRT is required for MagpieTTS TP builder tests",
    )


def _make_magpie_weights() -> dict[str, Any]:
    rng = np.random.RandomState(123)

    def rand(*shape: int) -> np.ndarray:
        return rng.randn(*shape).astype(np.float32)

    weights = {
        "_dec_layers": _LAYERS,
        "_dec_heads": _HEADS,
        "_dec_ffn": _FFN,
        "_hidden_size": _HIDDEN,
        "_num_codebooks": _NUM_CODEBOOKS,
        "_codebook_size": _CODEBOOK_SIZE,
        "_max_source_positions": _MAX_SRC,
        "_xa_n_heads": _XA_HEADS,
        "_xa_d_head": _XA_D_HEAD,
        "dec_pos_embedding": rand(16, _HIDDEN),
        "final_norm": rand(_HIDDEN),
        "w_out": rand(_HIDDEN, _NUM_CODEBOOKS * _CODEBOOK_SIZE),
        "w_out_bias": rand(_NUM_CODEBOOKS * _CODEBOOK_SIZE),
        "baked_context_lengths": np.array([3, 5], dtype=np.int32),
    }
    for i in range(_LAYERS):
        pfx = f"layer.{i}"
        for key in ("w_q", "w_k", "w_v", "w_o"):
            weights[f"{pfx}.{key}"] = rand(_HIDDEN, _HIDDEN)
        weights[f"{pfx}.w_fc1"] = rand(_HIDDEN, _FFN)
        weights[f"{pfx}.w_fc2"] = rand(_FFN, _HIDDEN)
        weights[f"{pfx}.input_norm"] = rand(_HIDDEN)
        weights[f"{pfx}.post_attn_norm"] = rand(_HIDDEN)
        weights[f"{pfx}.norm_xattn_query"] = rand(_HIDDEN)
        weights[f"{pfx}.norm_xattn_memory"] = rand(_HIDDEN)
        weights[f"{pfx}.cross_w_q"] = rand(_HIDDEN, _XA_HEADS * _XA_D_HEAD)
        weights[f"{pfx}.cross_w_k"] = rand(_HIDDEN, _XA_HEADS * _XA_D_HEAD)
        weights[f"{pfx}.cross_w_v"] = rand(_HIDDEN, _XA_HEADS * _XA_D_HEAD)
        weights[f"{pfx}.cross_w_o"] = rand(_XA_HEADS * _XA_D_HEAD, _HIDDEN)
    return weights


def test_magpie_tp_build_rejects_single_device_parallel_config():
    decoder_tp_builder = _magpie_tp_builder_module()

    with pytest.raises(ValueError, match="enabled parallel config"):
        decoder_tp_builder.build_magpie_tp_decoder_engine(
            object(),
            _make_magpie_weights(),
            max_cache_length=4,
            parallel_config=ParallelConfig(),
        )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("_hidden_size", _HIDDEN + 1, "hidden_size"),
        ("_dec_heads", _HEADS + 1, "decoder heads"),
        ("_dec_ffn", _FFN + 1, "FFN size"),
    ],
)
def test_magpie_tp_validation_rejects_non_divisible_dimensions(key, value, message):
    decoder_tp_builder = _magpie_tp_builder_module()
    weights = _make_magpie_weights()
    weights[key] = value

    with pytest.raises(ValueError, match=message):
        decoder_tp_builder.validate_magpie_decoder_tp(
            weights,
            ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
        )


@pytest.mark.parametrize(
    ("key", "shape", "message"),
    [
        ("layer.0.w_q", (_HIDDEN, _HIDDEN - 1), "last dimension"),
        ("layer.0.w_o", (_HIDDEN - 1, _HIDDEN), "first dimension"),
    ],
)
def test_magpie_tp_sharding_rejects_unshardable_weight_shapes(key, shape, message):
    decoder_tp_builder = _magpie_tp_builder_module()
    weights = _make_magpie_weights()
    weights[key] = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        decoder_tp_builder.shard_magpie_decoder_weights(
            weights,
            ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
        )


def test_magpie_tp_shards_rank_local_decoder_weights():
    decoder_tp_builder = _magpie_tp_builder_module()
    weights = _make_magpie_weights()
    shard = decoder_tp_builder.shard_magpie_decoder_weights(
        weights,
        ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1),
    )

    assert shard["_tensor_parallel_size"] == 2
    assert shard["_tensor_parallel_rank"] == 1
    assert shard["dec_pos_embedding"] is weights["dec_pos_embedding"]
    assert shard["final_norm"] is weights["final_norm"]
    assert shard["layer.0.cross_w_q"] is weights["layer.0.cross_w_q"]
    np.testing.assert_array_equal(
        shard["layer.0.w_q"],
        weights["layer.0.w_q"][:, _HIDDEN // 2:],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_k"],
        weights["layer.0.w_k"][:, _HIDDEN // 2:],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_v"],
        weights["layer.0.w_v"][:, _HIDDEN // 2:],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_o"],
        weights["layer.0.w_o"][_HIDDEN // 2:, :],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_fc1"],
        weights["layer.0.w_fc1"][:, _FFN // 2:],
    )
    np.testing.assert_array_equal(
        shard["layer.0.w_fc2"],
        weights["layer.0.w_fc2"][_FFN // 2:, :],
    )
