# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""glm SwiGLU weight shards must reconstruct the unsharded computation."""

import numpy as np
import pytest

from ..checkpoint_mapper import WeightDict
from ..config import ModelConfig
from ..parallel import ParallelConfig, shard_standard_decoder_weights


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_swiglu_rank_shards_reconstruct_weights_and_output(tp_size, dtype):
    config = ModelConfig(
        hidden_size=8,
        intermediate_size=24,
        num_attention_heads=8,
        num_key_value_heads=8,
    )
    rng = np.random.default_rng(42)
    weights = WeightDict(
        {
            "_attention_size": 8,
            "_kv_attention_size": 8,
            "_mlp_size": 24,
            "embedding": rng.normal(size=(16, 8)).astype(dtype),
            "layer.0.w_gate": rng.normal(scale=0.2, size=(8, 24)).astype(dtype),
            "layer.0.w_up": rng.normal(scale=0.2, size=(8, 24)).astype(dtype),
            "layer.0.w_down": rng.normal(scale=0.2, size=(24, 8)).astype(dtype),
            "layer.0.post_attn_norm": np.ones(8, dtype=dtype),
        }
    )
    original = {
        key: value.copy() for key, value in weights.items() if isinstance(value, np.ndarray)
    }
    shards = [
        shard_standard_decoder_weights(config, weights, ParallelConfig(tp_size, rank))
        for rank in range(tp_size)
    ]

    for key, axis in (("w_gate", 1), ("w_up", 1), ("w_down", 0)):
        full = weights[f"layer.0.{key}"]
        expected_shape = list(full.shape)
        expected_shape[axis] //= tp_size
        for shard in shards:
            part = shard[f"layer.0.{key}"]
            assert part.shape == tuple(expected_shape)
            assert part.dtype == dtype
            assert part.flags.c_contiguous
        np.testing.assert_array_equal(
            np.concatenate([shard[f"layer.0.{key}"] for shard in shards], axis=axis),
            full,
        )

    for rank, shard in enumerate(shards):
        assert shard["_mlp_size"] == 24 // tp_size
        assert isinstance(shard, WeightDict)
        for key in ("embedding", "layer.0.post_attn_norm"):
            np.testing.assert_array_equal(shard[key], weights[key])
        if tp_size > 1:
            assert shard["_tensor_parallel_rank"] == rank
    for key, value in original.items():
        np.testing.assert_array_equal(weights[key], value)
    assert weights["_mlp_size"] == 24

    inputs = rng.normal(size=(3, 8))

    def evaluate(rank_weights):
        gate = inputs @ rank_weights["layer.0.w_gate"].astype(np.float64)
        up = inputs @ rank_weights["layer.0.w_up"].astype(np.float64)
        return (gate / (1 + np.exp(-gate)) * up) @ rank_weights["layer.0.w_down"].astype(np.float64)

    # Row-parallel down projections are summed by the runtime ALL_REDUCE.
    np.testing.assert_allclose(
        sum(evaluate(shard) for shard in shards),
        evaluate(weights),
        rtol=1e-12,
        atol=1e-12,
    )
