"""Tests for decoder recipe and sharding policy layers."""

from __future__ import annotations

import numpy as np

from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.model_recipe import standard_decoder_recipe
from tensorrt_model_connect.parallel_config import ParallelConfig
from tensorrt_model_connect.sharding_policy import standard_decoder_sharding_policy


def _tiny_weights() -> dict[str, object]:
    weights: dict[str, object] = {}
    weights["_attention_size"] = 16
    weights["_kv_attention_size"] = 16
    weights["_mlp_size"] = 32
    weights["layer.0.w_q"] = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    weights["layer.0.w_k"] = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    weights["layer.0.w_v"] = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    weights["layer.0.w_o"] = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    weights["layer.0.w_gate"] = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    weights["layer.0.w_up"] = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    weights["layer.0.w_down"] = np.arange(32 * 16, dtype=np.float32).reshape(32, 16)
    weights["w_out"] = np.arange(16 * 100, dtype=np.float32).reshape(16, 100)
    return weights


def test_standard_decoder_recipe_names_shardable_regions() -> None:
    cfg = ModelConfig.create_tiny("qwen3")
    recipe = standard_decoder_recipe(cfg, family="qwen")

    assert recipe.component == "decoder"
    assert "decoder.layers.0.self_attn" in recipe.region_names()
    assert "decoder.layers.0.mlp" in recipe.region_names()
    assert "decoder.lm_head" in recipe.region_names()


def test_sharding_policy_preserves_single_device_weights() -> None:
    cfg = ModelConfig.create_tiny("qwen3")
    weights = _tiny_weights()
    policy = standard_decoder_sharding_policy(cfg, weights, ParallelConfig())

    assert policy.shard_weights() is weights


def test_sharding_policy_slices_decoder_tp_weights() -> None:
    cfg = ModelConfig.create_tiny("qwen3")
    weights = _tiny_weights()
    policy = standard_decoder_sharding_policy(
        cfg,
        weights,
        ParallelConfig(mode="tensor_parallel", tp_size=2, rank=1),
    )

    shard = policy.shard_weights()

    np.testing.assert_allclose(shard["layer.0.w_q"], weights["layer.0.w_q"][:, 8:])
    np.testing.assert_allclose(shard["layer.0.w_k"], weights["layer.0.w_k"][:, 8:])
    np.testing.assert_allclose(shard["layer.0.w_o"], weights["layer.0.w_o"][8:, :])
    np.testing.assert_allclose(shard["layer.0.w_gate"], weights["layer.0.w_gate"][:, 16:])
    np.testing.assert_allclose(shard["layer.0.w_down"], weights["layer.0.w_down"][16:, :])
    np.testing.assert_allclose(shard["w_out"], weights["w_out"])
    assert shard["_attention_size"] == 8
    assert shard["_kv_attention_size"] == 8
    assert shard["_mlp_size"] == 16


def test_sharding_policy_emits_region_and_collective_plans() -> None:
    cfg = ModelConfig.create_tiny("qwen3")
    weights = _tiny_weights()
    recipe = standard_decoder_recipe(cfg, family="qwen")
    policy = standard_decoder_sharding_policy(
        cfg,
        weights,
        ParallelConfig(mode="tensor_parallel", tp_size=2, rank=0),
        recipe=recipe,
    )

    assert {region.selector for region in policy.region_plans()} == {
        "decoder.layers[*].self_attn",
        "decoder.layers[*].mlp",
    }
    assert {collective.group for collective in policy.collective_plans()} == {"tp"}
