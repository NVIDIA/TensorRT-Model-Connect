# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for GPT-OSS YaRN RoPE resolution and sliding-window attention.

Trace: ARCH-FAM-001, UD-FAM-GPT-OSS
Intent: GPT-OSS config.json ships ``rope_scaling`` (yarn, factor 32) and
    alternating ``layer_types`` with ``sliding_window: 128``. The engine
    builder previously read only ``rope_parameters`` (silently skipping YaRN)
    and applied a full causal mask to every layer (ignoring the sliding
    window). Both defects produce garbage generation for prompts longer than
    the sliding window while short-prompt smoke tests still pass.
Preconditions: Synthetic tiny GPT-OSS configs/weights; no HF checkpoint.
Postconditions: RoPE tables honor rope_scaling with HF-exact YaRN semantics
    (attention factor, non-truncated correction range) and sliding layers
    attend only within the configured window.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

try:
    # NOTE: never import the ``plugin`` submodule directly here — the import
    # machinery would rebind the package attribute ``gpt_oss.plugin`` from
    # the plugin *instance* to the submodule and break sibling tests.
    import tensorrt_model_connect.families.gpt_oss as gpt_oss
    from tensorrt_model_connect.families.gpt_oss import graph_ops
    from tensorrt_model_connect.families.gpt_oss.config import ModelConfig
    from tensorrt_model_connect.families.gpt_oss.utils import (
        make_rope_half_tables,
        resolve_rope_parameters,
    )
except (ImportError, ModuleNotFoundError):
    pytest.skip("tensorrt_model_connect requires tensorrt", allow_module_level=True)


GPT_OSS_ROPE_SCALING = {
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "factor": 32.0,
    "original_max_position_embeddings": 4096,
    "rope_type": "yarn",
    "truncate": False,
}


def _hf_yarn_reference(
    positions: int,
    head_dim: int,
    base: float,
    factor: float,
    original_max: int,
    beta_fast: float,
    beta_slow: float,
    truncate: bool,
) -> tuple[np.ndarray, float]:
    """Independent port of transformers._compute_yarn_parameters."""

    def find_correction_dim(num_rotations: float) -> float:
        return (head_dim * math.log(original_max / (num_rotations * 2 * math.pi))
                ) / (2 * math.log(base))

    low = find_correction_dim(beta_fast)
    high = find_correction_dim(beta_slow)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    low = max(low, 0)
    high = min(high, head_dim - 1)
    if low == high:
        high += 0.001

    pos_freqs = base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    inv_extra = 1.0 / pos_freqs
    inv_inter = 1.0 / (factor * pos_freqs)
    ramp = np.clip(
        (np.arange(head_dim // 2, dtype=np.float64) - low) / (high - low), 0.0, 1.0)
    extrapolation_factor = 1.0 - ramp
    inv_freq = inv_inter * (1.0 - extrapolation_factor) + inv_extra * extrapolation_factor
    attention_factor = 0.1 * math.log(factor) + 1.0 if factor > 1.0 else 1.0
    return inv_freq, attention_factor


# ---------------------------------------------------------------------------
# RoPE parameter resolution
# ---------------------------------------------------------------------------

def test_resolve_rope_parameters_accepts_rope_scaling_key():
    """GPT-OSS Hub checkpoints serialize the yarn dict under rope_scaling."""
    config = ModelConfig.create_tiny(
        "gpt_oss", rope_scaling=GPT_OSS_ROPE_SCALING)
    assert resolve_rope_parameters(config) == GPT_OSS_ROPE_SCALING


def test_resolve_rope_parameters_prefers_rope_parameters_key():
    params = {"rope_type": "yarn", "factor": 8.0}
    config = ModelConfig.create_tiny(
        "gpt_oss",
        rope_parameters=params, rope_scaling=GPT_OSS_ROPE_SCALING)
    assert resolve_rope_parameters(config) == params


def test_resolve_rope_parameters_defaults_to_empty():
    config = ModelConfig.create_tiny("gpt_oss")
    assert resolve_rope_parameters(config) == {}


def test_make_rope_half_tables_applies_yarn_from_rope_scaling():
    """The original defect: rope_scaling configs fell back to default RoPE."""
    config = ModelConfig.create_tiny(
        "gpt_oss", rope_theta=150000.0,
        rope_scaling=dict(GPT_OSS_ROPE_SCALING))
    head_dim, window = 64, 16

    cos_yarn, sin_yarn = make_rope_half_tables(config, window, head_dim)
    cos_default = graph_ops.make_rope_table_half_dim(
        window, head_dim, 150000.0, True)

    assert not np.allclose(cos_yarn, cos_default), (
        "rope_scaling yarn config must not silently fall back to default RoPE")

    inv_freq, attention_factor = _hf_yarn_reference(
        window, head_dim, 150000.0, 32.0, 4096, 32.0, 1.0, truncate=False)
    positions = np.arange(window, dtype=np.float64)[:, None]
    expected_cos = np.cos(positions * inv_freq[None, :]) * attention_factor
    expected_sin = np.sin(positions * inv_freq[None, :]) * attention_factor
    np.testing.assert_allclose(cos_yarn, expected_cos, rtol=0, atol=1e-5)
    np.testing.assert_allclose(sin_yarn, expected_sin, rtol=0, atol=1e-5)


def test_yarn_table_truncate_flag_changes_correction_range():
    """truncate=False keeps float correction bounds (HF parity)."""
    kwargs = dict(
        scaling_factor=32.0, original_max_position_embeddings=4096,
        beta_fast=32.0, beta_slow=1.0)
    cos_no_trunc = graph_ops.make_yarn_rope_table_half_dim(
        16, 64, 150000.0, True, truncate=False, **kwargs)
    cos_trunc = graph_ops.make_yarn_rope_table_half_dim(
        16, 64, 150000.0, True, truncate=True, **kwargs)
    assert not np.allclose(cos_no_trunc, cos_trunc)


def test_yarn_table_explicit_attention_factor():
    kwargs = dict(
        scaling_factor=32.0, original_max_position_embeddings=4096,
        beta_fast=32.0, beta_slow=1.0, truncate=False)
    cos_default_factor = graph_ops.make_yarn_rope_table_half_dim(
        4, 64, 150000.0, True, **kwargs)
    cos_unit_factor = graph_ops.make_yarn_rope_table_half_dim(
        4, 64, 150000.0, True, attention_factor=1.0, **kwargs)
    expected_factor = 0.1 * math.log(32.0) + 1.0
    np.testing.assert_allclose(
        cos_default_factor, cos_unit_factor * expected_factor,
        rtol=0, atol=1e-6)
    # Position 0 cosine is exactly the attention factor.
    np.testing.assert_allclose(
        cos_default_factor[0], np.full(32, expected_factor),
        rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# Sliding-window attention (requires a CUDA-capable TensorRT build)
# ---------------------------------------------------------------------------

def _tiny_gpt_oss_config(layer_types: list[str], sliding_window: int) -> ModelConfig:
    return ModelConfig.create_tiny(
        "gpt_oss",
        vocab_size=32,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=150000.0,
        num_local_experts=2,
        num_experts_per_tok=1,
        layer_types=layer_types,
        sliding_window=sliding_window,
    )


def _tiny_gpt_oss_weights(config: ModelConfig, seed: int = 7):
    from tensorrt_model_connect.families.gpt_oss.checkpoint_mapper import WeightDict

    rng = np.random.default_rng(seed)
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_heads = config.num_attention_heads
    num_kv = config.num_key_value_heads
    head_dim = config.head_dim
    attn = num_heads * head_dim
    kv_attn = num_kv * head_dim
    experts = config.raw["num_local_experts"]
    inter = 16

    def w(*shape):
        return rng.standard_normal(shape).astype(np.float32) * 0.1

    weights = WeightDict()
    weights["embedding"] = w(vocab, hidden)
    weights["final_norm"] = np.ones(hidden, dtype=np.float32)
    weights["w_out"] = w(hidden, vocab)
    for i in range(config.num_hidden_layers):
        p = f"layer.{i}"
        weights[f"{p}.input_norm"] = np.ones(hidden, dtype=np.float32)
        weights[f"{p}.post_attn_norm"] = np.ones(hidden, dtype=np.float32)
        weights[f"{p}.w_q"] = w(hidden, attn)
        weights[f"{p}.w_k"] = w(hidden, kv_attn)
        weights[f"{p}.w_v"] = w(hidden, kv_attn)
        weights[f"{p}.w_o"] = w(attn, hidden)
        weights[f"{p}.q_bias"] = w(attn)
        weights[f"{p}.k_bias"] = w(kv_attn)
        weights[f"{p}.v_bias"] = w(kv_attn)
        weights[f"{p}.o_bias"] = w(hidden)
        weights[f"{p}.sinks"] = w(num_heads)
        weights[f"{p}.router"] = w(hidden, experts)
        weights[f"{p}.router_bias"] = w(experts)
        for e in range(experts):
            weights[f"{p}.expert.{e}.w_gate"] = w(hidden, inter)
            weights[f"{p}.expert.{e}.w_up"] = w(hidden, inter)
            weights[f"{p}.expert.{e}.w_down"] = w(inter, hidden)
            weights[f"{p}.expert.{e}.gate_bias"] = w(inter)
            weights[f"{p}.expert.{e}.up_bias"] = w(inter)
            weights[f"{p}.expert.{e}.down_bias"] = w(hidden)
    weights["_attention_size"] = attn
    weights["_num_experts"] = experts
    weights["_moe_intermediate_size"] = inter
    weights["_num_experts_per_tok"] = config.raw["num_experts_per_tok"]
    return weights


def _require_cuda() -> None:
    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart  # type: ignore[no-redef]
        except ImportError:
            pytest.skip("cuda-python is required for engine execution tests")
    err, count = cudart.cudaGetDeviceCount()
    if int(err) != 0 or count == 0:
        pytest.skip("No CUDA device available for engine execution tests")


@pytest.mark.gpu
@pytest.mark.trt
def test_sliding_layers_restrict_attention_to_window():
    """Engines with sliding layer_types must diverge from all-full engines
    exactly when the prompt outgrows the sliding window."""
    _require_cuda()
    from tensorrt_model_connect.families.gpt_oss.debug_runner import TrtRunner

    sliding_window = 4
    max_cache = 12
    num_layers = 2
    tokens = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

    plugin = gpt_oss.plugin
    logits_by_mode: dict[str, list[np.ndarray]] = {}
    for mode, layer_types in (
        ("sliding", ["sliding_attention", "full_attention"]),
        ("full", ["full_attention", "full_attention"]),
    ):
        config = _tiny_gpt_oss_config(layer_types, sliding_window)
        weights = _tiny_gpt_oss_weights(config)
        plan = plugin.build_engine(
            config, weights, max_cache, precision="fp32")
        runner = TrtRunner(
            engine_plan=plan, max_cache_length=max_cache,
            num_layers=num_layers)
        logits_by_mode[mode] = [
            runner.step(t)["logits"].flatten().copy() for t in tokens]
        del runner

    # While the context fits in the window the two engines are identical
    # (sliding penalty masks nothing until position_id > window - 1).
    for step in range(sliding_window):
        np.testing.assert_allclose(
            logits_by_mode["sliding"][step], logits_by_mode["full"][step],
            rtol=0, atol=1e-4,
            err_msg=f"step {step} must be unaffected by the sliding window")

    # Once the context exceeds the window the sliding engine must diverge.
    late_diffs = [
        float(np.max(np.abs(
            logits_by_mode["sliding"][step] - logits_by_mode["full"][step])))
        for step in range(sliding_window, len(tokens))
    ]
    assert max(late_diffs) > 1e-3, (
        "sliding_attention layers did not restrict attention to the window; "
        f"per-step max diffs: {late_diffs}")
