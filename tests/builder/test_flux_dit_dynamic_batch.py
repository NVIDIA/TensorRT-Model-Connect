"""Smoke tests for the FLUX.1 dynamic-batch DiT builder path."""

from __future__ import annotations

import numpy as np

from .conftest import requires_trt


def _rng_for(prefix: str) -> np.random.Generator:
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(prefix)) % (2**32)
    return np.random.default_rng(seed)


def _linear(weights: dict[str, np.ndarray], key: str, in_dim: int, out_dim: int) -> None:
    rng = _rng_for(key)
    weights[f"{key}.weight"] = rng.normal(0.0, 0.01, (in_dim, out_dim)).astype(np.float32)
    weights[f"{key}.bias"] = np.zeros((out_dim,), dtype=np.float32)


def _joint_block_weights(prefix: str, *, dim: int, head_dim: int, ffn_dim: int) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    _linear(weights, f"{prefix}.norm1.linear", dim, 6 * dim)
    _linear(weights, f"{prefix}.norm1_context.linear", dim, 6 * dim)

    for name in ("to_q", "to_k", "to_v"):
        _linear(weights, f"{prefix}.attn.{name}", dim, dim)
    _linear(weights, f"{prefix}.attn.to_out.0", dim, dim)

    for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
        _linear(weights, f"{prefix}.attn.{name}", dim, dim)
    _linear(weights, f"{prefix}.attn.to_add_out", dim, dim)

    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        weights[f"{prefix}.attn.{name}.weight"] = np.ones((head_dim,), dtype=np.float32)

    _linear(weights, f"{prefix}.ff.net.0.proj", dim, ffn_dim)
    _linear(weights, f"{prefix}.ff.net.2", ffn_dim, dim)
    _linear(weights, f"{prefix}.ff_context.net.0.proj", dim, ffn_dim)
    _linear(weights, f"{prefix}.ff_context.net.2", ffn_dim, dim)
    return weights


def _single_block_weights(prefix: str, *, dim: int, head_dim: int, ffn_dim: int) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    _linear(weights, f"{prefix}.norm.linear", dim, 3 * dim)
    for name in ("to_q", "to_k", "to_v"):
        _linear(weights, f"{prefix}.attn.{name}", dim, dim)
    for name in ("norm_q", "norm_k"):
        weights[f"{prefix}.attn.{name}.weight"] = np.ones((head_dim,), dtype=np.float32)
    _linear(weights, f"{prefix}.proj_mlp", dim, ffn_dim)
    _linear(weights, f"{prefix}.proj_out", dim + ffn_dim, dim)
    return weights


def _tiny_weights() -> dict[str, np.ndarray]:
    dim = 8
    head_dim = 4
    ffn_dim = 16
    out_channels = 4
    weights: dict[str, np.ndarray] = {}
    weights.update(_joint_block_weights(
        "transformer_blocks.0", dim=dim, head_dim=head_dim, ffn_dim=ffn_dim))
    weights.update(_single_block_weights(
        "single_transformer_blocks.0", dim=dim, head_dim=head_dim, ffn_dim=ffn_dim))
    _linear(weights, "norm_out.linear", dim, 2 * dim)
    _linear(weights, "proj_out", dim, out_channels)
    return weights


@requires_trt
def test_flux_dit_dynamic_batch_engine_builds_tiny_network():
    import tensorrt as trt
    from tensorrt_model_connect.families.flux.flux_dit_builder import (
        build_flux_dit_engine,
    )

    plan = build_flux_dit_engine(
        _tiny_weights(),
        dim=8,
        num_heads=2,
        num_layers=1,
        num_single_layers=1,
        num_img_tokens=2,
        text_seq_len=3,
        mlp_ratio=2.0,
        max_batch_size=2,
        verbose=False,
    )

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    ctx = engine.create_execution_context()

    for batch in (1, 2):
        assert ctx.set_input_shape("hidden_states", (batch, 2, 8))
        assert ctx.set_input_shape("encoder_hidden_states", (batch, 3, 8))
        assert ctx.set_input_shape("temb", (batch, 8))
        assert ctx.set_input_shape("rotary_cos", (batch, 5, 4))
        assert ctx.set_input_shape("rotary_sin", (batch, 5, 4))
        assert tuple(ctx.get_tensor_shape("output")) == (batch, 2, 4)

    assert ctx.set_input_shape("hidden_states", (3, 2, 8)) is False
