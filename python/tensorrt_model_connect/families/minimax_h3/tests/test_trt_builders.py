# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest


trt = pytest.importorskip("tensorrt")

from tensorrt_model_connect.families.minimax_h3.adaln_builder import (  # noqa: E402
    build_adaln_precompute_engine,
)
from tensorrt_model_connect.families.minimax_h3.config import MiniMaxH3Config  # noqa: E402
from tensorrt_model_connect.families.minimax_h3.dit_builder import build_dit_engine  # noqa: E402
from tensorrt_model_connect.families.minimax_h3 import graph_ops as op  # noqa: E402


def _weights(profile: MiniMaxH3Config) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)

    def weight(out_features: int, in_features: int) -> np.ndarray:
        return rng.normal(0.0, 0.02, (out_features, in_features)).astype(np.float32)

    def bias(features: int) -> np.ndarray:
        return np.zeros((features,), dtype=np.float32)

    state = {
        "proj_in.weight": weight(profile.hidden_size, profile.video_patch_dim),
        "proj_in.bias": bias(profile.hidden_size),
        "audio_proj_in.weight": weight(profile.hidden_size, profile.audio_in_channels),
        "audio_proj_in.bias": bias(profile.hidden_size),
        "context_embedder.weight": weight(profile.hidden_size, profile.text_dim),
        "context_embedder.bias": bias(profile.hidden_size),
        "time_embedder.linear_1.weight": weight(
            profile.timestep_hidden_size, profile.timestep_input_dim
        ),
        "time_embedder.linear_1.bias": bias(profile.timestep_hidden_size),
        "time_embedder.linear_2.weight": weight(
            profile.timestep_embed_dim, profile.timestep_hidden_size
        ),
        "time_embedder.linear_2.bias": bias(profile.timestep_embed_dim),
        "token_refiner.final_norm.weight": np.ones(profile.hidden_size, np.float32),
        "norm_out.norm.weight": np.ones(profile.hidden_size, np.float32),
        "norm_out.linear.weight": weight(2 * profile.hidden_size, profile.timestep_embed_dim),
        "norm_out.linear.bias": bias(2 * profile.hidden_size),
        "proj_out.weight": weight(profile.video_patch_dim, profile.hidden_size),
        "proj_out.bias": bias(profile.video_patch_dim),
        "audio_proj_out.weight": weight(profile.audio_in_channels, profile.hidden_size),
        "audio_proj_out.bias": bias(profile.audio_in_channels),
    }
    for prefix in [
        *(f"token_refiner.refiner_blocks.{i}" for i in range(profile.num_refiner_layers)),
        *(f"transformer_blocks.{i}" for i in range(profile.num_layers)),
    ]:
        state[f"{prefix}.norm1.weight"] = np.ones(profile.hidden_size, np.float32)
        state[f"{prefix}.norm2.weight"] = np.ones(profile.hidden_size, np.float32)
        for name in ("q", "k", "v"):
            state[f"{prefix}.attn.to_{name}.weight"] = weight(
                profile.attention_size, profile.hidden_size
            )
        state[f"{prefix}.attn.norm_q.weight"] = np.ones(profile.head_dim, np.float32)
        state[f"{prefix}.attn.norm_k.weight"] = np.ones(profile.head_dim, np.float32)
        state[f"{prefix}.attn.to_out.0.weight"] = weight(
            profile.hidden_size, profile.attention_size
        )
        state[f"{prefix}.ff.net.0.proj.weight"] = weight(2 * profile.ffn_dim, profile.hidden_size)
        state[f"{prefix}.ff.net.2.weight"] = weight(profile.hidden_size, profile.ffn_dim)
    for index in range(profile.num_layers):
        prefix = f"transformer_blocks.{index}.adaln_proj.linear"
        state[f"{prefix}.weight"] = weight(18 * profile.hidden_size, profile.timestep_embed_dim)
        state[f"{prefix}.bias"] = bias(18 * profile.hidden_size)
    return state


@pytest.mark.gpu
def test_tiny_native_h3_graphs_serialize() -> None:
    profile = MiniMaxH3Config(
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        num_heads=4,
        head_dim=8,
        ffn_dim=16,
        video_in_channels=2,
        audio_in_channels=2,
        text_dim=8,
        timestep_input_dim=4,
        timestep_hidden_size=8,
        timestep_embed_dim=4,
        rope_freq_dim=1,
        video_rows=4,
        audio_rows=2,
        text_rows=2,
        padded_sequence_length=8,
        max_timestep_count=2,
        context_parallel_size=1,
    )
    weights = _weights(profile)
    assert build_adaln_precompute_engine(weights, profile)
    assert build_dit_engine(weights, profile)


@pytest.mark.gpu
def test_native_linear_broadcasts_over_vae_batch() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    value = network.add_input("value", trt.float32, (2, 3, 4))
    output = op.linear(
        network,
        value,
        np.ones((5, 4), np.float32),
        np.zeros((5,), np.float32),
        compute_dtype=trt.float16,
    )
    output.name = "output"
    network.mark_output(output)
    assert tuple(output.shape) == (2, 3, 5)
    assert builder.build_serialized_network(network, config)


def test_native_network_contract_counts_iattention_and_fails_closed() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    q = network.add_input("q", trt.float16, (1, 2, 4, 8))
    k = network.add_input("k", trt.float16, (1, 2, 4, 8))
    v = network.add_input("v", trt.float16, (1, 2, 4, 8))
    attention = network.add_attention(q, k, v, trt.AttentionNormalizationOp.SOFTMAX, False)
    assert attention is not None
    contract = op.validate_native_network(network, expected_attentions=1, label="test network")
    assert contract == {
        "attention_input": 1,
        "attention_output": 1,
        "plugin": 0,
        "plugin_v2": 0,
        "plugin_v3": 0,
        "dist_collective": 0,
    }
    with pytest.raises(RuntimeError, match="native layer contract failed"):
        op.validate_native_network(network, expected_attentions=2, label="test network")
