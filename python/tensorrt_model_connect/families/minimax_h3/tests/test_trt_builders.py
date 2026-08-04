# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest


trt = pytest.importorskip("tensorrt")

from tensorrt_model_connect.families.minimax_h3.adaln_builder import (  # noqa: E402
    build_adaln_precompute_engine,
)
from tensorrt_model_connect.families.minimax_h3.config import (  # noqa: E402
    ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    DENOISER_DEFAULT_WORKSPACE_BYTES,
    MiniMaxH3Config,
    SOL_ENGINE_1344X768_124F,
    TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES,
    resolve_workspace_bytes,
)
from tensorrt_model_connect.families.minimax_h3.dit_builder import build_dit_engine  # noqa: E402
from tensorrt_model_connect.families.minimax_h3 import graph_ops as op  # noqa: E402
from tensorrt_model_connect.families.minimax_h3.text_encoder_builder import (  # noqa: E402
    build_text_encoder_engine,
)
from tensorrt_model_connect.families.minimax_h3.vae_builder import (  # noqa: E402
    build_vae_tile_decoder_engine,
)


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


@pytest.mark.parametrize(
    ("builder", "args", "default_bytes"),
    [
        (build_text_encoder_engine, ({},), TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES),
        (
            build_adaln_precompute_engine,
            ({}, SOL_ENGINE_1344X768_124F),
            ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
        ),
        (
            build_dit_engine,
            ({}, SOL_ENGINE_1344X768_124F),
            DENOISER_DEFAULT_WORKSPACE_BYTES,
        ),
        (build_vae_tile_decoder_engine, ({},), VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES),
    ],
)
@pytest.mark.parametrize("workspace_bytes", [None, 8 << 30])
def test_builders_apply_default_or_overridden_workspace(
    monkeypatch, builder, args, default_bytes: int, workspace_bytes: int | None
) -> None:
    observed = {}

    class WorkspaceConfigured(Exception):
        pass

    def capture(_config, supplied, *, default_bytes):
        observed.update(supplied=supplied, default_bytes=default_bytes)
        raise WorkspaceConfigured

    monkeypatch.setattr(op, "configure_workspace", capture)
    kwargs = {"workspace_bytes": workspace_bytes}
    if builder is build_text_encoder_engine:
        kwargs["sequence_length"] = 1
    with pytest.raises(WorkspaceConfigured):
        builder(*args, **kwargs)
    assert observed == {"supplied": workspace_bytes, "default_bytes": default_bytes}


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8589934592"])
def test_workspace_limit_rejects_non_positive_or_non_integer_values(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        resolve_workspace_bytes(value, default_bytes=64 << 30)


def test_workspace_limit_uses_default_or_exact_override() -> None:
    assert resolve_workspace_bytes(None, default_bytes=64 << 30) == 64 << 30
    assert resolve_workspace_bytes(8 << 30, default_bytes=64 << 30) == 8 << 30

    calls = []

    class FakeConfig:
        workspace_bytes = 0

        def set_memory_pool_limit(self, pool, workspace_bytes) -> None:
            calls.append((pool, workspace_bytes))
            self.workspace_bytes = workspace_bytes

        def get_memory_pool_limit(self, pool) -> int:
            assert pool == trt.MemoryPoolType.WORKSPACE
            return self.workspace_bytes

    assert op.configure_workspace(FakeConfig(), 8 << 30, default_bytes=64 << 30) == 8 << 30
    assert calls == [(trt.MemoryPoolType.WORKSPACE, 8 << 30)]

    class RejectingConfig(FakeConfig):
        def set_memory_pool_limit(self, pool, workspace_bytes) -> None:
            del pool, workspace_bytes

    with pytest.raises(RuntimeError, match="did not apply"):
        op.configure_workspace(RejectingConfig(), 8 << 30, default_bytes=64 << 30)


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
    assert build_adaln_precompute_engine(weights, profile, workspace_bytes=1 << 30)
    assert build_dit_engine(weights, profile, workspace_bytes=1 << 30)


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
    try:
        plan = builder.build_serialized_network(network, config)
    finally:
        op.release_weight_buffers(network)
    assert plan


@pytest.mark.gpu
def test_native_linear_serializes_checkpoint_bf16_without_fp32_constant() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    value = network.add_input("value", trt.bfloat16, (1, 2))
    weight = np.array([[1.5, -2.25], [3.125, 4.5]], dtype=ml_dtypes.bfloat16)
    explicit = trt.Weights(trt.bfloat16, weight.ctypes.data, weight.size)

    output = op.linear(network, value, weight)
    output.name = "output"
    network.mark_output(output)

    constants = [
        network.get_layer(index)
        for index in range(network.num_layers)
        if network.get_layer(index).type == trt.LayerType.CONSTANT
    ]
    assert len(constants) == 1
    assert constants[0].get_output(0).dtype == trt.bfloat16
    assert explicit.dtype == trt.bfloat16
    assert explicit.nbytes == weight.nbytes == 8
    assert output.dtype == trt.bfloat16
    try:
        plan = builder.build_serialized_network(network, config)
    finally:
        op.release_weight_buffers(network)
    assert plan


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
