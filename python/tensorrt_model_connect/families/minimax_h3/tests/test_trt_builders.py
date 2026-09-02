# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import ml_dtypes
import numpy as np
import pytest


trt = pytest.importorskip("tensorrt")

from tensorrt_model_connect.families.minimax_h3.adaln_builder import (  # noqa: E402
    build_adaln_precompute_engine,
)
from tensorrt_model_connect.families.minimax_h3.audio_vae_builder import (  # noqa: E402
    build_audio_vae_decoder_engine,
)
from tensorrt_model_connect.families.minimax_h3.config import (  # noqa: E402
    ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
    DENOISER_DEFAULT_WORKSPACE_BYTES,
    MiniMaxH3Config,
    SOL_ENGINE_1344X768_124F,
    TEXT_ENCODER_DEFAULT_WORKSPACE_BYTES,
    VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES,
    resolve_workspace_bytes,
)
from tensorrt_model_connect.families.minimax_h3.dit_builder import (  # noqa: E402
    build_dit_engine,
    build_dit_finish_engine,
    build_dit_head_engine,
    build_dit_tail_engine,
    build_dit_vsa_entry_engine,
    build_dit_vsa_finish_engine,
    build_dit_vsa_transition_engine,
    checkpoint_keys as dit_checkpoint_keys,
    finish_checkpoint_keys,
    head_checkpoint_keys,
    tail_checkpoint_keys,
)
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
        (
            build_audio_vae_decoder_engine,
            ({},),
            AUDIO_VAE_DECODER_DEFAULT_WORKSPACE_BYTES,
        ),
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

    class FakeBuilder:
        @staticmethod
        def create_network(_flags):
            return object()

        @staticmethod
        def create_builder_config():
            return object()

    monkeypatch.setattr(trt, "Builder", lambda _logger: FakeBuilder())
    monkeypatch.setattr(
        op,
        "configure_builder",
        lambda _config, *, weight_streaming=False: None,
    )
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


def test_first_block_cache_checkpoint_partitions_are_exact() -> None:
    profile = replace(SOL_ENGINE_1344X768_124F, first_block_cache=True)
    head = set(head_checkpoint_keys(profile))
    tail = set(tail_checkpoint_keys(profile))
    finish = set(finish_checkpoint_keys(profile))
    assert head
    assert tail
    assert finish
    assert not (head & tail or head & finish or tail & finish)
    assert head | tail | finish == set(dit_checkpoint_keys(profile))
    assert "transformer_blocks.0.norm1.weight" in head
    assert "transformer_blocks.1.norm1.weight" in tail
    assert "norm_out.norm.weight" in finish


def test_split_builders_require_explicit_first_block_cache_profile() -> None:
    for builder in (build_dit_head_engine, build_dit_tail_engine, build_dit_finish_engine):
        with pytest.raises(ValueError, match="first_block_cache=True"):
            builder({}, SOL_ENGINE_1344X768_124F)
    with pytest.raises(ValueError, match="requires the split DiT builders"):
        build_dit_engine({}, replace(SOL_ENGINE_1344X768_124F, first_block_cache=True))


@pytest.mark.gpu
def test_tiny_native_h3_graphs_serialize() -> None:
    profile = MiniMaxH3Config(
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        num_heads=4,
        # TensorRT-RTX keeps H3 IAttention non-decomposable for production
        # performance. Exercise a head dimension supported by its dedicated
        # BF16 attention kernel rather than a synthetic 8-wide head.
        head_dim=128,
        ffn_dim=16,
        video_in_channels=2,
        audio_in_channels=2,
        text_dim=8,
        timestep_input_dim=4,
        timestep_hidden_size=8,
        timestep_embed_dim=4,
        rope_freq_dim=1,
        min_video_rows=4,
        opt_video_rows=4,
        video_rows=4,
        min_audio_rows=2,
        opt_audio_rows=2,
        audio_rows=2,
        min_text_rows=1,
        opt_text_rows=2,
        text_rows=2,
        padded_sequence_length=8,
        max_timestep_count=2,
        context_parallel_size=1,
    )
    weights = _weights(profile)
    assert build_adaln_precompute_engine(weights, profile, workspace_bytes=1 << 30)
    assert build_dit_engine(weights, profile, workspace_bytes=1 << 30)

    split_profile = replace(profile, num_layers=2, first_block_cache=True)
    split_weights = _weights(split_profile)
    head_plan = build_dit_head_engine(split_weights, split_profile, workspace_bytes=1 << 30)
    tail_plan = build_dit_tail_engine(split_weights, split_profile, workspace_bytes=1 << 30)
    finish_plan = build_dit_finish_engine(split_weights, split_profile, workspace_bytes=1 << 30)
    assert head_plan and tail_plan and finish_plan

    runtime_logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(runtime_logger)
    expected = (
        (head_plan, {"head_hidden", "head_residual", "cache_metric"}),
        (tail_plan, {"tail_residual"}),
        (finish_plan, {"video_velocity", "audio_velocity"}),
    )
    for plan, expected_outputs in expected:
        engine = runtime.deserialize_cuda_engine(plan)
        assert engine is not None
        outputs = {
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
        }
        assert outputs == expected_outputs


@pytest.mark.gpu
def test_dynamic_finish_plan_preserves_124_and_345_frame_shapes() -> None:
    profile = MiniMaxH3Config(
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=0,
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
        video_rows=102816,
        audio_rows=1150,
        padded_sequence_length=104503,
        max_timestep_count=2,
        context_parallel_size=1,
        first_block_cache=True,
    )
    plan = build_dit_finish_engine(_weights(profile), profile, workspace_bytes=1 << 30)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    assert tuple(
        tuple(shape)
        for shape in engine.get_tensor_profile_shape("video_hidden_states", 0)
    ) == (
        (37296, profile.video_patch_dim),
        (37296, profile.video_patch_dim),
        (102816, profile.video_patch_dim),
    )
    assert tuple(
        tuple(shape)
        for shape in engine.get_tensor_profile_shape("audio_hidden_states", 0)
    ) == (
        (414, profile.audio_in_channels),
        (414, profile.audio_in_channels),
        (1150, profile.audio_in_channels),
    )

    for video_rows, audio_rows, packed_rows in (
        (37296, 414, 37711),
        (102816, 1150, 104503),
    ):
        finish = engine.create_execution_context()
        assert finish.set_input_shape("head_hidden", (packed_rows, profile.hidden_size))
        assert finish.set_input_shape("tail_residual", (packed_rows, profile.hidden_size))
        assert finish.set_input_shape("timestep_indices", (packed_rows,))
        assert finish.set_input_shape(
            "video_hidden_states", (video_rows, profile.video_patch_dim)
        )
        assert finish.set_input_shape(
            "audio_hidden_states", (audio_rows, profile.audio_in_channels)
        )
        assert tuple(finish.get_tensor_shape("video_velocity")) == (
            video_rows,
            profile.video_patch_dim,
        )
        assert tuple(finish.get_tensor_shape("audio_velocity")) == (
            audio_rows,
            profile.audio_in_channels,
        )


@pytest.mark.gpu
def test_dynamic_attention_plans_preserve_media_and_packed_shapes() -> None:
    profile = MiniMaxH3Config(
        hidden_size=128,
        num_layers=2,
        num_refiner_layers=0,
        num_heads=2,
        head_dim=128,
        ffn_dim=256,
        video_in_channels=2,
        audio_in_channels=2,
        text_dim=128,
        timestep_input_dim=4,
        timestep_hidden_size=128,
        timestep_embed_dim=64,
        rope_freq_dim=16,
        min_video_rows=64,
        opt_video_rows=64,
        video_rows=128,
        min_audio_rows=32,
        opt_audio_rows=32,
        audio_rows=64,
        min_text_rows=32,
        opt_text_rows=32,
        text_rows=64,
        padded_sequence_length=256,
        max_timestep_count=2,
        first_block_cache=True,
    )
    monolithic_profile = replace(profile, first_block_cache=False)
    monolithic_plan = build_dit_engine(
        _weights(monolithic_profile), monolithic_profile, workspace_bytes=1 << 30
    )
    head_plan = build_dit_head_engine(_weights(profile), profile, workspace_bytes=1 << 30)
    tail_plan = build_dit_tail_engine(_weights(profile), profile, workspace_bytes=1 << 30)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    monolithic_engine = runtime.deserialize_cuda_engine(monolithic_plan)
    head_engine = runtime.deserialize_cuda_engine(head_plan)
    tail_engine = runtime.deserialize_cuda_engine(tail_plan)
    assert monolithic_engine is not None and head_engine is not None and tail_engine is not None
    assert tuple(
        tuple(shape)
        for shape in head_engine.get_tensor_profile_shape("video_hidden_states", 0)
    ) == ((64, profile.video_patch_dim), (64, profile.video_patch_dim), (128, profile.video_patch_dim))
    assert tuple(
        tuple(shape) for shape in tail_engine.get_tensor_profile_shape("head_hidden", 0)
    ) == ((128, profile.hidden_size), (128, profile.hidden_size), (256, profile.hidden_size))

    for video_rows, audio_rows, text_rows, packed_rows in (
        (64, 32, 32, 128),
        (128, 64, 64, 256),
    ):
        monolithic = monolithic_engine.create_execution_context()
        assert monolithic.set_input_shape(
            "video_hidden_states", (video_rows, profile.video_patch_dim)
        )
        assert monolithic.set_input_shape(
            "audio_hidden_states", (audio_rows, profile.audio_in_channels)
        )
        assert monolithic.set_input_shape(
            "encoder_hidden_states", (text_rows, profile.text_dim)
        )
        assert monolithic.set_input_shape("position_ids", (packed_rows, 3))
        assert monolithic.set_input_shape("adaln_indices", (packed_rows,))
        assert monolithic.set_input_shape("timestep_indices", (packed_rows,))
        assert tuple(monolithic.get_tensor_shape("video_velocity")) == (
            video_rows,
            profile.video_patch_dim,
        )
        assert tuple(monolithic.get_tensor_shape("audio_velocity")) == (
            audio_rows,
            profile.audio_in_channels,
        )

        head = head_engine.create_execution_context()
        assert head.set_input_shape("video_hidden_states", (video_rows, profile.video_patch_dim))
        assert head.set_input_shape("audio_hidden_states", (audio_rows, profile.audio_in_channels))
        assert head.set_input_shape("encoder_hidden_states", (text_rows, profile.text_dim))
        assert head.set_input_shape("position_ids", (packed_rows, 3))
        assert head.set_input_shape("adaln_indices", (packed_rows,))
        assert head.set_input_shape("previous_head_residual", (packed_rows, profile.hidden_size))
        assert tuple(head.get_tensor_shape("head_hidden")) == (packed_rows, profile.hidden_size)
        assert tuple(head.get_tensor_shape("head_residual")) == (packed_rows, profile.hidden_size)

        tail = tail_engine.create_execution_context()
        assert tail.set_input_shape("head_hidden", (packed_rows, profile.hidden_size))
        assert tail.set_input_shape("position_ids", (packed_rows, 3))
        assert tail.set_input_shape("adaln_indices", (packed_rows,))
        assert tuple(tail.get_tensor_shape("tail_residual")) == (
            packed_rows,
            profile.hidden_size,
        )


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


@pytest.mark.gpu
@pytest.mark.trt
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


@pytest.mark.gpu
def test_segmented_vsa_entry_transition_finish_serialize_head_major_abi() -> None:
    profile = MiniMaxH3Config(
        hidden_size=128,
        num_layers=2,
        num_refiner_layers=0,
        num_heads=2,
        head_dim=128,
        ffn_dim=256,
        video_in_channels=2,
        audio_in_channels=2,
        text_dim=128,
        timestep_input_dim=4,
        timestep_hidden_size=128,
        timestep_embed_dim=64,
        rope_freq_dim=16,
        min_video_rows=64,
        opt_video_rows=64,
        video_rows=128,
        min_audio_rows=32,
        opt_audio_rows=32,
        audio_rows=64,
        min_text_rows=32,
        opt_text_rows=32,
        text_rows=64,
        padded_sequence_length=256,
        max_timestep_count=2,
    )
    weights = _weights(profile)
    for index in range(profile.num_layers):
        weights[
            f"transformer_blocks.{index}.attn.to_gate_compress.weight"
        ] = np.zeros((profile.attention_size, profile.hidden_size), np.float32)

    plans = {
        "entry": build_dit_vsa_entry_engine(
            dict(weights), profile, workspace_bytes=1 << 30
        ),
        "transition": build_dit_vsa_transition_engine(
            dict(weights), profile, 0, workspace_bytes=1 << 30
        ),
        "finish": build_dit_vsa_finish_engine(
            dict(weights), profile, workspace_bytes=1 << 30
        ),
    }
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engines = {
        name: runtime.deserialize_cuda_engine(plan) for name, plan in plans.items()
    }
    assert all(engine is not None for engine in engines.values())
    entry = engines["entry"]
    transition = engines["transition"]
    finish = engines["finish"]
    assert set(entry.get_tensor_name(index) for index in range(entry.num_io_tensors)) >= {
        "next_residual_hidden",
        "vsa_query",
        "vsa_key",
        "vsa_value",
        "vsa_gate",
    }
    assert tuple(
        tuple(shape)
        for shape in transition.get_tensor_profile_shape("vsa_attention_output", 0)
    ) == ((2, 128, 128), (2, 128, 128), (2, 256, 128))

    for rows in (128, 256):
        context = transition.create_execution_context()
        assert context.set_input_shape("residual_hidden", (rows, profile.hidden_size))
        assert context.set_input_shape(
            "vsa_attention_output", (profile.num_heads, rows, profile.head_dim)
        )
        assert context.set_input_shape("position_ids", (rows, 3))
        assert context.set_input_shape("adaln_indices", (rows,))
        assert tuple(context.get_tensor_shape("next_residual_hidden")) == (
            rows,
            profile.hidden_size,
        )
        for name in ("vsa_query", "vsa_key", "vsa_value", "vsa_gate"):
            assert tuple(context.get_tensor_shape(name)) == (
                profile.num_heads,
                rows,
                profile.head_dim,
            )
    assert set(finish.get_tensor_name(index) for index in range(finish.num_io_tensors)) >= {
        "vsa_attention_output",
        "video_velocity",
        "audio_velocity",
    }


def test_native_attention_preserves_checkpoint_bfloat16_range() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    q = network.add_input("q", trt.bfloat16, (4, 16))
    k = network.add_input("k", trt.bfloat16, (4, 16))
    v = network.add_input("v", trt.bfloat16, (4, 16))

    output = op.native_attention(
        network,
        q,
        k,
        v,
        rows=4,
        heads=2,
        head_dim=8,
        name="bf16_attention",
    )

    assert output.dtype == trt.bfloat16
    assert all(
        network.get_layer(index).get_output(0).dtype != trt.float16
        for index in range(network.num_layers)
    )


def test_fused_qkv_releases_consumed_source_arrays(monkeypatch) -> None:
    class Tensor:
        shape = (2, 6)

    class Layer:
        def get_output(self, _index):
            return object()

    class Network:
        def add_slice(self, *_args):
            return Layer()

    prefix = "transformer_blocks.0.attn"
    keys = [f"{prefix}.to_{name}.weight" for name in ("q", "k", "v")]
    weights = {key: np.ones((2, 2), dtype=np.float32) for key in keys}
    monkeypatch.setattr(op, "linear", lambda *_args, **_kwargs: Tensor())

    outputs = op.fused_qkv(
        Network(), object(), weights, prefix, consume_weights=True
    )

    assert len(outputs) == 3
    assert not any(key in weights for key in keys)
