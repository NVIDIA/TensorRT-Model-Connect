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
from tensorrt_model_connect.families.minimax_h3.config import (  # noqa: E402
    ADALN_PRECOMPUTE_DEFAULT_WORKSPACE_BYTES,
    DENOISER_DEFAULT_WORKSPACE_BYTES,
    FL2VA_KEYFRAME_COUNTS,
    FL2VA_KEYFRAME_ROWS_1344X768,
    MiniMaxH3Config,
    REF2VA_MAX_CONDITION_AUDIO_ROWS,
    REF2VA_MAX_CONDITION_VIDEO_ROWS,
    REF2VA_MAX_IMAGE_CONDITION_VIDEO_ROWS,
    REF2VA_MAX_STANDALONE_AUDIO_ROWS,
    REF2VA_MAX_TEXT_ROWS,
    REF2VA_MAX_VIDEO_CONDITION_VIDEO_ROWS,
    REF2VA_MAX_VIDEO_SOUNDTRACK_ROWS,
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
    build_fl2va_dit_engine,
    build_ref2va_dit_engine,
    checkpoint_keys as dit_checkpoint_keys,
    finish_checkpoint_keys,
    fl2va_optimization_profile_shapes,
    head_checkpoint_keys,
    ref2va_optimization_profile_shapes,
    tail_checkpoint_keys,
)
from tensorrt_model_connect.families.minimax_h3 import graph_ops as op  # noqa: E402
from tensorrt_model_connect.families.minimax_h3.text_encoder_builder import (  # noqa: E402
    build_text_encoder_engine,
)
from tensorrt_model_connect.families.minimax_h3.vae_builder import (  # noqa: E402
    build_vae_tile_decoder_engine,
)


def _tiny_profile() -> MiniMaxH3Config:
    return MiniMaxH3Config(
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
        min_text_rows=1,
        max_text_rows=3,
        padded_sequence_length=8,
        max_timestep_count=2,
        context_parallel_size=1,
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


def test_fl2va_profiles_are_padless_dynamic_text_and_exact_keyframe_modes() -> None:
    profile = SOL_ENGINE_1344X768_124F
    shapes = fl2va_optimization_profile_shapes(profile)
    assert len(shapes) == len(FL2VA_KEYFRAME_COUNTS) == 3
    # Qwen contributes one start and end token around each 1,008-row merged
    # vision block.  The 4,096-row bound is total conditioner output rows, so
    # the existing 537-row prompt and both keyframe presentations fit together.
    assert profile.text_rows + 2 * (FL2VA_KEYFRAME_ROWS_1344X768 + 2) <= profile.max_text_rows

    for keyframe_count, shape_map in zip(FL2VA_KEYFRAME_COUNTS, shapes):
        condition_rows = keyframe_count * FL2VA_KEYFRAME_ROWS_1344X768
        video_rows = profile.video_rows + condition_rows
        packed_media_rows = condition_rows + profile.audio_rows + profile.video_rows
        assert set(shape_map) == {
            "video_hidden_states",
            "encoder_hidden_states",
            "position_ids",
            "token_tags",
            "timestep_indices",
        }
        assert shape_map["video_hidden_states"] == (
            (video_rows, profile.video_patch_dim),
            (video_rows, profile.video_patch_dim),
            (video_rows, profile.video_patch_dim),
        )
        assert shape_map["encoder_hidden_states"] == (
            (profile.min_text_rows, profile.text_dim),
            (profile.text_rows, profile.text_dim),
            (profile.max_text_rows, profile.text_dim),
        )
        expected_packed_rows = tuple(
            (packed_media_rows + text_rows,)
            for text_rows in (
                profile.min_text_rows,
                profile.text_rows,
                profile.max_text_rows,
            )
        )
        assert shape_map["token_tags"] == expected_packed_rows
        assert shape_map["timestep_indices"] == expected_packed_rows
        assert shape_map["position_ids"] == tuple(
            (row_shape[0], 3) for row_shape in expected_packed_rows
        )


def test_fl2va_profile_rejects_fake_modes_and_wrong_checkpoint_partition() -> None:
    profile = SOL_ENGINE_1344X768_124F
    for keyframe_count in (-1, 3, True):
        with pytest.raises(ValueError, match="keyframe_count must be 0, 1, or 2"):
            profile.fl2va_condition_rows(keyframe_count)
    with pytest.raises(ValueError, match="checkpoint subfolder 'transformer'"):
        build_fl2va_dit_engine({}, profile, checkpoint_subfolder="transformer_ref")


def test_ref2va_profile_covers_model_card_maxima_without_padding() -> None:
    profile = SOL_ENGINE_1344X768_124F
    shapes = ref2va_optimization_profile_shapes(profile)
    assert profile.ref2va_max_text_rows == REF2VA_MAX_TEXT_ROWS == 262144
    assert profile.ref2va_max_condition_video_rows == REF2VA_MAX_CONDITION_VIDEO_ROWS == 258120
    assert REF2VA_MAX_CONDITION_VIDEO_ROWS == (
        REF2VA_MAX_IMAGE_CONDITION_VIDEO_ROWS + REF2VA_MAX_VIDEO_CONDITION_VIDEO_ROWS
    )
    assert profile.ref2va_max_condition_audio_rows == REF2VA_MAX_CONDITION_AUDIO_ROWS == 2408
    assert REF2VA_MAX_CONDITION_AUDIO_ROWS == (
        REF2VA_MAX_STANDALONE_AUDIO_ROWS + REF2VA_MAX_VIDEO_SOUNDTRACK_ROWS
    )
    assert REF2VA_MAX_STANDALONE_AUDIO_ROWS == REF2VA_MAX_VIDEO_SOUNDTRACK_ROWS == 1204
    assert shapes == {
        "video_hidden_states": ((41392, 96), (41392, 96), (295416, 96)),
        "audio_hidden_states": ((414, 32), (414, 32), (2822, 32)),
        "encoder_hidden_states": ((1, 5120), (8192, 5120), (262144, 5120)),
        "video_indices": ((41392,), (41392,), (295416,)),
        "audio_indices": ((414,), (414,), (2822,)),
        "position_ids": ((41807, 3), (49998, 3), (560382, 3)),
        "token_tags": ((41807,), (49998,), (560382,)),
        "timestep_indices": ((41807,), (49998,), (560382,)),
    }
    # The maximum is intentionally visible to native attention, not represented
    # by a smaller live sequence plus capacity padding.
    assert shapes["position_ids"][2][0] == (
        REF2VA_MAX_TEXT_ROWS
        + REF2VA_MAX_CONDITION_VIDEO_ROWS
        + REF2VA_MAX_CONDITION_AUDIO_ROWS
        + profile.audio_rows
        + profile.video_rows
    )
    assert shapes["position_ids"][2][0] ** 2 == 314027985924


def test_ref2va_profile_fails_closed_on_narrowing_or_wrong_weights() -> None:
    profile = SOL_ENGINE_1344X768_124F
    narrowed_profiles = (
        replace(profile, ref2va_max_text_rows=REF2VA_MAX_TEXT_ROWS - 1),
        replace(
            profile,
            ref2va_max_condition_video_rows=REF2VA_MAX_CONDITION_VIDEO_ROWS - 1,
        ),
        replace(
            profile,
            ref2va_max_condition_audio_rows=REF2VA_MAX_CONDITION_AUDIO_ROWS - 1,
        ),
    )
    for narrowed in narrowed_profiles:
        with pytest.raises(ValueError, match="may not silently narrow"):
            narrowed.validate()
    with pytest.raises(ValueError, match="checkpoint subfolder 'transformer_ref'"):
        build_ref2va_dit_engine({}, profile, checkpoint_subfolder="transformer")


@pytest.mark.gpu
def test_tiny_native_h3_graphs_serialize() -> None:
    profile = _tiny_profile()
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
def test_tiny_dynamic_fl2va_graph_serializes_all_exact_profiles() -> None:
    profile = _tiny_profile()
    plan = build_fl2va_dit_engine(
        _weights(profile),
        profile,
        workspace_bytes=1 << 30,
    )
    assert plan

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    assert engine.num_optimization_profiles == 3
    input_names = {
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
    }
    assert input_names == {
        "video_hidden_states",
        "audio_hidden_states",
        "encoder_hidden_states",
        "position_ids",
        "token_tags",
        "timestep_indices",
        "block_modulation_0",
        "final_modulation",
    }
    assert "adaln_indices" not in input_names
    assert "attention_mask" not in input_names
    assert {
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
    } == {"video_velocity", "audio_velocity"}

    for keyframe_count, shape_map in zip(
        FL2VA_KEYFRAME_COUNTS,
        fl2va_optimization_profile_shapes(profile),
    ):
        for name, expected_shapes in shape_map.items():
            assert (
                tuple(
                    tuple(shape) for shape in engine.get_tensor_profile_shape(name, keyframe_count)
                )
                == expected_shapes
            )
        context = engine.create_execution_context()
        assert context.set_optimization_profile_async(keyframe_count, 0)
        for name, (_, optimum, _) in shape_map.items():
            assert context.set_input_shape(name, optimum)
        assert context.infer_shapes() == []
        assert tuple(context.get_tensor_shape("video_velocity")) == (
            profile.video_rows,
            profile.video_patch_dim,
        )
        assert tuple(context.get_tensor_shape("audio_velocity")) == (
            profile.audio_rows,
            profile.audio_in_channels,
        )


@pytest.mark.gpu
def test_tiny_dynamic_ref2va_graph_serializes_request_order_scatter() -> None:
    # Tiny channel/layer dimensions keep this synthetic build tractable while
    # retaining the full model-card dynamic row maxima in the TRT profile.
    profile = _tiny_profile()
    plan = build_ref2va_dit_engine(
        _weights(profile),
        profile,
        workspace_bytes=1 << 30,
    )
    assert plan

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    assert engine.num_optimization_profiles == 1
    input_names = {
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
    }
    assert input_names == {
        "video_hidden_states",
        "audio_hidden_states",
        "encoder_hidden_states",
        "video_indices",
        "audio_indices",
        "position_ids",
        "token_tags",
        "timestep_indices",
        "block_modulation_0",
        "final_modulation",
    }
    assert "adaln_indices" not in input_names
    assert "attention_mask" not in input_names
    shape_map = ref2va_optimization_profile_shapes(profile)
    assert shape_map["position_ids"][2] == (522678, 3)
    for name, expected_shapes in shape_map.items():
        assert (
            tuple(tuple(shape) for shape in engine.get_tensor_profile_shape(name, 0))
            == expected_shapes
        )

    context = engine.create_execution_context()
    for name, (_, optimum, _) in shape_map.items():
        assert context.set_input_shape(name, optimum)
    assert context.infer_shapes() == []
    assert tuple(context.get_tensor_shape("video_velocity")) == (
        profile.video_rows,
        profile.video_patch_dim,
    )
    assert tuple(context.get_tensor_shape("audio_velocity")) == (
        profile.audio_rows,
        profile.audio_in_channels,
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
def test_rms_norm_multiplies_gamma_in_fp32_before_bf16_publication() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    value = network.add_input("value", trt.bfloat16, (2, 4))
    weight = np.asarray([0.75, 1.0, 1.25, 1.5], dtype=ml_dtypes.bfloat16)

    output = op.rms_norm(network, value, weight, width=4, eps=1.0e-6)
    product = network.get_layer(network.num_layers - 2)
    publication = network.get_layer(network.num_layers - 1)

    assert product.type == trt.LayerType.ELEMENTWISE
    assert product.get_input(0).dtype == trt.float32
    assert product.get_input(1).dtype == trt.float32
    assert publication.type == trt.LayerType.CAST
    assert output is publication.get_output(0)
    assert output.dtype == trt.bfloat16
    op.release_weight_buffers(network)


@pytest.mark.gpu
def test_qwen_rms_norm_rounds_normalized_hidden_before_bf16_gamma() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    value = network.add_input("value", trt.bfloat16, (2, 4))
    weight = np.asarray([0.75, 1.0, 1.25, 1.5], dtype=ml_dtypes.bfloat16)

    output = op.qwen_rms_norm(network, value, weight, width=4, eps=1.0e-6)
    product = network.get_layer(network.num_layers - 1)

    assert product.type == trt.LayerType.ELEMENTWISE
    assert product.get_input(0).dtype == trt.bfloat16
    assert product.get_input(1).dtype == trt.bfloat16
    assert output is product.get_output(0)
    assert output.dtype == trt.bfloat16
    op.release_weight_buffers(network)


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
def test_native_attention_preserves_checkpoint_bf16_at_iattention_boundary() -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    rows, heads, head_dim = 64, 2, 64
    width = heads * head_dim
    query = network.add_input("query", trt.bfloat16, (rows, width))
    key = network.add_input("key", trt.bfloat16, (rows, width))
    value = network.add_input("value", trt.bfloat16, (rows, width))

    output = op.native_attention(
        network,
        query,
        key,
        value,
        rows=rows,
        heads=heads,
        head_dim=head_dim,
        name="test.native_bf16_attention",
    )
    output.name = "output"
    network.mark_output(output)

    attention_inputs = [
        network.get_layer(index)
        for index in range(network.num_layers)
        if network.get_layer(index).type == trt.LayerType.ATTENTION_INPUT
    ]
    attention_outputs = [
        network.get_layer(index)
        for index in range(network.num_layers)
        if network.get_layer(index).type == trt.LayerType.ATTENTION_OUTPUT
    ]
    assert len(attention_inputs) == len(attention_outputs) == 1
    assert [attention_inputs[0].get_input(index).dtype for index in range(3)] == [
        trt.bfloat16,
        trt.bfloat16,
        trt.bfloat16,
    ]
    assert attention_outputs[0].get_input(0).dtype == trt.bfloat16
    assert attention_outputs[0].get_output(0).dtype == trt.bfloat16
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
