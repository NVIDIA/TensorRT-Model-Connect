# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from families.minimax_h3.ref2va_checkpoint import (
    REF2VA_ADALN_KEYS,
    REF2VA_DENOISER_KEYS,
    REF2VA_FINISH_KEYS,
    REF2VA_HEAD_KEYS,
    REF2VA_TAIL_KEYS,
)
from families.minimax_h3.ref2va_contract import (
    Ref2VADenoiserProfile,
    ref2va_denoiser_abi,
    ref2va_denoiser_profiles,
    ref2va_first_block_cache_abis,
)
from families.minimax_h3.quantized_checkpoint import ConvRotInt8Weight


@pytest.fixture
def builder_module():
    from families.minimax_h3 import trt_compat

    if not trt_compat.is_available("tensorrt"):
        if not trt_compat.is_available("tensorrt_rtx"):
            pytest.skip("TensorRT or TensorRT-RTX bindings are unavailable")
        trt_compat.configure_backend(rtx=True)
    from families.minimax_h3 import ref2va_dit_builder

    return ref2va_dit_builder


def test_cache_checkpoint_partitions_are_disjoint_and_exhaustive() -> None:
    head, tail, finish = map(set, (REF2VA_HEAD_KEYS, REF2VA_TAIL_KEYS, REF2VA_FINISH_KEYS))
    assert (len(head), len(tail), len(finish)) == (37, 490, 5)
    assert not head & tail and not head & finish and not tail & finish
    assert head | tail | finish == set(REF2VA_DENOISER_KEYS)
    assert not (head | tail | finish) & set(REF2VA_ADALN_KEYS)
    assert all(name.startswith("transformer_blocks.") for name in tail)
    assert not any(name.startswith("transformer_blocks.0.") for name in tail)
    assert "transformer_blocks.0.attn.to_q.weight" in head
    assert "audio_proj_out.weight" in finish


@pytest.mark.parametrize("index", (0, 1, 49))
def test_ref2va_transformer_routes_packed_int8_weights_without_dequantizing(
    builder_module, monkeypatch, index
) -> None:
    # Exercise the actual shared REF block, attention, fused-QKV, SwiGLU and
    # linear dispatch. Stub only tensor operations so no GPU build is needed.
    dense, op = builder_module.dense, builder_module.op
    tensor = SimpleNamespace(shape=(2, 12))
    network = SimpleNamespace(
        add_elementwise=lambda *_args: SimpleNamespace(get_output=lambda _index: tensor)
    )

    def passthrough(*_args, **_kwargs):
        return tensor

    for name in (
        "gather_rows",
        "rms_norm",
        "modulate",
        "gated_residual",
        "partial_rope",
        "native_attention",
        "dynamic_slice",
        "silu",
    ):
        monkeypatch.setattr(op, name, passthrough)
    monkeypatch.setattr(dense, "_slice_modulation", passthrough)
    monkeypatch.setattr(dense, "_per_head_norm", passthrough)

    def quantized(rows, *, fused=False):
        return ConvRotInt8Weight(
            qweight=np.zeros((rows, 4), dtype=np.int8),
            scale=np.ones((rows, 1), dtype=np.float32),
            group_size=4,
            is_full_fused_qkv=fused,
        )

    parent = quantized(12, fused=True)
    output, fc1, fc2 = (quantized(rows) for rows in (4, 8, 4))
    prefix = f"transformer_blocks.{index}"
    weights = {
        f"{prefix}.{name}": np.ones(4, dtype=np.float32)
        for name in ("norm1.weight", "norm2.weight", "attn.norm_q.weight", "attn.norm_k.weight")
    }
    for offset, name in enumerate(("q", "k", "v")):
        start, end = offset * 4, (offset + 1) * 4
        weights[f"{prefix}.attn.to_{name}.weight"] = ConvRotInt8Weight(
            qweight=parent.qweight[start:end],
            scale=parent.scale[start:end],
            group_size=4,
            packed_parent=parent,
            row_slice=(start, end),
        )
    weights[f"{prefix}.attn.to_out.0.weight"] = output
    weights[f"{prefix}.ff.net.0.proj.weight"] = fc1
    weights[f"{prefix}.ff.net.2.weight"] = fc2
    observed = []

    def native_int8(_network, _tensor, weight, bias=None):
        assert isinstance(weight, ConvRotInt8Weight)
        assert weight.qweight.dtype == np.int8
        observed.append(weight)
        return tensor

    monkeypatch.setattr(op, "_convrot_int8_linear", native_int8)
    dense._transformer_block(
        network,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        weights,
        builder_module.native_profile(),
        index,
        consume_weights=True,
    )
    assert len(observed) == 4
    assert all(
        actual is expected
        for actual, expected in zip(observed, (parent, output, fc1, fc2), strict=True)
    )
    assert not any(f"{prefix}.attn.to_{name}.weight" in weights for name in ("q", "k", "v"))


def test_ref2va_adaln_preserves_int8_weights_and_four_timestep_rows(
    builder_module, monkeypatch
) -> None:
    quantized = ConvRotInt8Weight(
        qweight=np.zeros((4, 4), dtype=np.int8),
        scale=np.ones((4, 1), dtype=np.float32),
        group_size=4,
    )
    weights = dict.fromkeys(REF2VA_ADALN_KEYS, np.zeros(4, dtype=np.float32))
    for index in range(50):
        weights[f"transformer_blocks.{index}.adaln_proj.linear.weight"] = quantized

    def build(values, profile, **kwargs):
        assert values is weights
        assert profile.max_timestep_count == 4
        assert profile.adaln_table_rows == 12
        assert sum(isinstance(value, ConvRotInt8Weight) for value in values.values()) == 50
        assert kwargs == {"consume_weights": True}
        return b"test-plan"

    monkeypatch.setattr(builder_module, "build_adaln_precompute_engine", build)
    assert (
        builder_module.build_ref2va_adaln_precompute_engine(weights, consume_weights=True)
        == b"test-plan"
    )


def test_cache_abi_preserves_scatter_indices_timesteps_and_public_capacity() -> None:
    for profile in ref2va_denoiser_profiles():
        abis = ref2va_first_block_cache_abis(profile)
        head, tail, finish = (abis[f"ref2va_dit_{part}"] for part in ("head", "tail", "finish"))
        assert (head.filename, tail.filename, finish.filename) == (
            "ref2va_dit_head.plan",
            "ref2va_dit_tail.plan",
            "ref2va_dit_finish.plan",
        )
        head_inputs = {value.name: value for value in head.inputs}
        assert len(head.inputs) == 12
        assert len(tail.inputs) == 52
        assert len(finish.inputs) == 6
        assert "timestep_indices" not in head_inputs
        assert head_inputs["block_modulation_0"].min_shape == (12, 6, 5376)
        assert finish.inputs[-1].min_shape == (4, 2, 5376)
        assert head_inputs["previous_head_residual"].max_shape == (profile.max_packed_rows, 5376)
        assert head_inputs["cache_video_indices"].min_shape == (1,)
        assert head_inputs["cache_video_indices"].opt_shape == (profile.opt_video_rows,)
        assert head_inputs["cache_audio_indices"].max_shape == (profile.max_audio_rows,)
        assert tuple(value.name for value in head.outputs) == (
            "head_hidden",
            "head_residual",
            "cache_metric",
        )
        assert finish.outputs == ref2va_denoiser_abi(profile).outputs
        assert tuple(value.name for value in finish.inputs[2:5]) == (
            "timestep_indices",
            "video_indices",
            "audio_indices",
        )


@pytest.mark.parametrize("part", ("head", "tail", "finish"))
def test_cache_builders_reject_incomplete_or_cross_partition_weights(builder_module, part) -> None:
    build = getattr(builder_module, f"build_ref2va_dit_{part}_engine")
    with pytest.raises(ValueError, match=f"cache {part} checkpoint partition mismatch"):
        build({})
    keys = getattr(builder_module, f"{part}_checkpoint_keys")()
    with pytest.raises(ValueError, match="unexpected=.*wrong_partition"):
        build(dict.fromkeys((*keys, "wrong_partition")))


@pytest.mark.parametrize("component", ("ref2va_dit_head", "ref2va_dit_tail", "ref2va_dit_finish"))
def test_split_profiles_match_each_plan_and_keep_public_fallback(builder_module, component) -> None:
    class Optimization:
        def __init__(self):
            self.extra_memory_target = 1.0
            self.shapes = {}

        def set_shape(self, name, *, min, opt, max):
            self.shapes[name] = (min, opt, max)

        def get_shape(self, name):
            return self.shapes[name]

    profiles = []

    def add_profile(profile):
        profiles.append(profile)
        return len(profiles) - 1

    builder_module._add_cache_optimization_profiles(
        SimpleNamespace(create_optimization_profile=Optimization),
        SimpleNamespace(add_optimization_profile=add_profile),
        Ref2VADenoiserProfile(),
        component,
    )
    assert len(profiles) == 2
    assert [profile.extra_memory_target for profile in profiles] == [1.0, 0.0]
    for recorded, capacity in zip(profiles, ref2va_denoiser_profiles(), strict=True):
        expected = {
            value.name: (value.min_shape, value.opt_shape, value.max_shape)
            for value in ref2va_first_block_cache_abis(capacity)[component].inputs
            if not value.name.startswith("block_modulation_") and value.name != "final_modulation"
        }
        assert recorded.shapes == expected


def test_target_metric_excludes_reference_rows_and_does_not_mask_audio(
    builder_module, monkeypatch
) -> None:
    trt = builder_module.trt

    class Layer:
        def __init__(self, value):
            self.value = value

        def get_output(self, _index):
            return (
                self.value.reshape(self.reshape_dims)
                if hasattr(self, "reshape_dims")
                else self.value
            )

    class Network:
        @staticmethod
        def add_elementwise(left, right, operation):
            functions = {
                trt.ElementWiseOperation.SUB: np.subtract,
                trt.ElementWiseOperation.DIV: np.divide,
                trt.ElementWiseOperation.MAX: np.maximum,
                trt.ElementWiseOperation.OR: np.logical_or,
            }
            return Layer(functions[operation](left, right))

        @staticmethod
        def add_unary(value, operation):
            functions = {trt.UnaryOperation.ABS: np.abs, trt.UnaryOperation.ISNAN: np.isnan}
            return Layer(functions[operation](value))

        @staticmethod
        def add_select(condition, when_true, when_false):
            return Layer(np.where(condition, when_true, when_false))

        @staticmethod
        def add_shuffle(value):
            return Layer(value)

        @staticmethod
        def add_reduce(value, operation, axes, keep_dimensions):
            assert operation == trt.ReduceOperation.SUM and axes == 3
            return Layer(np.sum(value, axis=(0, 1), keepdims=keep_dimensions))

    monkeypatch.setattr(
        builder_module.op, "cast", lambda _network, value, _dtype: value.astype(np.float32)
    )
    monkeypatch.setattr(
        builder_module.op, "gather_rows", lambda _network, value, indices: value[indices]
    )
    monkeypatch.setattr(builder_module.op, "constant", lambda _network, value: value)
    previous = np.ones((1002, 4), dtype=np.float32)
    current = previous.copy()
    # A long fixed reference is not allowed to dilute the target changes.
    current[1000] += 0.01
    current[1001] += 0.5
    video = builder_module._target_relative_change(Network(), current, previous, np.array([1000]))
    audio = builder_module._target_relative_change(Network(), current, previous, np.array([1001]))
    assert float(video.item()) == pytest.approx(0.01)
    assert float(audio.item()) == pytest.approx(0.5)
    assert float(builder_module._cache_metric(Network(), video, audio).item()) == pytest.approx(0.5)
    invalid = np.full((1, 1), np.nan, dtype=np.float32)
    assert np.isinf(builder_module._cache_metric(Network(), invalid, audio)).all()
    assert np.isinf(builder_module._cache_metric(Network(), video, invalid)).all()
    current[:1000] = 100
    assert (
        builder_module._target_relative_change(Network(), current, previous, np.array([1001]))
        == audio
    )
    first = builder_module._target_relative_change(
        Network(), current, np.zeros_like(previous), np.array([1001])
    )
    assert np.isfinite(first).all() and float(first.item()) > 1.0


def test_native_target_metric_graph_serializes(builder_module) -> None:
    trt = builder_module.trt
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 20)
    current = network.add_input("current", trt.bfloat16, (8, 16))
    previous = network.add_input("previous", trt.bfloat16, (8, 16))
    indices = network.add_input("indices", trt.int32, (2,))
    audio_indices = network.add_input("audio_indices", trt.int32, (1,))
    video_change = builder_module._target_relative_change(network, current, previous, indices)
    audio_change = builder_module._target_relative_change(network, current, previous, audio_indices)
    metric = builder_module._cache_metric(network, video_change, audio_change)
    metric.name = "metric"
    network.mark_output(metric)
    try:
        assert metric.dtype == trt.float32
        assert tuple(metric.shape) == (1,)
        plan = builder.build_serialized_network(network, config)
        assert plan is not None and len(bytes(plan)) > 0
    finally:
        builder_module.op.release_weight_buffers(network)


def test_split_finish_serializes_and_keeps_dynamic_scatter_output_rows(builder_module) -> None:
    profile = Ref2VADenoiserProfile(
        min_video_rows=1,
        opt_video_rows=2,
        max_video_rows=4,
        min_audio_rows=2,
        opt_audio_rows=4,
        max_audio_rows=6,
        min_text_rows=1,
        opt_text_rows=1,
        max_text_rows=2,
    )
    weights = {
        "norm_out.norm.weight": np.ones(5376, dtype=np.float32),
        "proj_out.weight": np.ones((96, 5376), dtype=np.float32),
        "proj_out.bias": np.zeros(96, dtype=np.float32),
        "audio_proj_out.weight": np.ones((32, 5376), dtype=np.float32),
        "audio_proj_out.bias": np.zeros(32, dtype=np.float32),
    }
    plan = builder_module.build_ref2va_dit_finish_engine(weights, profile, workspace_bytes=64 << 20)
    trt = builder_module.trt
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    expected = ref2va_first_block_cache_abis(profile)["ref2va_dit_finish"]
    for binding in expected.inputs:
        if binding.name == "final_modulation":
            continue
        assert tuple(
            tuple(shape) for shape in engine.get_tensor_profile_shape(binding.name, 0)
        ) == (binding.min_shape, binding.opt_shape, binding.max_shape)
    context = engine.create_execution_context()
    for video, audio, packed in ((1, 2, 4), (4, 6, 12)):
        assert context.set_input_shape("head_hidden", (packed, 5376))
        assert context.set_input_shape("tail_residual", (packed, 5376))
        assert context.set_input_shape("timestep_indices", (packed,))
        assert context.set_input_shape("video_indices", (video,))
        assert context.set_input_shape("audio_indices", (audio,))
        assert tuple(context.get_tensor_shape("video_velocity")) == (video, 96)
        assert tuple(context.get_tensor_shape("audio_velocity")) == (audio, 32)
