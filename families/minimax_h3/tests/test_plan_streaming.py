# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-buffer contracts and tiny native round trips; no trained/RAM qualification."""

import io

import pytest

from families.minimax_h3.graph_ops import emit_serialized_plan


class NativeBuffer(bytearray):
    def __bytes__(self):
        raise AssertionError("The streamed path copied the complete plan into Python bytes")


def test_streaming_retains_all_bytes_without_converting_the_native_buffer():
    payload = bytes(range(256)) * 8193
    source = NativeBuffer(payload)
    output = io.BytesIO()
    assert emit_serialized_plan(source, output) is None
    assert output.getvalue() == payload
    assert source == payload
    assert emit_serialized_plan(payload) == payload


def test_streaming_handles_short_writes_and_keeps_the_original_owner_alive():
    class ShortWriter(io.BytesIO):
        def write(self, value):
            return super().write(value[:7])

    output = ShortWriter()
    assert emit_serialized_plan(NativeBuffer(b"exact serialized plan content"), output) is None
    assert output.getvalue() == b"exact serialized plan content"


@pytest.mark.parametrize("written", [None, 0, -1, True, "4", 100])
def test_a_stalled_or_invalid_writer_cannot_publish_a_complete_plan(written):
    class InvalidWriter:
        def write(self, value):
            return written

    with pytest.raises(OSError):
        emit_serialized_plan(NativeBuffer(b"plan"), InvalidWriter())


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("input_timesteps", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
def test_native_adaln_serialization_round_trip(tmp_path, input_timesteps, streamed):
    import tensorrt as trt

    from families.minimax_h3.adaln_builder import build_adaln_precompute_engine
    from families.minimax_h3.config import MiniMaxH3Config
    from .test_trt_builders import _weights

    profile = MiniMaxH3Config(
        hidden_size=8,
        num_layers=2,
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
        min_text_rows=1,
        opt_text_rows=2,
        text_rows=2,
        padded_sequence_length=8,
        max_timestep_count=2,
        context_parallel_size=1,
    )
    path = tmp_path / "adaln.plan"
    options = dict(input_timesteps=input_timesteps, workspace_bytes=1 << 30)
    weights = _weights(profile)
    if streamed:
        with path.open("xb") as output:
            assert build_adaln_precompute_engine(weights, profile, output=output, **options) is None
    else:
        plan = build_adaln_precompute_engine(weights, profile, **options)
        assert isinstance(plan, bytes)
        path.write_bytes(plan)
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    assert engine is not None and engine.num_io_tensors == 4
    name = "timesteps" if input_timesteps else "timestep_features"
    assert engine.get_tensor_shape(name) == ((2,) if input_timesteps else (2, 4))
    assert engine.get_tensor_dtype(name) == trt.float32
    for name in ("block_modulation_0", "block_modulation_1"):
        assert engine.get_tensor_shape(name) == (profile.adaln_table_rows, 6, 8)
        assert engine.get_tensor_dtype(name) == trt.bfloat16
    assert engine.get_tensor_shape("final_modulation") == (2, 2, 8)
    assert engine.get_tensor_dtype("final_modulation") == trt.bfloat16
