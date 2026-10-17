# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference conditioning ABI and exact interleaved MRoPE coordinate mapping."""

import numpy as np
import pytest

from families.minimax_h3 import text_encoder_builder as text
from families.minimax_h3.tests.reference_language_case import (
    ATOL,
    RTOL,
    ROW_COUNTS,
    make_reference_language_case,
    reference_language_expected,
    reference_language_inputs,
)


def test_reference_conditioner_shapes_do_not_use_text_only_position_layout():
    shapes = text.conditioner_input_shapes(8192, reference=True)
    assert shapes["position_ids"] == [(3, 1), (3, 128), (3, 8192)]
    assert shapes["input_ids"] == [(1,), (128,), (8192,)]
    assert shapes["visual_mask"] == [(1, 1), (128, 1), (8192, 1)]
    assert len(shapes) == 7
    for name in (
        "visual_features",
        "deepstack_features_0",
        "deepstack_features_1",
        "deepstack_features_2",
    ):
        assert shapes[name] == [(1, 5120), (128, 5120), (8192, 5120)]


def test_text_only_conditioner_retains_its_two_inputs():
    assert text.conditioner_input_shapes(37) == {
        "input_ids": [(1,), (37,), (37,)],
        "position_ids": [(1,), (37,), (37,)],
    }


@pytest.mark.parametrize(
    "length,reference", [(0, False), (-1, True), (True, True), (8.0, False), (8, "yes")]
)
def test_conditioner_rejects_invalid_bounds_before_cuda(length, reference):
    with pytest.raises(ValueError):
        text.build_text_encoder_engine({}, sequence_length=length, reference=reference)


def test_qwen_interleaved_mrope_is_not_chunked_h3_dit_rope():
    axes = text.mrope_axes()
    assert axes.dtype == np.int32 and axes.shape == (64,)
    np.testing.assert_array_equal(np.bincount(axes), [24, 20, 20])
    np.testing.assert_array_equal(axes[:60], np.tile([0, 1, 2], 20))
    np.testing.assert_array_equal(axes[60:], [0, 0, 0, 0])
    # Independent transcription of the upstream overwrite operations: each
    # coordinate deliberately differs, so a one-axis cache cannot pass.
    positions = np.arange(3 * 7, dtype=np.float32).reshape(3, 1, 7)
    inverse = np.arange(1, 65, dtype=np.float32)
    frequencies = positions[..., None] * inverse
    expected = frequencies[0].copy()
    expected[..., 1:60:3] = frequencies[1, ..., 1:60:3]
    expected[..., 2:60:3] = frequencies[2, ..., 2:60:3]
    actual = positions[axes, 0].T * inverse
    np.testing.assert_array_equal(actual, expected[0])


def test_qwen_bfloat16_products_cannot_be_collapsed_before_the_sum():
    import torch

    first = torch.tensor(-5.4375, dtype=torch.bfloat16)
    rotated = torch.tensor(4.75, dtype=torch.bfloat16)
    cos = torch.tensor(0.984375, dtype=torch.bfloat16)
    sin = torch.tensor(0.1689453125, dtype=torch.bfloat16)
    separate = first * cos + rotated * sin
    collapsed = (first.float() * cos.float() + rotated.float() * sin.float()).bfloat16()
    assert separate.item() == -4.53125
    assert collapsed.item() == -4.5625


@pytest.mark.gpu
@pytest.mark.trt
@pytest.mark.parametrize("heads", [1, 2])
def test_qwen_rope_preserves_bfloat16_product_rounds(monkeypatch, heads):
    import tensorrt as trt
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

    monkeypatch.setattr(text, "HEAD_DIM", 8)
    rows = 13
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    values = network.add_input("values", trt.bfloat16, (rows, heads * 8))
    cos = network.add_input("cos", trt.bfloat16, (1, rows, 4))
    sin = network.add_input("sin", trt.bfloat16, (1, rows, 4))
    output = text._qwen_rope(network, values, cos, sin, rows=rows, heads=heads)
    output.name = "output"
    network.mark_output(output)
    try:
        plan = builder.build_serialized_network(network, config)
    finally:
        text.op.release_weight_buffers(network)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    context = engine.create_execution_context()
    assert context is not None
    q = (torch.arange(rows * heads * 8, device="cuda").float() * 0.173 - 7).bfloat16()
    q = q.reshape(rows, heads * 8)
    coordinates = torch.arange(rows, device="cuda").float().reshape(1, rows, 1)
    frequencies = torch.tensor([0.5, 0.17, 0.03, 0.001], device="cuda")[None, None]
    cos_half = (coordinates * frequencies).cos().bfloat16()
    sin_half = (coordinates * frequencies).sin().bfloat16()
    q4 = q.reshape(1, rows, heads, 8).transpose(1, 2)
    expected = (
        apply_rotary_pos_emb(
            q4, q4, torch.cat((cos_half, cos_half), -1), torch.cat((sin_half, sin_half), -1)
        )[0]
        .transpose(1, 2)
        .reshape_as(q)
    )
    actual = torch.empty_like(q)
    for name, value in dict(values=q, cos=cos_half, sin=sin_half, output=actual).items():
        assert context.set_tensor_address(name, value.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


@pytest.mark.gpu
@pytest.mark.trt
def test_reference_language_injects_visual_features_and_deepstack_before_hidden_state_boundary(
    monkeypatch, record_property
):
    import json
    import tensorrt as trt
    import torch

    original, weights = make_reference_language_case(monkeypatch)
    plan = text.build_text_encoder_engine(
        weights, sequence_length=16, reference=True, workspace_bytes=1 << 30
    )
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None and engine.num_io_tensors == 8
    assert engine.get_tensor_dtype("visual_mask") == trt.int32
    context = engine.create_execution_context()
    assert context is not None
    original.cuda()
    stream = torch.cuda.current_stream()
    results = []
    with torch.no_grad():
        for rows in ROW_COUNTS:
            inputs = reference_language_inputs(rows)
            expected = reference_language_expected(original, inputs)
            for name, value in inputs.items():
                assert context.set_input_shape(name, tuple(value.shape))
                assert context.set_tensor_address(name, value.data_ptr())
            actual = torch.empty((rows, 16), device="cuda", dtype=torch.float32)
            assert tuple(context.get_tensor_shape("encoder_hidden_states")) == (rows, 16)
            assert context.set_tensor_address("encoder_hidden_states", actual.data_ptr())
            assert context.execute_async_v3(stream.cuda_stream)
            stream.synchronize()
            error = (actual - expected.float()).abs()
            results.append(dict(rows=rows, max_abs_error=error.max().item()))
            torch.testing.assert_close(actual, expected.float(), rtol=RTOL, atol=ATOL)
    record_property("reference_language_comparisons", json.dumps(results, allow_nan=False))
