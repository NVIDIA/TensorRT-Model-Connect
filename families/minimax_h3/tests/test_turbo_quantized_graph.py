# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU graph contracts for an INT8 base with separate BF16 Turbo LoRA."""

import ast
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from families.minimax_h3.quantized_checkpoint import ConvRotInt8Weight
from families.minimax_h3.turbo_checkpoint import TurboLoraWeight, pack_turbo_qkv


def _bf16(shape):
    return np.ones(shape, dtype=ml_dtypes.bfloat16)


def _quantized(rows=8, width=4, **kwargs):
    return ConvRotInt8Weight(
        np.ones((rows, width), dtype=np.int8),
        np.ones((rows, 1), dtype=np.float32),
        4,
        **kwargs,
    )


@pytest.fixture
def graph():
    # Execute the actual graph functions with recording native-API doubles.
    # These tests check wiring/precision contracts, not TensorRT lowering or
    # the numerical accuracy of the separately tested ConvRot implementation.
    source = Path(__file__).resolve().parents[1] / "graph_ops.py"
    names = {"linear", "bf16_linear_with_fused_bias", "fused_qkv"}
    functions = [
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *functions], type_ignores=[]))
    matmuls, sums, quantized_calls, slices = [], [], [], []

    class Tensor:
        def __init__(self, shape, dtype="bf16", *, source=None, value=None):
            self.shape = shape
            self.dtype = dtype
            self.source = source
            self.value = value

    def layer(value):
        return SimpleNamespace(get_output=lambda _index: value)

    class Network:
        def add_matrix_multiply(self, left, left_op, right, right_op):
            assert left_op == "none" and right_op == "transpose"
            assert left.dtype == right.dtype
            result = Tensor((left.shape[0], right.shape[0]), left.dtype)
            matmuls.append((left, right, result))
            return layer(result)

        def add_elementwise(self, left, right, operation):
            assert operation == "sum"
            assert left.dtype == right.dtype
            result = Tensor(left.shape, left.dtype)
            sums.append((left, right, result))
            return layer(result)

    def constant(_network, array):
        dtype = "bf16" if array.dtype == np.dtype(ml_dtypes.bfloat16) else "fp32"
        return Tensor(array.shape, dtype, value=array)

    def cast(_network, tensor, dtype):
        if tensor.dtype == dtype:
            return tensor
        return Tensor(tensor.shape, dtype, source=tensor, value=tensor.value)

    def convrot(_network, tensor, weight, bias=None):
        result = Tensor((tensor.shape[0], weight.qweight.shape[0]))
        quantized_calls.append((tensor, weight, bias, result))
        return result

    def dynamic_slice(_network, tensor, starts, sizes):
        slices.append((tensor, starts, sizes))
        return Tensor((tensor.shape[0], sizes[1]), tensor.dtype)

    globals_ = {
        "np": np,
        "TurboLoraWeight": TurboLoraWeight,
        "ConvRotInt8Weight": ConvRotInt8Weight,
        "pack_turbo_qkv": pack_turbo_qkv,
        "trt": SimpleNamespace(
            bfloat16="bf16",
            float32="fp32",
            MatrixOperation=SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
            ElementWiseOperation=SimpleNamespace(SUM="sum"),
        ),
        "constant": constant,
        "weight_constant": constant,
        "cast": cast,
        "_convrot_int8_linear": convrot,
        "dynamic_slice": dynamic_slice,
    }
    exec(compile(module, str(source), "exec"), globals_)
    return SimpleNamespace(
        **globals_,
        network=Network(),
        input=Tensor((-1, 4)),
        matmuls=matmuls,
        sums=sums,
        quantized_calls=quantized_calls,
        slices=slices,
    )


@pytest.mark.parametrize("with_bias", [False, True])
def test_quantized_base_preserves_original_lora_input_and_bf16_add(graph, with_bias):
    base = _quantized()
    weight = TurboLoraWeight(base, _bf16((2, 4)), _bf16((8, 2)))
    bias = _bf16((8,)) if with_bias else None
    output = graph.linear(graph.network, graph.input, weight, bias)

    assert len(graph.quantized_calls) == 1
    original, actual_base, actual_bias, base_output = graph.quantized_calls[0]
    assert original is graph.input and actual_base is base and actual_bias is bias
    assert len(graph.matmuls) == 2
    first, second = graph.matmuls
    assert first[0] is graph.input  # Not the base branch's rotated activation.
    assert first[1].value is weight.lora_a
    assert second[0] is first[2] and second[1].value is weight.lora_b
    assert all(value.dtype == "bf16" for operation in graph.matmuls for value in operation)
    assert len(graph.sums) == 3
    assert graph.sums[0][0] is base_output
    assert graph.sums[1][0] is second[2]
    assert graph.sums[2][0] is graph.sums[0][2]
    assert graph.sums[2][1] is graph.sums[1][2]
    assert output is graph.sums[2][2] and output.dtype == "bf16"
    assert all(np.all(operation[1].value == 0) for operation in graph.sums[:2])


@pytest.mark.parametrize("with_bias", [False, True])
def test_original_bf16_turbo_path_keeps_three_linears(graph, with_bias):
    weight = TurboLoraWeight(_bf16((8, 4)), _bf16((2, 4)), _bf16((8, 2)))
    output = graph.linear(graph.network, graph.input, weight, _bf16((8,)) if with_bias else None)
    assert not graph.quantized_calls
    assert len(graph.matmuls) == 3
    assert graph.matmuls[0][0].dtype == ("fp32" if with_bias else "bf16")
    assert graph.matmuls[1][0] is graph.input
    assert graph.matmuls[2][0] is graph.matmuls[1][2]
    assert output.dtype == "bf16"


def test_plain_quantized_path_still_delegates_without_lora(graph):
    weight = _quantized()
    bias = _bf16((8,))
    output = graph.linear(graph.network, graph.input, weight, bias)
    assert graph.quantized_calls == [(graph.input, weight, bias, output)]
    assert not graph.matmuls and not graph.sums


@pytest.mark.parametrize("kwargs", [{"bf16": False}, {"compute_dtype": "fp32"}])
def test_quantized_turbo_rejects_non_bf16_lora_compute(graph, kwargs):
    weight = TurboLoraWeight(_quantized(), _bf16((2, 4)), _bf16((8, 2)))
    with pytest.raises(ValueError, match="LoRA linears require BF16 compute"):
        graph.linear(graph.network, graph.input, weight, **kwargs)
    assert not graph.quantized_calls and not graph.matmuls


def test_fused_quantized_turbo_qkv_reuses_parent_and_shared_a(graph):
    quantized_parent = _quantized(rows=12, is_full_fused_qkv=True)
    parent = TurboLoraWeight(
        quantized_parent, _bf16((2, 4)), _bf16((12, 2)), is_full_fused_qkv=True
    )
    children = []
    for start in (0, 4, 8):
        end = start + 4
        base = ConvRotInt8Weight(
            quantized_parent.qweight[start:end],
            quantized_parent.scale[start:end],
            4,
            packed_parent=quantized_parent,
            row_slice=(start, end),
        )
        children.append(
            TurboLoraWeight(
                base,
                parent.lora_a,
                parent.lora_b[start:end],
                packed_parent=parent,
                row_slice=(start, end),
            )
        )
    weights = dict(zip(("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"), children))
    result = graph.fused_qkv(graph.network, graph.input, weights, "attn", consume_weights=True)
    assert weights == {}
    assert graph.quantized_calls[0][1] is quantized_parent
    assert len(graph.quantized_calls) == 1 and len(graph.matmuls) == 2
    assert graph.matmuls[0][1].value is parent.lora_a
    assert graph.matmuls[1][1].value is parent.lora_b
    assert [start for _, start, _ in graph.slices] == [(0, 0), (0, 4), (0, 8)]
    assert all(tensor.shape == (-1, 4) for tensor in result)
