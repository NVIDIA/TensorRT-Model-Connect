# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.sam2_hoi import native_graph_ops


def _round_to_bf16(value):
    fp32 = np.ascontiguousarray(value, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray((rounded & np.uint32(0xFFFF0000)).view(np.float32))


class _FakeTensor:
    def __init__(self, dtype, values):
        self.dtype = dtype
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.shape = self.values.shape


class _FakeLayer:
    def __init__(self, output):
        self.output = output

    def get_output(self, index):
        assert index == 0
        return self.output


class _FakeActivationNetwork:
    def __init__(self, trt):
        self.trt = trt
        self.cast_dtypes = []
        self.elementwise_dtypes = []
        self.unary_dtypes = []

    def add_cast(self, tensor, dtype):
        self.cast_dtypes.append((tensor.dtype, dtype))
        values = _round_to_bf16(tensor.values) if dtype == self.trt.bfloat16 else tensor.values
        return _FakeLayer(_FakeTensor(dtype, values))

    def add_elementwise(self, left, right, operation):
        self.elementwise_dtypes.append((left.dtype, right.dtype))
        if operation == self.trt.ElementWiseOperation.PROD:
            values = np.multiply(left.values, right.values, dtype=np.float32)
        elif operation == getattr(self.trt.ElementWiseOperation, "DIV", None):
            with np.errstate(invalid="ignore"):
                values = np.divide(left.values, right.values, dtype=np.float32)
        elif operation == getattr(self.trt.ElementWiseOperation, "SUB", None):
            values = np.subtract(left.values, right.values, dtype=np.float32)
        else:
            assert operation == self.trt.ElementWiseOperation.SUM
            values = np.add(left.values, right.values, dtype=np.float32)
        return _FakeLayer(_FakeTensor(left.dtype, values))

    def add_unary(self, tensor, operation):
        self.unary_dtypes.append(tensor.dtype)
        if operation == getattr(self.trt.UnaryOperation, "ERF", None):
            values = np.asarray(
                [np.float32(math.erf(float(value))) for value in tensor.values.flat],
                dtype=np.float32,
            ).reshape(tensor.shape)
        elif operation == getattr(self.trt.UnaryOperation, "NEG", None):
            values = np.negative(tensor.values, dtype=np.float32)
        else:
            assert operation == self.trt.UnaryOperation.EXP
            values = np.exp(tensor.values, dtype=np.float32)
        return _FakeLayer(_FakeTensor(tensor.dtype, values))


def _gelu_reference(values):
    values = np.ascontiguousarray(values, dtype=np.float32)
    scaled = np.multiply(values, np.float32(1.0 / np.sqrt(2.0)), dtype=np.float32)
    erf = np.asarray(
        [np.float32(math.erf(float(value))) for value in scaled.flat],
        dtype=np.float32,
    ).reshape(values.shape)
    one_plus = np.add(np.float32(1.0), erf, dtype=np.float32)
    half_x = np.multiply(np.float32(0.5), values, dtype=np.float32)
    return np.multiply(half_x, one_plus, dtype=np.float32)


def _silu_reference(values):
    values = np.ascontiguousarray(values, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        negative = np.negative(values, dtype=np.float32)
        exponential = np.exp(negative, dtype=np.float32)
        denominator = np.add(np.float32(1.0), exponential, dtype=np.float32)
        return np.divide(values, denominator, dtype=np.float32)


def test_native_graph_ops_are_family_owned_and_parser_free():
    path = Path(native_graph_ops.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imports.extend(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            imports.append(statement.module)

    assert not any(name.startswith("tensorrt_model_connect.families.sam3") for name in imports)
    assert "OnnxParser" not in source
    assert "torch.onnx" not in source


@pytest.mark.parametrize(("value", "expected"), [("fp32", "fp32"), ("BF16", "bf16")])
def test_precision_normalization(value, expected):
    assert native_graph_ops.normalize_precision(value) == expected


@pytest.mark.parametrize("value", ["fp16", "int8", "", "bfloat16"])
def test_precision_normalization_fails_closed(value):
    with pytest.raises(ValueError, match="support fp32 or bf16"):
        native_graph_ops.normalize_precision(value)


def test_bf16_gelu_computes_exact_erf_formula_in_fp32_and_casts_back(monkeypatch):
    fake_trt = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        ElementWiseOperation=SimpleNamespace(PROD="prod", SUM="sum"),
        UnaryOperation=SimpleNamespace(ERF="erf"),
    )
    network = _FakeActivationNetwork(fake_trt)
    constants = []

    def fake_constant(_network, shape, values, *, precision):
        constants.append(precision)
        return _FakeTensor(fake_trt.float32, np.asarray(values, dtype=np.float32).reshape(shape))

    monkeypatch.setattr(native_graph_ops, "_trt", lambda: fake_trt)
    monkeypatch.setattr(native_graph_ops, "add_constant", fake_constant)
    values = _round_to_bf16(np.asarray([[-3.0, -1.0, -0.1, 0.1, 1.0, 3.0]]))

    output = native_graph_ops.add_activation(
        network,
        _FakeTensor(fake_trt.bfloat16, values),
        "gelu",
    )

    assert output.dtype == fake_trt.bfloat16
    assert network.cast_dtypes == [
        (fake_trt.bfloat16, fake_trt.float32),
        (fake_trt.float32, fake_trt.bfloat16),
    ]
    assert constants == ["fp32", "fp32", "fp32"]
    assert network.elementwise_dtypes == [(fake_trt.float32, fake_trt.float32)] * 4
    assert network.unary_dtypes == [fake_trt.float32]
    np.testing.assert_array_equal(output.values, _round_to_bf16(_gelu_reference(values)))


def test_bf16_silu_computes_source_formula_in_fp32_and_casts_back(monkeypatch):
    fake_trt = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        ElementWiseOperation=SimpleNamespace(DIV="div", PROD="prod", SUM="sum"),
        UnaryOperation=SimpleNamespace(EXP="exp", NEG="neg"),
    )
    network = _FakeActivationNetwork(fake_trt)
    constants = []

    def fake_constant(_network, shape, values, *, precision):
        constants.append((shape, precision))
        return _FakeTensor(fake_trt.float32, np.asarray(values, dtype=np.float32).reshape(shape))

    monkeypatch.setattr(native_graph_ops, "_trt", lambda: fake_trt)
    monkeypatch.setattr(native_graph_ops, "add_constant", fake_constant)
    values = _round_to_bf16(np.asarray([-8.0, -1.0, -0.1, 0.1, 1.0, 8.0]).reshape(1, 6, 1, 1))

    output = native_graph_ops.add_activation(
        network,
        _FakeTensor(fake_trt.bfloat16, values),
        "silu",
    )

    assert output.dtype == fake_trt.bfloat16
    assert network.cast_dtypes == [
        (fake_trt.bfloat16, fake_trt.float32),
        (fake_trt.float32, fake_trt.bfloat16),
    ]
    assert constants == [((1, 1, 1, 1), "fp32")]
    assert network.unary_dtypes == [fake_trt.float32, fake_trt.float32]
    assert network.elementwise_dtypes == [
        (fake_trt.float32, fake_trt.float32),
        (fake_trt.float32, fake_trt.float32),
    ]
    np.testing.assert_array_equal(output.values, _round_to_bf16(_silu_reference(values)))


def test_fp32_silu_keeps_source_formula_and_special_value_classes(monkeypatch):
    fake_trt = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        ElementWiseOperation=SimpleNamespace(DIV="div", PROD="prod", SUM="sum"),
        UnaryOperation=SimpleNamespace(EXP="exp", NEG="neg"),
    )
    network = _FakeActivationNetwork(fake_trt)

    def fake_constant(_network, shape, values, *, precision):
        assert shape == (1, 1, 1, 1)
        assert precision == "fp32"
        return _FakeTensor(fake_trt.float32, np.asarray(values, dtype=np.float32).reshape(shape))

    monkeypatch.setattr(native_graph_ops, "_trt", lambda: fake_trt)
    monkeypatch.setattr(native_graph_ops, "add_constant", fake_constant)
    values = np.asarray([0.0, -0.0, np.inf, -np.inf, np.nan, -8.0, 8.0], dtype=np.float32).reshape(
        1, 7, 1, 1
    )

    output = native_graph_ops.add_activation(
        network,
        _FakeTensor(fake_trt.float32, values),
        "silu",
    )
    expected = _silu_reference(values)

    assert output.dtype == fake_trt.float32
    assert network.cast_dtypes == []
    assert network.unary_dtypes == [fake_trt.float32, fake_trt.float32]
    assert network.elementwise_dtypes == [
        (fake_trt.float32, fake_trt.float32),
        (fake_trt.float32, fake_trt.float32),
    ]
    np.testing.assert_array_equal(np.isnan(output.values), np.isnan(expected))
    finite = ~np.isnan(expected)
    np.testing.assert_array_equal(output.values[finite], expected[finite])
    assert not np.signbit(output.values.reshape(-1)[0])
    assert np.signbit(output.values.reshape(-1)[1])


def test_fp32_gelu_keeps_existing_fp32_formula_without_casts(monkeypatch):
    fake_trt = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        ElementWiseOperation=SimpleNamespace(PROD="prod", SUM="sum"),
        UnaryOperation=SimpleNamespace(ERF="erf"),
    )
    network = _FakeActivationNetwork(fake_trt)

    def fake_constant(_network, shape, values, *, precision):
        assert precision == "fp32"
        return _FakeTensor(fake_trt.float32, np.asarray(values, dtype=np.float32).reshape(shape))

    monkeypatch.setattr(native_graph_ops, "_trt", lambda: fake_trt)
    monkeypatch.setattr(native_graph_ops, "add_constant", fake_constant)
    values = np.asarray([[-3.0, -1.0, -0.1, 0.1, 1.0, 3.0]], dtype=np.float32)

    output = native_graph_ops.add_activation(
        network,
        _FakeTensor(fake_trt.float32, values),
        "gelu",
    )

    assert output.dtype == fake_trt.float32
    assert network.cast_dtypes == []
    np.testing.assert_array_equal(output.values, _gelu_reference(values))


def test_batch_norm_folding_matches_eval_formula():
    weight = np.asarray([[[[2.0]]], [[[3.0]]]], dtype=np.float32)
    bias = np.asarray([1.0, -2.0], dtype=np.float32)
    gamma = np.asarray([4.0, 5.0], dtype=np.float32)
    beta = np.asarray([6.0, 7.0], dtype=np.float32)
    mean = np.asarray([0.5, -0.25], dtype=np.float32)
    variance = np.asarray([3.0, 8.0], dtype=np.float32)
    folded_weight, folded_bias = native_graph_ops.fold_batch_norm(
        weight,
        bias,
        gamma,
        beta,
        mean,
        variance,
        epsilon=1.0e-3,
    )

    scale = gamma / np.sqrt(variance + np.float32(1.0e-3))
    np.testing.assert_allclose(folded_weight, weight * scale.reshape(2, 1, 1, 1))
    np.testing.assert_allclose(folded_bias, (bias - mean) * scale + beta)
    assert folded_weight.flags.c_contiguous
    assert folded_bias.flags.c_contiguous


def test_batch_norm_affine_parameters_keep_running_statistics_fp32():
    gamma = np.asarray([4.0, 5.0], dtype=np.float32)
    beta = np.asarray([6.0, 7.0], dtype=np.float32)
    mean = np.asarray([0.5, -0.25], dtype=np.float32)
    variance = np.asarray([3.0, 8.0], dtype=np.float32)
    scale, shift = native_graph_ops.batch_norm_affine_parameters(
        gamma,
        beta,
        mean,
        variance,
        epsilon=1.0e-3,
    )

    expected_scale = gamma / np.sqrt(variance + np.float32(1.0e-3))
    np.testing.assert_allclose(scale, expected_scale)
    np.testing.assert_allclose(shift, beta - mean * expected_scale)
    assert scale.dtype == np.float32
    assert shift.dtype == np.float32
    assert scale.flags.c_contiguous
    assert shift.flags.c_contiguous


def test_batch_norm_exact_affine_uses_supplied_cuda_invstd_without_host_sqrt():
    gamma = np.asarray([4.0, 5.0], dtype=np.float32)
    beta = np.asarray([6.0, 7.0], dtype=np.float32)
    mean = np.asarray([0.5, -0.25], dtype=np.float32)
    invstd = np.asarray([0.25, 0.125], dtype=np.float32)

    scale, shift = native_graph_ops.batch_norm_affine_parameters_from_invstd(
        gamma,
        beta,
        mean,
        invstd,
    )

    expected_scale = np.multiply(gamma, invstd, dtype=np.float32)
    expected_shift = np.subtract(
        beta,
        np.multiply(mean, expected_scale, dtype=np.float32),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(scale, expected_scale)
    np.testing.assert_array_equal(shift, expected_shift)
    assert scale.dtype == shift.dtype == np.float32
    assert scale.flags.c_contiguous and shift.flags.c_contiguous
    source = Path(native_graph_ops.__file__).read_text(encoding="utf-8")
    exact_body = source.split("def batch_norm_affine_parameters_from_invstd", 1)[1].split(
        "\ndef ", 1
    )[0]
    assert "sqrt" not in exact_body


def test_batch_norm_exact_affine_rejects_parameter_shape_drift():
    with pytest.raises(ValueError, match="exact affine parameter shape drift"):
        native_graph_ops.batch_norm_affine_parameters_from_invstd(
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )


def test_batch_norm_exact_graph_preserves_source_fp32_operation_order(monkeypatch):
    fake_trt = SimpleNamespace(
        bfloat16="bf16",
        float32="fp32",
        ElementWiseOperation=SimpleNamespace(PROD="prod", SUB="sub", SUM="sum"),
    )
    network = _FakeActivationNetwork(fake_trt)
    constants = []

    def fake_constant(_network, shape, values, *, precision):
        value = np.ascontiguousarray(values, dtype=np.float32).reshape(shape)
        constants.append((value.copy(), precision))
        return _FakeTensor(fake_trt.float32, value)

    monkeypatch.setattr(native_graph_ops, "_trt", lambda: fake_trt)
    monkeypatch.setattr(native_graph_ops, "add_constant", fake_constant)
    source = np.asarray([[[[-172.0]]]], dtype=np.float32)
    mean = np.asarray([-114.6495361328125], dtype=np.float32)
    invstd = np.asarray([0.17770779132843018], dtype=np.float32)
    gamma = np.asarray([-12.564002990722656], dtype=np.float32)
    beta = np.asarray([-127.84300994873047], dtype=np.float32)

    output = native_graph_ops.add_batch_norm2d_affine_from_invstd(
        network,
        _FakeTensor(fake_trt.bfloat16, source),
        gamma,
        beta,
        mean,
        invstd,
        output_dtype=fake_trt.bfloat16,
    )

    centered = np.subtract(source, mean.reshape(1, 1, 1, 1), dtype=np.float32)
    gamma_centered = np.multiply(gamma.reshape(1, 1, 1, 1), centered, dtype=np.float32)
    scaled = np.multiply(gamma_centered, invstd.reshape(1, 1, 1, 1), dtype=np.float32)
    explicit = np.add(scaled, beta.reshape(1, 1, 1, 1), dtype=np.float32)
    wrong_order = np.add(
        np.multiply(
            np.multiply(centered, invstd.reshape(1, 1, 1, 1), dtype=np.float32),
            gamma.reshape(1, 1, 1, 1),
            dtype=np.float32,
        ),
        beta.reshape(1, 1, 1, 1),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(output.values, _round_to_bf16(explicit))
    assert not np.array_equal(_round_to_bf16(explicit), _round_to_bf16(wrong_order))
    assert network.cast_dtypes == [("bf16", "fp32"), ("fp32", "bf16")]
    assert network.elementwise_dtypes == [("fp32", "fp32")] * 4
    np.testing.assert_array_equal(constants[0][0], mean.reshape(1, 1, 1, 1))
    np.testing.assert_array_equal(constants[1][0], gamma.reshape(1, 1, 1, 1))
    np.testing.assert_array_equal(constants[2][0], invstd.reshape(1, 1, 1, 1))
    np.testing.assert_array_equal(constants[3][0], beta.reshape(1, 1, 1, 1))
    assert [precision for _value, precision in constants] == ["fp32"] * 4


def test_batch_norm_exact_graph_source_is_cast_add_mul_mul_add_cast() -> None:
    source = Path(native_graph_ops.__file__).read_text(encoding="utf-8")
    body = source.split("def add_batch_norm2d_affine_from_invstd", 1)[1].split("\ndef ", 1)[0]
    assert "batch_norm_affine_parameters_from_invstd" not in body
    ordered = (
        "compute = cast(network, inp, trt.float32)",
        "centered = network.add_elementwise(",
        "gamma_centered = network.add_elementwise(",
        "scaled = network.add_elementwise(",
        "shifted = network.add_elementwise(",
        "return cast(network, shifted.get_output(0), output_dtype)",
    )
    positions = [body.index(token) for token in ordered]
    assert positions == sorted(positions)
    assert body.count("add_constant(") == 4
    assert body.count("trt.ElementWiseOperation.SUB") == 1
    assert body.count("trt.ElementWiseOperation.SUM") == 1
    assert body.count("trt.ElementWiseOperation.PROD") == 2
    assert "gamma_tensor,\n        centered.get_output(0)" in body
    assert "gamma_centered.get_output(0),\n        invstd_tensor" in body


@pytest.mark.parametrize("lookup_name", ["get_creator", "get_plugin_creator"])
def test_plugin_creator_supports_current_and_legacy_registry_apis(monkeypatch, lookup_name):
    expected = object()

    class Registry:
        pass

    registry = Registry()
    setattr(
        registry,
        lookup_name,
        lambda name, version, namespace: (
            expected if (name, version, namespace) == ("ExactOp", "1", "") else None
        ),
    )
    fake_trt = SimpleNamespace(get_plugin_registry=lambda: registry)
    monkeypatch.setattr(native_graph_ops, "_trt", lambda: fake_trt)

    assert native_graph_ops.plugin_creator("ExactOp") is expected


def test_plugin_creator_fails_when_registry_has_no_lookup(monkeypatch):
    fake_trt = SimpleNamespace(get_plugin_registry=lambda: object())
    monkeypatch.setattr(native_graph_ops, "_trt", lambda: fake_trt)
    with pytest.raises(RuntimeError, match="no creator lookup API"):
        native_graph_ops.plugin_creator("ExactOp")
