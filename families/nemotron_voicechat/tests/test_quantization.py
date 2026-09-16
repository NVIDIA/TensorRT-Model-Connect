# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the VoiceChat-owned runtime-absmax W8A8 path."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest

from families.nemotron_voicechat import quantization


@pytest.fixture(autouse=True)
def _isolated_pointer_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        quantization, "_INT8_WEIGHT_KEEPALIVE", weakref.WeakKeyDictionary()
    )
    monkeypatch.setattr(quantization, "_INT8_FINALIZED_WEIGHT_NETWORKS", weakref.WeakSet())


def test_weight_scale_is_per_output_and_chunked() -> None:
    weight = np.array(
        [[0.0, -127.0, 4.0], [254.0, 0.0, -8.0]], dtype=np.float16
    )

    scale = quantization.derive_weight_scale(
        weight,
        chunk_bytes=3 * np.dtype(np.float32).itemsize,
    )

    np.testing.assert_allclose(scale, np.array([2.0, 1.0, 8.0 / 127.0]))
    assert scale.dtype == np.float32
    assert scale.flags.c_contiguous


def test_numpy_weight_packing_rounds_saturates_and_uses_bounded_chunks() -> None:
    packed, scale = quantization.quantize_int8_per_output_channel(
        np.array(
            [[0.6, 0.5, 300.0], [-0.6, -0.5, -300.0]],
            dtype=np.float32,
        ),
        np.array([0.5, 0.25, 2.0], dtype=np.float32),
        lhs_width=2,
        rhs_width=3,
        chunk_bytes=3 * np.dtype(np.float32).itemsize,
    )

    np.testing.assert_array_equal(
        packed,
        np.array([[1, 2, 127], [-1, -2, -128]], dtype=np.int8),
    )
    np.testing.assert_array_equal(scale, np.array([0.5, 0.25, 2.0], dtype=np.float32))
    assert packed.flags.c_contiguous
    assert scale.flags.c_contiguous


def test_context_derives_scales_and_falls_back_for_unselected_weights() -> None:
    calls: list[tuple] = []

    class GraphOps:
        @staticmethod
        def add_matmul_rhs_constant(*args, **kwargs):
            calls.append((args, kwargs))
            return "fp-matmul"

    context = quantization.VoiceChatQuantContext.from_weights(
        {
            "selected": np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32),
            "unselected": np.ones((2, 2), dtype=np.float32),
        },
        ["selected"],
        GraphOps,
    )

    np.testing.assert_allclose(
        context.weight_scales["selected"],
        np.array([3.0, 4.0], dtype=np.float32) / np.float32(127.0),
    )
    result = context.maybe_quantized_matmul(
        object(),
        object(),
        2,
        2,
        np.ones((2, 2), dtype=np.float32),
        "unselected",
    )
    assert result == "fp-matmul"
    assert calls


def test_model_selects_runtime_w8a8_projections_but_not_language_head() -> None:
    from families.nemotron_voicechat import model

    assert model._normalize_quantization(None) is None
    assert model._normalize_quantization("int8") == "int8_sq"
    assert model._normalize_quantization("int8-sq") == "int8_sq"
    with pytest.raises(ValueError, match="only int8/int8_sq"):
        model._normalize_quantization("fp8")

    weights: dict[str, object] = {
        "_layer_types": ["mamba2", "mlp", "attention"],
        "layer.0.mamba_in_proj": np.array(
            [[1.0, -4.0], [2.0, 3.0]], dtype=np.float32
        ),
        "layer.0.mamba_out_proj": np.ones((2, 2), dtype=np.float32),
        "layer.1.w_up": np.ones((2, 2), dtype=np.float32),
        "layer.1.w_down": np.ones((2, 2), dtype=np.float32),
        "layer.2.w_q": np.ones((2, 2), dtype=np.float32),
        "layer.2.w_k": np.ones((2, 1), dtype=np.float32),
        "layer.2.w_v": np.ones((2, 1), dtype=np.float32),
        "layer.2.w_o": np.ones((2, 2), dtype=np.float32),
        "w_lm_head": np.ones((2, 3), dtype=np.float32),
        "w_function_head": np.ones((2, 3), dtype=np.float32),
    }
    context = model._build_thinker_quant_context(
        weights,
        graph_ops_module=object(),
    )

    assert set(context.weight_scales) == set(
        model._thinker_quantized_weight_names(weights)
    )
    assert "w_function_head" in context.weight_scales
    assert "w_lm_head" not in context.weight_scales
    np.testing.assert_allclose(
        context.weight_scales["layer.0.mamba_in_proj"],
        np.array([2.0, 4.0], dtype=np.float32) / np.float32(127.0),
    )


def test_dynamic_w8a8_graph_uses_packed_weight_and_runtime_row_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        def __init__(self, dtype: str, shape: tuple[int, ...]):
            self.dtype = dtype
            self.shape = shape

    class Layer:
        def __init__(self, output: Tensor):
            self.output = output
            self.axis = None

        def get_output(self, index: int) -> Tensor:
            assert index == 0
            return self.output

    explicit_weights: list[object] = []

    class Weights:
        def __init__(self, dtype: str, pointer: int, size: int):
            self.dtype = dtype
            self.pointer = pointer
            self.size = size
            explicit_weights.append(self)

    trt = SimpleNamespace(
        float16="fp16",
        float32="fp32",
        int8="int8",
        Weights=Weights,
        MatrixOperation=SimpleNamespace(NONE="none"),
        UnaryOperation=SimpleNamespace(ABS="abs", ROUND="round"),
        ReduceOperation=SimpleNamespace(MAX="reduce_max"),
        ElementWiseOperation=SimpleNamespace(
            MAX="max", MIN="min", DIV="div", PROD="prod"
        ),
    )
    monkeypatch.setattr(quantization, "_trt", lambda: trt)

    calls: dict[str, list] = {
        "constant": [],
        "cast": [],
        "unary": [],
        "reduce": [],
        "elementwise": [],
        "dequantize": [],
        "matmul": [],
    }

    class Network:
        def add_constant(self, shape, weights):
            calls["constant"].append((shape, weights))
            return Layer(Tensor(weights.dtype, shape))

        def add_cast(self, tensor, dtype):
            layer = Layer(Tensor(dtype, tensor.shape))
            calls["cast"].append((tensor, dtype, layer))
            return layer

        def add_unary(self, tensor, operation):
            calls["unary"].append((tensor, operation))
            return Layer(Tensor(tensor.dtype, tensor.shape))

        def add_reduce(self, tensor, operation, axes, keep_dims):
            calls["reduce"].append((tensor, operation, axes, keep_dims))
            return Layer(Tensor(tensor.dtype, (tensor.shape[0], 1)))

        def add_elementwise(self, lhs, rhs, operation):
            output = Tensor(lhs.dtype, lhs.shape)
            calls["elementwise"].append((lhs, rhs, operation, output))
            return Layer(output)

        def add_dequantize(self, tensor, scale, dtype):
            layer = Layer(Tensor(dtype, tensor.shape))
            calls["dequantize"].append((tensor, scale, dtype, layer))
            return layer

        def add_matrix_multiply(self, lhs, lhs_op, rhs, rhs_op):
            calls["matmul"].append((lhs, lhs_op, rhs, rhs_op))
            return Layer(Tensor(lhs.dtype, (lhs.shape[0], rhs.shape[-1])))

    graph_constants: list[tuple] = []

    class GraphOps:
        @staticmethod
        def add_constant(_network, shape, values, *, dtype):
            graph_constants.append((shape, np.array(values, copy=True), dtype))
            return Tensor(trt.float32, shape)

    network = Network()
    context = quantization.VoiceChatQuantContext(
        weight_scales={"projection": np.array([0.5, 1.0], dtype=np.float32)},
        graph_ops=GraphOps,
    )
    result = context.maybe_quantized_matmul(
        network,
        Tensor(trt.float16, (3, 2)),
        2,
        2,
        np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32),
        "projection",
        dtype=np.float32,
    )

    assert result.dtype == trt.float16
    assert len(explicit_weights) == 1
    assert explicit_weights[0].dtype == trt.int8
    assert explicit_weights[0].size == 4
    kept = quantization._INT8_WEIGHT_KEEPALIVE[network]
    np.testing.assert_array_equal(kept[0], np.array([[4, 4], [12, 8]], dtype=np.int8))
    assert explicit_weights[0].pointer == kept[0].ctypes.data

    assert calls["reduce"][0][1:] == (trt.ReduceOperation.MAX, 0b10, True)
    assert [call[1] for call in calls["unary"]] == [
        trt.UnaryOperation.ABS,
        trt.UnaryOperation.ROUND,
    ]
    assert [call[2] for call in calls["elementwise"]] == [
        trt.ElementWiseOperation.MAX,
        trt.ElementWiseOperation.DIV,
        trt.ElementWiseOperation.DIV,
        trt.ElementWiseOperation.MAX,
        trt.ElementWiseOperation.MIN,
        trt.ElementWiseOperation.PROD,
    ]
    assert len(calls["dequantize"]) == 2
    assert calls["dequantize"][0][0].dtype == trt.int8
    assert calls["dequantize"][0][3].axis == 1
    assert calls["dequantize"][1][1].shape == ()
    assert not hasattr(network, "add_quantize")
    assert [entry[0] for entry in graph_constants] == [
        (2,),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (),
    ]


@pytest.mark.parametrize("raises", [False, True])
def test_serialization_releases_pointer_buffers_and_makes_network_one_shot(
    raises: bool,
) -> None:
    class Network:
        pass

    network = Network()
    quantization._retain_int8_weight_buffer(network, np.ones(4, dtype=np.int8))

    class Builder:
        def build_serialized_network(self, received_network, received_config):
            assert received_network is network
            assert received_config == "config"
            assert network in quantization._INT8_WEIGHT_KEEPALIVE
            if raises:
                raise RuntimeError("serialization failed")
            return b"plan"

    if raises:
        with pytest.raises(RuntimeError, match="serialization failed"):
            quantization.build_serialized_network(Builder(), network, "config")
    else:
        assert (
            quantization.build_serialized_network(Builder(), network, "config")
            == b"plan"
        )

    assert network not in quantization._INT8_WEIGHT_KEEPALIVE
    with pytest.raises(RuntimeError, match="serialized only once"):
        quantization.prepare_int8_weight_serialization(network)


def test_abandoned_network_releases_pointer_buffers_on_collection() -> None:
    class Network:
        pass

    network = Network()
    buffer = np.ones(4, dtype=np.int8)
    network_ref = weakref.ref(network)
    buffer_ref = weakref.ref(buffer)
    quantization._retain_int8_weight_buffer(network, buffer)

    del buffer
    del network
    gc.collect()

    assert network_ref() is None
    assert buffer_ref() is None
    assert not quantization._INT8_WEIGHT_KEEPALIVE
    assert quantization.prepare_int8_weight_serialization(Network()) is False


def test_outer_build_scope_releases_prior_buffers_after_later_failure() -> None:
    class Network:
        pass

    network = Network()
    with pytest.raises(RuntimeError, match="later graph construction failed"):
        with quantization.int8_weight_build_scope(network):
            quantization._retain_int8_weight_buffer(network, np.ones(4, dtype=np.int8))
            assert network in quantization._INT8_WEIGHT_KEEPALIVE
            raise RuntimeError("later graph construction failed")

    assert network not in quantization._INT8_WEIGHT_KEEPALIVE
    with pytest.raises(RuntimeError, match="serialized only once"):
        quantization.prepare_int8_weight_serialization(network)


def test_pointer_backed_network_must_support_weak_references() -> None:
    with pytest.raises(TypeError, match="must be weak-referenceable and hashable"):
        quantization._retain_int8_weight_buffer(object(), np.ones(4, dtype=np.int8))


def test_graph_construction_failure_releases_sibling_pointer_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        dtype = "fp32"
        shape = (1, 2)

    class Layer:
        axis = None

        def get_output(self, index):
            assert index == 0
            return Tensor()

    class Weights:
        def __init__(self, dtype, pointer, size):
            del pointer, size
            self.dtype = dtype

    trt = SimpleNamespace(int8="int8", Weights=Weights)
    monkeypatch.setattr(quantization, "_trt", lambda: trt)

    class Network:
        def add_constant(self, _shape, _weights):
            return Layer()

    class FailingGraphOps:
        @staticmethod
        def add_constant(*_args, **_kwargs):
            raise RuntimeError("scale graph construction failed")

    network = Network()
    quantization._retain_int8_weight_buffer(network, np.array([7], dtype=np.int8))
    context = quantization.VoiceChatQuantContext(
        weight_scales={"projection": np.array([0.5, 0.5], dtype=np.float32)},
        graph_ops=FailingGraphOps,
    )

    with pytest.raises(RuntimeError, match="scale graph construction failed"):
        context.maybe_quantized_matmul(
            network,
            Tensor(),
            2,
            2,
            np.ones((2, 2), dtype=np.float32),
            "projection",
        )

    assert network not in quantization._INT8_WEIGHT_KEEPALIVE
    with pytest.raises(RuntimeError, match="serialized only once"):
        quantization.prepare_int8_weight_serialization(network)
