# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT topology checks for qualified Qwen native-KV attention paths."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from tests.builder.conftest import requires_trt

trt = pytest.importorskip("tensorrt")

_FAMILIES = (
    "tensorrt_model_connect.families.qwen.graph_ops",
    "tensorrt_model_connect.families.llama.graph_ops",
)


class _FakeTensor:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype


class _FakeLayer:
    def __init__(self, output):
        self._output = output

    def get_output(self, index):
        assert index == 0
        return self._output


class _FakeNetwork:
    def __init__(self):
        self.calls = []

    def _layer(self, operation, dtype):
        return _FakeLayer(_FakeTensor(f"{operation}_{len(self.calls)}", dtype))

    def add_kv_cache_update(self, cache, update, indices, mode):
        layer = self._layer("kv_cache_update", cache.dtype)
        self.calls.append(("kv_cache_update", cache, update, indices, mode, layer))
        return layer

    def add_cast(self, tensor, dtype):
        layer = self._layer("cast", dtype)
        self.calls.append(("cast", tensor, dtype, layer))
        return layer

    def add_elementwise(self, lhs, rhs, operation):
        layer = self._layer("elementwise", lhs.dtype)
        self.calls.append(("elementwise", lhs, rhs, operation, layer))
        return layer

    def add_attention_v2(self, q, k, v, normalization, causal_mask):
        layer = self._layer("attention", q.dtype)
        self.calls.append(("attention", q, k, v, normalization, causal_mask, layer))
        return layer


class _AttentionRecordingNetwork:
    def __init__(self, network):
        self._network = network
        self.attention = None

    def __getattr__(self, name):
        return getattr(self._network, name)

    def add_attention_v2(self, *args, **kwargs):
        self.attention = self._network.add_attention_v2(*args, **kwargs)
        return self.attention


def _add_native_attention(monkeypatch, graph_ops, dtype):
    network = _FakeNetwork()
    heads, kv_heads, head_dim = 4, 2, 128
    tensors = {
        "q": _FakeTensor("q", dtype),
        "k": _FakeTensor("k", dtype),
        "v": _FakeTensor("v", dtype),
        "cache_k": _FakeTensor("cache_k", dtype),
        "cache_v": _FakeTensor("cache_v", dtype),
        "write_indices": _FakeTensor("cache_write_indices", trt.int32),
        "lengths": _FakeTensor("key_value_lengths", trt.int32),
    }
    monkeypatch.setattr(
        graph_ops,
        "reshape_rows_to_heads_4d",
        lambda _network, tensor, *_args, **_kwargs: tensor,
    )
    monkeypatch.setattr(
        graph_ops,
        "reshape_heads_4d_to_rows",
        lambda _network, tensor, *_args, **_kwargs: tensor,
    )
    result = graph_ops.add_native_kv_cache_attention_from_rows(
        network,
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["cache_k"],
        tensors["cache_v"],
        tensors["write_indices"],
        tensors["lengths"],
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        q_seq=1,
    )
    return network, tensors, result


def _dual_profile_builder_module():
    # The Qwen package lazy-loads its plugin; enter through that public module
    # so its builder imports are initialized in the same order as production.
    importlib.import_module("tensorrt_model_connect.families.qwen.plugin")
    return importlib.import_module(
        "tensorrt_model_connect.families.qwen.dual_profile_decoder_builder"
    )


@pytest.mark.parametrize(
    (
        "native_kv_cache",
        "explicit_mask",
        "profile_mode",
        "cache_length",
        "opt_length",
        "requested",
        "expected",
    ),
    [
        (True, True, "prefill", 16384, 64, None, (64, 64)),
        (True, True, "prefill", 16384, 128, 128, (64, 64)),
        (True, True, "prefill", 16384, 64, 32, (32, 32)),
        (True, True, "dual_profile", 16384, 128, None, (64, 64)),
        (True, True, "decode", 16384, 64, None, (64, 16384)),
        (True, False, "prefill", 40960, 64, None, (64, 32768)),
        (False, False, "prefill", 40960, 64, None, (64, 40960)),
    ],
)
def test_explicit_mask_prefill_limit(
    native_kv_cache,
    explicit_mask,
    profile_mode,
    cache_length,
    opt_length,
    requested,
    expected,
):
    builder = _dual_profile_builder_module()

    assert builder._resolve_prefill_lengths(
        cache_length,
        opt_length,
        requested,
        native_kv_cache=native_kv_cache,
        explicit_native_attention_mask=explicit_mask,
        profile_mode=profile_mode,
    ) == expected


@pytest.mark.parametrize("compute_capability", [(8, 6), (12, 1)])
def test_current_cuda_compute_capability(monkeypatch, compute_capability):
    builder = _dual_profile_builder_module()
    properties = SimpleNamespace(
        major=compute_capability[0], minor=compute_capability[1]
    )
    runtime = SimpleNamespace(
        cudaGetDevice=lambda: (0, 0),
        cudaGetDeviceProperties=lambda _device: (0, properties),
    )
    monkeypatch.setattr(builder, "_cuda_runtime", lambda: runtime)

    assert builder._current_cuda_compute_capability() == compute_capability


def test_current_cuda_compute_capability_fails_closed(monkeypatch):
    builder = _dual_profile_builder_module()
    runtime = SimpleNamespace(cudaGetDevice=lambda: (7, 0))
    monkeypatch.setattr(builder, "_cuda_runtime", lambda: runtime)

    with pytest.raises(RuntimeError, match="cudaGetDevice failed"):
        builder._current_cuda_compute_capability()


@requires_trt
def test_qwen_explicit_mask_omits_native_lengths_on_real_trt_network():
    graph_ops = importlib.import_module(
        "tensorrt_model_connect.families.qwen.graph_ops"
    )
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    raw_network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    network = _AttentionRecordingNetwork(raw_network)
    q = network.add_input("q", trt.bfloat16, (1, 512))
    k = network.add_input("k", trt.bfloat16, (1, 256))
    v = network.add_input("v", trt.bfloat16, (1, 256))
    cache_k = network.add_input("cache_k", trt.bfloat16, (1, 2, 8, 128))
    cache_v = network.add_input("cache_v", trt.bfloat16, (1, 2, 8, 128))
    write_indices = network.add_input(
        "cache_write_indices", trt.int32, (1,)
    )
    lengths = network.add_input("key_value_lengths", trt.int32, (1,))
    mask = network.add_input("explicit_mask", trt.bfloat16, (1, 1, 1, 8))

    result = graph_ops.add_native_kv_cache_attention_from_rows(
        network,
        q,
        k,
        v,
        cache_k,
        cache_v,
        write_indices,
        lengths,
        num_heads=4,
        num_kv_heads=2,
        head_dim=128,
        q_seq=1,
        explicit_mask=mask,
    )

    attention = network.attention
    assert isinstance(attention, trt.IAttention)
    assert attention.num_inputs == 4
    assert attention.mask is mask
    assert attention.key_value_lengths is None
    assert attention.causal_kind == trt.CausalMaskKind.NONE
    assert attention.decomposable is False
    assert attention.get_input(1) is result["present_k"]
    assert attention.get_input(2) is result["present_v"]


@requires_trt
def test_qwen_fp16_explicit_mask_uses_bf16_native_attention_boundary():
    graph_ops = importlib.import_module(
        "tensorrt_model_connect.families.qwen.graph_ops"
    )
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    raw_network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    network = _AttentionRecordingNetwork(raw_network)
    q = network.add_input("q", trt.float16, (1, 512))
    k = network.add_input("k", trt.float16, (1, 256))
    v = network.add_input("v", trt.float16, (1, 256))
    cache_k = network.add_input("cache_k", trt.bfloat16, (1, 2, 8, 128))
    cache_v = network.add_input("cache_v", trt.bfloat16, (1, 2, 8, 128))
    write_indices = network.add_input(
        "cache_write_indices", trt.int32, (1,)
    )
    lengths = network.add_input("key_value_lengths", trt.int32, (1,))
    mask = network.add_input("explicit_mask", trt.bfloat16, (1, 1, 1, 8))

    result = graph_ops.add_native_kv_cache_attention_from_rows(
        network,
        q,
        k,
        v,
        cache_k,
        cache_v,
        write_indices,
        lengths,
        num_heads=4,
        num_kv_heads=2,
        head_dim=128,
        q_seq=1,
        explicit_mask=mask,
        attention_dtype=trt.bfloat16,
    )

    attention = network.attention
    assert isinstance(attention, trt.IAttention)
    assert attention.num_inputs == 4
    assert attention.get_input(0).dtype == trt.bfloat16
    assert attention.get_input(1).dtype == trt.bfloat16
    assert attention.get_input(2).dtype == trt.bfloat16
    assert attention.mask is mask
    assert attention.key_value_lengths is None
    assert attention.causal_kind == trt.CausalMaskKind.NONE
    assert attention.decomposable is False
    assert result["present_k"].dtype == trt.bfloat16
    assert result["present_v"].dtype == trt.bfloat16
    assert result["context"].dtype == trt.float16


@pytest.mark.parametrize("module_name", _FAMILIES)
def test_bf16_uses_exact_fp32_scale_before_fused_attention(
    monkeypatch,
    module_name,
):
    graph_ops = importlib.import_module(module_name)
    constants: list[tuple[tuple[int, ...], np.ndarray, np.dtype]] = []

    def _record(network, shape, values, dtype=np.float32):
        constants.append((shape, np.asarray(values).copy(), np.dtype(dtype)))
        tensor = _FakeTensor("scale", trt.float32)
        network.calls.append(("constant", tensor))
        return tensor

    monkeypatch.setattr(graph_ops, "add_constant", _record)
    network, tensors, result = _add_native_attention(monkeypatch, graph_ops, trt.bfloat16)
    relevant = [
        call
        for call in network.calls
        if call[0] in {"cast", "constant", "elementwise", "attention"}
    ]

    assert len(constants) == 1
    shape, values, dtype = constants[0]
    assert shape == (1, 1, 1, 1)
    assert dtype == np.dtype(np.float32)
    assert float(values.item()) == pytest.approx(1.0 / np.sqrt(128), rel=0.0, abs=0.0)
    assert [call[0] for call in relevant] == [
        "cast",
        "constant",
        "elementwise",
        "cast",
        "attention",
    ]
    upcast, constant, product, downcast, attention = relevant
    assert upcast[1] is tensors["q"]
    assert upcast[2] == trt.float32
    assert upcast[3].get_output(0).dtype == trt.float32
    assert constant[1].dtype == trt.float32
    assert product[1] is upcast[3].get_output(0)
    assert product[2] is constant[1]
    assert product[3] == trt.ElementWiseOperation.PROD
    assert downcast[1] is product[4].get_output(0)
    assert downcast[2] == trt.bfloat16
    assert attention[1] is downcast[3].get_output(0)
    assert attention[2] is result["present_k"]
    assert attention[3] is result["present_v"]
    assert attention[4] == trt.AttentionNormalizationOp.SOFTMAX
    assert attention[5] == trt.CausalMaskKind.LOWER_RIGHT
    assert attention[6].decomposable is False
    assert attention[6].key_value_lengths is tensors["lengths"]
    assert result["context"] is attention[6].get_output(0)


def test_qwen_fp16_uses_exact_fp32_scale_before_fused_attention(monkeypatch):
    graph_ops = importlib.import_module("tensorrt_model_connect.families.qwen.graph_ops")
    constants: list[tuple[tuple[int, ...], np.ndarray, np.dtype]] = []

    def _record(network, shape, values, dtype=np.float32):
        constants.append((shape, np.asarray(values).copy(), np.dtype(dtype)))
        tensor = _FakeTensor("scale", trt.float32)
        network.calls.append(("constant", tensor))
        return tensor

    monkeypatch.setattr(graph_ops, "add_constant", _record)
    network, tensors, result = _add_native_attention(monkeypatch, graph_ops, trt.float16)
    relevant = [
        call
        for call in network.calls
        if call[0] in {"cast", "constant", "elementwise", "attention"}
    ]

    assert len(constants) == 1
    assert [call[0] for call in relevant] == [
        "cast",
        "constant",
        "elementwise",
        "cast",
        "attention",
    ]
    upcast, constant, product, downcast, attention = relevant
    assert upcast[1] is tensors["q"]
    assert upcast[2] == trt.float32
    assert constant[1].dtype == trt.float32
    assert product[3] == trt.ElementWiseOperation.PROD
    assert downcast[2] == trt.float16
    assert attention[1] is downcast[3].get_output(0)
    assert attention[2] is result["present_k"]
    assert attention[3] is result["present_v"]


def test_llama_unqualified_fp16_graph_fails_closed(monkeypatch):
    graph_ops = importlib.import_module("tensorrt_model_connect.families.llama.graph_ops")

    with pytest.raises(ValueError, match="requires BF16"):
        _add_native_attention(monkeypatch, graph_ops, trt.float16)
