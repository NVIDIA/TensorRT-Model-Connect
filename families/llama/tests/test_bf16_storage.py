# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checkpoint and typed graph-prefix checks, without TensorRT execution."""

import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
from safetensors.numpy import save_file

from ..checkpoint_mapper import load_standard_weights
from ..config import ModelConfig


class _Tensor:
    def __init__(self, values, dtype="fp32", shape=None):
        storage = {"fp32": np.float32, "fp16": np.float16, "bf16": ml_dtypes.bfloat16}
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            self.values = np.asarray(values).astype(storage.get(dtype, np.int32)).astype(np.float32)
        self.dtype = dtype
        self.shape = self.values.shape if shape is None else shape


def _layer(tensor):
    return SimpleNamespace(get_output=lambda index: tensor)


class _PrefixComplete(Exception):
    pass


class _Network:
    def __init__(self, stop_at_epsilon=False):
        self.constants = []
        self.inputs = {}
        self.stop_at_epsilon = stop_at_epsilon

    def add_constant(self, shape, values):
        values = values.reshape(shape)
        self.constants.append(values.copy())
        if self.stop_at_epsilon and shape == (1, 1):
            raise _PrefixComplete
        dtype = "fp16" if values.dtype == np.float16 else "fp32"
        return _layer(_Tensor(values, dtype))

    def add_input(self, name, dtype, shape):
        tensor = _Tensor(np.zeros(tuple(max(1, n) for n in shape)), dtype, shape)
        self.inputs[name] = tensor
        return tensor

    def add_cast(self, tensor, dtype):
        return _layer(_Tensor(tensor.values, dtype, tensor.shape))

    def add_elementwise(self, lhs, rhs, operation):
        assert lhs.dtype == rhs.dtype, "mixed elementwise dtypes"
        function = {"prod": np.multiply, "sum": np.add, "sub": np.subtract}[operation]
        return _layer(_Tensor(function(lhs.values, rhs.values), lhs.dtype))

    def add_reduce(self, tensor, operation, axes, keep_dims):
        assert operation == "avg"
        dimensions = tuple(i for i in range(tensor.values.ndim) if axes & (1 << i))
        return _layer(
            _Tensor(tensor.values.mean(axis=dimensions, keepdims=keep_dims), tensor.dtype)
        )

    def add_unary(self, tensor, operation):
        function = {"sqrt": np.sqrt, "recip": np.reciprocal}[operation]
        return _layer(_Tensor(function(tensor.values), tensor.dtype))

    def add_shuffle(self, tensor):
        shuffle = SimpleNamespace(reshape_dims=tensor.shape)
        shuffle.get_output = lambda index: _Tensor(
            tensor.values.reshape(shuffle.reshape_dims), tensor.dtype
        )
        return shuffle


@pytest.fixture
def family(monkeypatch):
    network = _Network()
    config = SimpleNamespace(add_optimization_profile=lambda profile: None)
    builder = SimpleNamespace(
        create_network=lambda flags: network,
        create_builder_config=lambda: config,
        create_optimization_profile=lambda: SimpleNamespace(set_shape=lambda *args: None),
    )
    logger = type("Logger", (), {"VERBOSE": 1, "WARNING": 2, "__init__": lambda self, level: None})
    trt = SimpleNamespace(
        float16="fp16",
        float32="fp32",
        bfloat16="bf16",
        int32="int32",
        Weights=lambda values: values,
        Logger=logger,
        Builder=lambda logger: builder,
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=1),
        ElementWiseOperation=SimpleNamespace(PROD="prod", SUM="sum", SUB="sub"),
        ReduceOperation=SimpleNamespace(AVG="avg"),
        UnaryOperation=SimpleNamespace(SQRT="sqrt", RECIP="recip"),
    )
    name = "_llama_bf16_storage_test"
    package = ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setitem(sys.modules, name, package)
    try:
        yield SimpleNamespace(
            graph=importlib.import_module(name + ".graph_ops"),
            standard=importlib.import_module(name + ".standard_decoder_builder"),
            dual=importlib.import_module(name + ".dual_profile_decoder_builder"),
            network=network,
        )
    finally:
        for key in tuple(sys.modules):
            if key.startswith(name + "."):
                del sys.modules[key]


def _checkpoint(tmp_path, tied, *, wide=True):
    config = ModelConfig(
        model_type="llama",
        hidden_size=4,
        vocab_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=4,
        rms_norm_eps=2**-30,
        raw={"_decoder_engine_role": "decode"},
    )
    values = np.array(
        [2**20, -(2**20), 2**-30, -(2**-30)] if wide else [1024, -1024, 0.125, -0.125],
        dtype=ml_dtypes.bfloat16,
    )
    matrix = np.tile(values, (4, 1))
    tensors = {"model.embed_tokens.weight": matrix}
    if not tied:
        tensors["lm_head.weight"] = matrix
    for norm in ["input_layernorm", "post_attention_layernorm"]:
        tensors[f"model.layers.0.{norm}.weight"] = np.ones(4, dtype=ml_dtypes.bfloat16)
    for projection in [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]:
        tensors[f"model.layers.0.{projection}.weight"] = matrix
    save_file(tensors, tmp_path / "model.safetensors")
    return config, matrix.astype(np.float32)


@pytest.mark.parametrize("precision", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("tied", [False, True])
def test_checkpoint_preserves_requested_storage_range(tmp_path, precision, tied):
    config, source = _checkpoint(tmp_path, tied, wide=precision != "fp16")
    dtype = np.float16 if precision == "fp16" else np.float32
    with np.errstate(over="ignore", under="ignore"):
        weights = load_standard_weights(tmp_path, config, precision=precision)
        expected = source.astype(dtype)
    for key in [
        "embedding",
        "w_out",
        "layer.0.w_q",
        "layer.0.w_k",
        "layer.0.w_v",
        "layer.0.w_o",
        "layer.0.w_gate",
        "layer.0.w_up",
        "layer.0.w_down",
    ]:
        assert weights[key].dtype == dtype
        np.testing.assert_array_equal(weights[key], expected if key == "embedding" else expected.T)


@pytest.mark.parametrize("precision", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("builder", ["standard", "dual"])
def test_decoder_prefix_preserves_constant_storage(family, tmp_path, precision, builder):
    config, values = _checkpoint(tmp_path, True, wide=precision != "fp16")
    weights = {
        "embedding": values,
        "position_embedding": values,
        "final_norm": np.ones(4),
        "w_out": values.T,
    }
    family.network.stop_at_epsilon = True
    module = getattr(family, builder)
    function = getattr(
        module, f"build_{'standard' if builder == 'standard' else 'dual_profile'}_decoder_engine"
    )
    kwargs = {} if builder == "standard" else {"native_kv_cache": False, "profile_mode": "decode"}
    with np.errstate(over="ignore", under="ignore"):
        with pytest.raises(_PrefixComplete):
            function(config, weights, 4, precision=precision, position_type="learned", **kwargs)
        dtype = np.float16 if precision == "fp16" else np.float32
        expected = values.astype(dtype)
    for constant in family.network.constants[:2]:
        assert constant.dtype == dtype
        np.testing.assert_array_equal(constant, expected)
    assert (
        family.network.inputs["cache_k_0"].dtype
        == {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}[precision]
    )
    if precision == "bf16":
        np.testing.assert_array_equal(family.network.constants[-1], [[2**-30]])


@pytest.mark.parametrize(
    "dtype,storage", [("bf16", np.float32), ("fp16", np.float16), ("fp32", np.float32)]
)
@pytest.mark.parametrize("norm", ["rms", "head", "layer"])
def test_norm_uses_fp32_compute_with_independent_storage(family, dtype, storage, norm):
    inputs = _Tensor([[1.0, 2.0, 3.0, 4.0]], dtype)
    epsilon = _Tensor([[1e-5]], "fp32")
    gamma = np.ones(4, dtype=np.float32)
    kwargs = {"dtype": storage}
    if norm == "rms":
        result = family.graph.add_rms_norm(family.network, inputs, 4, gamma, epsilon, **kwargs)
        expected = inputs.values / np.sqrt(np.mean(inputs.values**2) + 1e-5)
    elif norm == "head":
        result = family.graph.add_rms_norm_per_head(
            family.network, inputs, 2, 2, gamma, epsilon, **kwargs
        )
        reshaped = inputs.values.reshape(1, 2, 2)
        expected = (reshaped / np.sqrt(np.mean(reshaped**2, axis=2, keepdims=True) + 1e-5)).reshape(
            1, 4
        )
    else:
        result = family.graph.add_layer_norm(
            family.network, inputs, 4, gamma, np.zeros(4), epsilon, **kwargs
        )
        centered = inputs.values - inputs.values.mean(axis=1, keepdims=True)
        expected = centered / np.sqrt(np.mean(centered**2) + 1e-5)
    assert result.dtype == dtype
    np.testing.assert_allclose(result.values, _Tensor(expected, dtype).values, rtol=1e-6, atol=1e-7)
