# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from types import SimpleNamespace

from families.minimax_h3 import vae_builder as vae
from families.minimax_h3.tests.vae_decoder_prefix_localize import compare, save_array


def test_prefix_diagnostic_keeps_dtype_evidence_when_values_match():
    actual = np.ones((2, 4), np.float16)
    expected = np.ones((2, 4), np.float32)
    result = compare(actual, expected)
    assert not result["dtype_equal"] and not result["byte_exact"]
    assert result["mismatched_elements"] == 0 and result["compared_elements"] == 8


def test_prefix_diagnostic_checks_near_zero_without_comparison_downcast():
    actual = np.array([0.0001, 1.0], np.float32)
    expected = np.array([0.0, 1.0], np.float32)
    result = compare(actual, expected)
    assert result["dtype_equal"] and not result["byte_exact"]
    assert result["mismatched_elements"] == 1 and result["compared_elements"] == 2


def test_prefix_evidence_retains_full_unique_payloads_without_rewriting(tmp_path):
    saved = {}
    value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    first = save_array(tmp_path, "first", value, saved)
    alias = save_array(tmp_path, "second-rebuild", value.copy(), saved)
    assert first == alias and not (tmp_path / "second-rebuild.npy").exists()
    changed = value.copy()
    changed[1, 2, 3] += 0.25
    second = save_array(tmp_path, "different", changed, saved)
    reshaped = save_array(tmp_path, "different-shape", value.reshape(4, 6), saved)
    half = save_array(tmp_path, "different-dtype", value.astype(np.float16), saved)
    assert len(saved) == 4 and len(list(tmp_path.iterdir())) == 4
    for record, expected in [
        (first, value),
        (second, changed),
        (reshaped, value.reshape(4, 6)),
        (half, value.astype(np.float16)),
    ]:
        actual = np.load(tmp_path / record["path"], allow_pickle=False)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("batch", [1, 28])
def test_video_decoder_preserves_fp32_registers_before_first_residual(monkeypatch, batch):
    # Typed graph construction only, not a trained-model or numerical oracle.
    # Upstream proj_in is autocast FP16, while the learned register parameter
    # stays FP32; torch.cat promotes the initial residual sequence to FP32.
    class CapturedPacking(Exception):
        pass

    def tensor(shape, dtype, values=None):
        return SimpleNamespace(shape=shape, dtype=dtype, values=values)

    def layer(value):
        return SimpleNamespace(get_output=lambda _index: value)

    registers = np.full((1, 4, 2048), 1.0001, dtype=np.float32)

    class Network:
        def add_input(self, _name, dtype, shape):
            return tensor(shape, dtype)

        def add_shuffle(self, value):
            return layer(value)

        def add_cast(self, value, dtype):
            return layer(tensor(value.shape, dtype, value.values))

        def add_concatenation(self, values):
            assert len(values) == 3
            assert [value.dtype for value in values] == [vae.trt.float32] * 3
            assert [tuple(value.shape) for value in values] == [
                (batch, 1792, 2048),
                (batch, 4, 2048),
                (batch, 1, 2048),
            ]
            np.testing.assert_array_equal(
                values[1].values, np.broadcast_to(registers, (batch, 4, 2048))
            )
            raise CapturedPacking

    class Config:
        def clear_flag(self, _flag):
            pass

    class Builder:
        def create_network(self, _flags):
            return Network()

        def create_builder_config(self):
            return Config()

    calls = []

    def linear(_network, _value, _weight, _bias, *, compute_dtype):
        calls.append(compute_dtype)
        return tensor((batch, 1792, 2048), compute_dtype)

    def constant(_network, value):
        value = np.asarray(value)
        assert value.dtype == np.float32
        return tensor(value.shape, vae.trt.float32, value)

    monkeypatch.setattr(vae.trt, "Builder", lambda _logger: Builder())
    monkeypatch.setattr(vae.op, "configure_workspace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vae.op, "linear", linear)
    monkeypatch.setattr(vae.op, "constant", constant)
    monkeypatch.setattr(vae.op, "weight_constant", constant)
    monkeypatch.setattr(
        vae, "_post_quant_conv", lambda *_args: tensor((batch, 1792, 24), vae.trt.float16)
    )
    weights = {
        "post_quant_conv.weight": np.zeros((24, 24), np.float32),
        "post_quant_conv.bias": np.zeros(24, np.float32),
        "decoder.proj_in.weight": np.zeros((1, 1), np.float32),
        "decoder.proj_in.bias": np.zeros(1, np.float32),
        "decoder.register_tokens": registers,
    }
    with pytest.raises(CapturedPacking):
        vae.build_vae_tile_decoder_engine(weights, batch_size=batch)
    assert calls == [vae.trt.float16]


@pytest.mark.parametrize("batch", [1, 28])
def test_video_post_quantization_preserves_half_operands_float_accumulation_and_bias(batch):
    # Original trained parameter shapes and autocast dtypes; no GPU or model
    # math is supplied by this graph-contract fixture.
    recorded = {}
    source = SimpleNamespace(shape=(batch, 24, 7, 16, 16), dtype=vae.trt.float32)

    def tensor(dtype, label):
        return SimpleNamespace(shape=source.shape, dtype=dtype, label=label)

    class Shuffle:
        def get_output(self, _index):
            # Permutation storage has eight slots; only this input's rank is active.
            assert tuple(self.first_transpose[index] for index in range(5)) == (0, 2, 3, 4, 1)
            assert self.reshape_dims == (batch, 1792, 24)
            return SimpleNamespace(shape=self.reshape_dims, dtype=vae.trt.float16)

    class Network:
        def add_cast(self, value, dtype):
            if value is source:
                assert dtype == vae.trt.float16
                label = "half-input"
            elif value.label == "half-input":
                assert dtype == vae.trt.float32
                label = "float-input"
            elif value.label == "float-convolution":
                assert dtype == vae.trt.float16
                label = "half-convolution"
            elif value.label == "half-convolution":
                assert dtype == vae.trt.float32
                label = "wide-convolution"
            else:
                assert value.label == "wide-sum" and dtype == vae.trt.float16
                label = "half-output"
            return SimpleNamespace(get_output=lambda _index: tensor(dtype, label))

        def add_convolution_nd(self, value, channels, kernel_shape, kernel, bias):
            assert value.dtype == vae.trt.float32 and value.shape == source.shape
            assert value.label == "float-input"
            assert channels == 24 and kernel_shape == (1, 1, 1)
            assert kernel.dtype == np.float32 and kernel.flags.c_contiguous
            assert isinstance(bias, vae.trt.Weights) and bias.dtype == vae.trt.float32
            assert bias.size == 0, "bias must follow the half-rounded convolution"
            recorded.update(kernel=kernel, bias=bias)
            recorded["layer"] = SimpleNamespace(
                get_output=lambda _index: tensor(vae.trt.float32, "float-convolution")
            )
            return recorded["layer"]

        def add_constant(self, shape, values):
            assert tuple(shape) == (1, 24, 1, 1, 1) and values.dtype == np.float32
            # Widen the autocast-half bias, not the original FP32 parameter.
            expected = weights["post_quant_conv.bias"].astype(np.float16).astype(np.float32)
            np.testing.assert_array_equal(values.reshape(24), expected)
            recorded["bias_constant"] = values
            return SimpleNamespace(get_output=lambda _index: tensor(vae.trt.float32, "wide-bias"))

        def add_elementwise(self, left, right, operation):
            assert left.label == "wide-convolution" and right.label == "wide-bias"
            assert left.dtype == right.dtype == vae.trt.float32
            assert operation == vae.trt.ElementWiseOperation.SUM
            return SimpleNamespace(get_output=lambda _index: tensor(vae.trt.float32, "wide-sum"))

        def add_shuffle(self, value):
            assert value.dtype == vae.trt.float16 and value.shape == source.shape
            assert value.label == "half-output"
            return Shuffle()

    weights = {
        "post_quant_conv.weight": np.arange(576, dtype=np.float32).reshape(24, 24, 1, 1, 1) / 1000,
        "post_quant_conv.bias": np.linspace(-0.01, 0.01, 24, dtype=np.float32),
    }
    network = Network()
    try:
        result = vae._post_quant_conv(network, source, weights)
        assert result.shape == (batch, 1792, 24) and result.dtype == vae.trt.float16
        assert recorded["layer"].stride_nd == (1, 1, 1) and recorded["layer"].padding_nd == (
            0,
            0,
            0,
        )
        np.testing.assert_array_equal(
            recorded["kernel"],
            weights["post_quant_conv.weight"].astype(np.float16).astype(np.float32),
        )
        for name in ("kernel", "bias_constant"):
            assert any(
                value is recorded[name] for value in vae.op._WEIGHT_BUFFER_KEEPALIVE[id(network)]
            )
    finally:
        vae.op.release_weight_buffers(network)


def test_video_post_quantization_rejected_convolution_does_not_retain_buffers():
    source = SimpleNamespace(shape=(1, 24, 7, 16, 16), dtype=vae.trt.float16)
    network = SimpleNamespace(
        add_convolution_nd=lambda *_args: None,
        add_cast=lambda value, dtype: SimpleNamespace(
            get_output=lambda _index: SimpleNamespace(shape=value.shape, dtype=dtype)
        ),
    )
    weights = {
        "post_quant_conv.weight": np.zeros((24, 24, 1, 1, 1), np.float32),
        "post_quant_conv.bias": np.zeros(24, np.float32),
    }
    with pytest.raises(RuntimeError, match="rejected"):
        vae._post_quant_conv(network, source, weights)
    assert id(network) not in vae.op._WEIGHT_BUFFER_KEEPALIVE


@pytest.mark.parametrize("batch", [0, -1, 2, 29, True, 1.0, "1", None])
def test_video_decoder_rejects_invalid_tile_batch_before_cuda(batch):
    with pytest.raises(ValueError, match="batch_size must be 1 or 28"):
        vae.build_vae_tile_decoder_engine({}, batch_size=batch)


@pytest.mark.parametrize("batch", [1, 28])
def test_video_decoder_static_batch_threads_through_typed_input(monkeypatch, batch):
    class CapturedInput(Exception):
        pass

    class Network:
        def add_input(self, name, dtype, shape):
            assert name == "latent_tiles" and dtype == vae.trt.float32
            assert shape == (batch, 24, 7, 16, 16)
            raise CapturedInput

    class Config:
        def clear_flag(self, flag):
            assert flag == vae.trt.BuilderFlag.TF32

        def set_memory_pool_limit(self, _pool, value):
            self.limit = value

        def get_memory_pool_limit(self, _pool):
            return self.limit

    class Builder:
        def create_network(self, flags):
            assert flags == 1 << int(vae.trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            return Network()

        def create_builder_config(self):
            return Config()

    monkeypatch.setattr(vae.trt, "Builder", lambda _logger: Builder())
    with pytest.raises(CapturedInput):
        vae.build_vae_tile_decoder_engine({}, batch_size=batch, workspace_bytes=1 << 30)
