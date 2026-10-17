# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from types import SimpleNamespace

from families.minimax_h3 import vae_builder as vae
from families.minimax_h3.tests.vae_decoder_prefix_localize import compare


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
    weights = {
        "post_quant_conv.weight": np.zeros((24, 24), np.float32),
        "post_quant_conv.bias": np.zeros(24, np.float32),
        "decoder.proj_in.weight": np.zeros((1, 1), np.float32),
        "decoder.proj_in.bias": np.zeros(1, np.float32),
        "decoder.register_tokens": registers,
    }
    with pytest.raises(CapturedPacking):
        vae.build_vae_tile_decoder_engine(weights, batch_size=batch)
    assert calls == [vae.trt.float16, vae.trt.float16]


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
