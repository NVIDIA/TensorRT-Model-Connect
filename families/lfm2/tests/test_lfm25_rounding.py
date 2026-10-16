# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the owning graph helpers against the actual BF16 Torch operations.

These tests require TensorRT and CUDA. They build tiny networks, without loading
checkpoint weights, and retain exact equality as the rounding oracle.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def gpu():
    torch = pytest.importorskip("torch")
    trt = pytest.importorskip("tensorrt")
    if not torch.cuda.is_available():
        pytest.skip("LFM2 BF16 rounding regression requires CUDA")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("LFM2 BF16 rounding regression requires BF16 CUDA support")
    from families.lfm2 import model

    return torch, trt, model


class TinyEngine:
    def __init__(self, torch, trt, define, profiles=None):
        self.torch = torch
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        define(self.network)
        config = self.builder.create_builder_config()
        config.builder_optimization_level = 1
        if profiles:
            profile = self.builder.create_optimization_profile()
            for name, shapes in profiles.items():
                profile.set_shape(name, *shapes)
            assert config.add_optimization_profile(profile) >= 0
        plan = self.builder.build_serialized_network(self.network, config)
        assert plan is not None, "actual family helper failed to build"
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan)
        assert self.engine is not None
        self.context = self.engine.create_execution_context()
        assert self.context is not None

    def __call__(self, inputs):
        torch, trt = self.torch, self.trt
        dtypes = {
            trt.float32: torch.float32,
            trt.bfloat16: torch.bfloat16,
            trt.int32: torch.int32,
            trt.bool: torch.bool,
        }
        buffers = {}
        outputs = {}
        for name, value in inputs.items():
            if -1 in tuple(self.engine.get_tensor_shape(name)):
                assert self.context.set_input_shape(name, tuple(value.shape))
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            dtype = dtypes[self.engine.get_tensor_dtype(name)]
            shape = tuple(self.context.get_tensor_shape(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                value = inputs[name].contiguous()
                assert value.dtype == dtype and tuple(value.shape) == shape
            else:
                value = torch.empty(shape, dtype=dtype, device="cuda")
                outputs[name] = value
            buffers[name] = value
            assert self.context.set_tensor_address(name, value.data_ptr())
        assert self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return outputs


def _output(network, tensor, name):
    tensor.name = name
    network.mark_output(tensor)


def test_silu_rounds_once_like_torch_functional(gpu):
    torch, trt, model = gpu
    # Exercise thousands of distinct BF16 values, including the negative lobe
    # and the positive range where rounding the sigmoid early changes results.
    values = torch.linspace(-12, 12, 16384, device="cuda").reshape(1, -1).bfloat16()
    expected = torch.nn.functional.silu(values)
    prematurely_rounded = values * torch.sigmoid(values)
    assert torch.any(expected != prematurely_rounded), "fixture must expose double rounding"

    def define(network):
        input_tensor = network.add_input("input", trt.bfloat16, tuple(values.shape))
        _output(network, model._silu(network, input_tensor), "actual")

    engine = TinyEngine(torch, trt, define)
    actual = engine({"input": values})["actual"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("hidden", [4, 8])
def test_conv_selects_prefill_or_cached_decode_product_rounding(gpu, hidden):
    torch, trt, model = gpu
    # Every value is exactly representable in BF16. B=C=1 and identity X/output
    # projections isolate the real helper's convolution and state transitions.
    generator = torch.Generator(device="cuda").manual_seed(193)
    present = torch.randn((hidden, 3), generator=generator, device="cuda").bfloat16()
    conv_weight = torch.randn((hidden, 3), generator=generator, device="cuda").bfloat16()
    state = torch.cat(
        (torch.full((hidden, 1), 7, dtype=torch.bfloat16, device="cuda"), present[:, :2]), dim=1
    )
    values = present[:, 2].reshape(1, hidden).contiguous()
    prefill = torch.nn.functional.conv1d(
        present.unsqueeze(0), conv_weight.unsqueeze(1), groups=hidden
    ).reshape(1, hidden)
    decode = torch.sum(present * conv_weight, dim=1).reshape(1, hidden)
    assert torch.any(prefill != decode), "fixture must distinguish the two reference phases"

    projection = np.zeros((hidden, hidden * 3), dtype=np.float32)
    projection[:, hidden * 2 :] = np.eye(hidden, dtype=np.float32)
    weights = {
        "test.conv_in": projection,
        "test.conv_in_bias": np.concatenate((np.ones(hidden * 2), np.zeros(hidden))).astype(
            np.float32
        ),
        "test.conv_weight": conv_weight.float().cpu().numpy(),
        "test.conv_out": np.eye(hidden, dtype=np.float32),
    }

    def define(network):
        input_tensor = network.add_input("input", trt.bfloat16, (1, hidden))
        state_tensor = network.add_input("state", trt.bfloat16, (hidden, 3))
        phase_input = network.add_input("decode_phase", trt.int32, (1, 1))
        zero = network.add_constant((1, 1), np.zeros((1, 1), dtype=np.int32)).get_output(0)
        phase = network.add_elementwise(
            phase_input, zero, trt.ElementWiseOperation.GREATER
        ).get_output(0)
        output, updated = model._add_conv_layer(
            network,
            input_tensor,
            state_tensor,
            phase,
            weights,
            position_id=network.add_constant((1,), np.array([3], dtype=np.int32)).get_output(0),
            prefix="test",
            config=SimpleNamespace(hidden_size=hidden, conv_l_cache=3),
            storage_dtype=np.dtype(np.float32),
            work_dtype=trt.bfloat16,
        )
        _output(network, output, "actual")
        _output(network, updated, "present")

    engine = TinyEngine(torch, trt, define)
    assert engine.engine.get_tensor_dtype("decode_phase") == trt.int32
    assert tuple(engine.engine.get_tensor_shape("decode_phase")) == (1, 1)
    # Alternating phases on one execution context catches stale/unbound phase
    # inputs as well as a graph that always uses one product-rounding policy.
    for phase in (0, 1, 0, 1):
        result = engine(
            {
                "input": values,
                "state": state,
                "decode_phase": torch.tensor([[phase]], device="cuda", dtype=torch.int32),
            }
        )
        torch.testing.assert_close(result["present"], present, rtol=0, atol=0)
        torch.testing.assert_close(result["actual"], decode if phase else prefill, rtol=0, atol=0)


@pytest.mark.parametrize("num_heads", [2, 4])
@pytest.mark.parametrize("rows", [1, 17])
def test_rotary_rounds_cosine_and_sine_products_separately(gpu, num_heads, rows):
    torch, trt, model = gpu
    head_dim = 64
    generator = torch.Generator(device="cuda").manual_seed(193)
    values = (
        torch.randn((rows, num_heads * head_dim), generator=generator, device="cuda") * 2
    ).bfloat16()
    heads = values.reshape(rows, num_heads, head_dim)
    rotated = torch.cat((-heads[..., head_dim // 2 :], heads[..., : head_dim // 2]), dim=-1)
    frequencies = torch.linspace(0.001, 1, head_dim // 2, device="cuda")

    def define(network):
        input_tensor = network.add_input("input", trt.bfloat16, (rows, num_heads * head_dim))
        cosine = network.add_input("cosine", trt.bfloat16, (1, rows, head_dim // 2))
        sine = network.add_input("sine", trt.bfloat16, (1, rows, head_dim // 2))
        output = model._apply_rope(
            network, input_tensor, cosine, sine, num_heads=num_heads, head_dim=head_dim
        )
        _output(network, output, "actual")

    engine = TinyEngine(torch, trt, define)
    distinguishes_single_rounding = False
    for position in (0, 1, 17, 4095, 127999):
        positions = torch.arange(rows, device="cuda") + position
        angles = positions[:, None] * frequencies[None, :]
        cosine = angles.cos().bfloat16().reshape(1, rows, head_dim // 2)
        sine = angles.sin().bfloat16().reshape(1, rows, head_dim // 2)
        full_cosine = torch.cat((cosine, cosine), dim=-1).reshape(rows, 1, head_dim)
        full_sine = torch.cat((sine, sine), dim=-1).reshape(rows, 1, head_dim)
        # Each eager Torch multiplication produces BF16 before the addition.
        expected = (heads * full_cosine + rotated * full_sine).reshape_as(values)
        single_rounding = (
            (heads.float() * full_cosine.float() + rotated.float() * full_sine.float())
            .bfloat16()
            .reshape_as(values)
        )
        distinguishes_single_rounding |= bool(torch.any(expected != single_rounding))
        actual = engine({"input": values, "cosine": cosine, "sine": sine})["actual"]
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0, msg=f"rotary cache position {position}"
        )
    assert distinguishes_single_rounding, "fixture must expose fused rotary rounding"


@pytest.mark.parametrize("cache_length", [1, 3, 4])
@pytest.mark.parametrize("batched", [False, True])
def test_short_prefix_cache_updates_match_original_four_decode_steps(gpu, cache_length, batched):
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv

    torch, trt, model = gpu
    hidden = 4
    config = SimpleNamespace(hidden_size=hidden, conv_L_cache=cache_length, conv_bias=False)
    original = Lfm2ShortConv(config, 0).to(device="cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        original.in_proj.weight.zero_()
        original.in_proj.weight[: hidden * 2, 0] = 1  # B=C=1 from the constant first feature.
        original.in_proj.weight[hidden * 2 :] = torch.eye(hidden, device="cuda")
        original.out_proj.weight.copy_(torch.eye(hidden, device="cuda"))
        taps = torch.tensor([2.0**index for index in range(cache_length)], device="cuda")
        original.conv.weight.copy_(taps.reshape(1, 1, -1).expand(hidden, 1, -1))
    weights = {
        "test.conv_in": original.in_proj.weight.detach().float().T.cpu().numpy().copy(),
        "test.conv_weight": original.conv.weight.detach().float()[:, 0].cpu().numpy().copy(),
        "test.conv_out": original.out_proj.weight.detach().float().T.cpu().numpy().copy(),
    }

    def define(network):
        value = network.add_input("input", trt.bfloat16, (-1 if batched else 1, hidden))
        state = network.add_input("state", trt.bfloat16, (hidden, cache_length))
        position = network.add_input("position", trt.int32, (-1 if batched else 1,))
        decode = network.add_input("decode", trt.int32, (1, 1))
        zero = network.add_constant((1, 1), np.zeros((1, 1), np.int32)).get_output(0)
        phase = network.add_elementwise(decode, zero, trt.ElementWiseOperation.GREATER).get_output(
            0
        )
        conv_builder = model._add_batched_conv_layer if batched else model._add_conv_layer
        output, present = conv_builder(
            network,
            value,
            state,
            phase,
            weights,
            position_id=position,
            prefix="test",
            config=SimpleNamespace(hidden_size=hidden, conv_l_cache=cache_length),
            storage_dtype=np.dtype(np.float32),
            work_dtype=trt.bfloat16,
        )
        _output(network, output, "actual")
        _output(network, present, "present")

    profiles = (
        {
            "input": ((1, hidden), (17, hidden), (17, hidden)),
            "position": ((1,), (17,), (17,)),
        }
        if batched
        else None
    )
    engine = TinyEngine(torch, trt, define, profiles)
    for prefix_length in (1, 2, 17):
        values = torch.tensor(
            [[1, 3 * step + 2, 3 * step + 3, 3 * step + 4] for step in range(prefix_length + 4)],
            device="cuda",
            dtype=torch.bfloat16,
        ).unsqueeze(0)
        cache = SimpleNamespace(
            conv_cache=[torch.zeros((1, hidden, cache_length), device="cuda", dtype=torch.bfloat16)]
        )
        state = cache.conv_cache[0][0].clone()
        with torch.no_grad():
            prefill = original.slow_forward(
                values[:, :prefix_length], cache, torch.arange(prefix_length, device="cuda")
            )
        if batched:
            result = engine(
                {
                    "input": values[0, :prefix_length],
                    "state": state,
                    "position": torch.arange(prefix_length, device="cuda", dtype=torch.int32),
                    "decode": torch.zeros((1, 1), device="cuda", dtype=torch.int32),
                }
            )
            torch.testing.assert_close(result["actual"], prefill[0], rtol=0, atol=0)
            state = result["present"]
        for position in () if batched else range(prefix_length):
            result = engine(
                {
                    "input": values[:, position],
                    "state": state,
                    "position": torch.tensor([position], device="cuda", dtype=torch.int32),
                    "decode": torch.zeros((1, 1), device="cuda", dtype=torch.int32),
                }
            )
            torch.testing.assert_close(result["actual"], prefill[:, position], rtol=0, atol=0)
            state = result["present"]
        torch.testing.assert_close(state, cache.conv_cache[0][0], rtol=0, atol=0)
        for step in range(4):
            position = prefix_length + step
            with torch.no_grad():
                expected = original.slow_forward(
                    values[:, position : position + 1],
                    cache,
                    torch.tensor([position], device="cuda"),
                )
            result = engine(
                {
                    "input": values[:, position],
                    "state": state,
                    "position": torch.tensor([position], device="cuda", dtype=torch.int32),
                    "decode": torch.ones((1, 1), device="cuda", dtype=torch.int32),
                }
            )
            torch.testing.assert_close(result["actual"], expected[:, 0], rtol=0, atol=0)
            torch.testing.assert_close(result["present"], cache.conv_cache[0][0], rtol=0, atol=0)
            state = result["present"]
