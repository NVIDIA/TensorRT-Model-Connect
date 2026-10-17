# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the owning graph helpers against the actual BF16 Torch operations.

These tests require TensorRT and CUDA. They build tiny networks, without loading
checkpoint weights, and retain exact equality as the rounding oracle.
"""

from __future__ import annotations

import os
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

    # CI supplies the native build when no installed SDK prefix was selected.
    prefix = os.environ.get("CMAKE_PREFIX_PATH") or os.environ.get("TRTMC_NATIVE_BUILD_DIR")
    with pytest.MonkeyPatch.context() as environment:
        if prefix:
            environment.setenv("CMAKE_PREFIX_PATH", prefix)
        yield torch, trt, model


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
        self._load_plan(plan)

    @classmethod
    def from_plan(cls, torch, trt, plan):
        """Execute a plan built by an owning builder without rebuilding its graph."""
        instance = cls.__new__(cls)
        instance.torch, instance.trt = torch, trt
        instance.logger = trt.Logger(trt.Logger.WARNING)
        instance._load_plan(plan)
        return instance

    def _load_plan(self, plan):
        self.runtime = self.trt.Runtime(self.logger)
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


def _projection_residual_fixture(precision, rows):
    # Two nonzero products, both exactly representable. delta is half an ULP at 1.
    delta = 2.0 ** (-8 if precision == "bf16" else -11)
    sign = np.where(np.arange(rows) % 2, -1.0, 1.0).astype(np.float32)[:, None]
    x = np.zeros((rows, 2048), np.float32)
    x[:, :2] = sign * np.array([1, delta], np.float32)
    w = np.zeros((2048, 2048), np.float32)  # owning matmul layout[in,out]
    w[0] = np.tile([1, -1, 1, -1, 2, -2, 0, 1], 256)
    w[1] = np.tile([1, -1, 1, -1, 2, -2, 1, 0], 256)
    h = sign * np.tile(np.array([1, -1, -1, 1, 2, -2, 1, 1], np.float32) * delta, (rows, 256))
    projection = sign * np.tile(np.array([1, -1, 1, -1, 2, -2, delta, 1], np.float32), (rows, 256))
    # Literal dyadic oracles, independent of any graph or floating-point reducer.
    # Channels 0/1/4/5 tie at an even lower significand after each separate rounding.
    separate = sign * np.tile(
        np.array([1, -1, 1 - delta, -1 + delta, 2, -2, 2 * delta, 1], np.float32), (rows, 256)
    )
    fused = sign * np.tile(
        np.array(
            [1 + 2 * delta, -1 - 2 * delta, 1, -1, 2 + 4 * delta, -2 - 4 * delta, 2 * delta, 1],
            np.float32,
        ),
        (rows, 256),
    )
    assert np.count_nonzero(separate != fused) == rows * 1536
    # Channel 6 agrees and is nonzero while channel 0 differs: not uniform scaling.
    assert np.all(separate[:, 6] == fused[:, 6]) and np.all(separate[:, 6] != 0)
    assert np.all(separate[:, 0] != fused[:, 0])
    return x, w, h, projection, separate, fused


def _projection_residual_torch_oracle(torch, precision, rows, device):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    x, w, h, projection, expected, fused = _projection_residual_fixture(precision, rows)
    tx, tw, th = [torch.tensor(a, dtype=dtype, device=device) for a in (x, w, h)]
    for tensor, values in [(tx, x), (tw, w), (th, h)]:
        assert np.array_equal(tensor.float().cpu().numpy(), values), (
            "operands must be exactly representable"
        )
    actual_projection = torch.nn.functional.linear(tx, tw.T.contiguous())
    actual = actual_projection + th
    assert np.array_equal(actual_projection.float().cpu().numpy(), projection)
    assert np.array_equal(actual.float().cpu().numpy(), expected)
    return tx, th, expected, fused


def _check_projection_residual_graph(gpu, precision):
    """Two profiles, one graph; no tactic selection or checkpoint dependency."""
    torch, trt, model = gpu
    work_dtype = trt.bfloat16 if precision == "bf16" else trt.float16
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = network.add_input("input", work_dtype, (-1, 2048))
    h = network.add_input("residual", work_dtype, (-1, 2048))
    w = _projection_residual_fixture(precision, 1)[1]
    projected = model._matmul(
        network,
        x,
        w,
        lhs_width=2048,
        rhs_width=2048,
        storage_dtype=np.dtype(np.float32),
        work_dtype=work_dtype,
        name="projection",
    )
    result = model._residual_sum(network, projected, h)
    assert result.dtype == work_dtype
    result.name = "actual"
    network.mark_output(result)
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    for sizes in [(1, 17, 17), (1, 1, 1)]:
        profile = builder.create_optimization_profile()
        for name in ("input", "residual"):
            shapes = tuple((r, 2048) for r in sizes)
            profile.set_shape(name, *shapes)
            assert tuple(tuple(v) for v in profile.get_shape(name)) == shapes
        assert config.add_optimization_profile(profile) >= 0
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    stream = torch.cuda.current_stream().cuda_stream
    for profile_index, rows in [(0, 17), (1, 1)]:
        context = engine.create_execution_context()
        assert context is not None and context.set_optimization_profile_async(profile_index, stream)
        tx, th, expected, fused = _projection_residual_torch_oracle(torch, precision, rows, "cuda")
        assert context.set_input_shape("input", tuple(tx.shape))
        assert context.set_input_shape("residual", tuple(th.shape))
        output = torch.empty_like(th)
        for name, value in [("input", tx), ("residual", th), ("actual", output)]:
            assert context.set_tensor_address(name, value.data_ptr())
        assert context.execute_async_v3(stream)
        torch.cuda.synchronize()
        actual = output.float().cpu().numpy()
        assert np.array_equal(actual, expected), (
            f"{precision} profile{profile_index} lost the model-dtype projection rounding"
        )
        assert not np.array_equal(actual, fused), "fixture must reject one-rounding fusion"


@pytest.mark.parametrize("precision", ["bf16", "fp16"])
def test_projection_is_rounded_before_residual(gpu, precision):
    _check_projection_residual_graph(gpu, precision)


@pytest.mark.parametrize("heads", [8, 32])
def test_qk_mean64_matches_torch_for_decode_prefill_and_tail_rows(gpu, heads):
    torch, trt, model = gpu
    bridge, sampler = model._sampler_libraries()
    extension = model._register_sampler(bridge, sampler)
    model._register_attention(extension, model._attention_library(bridge))

    def define(network):
        squares = network.add_input("squares", trt.float32, (-1, heads, 64))
        _output(network, model._mean64(network, squares, bridge), "actual")

    engine = TinyEngine(
        torch,
        trt,
        define,
        {"squares": ((1, heads, 64), (17, heads, 64), (20, heads, 64))},
    )
    for rows in (1, 3, 17, 20):
        # BF16 values with varied exponents exercise FP32 accumulation rounding.
        # R3/H8 and R17/H8 leave a partial final reduction block.
        index = torch.arange(rows * heads * 64, device="cuda", dtype=torch.int64)
        mantissa = 1 + (index % 7).float() / 8
        exponent = ((index // 7) % 25 - 12).to(torch.int32)
        values = torch.ldexp(mantissa, exponent).reshape(rows, heads, 64).bfloat16()
        squares = values.float().square()
        expected = squares.mean(dim=-1, keepdim=True)
        actual = engine({"squares": squares})["actual"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [8, 32])
def test_qk_rsqrt_preserves_torch_fp32_rounding(gpu, heads):
    torch, trt, model = gpu
    bridge, sampler = model._sampler_libraries()
    extension = model._register_sampler(bridge, sampler)
    model._register_attention(extension, model._attention_library(bridge))

    def define(network):
        variance = network.add_input("variance", trt.float32, (-1, heads, 1))
        _output(network, model._rsqrt(network, variance, bridge), "actual")

    engine = TinyEngine(
        torch,
        trt,
        define,
        {"variance": ((1, heads, 1), (17, heads, 1), (20, heads, 1))},
    )
    # This positive variance exposes a BF16 rounding boundary after the FP32
    # normalization product. The oracle remains the actual Torch CUDA operator.
    witness = np.array([0x3A500D46], dtype=np.uint32).view(np.float32)[0]
    witness = np.float32(witness + np.float32(1e-5))
    distinguishes_split_operations = False
    for rows in (1, 3, 17, 20):
        index = torch.arange(rows * heads, dtype=torch.int32, device="cuda")
        mantissa = 1 + (index % 23).float() / 32
        exponent = (index // 23) % 21 - 10
        variance = torch.ldexp(mantissa, exponent).reshape(rows, heads, 1)
        variance[0, 0, 0] = float(witness)
        expected = torch.rsqrt(variance)
        split_operations = torch.reciprocal(torch.sqrt(variance))
        distinguishes_split_operations |= bool(torch.any(expected != split_operations))
        actual = engine({"variance": variance})["actual"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert distinguishes_split_operations, "fixture must distinguish the reference operator"


def test_wide_rsqrt_preserves_torch_fp32_rounding(gpu):
    torch, trt, model = gpu
    bridge, sampler = model._sampler_libraries()
    extension = model._register_sampler(bridge, sampler)
    model._register_attention(extension, model._attention_library(bridge))

    def define(network):
        variance = network.add_input("variance", trt.float32, (-1, 1))
        _output(network, model._rsqrt(network, variance, bridge), "actual")

    engine = TinyEngine(
        torch,
        trt,
        define,
        {"variance": ((1, 1), (17, 1), (257, 1))},
    )
    # This wide-normalization variance exposes the split operation's rounding.
    # R257 also exercises a partial second block with the rank-two contract.
    witness = np.array([0x3AA733D3], dtype=np.uint32).view(np.float32)[0]
    witness = np.float32(witness + np.float32(1e-5))
    distinguishes_split_operations = False
    for rows in (1, 3, 17, 20, 257):
        index = torch.arange(rows, dtype=torch.int32, device="cuda")
        mantissa = 1 + (index % 23).float() / 32
        exponent = (index // 23) % 21 - 10
        variance = torch.ldexp(mantissa, exponent).reshape(rows, 1)
        variance[0, 0] = float(witness)
        expected = torch.rsqrt(variance)
        split_operations = torch.reciprocal(torch.sqrt(variance))
        distinguishes_split_operations |= bool(torch.any(expected != split_operations))
        actual = engine({"variance": variance})["actual"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert distinguishes_split_operations, "fixture must distinguish the reference operator"


def test_decode_sum3_preserves_order_and_positive_zero(gpu):
    torch, trt, model = gpu
    bridge, sampler = model._sampler_libraries()
    extension = model._register_sampler(bridge, sampler)
    model._register_attention(extension, model._attention_library(bridge))

    def define(network):
        products = network.add_input("products", trt.float32, (1, 2048, 3))
        _output(network, model._sum3(network, products, bridge), "actual")

    engine = TinyEngine(torch, trt, define)
    # Literal exact-bit oracles distinguish the width-three addition order and
    # positive-zero identities. The first triplet lands on a BF16 midpoint;
    # cancellation (1 + 2^-24) + (-1) is zero, unlike (1 - 1) + 2^-24.
    triplets = np.array(
        [
            [0xB0A10000, 0xBAC00000, 0xBC690000],
            [0x80000000, 0x80000000, 0x80000000],
            [0x3F800000, 0xBF800000, 0x33800000],
        ],
        dtype=np.uint32,
    )
    expected_bits = np.array([0xBC808000, 0x00000000, 0x00000000], dtype=np.uint32)
    index = np.arange(2048) % len(triplets)
    products = torch.from_numpy(triplets[index].view(np.float32).reshape(1, 2048, 3)).cuda()
    actual = engine({"products": products})["actual"].cpu().numpy().view(np.uint32)
    assert np.array_equal(actual, expected_bits[index].reshape(1, 2048)), (
        "decode SUM3 must preserve the declared FP32 order and positive zero"
    )


def test_wide_mean2048_matches_torch_across_reduction_geometries(gpu):
    torch, trt, model = gpu
    bridge, sampler = model._sampler_libraries()
    extension = model._register_sampler(bridge, sampler)
    model._register_attention(extension, model._attention_library(bridge))

    def define(network):
        squares = network.add_input("squares", trt.float32, (-1, 2048))
        _output(network, model._mean2048(network, squares, bridge), "actual")

    engine = TinyEngine(
        torch,
        trt,
        define,
        {"squares": ((1, 2048), (17, 2048), (20, 2048))},
    )
    # Cover every block-width branch and incomplete final row groups with the
    # same deterministic BF16 values spanning a wide exponent range.
    for rows in (1, 2, 3, 4, 8, 16, 17, 20):
        index = torch.arange(rows * 2048, device="cuda", dtype=torch.int64)
        mantissa = 1 + (index % 7).float() / 8
        exponent = ((index // 7) % 25 - 12).to(torch.int32)
        values = torch.ldexp(mantissa, exponent).reshape(rows, 2048).bfloat16()
        squares = values.float().square()
        expected = squares.mean(dim=-1, keepdim=True)
        actual = engine({"squares": squares})["actual"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_kv_projection_matches_torch_for_decode_and_prefill(gpu):
    torch, trt, model = gpu
    bridge, sampler = model._sampler_libraries()
    extension = model._register_sampler(bridge, sampler)
    model._register_w2(extension, model._w2_library(bridge), batched_prefill=True)
    generator = torch.Generator(device="cuda").manual_seed(193)
    weight = torch.randn((512, 2048), generator=generator, device="cuda").bfloat16()
    # The checkpoint mapper supplies [input, output]; the reference Linear
    # operator stores [output, input]. Exercise that actual owning conversion.
    mapped = np.ascontiguousarray(weight.float().cpu().numpy().T)

    def define(network):
        values = network.add_input("input", trt.bfloat16, (-1, 2048))
        projected = model._add_kv_projection(
            network, values, mapped, bridge, name="test.kv_projection"
        )
        _output(network, projected, "actual")

    engine = TinyEngine(
        torch,
        trt,
        define,
        {"input": ((1, 2048), (17, 2048), (20, 2048))},
    )
    for rows in (1, 3, 17, 20):
        values = torch.randn((1, rows, 2048), generator=generator, device="cuda").bfloat16()
        expected = torch.nn.functional.linear(values, weight).reshape(rows, 512)
        actual = engine({"input": values.reshape(rows, 2048)})["actual"]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
