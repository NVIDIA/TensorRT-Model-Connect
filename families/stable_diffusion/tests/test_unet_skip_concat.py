# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The UNet's skip concatenation runs on every device, Jetson Thor (sm_110) included."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from families.stable_diffusion import unet_builder  # noqa: E402

CHANNELS, SIDE = 64, 8


def _skip_network():
    """``concat(conv_b(skip), skip)`` with ``skip = conv_a(x)``: the shape of every UNet skip connection."""
    rng = np.random.default_rng(0)
    builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    x = network.add_input("x", trt.float16, (1, CHANNELS, SIDE, SIDE))
    weights = []

    def conv(tensor):
        weight = (rng.standard_normal((CHANNELS, CHANNELS, 3, 3)) * 0.05).astype(np.float16)
        bias = (rng.standard_normal(CHANNELS) * 0.05).astype(np.float16)
        weights.append((weight, bias))
        layer = network.add_convolution_nd(tensor, CHANNELS, (3, 3), trt.Weights(weight), trt.Weights(bias))
        layer.padding_nd = (1, 1)
        return layer.get_output(0)

    skip = conv(x)
    y = unet_builder._skip_concat(network, conv(skip), skip)
    y.name = "y"
    network.mark_output(y)
    return builder, network, weights


def test_skip_concat_is_fp32_and_returns_the_work_dtype():
    _, network, _ = _skip_network()
    concats = [network.get_layer(i) for i in range(network.num_layers)
               if network.get_layer(i).type == trt.LayerType.CONCATENATION]
    assert len(concats) == 1
    concat = concats[0]
    assert [concat.get_input(i).dtype for i in range(concat.num_inputs)] == [trt.float32, trt.float32]
    assert network.get_output(0).dtype == trt.float16


def test_skip_concat_engine_runs_and_matches_torch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    builder, network, weights = _skip_network()
    config = builder.create_builder_config()
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(plan)
    context = engine.create_execution_context()
    assert context is not None  # TensorRT 11.4 on sm_110 fails here for a plain FP16 concatenation (#2190)

    x = torch.randn((1, CHANNELS, SIDE, SIDE), device="cuda", dtype=torch.float16)
    y = torch.empty((1, 2 * CHANNELS, SIDE, SIDE), device="cuda", dtype=torch.float16)
    context.set_tensor_address("x", x.data_ptr())
    context.set_tensor_address("y", y.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    (weight_a, bias_a), (weight_b, bias_b) = [
        (torch.from_numpy(w).float().cuda(), torch.from_numpy(b).float().cuda()) for w, b in weights]
    skip = torch.nn.functional.conv2d(x.float(), weight_a, bias_a, padding=1)
    expected = torch.cat([torch.nn.functional.conv2d(skip, weight_b, bias_b, padding=1), skip], dim=1)
    torch.testing.assert_close(y.float(), expected, rtol=2e-2, atol=2e-2)
