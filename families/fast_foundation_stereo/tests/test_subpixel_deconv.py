# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kernel-4, stride-2, padding-1 deconvolutions run as four 2x2 convolutions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _transposed_convolution(x, weight, bias):
    """NumPy reference: ``ConvTranspose2d(kernel 4, stride 2, padding 1)``, weight ``[C_in, C_out, 4, 4]``."""
    batch, _, height, width = x.shape
    out = np.zeros((batch, weight.shape[1], 2 * height, 2 * width), np.float64)
    for i in range(height):
        for j in range(width):
            for kh in range(4):
                for kw in range(4):
                    oh, ow = 2 * i - 1 + kh, 2 * j - 1 + kw
                    if 0 <= oh < 2 * height and 0 <= ow < 2 * width:
                        out[:, :, oh, ow] += x[:, :, i, j] @ weight[:, :, kh, kw]
    return out + bias[None, :, None, None]


def _module(weight, bias, **overrides):
    values = dict(
        weight=weight,
        bias=bias,
        out_channels=weight.shape[1],
        kernel_size=(4, 4),
        stride=(2, 2),
        padding=(1, 1),
        output_padding=(0, 0),
        dilation=(1, 1),
        groups=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_only_kernel4_stride2_padding1_deconvolutions_use_the_subpixel_form() -> None:
    from families.fast_foundation_stereo.native_graph import NativeGraph

    graph = NativeGraph(
        SimpleNamespace(), SimpleNamespace(float16="float16", float32="float32"), fp16=True
    )
    tensor = SimpleNamespace(shape=(2, 3, 5, 6))
    weight = np.zeros((3, 4, 4, 4), np.float32)
    assert graph._subpixel_deconvolution(_module(weight, None), 2, tensor)
    assert not graph._subpixel_deconvolution(_module(weight, None), 3, tensor)
    assert not graph._subpixel_deconvolution(_module(weight, None, stride=(1, 1)), 2, tensor)
    assert not graph._subpixel_deconvolution(_module(weight, None, groups=3), 2, tensor)
    assert not graph._subpixel_deconvolution(
        _module(weight, None, output_padding=(1, 1)), 2, tensor
    )
    assert not graph._subpixel_deconvolution(
        _module(weight, None), 2, SimpleNamespace(shape=(-1, 3, 5, 6))
    )


@pytest.mark.gpu
@pytest.mark.trt
def test_subpixel_deconvolution_matches_a_transposed_convolution() -> None:
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from families.fast_foundation_stereo.native_graph import NativeGraph

    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 5, 6)).astype(np.float32)
    weight = rng.standard_normal((3, 4, 4, 4)).astype(np.float32)
    bias = rng.standard_normal(4).astype(np.float32)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    graph = NativeGraph(network, trt, fp16=False)
    source = network.add_input("x", trt.float32, x.shape)
    output = graph.deconv2d(source, _module(weight, bias))
    output.name = "y"
    network.mark_output(output)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    engine = trt.Runtime(logger).deserialize_cuda_engine(
        builder.build_serialized_network(network, config)
    )
    assert not any(
        layer.type == trt.LayerType.DECONVOLUTION
        for layer in (network.get_layer(index) for index in range(network.num_layers))
    )

    context = engine.create_execution_context()
    device_x = torch.from_numpy(x).cuda()
    device_y = torch.empty((2, 4, 10, 12), dtype=torch.float32, device="cuda")
    context.set_tensor_address("x", device_x.data_ptr())
    context.set_tensor_address("y", device_y.data_ptr())
    assert context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    np.testing.assert_allclose(
        device_y.cpu().numpy(), _transposed_convolution(x, weight, bias), rtol=1e-4, atol=1e-4
    )
