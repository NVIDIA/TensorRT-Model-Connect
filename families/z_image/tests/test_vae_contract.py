# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Z-Image VAE convolution precision contracts."""

from __future__ import annotations

import numpy as np
import pytest
import tensorrt as trt

from families.z_image import graph_ops


class _Tensor:
    def __init__(self, dtype, shape: tuple = ()):
        self.dtype = dtype
        self.shape = shape


class _Layer:
    def __init__(self, output: _Tensor):
        self._output = output
        self.inputs: dict[int, _Tensor] = {}

    def get_output(self, _index: int = 0) -> _Tensor:
        return self._output

    def set_input(self, index: int, tensor: _Tensor) -> None:
        self.inputs[index] = tensor


class _Network:
    def __init__(self):
        self.convolutions: list[tuple[_Layer, dict]] = []
        self.constant_shapes: list[tuple] = []

    def add_convolution_nd(self, inp, **kwargs) -> _Layer:
        layer = _Layer(_Tensor(inp.dtype))
        self.convolutions.append((layer, kwargs))
        return layer

    def add_constant(self, shape, _weights) -> _Layer:
        self.constant_shapes.append(tuple(shape))
        return _Layer(_Tensor(trt.float32, tuple(shape)))

    def add_cast(self, tensor, dtype) -> _Layer:
        return _Layer(_Tensor(dtype, tensor.shape))


@pytest.mark.parametrize("dtype", [trt.float16, trt.bfloat16])
def test_conv2d_weights_follow_the_input_dtype(dtype) -> None:
    network = _Network()
    weight = np.ones((4, 2, 3, 3), dtype=np.float32)
    output = graph_ops.add_conv2d(
        network,
        _Tensor(dtype),
        weight,
        np.zeros((4,), dtype=np.float32),
        out_channels=4,
        kernel_size=(3, 3),
        padding=(1, 1),
        dtype=np.float16,
    )

    ((conv, kwargs),) = network.convolutions
    assert output.dtype == dtype
    assert conv.padding_nd == (1, 1)
    if dtype == trt.float16:
        assert kwargs["kernel"].dtype == trt.float16
        assert conv.inputs == {}
    else:
        # numpy has no bf16: the kernel and the bias arrive as bf16 tensors cast from fp32 constants.
        assert kwargs["kernel"].size == 0 and kwargs["bias"].size == 0
        assert network.constant_shapes == [(4, 2, 3, 3), (4,)]
        assert {index: tensor.dtype for index, tensor in conv.inputs.items()} == {
            1: trt.bfloat16,
            2: trt.bfloat16,
        }
