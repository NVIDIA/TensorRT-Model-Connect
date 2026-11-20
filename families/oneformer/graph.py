# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT network helpers owned by the OneFormer family.

The network is strongly typed and float32 throughout (the builder disables
TF32). Every spatial size is dynamic: image height and width are runtime
dimensions, so padding, window partitions, pyramid splits and positions are
computed from int32 shape tensors in the graph.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

E = trt.ElementWiseOperation


class Graph:
    """One float32 network under construction and the constants it references."""

    def __init__(self, network):
        self.network = network
        self._keep: list[np.ndarray] = []

    # Constants and shape arithmetic -------------------------------------------------

    def constant(self, values, dtype=np.float32, shape=None):
        """A constant tensor; 0-d values stay 0-d (``np.ascontiguousarray`` would make them 1-d)."""
        array = np.asarray(values, dtype=dtype)
        if shape is not None:
            array = array.reshape(shape)
        if not array.flags.c_contiguous:
            array = array.copy()
        self._keep.append(array)
        return self.network.add_constant(array.shape, trt.Weights(array)).get_output(0)

    def f32(self, values, shape=None):
        return self.constant(values, np.float32, shape)

    def i32(self, values):
        return self.constant(np.asarray(values, dtype=np.int32).reshape(-1), np.int32)

    def shape(self, tensor):
        shape = self.network.add_shape(tensor).get_output(0)
        return self.network.add_cast(shape, trt.int32).get_output(0)

    def dim(self, tensor, axis: int):
        """The size of one axis as an int32 tensor of shape (1,)."""
        return self.network.add_gather(self.shape(tensor), self.i32([axis]), 0).get_output(0)

    def _int(self, value):
        return self.i32([int(value)]) if isinstance(value, (int, np.integer)) else value

    def vec(self, parts):
        """Concatenate ints and (1,) int32 tensors into one shape vector."""
        pieces = [self._int(part) for part in parts]
        if len(pieces) == 1:
            return pieces[0]
        layer = self.network.add_concatenation(pieces)
        layer.axis = 0
        return layer.get_output(0)

    def iop(self, left, right, op):
        return self.network.add_elementwise(self._int(left), self._int(right), op).get_output(0)

    def ceil_to(self, value, multiple: int):
        """((value + multiple - 1) // multiple) * multiple on an int32 shape tensor."""
        bumped = self.iop(value, multiple - 1, E.SUM)
        return self.iop(self.iop(bumped, multiple, E.FLOOR_DIV), multiple, E.PROD)

    def as_float(self, value):
        return self.network.add_cast(value, trt.float32).get_output(0)

    # Layout ---------------------------------------------------------------------------

    def reshape(self, tensor, dims, first=None, second=None):
        layer = self.network.add_shuffle(tensor)
        if first is not None:
            layer.first_transpose = tuple(first)
        if dims is not None:
            if all(isinstance(value, (int, np.integer)) for value in dims):
                layer.reshape_dims = tuple(int(value) for value in dims)
            else:
                layer.set_input(1, self.vec(dims))
        if second is not None:
            layer.second_transpose = tuple(second)
        return layer.get_output(0)

    def transpose(self, tensor, perm):
        layer = self.network.add_shuffle(tensor)
        layer.first_transpose = tuple(perm)
        return layer.get_output(0)

    def slice(self, tensor, start, size, stride=None, mode=None):
        rank = len(tensor.shape)
        stride = stride if stride is not None else [1] * rank
        layer = self.network.add_slice(tensor, (0,) * rank, (1,) * rank, (1,) * rank)
        layer.set_input(1, self.vec(start))
        layer.set_input(2, self.vec(size))
        layer.set_input(3, self.vec(stride))
        if mode is not None:
            layer.mode = mode
        return layer.get_output(0)

    def concat(self, tensors, axis: int):
        layer = self.network.add_concatenation(list(tensors))
        layer.axis = axis
        return layer.get_output(0)

    def cast(self, tensor, dtype):
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    # Arithmetic -----------------------------------------------------------------------

    def ew(self, left, right, op):
        return self.network.add_elementwise(left, right, op).get_output(0)

    def add(self, left, right):
        return self.ew(left, right, E.SUM)

    def unary(self, tensor, op):
        return self.network.add_unary(tensor, op).get_output(0)

    def activation(self, tensor, kind):
        return self.network.add_activation(tensor, kind).get_output(0)

    def relu(self, tensor):
        return self.activation(tensor, trt.ActivationType.RELU)

    def sigmoid(self, tensor):
        return self.activation(tensor, trt.ActivationType.SIGMOID)

    def softmax(self, tensor, axis: int):
        layer = self.network.add_softmax(tensor)
        layer.axes = 1 << axis
        return layer.get_output(0)

    def matmul(self, left, right, transpose_right=False):
        op_right = trt.MatrixOperation.TRANSPOSE if transpose_right else trt.MatrixOperation.NONE
        return self.network.add_matrix_multiply(
            left, trt.MatrixOperation.NONE, right, op_right
        ).get_output(0)

    def scalar(self, tensor, value):
        """A broadcastable float32 constant of ``tensor``'s rank."""
        return self.constant([value], np.float32, (1,) * len(tensor.shape))

    # Layers ---------------------------------------------------------------------------

    def linear(self, tensor, weight, bias=None):
        """``tensor @ weight.T + bias`` with torch.nn.Linear's weight layout."""
        rank = len(tensor.shape)
        out_features, in_features = weight.shape
        lead = (1,) * (rank - 2)
        matrix = self.constant(weight, shape=lead + (out_features, in_features))
        out = self.matmul(tensor, matrix, transpose_right=True)
        if bias is None:
            return out
        return self.add(out, self.constant(bias, shape=(1,) * (rank - 1) + (out_features,)))

    def layer_norm(self, tensor, gamma, beta, eps: float):
        """torch.nn.LayerNorm over the last axis."""
        rank = len(tensor.shape)
        shape = (1,) * (rank - 1) + (int(gamma.shape[0]),)
        layer = self.network.add_normalization(
            tensor, self.f32(gamma, shape), self.f32(beta, shape), 1 << (rank - 1)
        )
        layer.epsilon = float(eps)
        return layer.get_output(0)

    def group_norm(self, tensor, gamma, beta, groups: int, eps: float):
        """torch.nn.GroupNorm on NCHW: statistics over each group's channels and pixels."""
        channels = int(tensor.shape[1])
        grouped = self.reshape(tensor, (1, groups, -1))
        mean = self.network.add_reduce(grouped, trt.ReduceOperation.AVG, 1 << 2, True).get_output(0)
        centred = self.ew(grouped, mean, E.SUB)
        variance = self.network.add_reduce(
            self.ew(centred, centred, E.PROD), trt.ReduceOperation.AVG, 1 << 2, True
        ).get_output(0)
        deviation = self.unary(
            self.add(variance, self.scalar(variance, eps)), trt.UnaryOperation.SQRT
        )
        normed = self.ew(centred, deviation, E.DIV)
        layer = self.network.add_shuffle(normed)
        layer.set_input(1, self.shape(tensor))
        restored = layer.get_output(0)
        scaled = self.ew(restored, self.f32(gamma, (1, channels, 1, 1)), E.PROD)
        return self.add(scaled, self.f32(beta, (1, channels, 1, 1)))

    def conv(self, tensor, weight, bias=None, stride=1, padding=0):
        weight = np.asarray(weight, dtype=np.float32)
        if not weight.flags.c_contiguous:
            weight = weight.copy()
        self._keep.append(weight)
        out_channels, _, kernel_h, kernel_w = weight.shape
        if bias is not None:
            bias = np.asarray(bias, dtype=np.float32).copy()
            self._keep.append(bias)
        layer = self.network.add_convolution_nd(
            tensor,
            int(out_channels),
            (int(kernel_h), int(kernel_w)),
            trt.Weights(weight),
            trt.Weights(bias) if bias is not None else trt.Weights(),
        )
        layer.stride_nd = (stride, stride)
        layer.padding_nd = (padding, padding)
        return layer.get_output(0)

    def resize_bilinear(self, tensor, shape_vector):
        """F.interpolate(mode="bilinear", align_corners=False) to an explicit output shape."""
        layer = self.network.add_resize(tensor)
        layer.set_input(1, shape_vector)
        layer.resize_mode = trt.InterpolationMode.LINEAR
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        layer.selector_for_single_pixel = trt.ResizeSelector.FORMULA
        return layer.get_output(0)

    def grid_sample(self, values, grid):
        """F.grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)."""
        layer = self.network.add_grid_sample(values, grid)
        layer.interpolation_mode = trt.InterpolationMode.LINEAR
        layer.sample_mode = trt.SampleMode.FILL
        layer.align_corners = False
        return layer.get_output(0)

    def grid(self, extents, alpha: float, delta):
        """LINSPACE fill: ``alpha + sum(delta[i] * index[i])`` over a runtime shape, float32."""
        layer = self.network.add_fill((1,) * len(delta), trt.FillOperation.LINSPACE, trt.float32)
        layer.set_input(0, self.vec(extents))
        layer.set_input(1, self.f32(np.float32(alpha)))
        layer.set_input(2, self.f32(np.asarray(delta, dtype=np.float32)))
        return layer.get_output(0)
