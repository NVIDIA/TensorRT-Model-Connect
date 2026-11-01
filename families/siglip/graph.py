# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for SigLIP.

Both towers are pre-norm transformers over a row tensor ``[seq, width]``, so
every helper takes and returns that layout. Two helpers differ from the CLIP
family's otherwise identical set, and both differences are load-bearing:

* the activation is the **tanh approximation** of GELU, not QuickGELU and not
  the exact GELU;
* ``add_attention`` lets the query carry a different row count from the keys,
  because the vision tower is pooled by attending a single learned query row
  against the whole patch grid.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt


def add_constant(network, shape, values, dtype=np.float32):
    data = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)


def _as_float32(network, tensor):
    """Cast to fp32 for a reduction's statistics, and remember the way back."""
    if tensor.dtype == trt.float32:
        return tensor, None
    return network.add_cast(tensor, trt.float32).get_output(0), tensor.dtype


def _restore_dtype(network, tensor, original):
    if original is None:
        return tensor
    return network.add_cast(tensor, original).get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps):
    source, original = _as_float32(network, inp)
    axis = len(inp.shape) - 1
    mean = network.add_reduce(source, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    centred = network.add_elementwise(source, mean, trt.ElementWiseOperation.SUB).get_output(0)
    square = network.add_elementwise(centred, centred, trt.ElementWiseOperation.PROD).get_output(0)
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    width = int(inp.shape[-1])
    epsilon = add_constant(network, tuple([1] * len(inp.shape)), np.full((1,), eps))
    shifted = network.add_elementwise(variance, epsilon, trt.ElementWiseOperation.SUM).get_output(0)
    deviation = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    normed = network.add_elementwise(centred, deviation,
                                     trt.ElementWiseOperation.DIV).get_output(0)
    shape = tuple([1] * (len(inp.shape) - 1) + [width])
    scaled = network.add_elementwise(normed, add_constant(network, shape, gamma),
                                     trt.ElementWiseOperation.PROD).get_output(0)
    affine = network.add_elementwise(scaled, add_constant(network, shape, beta),
                                     trt.ElementWiseOperation.SUM).get_output(0)
    return _restore_dtype(network, affine, original)


def add_linear(network, inp, weight, bias=None, dtype=np.float32, *, matmul=None):
    """y = x @ W^T + b, matching torch.nn.Linear's weight layout.

    ``matmul`` (``(network, input) -> output``) replaces the matrix product, which is how
    a quantized projection takes its place; the bias is added the same way either way.
    """
    rank = len(inp.shape)
    if matmul is not None:
        out = matmul(network, inp)
    else:
        weight = np.ascontiguousarray(np.asarray(weight, dtype=dtype))
        shape = (1,) * (rank - 2) + tuple(weight.shape)
        constant = add_constant(network, shape, weight.reshape(shape), dtype=dtype)
        out = network.add_matrix_multiply(
            inp, trt.MatrixOperation.NONE, constant,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
    if bias is None:
        return out
    bias_shape = (1,) * (rank - 1) + (-1,)
    bias = np.asarray(bias, dtype=dtype).reshape(bias_shape)
    return network.add_elementwise(out, add_constant(network, bias.shape, bias, dtype=dtype),
                                   trt.ElementWiseOperation.SUM).get_output(0)


def add_conv2d(network, inp, weight, bias=None, stride=(1, 1), dtype=np.float32):
    weight = np.ascontiguousarray(np.asarray(weight, dtype=dtype))
    out_channels, _, kernel_h, kernel_w = weight.shape
    layer = network.add_convolution_nd(
        inp, int(out_channels), (int(kernel_h), int(kernel_w)), trt.Weights(weight),
        trt.Weights(np.ascontiguousarray(np.asarray(bias, dtype=dtype)))
        if bias is not None else trt.Weights())
    layer.stride_nd = tuple(int(v) for v in stride)
    return layer.get_output(0)


def add_gelu_tanh(network, inp, dtype=np.float32):
    """``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))``.

    Both towers set ``hidden_act: gelu_pytorch_tanh``. This is the tanh
    approximation, which is a different function from the exact GELU that the
    error-function form gives: they agree to about 1e-3, which a pooled cosine
    will not notice and the logits will. TensorRT's GELU activation is spelled
    out here rather than trusted to pick the same one.
    """
    rank = len(inp.shape)
    ones = tuple([1] * rank)

    def constant(value):
        return add_constant(network, ones, np.full((1,), value), dtype=dtype)

    def prod(left, right):
        return network.add_elementwise(left, right, trt.ElementWiseOperation.PROD).get_output(0)

    cube = prod(prod(inp, inp), inp)
    inner = network.add_elementwise(inp, prod(constant(0.044715), cube),
                                    trt.ElementWiseOperation.SUM).get_output(0)
    scaled = prod(constant(np.sqrt(2.0 / np.pi)), inner)
    tanh = network.add_activation(scaled, trt.ActivationType.TANH).get_output(0)
    gate = network.add_elementwise(constant(1.0), tanh,
                                   trt.ElementWiseOperation.SUM).get_output(0)
    return prod(prod(constant(0.5), inp), gate)


def add_attention(network, query, key, value, *, num_heads, head_dim):
    """Multi-head attention over rows, with the query free to be shorter.

    ``query`` is ``[q_seq, width]`` and ``key``/``value`` are ``[kv_seq, width]``.
    Self-attention passes the same length three times; the pooling head passes
    a single query row against the whole patch grid. Neither tower is masked -
    SigLIP's text encoder is bidirectional, unlike CLIP's - so there is no mask
    argument to forget to pass.
    """
    q_seq = int(query.shape[0])
    kv_seq = int(key.shape[0])
    width = num_heads * head_dim

    def to_heads(tensor, seq):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (1, seq, num_heads, head_dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    scores = network.add_matrix_multiply(
        to_heads(query, q_seq), trt.MatrixOperation.NONE,
        to_heads(key, kv_seq), trt.MatrixOperation.TRANSPOSE).get_output(0)
    scale = add_constant(network, (1, 1, 1, 1), np.full((1,), 1.0 / np.sqrt(head_dim)),
                         dtype=np.dtype(trt.nptype(scores.dtype)))
    scores = network.add_elementwise(scores, scale, trt.ElementWiseOperation.PROD).get_output(0)
    weights = network.add_softmax(scores)
    weights.axes = 1 << 3
    context = network.add_matrix_multiply(
        weights.get_output(0), trt.MatrixOperation.NONE,
        to_heads(value, kv_seq), trt.MatrixOperation.NONE).get_output(0)
    back = network.add_shuffle(context)
    back.first_transpose = (0, 2, 1, 3)
    back.reshape_dims = (q_seq, width)
    return back.get_output(0)


def add_l2_normalise(network, inp):
    """Row-wise L2 normalisation, taken in fp32 for the same reason as the norms."""
    source, original = _as_float32(network, inp)
    axis = len(inp.shape) - 1
    square = network.add_elementwise(source, source, trt.ElementWiseOperation.PROD).get_output(0)
    total = network.add_reduce(square, trt.ReduceOperation.SUM, 1 << axis, True).get_output(0)
    norm = network.add_unary(total, trt.UnaryOperation.SQRT).get_output(0)
    divided = network.add_elementwise(source, norm, trt.ElementWiseOperation.DIV).get_output(0)
    return _restore_dtype(network, divided, original)
