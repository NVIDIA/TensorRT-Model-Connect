# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph helpers for TIPSv2.

Both towers are pre-norm transformers over a row tensor ``[seq, width]``, so every helper takes
and returns that layout. Three things here are TIPSv2's and not a sibling family's:

* the vision MLP uses the **error-function** GELU (``nn.GELU()`` with its default
  ``approximate="none"``), not the tanh approximation. The two differ by about 1e-3, which a
  pooled cosine will not notice and the relevance logit will;
* the text MLP uses **ReLU**, and masks its hidden *and* its output rows with the padding mask;
* the text tower's position signal is **computed**, not learned: a sinusoid that this module
  reproduces in the same float32 order the reference uses.

**Precision.** In an FP16 engine the projections and the MLPs run at half width and these do not:
the layer-norm statistics, the whole attention, the GELU, and (in ``text_builder``) the masked
average pooling. Each is a sum or a cancellation that half precision gets wrong, not a
precaution; the reasons are at the functions.

Two rules hold everywhere an infinity could appear. The attention mask is a large **finite**
negative number rather than the reference's ``-inf`` (``mask_additive_value``), so a masked score
against a zero weight is a zero and not a NaN. And anything a projection hands to a wider stage
is held inside the finite range of the dtype it came from (``clamp_to_finite``), because widening
an infinity keeps it and every path onwards from one ends in a NaN.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

# The additive mask has to push a key out of the softmax without becoming an infinity: the
# reference writes float("-inf"), and in a half-precision graph that value multiplied by a zero
# mask is a NaN. These are far below any score this network produces (the largest |score| measured
# on the checkpoint is about 1e2) and are finite in their own dtype.
_MASK_VALUE = {np.float32: -1.0e30, np.float16: -6.0e4}


def finite_range(dtype) -> float:
    """The largest finite magnitude ``dtype`` can hold."""
    return float(np.finfo(np.dtype(dtype)).max)


def clamp_to_finite(network, tensor, dtype):
    """Hold ``tensor`` inside the finite range of ``dtype``.

    Every projection in this family runs in the engine's dtype, so in FP16 a row past +-65504
    leaves it as an infinity. Widening afterwards keeps the infinity, and from there every path
    ends in a NaN: `inf * 0` wherever a mask or a zero weight meets it, `inf - inf` wherever two
    signs do, and a softmax whose largest score is an infinity.

    Clamping at the dtype's **own** largest finite value is what makes this free rather than a
    tradeoff: no value that dtype can carry is changed, so the clamp is invisible to every real
    activation and only an overflow sees it. The caller clamps after widening, so the comparison
    itself is exact.
    """
    # Two dtypes meet here and they are not the same thing. The **bound** comes from `dtype`, the
    # precision the value was produced in, because that is what decides when it overflows. The
    # **constants** are built in the tensor's own dtype, because a strongly typed network refuses
    # an elementwise operation whose operands disagree - and this helper is called both on a
    # tensor still in the engine's dtype and on one already widened to FP32.
    constant_dtype = trt.nptype(tensor.dtype)
    # The narrower of the two, which is the only bound that is both meaningful and
    # representable: a value cannot be held inside a range wider than its own container, and a
    # constant of the wider range would itself become an infinity on the way into the narrower
    # one - a clamp against infinity being no clamp at all.
    bound = min(finite_range(dtype), finite_range(constant_dtype))
    rank = len(tensor.shape)
    ones = tuple([1] * rank)
    limited = network.add_elementwise(
        tensor, add_constant(network, ones, np.full((1,), bound), dtype=constant_dtype),
        trt.ElementWiseOperation.MIN).get_output(0)
    return network.add_elementwise(
        limited, add_constant(network, ones, np.full((1,), -bound), dtype=constant_dtype),
        trt.ElementWiseOperation.MAX).get_output(0)


def mask_additive_value(dtype) -> float:
    """The finite stand-in for ``-inf`` in ``dtype``'s attention mask."""
    key = np.dtype(dtype).type
    if key not in _MASK_VALUE:
        raise NotImplementedError(f"tipsv2 has no attention mask value for {np.dtype(dtype)}")
    return _MASK_VALUE[key]


def narrowed(values, dtype, what="a constant"):
    """``values`` in ``dtype``, refused rather than silently narrowed into an infinity.

    Every weight of both towers passes through here, and an FP16 build converts each one on the
    way in. A finite FP32 weight above 65504 becomes an infinity in that conversion, and an
    infinite weight makes its matmul produce NaNs before any activation clamp can see them -
    silently, in a build that otherwise succeeds. The released so400m checkpoint's largest weight
    is 1.12, nowhere near it; this is here so that a later variant that is near it fails the
    build with its own name on it instead.

    **The check has to come before the conversion, and every weight has to reach it.** Narrowing
    first and inspecting the result afterwards cannot work: the information that the value was
    once finite is already gone. So this function owns both steps, and a caller that needs raw
    ``trt.Weights`` rather than a constant layer calls it rather than converting on its own.
    """
    source = np.asarray(values)
    with np.errstate(over="ignore"):  # the overflow is detected and reported below, not ignored
        data = np.ascontiguousarray(source.astype(dtype))
    if np.issubdtype(data.dtype, np.floating):
        lost = np.isfinite(source.astype(np.float64)) & ~np.isfinite(data)
        if lost.any():
            largest = float(np.abs(source.astype(np.float64)[lost]).max())
            raise ValueError(
                f"tipsv2 cannot build {what} at {np.dtype(dtype)}: a value of {largest} becomes "
                f"an infinity there ({int(lost.sum())} of {source.size} values)")
    return data


def add_constant(network, shape, values, dtype=np.float32, what="a constant"):
    """One constant, refused rather than silently narrowed into an infinity (see `narrowed`)."""
    data = np.ascontiguousarray(narrowed(values, dtype, what).reshape(shape))
    return network.add_constant(shape, data).get_output(0)


def add_sum(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)


def add_prod(network, left, right):
    return network.add_elementwise(left, right, trt.ElementWiseOperation.PROD).get_output(0)


def add_residual(network, stream, branch, dtype):
    """``stream + branch``, keeping the residual stream finite.

    Both towers are a stack of residual additions, and that stack is the one place where the
    clamps elsewhere in this file are not enough on their own. An FP16 overflow anywhere makes a
    branch an infinity; if the stream has already overflowed the other way, the addition is
    ``inf + -inf``, a **NaN** - and a NaN is the one value no later clamp can take back, because
    a clamp compares and a comparison against a NaN is false.

    So the invariant is maintained at every step rather than repaired at the end: the stream is
    finite before the addition, therefore the addition is finite-or-infinite but never a NaN,
    and clamping the result makes it finite again for the next block. An overflowed branch shows
    up as a saturated channel, which the next layer norm can still work with.
    """
    return clamp_to_finite(network, add_sum(network, stream, branch), dtype)


_TRT_DTYPE = {np.float32: trt.float32, np.float16: trt.float16}


def trt_dtype(dtype):
    """The TensorRT dtype this family builds in."""
    key = np.dtype(dtype).type
    if key not in _TRT_DTYPE:
        raise NotImplementedError(f"tipsv2 does not build in {np.dtype(dtype)}")
    return _TRT_DTYPE[key]


def to_dtype(network, tensor, dtype):
    """Cast a tensor into the network's working dtype, or leave it alone."""
    target = trt_dtype(dtype)
    if tensor.dtype == target:
        return tensor
    return network.add_cast(tensor, target).get_output(0)


def as_float32(network, tensor):
    """Cast to FP32 for a reduction, and remember the way back."""
    if tensor.dtype == trt.float32:
        return tensor, None
    return network.add_cast(tensor, trt.float32).get_output(0), tensor.dtype


def restore_dtype(network, tensor, original):
    if original is None:
        return tensor
    return network.add_cast(tensor, original).get_output(0)


def add_layer_norm(network, inp, gamma, beta, eps):
    """Row-wise layer norm whose mean, variance and reciprocal are taken in FP32.

    The statistics are a sum over the full width (1152 terms here); in FP16 that sum both loses
    the small channels and can overflow on a wide residual stream, so the whole statistic and the
    affine that follows run in FP32 and only the result is cast back.
    """
    source, original = as_float32(network, inp)
    narrow = trt.nptype(original if original is not None else inp.dtype)
    # The last place an infinity could still enter a wider stage: a projection whose own output
    # overflowed arrives here, the mean of its row is an infinity, and `inf - inf` is a NaN that
    # the rest of the tower then carries. With this, no FP32 stage in either tower can be handed
    # an infinity by an FP16 one.
    source = clamp_to_finite(network, source, narrow)
    axis = len(inp.shape) - 1
    mean = network.add_reduce(source, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    centred = network.add_elementwise(source, mean, trt.ElementWiseOperation.SUB).get_output(0)
    square = add_prod(network, centred, centred)
    variance = network.add_reduce(square, trt.ReduceOperation.AVG, 1 << axis, True).get_output(0)
    width = int(inp.shape[-1])
    epsilon = add_constant(network, tuple([1] * len(inp.shape)), np.full((1,), eps))
    shifted = add_sum(network, variance, epsilon)
    deviation = network.add_unary(shifted, trt.UnaryOperation.SQRT).get_output(0)
    normed = network.add_elementwise(centred, deviation,
                                     trt.ElementWiseOperation.DIV).get_output(0)
    shape = tuple([1] * (len(inp.shape) - 1) + [width])
    scaled = add_prod(network, normed, add_constant(network, shape, gamma))
    affine = add_sum(network, scaled, add_constant(network, shape, beta))
    # And the way out, for the same reason the way in is guarded. The affine is computed in FP32
    # and goes back to the engine's dtype; a finite FP32 result past 65504 becomes an infinity in
    # that narrowing, and the projection that reads it next turns that into a NaN. Clamping to
    # the **destination** range before the cast changes no value that dtype could have carried.
    return restore_dtype(network, clamp_to_finite(network, affine, narrow), original)


def add_linear(network, inp, weight, bias=None, dtype=np.float32):
    """``y = x @ W^T + b``, matching ``torch.nn.Linear``'s weight layout.

    The weight and the bias are handed over as the checkpoint stores them; `add_constant` is what
    narrows them, so its overflow check sees values that are still finite.
    """
    rank = len(inp.shape)
    weight = np.asarray(weight)
    shape = (1,) * (rank - 2) + tuple(weight.shape)
    constant = add_constant(network, shape, weight, dtype=dtype, what="a linear weight")
    out = network.add_matrix_multiply(inp, trt.MatrixOperation.NONE, constant,
                                      trt.MatrixOperation.TRANSPOSE).get_output(0)
    if bias is None:
        return out
    bias = np.asarray(bias)
    bias_shape = (1,) * (rank - 1) + (int(bias.size),)
    return add_sum(network, out,
                   add_constant(network, bias_shape, bias, dtype=dtype, what="a linear bias"))


def add_conv2d(network, inp, weight, bias=None, stride=(1, 1), dtype=np.float32):
    """The patch embedding. Its weights go in as `trt.Weights`, so they are narrowed by hand.

    Both arrays stay bound to a name until the layer is added: `trt.Weights` does not own the
    buffer it is given.
    """
    weight = narrowed(weight, dtype, "a convolution weight")
    bias = None if bias is None else narrowed(bias, dtype, "a convolution bias")
    out_channels, _, kernel_h, kernel_w = weight.shape
    layer = network.add_convolution_nd(
        inp, int(out_channels), (int(kernel_h), int(kernel_w)), trt.Weights(weight),
        trt.Weights() if bias is None else trt.Weights(bias))
    layer.stride_nd = tuple(int(v) for v in stride)
    return layer.get_output(0)


def add_gelu_exact(network, inp, dtype=np.float32):
    """``0.5 * x * (1 + erf(x / sqrt(2)))`` - the error-function GELU, spelled out, in FP32.

    The vision tower is built with ``nn.GELU()``, whose default ``approximate="none"`` is this
    function and **not** the tanh approximation other CLIP-shaped families use. The two agree to
    about 1e-3. Writing it out rather than calling TensorRT's GELU activation keeps the choice
    visible and keeps the builder from silently getting the other one.

    The arithmetic is FP32 even in an FP16 engine, because ``1 + erf(x)`` cancels: at x = -4 the
    erf is -0.99998, which rounds to exactly -1 in FP16, so the gate becomes 0 and the whole
    activation returns 0 where the answer is -1.27e-4 - a value FP16 represents perfectly well.
    The input is cast in and the result cast back, so only this one expression widens.
    """
    source, original = as_float32(network, inp)
    rank = len(source.shape)
    ones = tuple([1] * rank)

    def constant(value):
        return add_constant(network, ones, np.full((1,), value))

    # `0.5 * -inf * (1 + erf(-inf))` is `-inf * 0`, a NaN, and a surviving `+inf` becomes one in
    # the projection that follows. GELU of the clamp's bound is that bound, and of its negative a
    # signed zero, which is what every more negative input gives too.
    source = clamp_to_finite(network, source, dtype)
    scaled = add_prod(network, source, constant(1.0 / np.sqrt(2.0)))
    erf = network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
    gate = add_sum(network, constant(1.0), erf)
    activated = add_prod(network, add_prod(network, constant(0.5), source), gate)
    return restore_dtype(network, activated, original)


def add_relu(network, inp):
    return network.add_activation(inp, trt.ActivationType.RELU).get_output(0)


def add_attention(network, query, key, value, *, num_heads, head_dim, additive_mask=None,
                  dtype=np.float32):
    """Multi-head attention over rows ``[seq, width]``, **entirely in FP32**.

    The query is scaled before the product, which is where both towers' references put it
    (``qkv[0] * self.scale`` in the vision block, ``q / sqrt(E)`` inside
    ``nn.MultiheadAttention``).

    All three stages - the score product, the softmax and the context product - are FP32 even in
    an FP16 engine. The score of one head is a sum over ``head_dim`` products of two projected
    rows, with nothing bounding it: an attention without a fused kernel accumulates such a sum in
    half precision, which both overflows on large rows and drops the small terms of the softmax's
    1,026 exponentials. The projections on either side stay in the engine's dtype, so the weights
    and the activations are still half-width; only this expression is widened.

    ``additive_mask`` is ``[1, 1, 1, kv_seq]``: 0 for a key that may be attended and a large
    finite negative number for one that may not. It is added to the scores before the softmax,
    which is where the reference's ``attn_mask`` of 0 and ``-inf`` goes.
    """
    q_seq = int(query.shape[0])
    kv_seq = int(key.shape[0])
    width = num_heads * head_dim

    def to_heads(tensor, seq):
        # Widen first, then clamp: the projections run in the engine's dtype, and an FP16 one
        # that overflowed arrives here as an infinity that the cast preserves. From there the
        # score product, the softmax and the value product all end in NaNs.
        wide, _ = as_float32(network, tensor)
        layer = network.add_shuffle(clamp_to_finite(network, wide, dtype))
        layer.reshape_dims = (1, seq, num_heads, head_dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    original = query.dtype
    scaled_query = add_prod(network, to_heads(query, q_seq),
                            add_constant(network, (1, 1, 1, 1),
                                         np.full((1,), 1.0 / np.sqrt(head_dim))))
    scores = network.add_matrix_multiply(scaled_query, trt.MatrixOperation.NONE,
                                         to_heads(key, kv_seq),
                                         trt.MatrixOperation.TRANSPOSE).get_output(0)
    if additive_mask is not None:
        mask, _ = as_float32(network, additive_mask)
        scores = add_sum(network, scores, mask)
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 3
    context = network.add_matrix_multiply(softmax.get_output(0), trt.MatrixOperation.NONE,
                                          to_heads(value, kv_seq),
                                          trt.MatrixOperation.NONE).get_output(0)
    back = network.add_shuffle(context)
    back.first_transpose = (0, 2, 1, 3)
    back.reshape_dims = (q_seq, width)
    if back.get_output(0).dtype == original:
        return back.get_output(0)
    return network.add_cast(back.get_output(0), original).get_output(0)


def sinusoidal_position_signal(seq_length: int, width: int) -> np.ndarray:
    """The text tower's computed position signal, in the reference's float32 order.

    ``PositionalEmbedding`` in ``text_encoder.py``, with ``min_timescale=1`` and
    ``max_timescale=10000``: half the channels are sines and half are cosines of the same
    geometric timescales, concatenated in that order (**not** interleaved).

    The chain is reproduced operation by operation and in float32, the dtype the reference
    computes it in. It is not bit-identical to it - ``exp`` and ``sin`` differ by an ulp or two
    between libraries, measured at 3.9e-6 absolute on a signal bounded by 1, which the embedding
    it is added to (scaled by sqrt(1152) = 33.9) does not resolve. A float64 table rounded at the
    end is closer to the mathematical value (3.0e-8 against 5.2e-6) but no closer to the
    reference, so the reference's own order is what is kept.
    """
    if width % 2:
        # F.pad would add one zero channel; no TIPSv2 width is odd and the untested path is
        # refused rather than guessed.
        raise NotImplementedError(f"tipsv2 position signal needs an even width, not {width}")
    num_timescales = width // 2
    log_increment = (np.log(np.float32(10000.0) / np.float32(1.0))
                     / np.maximum(np.float32(num_timescales) - np.float32(1.0), np.float32(1.0)))
    inv_timescales = np.float32(1.0) * np.exp(
        np.arange(num_timescales, dtype=np.float32) * -log_increment)
    position = np.arange(seq_length, dtype=np.float32)
    scaled_time = position[:, None] * inv_timescales[None, :]
    return np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1).astype(np.float32)
