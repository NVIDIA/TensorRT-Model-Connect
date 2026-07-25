# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned TensorRT graph operations for Python engine builds.

Tensor names and shapes must stay compatible with the C++ bundle runtime.
"""

from __future__ import annotations

import numpy as np
from tensorrt_model_connect import trt_compat


trt = trt_compat.get_trt()


def _cast_back_to_trt_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Cast a tensor back to the original TRT runtime dtype after FP32 compute."""
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


def add_constant(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    values: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add a constant tensor in the given *dtype* (default float32)."""
    weights = trt.Weights(np.ascontiguousarray(values, dtype=dtype))
    layer = network.add_constant(shape, weights)
    return layer.get_output(0)


def add_matmul_rhs_constant(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    lhs_width: int,
    rhs_width: int,
    rhs_weights: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Matrix multiply: lhs @ rhs_constant.  rhs is [lhs_width, rhs_width]."""
    rank = len(tuple(lhs.shape))
    rhs_shape = (lhs_width, rhs_width) if rank <= 2 else (1,) * (rank - 2) + (lhs_width, rhs_width)
    rhs = add_constant(
        network,
        rhs_shape,
        np.asarray(rhs_weights).reshape(rhs_shape),
        dtype=dtype,
    )
    rhs = _cast_back_to_trt_dtype(network, rhs, lhs.dtype)
    mm = network.add_matrix_multiply(
        lhs,
        trt.MatrixOperation.NONE,
        rhs,
        trt.MatrixOperation.NONE,
    )
    return _cast_back_to_trt_dtype(network, mm.get_output(0), lhs.dtype)


def add_bias_sum(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    width: int,
    bias: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Element-wise add a bias broadcast over all non-feature axes."""
    rank = len(tuple(inp.shape))
    bias_shape = (width,) if rank <= 1 else (1,) * (rank - 1) + (width,)
    bias_t = add_constant(network, bias_shape, np.asarray(bias).reshape(bias_shape), dtype=dtype)
    bias_t = _cast_back_to_trt_dtype(network, bias_t, inp.dtype)
    s = network.add_elementwise(inp, bias_t, trt.ElementWiseOperation.SUM)
    return _cast_back_to_trt_dtype(network, s.get_output(0), inp.dtype)


def add_rms_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """RMSNorm: gamma * (x / sqrt(mean(x^2) + eps)).

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.

    TRT's native normalization API implements mean-centered LayerNorm, not
    RMSNorm, so this remains a manual shared implementation.
    """
    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom_in = network.add_elementwise(mean.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(inp, recip.get_output(0), trt.ElementWiseOperation.PROD)
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    result = scaled.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """LayerNorm: gamma * ((x - mean) / sqrt(var + eps)) + beta.

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        eps_tensor = network.add_cast(eps_tensor, trt.float32).get_output(0)
    # mean = reduce_mean(x)
    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # x - mean
    centered = network.add_elementwise(inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    # variance = mean((x - mean)^2)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0), trt.ElementWiseOperation.PROD
    )
    var = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    # sqrt(var + eps)
    denom_in = network.add_elementwise(var.get_output(0), eps_tensor, trt.ElementWiseOperation.SUM)
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    # normalized = (x - mean) / sqrt(var + eps)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD
    )
    # gamma * normalized + beta
    gamma_t = add_constant(network, (1, hidden_size), gamma, dtype=np.float32)
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    beta_t = add_constant(network, (1, hidden_size), beta, dtype=np.float32)
    result = network.add_elementwise(scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM)
    result = result.get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_gelu_new(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GELU (tanh approximation): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).

    Constants are cast to ``inp.dtype`` so the elementwise ops are valid in
    a STRONGLY_TYPED network when ``inp`` is bf16 (storage np_dtype is
    fp16, runtime trt_dtype is bfloat16) or any other non-matching combo.
    """
    target_dtype = inp.dtype
    const_shape = (1,) * max(1, len(tuple(inp.shape)))

    def _const(name, value):
        c = add_constant(network, const_shape, np.array([value], dtype=np.float32), dtype=dtype)
        return _cast_back_to_trt_dtype(network, c, target_dtype)

    # x^3
    x_sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    x_cu = network.add_elementwise(x_sq.get_output(0), inp, trt.ElementWiseOperation.PROD)
    # 0.044715 * x^3
    coeff = _const("coeff", 0.044715)
    scaled_cube = network.add_elementwise(x_cu.get_output(0), coeff, trt.ElementWiseOperation.PROD)
    # x + 0.044715 * x^3
    inner_sum = network.add_elementwise(
        inp, scaled_cube.get_output(0), trt.ElementWiseOperation.SUM
    )
    # sqrt(2/pi) * (x + 0.044715 * x^3)
    sqrt_2_over_pi = _const("sqrt_2_over_pi", np.sqrt(2.0 / np.pi))
    tanh_arg = network.add_elementwise(
        sqrt_2_over_pi, inner_sum.get_output(0), trt.ElementWiseOperation.PROD
    )
    # tanh(...)
    tanh_l = network.add_activation(tanh_arg.get_output(0), trt.ActivationType.TANH)
    # 1 + tanh(...)
    one = _const("one", 1.0)
    one_plus_tanh = network.add_elementwise(one, tanh_l.get_output(0), trt.ElementWiseOperation.SUM)
    # 0.5 * x
    half = _const("half", 0.5)
    half_x = network.add_elementwise(half, inp, trt.ElementWiseOperation.PROD)
    # 0.5 * x * (1 + tanh(...))
    result = network.add_elementwise(
        half_x.get_output(0), one_plus_tanh.get_output(0), trt.ElementWiseOperation.PROD
    )
    return result.get_output(0)


# ---------------------------------------------------------------------------
# Diffusion graph ops — used by DiT, T5, VAE builders
# ---------------------------------------------------------------------------


def add_group_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_channels: int,
    num_groups: int,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GroupNorm: split channels into groups, normalize each group.

    Input: [..., num_channels] (last dim is channels).
    Output: same shape.

    TRT 10 does not have a native GroupNorm layer, so we reshape to
    [batch, num_groups, group_size], normalize, reshape back, then apply
    affine (gamma, beta).

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    # Reshape: [B, C] or [B, C, ...] — we handle the 2D case [seq, C]
    # and the 5D case [B, C, T, H, W] for VAE.
    need_cast = dtype != np.float32
    ndims = len(inp.shape)
    group_size = num_channels // num_groups
    output_dtype = inp.dtype

    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)

    if ndims == 2:
        # [seq, C] -> [seq, G, Gs]
        reshape_in = network.add_shuffle(inp)
        reshape_in.reshape_dims = (-1, num_groups, group_size)
        x = reshape_in.get_output(0)

        # Normalize over group_size dim (dim=2)
        eps_t = add_constant(
            network, (1, 1, 1), np.array([eps], dtype=np.float32), dtype=np.float32
        )
        sq = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD)
        mean = network.add_reduce(x, trt.ReduceOperation.AVG, 1 << 2, keep_dims=True)
        mean_sq = network.add_reduce(
            sq.get_output(0), trt.ReduceOperation.AVG, 1 << 2, keep_dims=True
        )
        var = network.add_elementwise(
            mean_sq.get_output(0),
            network.add_elementwise(
                mean.get_output(0), mean.get_output(0), trt.ElementWiseOperation.PROD
            ).get_output(0),
            trt.ElementWiseOperation.SUB,
        )
        denom = network.add_unary(
            network.add_elementwise(
                var.get_output(0), eps_t, trt.ElementWiseOperation.SUM
            ).get_output(0),
            trt.UnaryOperation.SQRT,
        )
        recip = network.add_unary(denom.get_output(0), trt.UnaryOperation.RECIP)
        centered = network.add_elementwise(x, mean.get_output(0), trt.ElementWiseOperation.SUB)
        normalized = network.add_elementwise(
            centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD
        )

        # Reshape back to [seq, C]
        reshape_out = network.add_shuffle(normalized.get_output(0))
        reshape_out.reshape_dims = (-1, num_channels)
        result = reshape_out.get_output(0)

    elif ndims == 5:
        # [B, C, T, H, W] — use TRT INormalizationLayer (GroupNorm mode)
        # Reshape to [B, G, Gs, T, H, W], norm over dims 2,3,4,5, reshape back
        # But simpler: use the fact that TRT GroupNorm can work on NCHW-like tensors.
        # We treat [B, C, T, H, W] directly, normalizing over (Gs, T, H, W) per group.
        b, c, t, h, w = inp.shape
        reshape_in = network.add_shuffle(inp)
        reshape_in.reshape_dims = (b, num_groups, group_size, t, h, w)
        x = reshape_in.get_output(0)

        # Reduce over dims 2,3,4,5 (group_size, T, H, W)
        reduce_axes = (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
        eps_t = add_constant(
            network, (1, 1, 1, 1, 1, 1), np.array([eps], dtype=np.float32), dtype=np.float32
        )
        sq = network.add_elementwise(x, x, trt.ElementWiseOperation.PROD)
        mean = network.add_reduce(x, trt.ReduceOperation.AVG, reduce_axes, keep_dims=True)
        mean_sq = network.add_reduce(
            sq.get_output(0), trt.ReduceOperation.AVG, reduce_axes, keep_dims=True
        )
        var = network.add_elementwise(
            mean_sq.get_output(0),
            network.add_elementwise(
                mean.get_output(0), mean.get_output(0), trt.ElementWiseOperation.PROD
            ).get_output(0),
            trt.ElementWiseOperation.SUB,
        )
        denom = network.add_unary(
            network.add_elementwise(
                var.get_output(0), eps_t, trt.ElementWiseOperation.SUM
            ).get_output(0),
            trt.UnaryOperation.SQRT,
        )
        recip = network.add_unary(denom.get_output(0), trt.UnaryOperation.RECIP)
        centered = network.add_elementwise(x, mean.get_output(0), trt.ElementWiseOperation.SUB)
        normalized = network.add_elementwise(
            centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD
        )

        # Reshape back to [B, C, T, H, W]
        reshape_out = network.add_shuffle(normalized.get_output(0))
        reshape_out.reshape_dims = (b, c, t, h, w)
        result = reshape_out.get_output(0)
    else:
        raise ValueError(f"add_group_norm: unsupported ndims={ndims}")

    # Affine: gamma * result + beta (broadcast over spatial dims)
    if ndims == 2:
        gamma_t = add_constant(network, (1, num_channels), gamma, dtype=np.float32)
        beta_t = add_constant(network, (1, num_channels), beta, dtype=np.float32)
    else:
        gamma_t = add_constant(
            network, (1, num_channels, 1, 1, 1), gamma.reshape(1, -1, 1, 1, 1), dtype=np.float32
        )
        beta_t = add_constant(
            network, (1, num_channels, 1, 1, 1), beta.reshape(1, -1, 1, 1, 1), dtype=np.float32
        )
    scaled = network.add_elementwise(result, gamma_t, trt.ElementWiseOperation.PROD)
    result = network.add_elementwise(
        scaled.get_output(0), beta_t, trt.ElementWiseOperation.SUM
    ).get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_silu(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
) -> trt.ITensor:
    """SiLU (Swish): x * sigmoid(x)."""
    sigmoid = network.add_activation(inp, trt.ActivationType.SIGMOID)
    return network.add_elementwise(
        inp, sigmoid.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)


def add_conv3d_as_conv2d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """3D convolution decomposed as 2D convolution over fused (T*C) channels.

    Input: [B, C_in, T, H, W]
    Weight: [C_out, C_in, Kt, Kh, Kw]
    Output: [B, C_out, T_out, H_out, W_out]

    For temporal kernel Kt=1, this is a standard spatial conv applied to each frame.
    For Kt>1, we reshape [B, C_in, T, H, W] -> [B, C_in*Kt, T_out, H, W] using
    a sliding-window gather, then apply Conv2D with [C_out, C_in*Kt, Kh, Kw].
    """
    b, c_in, t, h, w = inp.shape
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    pt, ph, pw = padding

    if kt == 1 and st == 1 and pt == 0:
        # Simple case: per-frame spatial conv
        # Reshape [B, C, T, H, W] -> [B*T, C, H, W]
        reshape_in = network.add_shuffle(inp)
        reshape_in.first_transpose = trt.Permutation([0, 2, 1, 3, 4])
        reshape_in.reshape_dims = (b * t, c_in, h, w)

        # Weight: [C_out, C_in, 1, Kh, Kw] -> [C_out, C_in, Kh, Kw]
        w2d = weight.reshape(out_channels, c_in, kh, kw)
        conv_w = trt.Weights(np.ascontiguousarray(w2d, dtype=dtype))
        conv_b = trt.Weights()
        if bias is not None:
            conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

        conv = network.add_convolution_nd(
            reshape_in.get_output(0),
            num_output_maps=out_channels,
            kernel_shape=(kh, kw),
            kernel=conv_w,
            bias=conv_b,
        )
        conv.stride_nd = (sh, sw)
        conv.padding_nd = (ph, pw)

        # Reshape back [B*T, C_out, H', W'] -> [B, C_out, T, H', W']
        h_out = (h + 2 * ph - kh) // sh + 1
        w_out = (w + 2 * pw - kw) // sw + 1
        reshape_out = network.add_shuffle(conv.get_output(0))
        reshape_out.reshape_dims = (b, t, out_channels, h_out, w_out)
        reshape_out.second_transpose = trt.Permutation([0, 2, 1, 3, 4])
        return reshape_out.get_output(0)
    else:
        # General case: temporal kernel > 1
        # Pad temporally if needed
        if pt > 0:
            # Zero-pad [B, C, T, H, W] -> [B, C, T+2*pt, H, W]
            pad_layer = network.add_padding_nd(
                inp,
                pre_padding=(0, pt, 0),
                post_padding=(0, pt, 0),
            )
            inp = pad_layer.get_output(0)

        # For causal conv we handle this via the cache mechanism externally,
        # so here we just do a per-frame conv with gathered temporal neighbors.
        # Reshape [B, C, T_padded, H, W] -> sliding window gather -> Conv2D
        # This is complex in pure TRT graph, so for now we use the simple
        # kernel=1 path and handle temporal via caching externally.
        raise NotImplementedError(
            f"Conv3D with kt={kt} not yet implemented in TRT graph. "
            "Use causal caching with kt=1 per-frame convolutions instead."
        )


def add_causal_conv3d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    cache: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
    padding_hw: tuple[int, int] = (0, 0),
    dtype: np.dtype = np.float32,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Causal 3D convolution with temporal cache.

    Input: [B, C_in, T, H, W] (T >= 1)
    Cache: [B, C_in, Kt-1, H, W] (previous frames)
    Weight: [C_out, C_in, Kt, Kh, Kw]

    Returns: (output [B, C_out, T, H', W'], updated_cache [B, C_in, Kt-1, H, W])

    The cache stores Kt-1 previous frames. We concatenate cache + input along
    temporal dim, then apply convolution. For T=1 uses optimized 2D decomposition,
    for T>1 uses native 3D convolution.
    """
    b, c_in, t_in, h, w = inp.shape
    kt, kh, kw = kernel_size
    ph, pw = padding_hw

    if kt == 1:
        # No temporal dependency, just spatial conv
        result = add_conv3d_as_conv2d(
            network,
            inp,
            weight,
            bias,
            out_channels,
            kernel_size=(1, kh, kw),
            stride=stride,
            padding=(0, ph, pw),
            dtype=dtype,
        )
        # Cache is unchanged
        return result, cache

    # Concatenate cache + input along temporal dim:
    # [B, C, Kt-1, H, W] cat [B, C, T, H, W] -> [B, C, Kt-1+T, H, W]
    concat = network.add_concatenation([cache, inp])
    concat.axis = 2  # temporal dim
    full_temporal = concat.get_output(0)

    if t_in == 1:
        # Optimized T=1 path: reshape to 2D and use Conv2D
        # full_temporal is [B, C_in, Kt, H, W]
        reshape_in = network.add_shuffle(full_temporal)
        reshape_in.reshape_dims = (b, c_in * kt, h, w)

        w2d = weight.reshape(out_channels, c_in * kt, kh, kw)
        conv_w = trt.Weights(np.ascontiguousarray(w2d, dtype=dtype))
        conv_b = trt.Weights()
        if bias is not None:
            conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

        conv = network.add_convolution_nd(
            reshape_in.get_output(0),
            num_output_maps=out_channels,
            kernel_shape=(kh, kw),
            kernel=conv_w,
            bias=conv_b,
        )
        conv.stride_nd = (stride[1], stride[2])
        conv.padding_nd = (ph, pw)

        h_out = (h + 2 * ph - kh) // stride[1] + 1
        w_out = (w + 2 * pw - kw) // stride[2] + 1
        reshape_out = network.add_shuffle(conv.get_output(0))
        reshape_out.reshape_dims = (b, out_channels, 1, h_out, w_out)
        result = reshape_out.get_output(0)
    else:
        # General T>1 path: native 3D convolution
        # full_temporal is [B, C_in, Kt-1+T, H, W]
        w3d = weight.reshape(out_channels, c_in, kt, kh, kw)
        conv_w = trt.Weights(np.ascontiguousarray(w3d, dtype=dtype))
        conv_b = trt.Weights()
        if bias is not None:
            conv_b = trt.Weights(np.ascontiguousarray(bias, dtype=dtype))

        conv = network.add_convolution_nd(
            full_temporal,
            num_output_maps=out_channels,
            kernel_shape=(kt, kh, kw),
            kernel=conv_w,
            bias=conv_b,
        )
        conv.stride_nd = (stride[0], stride[1], stride[2])
        conv.padding_nd = (0, ph, pw)  # No temporal padding (cache provides it)
        result = conv.get_output(0)  # [B, C_out, T, H', W']

    # Update cache: last Kt-1 frames from the concatenated tensor
    total_t = (kt - 1) + t_in
    cache_start_t = total_t - (kt - 1)  # = t_in
    if kt > 1:
        slice_layer = network.add_slice(
            full_temporal,
            start=(0, 0, cache_start_t, 0, 0),
            shape=(b, c_in, kt - 1, h, w),
            stride=(1, 1, 1, 1, 1),
        )
        new_cache = slice_layer.get_output(0)
    else:
        new_cache = cache

    return result, new_cache


def add_spatial_upsample(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    scale_factor: int = 2,
) -> trt.ITensor:
    """Spatial nearest-neighbor upsampling for 5D tensor [B, C, T, H, W].

    Output: [B, C, T, H*scale, W*scale]
    """
    b, c, t, h, w = inp.shape
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.NEAREST
    resize.shape = (b, c, t, h * scale_factor, w * scale_factor)
    return resize.get_output(0)


def add_spatial_upsample_with_conv(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weight: np.ndarray,
    bias: np.ndarray | None,
    scale: int = 2,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Spatial nearest-neighbor 2x upsample + Conv3D(1,3,3) smoothing.

    Matches HF WanResample's nn.Sequential(Upsample(2x), Conv3d(1,3,3)).

    Input: [B, C_in, T, H, W]
    Weight: [C_out, C_in, 1, 3, 3]  (C_out detected from weight shape)
    Output: [B, C_out, T, H*scale, W*scale]
    """
    out_channels = weight.shape[0]

    # Step 1: nearest-neighbor 2x spatial
    upsampled = add_spatial_upsample(network, inp, scale)

    # Step 2: Conv3D(1,3,3) = per-frame 2D conv with 3x3 kernel
    result = add_conv3d_as_conv2d(
        network,
        upsampled,
        weight=weight,
        bias=bias,
        out_channels=out_channels,
        kernel_size=(1, 3, 3),
        padding=(0, 1, 1),
        dtype=dtype,
    )
    return result


def add_l2_channel_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_channels: int,
    gamma: np.ndarray,
    eps: float = 1e-6,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """L2 channel norm: F.normalize(x, dim=1) * sqrt(C) * gamma.

    L2-normalizes over channel dimension (axis=1), then scales by
    sqrt(num_channels) and learnable gamma.

    Input: [B, C, T, H, W] (5D tensor)
    gamma: [C, 1, 1, 1] reshaped to [1, C, 1, 1, 1] for broadcast
    Output: same shape

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
    # L2 norm over channel dim: ||x||_2 = sqrt(sum(x^2, dim=1))
    sq = network.add_elementwise(inp, inp, trt.ElementWiseOperation.PROD)
    sum_sq = network.add_reduce(sq.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)

    eps_t = add_constant(
        network, (1, 1, 1, 1, 1), np.array([eps], dtype=np.float32), dtype=np.float32
    )
    denom_in = network.add_elementwise(sum_sq.get_output(0), eps_t, trt.ElementWiseOperation.SUM)
    norm = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(norm.get_output(0), trt.UnaryOperation.RECIP)

    # normalized = x / ||x||_2
    normalized = network.add_elementwise(inp, recip.get_output(0), trt.ElementWiseOperation.PROD)

    # Scale by sqrt(C) * gamma  →  gamma_scaled shape [1, C, 1, 1, 1]
    gamma_flat = gamma.flatten()[:num_channels]
    scale = np.sqrt(num_channels) * gamma_flat
    scale_t = add_constant(
        network,
        (1, num_channels, 1, 1, 1),
        scale.reshape(1, num_channels, 1, 1, 1),
        dtype=np.float32,
    )

    result = network.add_elementwise(
        normalized.get_output(0), scale_t, trt.ElementWiseOperation.PROD
    ).get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def add_temporal_pixel_shuffle(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    factor: int = 2,
) -> trt.ITensor:
    """Temporal pixel shuffle: [B, factor*C, T, H, W] → [B, C, factor*T, H, W].

    Interleaves temporal frames by splitting the channel dimension and
    folding it into the temporal dimension.
    """
    b, c_total, t, h, w = inp.shape
    c = c_total // factor  # output channels

    # Reshape: [B, factor*C, T, H, W] → [B, factor, C, T, H, W]
    reshape1 = network.add_shuffle(inp)
    reshape1.reshape_dims = (b, factor, c, t, h, w)

    # Permute: [B, factor, C, T, H, W] → [B, C, T, factor, H, W]
    transpose = network.add_shuffle(reshape1.get_output(0))
    transpose.first_transpose = trt.Permutation([0, 2, 3, 1, 4, 5])

    # Reshape: [B, C, T, factor, H, W] → [B, C, factor*T, H, W]
    reshape2 = network.add_shuffle(transpose.get_output(0))
    reshape2.reshape_dims = (b, c, factor * t, h, w)

    return reshape2.get_output(0)


def add_adaptive_layernorm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    scale: trt.ITensor,
    shift: trt.ITensor,
    hidden_size: int,
    eps: float = 1e-5,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Adaptive LayerNorm (AdaLN): norm(x) * (1 + scale) + shift.

    Used by DiT blocks. The scale and shift come from the timestep MLP.

    Input: [seq, hidden_size]
    scale: [1, hidden_size]
    shift: [1, hidden_size]
    Output: [seq, hidden_size]

    FP32 precision boundary: when dtype != float32, casts to FP32 before
    norm computation for numerical stability, then casts back.
    """
    need_cast = dtype != np.float32
    output_dtype = inp.dtype
    if need_cast:
        inp = network.add_cast(inp, trt.float32).get_output(0)
        scale = network.add_cast(scale, trt.float32).get_output(0)
        shift = network.add_cast(shift, trt.float32).get_output(0)
    # Standard LayerNorm without affine
    eps_t = add_constant(network, (1, 1), np.array([eps], dtype=np.float32), dtype=np.float32)
    mean = network.add_reduce(inp, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    centered = network.add_elementwise(inp, mean.get_output(0), trt.ElementWiseOperation.SUB)
    sq = network.add_elementwise(
        centered.get_output(0), centered.get_output(0), trt.ElementWiseOperation.PROD
    )
    var = network.add_reduce(sq.get_output(0), trt.ReduceOperation.AVG, 1 << 1, keep_dims=True)
    denom = network.add_unary(
        network.add_elementwise(var.get_output(0), eps_t, trt.ElementWiseOperation.SUM).get_output(
            0
        ),
        trt.UnaryOperation.SQRT,
    )
    recip = network.add_unary(denom.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        centered.get_output(0), recip.get_output(0), trt.ElementWiseOperation.PROD
    )

    # Adaptive modulation: norm(x) * (1 + scale) + shift
    one = add_constant(network, (1, 1), np.array([1.0], dtype=np.float32), dtype=np.float32)
    scale_plus_one = network.add_elementwise(one, scale, trt.ElementWiseOperation.SUM)
    scaled = network.add_elementwise(
        normalized.get_output(0), scale_plus_one.get_output(0), trt.ElementWiseOperation.PROD
    )
    result = network.add_elementwise(
        scaled.get_output(0), shift, trt.ElementWiseOperation.SUM
    ).get_output(0)
    if need_cast:
        result = _cast_back_to_trt_dtype(network, result, output_dtype)
    return result


def make_t5_relative_position_bias(
    num_heads: int,
    max_seq_len: int,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> np.ndarray:
    """Compute T5-style relative position bias table.

    Returns: [num_heads, max_seq_len, max_seq_len] float32 bias table.
    This is baked as a constant into the TRT graph.
    """

    def _relative_position_bucket(
        relative_position: np.ndarray,
        bidirectional: bool = True,
        num_bkts: int = 32,
        max_dist: int = 128,
    ) -> np.ndarray:
        """Map relative position to bucket index (T5 algorithm)."""
        ret = np.zeros_like(relative_position, dtype=np.int32)
        n = -relative_position
        if bidirectional:
            num_bkts //= 2
            ret += (n < 0).astype(np.int32) * num_bkts
            n = np.abs(n)
        else:
            n = np.maximum(n, 0)

        max_exact = num_bkts // 2
        is_small = n < max_exact

        # Clamp to avoid log(0)
        n_clamped = np.maximum(n.astype(np.float32), 1)
        val_if_large = max_exact + (
            np.log(n_clamped / max_exact) / np.log(max_dist / max_exact) * (num_bkts - max_exact)
        ).astype(np.int32)
        val_if_large = np.minimum(val_if_large, num_bkts - 1)

        ret += np.where(is_small, n, val_if_large)
        return ret

    # Build relative position matrix
    context_position = np.arange(max_seq_len, dtype=np.int32)[:, None]
    memory_position = np.arange(max_seq_len, dtype=np.int32)[None, :]
    relative_position = memory_position - context_position

    buckets = _relative_position_bucket(
        relative_position,
        bidirectional=True,
        num_bkts=num_buckets,
        max_dist=max_distance,
    )

    return buckets.astype(np.int32)


# Alias: add_gelu_tanh is the same as add_gelu_new (tanh approximation)
add_gelu_tanh = add_gelu_new


def validate_native_rope_dim(
    rotary_embedding_dim: int,
    *,
    field_name: str = "rotary_embedding_dim",
) -> int:
    """Validate the dimension contract required by TRT native RoPE."""
    rotary_embedding_dim = int(rotary_embedding_dim)
    if rotary_embedding_dim < 2 or rotary_embedding_dim % 2 != 0:
        raise ValueError(
            f"TRT native RoPE requires {field_name} to be an even value >= 2; "
            f"got {rotary_embedding_dim}"
        )
    return rotary_embedding_dim


def reshape_rows_to_heads_4d(
    network: trt.INetworkDefinition,
    x: trt.ITensor,
    num_heads: int,
    head_dim: int,
    sequence_length: int | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape [S, H * D] rows into [1, H, S, D].

    The transpose is required for S > 1 because each input row contains all
    heads for one token. ``sequence_length=None`` means runtime-dynamic S.
    """
    seq_dim = -1 if sequence_length is None else sequence_length
    r1 = network.add_shuffle(x)
    if tag:
        r1.name = tag + "_s_h_d"
    r1.reshape_dims = (seq_dim, num_heads, head_dim)
    r1.second_transpose = trt.Permutation([1, 0, 2])

    r2 = network.add_shuffle(r1.get_output(0))
    if tag:
        r2.name = tag + "_1_h_s_d"
    r2.reshape_dims = (1, num_heads, seq_dim, head_dim)
    return r2.get_output(0)


def reshape_heads_4d_to_rows(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    attention_size: int,
    sequence_length: int | None = None,
    tag: str | None = None,
) -> trt.ITensor:
    """Reshape [1, H, S, D] back to [S, H * D]."""
    seq_dim = -1 if sequence_length is None else sequence_length
    out = network.add_shuffle(x_4d)
    if tag:
        out.name = tag + "_s_h_d"
    out.first_transpose = trt.Permutation([0, 2, 1, 3])
    out.reshape_dims = (seq_dim, attention_size)
    return out.get_output(0)


def add_apply_rope_native_sequence(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_3d: trt.ITensor,
    sin_cache_3d: trt.ITensor,
    rotary_embedding_dim: int,
    interleaved: bool = False,
    sequence_length: int | None = None,
) -> trt.ITensor:
    """Apply native RoPE with per-position caches [1, Sq, rotary_dim / 2]."""
    rotary_embedding_dim = validate_native_rope_dim(rotary_embedding_dim)
    attention_size = num_heads * head_dim
    inp_4d = reshape_rows_to_heads_4d(network, inp, num_heads, head_dim, sequence_length)
    rope = network.add_rotary_embedding(
        inp_4d,
        cos_cache_3d,
        sin_cache_3d,
        interleaved,
        rotary_embedding_dim,
    )
    return reshape_heads_4d_to_rows(network, rope.get_output(0), attention_size, sequence_length)


def add_apply_rope_native_from_full_cache(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    num_heads: int,
    head_dim: int,
    cos_cache_full: trt.ITensor,
    sin_cache_full: trt.ITensor,
    sequence_length: int,
    interleaved: bool = False,
    rotary_embedding_dim: int | None = None,
) -> trt.ITensor:
    """Apply native RoPE from per-head full-dim cos/sin caches [Sq, D]."""
    rope_dim = head_dim if rotary_embedding_dim is None else rotary_embedding_dim
    rope_dim = validate_native_rope_dim(rope_dim)
    half = rope_dim // 2
    stride = 2 if interleaved else 1
    cos_half = network.add_slice(
        cos_cache_full, start=(0, 0), shape=(sequence_length, half), stride=(1, stride)
    )
    sin_half = network.add_slice(
        sin_cache_full, start=(0, 0), shape=(sequence_length, half), stride=(1, stride)
    )
    cos_3d = network.add_shuffle(cos_half.get_output(0))
    cos_3d.reshape_dims = (1, sequence_length, half)
    sin_3d = network.add_shuffle(sin_half.get_output(0))
    sin_3d.reshape_dims = (1, sequence_length, half)
    return add_apply_rope_native_sequence(
        network,
        inp,
        num_heads,
        head_dim,
        cos_3d.get_output(0),
        sin_3d.get_output(0),
        rotary_embedding_dim=rope_dim,
        interleaved=interleaved,
        sequence_length=sequence_length,
    )


def add_attention_core(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    causal: bool = False,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
    fp32_accumulation: bool = False,
) -> trt.ITensor:
    """Scaled dot-product attention via TRT native IAttention layer.

    Replaces the manual Q@K^T → scale → softmax → @V chain.  TRT 10 fuses
    this into a single kernel when a compatible implementation is available;
    decomposable=True ensures a correct fallback to primitives otherwise.

    NOTE: TRT IAttention computes raw BMM1 = Q @ K^T without any built-in
    1/sqrt(D) scaling.  We pre-scale Q by 1/sqrt(D) so that the fused kernel
    computes the standard scaled dot-product attention formula.

    Args:
        q_4d:    Query  [B, H, q_seq, D].
        k_4d:    Key    [B, H, kv_seq, D].
        v_4d:    Value  [B, H, kv_seq, D].
        causal:  Apply causal (autoregressive) mask.  Mutually exclusive
                 with ``mask``.
        mask:    Optional additive float mask [B, H, q_seq, kv_seq] added
                 to scaled logits before softmax.  Cannot be used with
                 causal=True.
        scale:   Optional Q pre-scale factor.  Defaults to 1/sqrt(D).
        fp32_accumulation:
                 Cast Q/K/V to FP32 before IAttention, then cast the context
                 back to the original Q dtype.  TRT may still select a
                 Half-input fused MHA tactic after optimizing the casts, while
                 keeping the IAttention accumulation/output boundary in FP32.

    Returns:
        Context tensor [B, H, q_seq, D].
    """
    output_dtype = q_4d.dtype
    if fp32_accumulation and output_dtype != trt.float32:
        q_4d = network.add_cast(q_4d, trt.float32).get_output(0)
        k_4d = network.add_cast(k_4d, trt.float32).get_output(0)
        v_4d = network.add_cast(v_4d, trt.float32).get_output(0)
        if mask is not None and mask.dtype != trt.float32:
            mask = network.add_cast(mask, trt.float32).get_output(0)

    # Pre-scale Q: TRT IAttention does not apply score scaling itself.
    # Match the scale constant's dtype to Q's dtype: in strongly-typed networks
    # a FP32 constant mixed with a FP16/BF16 Q causes add_elementwise to emit
    # a type-mismatch error and produce a tensor with corrupted dimensions,
    # which makes add_attention return None.
    if scale is None:
        head_dim = q_4d.shape[-1]
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    # Use FP16 weights directly for FP16; BF16 has no numpy native type so
    # create as FP32 and cast; FP32 falls through to the default.
    scale_np_dtype = np.float16 if q_4d.dtype == trt.float16 else np.float32
    scale_t = add_constant(network, (1, 1, 1, 1), np.array([[[[scale]]]]), dtype=scale_np_dtype)
    if q_4d.dtype == trt.bfloat16:
        scale_t = network.add_cast(scale_t, trt.bfloat16).get_output(0)
    q_scaled = network.add_elementwise(q_4d, scale_t, trt.ElementWiseOperation.PROD)

    attn = network.add_attention(
        q_scaled.get_output(0),
        k_4d,
        v_4d,
        trt.AttentionNormalizationOp.SOFTMAX,
        causal,
    )
    # Allow TRT to decompose into primitive ops when no fused kernel is
    # available (e.g. unsupported head-dim or dtype).  This guarantees
    # correctness on any configuration at the cost of potential performance.
    attn.decomposable = True
    if mask is not None and not causal:
        attn.mask = mask
    return _cast_back_to_trt_dtype(network, attn.get_output(0), output_dtype)


def _scalar_constant_for_trt_dtype(
    network: trt.INetworkDefinition,
    shape: tuple[int, ...],
    value: float,
    dtype: trt.DataType,
) -> trt.ITensor:
    np_dtype = np.float16 if dtype == trt.float16 else np.float32
    const = add_constant(network, shape, np.full(shape, value, dtype=np_dtype), dtype=np_dtype)
    if dtype == trt.bfloat16:
        const = network.add_cast(const, trt.bfloat16).get_output(0)
    return const


def add_tanh_softcap(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    cap: float,
    *,
    scalar_shape: tuple[int, ...],
) -> trt.ITensor:
    """Apply ``tanh(tensor / cap) * cap`` using scalar broadcasting."""
    cap_t = _scalar_constant_for_trt_dtype(network, scalar_shape, float(cap), tensor.dtype)
    scaled = network.add_elementwise(tensor, cap_t, trt.ElementWiseOperation.DIV).get_output(0)
    capped = network.add_activation(scaled, trt.ActivationType.TANH).get_output(0)
    return network.add_elementwise(capped, cap_t, trt.ElementWiseOperation.PROD).get_output(0)


def _repeat_kv_heads_4d(
    network: trt.INetworkDefinition,
    x_4d: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> trt.ITensor:
    if num_kv_heads == num_heads:
        return x_4d
    if num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")

    repeat = num_heads // num_kv_heads
    if num_kv_heads == 1:
        concat = network.add_concatenation([x_4d] * repeat)
        concat.axis = 1
        return concat.get_output(0)

    x_shape = network.add_shape(x_4d).get_output(0)
    one = add_constant(network, (1,), np.array([1], dtype=np.int64), dtype=np.int64)
    seq = network.add_slice(x_shape, start=(2,), shape=(1,), stride=(1,))
    dim = add_constant(network, (1,), np.array([head_dim], dtype=np.int64), dtype=np.int64)
    slice_shape = network.add_concatenation([one, one, seq.get_output(0), dim])
    slice_shape.axis = 0

    repeated = []
    for head_idx in range(num_kv_heads):
        head_slice = network.add_slice(
            x_4d, start=(0, head_idx, 0, 0), shape=(1, 1, 1, head_dim), stride=(1, 1, 1, 1)
        )
        head_slice.set_input(2, slice_shape.get_output(0))
        repeated.extend([head_slice.get_output(0)] * repeat)

    concat = network.add_concatenation(repeated)
    concat.axis = 1
    return concat.get_output(0)


def _add_attention_core_with_logit_softcap(
    network: trt.INetworkDefinition,
    q_4d: trt.ITensor,
    k_4d: trt.ITensor,
    v_4d: trt.ITensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mask: trt.ITensor | None,
    scale: float,
    logit_softcap: float,
) -> trt.ITensor:
    output_dtype = q_4d.dtype
    k_4d = _repeat_kv_heads_4d(
        network, k_4d, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    v_4d = _repeat_kv_heads_4d(
        network, v_4d, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim
    )

    score_q = q_4d
    score_k = k_4d
    score_mask = mask
    if output_dtype != trt.float32:
        score_q = network.add_cast(score_q, trt.float32).get_output(0)
        score_k = network.add_cast(score_k, trt.float32).get_output(0)
        if score_mask is not None and score_mask.dtype != trt.float32:
            score_mask = network.add_cast(score_mask, trt.float32).get_output(0)

    scale_t = _scalar_constant_for_trt_dtype(network, (1, 1, 1, 1), scale, score_q.dtype)
    scores = network.add_matrix_multiply(
        score_q, trt.MatrixOperation.NONE, score_k, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scores = network.add_elementwise(scores, scale_t, trt.ElementWiseOperation.PROD).get_output(0)

    scores = add_tanh_softcap(network, scores, logit_softcap, scalar_shape=(1, 1, 1, 1))

    if score_mask is not None:
        scores = network.add_elementwise(
            scores, score_mask, trt.ElementWiseOperation.SUM
        ).get_output(0)

    probs = network.add_softmax(scores)
    probs.axes = 1 << 3
    probs_t = probs.get_output(0)
    if probs_t.dtype != output_dtype:
        probs_t = network.add_cast(probs_t, output_dtype).get_output(0)

    context = network.add_matrix_multiply(
        probs_t, trt.MatrixOperation.NONE, v_4d, trt.MatrixOperation.NONE
    ).get_output(0)
    return _cast_back_to_trt_dtype(network, context, output_dtype)


def add_attention_from_rows(
    network: trt.INetworkDefinition,
    q: trt.ITensor,
    k: trt.ITensor,
    v: trt.ITensor,
    *,
    num_heads: int,
    head_dim: int,
    num_kv_heads: int | None = None,
    q_seq: int | None,
    kv_seq: int | None,
    causal: bool = False,
    mask: trt.ITensor | None = None,
    scale: float | None = None,
    logit_softcap: float | None = None,
    fp32_accumulation: bool = False,
    tag: str | None = None,
) -> trt.ITensor:
    """Native IAttention for row-major [S, H * D] Q/K/V tensors.

    ``num_kv_heads`` can be smaller than ``num_heads`` for GQA/MQA. TRT
    native IAttention supports this directly, so callers should not expand K/V
    heads unless the model semantics require per-query-head K/V values.
    """
    attention_size = num_heads * head_dim
    kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    q_4d = reshape_rows_to_heads_4d(
        network,
        q,
        num_heads,
        head_dim,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".q",
    )
    k_4d = reshape_rows_to_heads_4d(
        network,
        k,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".k",
    )
    v_4d = reshape_rows_to_heads_4d(
        network,
        v,
        kv_heads,
        head_dim,
        sequence_length=kv_seq,
        tag=None if tag is None else tag + ".v",
    )
    if scale is None:
        scale = float(1.0 / np.sqrt(head_dim)) if head_dim > 0 else 1.0
    if logit_softcap is not None and float(logit_softcap) > 0.0:
        if causal:
            raise NotImplementedError("logit_softcap attention requires an explicit additive mask")
        ctx_4d = _add_attention_core_with_logit_softcap(
            network,
            q_4d,
            k_4d,
            v_4d,
            num_heads=num_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            mask=mask,
            scale=scale,
            logit_softcap=float(logit_softcap),
        )
    else:
        ctx_4d = add_attention_core(
            network,
            q_4d,
            k_4d,
            v_4d,
            causal=causal,
            mask=mask,
            scale=scale,
            fp32_accumulation=fp32_accumulation,
        )
    return reshape_heads_4d_to_rows(
        network,
        ctx_4d,
        attention_size,
        sequence_length=q_seq,
        tag=None if tag is None else tag + ".ctx",
    )


# Backward-compatible name used by existing tests and call sites.
_add_attention_core = add_attention_core
