"""Qwen-Image VAE decoder builder.

Builds a TensorRT engine for the ``AutoencoderKLQwenImage`` decoder running
in image (T=1) mode. The Qwen-Image VAE is a 3D causal-conv VAE forked from
Wan2.1 -- same architecture, same weight naming, different default
``temperal_downsample`` and ``latents_mean`` / ``latents_std`` vectors.

Architectural model (single-frame decode path; see
``references/diffusers/.../autoencoder_kl_qwenimage.py``):

  Input: latent z of shape [B, z_dim, T=1, H_lat, W_lat] (z_dim=16).
  Output: pixels of shape [B, 3, T=1, H_lat*8, W_lat*8] in [-1, 1].

  Forward (HF ``_decode`` with num_frame=1):
    z -> post_quant_conv (1x1x1 Conv3d, z_dim -> z_dim)
       -> decoder(z_single_frame, fresh feat_cache)
            conv_in: CausalConv3d(z_dim -> base_dim*dim_mult[-1], k=(3,3,3))
            mid_block: ResBlock -> SpatialAttention -> ResBlock
            up_blocks (reversed dim_mult, iterate level 0..N-1):
              N+1 ResBlocks (channel change at block 0, in_ch -> out_ch)
              if level < N-1:
                if temperal_upsample[level]:  # reversed [F,T,T] => [T,T,F]
                  -- time_conv path: in image mode (fresh cache marked "Rep")
                     the HF code SKIPS the time_conv call entirely; it only
                     marks the cache slot. Spatial upsample is still applied.
                spatial upsample: nearest 2x + Conv3d(1,3,3) smoothing
            norm_out (L2 channel norm with .gamma) -> SiLU
            conv_out: CausalConv3d(prev_ch -> 3, k=(3,3,3))
       -> clamp(-1, 1)

  Per-channel ``latents_mean`` and ``latents_std`` (length=z_dim float
  vectors) un-normalise the latent BEFORE the VAE: HF's pipeline does
  ``z = z / std + mean`` before calling ``vae.decode``. Equivalently the
  graph can absorb this as a pre-multiplicative scale (`std`) + per-channel
  bias (`mean`). We expose two builder modes:

    * ``apply_latent_unnorm=False`` (default): the engine consumes a latent
      already un-normalised by the caller (matches HF's ``vae.decode(z_raw)``
      contract). Synthetic test uses this for exact parity.
    * ``apply_latent_unnorm=True``: the engine receives a normalised latent
      ``z_norm = (z_raw - mean) / std`` and the graph applies
      ``z_raw = z_norm * std + mean`` before ``post_quant_conv``. Convenient
      when wiring a full T2I pipeline whose denoiser emits ``z_norm``.

  Causal-conv handling: for image (T=1) mode HF's ``QwenImageCausalConv3d``
  with no prior cache reduces to a standard Conv3d with zero left-padding
  in time. We implement this by feeding zero-valued cache tensors of shape
  [B, C, Kt-1=2, H, W] as TRT constants into the existing
  ``graph_ops.add_causal_conv3d`` helper -- no temporal expansion since
  ``num_frame=1`` and HF skips time_conv on the fresh-cache "Rep" path.

  Approach (per spec): Option A -- image-mode-only 3D graph with T=1 held
  constant throughout. The graph emits exactly the same numbers as HF's
  ``vae.decode(z)`` with ``return_dict=True`` and reads ``.sample``.

Precision: the network is STRONGLY_TYPED with fp32 inputs/outputs but
runs the heavy decoder compute internally in bf16. The latent input is
cast to bf16 immediately after un-normalisation (which itself uses fp32
constants for accuracy). Conv/upsample/attention/residual weights are
materialised as bf16 buffers (anchored in a module-level list to survive
GC during engine build). L2 channel norms cast back to fp32 internally for
the variance reduction, then cast back to bf16 at the boundary. The final
clamp/output is cast back to fp32 right before ``mark_output``.

This mirrors HF's ``AutoencoderKLQwenImage.to(dtype=bfloat16)`` behaviour:
the VAE accepts an fp32 latent which the HF pipeline immediately
``.to(self.vae.dtype)``s, runs the whole module in bf16, and the C++/Python
runtime sees fp32 buffers at the bound IO tensors.

Trace: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01, UT-QWEN-IMAGE-VAE-001.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

from tensorrt_model_connect import trt_compat

from ... import graph_ops


trt = trt_compat.get_trt()


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class QwenImageVAEConfig:
    """Architecture parameters for the Qwen-Image VAE decoder.

    Real Qwen-Image values (matching ``Qwen/Qwen-Image-2512/vae/config.json``):
        z_dim=16, base_dim=96, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
        temperal_downsample=[False, True, True].

    Note: HF uses the misspelled key ``temperal_downsample`` in its config;
    we mirror that spelling for the raw-JSON loader but expose the corrected
    field name ``temporal_downsample`` here. The constructor accepts either
    via the loader.
    """

    z_dim: int = 16
    base_dim: int = 96
    dim_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    # HF stores ``temperal_downsample`` (the typo). For image decoding the
    # decoder iterates over the *reverse* of this list (the "upsample"
    # schedule) and -- in image mode -- skips the time_conv even where the
    # schedule is True, so the value only governs which weights need to be
    # present in the checkpoint.
    temporal_downsample: list[bool] = field(
        default_factory=lambda: [False, True, True]
    )
    input_channels: int = 3
    attn_scales: list[float] = field(default_factory=list)
    latents_mean: list[float] = field(
        default_factory=lambda: [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517,
            1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497,
            0.2503, -0.2921,
        ]
    )
    latents_std: list[float] = field(
        default_factory=lambda: [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ]
    )
    # Numerical eps for L2 channel norm. HF default for F.normalize is 1e-12.
    norm_eps: float = 1e-12

    @property
    def spatial_compression_ratio(self) -> int:
        """Total spatial downscale = 2 ** len(dim_mult) (matches HF prop)."""
        return 2 ** len(self.temporal_downsample)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Module-level anchor list keeping bf16 numpy buffers alive while TRT
# consumes them via raw pointers in ``trt.Weights(trt.bfloat16, ptr, size)``.
# Cleared at the top of every ``build_qwen_image_vae_decoder_engine`` call.
_BF16_WEIGHT_REFS: list[np.ndarray] = []


def _as_numpy(value, *, name: str) -> np.ndarray:
    """Coerce torch/numpy to a contiguous float32 numpy array."""
    if isinstance(value, np.ndarray):
        arr = value
    else:
        try:
            import torch
        except ImportError:  # pragma: no cover
            raise TypeError(
                f"weight {name!r} is not numpy and torch is unavailable"
            )
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().to(torch.float32).numpy()
        else:
            raise TypeError(
                f"weight {name!r} has unsupported type {type(value).__name__}"
            )
    return np.ascontiguousarray(arr, dtype=np.float32)


def _to_bf16(network: "trt.INetworkDefinition", tensor: "trt.ITensor") -> "trt.ITensor":
    """Cast a tensor to bf16 (no-op if already bf16)."""
    if tensor.dtype == trt.bfloat16:
        return tensor
    return network.add_cast(tensor, trt.bfloat16).get_output(0)


def _to_fp32(network: "trt.INetworkDefinition", tensor: "trt.ITensor") -> "trt.ITensor":
    """Cast a tensor to fp32 (no-op if already fp32)."""
    if tensor.dtype == trt.float32:
        return tensor
    return network.add_cast(tensor, trt.float32).get_output(0)


def _bf16_weights(data_fp32: np.ndarray) -> "trt.Weights":
    """Wrap a fp32 numpy array as a bf16 ``trt.Weights``.

    Anchors the bf16 buffer in ``_BF16_WEIGHT_REFS`` to keep it alive while
    TRT consumes it via raw pointer.
    """
    import ml_dtypes
    bf16_arr = np.ascontiguousarray(
        np.asarray(data_fp32, dtype=np.float32).astype(ml_dtypes.bfloat16)
    )
    _BF16_WEIGHT_REFS.append(bf16_arr)
    return trt.Weights(trt.bfloat16, bf16_arr.ctypes.data, bf16_arr.size)


def _bf16_conv2d(
    network: "trt.INetworkDefinition",
    inp: "trt.ITensor",
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_hw: tuple[int, int],
    stride_hw: tuple[int, int],
    padding_hw: tuple[int, int],
) -> "trt.ITensor":
    """2D conv with bf16 weights/bias on a bf16 input.

    ``weight`` shape is the original 4D conv kernel [C_out, C_in, Kh, Kw].
    """
    conv = network.add_convolution_nd(
        inp,
        num_output_maps=out_channels,
        kernel_shape=kernel_hw,
        kernel=_bf16_weights(weight),
        bias=_bf16_weights(bias) if bias is not None else trt.Weights(),
    )
    conv.stride_nd = stride_hw
    conv.padding_nd = padding_hw
    return conv.get_output(0)


def _bf16_conv3d_as_conv2d(
    network: "trt.INetworkDefinition",
    inp: "trt.ITensor",
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int] = (0, 0, 0),
) -> "trt.ITensor":
    """3D conv (kt=1 path) decomposed to 2D, with bf16 weights.

    Input: [B, C_in, T, H, W] bf16.
    Output: [B, C_out, T, H_out, W_out] bf16.
    Mirrors ``graph_ops.add_conv3d_as_conv2d`` for the ``kt=1`` branch
    only -- the kt>1 branch is handled by ``_bf16_causal_conv3d``.
    """
    b, c_in, t, h, w = inp.shape
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    pt, ph, pw = padding
    if kt != 1 or st != 1 or pt != 0:
        raise NotImplementedError(
            "_bf16_conv3d_as_conv2d only supports kt=1 spatial-only convs"
        )

    # Reshape [B, C, T, H, W] -> [B*T, C, H, W]
    reshape_in = network.add_shuffle(inp)
    reshape_in.first_transpose = trt.Permutation([0, 2, 1, 3, 4])
    reshape_in.reshape_dims = (b * t, c_in, h, w)

    # Weight: [C_out, C_in, 1, Kh, Kw] -> [C_out, C_in, Kh, Kw]
    w2d = weight.reshape(out_channels, c_in, kh, kw)
    conv_out = _bf16_conv2d(
        network, reshape_in.get_output(0),
        w2d, bias,
        out_channels=out_channels,
        kernel_hw=(kh, kw),
        stride_hw=(sh, sw),
        padding_hw=(ph, pw),
    )

    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1
    reshape_out = network.add_shuffle(conv_out)
    reshape_out.reshape_dims = (b, t, out_channels, h_out, w_out)
    reshape_out.second_transpose = trt.Permutation([0, 2, 1, 3, 4])
    return reshape_out.get_output(0)


def _bf16_causal_conv3d(
    network: "trt.INetworkDefinition",
    inp: "trt.ITensor",
    cache: "trt.ITensor",
    weight: np.ndarray,
    bias: np.ndarray | None,
    out_channels: int,
    kernel_size: tuple[int, int, int],
    padding_hw: tuple[int, int] = (0, 0),
) -> "trt.ITensor":
    """Image-mode (T=1) causal Conv3d with bf16 weights.

    Mirrors ``graph_ops.add_causal_conv3d`` for the ``t_in==1, kt>1`` path:
    concatenate the [B, C, Kt-1, H, W] cache with the [B, C, 1, H, W]
    input and run a single fused 2D conv whose [C_in*Kt, Kh, Kw] kernel
    multiplies the cached+current frames together. For ``kt=1`` we fall
    through to ``_bf16_conv3d_as_conv2d``.
    """
    b, c_in, t_in, h, w = inp.shape
    kt, kh, kw = kernel_size
    ph, pw = padding_hw

    if kt == 1:
        return _bf16_conv3d_as_conv2d(
            network, inp, weight, bias, out_channels,
            kernel_size=(1, kh, kw),
            padding=(0, ph, pw),
        )
    if t_in != 1:
        raise NotImplementedError(
            "_bf16_causal_conv3d only supports T=1 image-mode causal conv"
        )

    # Concatenate cache + input along temporal dim: [B, C, Kt, H, W]
    concat = network.add_concatenation([cache, inp])
    concat.axis = 2
    full_temporal = concat.get_output(0)

    # Reshape to 2D for a fused conv: [B, C_in*Kt, H, W]
    reshape_in = network.add_shuffle(full_temporal)
    reshape_in.reshape_dims = (b, c_in * kt, h, w)

    w2d = weight.reshape(out_channels, c_in * kt, kh, kw)
    conv_out = _bf16_conv2d(
        network, reshape_in.get_output(0),
        w2d, bias,
        out_channels=out_channels,
        kernel_hw=(kh, kw),
        stride_hw=(1, 1),
        padding_hw=(ph, pw),
    )

    h_out = (h + 2 * ph - kh) + 1
    w_out = (w + 2 * pw - kw) + 1
    reshape_out = network.add_shuffle(conv_out)
    reshape_out.reshape_dims = (b, out_channels, 1, h_out, w_out)
    return reshape_out.get_output(0)


def _bf16_spatial_upsample_with_conv(
    network: "trt.INetworkDefinition",
    inp: "trt.ITensor",
    weight: np.ndarray,
    bias: np.ndarray | None,
    scale: int = 2,
) -> "trt.ITensor":
    """Nearest-neighbour 2x spatial upsample + 1x3x3 Conv3d (bf16 weights)."""
    b, c, t, h, w = inp.shape
    resize = network.add_resize(inp)
    resize.resize_mode = trt.InterpolationMode.NEAREST
    resize.shape = (b, c, t, h * scale, w * scale)
    upsampled = resize.get_output(0)

    out_channels = weight.shape[0]
    return _bf16_conv3d_as_conv2d(
        network, upsampled,
        weight=weight, bias=bias,
        out_channels=out_channels,
        kernel_size=(1, 3, 3),
        padding=(0, 1, 1),
    )


def _bf16_spatial_downsample_with_conv(
    network: "trt.INetworkDefinition",
    inp: "trt.ITensor",
    weight: np.ndarray,
    bias: np.ndarray | None,
) -> "trt.ITensor":
    """ZeroPad2d((0,1,0,1)) + stride-2 Conv2d over each frame.

    Mirrors QwenImageResample(mode="downsample2d"/"downsample3d") for the
    first image frame. For downsample3d, HF caches the first frame and skips
    the temporal convolution, so image-mode encoding only needs this spatial
    branch.
    """
    b, c, t, h, w = inp.shape

    # Pad right and bottom by 1: [B,C,T,H,W] -> [B*T,C,H+1,W+1].
    reshape_in = network.add_shuffle(inp)
    reshape_in.first_transpose = trt.Permutation([0, 2, 1, 3, 4])
    reshape_in.reshape_dims = (b * t, c, h, w)

    pad = network.add_padding_nd(
        reshape_in.get_output(0),
        pre_padding=(0, 0),
        post_padding=(1, 1),
    )

    conv_out = _bf16_conv2d(
        network,
        pad.get_output(0),
        weight,
        bias,
        out_channels=weight.shape[0],
        kernel_hw=(3, 3),
        stride_hw=(2, 2),
        padding_hw=(0, 0),
    )

    h_out = (h + 1 - 3) // 2 + 1
    w_out = (w + 1 - 3) // 2 + 1
    reshape_out = network.add_shuffle(conv_out)
    reshape_out.reshape_dims = (b, t, weight.shape[0], h_out, w_out)
    reshape_out.second_transpose = trt.Permutation([0, 2, 1, 3, 4])
    return reshape_out.get_output(0)


def _bf16_l2_channel_norm(
    network: "trt.INetworkDefinition",
    inp: "trt.ITensor",
    num_channels: int,
    gamma: np.ndarray,
    eps: float,
) -> "trt.ITensor":
    """L2 channel norm with bf16 IO; reduction performed in fp32.

    Implements ``F.normalize(x, dim=1) * sqrt(C) * gamma``. We cast to fp32
    around the variance reduction (the only numerically-sensitive part) and
    cast the result back to bf16 so downstream ops stay in compute dtype.
    """
    inp_fp32 = _to_fp32(network, inp)

    sq = network.add_elementwise(
        inp_fp32, inp_fp32, trt.ElementWiseOperation.PROD).get_output(0)
    sum_sq = network.add_reduce(
        sq, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True).get_output(0)
    eps_t = graph_ops.add_constant(
        network, (1, 1, 1, 1, 1),
        np.array([eps], dtype=np.float32), dtype=np.float32)
    denom = network.add_elementwise(
        sum_sq, eps_t, trt.ElementWiseOperation.SUM).get_output(0)
    norm = network.add_unary(denom, trt.UnaryOperation.SQRT).get_output(0)
    recip = network.add_unary(norm, trt.UnaryOperation.RECIP).get_output(0)
    normalized = network.add_elementwise(
        inp_fp32, recip, trt.ElementWiseOperation.PROD).get_output(0)

    # scale = sqrt(C) * gamma  (fp32 constant)
    gamma_flat = gamma.flatten()[:num_channels]
    scale = (np.sqrt(num_channels) * gamma_flat).reshape(1, num_channels, 1, 1, 1)
    scale_t = graph_ops.add_constant(
        network, (1, num_channels, 1, 1, 1), scale, dtype=np.float32)
    result_fp32 = network.add_elementwise(
        normalized, scale_t, trt.ElementWiseOperation.PROD).get_output(0)

    return _to_bf16(network, result_fp32)


def _bf16_silu(
    network: "trt.INetworkDefinition", inp: "trt.ITensor"
) -> "trt.ITensor":
    """SiLU/Swish on a bf16 tensor: ``x * sigmoid(x)``.

    Activations and elementwise products preserve input dtype in
    STRONGLY_TYPED networks, so this stays in bf16 end-to-end.
    """
    sig = network.add_activation(inp, trt.ActivationType.SIGMOID).get_output(0)
    return network.add_elementwise(
        inp, sig, trt.ElementWiseOperation.PROD).get_output(0)


def _bf16_vae_resblock_3d(
    network: "trt.INetworkDefinition",
    inp: "trt.ITensor",
    cache_in1: "trt.ITensor",
    cache_in2: "trt.ITensor",
    *,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    in_channels: int,
    out_channels: int,
    eps: float,
) -> "trt.ITensor":
    """3D VAE residual block running in bf16.

    Norm -> SiLU -> CausalConv3d(3,3,3) -> Norm -> SiLU -> CausalConv3d(3,3,3) + shortcut.
    Norms cast to fp32 internally for the variance reduction.
    """
    g1 = weights[f"{prefix}.norm1.gamma"]
    h = _bf16_l2_channel_norm(network, inp, in_channels, g1, eps)
    h = _bf16_silu(network, h)
    h = _bf16_causal_conv3d(
        network, h, cache_in1,
        weight=weights[f"{prefix}.conv1.weight"],
        bias=weights.get(f"{prefix}.conv1.bias"),
        out_channels=out_channels,
        kernel_size=(3, 3, 3),
        padding_hw=(1, 1),
    )

    g2 = weights[f"{prefix}.norm2.gamma"]
    h = _bf16_l2_channel_norm(network, h, out_channels, g2, eps)
    h = _bf16_silu(network, h)
    h = _bf16_causal_conv3d(
        network, h, cache_in2,
        weight=weights[f"{prefix}.conv2.weight"],
        bias=weights.get(f"{prefix}.conv2.bias"),
        out_channels=out_channels,
        kernel_size=(3, 3, 3),
        padding_hw=(1, 1),
    )

    if in_channels != out_channels:
        shortcut = _bf16_conv3d_as_conv2d(
            network, inp,
            weight=weights[f"{prefix}.conv_shortcut.weight"],
            bias=weights.get(f"{prefix}.conv_shortcut.bias"),
            out_channels=out_channels,
            kernel_size=(1, 1, 1),
        )
    else:
        shortcut = inp

    return network.add_elementwise(
        h, shortcut, trt.ElementWiseOperation.SUM).get_output(0)


def _bf16_vae_spatial_attention(
    network: "trt.INetworkDefinition",
    inp: "trt.ITensor",
    *,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    channels: int,
    eps: float,
) -> "trt.ITensor":
    """Single-head spatial self-attention over [B, C, T, H, W] in bf16.

    Mirrors ``graph_blocks.add_vae_spatial_attention(norm_type=l2_channel_norm)``
    but with bf16 conv/matmul weights. Norm reductions cast to fp32; the
    softmax/IAttention runs natively on bf16 Q/K/V.
    """
    b, c, t, h, w = inp.shape
    bt = b * t
    hw = h * w
    identity = inp

    normed = _bf16_l2_channel_norm(
        network, inp, channels, weights[f"{prefix}.norm.gamma"], eps)

    # QKV projection: 1x1 conv per spatial position, applied per (B*T) frame.
    # The to_qkv weight on disk is [3C, C, 1, 1] (HF Conv2d).
    qkv_w = weights[f"{prefix}.to_qkv.weight"]
    if qkv_w.ndim == 4:
        qkv_w_5d = qkv_w.reshape(3 * c, c, 1, qkv_w.shape[2], qkv_w.shape[3])
    else:
        qkv_w_5d = qkv_w.reshape(3 * c, c, 1, 1, 1)
    qkv = _bf16_conv3d_as_conv2d(
        network, normed,
        weight=qkv_w_5d,
        bias=weights.get(f"{prefix}.to_qkv.bias"),
        out_channels=3 * c,
        kernel_size=(1, 1, 1),
    )
    # qkv: [B, 3C, T, H, W] -> [BT, HW, 3C]
    qkv_flat = network.add_shuffle(qkv)
    qkv_flat.first_transpose = trt.Permutation([0, 2, 3, 4, 1])  # [B,T,H,W,3C]
    qkv_flat.reshape_dims = (bt, hw, 3 * c)

    q_slice = network.add_slice(
        qkv_flat.get_output(0),
        start=(0, 0, 0), shape=(bt, hw, c), stride=(1, 1, 1)).get_output(0)
    k_slice = network.add_slice(
        qkv_flat.get_output(0),
        start=(0, 0, c), shape=(bt, hw, c), stride=(1, 1, 1)).get_output(0)
    v_slice = network.add_slice(
        qkv_flat.get_output(0),
        start=(0, 0, 2 * c), shape=(bt, hw, c), stride=(1, 1, 1)).get_output(0)

    q_4d = network.add_shuffle(q_slice)
    q_4d.reshape_dims = (bt, 1, hw, c)
    k_4d = network.add_shuffle(k_slice)
    k_4d.reshape_dims = (bt, 1, hw, c)
    v_4d = network.add_shuffle(v_slice)
    v_4d.reshape_dims = (bt, 1, hw, c)

    attn_scale = 1.0 / np.sqrt(max(c, 1))
    ctx = graph_ops.add_attention_core(
        network,
        q_4d.get_output(0),
        k_4d.get_output(0),
        v_4d.get_output(0),
        scale=float(attn_scale),
    )  # [BT, 1, HW, C]

    # Back to [B, C, T, H, W]
    ctx_5d = network.add_shuffle(ctx)
    ctx_5d.reshape_dims = (b, t, h, w, c)
    ctx_5d.second_transpose = trt.Permutation([0, 4, 1, 2, 3])

    # Output projection: 1x1 conv. proj.weight on disk is [C, C, 1, 1].
    proj_w = weights[f"{prefix}.proj.weight"]
    if proj_w.ndim == 4:
        proj_w_5d = proj_w.reshape(c, c, 1, proj_w.shape[2], proj_w.shape[3])
    else:
        proj_w_5d = proj_w.reshape(c, c, 1, 1, 1)
    out = _bf16_conv3d_as_conv2d(
        network, ctx_5d.get_output(0),
        weight=proj_w_5d,
        bias=weights.get(f"{prefix}.proj.bias"),
        out_channels=c,
        kernel_size=(1, 1, 1),
    )

    return network.add_elementwise(
        out, identity, trt.ElementWiseOperation.SUM).get_output(0)


def _zero_cache_constant(
    network: "trt.INetworkDefinition",
    *,
    batch: int,
    channels: int,
    h: int,
    w: int,
    t_cache: int = 2,
) -> "trt.ITensor":
    """Bf16 zero cache tensor for an image-mode causal conv.

    Shape: [B, C, t_cache, H, W]. With zeros the causal conv produces
    left-zero-padded output, exactly matching HF's first-frame behaviour
    where ``cache_x is None``.
    """
    shape = (batch, channels, t_cache, h, w)
    zeros = np.zeros(shape, dtype=np.float32)
    layer = network.add_constant(shape, _bf16_weights(zeros))
    return layer.get_output(0)


def _make_unnorm_constants(
    cfg: QwenImageVAEConfig,
    z_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape latents_mean / latents_std into [1, z_dim, 1, 1, 1] for broadcast."""
    mean = np.asarray(cfg.latents_mean, dtype=np.float32)
    std = np.asarray(cfg.latents_std, dtype=np.float32)
    if mean.shape != (z_dim,) or std.shape != (z_dim,):
        raise ValueError(
            f"latents_mean/std length {mean.shape}/{std.shape} mismatch z_dim={z_dim}"
        )
    return (
        mean.reshape(1, z_dim, 1, 1, 1),
        std.reshape(1, z_dim, 1, 1, 1),
    )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_qwen_image_vae_encoder_engine(
    cfg: QwenImageVAEConfig,
    weights: Mapping[str, "np.ndarray"],
    out_path: Path | str,
    *,
    image_h: int = 1024,
    image_w: int = 1024,
    verbose: bool = False,
) -> Path:
    """Build the Qwen-Image VAE encoder TRT engine and serialise the plan.

    The engine consumes an RGB image tensor in HF VAE convention
    ``[1, 3, 1, H, W]`` with values in ``[-1, 1]`` and emits the posterior
    mode/mean ``[1, z_dim, 1, H/8, W/8]`` named ``latent``. The C++ runtime
    applies Qwen's `(latent - latents_mean) / latents_std` normalization
    after this engine returns.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.input_channels != 3:
        raise ValueError(
            "Qwen-Image VAE only supports input_channels=3 (RGB input)"
        )
    if image_h <= 0 or image_w <= 0:
        raise ValueError("image_h/image_w must be positive")
    if image_h % cfg.spatial_compression_ratio != 0 or image_w % cfg.spatial_compression_ratio != 0:
        raise ValueError(
            "VAE encoder image_h/image_w must be divisible by spatial_compression_ratio"
        )

    _BF16_WEIGHT_REFS.clear()

    z_dim = cfg.z_dim
    base = cfg.base_dim
    dim_mult = list(cfg.dim_mult)
    num_res = cfg.num_res_blocks
    channels_list = [base * m for m in [1] + dim_mult]

    def take(name: str) -> np.ndarray:
        if name not in weights:
            raise KeyError(f"missing required weight: {name!r}")
        return _as_numpy(weights[name], name=name)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)

    batch = 1
    cur_h, cur_w = image_h, image_w

    image_fp32 = network.add_input(
        "image", trt.float32, (batch, 3, 1, image_h, image_w)
    )
    x = _to_bf16(network, image_fp32)

    # Initial image-mode causal conv.
    ci_cache = _zero_cache_constant(
        network, batch=batch, channels=3, h=cur_h, w=cur_w)
    x = _bf16_causal_conv3d(
        network, x, ci_cache,
        weight=take("encoder.conv_in.weight"),
        bias=take("encoder.conv_in.bias"),
        out_channels=channels_list[0],
        kernel_size=(3, 3, 3),
        padding_hw=(1, 1),
    )
    cur_ch = channels_list[0]

    scale = 1.0
    for level, (in_ch_base, out_ch) in enumerate(zip(channels_list[:-1], channels_list[1:])):
        in_ch = in_ch_base
        for block in range(num_res):
            prefix = f"encoder.down_blocks.{level * (num_res + 1) + block}"
            c1 = _zero_cache_constant(
                network, batch=batch, channels=in_ch, h=cur_h, w=cur_w)
            c2 = _zero_cache_constant(
                network, batch=batch, channels=out_ch, h=cur_h, w=cur_w)
            rb_weights = _gather_resblock_weights(weights, prefix, in_ch, out_ch)
            x = _bf16_vae_resblock_3d(
                network, x, c1, c2,
                weights=rb_weights, prefix=prefix,
                in_channels=in_ch, out_channels=out_ch,
                eps=cfg.norm_eps,
            )
            in_ch = out_ch
            cur_ch = out_ch

            if scale in getattr(cfg, "attn_scales", []):
                raise NotImplementedError(
                    "Qwen-Image VAE encoder attention-in-down-blocks is not implemented"
                )

        if level != len(dim_mult) - 1:
            down_prefix = f"encoder.down_blocks.{level * (num_res + 1) + num_res}.resample.1"
            x = _bf16_spatial_downsample_with_conv(
                network,
                x,
                weight=take(f"{down_prefix}.weight"),
                bias=take(f"{down_prefix}.bias"),
            )
            cur_h //= 2
            cur_w //= 2
            scale /= 2.0
            if verbose:
                print(
                    f"[qwen-image-vae-encoder] down{level}: ch={cur_ch}, {cur_h}x{cur_w}",
                    file=sys.stderr,
                )

    # Middle block: resnet.0 -> attention.0 -> resnet.1.
    mid_ch = cur_ch
    for mi in range(2):
        prefix = f"encoder.mid_block.resnets.{mi}"
        c1 = _zero_cache_constant(
            network, batch=batch, channels=mid_ch, h=cur_h, w=cur_w)
        c2 = _zero_cache_constant(
            network, batch=batch, channels=mid_ch, h=cur_h, w=cur_w)
        rb_weights = _gather_resblock_weights(weights, prefix, mid_ch, mid_ch)
        x = _bf16_vae_resblock_3d(
            network, x, c1, c2,
            weights=rb_weights, prefix=prefix,
            in_channels=mid_ch, out_channels=mid_ch,
            eps=cfg.norm_eps,
        )
        if mi == 0:
            attn_prefix = "encoder.mid_block.attentions.0"
            attn_weights = _gather_attention_weights(weights, attn_prefix, mid_ch)
            x = _bf16_vae_spatial_attention(
                network, x,
                weights=attn_weights,
                prefix=attn_prefix,
                channels=mid_ch,
                eps=cfg.norm_eps,
            )

    x = _bf16_l2_channel_norm(
        network, x, cur_ch, take("encoder.norm_out.gamma"), cfg.norm_eps)
    x = _bf16_silu(network, x)

    co_cache = _zero_cache_constant(
        network, batch=batch, channels=cur_ch, h=cur_h, w=cur_w)
    x = _bf16_causal_conv3d(
        network, x, co_cache,
        weight=take("encoder.conv_out.weight"),
        bias=take("encoder.conv_out.bias"),
        out_channels=z_dim * 2,
        kernel_size=(3, 3, 3),
        padding_hw=(1, 1),
    )

    moments = _bf16_conv3d_as_conv2d(
        network,
        x,
        weight=take("quant_conv.weight"),
        bias=take("quant_conv.bias"),
        out_channels=z_dim * 2,
        kernel_size=(1, 1, 1),
    )

    # DiagonalGaussianDistribution.mode() returns the mean half.
    mean = network.add_slice(
        moments,
        start=(0, 0, 0, 0, 0),
        shape=(batch, z_dim, 1, cur_h, cur_w),
        stride=(1, 1, 1, 1, 1),
    ).get_output(0)
    mean = _to_fp32(network, mean)
    mean.name = "latent"
    network.mark_output(mean)

    if verbose:
        print(
            f"[qwen-image-vae-encoder] Building encoder "
            f"(image={image_h}x{image_w}, latent={cur_h}x{cur_w}, z={z_dim})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialisation failed for Qwen-Image VAE encoder")

    with open(out_path, "wb") as f:
        f.write(bytes(plan))
    return out_path


def build_qwen_image_vae_decoder_engine(
    cfg: QwenImageVAEConfig,
    weights: Mapping[str, "np.ndarray"],
    out_path: Path | str,
    *,
    h_lat: int = 32,
    w_lat: int = 32,
    apply_latent_unnorm: bool = False,
    max_batch_size: int = 1,
    opt_batch_size: int | None = None,
    verbose: bool = False,
) -> Path:
    """Build the Qwen-Image VAE decoder TRT engine and serialise the plan.

    Args:
        cfg: Architecture configuration.
        weights: Mapping of HF-style decoder weight names. Required keys
            (using ``decoder.`` prefix as in diffusers checkpoints):
              - ``post_quant_conv.weight``                 [z, z, 1, 1, 1]
              - ``post_quant_conv.bias``                   [z]
              - ``decoder.conv_in.weight``                 [base*dim_mult[-1], z, 3, 3, 3]
              - ``decoder.conv_in.bias``                   [base*dim_mult[-1]]
              - mid-block ResBlock pairs (resnets.0/1) with norm{1,2}.gamma,
                conv{1,2}.weight/.bias, and optional conv_shortcut for
                channel-mismatched resnets.
              - mid-block attention (attentions.0) with norm.gamma,
                to_qkv.weight/.bias, proj.weight/.bias.
              - up_blocks for each level: (num_res_blocks+1) ResBlocks and
                upsamplers.0.resample.1.weight/.bias (spatial conv) and
                upsamplers.0.time_conv.weight/.bias (temporal conv -- LOADED
                but UNUSED in image mode).
              - ``decoder.norm_out.gamma``                 [out_ch, 1, 1, 1]
              - ``decoder.conv_out.weight``                [3, last_ch, 3, 3, 3]
              - ``decoder.conv_out.bias``                  [3]
        out_path: Where to write the serialised TRT plan.
        h_lat, w_lat: Latent spatial dims (B is fixed to 1). Output will be
            ``[1, 3, 1, h_lat*8, w_lat*8]``.
        apply_latent_unnorm: If True, the engine receives the *normalised*
            latent (post-DiT) and applies ``z * std + mean`` internally. If
            False (default), the engine consumes a latent already in HF's
            ``vae.decode`` input convention.
        max_batch_size: Accepted for API uniformity with the text encoder and
            DiT builders, but per the diffusion batch-inference design
            (Decision E, 2026-05-19) the VAE always caps at 1 — the C++
            pipeline loops over per-sample latents calling this engine once
            per latent. The leading batch dim stays statically baked at 1
            regardless of the requested cap, so no dynamic TRT profile is
            attached. Values >1 are accepted but logged-and-clamped so the
            family plugin can pass the same cap through to every builder
            without special-casing the VAE.
        opt_batch_size: Accepted for API uniformity; unused (see
            ``max_batch_size``).
        verbose: Enable TRT verbose logging.

    Returns:
        Resolved ``Path`` to the written plan file.
    """
    if max_batch_size < 1:
        raise ValueError(
            f"max_batch_size must be >= 1 (got {max_batch_size})"
        )
    # Per design Decision E (2026-05-19) the VAE is always sliced at B=1.
    # The kwarg above is accepted for API uniformity; we deliberately ignore
    # the requested cap and keep the engine statically batched at 1.
    del opt_batch_size  # Unused by design — see docstring.
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.input_channels != 3:
        raise ValueError(
            "Qwen-Image VAE only supports input_channels=3 (RGB output)"
        )

    # Reset the bf16 anchor list for this build so we never accidentally
    # keep references to buffers from a previous build alive.
    _BF16_WEIGHT_REFS.clear()

    z_dim = cfg.z_dim
    base = cfg.base_dim
    dim_mult = list(cfg.dim_mult)
    num_levels = len(dim_mult)
    num_res = cfg.num_res_blocks

    channels_list = [base * m for m in dim_mult]  # encoder order
    dec_channels = list(reversed(channels_list))  # decoder iteration order
    mid_ch = dec_channels[0]

    def take(name: str) -> np.ndarray:
        if name not in weights:
            raise KeyError(f"missing required weight: {name!r}")
        return _as_numpy(weights[name], name=name)

    # ---- Build TRT network. -----
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.clear_flag(trt.BuilderFlag.TF32)

    # All internal tensor shapes use [B, C, T, H, W]; B=1, T=1 throughout.
    batch = 1
    cur_h, cur_w = h_lat, w_lat

    # ---- Input (fp32) and optional un-normalisation in fp32 ---
    # The un-norm scalars are small enough (z_dim=16) that fp32 cost is
    # negligible, and doing the multiply/add in fp32 protects against bf16
    # rounding of the latents_mean/std vectors which carry meaningful
    # precision (e.g. -0.7571 vs -0.7578 in bf16).
    latent_fp32 = network.add_input(
        "latent", trt.float32, (batch, z_dim, 1, h_lat, w_lat)
    )

    x_fp32 = latent_fp32
    if apply_latent_unnorm:
        mean_arr, std_arr = _make_unnorm_constants(cfg, z_dim)
        std_const = graph_ops.add_constant(
            network, std_arr.shape, std_arr, dtype=np.float32)
        mean_const = graph_ops.add_constant(
            network, mean_arr.shape, mean_arr, dtype=np.float32)
        scaled = network.add_elementwise(
            x_fp32, std_const, trt.ElementWiseOperation.PROD).get_output(0)
        x_fp32 = network.add_elementwise(
            scaled, mean_const, trt.ElementWiseOperation.SUM).get_output(0)

    # Cast to bf16 for the rest of the network.
    x = _to_bf16(network, x_fp32)

    # ---- post_quant_conv (1x1x1 Conv3D, z_dim -> z_dim) ---
    x = _bf16_conv3d_as_conv2d(
        network, x,
        weight=take("post_quant_conv.weight"),
        bias=take("post_quant_conv.bias"),
        out_channels=z_dim,
        kernel_size=(1, 1, 1),
    )

    # ---- conv_in (CausalConv3d, z_dim -> mid_ch, k=(3,3,3)) ---
    ci_cache = _zero_cache_constant(
        network, batch=batch, channels=z_dim, h=cur_h, w=cur_w)
    x = _bf16_causal_conv3d(
        network, x, ci_cache,
        weight=take("decoder.conv_in.weight"),
        bias=take("decoder.conv_in.bias"),
        out_channels=mid_ch,
        kernel_size=(3, 3, 3),
        padding_hw=(1, 1),
    )
    cur_ch = mid_ch
    if verbose:
        print(f"[qwen-image-vae] conv_in: ch={cur_ch}, {cur_h}x{cur_w}",
              file=sys.stderr)

    # ---- mid_block: resnet.0 -> attention -> resnet.1 ---
    for mi in range(2):
        prefix = f"decoder.mid_block.resnets.{mi}"
        c1 = _zero_cache_constant(
            network, batch=batch, channels=mid_ch, h=cur_h, w=cur_w)
        c2 = _zero_cache_constant(
            network, batch=batch, channels=mid_ch, h=cur_h, w=cur_w)
        rb_weights = _gather_resblock_weights(weights, prefix, mid_ch, mid_ch)
        x = _bf16_vae_resblock_3d(
            network, x, c1, c2,
            weights=rb_weights, prefix=prefix,
            in_channels=mid_ch, out_channels=mid_ch,
            eps=cfg.norm_eps,
        )
        if mi == 0:
            attn_weights = _gather_attention_weights(
                weights, "decoder.mid_block.attentions.0", mid_ch)
            x = _bf16_vae_spatial_attention(
                network, x,
                weights=attn_weights,
                prefix="decoder.mid_block.attentions.0",
                channels=mid_ch,
                eps=cfg.norm_eps,
            )
    if verbose:
        print(f"[qwen-image-vae] mid done: ch={cur_ch}, {cur_h}x{cur_w}",
              file=sys.stderr)

    # ---- up_blocks ---
    prev_ch = mid_ch
    for level in range(num_levels):
        out_ch = dec_channels[level]
        has_spatial = level < num_levels - 1
        # Image mode: time_conv is SKIPPED inside HF (Rep path); weights are
        # required to exist on disk but the graph never references them.

        # 3 resnets per block (num_res_blocks + 1)
        for blk in range(num_res + 1):
            prefix = f"decoder.up_blocks.{level}.resnets.{blk}"
            in_ch = prev_ch if blk == 0 else out_ch
            c1 = _zero_cache_constant(
                network, batch=batch, channels=in_ch, h=cur_h, w=cur_w)
            c2 = _zero_cache_constant(
                network, batch=batch, channels=out_ch, h=cur_h, w=cur_w)
            rb_weights = _gather_resblock_weights(
                weights, prefix, in_ch, out_ch)
            x = _bf16_vae_resblock_3d(
                network, x, c1, c2,
                weights=rb_weights, prefix=prefix,
                in_channels=in_ch, out_channels=out_ch,
                eps=cfg.norm_eps,
            )
            prev_ch = out_ch
        cur_ch = out_ch

        if verbose:
            print(
                f"[qwen-image-vae] up{level}: ch={out_ch}, {cur_h}x{cur_w}",
                file=sys.stderr,
            )

        # Spatial upsample (HF QwenImageResample.resample = Upsample(2x) +
        # Conv2d(C, C//2 or same, 3, padding=1)). The conv changes channels.
        if has_spatial:
            sp_prefix = (
                f"decoder.up_blocks.{level}.upsamplers.0.resample.1"
            )
            sp_w = take(f"{sp_prefix}.weight")
            sp_b = take(f"{sp_prefix}.bias")
            # Weight shape from diffusers Conv2d is [C_out, C_in, 3, 3];
            # the bf16 spatial-upsample helper expects [C_out, C_in, 1, 3, 3].
            if sp_w.ndim == 4:
                sp_w = sp_w.reshape(
                    sp_w.shape[0], sp_w.shape[1], 1, sp_w.shape[2], sp_w.shape[3]
                )
            x = _bf16_spatial_upsample_with_conv(
                network, x,
                weight=sp_w, bias=sp_b, scale=2,
            )
            prev_ch = sp_w.shape[0]
            cur_h *= 2
            cur_w *= 2
            cur_ch = prev_ch
            if verbose:
                print(
                    f"[qwen-image-vae]   spatial 2x -> ch={prev_ch}, "
                    f"{cur_h}x{cur_w}",
                    file=sys.stderr,
                )

    # ---- norm_out (L2 channel norm) + SiLU (bf16 IO, fp32 reduction) ---
    x = _bf16_l2_channel_norm(
        network, x, cur_ch, take("decoder.norm_out.gamma"), cfg.norm_eps)
    x = _bf16_silu(network, x)

    # ---- conv_out (CausalConv3d, cur_ch -> 3, k=(3,3,3)) ---
    co_cache = _zero_cache_constant(
        network, batch=batch, channels=cur_ch, h=cur_h, w=cur_w)
    x = _bf16_causal_conv3d(
        network, x, co_cache,
        weight=take("decoder.conv_out.weight"),
        bias=take("decoder.conv_out.bias"),
        out_channels=3,
        kernel_size=(3, 3, 3),
        padding_hw=(1, 1),
    )

    # ---- Cast back to fp32 before clamp + output ---
    x = _to_fp32(network, x)

    lo_const = graph_ops.add_constant(
        network, (1, 1, 1, 1, 1),
        np.array([-1.0], dtype=np.float32),
        dtype=np.float32,
    )
    hi_const = graph_ops.add_constant(
        network, (1, 1, 1, 1, 1),
        np.array([1.0], dtype=np.float32),
        dtype=np.float32,
    )
    x = network.add_elementwise(
        x, lo_const, trt.ElementWiseOperation.MAX).get_output(0)
    x = network.add_elementwise(
        x, hi_const, trt.ElementWiseOperation.MIN).get_output(0)

    # Mark output (fp32).
    x.name = "image"
    network.mark_output(x)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TRT engine serialisation failed for Qwen-Image VAE")

    with open(out_path, "wb") as f:
        f.write(bytes(plan))
    return out_path


# ---------------------------------------------------------------------------
# Weight gathering helpers (subset selection for graph_blocks helpers)
# ---------------------------------------------------------------------------

def _gather_resblock_weights(
    weights: Mapping[str, "np.ndarray"],
    prefix: str,
    in_channels: int,
    out_channels: int,
) -> dict[str, np.ndarray]:
    """Pull the resblock keys out of the full weight dict.

    Returns a small dict containing the keys ``_bf16_vae_resblock_3d`` needs.
    """
    out: dict[str, np.ndarray] = {}
    keys = [
        f"{prefix}.norm1.gamma",
        f"{prefix}.norm2.gamma",
        f"{prefix}.conv1.weight",
        f"{prefix}.conv1.bias",
        f"{prefix}.conv2.weight",
        f"{prefix}.conv2.bias",
    ]
    for k in keys:
        if k not in weights:
            raise KeyError(f"missing required weight: {k!r}")
        out[k] = _as_numpy(weights[k], name=k)
    if in_channels != out_channels:
        sc_key = f"{prefix}.conv_shortcut"
        out[f"{sc_key}.weight"] = _as_numpy(
            weights[f"{sc_key}.weight"], name=f"{sc_key}.weight"
        )
        if f"{sc_key}.bias" in weights:
            out[f"{sc_key}.bias"] = _as_numpy(
                weights[f"{sc_key}.bias"], name=f"{sc_key}.bias"
            )
    return out


def _gather_attention_weights(
    weights: Mapping[str, "np.ndarray"],
    prefix: str,
    channels: int,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    keys = [
        f"{prefix}.norm.gamma",
        f"{prefix}.to_qkv.weight",
        f"{prefix}.to_qkv.bias",
        f"{prefix}.proj.weight",
        f"{prefix}.proj.bias",
    ]
    for k in keys:
        if k not in weights:
            raise KeyError(f"missing required weight: {k!r}")
        out[k] = _as_numpy(weights[k], name=k)
    return out


# ---------------------------------------------------------------------------
# Real-weight loader
# ---------------------------------------------------------------------------

def load_qwen_image_vae_weights(
    vae_dir: "str | Path",
) -> tuple[QwenImageVAEConfig, dict[str, np.ndarray]]:
    """Load Qwen-Image VAE config + decoder weights from a diffusers VAE dir.

    Reads ``vae_dir/config.json`` (HF VAE config) and the ``*.safetensors``
    files; returns the parsed ``QwenImageVAEConfig`` and a dict of decoder
    weights (encoder keys are filtered out).
    """
    vae_dir = Path(vae_dir)
    config_path = vae_dir / "config.json"
    with open(config_path, "r") as f:
        raw = json.load(f)

    # HF uses 'temperal_downsample' (sic) -- accept both spellings.
    temp_ds = raw.get("temperal_downsample") or raw.get("temporal_downsample")
    if temp_ds is None:
        temp_ds = [False, True, True]

    cfg = QwenImageVAEConfig(
        z_dim=int(raw.get("z_dim", 16)),
        base_dim=int(raw.get("base_dim", 96)),
        dim_mult=list(raw.get("dim_mult", [1, 2, 4, 4])),
        num_res_blocks=int(raw.get("num_res_blocks", 2)),
        temporal_downsample=list(temp_ds),
        input_channels=int(raw.get("input_channels", 3)),
        attn_scales=list(raw.get("attn_scales", [])),
        latents_mean=list(raw.get("latents_mean", [])) or QwenImageVAEConfig().latents_mean,
        latents_std=list(raw.get("latents_std", [])) or QwenImageVAEConfig().latents_std,
    )

    # Pull VAE weights needed by both T2I decode and Edit encode paths. We open via the torch
    # framework because Qwen-Image VAE checkpoints store bf16 weights and
    # numpy has no bf16 dtype; torch handles the conversion.
    weights: dict[str, np.ndarray] = {}
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError(
            "safetensors is required to load HF VAE weights"
        ) from exc
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "torch is required to load Qwen-Image VAE weights (bf16 dtype)"
        ) from exc

    st_files = sorted(vae_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(
            f"no *.safetensors under {vae_dir!r}"
        )
    for st_path in st_files:
        with safe_open(str(st_path), framework="pt") as st:
            for k in st.keys():
                if not (
                    k.startswith("decoder.") or k.startswith("encoder.") or
                    k.startswith("post_quant_conv.") or k.startswith("quant_conv.")
                ):
                    continue
                t = st.get_tensor(k)
                weights[k] = t.detach().to(torch.float32).cpu().numpy()
    if not weights:
        raise RuntimeError(
            f"no VAE weights found under {vae_dir!r}"
        )
    return cfg, weights


__all__ = [
    "QwenImageVAEConfig",
    "build_qwen_image_vae_encoder_engine",
    "build_qwen_image_vae_decoder_engine",
    "load_qwen_image_vae_weights",
]
