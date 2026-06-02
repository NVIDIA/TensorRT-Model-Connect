"""Qwen-Image MMDiT denoiser engine builder.

Builds the Qwen-Image MMDiT denoiser engine (patchify, timestep embedding,
RoPE, joint blocks, full stack, and HF weight loader).

The Qwen-Image transformer is joint-stream MMDiT only (no FLUX-style single
blocks for the image-only tail). 60 joint blocks, hidden_size=3072,
24 heads, head_dim=128, rope_axes_dim=[16, 56, 56], rope_theta=10000.

Patchify / unpatchify layout convention
---------------------------------------

Matches diffusers ``QwenImagePipeline._pack_latents``::

    # diffusers/src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py:334-339
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)

i.e. einops ``b c (h p1) (w p2) -> b (h w) (c p1 p2)``. Within each output
patch the C channels come first, then the ``p1 * p2`` intra-patch pixels.

Trace IDs: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Union

import numpy as np

from tensorrt_model_connect import trt_compat
from ... import graph_ops

trt = trt_compat.get_trt()

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# BF16 internal-compute helpers.
#
# Strategy: STRONGLY_TYPED network with bf16 weights/activations for the heavy
# compute (matmul, attention, AdaLN, GELU FFN, RoPE complex muls). The
# network's IO stays fp32 -- inputs are cast to bf16 at the boundary and
# outputs are cast back to fp32 before mark_output, so the C++ runtime
# (qwen_image_pipeline.cpp) and Python QwenImageDebugRunner can keep
# binding fp32 host buffers without changes. Mirrors the pattern used by
# flux2_dit_builder.py.
#
# RoPE cos/sin tables and the eps constants stay fp32 at construction time
# and are cast to bf16 at point-of-use (the numerical sensitivity of the
# tables is preserved that way; flux2 does the same).
# ---------------------------------------------------------------------------

_CAST_DTYPE = trt.bfloat16

# Anchors for bf16 numpy arrays so Python GC doesn't free them before
# builder.build_serialized_network() returns. Without this, TRT silently
# reads garbage (or segfaults).
_weight_refs: list[np.ndarray] = []


def _np_reduced_dtype():
    """numpy dtype matching ``_CAST_DTYPE`` (bf16 -> ml_dtypes.bfloat16)."""
    import ml_dtypes
    return ml_dtypes.bfloat16


def _to_compute_dtype(network, tensor):
    """Cast a tensor to the reduced compute dtype if not already."""
    if tensor.dtype == _CAST_DTYPE:
        return tensor
    return network.add_cast(tensor, _CAST_DTYPE).get_output(0)


def _to_fp32(network, tensor):
    """Cast a tensor back to fp32 if not already."""
    if tensor.dtype == trt.float32:
        return tensor
    return network.add_cast(tensor, trt.float32).get_output(0)


def _make_reduced_weights(data_fp32: np.ndarray):
    """Build (trt.Weights, anchored_ndarray) in the reduced compute dtype."""
    import ml_dtypes
    if (
        isinstance(data_fp32, np.ndarray)
        and data_fp32.dtype == ml_dtypes.bfloat16
        and data_fp32.flags.c_contiguous
    ):
        bf16_arr = data_fp32
    else:
        bf16_arr = np.ascontiguousarray(np.asarray(data_fp32).astype(ml_dtypes.bfloat16))
    w = trt.Weights(trt.bfloat16, bf16_arr.ctypes.data, bf16_arr.size)
    return w, bf16_arr


def _add_constant_reduced(network, shape, values_fp32) -> "trt.ITensor":
    """Add a constant in the reduced compute dtype (anchors the bf16 array)."""
    w, arr_ref = _make_reduced_weights(values_fp32)
    _weight_refs.append(arr_ref)
    return network.add_constant(shape, w).get_output(0)


def _reset_weight_refs() -> None:
    """Clear the bf16 array anchor list before a fresh engine build."""
    global _weight_refs
    _weight_refs = []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_builder(verbose: bool):
    """Create a TRT builder + STRONGLY_TYPED network with sane defaults.

    Also resets the bf16 weight-anchor list so a fresh build doesn't keep
    references to arrays from a prior build (would only matter for tests
    that build many engines in one process, but it's cheap insurance).
    """
    _reset_weight_refs()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    # Shape ops are tiny; 256 MiB workspace is plenty.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    return builder, config, network


def _serialize_and_write(builder, network, config, out_path: PathLike, label: str) -> Path:
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"{label} TRT engine build failed")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(bytes(plan))
    return out_path


def _check_divisible(h_lat: int, w_lat: int, patch_size: int) -> None:
    if h_lat % patch_size != 0 or w_lat % patch_size != 0:
        raise ValueError(
            f"latent dims ({h_lat}, {w_lat}) must be divisible by "
            f"patch_size={patch_size}"
        )


# ---------------------------------------------------------------------------
# Patchify
# ---------------------------------------------------------------------------

def build_patchify_engine(
    out_path: PathLike,
    *,
    batch_size: int = 1,
    channels: int,
    patch_size: int,
    h_lat: int,
    w_lat: int,
    verbose: bool = False,
) -> Path:
    """Build a TRT engine that performs the Qwen-Image patchify reshape.

    Input  ``latent``   shape: ``[batch_size, channels, h_lat, w_lat]``  fp32
    Output ``patched``  shape: ``[batch_size, (h_lat/p)*(w_lat/p), channels*p*p]``

    where ``p = patch_size``.

    The op composes two TRT shuffle layers (reshape + transpose, then a
    final reshape), implementing the diffusers layout
    ``b c (h p1) (w p2) -> b (h w) (c p1 p2)``.

    The engine has fully static shapes -- the (B, C, H, W, p) constants are
    baked in at build time. Callers wanting a different size must build a
    fresh engine.
    """
    _check_divisible(h_lat, w_lat, patch_size)

    p = patch_size
    h_grid = h_lat // p
    w_grid = w_lat // p
    num_patches = h_grid * w_grid
    patch_dim = channels * p * p

    builder, config, network = _make_builder(verbose)

    inp = network.add_input(
        "latent", trt.float32, (batch_size, channels, h_lat, w_lat)
    )

    # Shuffle 1: reshape [B, C, H, W] -> [B, C, H/p, p, W/p, p]
    # then permute to [B, H/p, W/p, C, p, p].
    shuffle1 = network.add_shuffle(inp)
    shuffle1.reshape_dims = (batch_size, channels, h_grid, p, w_grid, p)
    shuffle1.second_transpose = trt.Permutation([0, 2, 4, 1, 3, 5])

    # Shuffle 2: collapse spatial grid and intra-patch dims.
    # [B, H/p, W/p, C, p, p] -> [B, (H/p)*(W/p), C*p*p]
    shuffle2 = network.add_shuffle(shuffle1.get_output(0))
    shuffle2.reshape_dims = (batch_size, num_patches, patch_dim)

    out = shuffle2.get_output(0)
    out.name = "patched"
    network.mark_output(out)

    print(
        f"[qwen-image-dit] Building patchify engine "
        f"(B={batch_size}, C={channels}, H={h_lat}, W={w_lat}, p={p}) "
        f"-> [{batch_size}, {num_patches}, {patch_dim}]",
        file=sys.stderr,
    )
    return _serialize_and_write(builder, network, config, out_path, "patchify")


# ---------------------------------------------------------------------------
# Unpatchify
# ---------------------------------------------------------------------------

def build_unpatchify_engine(
    out_path: PathLike,
    *,
    batch_size: int = 1,
    channels: int,
    patch_size: int,
    h_lat: int,
    w_lat: int,
    verbose: bool = False,
) -> Path:
    """Inverse of :func:`build_patchify_engine`.

    Input  ``patched`` shape: ``[batch_size, (h_lat/p)*(w_lat/p), channels*p*p]``
    Output ``latent``  shape: ``[batch_size, channels, h_lat, w_lat]``

    Composes two TRT shuffle layers, implementing the diffusers layout
    ``b (h w) (c p1 p2) -> b c (h p1) (w p2)``.
    """
    _check_divisible(h_lat, w_lat, patch_size)

    p = patch_size
    h_grid = h_lat // p
    w_grid = w_lat // p
    num_patches = h_grid * w_grid
    patch_dim = channels * p * p

    builder, config, network = _make_builder(verbose)

    inp = network.add_input(
        "patched", trt.float32, (batch_size, num_patches, patch_dim)
    )

    # Shuffle 1: reshape [B, N, D] -> [B, H/p, W/p, C, p, p]
    # then permute to [B, C, H/p, p, W/p, p].
    shuffle1 = network.add_shuffle(inp)
    shuffle1.reshape_dims = (batch_size, h_grid, w_grid, channels, p, p)
    shuffle1.second_transpose = trt.Permutation([0, 3, 1, 4, 2, 5])

    # Shuffle 2: collapse to [B, C, H, W].
    shuffle2 = network.add_shuffle(shuffle1.get_output(0))
    shuffle2.reshape_dims = (batch_size, channels, h_lat, w_lat)

    out = shuffle2.get_output(0)
    out.name = "latent"
    network.mark_output(out)

    print(
        f"[qwen-image-dit] Building unpatchify engine "
        f"(B={batch_size}, N={num_patches}, D={patch_dim}, p={p}) "
        f"-> [{batch_size}, {channels}, {h_lat}, {w_lat}]",
        file=sys.stderr,
    )
    return _serialize_and_write(builder, network, config, out_path, "unpatchify")


# ---------------------------------------------------------------------------
# Timestep embedding (sinusoidal + 2-layer SiLU MLP)
#
# Matches diffusers ``QwenTimestepProjEmbeddings`` (transformer_qwenimage.py:174):
#
#     time_proj  = Timesteps(num_channels=256, flip_sin_to_cos=True,
#                            downscale_freq_shift=0, scale=1000)
#     timestep_embedder = TimestepEmbedding(in_channels=256,
#                                           time_embed_dim=hidden_size,
#                                           act_fn="silu")
#
# i.e. sinusoidal embedding (256 channels) -> Linear(256, hidden) -> SiLU
# -> Linear(hidden, hidden).
# ---------------------------------------------------------------------------

def _as_numpy(arr) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float32))


def _add_get_timestep_embedding(
    network,
    timestep,
    *,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1000.0,
    max_period: float = 10000.0,
):
    """In-network port of ``diffusers.models.embeddings.get_timestep_embedding``.

    Input  ``timestep``: ITensor of shape ``[N]`` (fp32).
    Output ITensor of shape ``[N, embedding_dim]`` (fp32).
    """
    assert embedding_dim % 2 == 0, "embedding_dim must be even"
    half_dim = embedding_dim // 2

    # exponent = -log(max_period) * arange(half_dim) / (half_dim - downscale_freq_shift)
    # emb_base = exp(exponent)
    base = -math.log(max_period) * np.arange(half_dim, dtype=np.float32)
    base = base / (half_dim - downscale_freq_shift)
    base = np.exp(base).reshape(1, half_dim) * float(scale)

    base_const = network.add_constant((1, half_dim), trt.Weights(base)).get_output(0)

    # ts: [N] -> [N, 1]
    ts_reshape = network.add_shuffle(timestep)
    ts_reshape.reshape_dims = (-1, 1)
    ts_2d = ts_reshape.get_output(0)

    # emb = ts * base                              [N, half_dim]
    prod = network.add_elementwise(ts_2d, base_const, trt.ElementWiseOperation.PROD)
    arg = prod.get_output(0)

    sin_p = network.add_unary(arg, trt.UnaryOperation.SIN).get_output(0)
    cos_p = network.add_unary(arg, trt.UnaryOperation.COS).get_output(0)

    if flip_sin_to_cos:
        # diffusers: emb = cat([sin, cos]); if flip_sin_to_cos: emb = cat([cos, sin])
        order = [cos_p, sin_p]
    else:
        order = [sin_p, cos_p]
    cat = network.add_concatenation(order)
    cat.axis = 1
    return cat.get_output(0)


def _add_linear(network, x, weight_np: np.ndarray, bias_np: np.ndarray):
    """Linear(in -> out): y = x @ W^T + b in reduced compute dtype.

    weight_np shape: [out_dim, in_dim]; bias_np shape: [out_dim].
    Input x shape:  [N, in_dim]; output: [N, out_dim].
    Caller-supplied weights are fp32; this helper internally converts them
    to bf16 (anchored against GC) and emits a bf16 matmul + bias-add. ``x``
    is cast to bf16 at the boundary if needed; output stays in bf16.
    """
    out_dim, in_dim = weight_np.shape
    x_c = _to_compute_dtype(network, x)
    w_const = _add_constant_reduced(network, (out_dim, in_dim), np.ascontiguousarray(weight_np, dtype=np.float32))
    # x [N, in_dim] @ W^T [in_dim, out_dim] = [N, out_dim]
    matmul = network.add_matrix_multiply(
        x_c, trt.MatrixOperation.NONE, w_const, trt.MatrixOperation.TRANSPOSE
    )
    y = matmul.get_output(0)
    if bias_np is not None:
        b_const = _add_constant_reduced(network, (1, out_dim), np.ascontiguousarray(bias_np, dtype=np.float32))
        add = network.add_elementwise(y, b_const, trt.ElementWiseOperation.SUM)
        y = add.get_output(0)
    return y


def build_timestep_embed_engine(
    out_path: PathLike,
    *,
    weights: Mapping[str, "np.ndarray | object"],
    in_dim: int = 256,
    out_dim: int = 1024,
    batch_size: int = 1,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1000.0,
    max_period: float = 10000.0,
    verbose: bool = False,
) -> Path:
    """Build a TRT engine for the Qwen-Image timestep embedding.

    Pipeline (matches diffusers ``QwenTimestepProjEmbeddings``):

        timestep [N] --(sinusoidal)--> [N, in_dim]
                     --linear_1--> [N, out_dim]
                     --SiLU-->      [N, out_dim]
                     --linear_2--> [N, out_dim]

    Args:
        out_path: where to write the serialised TRT plan.
        weights: dict with keys
            ``"linear_1.weight"`` [out_dim, in_dim],
            ``"linear_1.bias"``   [out_dim],
            ``"linear_2.weight"`` [out_dim, out_dim],
            ``"linear_2.bias"``   [out_dim].
        in_dim:  sinusoidal embedding width (Qwen-Image: 256).
        out_dim: MLP hidden / output width (Qwen-Image: hidden_size = 3072).
        batch_size: number of timesteps fed at once. Bound to a static
            shape -- callers wanting a different batch must rebuild.
        flip_sin_to_cos, downscale_freq_shift, scale, max_period: forwarded
            to the sinusoidal embedding. Defaults match QwenImage.

    Returns:
        Path to the written plan.
    """
    required = ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias")
    for key in required:
        if key not in weights:
            raise KeyError(f"build_timestep_embed_engine: missing weight {key!r}")

    w1 = _as_numpy(weights["linear_1.weight"])
    b1 = _as_numpy(weights["linear_1.bias"])
    w2 = _as_numpy(weights["linear_2.weight"])
    b2 = _as_numpy(weights["linear_2.bias"])

    if w1.shape != (out_dim, in_dim):
        raise ValueError(f"linear_1.weight shape {w1.shape} != ({out_dim}, {in_dim})")
    if b1.shape != (out_dim,):
        raise ValueError(f"linear_1.bias shape {b1.shape} != ({out_dim},)")
    if w2.shape != (out_dim, out_dim):
        raise ValueError(f"linear_2.weight shape {w2.shape} != ({out_dim}, {out_dim})")
    if b2.shape != (out_dim,):
        raise ValueError(f"linear_2.bias shape {b2.shape} != ({out_dim},)")

    builder, config, network = _make_builder(verbose)

    timestep = network.add_input("timestep", trt.float32, (batch_size,))

    sample = _add_get_timestep_embedding(
        network,
        timestep,
        embedding_dim=in_dim,
        flip_sin_to_cos=flip_sin_to_cos,
        downscale_freq_shift=downscale_freq_shift,
        scale=scale,
        max_period=max_period,
    )
    h = _add_linear(network, sample, w1, b1)
    h = graph_ops.add_silu(network, h)
    h = _add_linear(network, h, w2, b2)
    # Engine output is fp32 by contract (matches the C++ runtime + Python
    # debug runner's fp32 host buffer bindings); internal compute stays bf16.
    h = _to_fp32(network, h)
    h.name = "timestep_emb"
    network.mark_output(h)

    print(
        f"[qwen-image-dit] Building timestep embed engine "
        f"(N={batch_size}, in_dim={in_dim}, out_dim={out_dim}) "
        f"-> [{batch_size}, {out_dim}]",
        file=sys.stderr,
    )
    return _serialize_and_write(builder, network, config, out_path, "timestep_embed")


# ---------------------------------------------------------------------------
# 3-axis RoPE (frame, height, width) for the Qwen-Image MMDiT denoiser.
#
# Matches diffusers ``QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56],
# scale_rope=True)`` (transformer_qwenimage.py:199-321):
#
#   * For each axis k the complex freqs at position i are
#         freqs[i, j] = exp(i * theta**(-2j/axes_dim[k]))    j in [0, axes_dim[k]/2)
#     (rope_params: torch.polar(ones_like(angles), angles)).
#   * Image (T=1) positions for token (h, w) are
#         frame_idx = 0
#         h_idx in {-(H - H//2), ..., -1, 0, 1, ..., H//2 - 1}  (scale_rope=True)
#         w_idx analogous, length W.
#     Concatenated along the last (head_dim/2) axis.
#   * Text positions: pos_freqs rows in
#         [max_vid_index, max_vid_index + n_text), where
#         max_vid_index = max(H//2, W//2).
#     Same row used for all three axes (the pos_freqs table is built once
#     from the same pos_index across axes).
#
# Output layout: concatenated as [img_freqs | txt_freqs] along the seq axis,
# giving a single (cos, sin) pair of shape [seq_len, head_dim] suitable for
# the attention application
#     out = x * cos + rotate_half(x) * sin
# (the cos/sin tables are "expanded" from the complex form by interleaved
# duplication along the last axis: cos[..., 2j] = cos[..., 2j+1] = real;
# sin[..., 2j] = sin[..., 2j+1] = imag).
# ---------------------------------------------------------------------------

_ROPE_PRECOMPUTE_LEN = 4096  # diffusers uses arange(4096) for pos/neg indices


def _rope_params_complex(index: np.ndarray, dim: int, theta: float) -> np.ndarray:
    """Return complex freqs of shape ``[len(index), dim // 2]`` (cos + i sin)."""
    assert dim % 2 == 0
    inv_freq = 1.0 / np.power(
        theta, np.arange(0, dim, 2, dtype=np.float64) / float(dim)
    )
    angles = np.outer(index.astype(np.float64), inv_freq)
    return np.cos(angles) + 1j * np.sin(angles)


def _qwen_rope_axis_freqs(
    axes_dim: list[int], theta: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return positive/negative complex RoPE tables split by Qwen image axis."""
    pos_index = np.arange(_ROPE_PRECOMPUTE_LEN)
    neg_index = pos_index[::-1] * -1 - 1  # [-1, -2, ..., -_ROPE_PRECOMPUTE_LEN]

    pos_axis_freqs = [
        _rope_params_complex(pos_index, axes_dim[k], theta) for k in range(3)
    ]
    neg_axis_freqs = [
        _rope_params_complex(neg_index, axes_dim[k], theta) for k in range(3)
    ]
    return pos_axis_freqs, neg_axis_freqs


def _precompute_qwen_image_freqs(
    *,
    axes_dim: list[int],
    pos_axis_freqs: list[np.ndarray],
    neg_axis_freqs: list[np.ndarray],
    h_lat: int,
    w_lat: int,
    frame_index: int,
) -> np.ndarray:
    """Pre-compute one image/grid chunk of Qwen 3-axis RoPE complex freqs.

    ``frame_index`` mirrors ``QwenEmbedRope._compute_video_freqs(..., idx)``
    and is what distinguishes multiple Edit condition-image grids from the
    generated-image grid.
    """
    head_dim = sum(axes_dim)
    splits = [a // 2 for a in axes_dim]

    frame = 1
    H, W = h_lat, w_lat

    # Frame axis (pos rows [0:1], shared across all (h, w)).
    freqs_frame = np.broadcast_to(
        pos_axis_freqs[0][frame_index : frame_index + frame].reshape(
            frame, 1, 1, splits[0]
        ),
        (frame, H, W, splits[0]),
    )

    # Height: scale_rope=True split [-(H - H//2):] from neg, then [:H//2] from pos.
    h_neg = neg_axis_freqs[1][-(H - H // 2):]
    h_pos = pos_axis_freqs[1][: H // 2]
    h_combined = np.concatenate([h_neg, h_pos], axis=0)  # [H, splits[1]]
    freqs_height = np.broadcast_to(
        h_combined.reshape(1, H, 1, splits[1]), (frame, H, W, splits[1])
    )

    w_neg = neg_axis_freqs[2][-(W - W // 2):]
    w_pos = pos_axis_freqs[2][: W // 2]
    w_combined = np.concatenate([w_neg, w_pos], axis=0)
    freqs_width = np.broadcast_to(
        w_combined.reshape(1, 1, W, splits[2]), (frame, H, W, splits[2])
    )

    return np.concatenate(
        [freqs_frame, freqs_height, freqs_width], axis=-1
    ).reshape(frame * H * W, head_dim // 2)


def _expand_qwen_rope_complex(combined: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expand complex Qwen RoPE freqs to duplicated real-valued cos/sin rows."""
    # Expand to [seq, head_dim] via interleaved duplication: each complex
    # entry contributes (cos, sin) repeated twice -- one for each of the
    # two real elements in a rotation pair.
    real = np.repeat(combined.real, 2, axis=-1).astype(np.float32)
    imag = np.repeat(combined.imag, 2, axis=-1).astype(np.float32)
    return np.ascontiguousarray(real), np.ascontiguousarray(imag)


def _precompute_qwen_rope_tables_for_shapes(
    axes_dim: list[int],
    image_shapes: list[tuple[int, int]],
    n_text: int,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute Qwen image/text RoPE tables for one or more image grids.

    ``image_shapes`` contains packed-token grids ``(h, w)`` in the order they
    are concatenated into the denoiser's image stream. T2I passes one shape.
    Edit passes the generated-image shape first, followed by VAE-condition
    image shapes. This mirrors diffusers ``QwenEmbedRope.forward`` where
    ``img_shapes`` is a list like ``[(gen_f, gen_h, gen_w),
    (cond_f, cond_h, cond_w)]``.
    """
    if len(axes_dim) != 3:
        raise ValueError(f"axes_dim must have 3 entries, got {axes_dim!r}")
    if any(a % 2 != 0 for a in axes_dim):
        raise ValueError(f"each axis dim must be even, got {axes_dim!r}")
    if not image_shapes:
        raise ValueError("image_shapes must contain at least one image grid")
    if n_text < 0:
        raise ValueError(f"n_text must be non-negative, got {n_text}")

    pos_axis_freqs, neg_axis_freqs = _qwen_rope_axis_freqs(axes_dim, theta)
    vid_freqs: list[np.ndarray] = []
    max_vid_index = 0
    for idx, (h_lat, w_lat) in enumerate(image_shapes):
        if h_lat <= 0 or w_lat <= 0:
            raise ValueError(
                f"image_shapes[{idx}] must be positive, got {(h_lat, w_lat)!r}"
            )
        max_vid_index = max(max_vid_index, h_lat // 2, w_lat // 2)
        vid_freqs.append(
            _precompute_qwen_image_freqs(
                axes_dim=axes_dim,
                pos_axis_freqs=pos_axis_freqs,
                neg_axis_freqs=neg_axis_freqs,
                h_lat=h_lat,
                w_lat=w_lat,
                frame_index=idx,
            )
        )

    if max_vid_index + n_text > _ROPE_PRECOMPUTE_LEN:
        raise ValueError(
            "n_text + max image grid half-extent exceeds the 4096-row "
            "Qwen RoPE pre-compute budget"
        )

    # Text freqs: rows [max_vid_index, max_vid_index + n_text) from the
    # concatenated positive freqs across all axes.
    pos_freqs_all = np.concatenate(pos_axis_freqs, axis=1)  # [4096, head_dim/2]
    txt_freqs = pos_freqs_all[max_vid_index : max_vid_index + n_text]

    combined = np.concatenate([*vid_freqs, txt_freqs], axis=0)  # [seq, head_dim/2]
    return _expand_qwen_rope_complex(combined)


def _precompute_qwen_rope_tables(
    axes_dim: list[int],
    h_lat: int,
    w_lat: int,
    n_text: int,
    theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute the [seq_len, head_dim] cos/sin tables in NumPy.

    Mirrors :class:`diffusers.models.transformers.transformer_qwenimage.QwenEmbedRope`
    with ``scale_rope=True``, ``frame=1``, ``max_txt_seq_len=n_text`` for
    the single-grid T2I case.
    """
    return _precompute_qwen_rope_tables_for_shapes(
        axes_dim, [(h_lat, w_lat)], n_text, theta
    )


def build_rope_3axis_engine(
    out_path: PathLike,
    *,
    axes_dim: list[int],
    h_lat: int,
    w_lat: int,
    n_text: int,
    theta: float = 10000.0,
    verbose: bool = False,
) -> Path:
    """Build a TRT engine that returns the Qwen-Image (cos, sin) RoPE tables.

    The tables are fully baked into the engine as constants -- no runtime
    inputs. Callers that need a different (h_lat, w_lat, n_text) build a
    fresh engine.

    Args:
        out_path: where to write the serialised TRT plan.
        axes_dim: per-axis head-dim budget [frame, height, width]. Must sum
            to ``head_dim``. For Qwen-Image: ``[16, 56, 56]`` (head_dim=128).
        h_lat, w_lat: packed-token grid size after patchify (each token
            covers a 2x2 latent patch).
        n_text: number of text tokens in the joint sequence.
        theta: RoPE base frequency. Qwen-Image uses 10000.0 (hardcoded in
            ``transformer_qwenimage.py:802``).

    Returns:
        Path to the written plan. The engine has two outputs
        ``rope_cos`` and ``rope_sin``, both of shape
        ``[n_text + h_lat * w_lat, sum(axes_dim)]``, fp32.
    """
    if len(axes_dim) != 3:
        raise ValueError(f"axes_dim must have 3 entries, got {axes_dim!r}")
    if any(a % 2 != 0 for a in axes_dim):
        raise ValueError(f"each axis dim must be even, got {axes_dim!r}")
    if h_lat <= 0 or w_lat <= 0 or n_text < 0:
        raise ValueError(
            f"h_lat={h_lat}, w_lat={w_lat}, n_text={n_text} must be positive"
        )
    max_vid_index = max(h_lat // 2, w_lat // 2)
    if max_vid_index + n_text > _ROPE_PRECOMPUTE_LEN:
        raise ValueError(
            "n_text + max(h_lat // 2, w_lat // 2) exceeds the 4096-row "
            "pre-compute budget; increase _ROPE_PRECOMPUTE_LEN"
        )

    cos_table, sin_table = _precompute_qwen_rope_tables(
        axes_dim, h_lat, w_lat, n_text, theta
    )

    builder, config, network = _make_builder(verbose)

    seq_len, head_dim = cos_table.shape
    cos_const = network.add_constant(
        (seq_len, head_dim), trt.Weights(cos_table)
    ).get_output(0)
    sin_const = network.add_constant(
        (seq_len, head_dim), trt.Weights(sin_table)
    ).get_output(0)

    # Pass-through identity shuffles so we can give the outputs stable names
    # (TRT requires marked outputs to be the output of some layer; you cannot
    # mark a constant directly).
    cos_sh = network.add_shuffle(cos_const)
    cos_sh.reshape_dims = (seq_len, head_dim)
    cos_out = cos_sh.get_output(0)
    cos_out.name = "rope_cos"

    sin_sh = network.add_shuffle(sin_const)
    sin_sh.reshape_dims = (seq_len, head_dim)
    sin_out = sin_sh.get_output(0)
    sin_out.name = "rope_sin"

    network.mark_output(cos_out)
    network.mark_output(sin_out)

    print(
        f"[qwen-image-dit] Building 3-axis RoPE engine "
        f"(axes_dim={axes_dim}, h_lat={h_lat}, w_lat={w_lat}, "
        f"n_text={n_text}, theta={theta}) "
        f"-> ({seq_len}, {head_dim}) x 2",
        file=sys.stderr,
    )
    return _serialize_and_write(builder, network, config, out_path, "rope_3axis")


# ---------------------------------------------------------------------------
# MMDiT joint-stream block.
#
# This implements one ``QwenImageTransformerBlock`` from diffusers
# ``transformer_qwenimage.py``. The math (verified against
# diffusers/src/diffusers/models/transformers/transformer_qwenimage.py
# at lines 473-730):
#
#   img_mod_params = Linear(SiLU(temb))                        # [B, 6*dim]
#   txt_mod_params = Linear(SiLU(temb))                        # [B, 6*dim]
#   img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)       # each [B, 3*dim]
#   txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)
#   (shift_msa, scale_msa, gate_msa) = mod1.chunk(3, dim=-1)
#   (shift_mlp, scale_mlp, gate_mlp) = mod2.chunk(3, dim=-1)
#
#   img_normed = LayerNorm(img_tokens)  *  (1 + scale_msa_img) + shift_msa_img
#   txt_normed = LayerNorm(txt_tokens)  *  (1 + scale_msa_txt) + shift_msa_txt
#
#   img_q = to_q(img_normed); img_k = to_k(img_normed); img_v = to_v(img_normed)
#   txt_q = add_q_proj(txt_normed); txt_k = add_k_proj(txt_normed); txt_v = add_v_proj(txt_normed)
#
#   # qk_norm="rms_norm" → RMSNorm over head_dim, per-head, elementwise_affine
#   img_q = norm_q(img_q.unflatten(-1, H, D));     img_k = norm_k(...)
#   txt_q = norm_added_q(txt_q.unflatten(-1, H, D)); txt_k = norm_added_k(...)
#
#   # RoPE (apply_rotary_emb_qwen, use_real=False) — pair-based rotation
#   #   x[..., 2j], x[..., 2j+1] = (x[..., 2j]*c - x[..., 2j+1]*s,
#   #                                x[..., 2j]*s + x[..., 2j+1]*c)
#   # where c, s come from `rope_cos`/`rope_sin` (interleaved-duplicated;
#   # cos[..., 2j] = cos[..., 2j+1] = real_part).
#   img_q/img_k <- RoPE(img_freqs)     txt_q/txt_k <- RoPE(txt_freqs)
#
#   # Joint attention with concat order [txt, img] along seq dim.
#   joint_q = cat([txt_q, img_q], dim=seq)
#   joint_k = cat([txt_k, img_k], dim=seq)
#   joint_v = cat([txt_v, img_v], dim=seq)
#   joint_out = SDPA(joint_q, joint_k, joint_v)
#   txt_attn_out, img_attn_out = split(joint_out, [N_txt, N_img], dim=seq)
#   img_attn_out = to_out.0(img_attn_out)
#   txt_attn_out = to_add_out(txt_attn_out)
#
#   # Gated residual.
#   hs_img  = img_tokens + gate_msa_img.unsqueeze(1) * img_attn_out
#   hs_txt  = txt_tokens + gate_msa_txt.unsqueeze(1) * txt_attn_out
#
#   # MLP step: LayerNorm + modulate(mod2) + FF (GELU-approximate "tanh").
#   img_n2   = LayerNorm(hs_img)   * (1 + scale_mlp_img) + shift_mlp_img
#   img_mlp  = Linear(GELU_tanh(Linear(img_n2)))
#   hs_img  += gate_mlp_img.unsqueeze(1) * img_mlp
#   # ... same for text via ff_context/txt_mlp ...
#
#   return (txt_out, img_out)
#
# Norms in the block are plain ``nn.LayerNorm(dim, elementwise_affine=False)``
# i.e. no learnable scale/bias on the norm itself. The modulation parameters
# come from ``img_mod.1`` / ``txt_mod.1`` (nn.Linear → 6*dim).
# ---------------------------------------------------------------------------


@dataclass
class JointBlockConfig:
    """Static configuration for one Qwen-Image MMDiT joint block.

    Defaults match ``Qwen/Qwen-Image-2512`` (hidden_size=3072, 24 heads,
    head_dim=128, MLP mult=4 → intermediate=12288).
    """

    hidden_size: int = 3072
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    intermediate_size: int = 12288  # FeedForward(dim, mult=4) → 4 * 3072.
    rms_norm_eps: float = 1e-6
    layer_norm_eps: float = 1e-6


# ---------------------------------------------------------------------------
# Internal TRT helpers (joint block only).
#
# TODO: consider promoting _add_modulate, _add_gate_residual, and
# _add_layernorm_no_affine_3d to graph_blocks now that the shape contract
# has been confirmed to hold for full-stack composition. Defer until
# end-to-end image generation works (avoid disrupting the working pipeline
# mid-build-out).
# ---------------------------------------------------------------------------

def _add_2d_constant(network, arr: np.ndarray) -> "trt.ITensor":
    """Add a small fp32 constant (e.g. eps, ones/zeros for LayerNorm, RoPE).

    Kept fp32 by design -- numerically sensitive values that we don't want
    to quantize at construction time. Callers cast to the compute dtype at
    point-of-use when needed.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return network.add_constant(arr.shape, trt.Weights(arr)).get_output(0)


def _add_linear_2d(network, x, in_dim: int, out_dim: int, w_hf: np.ndarray,
                   b_hf: "np.ndarray | None"):
    """Linear projection for [N, in_dim] input. ``w_hf`` is HF-order [out, in].

    Returns [N, out_dim]. Internally transposes weight to [in, out] and emits
    a bf16 matmul + optional bias-add. ``x`` is cast to bf16 at the boundary.
    Weights are converted fp32 -> bf16 once at build time and anchored.
    """
    x_c = _to_compute_dtype(network, x)
    w_t = np.ascontiguousarray(w_hf.T, dtype=np.float32)  # [in, out]
    w_const = _add_constant_reduced(network, (in_dim, out_dim), w_t)
    mm = network.add_matrix_multiply(
        x_c, trt.MatrixOperation.NONE, w_const, trt.MatrixOperation.NONE,
    )
    y = mm.get_output(0)
    if b_hf is not None:
        b_const = _add_constant_reduced(network, (1, out_dim), np.asarray(b_hf, dtype=np.float32))
        y = network.add_elementwise(y, b_const, trt.ElementWiseOperation.SUM).get_output(0)
    return y


def _add_layernorm_no_affine_3d(network, x, hidden_size: int, eps: float):
    """LayerNorm over last dim for [B, S, D] tensors, no learnable params.

    Input is cast to the bf16 compute dtype on the way in (so the layer's
    in/out tensors match the surrounding bf16 graph under STRONGLY_TYPED);
    the reduction itself runs in fp32 (``compute_precision=trt.float32``)
    so the LayerNorm mean/var don't lose precision -- matches what
    ``graph_ops.add_layer_norm_native`` does for flux2.
    """
    x_c = _to_compute_dtype(network, x)
    ones = _add_constant_reduced(
        network, (1, 1, hidden_size),
        np.ones((1, 1, hidden_size), dtype=np.float32),
    )
    zeros = _add_constant_reduced(
        network, (1, 1, hidden_size),
        np.zeros((1, 1, hidden_size), dtype=np.float32),
    )
    norm = network.add_normalization(x_c, ones, zeros, axesMask=1 << 2)
    norm.epsilon = float(eps)
    norm.compute_precision = trt.float32
    return norm.get_output(0)


def _add_modulate(network, x_3d, shift_2d, scale_2d, hidden_size: int):
    """x_3d * (1 + scale_2d[:, None, :]) + shift_2d[:, None, :], in bf16.

    x_3d:    [B, S, D]  (already bf16; otherwise cast at boundary)
    shift_2d, scale_2d: [B, D]  (bf16 from the modulation Linear)
    """
    # Reshape to [B, 1, D] for broadcasting over S.
    shift_3d = network.add_shuffle(shift_2d)
    shift_3d.reshape_dims = (-1, 1, hidden_size)
    scale_3d = network.add_shuffle(scale_2d)
    scale_3d.reshape_dims = (-1, 1, hidden_size)

    one_const = _add_constant_reduced(
        network, (1, 1, 1), np.ones((1, 1, 1), dtype=np.float32),
    )
    one_plus_scale = network.add_elementwise(
        scale_3d.get_output(0), one_const, trt.ElementWiseOperation.SUM
    ).get_output(0)

    x_c = _to_compute_dtype(network, x_3d)
    scaled = network.add_elementwise(
        x_c, one_plus_scale, trt.ElementWiseOperation.PROD
    ).get_output(0)
    shifted = network.add_elementwise(
        scaled, shift_3d.get_output(0), trt.ElementWiseOperation.SUM
    ).get_output(0)
    return shifted


def _add_gate_residual(network, residual_3d, gate_2d, branch_3d, hidden_size: int):
    """residual + gate.unsqueeze(1) * branch, in bf16.

    All inputs may originate from mixed dtypes; cast each operand to the
    compute dtype so the elementwise adds remain valid in a STRONGLY_TYPED
    network.
    """
    gate_3d = network.add_shuffle(gate_2d)
    gate_3d.reshape_dims = (-1, 1, hidden_size)
    branch_c = _to_compute_dtype(network, branch_3d)
    residual_c = _to_compute_dtype(network, residual_3d)
    gated = network.add_elementwise(
        gate_3d.get_output(0), branch_c, trt.ElementWiseOperation.PROD
    ).get_output(0)
    out = network.add_elementwise(
        residual_c, gated, trt.ElementWiseOperation.SUM
    ).get_output(0)
    return out


def _add_rms_norm_per_head(network, x_3d, num_heads: int, head_dim: int,
                           gamma: np.ndarray, eps: float, seq_len: int):
    """RMSNorm over head_dim for [B, S, H*D] (B is fixed in static engine).

    Reuses graph_ops.add_rms_norm_per_head, which auto-casts to fp32 for
    numerical stability when its ``dtype`` parameter is non-fp32 and casts
    back to the input dtype afterwards. The bf16 storage path is selected
    here so the in/out tensor dtype stays bf16 while the reduction runs in
    fp32 -- matches the QK-norm precision pattern of flux2/Qwen-VL.
    """
    # [B, S, H*D] -> [S, H*D] (B=1).
    sq = network.add_shuffle(x_3d)
    sq.reshape_dims = (seq_len, num_heads * head_dim)
    eps_t = _add_2d_constant(network, np.array([eps], dtype=np.float32))
    eps_scalar = network.add_shuffle(eps_t)
    eps_scalar.reshape_dims = ()
    out_2d = graph_ops.add_rms_norm_per_head(
        network, sq.get_output(0), num_heads, head_dim, gamma,
        eps_scalar.get_output(0), sequence_length=seq_len,
        dtype=np.float16,  # signal "non-fp32 in/out, fp32 compute internally"
    )
    # [S, H*D] -> [B=1, S, H*D].
    out_3d = network.add_shuffle(out_2d)
    out_3d.reshape_dims = (1, seq_len, num_heads * head_dim)
    return out_3d.get_output(0)


def _add_rope_pair(network, x_3d, cos_2d, sin_2d, num_heads: int,
                   head_dim: int, seq_len: int):
    """Apply Qwen pair-based RoPE.

    x_3d:    [B=1, S, H*D]
    cos_2d:  [S, D]  (interleaved duplicated: cos[..., 2j] = cos[..., 2j+1])
    sin_2d:  [S, D]  (similar)

    Pair-based rotation:
        x'[..., 2j]   = x[..., 2j]   * cos[..., 2j]   - x[..., 2j+1] * sin[..., 2j]
        x'[..., 2j+1] = x[..., 2j+1] * cos[..., 2j+1] + x[..., 2j]   * sin[..., 2j+1]

    Returns [B=1, S, H*D].
    """
    # [B=1, S, H*D] -> [S, H*D] -> [S, H, D] -> [S, H, D/2, 2]
    rb = network.add_shuffle(x_3d)
    rb.reshape_dims = (seq_len, num_heads, head_dim // 2, 2)
    x_pairs = rb.get_output(0)

    # x_real = x[..., 0], x_imag = x[..., 1]  (along last axis).
    x_real_sl = network.add_slice(
        x_pairs, start=(0, 0, 0, 0),
        shape=(seq_len, num_heads, head_dim // 2, 1), stride=(1, 1, 1, 1),
    )
    x_imag_sl = network.add_slice(
        x_pairs, start=(0, 0, 0, 1),
        shape=(seq_len, num_heads, head_dim // 2, 1), stride=(1, 1, 1, 1),
    )
    x_real = network.add_shuffle(x_real_sl.get_output(0))
    x_real.reshape_dims = (seq_len, num_heads, head_dim // 2)
    x_imag = network.add_shuffle(x_imag_sl.get_output(0))
    x_imag.reshape_dims = (seq_len, num_heads, head_dim // 2)

    # cos_pair / sin_pair: take every-other element from cos/sin (they are
    # interleaved-duplicated, so cos_pair[..., j] = cos[..., 2j]).
    # RoPE tables are stored as fp32 for numerical sensitivity; cast to the
    # bf16 compute dtype at point-of-use so the complex muls below stay in
    # bf16 (matches flux2_dit_builder.py).
    cos_pair_sl = network.add_slice(
        cos_2d, start=(0, 0), shape=(seq_len, head_dim // 2), stride=(1, 2),
    )
    sin_pair_sl = network.add_slice(
        sin_2d, start=(0, 0), shape=(seq_len, head_dim // 2), stride=(1, 2),
    )
    cos_pair_c = _to_compute_dtype(network, cos_pair_sl.get_output(0))
    sin_pair_c = _to_compute_dtype(network, sin_pair_sl.get_output(0))
    # Reshape cos/sin to [S, 1, D/2] for broadcast across heads.
    cos_3d = network.add_shuffle(cos_pair_c)
    cos_3d.reshape_dims = (seq_len, 1, head_dim // 2)
    sin_3d = network.add_shuffle(sin_pair_c)
    sin_3d.reshape_dims = (seq_len, 1, head_dim // 2)

    # new_real = x_real * cos - x_imag * sin
    r_cos = network.add_elementwise(
        x_real.get_output(0), cos_3d.get_output(0), trt.ElementWiseOperation.PROD,
    ).get_output(0)
    i_sin = network.add_elementwise(
        x_imag.get_output(0), sin_3d.get_output(0), trt.ElementWiseOperation.PROD,
    ).get_output(0)
    new_real = network.add_elementwise(
        r_cos, i_sin, trt.ElementWiseOperation.SUB,
    ).get_output(0)

    # new_imag = x_real * sin + x_imag * cos
    r_sin = network.add_elementwise(
        x_real.get_output(0), sin_3d.get_output(0), trt.ElementWiseOperation.PROD,
    ).get_output(0)
    i_cos = network.add_elementwise(
        x_imag.get_output(0), cos_3d.get_output(0), trt.ElementWiseOperation.PROD,
    ).get_output(0)
    new_imag = network.add_elementwise(
        r_sin, i_cos, trt.ElementWiseOperation.SUM,
    ).get_output(0)

    # Stack along the last dim: [S, H, D/2] + [S, H, D/2] -> [S, H, D/2, 2]
    nr_4d = network.add_shuffle(new_real)
    nr_4d.reshape_dims = (seq_len, num_heads, head_dim // 2, 1)
    ni_4d = network.add_shuffle(new_imag)
    ni_4d.reshape_dims = (seq_len, num_heads, head_dim // 2, 1)
    cat = network.add_concatenation([nr_4d.get_output(0), ni_4d.get_output(0)])
    cat.axis = 3

    # [S, H, D/2, 2] -> [B=1, S, H*D]
    flat = network.add_shuffle(cat.get_output(0))
    flat.reshape_dims = (1, seq_len, num_heads * head_dim)
    return flat.get_output(0)


def _add_joint_attention(
    network, q_img_3d, k_img_3d, v_img_3d,
    q_txt_3d, k_txt_3d, v_txt_3d,
    num_heads: int, head_dim: int, n_img: int, n_txt: int,
):
    """Joint attention with concat order [txt, img].

    Inputs are [B=1, S, H*D] for each stream. Returns
    (attn_txt [B=1, N_txt, H*D], attn_img [B=1, N_img, H*D]).
    """
    seq_total = n_txt + n_img
    # Concat along the sequence axis: [B=1, N_txt + N_img, H*D].
    q = network.add_concatenation([q_txt_3d, q_img_3d])
    q.axis = 1
    k = network.add_concatenation([k_txt_3d, k_img_3d])
    k.axis = 1
    v = network.add_concatenation([v_txt_3d, v_img_3d])
    v.axis = 1

    # Reshape [B=1, S, H*D] -> [B=1, S, H, D] -> [B=1, H, S, D].
    def to_4d(x_3d):
        r1 = network.add_shuffle(x_3d)
        r1.reshape_dims = (1, seq_total, num_heads, head_dim)
        r1.second_transpose = trt.Permutation([0, 2, 1, 3])
        return r1.get_output(0)

    q_4d = to_4d(q.get_output(0))
    k_4d = to_4d(k.get_output(0))
    v_4d = to_4d(v.get_output(0))

    ctx_4d = graph_ops.add_attention_core(
        network, q_4d, k_4d, v_4d, causal=False, mask=None,
        scale=float(1.0 / math.sqrt(head_dim)),
    )

    # [B=1, H, S, D] -> [B=1, S, H, D] -> [B=1, S, H*D]
    out_shuffle = network.add_shuffle(ctx_4d)
    out_shuffle.first_transpose = trt.Permutation([0, 2, 1, 3])
    out_shuffle.reshape_dims = (1, seq_total, num_heads * head_dim)
    out_3d = out_shuffle.get_output(0)

    # Split back into [B, N_txt, H*D] and [B, N_img, H*D].
    txt_sl = network.add_slice(
        out_3d, start=(0, 0, 0),
        shape=(1, n_txt, num_heads * head_dim), stride=(1, 1, 1),
    )
    img_sl = network.add_slice(
        out_3d, start=(0, n_txt, 0),
        shape=(1, n_img, num_heads * head_dim), stride=(1, 1, 1),
    )
    return txt_sl.get_output(0), img_sl.get_output(0)


def _add_mlp_block(network, x_3d, hidden_size: int, intermediate_size: int,
                   weights: Mapping[str, np.ndarray], prefix: str,
                   seq_len: int):
    """FeedForward block: Linear -> GELU(tanh) -> Linear.

    Diffusers FeedForward (gelu-approximate, mult=4, bias=True):
        net.0 = GELU(approximate="tanh"): Linear(dim, inner) followed by GELU
                in forward — but ``self.net[0]`` is the ``GELU`` module so the
                weight key is ``net.0.proj.weight``.
        net.1 = Dropout (skipped at inference)
        net.2 = Linear(inner, dim)
    """
    w1 = np.asarray(weights[f"{prefix}.net.0.proj.weight"], dtype=np.float32)  # [inner, hidden]
    b1 = np.asarray(weights[f"{prefix}.net.0.proj.bias"], dtype=np.float32)
    w2 = np.asarray(weights[f"{prefix}.net.2.weight"], dtype=np.float32)        # [hidden, inner]
    b2 = np.asarray(weights[f"{prefix}.net.2.bias"], dtype=np.float32)

    # Flatten to [S, D] for matmul, then back to [B=1, S, D'].
    flat = network.add_shuffle(x_3d)
    flat.reshape_dims = (seq_len, hidden_size)
    h = _add_linear_2d(network, flat.get_output(0), hidden_size, intermediate_size, w1, b1)
    h = graph_ops.add_gelu_tanh(network, h)
    h = _add_linear_2d(network, h, intermediate_size, hidden_size, w2, b2)
    unflat = network.add_shuffle(h)
    unflat.reshape_dims = (1, seq_len, hidden_size)
    return unflat.get_output(0)


def _add_qkv_with_norm_and_rope(
    network, x_3d, weights: Mapping[str, np.ndarray],
    *, hidden_size: int, num_heads: int, head_dim: int, seq_len: int,
    q_key: str, k_key: str, v_key: str,
    norm_q_key: str | None, norm_k_key: str | None,
    rms_eps: float,
    cos_2d, sin_2d,
):
    """Compute QKV for one stream with q-norm/k-norm + RoPE.

    Inputs are [B=1, S, D]. Returns (q_3d, k_3d, v_3d) each [B=1, S, H*D].
    Q and K get q-norm/k-norm + RoPE applied. V is passed through.
    """
    flat = network.add_shuffle(x_3d)
    flat.reshape_dims = (seq_len, hidden_size)
    x_flat = flat.get_output(0)

    def _proj(weight_key: str):
        w_hf = np.asarray(weights[f"{weight_key}.weight"], dtype=np.float32)
        b_hf = weights.get(f"{weight_key}.bias")
        b = np.asarray(b_hf, dtype=np.float32) if b_hf is not None else None
        return _add_linear_2d(network, x_flat, hidden_size, hidden_size, w_hf, b)

    q = _proj(q_key)
    k = _proj(k_key)
    v = _proj(v_key)

    # Reshape to [B=1, S, H*D] for downstream helpers.
    def to_3d(t):
        s = network.add_shuffle(t)
        s.reshape_dims = (1, seq_len, num_heads * head_dim)
        return s.get_output(0)

    q_3d = to_3d(q)
    k_3d = to_3d(k)
    v_3d = to_3d(v)

    # qk-norm: RMSNorm over head_dim with weight [head_dim].
    if norm_q_key is not None and f"{norm_q_key}.weight" in weights:
        gamma = np.asarray(weights[f"{norm_q_key}.weight"], dtype=np.float32)
        q_3d = _add_rms_norm_per_head(
            network, q_3d, num_heads, head_dim, gamma, rms_eps, seq_len,
        )
    if norm_k_key is not None and f"{norm_k_key}.weight" in weights:
        gamma = np.asarray(weights[f"{norm_k_key}.weight"], dtype=np.float32)
        k_3d = _add_rms_norm_per_head(
            network, k_3d, num_heads, head_dim, gamma, rms_eps, seq_len,
        )

    # RoPE on Q and K (V untouched).
    q_3d = _add_rope_pair(network, q_3d, cos_2d, sin_2d, num_heads, head_dim, seq_len)
    k_3d = _add_rope_pair(network, k_3d, cos_2d, sin_2d, num_heads, head_dim, seq_len)
    return q_3d, k_3d, v_3d


# ---------------------------------------------------------------------------
# Reusable joint-block graph helper.
#
# This is the inlined math for one ``QwenImageTransformerBlock``. Both
# ``build_joint_block_engine`` (single-block engine, used by the HF parity
# test) and ``build_qwen_image_dit_engine`` (full denoiser) call this.
# Single source of truth keeps the per-block math from drifting between
# the two builders.
#
# Inputs are pre-built ITensors (the caller is responsible for adding the
# network inputs / constants); the helper returns ``(img_out_3d, txt_out_3d)``.
# ---------------------------------------------------------------------------

def _add_joint_block_graph(
    network,
    *,
    img_3d,                # [B, n_img, hidden]
    txt_3d,                # [B, n_text, hidden]
    temb_2d,               # [B, hidden]
    cos_img,               # [n_img, head_dim]
    sin_img,               # [n_img, head_dim]
    cos_txt,               # [n_text, head_dim]
    sin_txt,               # [n_text, head_dim]
    weights: Mapping[str, "np.ndarray"],
    weights_prefix: str = "",
    cfg: "JointBlockConfig",
    n_img: int,
    n_text: int,
    batch_size: int = 1,
):
    """Build the math of one Qwen-Image MMDiT joint block.

    ``weights_prefix`` is prepended to every state-dict key lookup, so the
    full-denoiser builder can pass ``"transformer_blocks.0."`` and get the
    same block-local keys (``img_mod.1.weight``, etc.).

    Returns ``(img_out_3d, txt_out_3d)``, both [B, S_*, hidden].
    """
    if batch_size != 1:
        raise NotImplementedError(
            "_add_joint_block_graph currently supports batch_size=1 only"
        )
    H = cfg.num_attention_heads
    D = cfg.attention_head_dim
    dim = cfg.hidden_size

    def _w(key: str) -> np.ndarray:
        return np.asarray(weights[f"{weights_prefix}{key}"], dtype=np.float32)

    def _w_opt(key: str):
        v = weights.get(f"{weights_prefix}{key}")
        return None if v is None else np.asarray(v, dtype=np.float32)

    # Wrap weights so qkv-helper can use ``weights[key]`` directly with the
    # prefix already baked in.
    class _PrefixedWeights:
        def __init__(self, base: Mapping[str, np.ndarray], prefix: str):
            self._base = base
            self._prefix = prefix

        def __getitem__(self, key: str):
            return self._base[f"{self._prefix}{key}"]

        def get(self, key: str, default=None):
            return self._base.get(f"{self._prefix}{key}", default)

        def __contains__(self, key: str) -> bool:
            return f"{self._prefix}{key}" in self._base

    prefixed = _PrefixedWeights(weights, weights_prefix)

    # ----- AdaLN modulation: SiLU(temb) -> Linear -> 6*dim, chunk 2 then 3.
    img_mod_w = _w("img_mod.1.weight")
    img_mod_b = _w("img_mod.1.bias")
    txt_mod_w = _w("txt_mod.1.weight")
    txt_mod_b = _w("txt_mod.1.bias")

    temb_silu = graph_ops.add_silu(network, temb_2d)
    img_mod_params = _add_linear_2d(
        network, temb_silu, dim, 6 * dim, img_mod_w, img_mod_b,
    )
    txt_mod_params = _add_linear_2d(
        network, temb_silu, dim, 6 * dim, txt_mod_w, txt_mod_b,
    )

    def _six_chunks(mod_params):
        chunks = []
        for i in range(6):
            sl = network.add_slice(
                mod_params, start=(0, i * dim),
                shape=(batch_size, dim), stride=(1, 1),
            )
            chunks.append(sl.get_output(0))
        return chunks

    img_shift_msa, img_scale_msa, img_gate_msa, \
        img_shift_mlp, img_scale_mlp, img_gate_mlp = _six_chunks(img_mod_params)
    txt_shift_msa, txt_scale_msa, txt_gate_msa, \
        txt_shift_mlp, txt_scale_mlp, txt_gate_mlp = _six_chunks(txt_mod_params)

    # ----- norm1 + modulate per stream.
    img_normed = _add_layernorm_no_affine_3d(network, img_3d, dim, cfg.layer_norm_eps)
    img_modulated = _add_modulate(network, img_normed, img_shift_msa, img_scale_msa, dim)
    txt_normed = _add_layernorm_no_affine_3d(network, txt_3d, dim, cfg.layer_norm_eps)
    txt_modulated = _add_modulate(network, txt_normed, txt_shift_msa, txt_scale_msa, dim)

    # ----- QKV + qk-norm + RoPE.
    img_q, img_k, img_v = _add_qkv_with_norm_and_rope(
        network, img_modulated, prefixed,
        hidden_size=dim, num_heads=H, head_dim=D, seq_len=n_img,
        q_key="attn.to_q", k_key="attn.to_k", v_key="attn.to_v",
        norm_q_key="attn.norm_q", norm_k_key="attn.norm_k",
        rms_eps=cfg.rms_norm_eps, cos_2d=cos_img, sin_2d=sin_img,
    )
    txt_q, txt_k, txt_v = _add_qkv_with_norm_and_rope(
        network, txt_modulated, prefixed,
        hidden_size=dim, num_heads=H, head_dim=D, seq_len=n_text,
        q_key="attn.add_q_proj", k_key="attn.add_k_proj", v_key="attn.add_v_proj",
        norm_q_key="attn.norm_added_q", norm_k_key="attn.norm_added_k",
        rms_eps=cfg.rms_norm_eps, cos_2d=cos_txt, sin_2d=sin_txt,
    )

    # ----- joint attention; concat order [txt, img].
    attn_txt_3d, attn_img_3d = _add_joint_attention(
        network, img_q, img_k, img_v, txt_q, txt_k, txt_v,
        num_heads=H, head_dim=D, n_img=n_img, n_txt=n_text,
    )

    # ----- output projections (separate for each stream).
    def _out_proj(x_3d, key_suffix: str, seq_len: int):
        w = _w(f"{key_suffix}.weight")
        b = _w_opt(f"{key_suffix}.bias")
        flat = network.add_shuffle(x_3d)
        flat.reshape_dims = (seq_len, dim)
        proj = _add_linear_2d(network, flat.get_output(0), dim, dim, w, b)
        unflat = network.add_shuffle(proj)
        unflat.reshape_dims = (1, seq_len, dim)
        return unflat.get_output(0)

    img_attn_out = _out_proj(attn_img_3d, "attn.to_out.0", n_img)
    txt_attn_out = _out_proj(attn_txt_3d, "attn.to_add_out", n_text)

    # ----- gated residual (post-attention).
    hs_img = _add_gate_residual(network, img_3d, img_gate_msa, img_attn_out, dim)
    hs_txt = _add_gate_residual(network, txt_3d, txt_gate_msa, txt_attn_out, dim)

    # ----- norm2 + modulate(mod2) + MLP + gated residual.
    img_n2 = _add_layernorm_no_affine_3d(network, hs_img, dim, cfg.layer_norm_eps)
    img_mod2_out = _add_modulate(network, img_n2, img_shift_mlp, img_scale_mlp, dim)
    img_mlp_out = _add_mlp_block(
        network, img_mod2_out, dim, cfg.intermediate_size, prefixed, "img_mlp", n_img,
    )
    img_out = _add_gate_residual(network, hs_img, img_gate_mlp, img_mlp_out, dim)

    txt_n2 = _add_layernorm_no_affine_3d(network, hs_txt, dim, cfg.layer_norm_eps)
    txt_mod2_out = _add_modulate(network, txt_n2, txt_shift_mlp, txt_scale_mlp, dim)
    txt_mlp_out = _add_mlp_block(
        network, txt_mod2_out, dim, cfg.intermediate_size, prefixed, "txt_mlp", n_text,
    )
    txt_out = _add_gate_residual(network, hs_txt, txt_gate_mlp, txt_mlp_out, dim)
    return img_out, txt_out


# ---------------------------------------------------------------------------
# Public builder + loader.
# ---------------------------------------------------------------------------

def build_joint_block_engine(
    cfg: JointBlockConfig,
    weights: Mapping[str, "np.ndarray"],
    out_path: PathLike,
    *,
    batch_size: int = 1,
    n_img: int,
    n_text: int,
    verbose: bool = False,
) -> Path:
    """Build a TRT engine that performs one Qwen-Image MMDiT joint block.

    Engine inputs (fp32):
      ``img_tokens``  [batch_size, n_img,  hidden_size]
      ``txt_tokens``  [batch_size, n_text, hidden_size]
      ``temb``        [batch_size, hidden_size]
      ``rope_cos``    [n_img + n_text, head_dim]  (img tokens first, then txt)
      ``rope_sin``    [n_img + n_text, head_dim]

    Engine outputs (fp32):
      ``img_out``  [batch_size, n_img,  hidden_size]
      ``txt_out``  [batch_size, n_text, hidden_size]

    Required keys in ``weights`` (block-local names, matching the diffusers
    ``QwenImageTransformerBlock`` state-dict):

      img_mod.1.weight       [6*hidden, hidden]      img_mod.1.bias [6*hidden]
      txt_mod.1.weight       [6*hidden, hidden]      txt_mod.1.bias [6*hidden]
      attn.to_q.weight       [hidden, hidden]        attn.to_q.bias [hidden]
      attn.to_k.weight       [hidden, hidden]        attn.to_k.bias [hidden]
      attn.to_v.weight       [hidden, hidden]        attn.to_v.bias [hidden]
      attn.add_q_proj.weight [hidden, hidden]        attn.add_q_proj.bias
      attn.add_k_proj.weight [hidden, hidden]        attn.add_k_proj.bias
      attn.add_v_proj.weight [hidden, hidden]        attn.add_v_proj.bias
      attn.to_out.0.weight   [hidden, hidden]        attn.to_out.0.bias
      attn.to_add_out.weight [hidden, hidden]        attn.to_add_out.bias
      attn.norm_q.weight        [head_dim]
      attn.norm_k.weight        [head_dim]
      attn.norm_added_q.weight  [head_dim]
      attn.norm_added_k.weight  [head_dim]
      img_mlp.net.0.proj.weight   [intermediate, hidden]   img_mlp.net.0.proj.bias
      img_mlp.net.2.weight        [hidden, intermediate]   img_mlp.net.2.bias
      txt_mlp.net.0.proj.weight   [intermediate, hidden]   txt_mlp.net.0.proj.bias
      txt_mlp.net.2.weight        [hidden, intermediate]   txt_mlp.net.2.bias

    Implementation builds in fp32. Per Phase-1 spec deviation, this is
    expected to be sufficient for cosine >= 0.99 vs bf16 HF reference.

    Returns the path to the written serialized plan.
    """
    if batch_size != 1:
        raise NotImplementedError(
            "build_joint_block_engine currently supports batch_size=1 only"
        )
    H = cfg.num_attention_heads
    D = cfg.attention_head_dim
    dim = cfg.hidden_size
    if H * D != dim:
        raise ValueError(
            f"hidden_size ({dim}) != num_heads ({H}) * head_dim ({D})"
        )

    builder, config, network = _make_builder(verbose)

    img_in = network.add_input("img_tokens", trt.float32, (batch_size, n_img, dim))
    txt_in = network.add_input("txt_tokens", trt.float32, (batch_size, n_text, dim))
    temb_in = network.add_input("temb", trt.float32, (batch_size, dim))
    cos_in = network.add_input("rope_cos", trt.float32, (n_img + n_text, D))
    sin_in = network.add_input("rope_sin", trt.float32, (n_img + n_text, D))

    # The RoPE table is provided in [img | txt] order matching the RoPE
    # builder output. Split it into img and txt sub-tables.
    cos_img_sl = network.add_slice(cos_in, start=(0, 0), shape=(n_img, D), stride=(1, 1))
    sin_img_sl = network.add_slice(sin_in, start=(0, 0), shape=(n_img, D), stride=(1, 1))
    cos_txt_sl = network.add_slice(cos_in, start=(n_img, 0), shape=(n_text, D), stride=(1, 1))
    sin_txt_sl = network.add_slice(sin_in, start=(n_img, 0), shape=(n_text, D), stride=(1, 1))
    cos_img = cos_img_sl.get_output(0)
    sin_img = sin_img_sl.get_output(0)
    cos_txt = cos_txt_sl.get_output(0)
    sin_txt = sin_txt_sl.get_output(0)

    img_out, txt_out = _add_joint_block_graph(
        network,
        img_3d=img_in,
        txt_3d=txt_in,
        temb_2d=temb_in,
        cos_img=cos_img,
        sin_img=sin_img,
        cos_txt=cos_txt,
        sin_txt=sin_txt,
        weights=weights,
        weights_prefix="",
        cfg=cfg,
        n_img=n_img,
        n_text=n_text,
        batch_size=batch_size,
    )

    # Cast bf16 internal-compute results back to fp32 for the engine output
    # boundary -- C++ runtime / Python debug runner bind fp32 host buffers.
    img_out = _to_fp32(network, img_out)
    txt_out = _to_fp32(network, txt_out)
    img_out.name = "img_out"
    txt_out.name = "txt_out"
    network.mark_output(img_out)
    network.mark_output(txt_out)

    print(
        f"[qwen-image-dit] Building joint block engine "
        f"(B={batch_size}, N_img={n_img}, N_txt={n_text}, "
        f"hidden={dim}, heads={H}, head_dim={D})",
        file=sys.stderr,
    )
    return _serialize_and_write(builder, network, config, out_path, "joint_block")


def _load_safetensors_dir(model_dir: Path) -> tuple[dict[str, "object"], list]:
    """Open all *.safetensors in model_dir. Returns (reader_dict, opened_objs).

    The reader_dict maps tensor names to the reader that contains them. We
    use the ``torch`` framework so that bf16/fp16 tensors load as torch
    tensors (numpy lacks a native bf16 dtype). Callers convert to fp32
    NumPy via ``.float().cpu().numpy()`` after fetching.
    """
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise RuntimeError(
            "safetensors not installed; cannot load joint block weights"
        ) from e
    readers: dict[str, object] = {}
    opened = []
    for f in sorted(model_dir.glob("*.safetensors")):
        r = safe_open(str(f), framework="pt")
        opened.append(r)
        for key in r.keys():
            readers[key] = r
    if not readers:
        raise FileNotFoundError(
            f"No .safetensors shards found in {model_dir}"
        )
    return readers, opened


def load_joint_block_weights(
    transformer_dir: PathLike,
    *,
    block_index: int = 0,
) -> tuple[JointBlockConfig, dict[str, np.ndarray]]:
    """Load weights for ``transformer_blocks.{block_index}.*`` from
    a diffusers-format transformer checkpoint directory.

    Reads ``config.json`` to populate :class:`JointBlockConfig`.
    Returns ``(cfg, weights)`` where ``weights`` keys are stripped of the
    ``transformer_blocks.{block_index}.`` prefix.
    """
    import json
    import torch

    transformer_dir = Path(transformer_dir)
    config_path = transformer_dir / "config.json"
    with open(config_path, "r") as f:
        cfg_json = json.load(f)

    num_heads = int(cfg_json.get("num_attention_heads", 24))
    head_dim = int(cfg_json.get("attention_head_dim", 128))
    hidden = num_heads * head_dim
    cfg = JointBlockConfig(
        hidden_size=hidden,
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        # FeedForward(dim, mult=4) → 4 * hidden.
        intermediate_size=4 * hidden,
        rms_norm_eps=1e-6,
        layer_norm_eps=1e-6,
    )

    readers, opened = _load_safetensors_dir(transformer_dir)
    try:
        prefix = f"transformer_blocks.{block_index}."
        block_keys = [k for k in readers if k.startswith(prefix)]
        if not block_keys:
            raise KeyError(
                f"No tensors found under {prefix} in {transformer_dir}"
            )
        out: dict[str, np.ndarray] = {}
        for full_key in block_keys:
            local_key = full_key[len(prefix):]
            t = readers[full_key].get_tensor(full_key)
            # torch tensor (bf16 / fp16 / fp32) -> fp32 numpy.
            arr = t.detach().to(dtype=torch.float32).cpu().numpy()
            out[local_key] = np.ascontiguousarray(arr, dtype=np.float32)
    finally:
        # safe_open objects are context managers, but we manually opened
        # them; freeing the references lets them close.
        del opened
    return cfg, out


# ---------------------------------------------------------------------------
# Full MMDiT denoiser stack.
#
# Stacks 60 joint blocks plus the surrounding scaffolding from
# ``diffusers.models.transformers.transformer_qwenimage.QwenImageTransformer2DModel.forward``
# (lines 879-957 in the diffusers checkout) into one TRT engine:
#
#     hidden_states     = self.img_in(hidden_states)         # Linear [in_ch -> hidden]
#     encoder_hs        = self.txt_norm(encoder_hidden_states)  # RMSNorm over text_embed_dim
#     encoder_hs        = self.txt_in(encoder_hs)            # Linear [text_d -> hidden]
#     temb              = self.time_text_embed(timestep, ...)  # sinusoidal + MLP
#     image_rotary_emb  = self.pos_embed(img_shapes, max_txt_seq_len, device)
#     for block in self.transformer_blocks:
#         encoder_hs, hidden_states = block(hidden_states, encoder_hs, ...)
#     hidden_states     = self.norm_out(hidden_states, temb)  # AdaLayerNormContinuous
#     output            = self.proj_out(hidden_states)        # Linear [hidden -> p*p*out_ch]
#
# AdaLayerNormContinuous (diffusers/models/normalization.py:307-351):
#     emb         = self.linear(self.silu(temb))             # [B, 2*hidden]
#     scale, shift = torch.chunk(emb, 2, dim=1)              # scale FIRST, then shift
#     x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
# elementwise_affine=False -> the inner LayerNorm has no learnable params.
#
# Trace IDs: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01.
# ---------------------------------------------------------------------------


@dataclass
class QwenImageDiTConfig:
    """Configuration for the full Qwen-Image MMDiT denoiser.

    Defaults match ``Qwen/Qwen-Image-2512`` (60 joint blocks, hidden=3072,
    24 heads, head_dim=128, rope_axes_dim=[16, 56, 56], rope_theta=10000).
    """

    in_channels: int = 64
    out_channels: int = 16
    patch_size: int = 2
    hidden_size: int = 3072
    num_joint_blocks: int = 60
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    intermediate_size: int = 12288     # FeedForward(dim, mult=4) -> 4 * hidden.
    text_embed_dim: int = 3584         # joint_attention_dim (Qwen2.5-VL hidden).
    rope_axes_dim: list[int] = field(default_factory=lambda: [16, 56, 56])
    rope_theta: float = 10000.0
    timestep_embed_dim: int = 256      # sinusoidal embed width pre-MLP.
    rms_norm_eps: float = 1e-6
    layer_norm_eps: float = 1e-6
    max_image_tokens: int = 8192
    max_text_tokens: int = 1024
    guidance_embeds: bool = False      # Qwen-Image-2512: False.


def _joint_block_cfg_from(cfg: QwenImageDiTConfig) -> JointBlockConfig:
    """Adapter so the joint-block helper can be called with its own config."""
    return JointBlockConfig(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        attention_head_dim=cfg.attention_head_dim,
        intermediate_size=cfg.intermediate_size,
        rms_norm_eps=cfg.rms_norm_eps,
        layer_norm_eps=cfg.layer_norm_eps,
    )


def _add_linear_3d(network, x_3d, in_dim: int, out_dim: int, w_hf: np.ndarray,
                   b_hf: "np.ndarray | None", seq_len: int, batch_size: int = 1):
    """Linear on a [B, S, in_dim] tensor -> [B, S, out_dim].

    Flattens to [B*S, in_dim], applies the existing _add_linear_2d helper,
    then reshapes back. ``batch_size`` is bound statically.
    """
    flat = network.add_shuffle(x_3d)
    flat.reshape_dims = (batch_size * seq_len, in_dim)
    y = _add_linear_2d(network, flat.get_output(0), in_dim, out_dim, w_hf, b_hf)
    unflat = network.add_shuffle(y)
    unflat.reshape_dims = (batch_size, seq_len, out_dim)
    return unflat.get_output(0)


def _add_rms_norm_last_dim_3d(network, x_3d, hidden_size: int, gamma: np.ndarray,
                              eps: float):
    """RMSNorm over the last axis of a [B, S, D] tensor.

    Implements ``gamma * x / sqrt(mean(x^2, dim=-1, keepdim=True) + eps)``,
    matching ``diffusers.models.normalization.RMSNorm`` with
    ``elementwise_affine=True`` and no bias. ``gamma`` shape: [hidden_size].

    We don't reuse :func:`graph_ops.add_rms_norm` because that helper is
    hard-coded for 2D [N, D] inputs (reduces over axis 1).
    """
    sq = network.add_elementwise(x_3d, x_3d, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(
        sq.get_output(0), trt.ReduceOperation.AVG, 1 << 2, keep_dims=True,
    )
    eps_const = _add_2d_constant(
        network, np.array([[[eps]]], dtype=np.float32),
    )
    denom_in = network.add_elementwise(
        mean.get_output(0), eps_const, trt.ElementWiseOperation.SUM,
    )
    sqrt_l = network.add_unary(denom_in.get_output(0), trt.UnaryOperation.SQRT)
    recip = network.add_unary(sqrt_l.get_output(0), trt.UnaryOperation.RECIP)
    normalized = network.add_elementwise(
        x_3d, recip.get_output(0), trt.ElementWiseOperation.PROD,
    )
    gamma_const = _add_2d_constant(
        network, np.asarray(gamma, dtype=np.float32).reshape(1, 1, hidden_size),
    )
    scaled = network.add_elementwise(
        normalized.get_output(0), gamma_const, trt.ElementWiseOperation.PROD,
    )
    return scaled.get_output(0)


def _add_time_text_embed(
    network,
    timestep,
    *,
    weights: Mapping[str, np.ndarray],
    in_dim: int,
    hidden_size: int,
):
    """In-network port of ``QwenTimestepProjEmbeddings`` (the no-guidance path).

    timestep [B] -> sinusoidal [B, in_dim] -> linear_1 [B, hidden]
                 -> SiLU -> linear_2 [B, hidden]
    """
    sample = _add_get_timestep_embedding(
        network, timestep,
        embedding_dim=in_dim,
        flip_sin_to_cos=True,
        downscale_freq_shift=0.0,
        scale=1000.0,
        max_period=10000.0,
    )
    w1 = np.asarray(
        weights["time_text_embed.timestep_embedder.linear_1.weight"], dtype=np.float32,
    )
    b1 = np.asarray(
        weights["time_text_embed.timestep_embedder.linear_1.bias"], dtype=np.float32,
    )
    w2 = np.asarray(
        weights["time_text_embed.timestep_embedder.linear_2.weight"], dtype=np.float32,
    )
    b2 = np.asarray(
        weights["time_text_embed.timestep_embedder.linear_2.bias"], dtype=np.float32,
    )
    h = _add_linear(network, sample, w1, b1)
    h = graph_ops.add_silu(network, h)
    h = _add_linear(network, h, w2, b2)
    return h  # [B, hidden]


def _add_norm_out_3d(
    network,
    x_3d,
    temb_2d,
    *,
    weights: Mapping[str, np.ndarray],
    hidden_size: int,
    eps: float,
    batch_size: int,
):
    """Apply ``AdaLayerNormContinuous(elementwise_affine=False)`` for [B, S, D].

    Matches diffusers.models.normalization.AdaLayerNormContinuous.forward
    (normalization.py:346-351):
        emb         = self.linear(self.silu(temb))         # [B, 2*D]
        scale, shift = torch.chunk(emb, 2, dim=1)          # scale FIRST.
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
    where ``self.norm`` is ``LayerNorm(D, eps, elementwise_affine=False)``.
    """
    w = np.asarray(weights["norm_out.linear.weight"], dtype=np.float32)  # [2*D, D]
    b = np.asarray(weights["norm_out.linear.bias"], dtype=np.float32)    # [2*D]
    temb_silu = graph_ops.add_silu(network, temb_2d)
    emb = _add_linear_2d(network, temb_silu, hidden_size, 2 * hidden_size, w, b)
    # chunk(2, dim=1) -> (scale, shift).  diffusers convention is scale-first.
    scale_sl = network.add_slice(
        emb, start=(0, 0),
        shape=(batch_size, hidden_size), stride=(1, 1),
    )
    shift_sl = network.add_slice(
        emb, start=(0, hidden_size),
        shape=(batch_size, hidden_size), stride=(1, 1),
    )
    scale = scale_sl.get_output(0)
    shift = shift_sl.get_output(0)

    # LayerNorm(x), no affine.
    x_normed = _add_layernorm_no_affine_3d(network, x_3d, hidden_size, eps)
    # (1 + scale)[:, None, :] and shift[:, None, :]
    scale_3d = network.add_shuffle(scale)
    scale_3d.reshape_dims = (-1, 1, hidden_size)
    shift_3d = network.add_shuffle(shift)
    shift_3d.reshape_dims = (-1, 1, hidden_size)
    # bf16 constant so it matches scale_3d's dtype under STRONGLY_TYPED.
    one_const = _add_constant_reduced(
        network, (1, 1, 1), np.ones((1, 1, 1), dtype=np.float32),
    )
    one_plus_scale = network.add_elementwise(
        scale_3d.get_output(0), one_const, trt.ElementWiseOperation.SUM,
    ).get_output(0)
    scaled = network.add_elementwise(
        x_normed, one_plus_scale, trt.ElementWiseOperation.PROD,
    ).get_output(0)
    shifted = network.add_elementwise(
        scaled, shift_3d.get_output(0), trt.ElementWiseOperation.SUM,
    ).get_output(0)
    return shifted


def _validate_full_weights(
    cfg: QwenImageDiTConfig, weights: Mapping[str, np.ndarray],
) -> None:
    """Lightweight schema check on the weight dict.

    Only checks presence (and shapes for the top-level params). Per-block
    keys are checked by the joint-block helper at build time.
    """
    H = cfg.hidden_size
    in_ch = cfg.in_channels
    out_ch = cfg.out_channels
    p = cfg.patch_size
    txt_d = cfg.text_embed_dim

    required = {
        "img_in.weight": (H, in_ch),
        "img_in.bias": (H,),
        "txt_norm.weight": (txt_d,),
        "txt_in.weight": (H, txt_d),
        "txt_in.bias": (H,),
        "time_text_embed.timestep_embedder.linear_1.weight": (H, cfg.timestep_embed_dim),
        "time_text_embed.timestep_embedder.linear_1.bias": (H,),
        "time_text_embed.timestep_embedder.linear_2.weight": (H, H),
        "time_text_embed.timestep_embedder.linear_2.bias": (H,),
        "norm_out.linear.weight": (2 * H, H),
        "norm_out.linear.bias": (2 * H,),
        "proj_out.weight": (out_ch * p * p, H),
        "proj_out.bias": (out_ch * p * p,),
    }
    for key, want in required.items():
        if key not in weights:
            raise KeyError(f"build_qwen_image_dit_engine: missing weight {key!r}")
        arr = np.asarray(weights[key])
        if tuple(arr.shape) != tuple(want):
            raise ValueError(
                f"build_qwen_image_dit_engine: {key!r} shape {arr.shape} != {want}"
            )


def build_qwen_image_dit_engine(
    cfg: QwenImageDiTConfig,
    weights: Mapping[str, "np.ndarray"],
    out_path: PathLike,
    *,
    h_lat: int,
    w_lat: int,
    n_text: int,
    image_token_shapes: list[tuple[int, int]] | None = None,
    batch_size: int = 1,
    max_batch_size: int = 1,
    opt_batch_size: int | None = None,
    verbose: bool = False,
) -> Path:
    """Build the full Qwen-Image MMDiT denoiser as a single TRT engine.

    The denoiser stacks ``cfg.num_joint_blocks`` ``QwenImageTransformerBlock``
    instances plus the surrounding scaffolding (``img_in``, ``txt_norm``,
    ``txt_in``, ``time_text_embed``, final ``AdaLayerNormContinuous``,
    ``proj_out``).

    Engine inputs (fp32):
      ``img_patched`` [B, N_img, in_channels]
          packed-patch latents (already produced by the patchify engine /
          diffusers pack_latents). For Qwen-Image: in_channels = 64.
          T2I uses ``N_img = h_lat * w_lat``. Edit may pass
          ``image_token_shapes=[(gen_h, gen_w), (cond_h, cond_w), ...]``;
          then ``N_img`` is the sum of those packed-token grids, matching
          diffusers' ``torch.cat([latents, image_latents], dim=1)`` input.
      ``txt_hidden`` [B, N_txt = n_text, text_embed_dim]
          text encoder hidden states (Qwen2.5-VL hidden = 3584).
      ``timestep``  [B]
          diffusion timestep (fp32; will be cast/multiplied by ``scale=1000``
          inside the sinusoidal embedding, matching diffusers).

    When ``max_batch_size == 1`` every input has a statically baked leading
    dim of 1 (byte-for-byte identical to the pre-batch build). When
    ``max_batch_size > 1`` the leading dim is replaced with ``-1`` and a
    TensorRT optimization profile attaches with
    ``kMIN=1, kOPT=opt_batch_size, kMAX=max_batch_size`` per design Decisions
    A and C (2026-05-19).

    Engine output (fp32):
      ``noise_patched`` [B, N_img, out_channels * patch_size**2]
          packed-patch noise prediction. The runtime unpatchifies this
          before handing it to the VAE decoder.

    ``h_lat`` and ``w_lat`` describe the packed-token grid (= latent-grid
    divided by ``cfg.patch_size``). They bake into the engine via the RoPE
    tables, so callers wanting a different grid must build a fresh engine.
    For Edit, the first ``image_token_shapes`` entry must match
    ``(h_lat, w_lat)`` and represents the generated-image tokens; subsequent
    entries represent condition-image VAE latents.

    Real Qwen-Image-2512 weights load via the diffusers state-dict keys
    (no remapping); see ``_validate_full_weights`` for the full key list.

    Args:
        batch_size: Legacy static-batch knob retained for backwards
            compatibility with existing callers that drove a fixed B>1
            engine. Prefer ``max_batch_size`` for the dynamic-profile path.
            When both default to 1 the build is byte-for-byte identical to
            today's behavior.
        max_batch_size: Maximum DiT batch the engine should accept. Drives
            the dynamic optimization profile when ``>1`` (Decisions A and C).
            Per Decision C the Qwen-Image DiT cap is 4.
        opt_batch_size: ``kOPT`` for the dynamic batch dim. Ignored when
            ``max_batch_size==1``. Defaults to ``min(max_batch_size, 4)``.

    Returns the path to the written serialised plan.
    """
    if max_batch_size < 1:
        raise ValueError(
            f"max_batch_size must be >= 1 (got {max_batch_size})"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1 (got {batch_size})")
    if cfg.num_attention_heads * cfg.attention_head_dim != cfg.hidden_size:
        raise ValueError(
            f"hidden_size ({cfg.hidden_size}) != num_heads ({cfg.num_attention_heads}) "
            f"* head_dim ({cfg.attention_head_dim})"
        )
    if sum(cfg.rope_axes_dim) != cfg.attention_head_dim:
        raise ValueError(
            f"sum(rope_axes_dim) ({sum(cfg.rope_axes_dim)}) != head_dim "
            f"({cfg.attention_head_dim})"
        )
    if cfg.guidance_embeds:
        raise NotImplementedError(
            "build_qwen_image_dit_engine does not support guidance_embeds=True"
        )

    _validate_full_weights(cfg, weights)

    if image_token_shapes is None:
        image_token_shapes = [(h_lat, w_lat)]
    else:
        image_token_shapes = [(int(h), int(w)) for h, w in image_token_shapes]
        if not image_token_shapes:
            raise ValueError("image_token_shapes must not be empty")
        if image_token_shapes[0] != (h_lat, w_lat):
            raise ValueError(
                "image_token_shapes[0] must match (h_lat, w_lat); got "
                f"{image_token_shapes[0]!r} vs {(h_lat, w_lat)!r}"
            )
    n_img = sum(h * w for h, w in image_token_shapes)
    H_dim = cfg.hidden_size
    head_dim = cfg.attention_head_dim
    in_ch = cfg.in_channels
    out_ch = cfg.out_channels
    p = cfg.patch_size
    txt_d = cfg.text_embed_dim

    # Pre-compute RoPE tables (baked as constants).
    cos_table_np, sin_table_np = _precompute_qwen_rope_tables_for_shapes(
        list(cfg.rope_axes_dim), image_token_shapes, n_text, cfg.rope_theta,
    )
    seq_total = n_img + n_text
    assert cos_table_np.shape == (seq_total, head_dim), (
        f"rope cos shape {cos_table_np.shape} != ({seq_total}, {head_dim})"
    )

    builder, config, network = _make_builder(verbose)

    # When ``max_batch_size > 1`` we expose a dynamic leading batch dim via a
    # TensorRT optimization profile (design Decision A: one wide profile per
    # component). Otherwise the engine remains statically batched at
    # ``batch_size`` (today's behavior). The internal compute graph still
    # uses the static ``batch_size`` everywhere it bakes a shape — full
    # dynamic-batch correctness for the inner blocks is a Phase 1.5 follow-up
    # (see RFC 2026-05-11). The profile call below is the source of truth
    # for the engine's batch envelope.
    use_dynamic_batch = max_batch_size > 1
    input_batch = -1 if use_dynamic_batch else batch_size

    img_patched = network.add_input(
        "img_patched", trt.float32, (input_batch, n_img, in_ch),
    )
    txt_hidden = network.add_input(
        "txt_hidden", trt.float32, (input_batch, n_text, txt_d),
    )
    timestep = network.add_input("timestep", trt.float32, (input_batch,))

    if use_dynamic_batch:
        from ...engine_builder import add_dynamic_batch_profile

        opt_batch = (
            min(max_batch_size, 4) if opt_batch_size is None else opt_batch_size
        )
        add_dynamic_batch_profile(
            builder,
            config,
            network,
            input_names=["img_patched", "txt_hidden", "timestep"],
            max_batch=max_batch_size,
            opt_batch=opt_batch,
            static_shape={
                "img_patched": (n_img, in_ch),
                "txt_hidden": (n_text, txt_d),
                "timestep": (),
            },
        )

    # ----- img_in: Linear(in_ch -> hidden) over [B, N_img, in_ch].
    img_in_w = np.asarray(weights["img_in.weight"], dtype=np.float32)
    img_in_b = np.asarray(weights["img_in.bias"], dtype=np.float32)
    img_tokens = _add_linear_3d(
        network, img_patched, in_ch, H_dim, img_in_w, img_in_b,
        seq_len=n_img, batch_size=batch_size,
    )

    # ----- txt_norm: RMSNorm over text_embed_dim, then txt_in: Linear(text_d -> hidden).
    txt_norm_gamma = np.asarray(weights["txt_norm.weight"], dtype=np.float32)
    txt_normed = _add_rms_norm_last_dim_3d(
        network, txt_hidden, txt_d, txt_norm_gamma, cfg.rms_norm_eps,
    )
    txt_in_w = np.asarray(weights["txt_in.weight"], dtype=np.float32)
    txt_in_b = np.asarray(weights["txt_in.bias"], dtype=np.float32)
    txt_tokens = _add_linear_3d(
        network, txt_normed, txt_d, H_dim, txt_in_w, txt_in_b,
        seq_len=n_text, batch_size=batch_size,
    )

    # ----- time_text_embed: sinusoidal + MLP -> [B, hidden].
    temb = _add_time_text_embed(
        network, timestep, weights=weights,
        in_dim=cfg.timestep_embed_dim, hidden_size=H_dim,
    )

    # ----- RoPE cos/sin constants split into img and txt sub-tables.
    # _precompute_qwen_rope_tables returns [vid_freqs (n_img rows) | txt_freqs (n_text rows)].
    cos_const = network.add_constant(
        (seq_total, head_dim), trt.Weights(cos_table_np),
    ).get_output(0)
    sin_const = network.add_constant(
        (seq_total, head_dim), trt.Weights(sin_table_np),
    ).get_output(0)
    cos_img_sl = network.add_slice(cos_const, start=(0, 0),
                                   shape=(n_img, head_dim), stride=(1, 1))
    sin_img_sl = network.add_slice(sin_const, start=(0, 0),
                                   shape=(n_img, head_dim), stride=(1, 1))
    cos_txt_sl = network.add_slice(cos_const, start=(n_img, 0),
                                   shape=(n_text, head_dim), stride=(1, 1))
    sin_txt_sl = network.add_slice(sin_const, start=(n_img, 0),
                                   shape=(n_text, head_dim), stride=(1, 1))

    # ----- Joint blocks loop.
    jb_cfg = _joint_block_cfg_from(cfg)
    cur_img = img_tokens
    cur_txt = txt_tokens
    for i in range(cfg.num_joint_blocks):
        prefix = f"transformer_blocks.{i}."
        cur_img, cur_txt = _add_joint_block_graph(
            network,
            img_3d=cur_img,
            txt_3d=cur_txt,
            temb_2d=temb,
            cos_img=cos_img_sl.get_output(0),
            sin_img=sin_img_sl.get_output(0),
            cos_txt=cos_txt_sl.get_output(0),
            sin_txt=sin_txt_sl.get_output(0),
            weights=weights,
            weights_prefix=prefix,
            cfg=jb_cfg,
            n_img=n_img,
            n_text=n_text,
            batch_size=batch_size,
        )

    # ----- AdaLayerNormContinuous(elementwise_affine=False) -> proj_out.
    normed = _add_norm_out_3d(
        network, cur_img, temb, weights=weights,
        hidden_size=H_dim, eps=cfg.layer_norm_eps, batch_size=batch_size,
    )
    proj_w = np.asarray(weights["proj_out.weight"], dtype=np.float32)
    proj_b = np.asarray(weights["proj_out.bias"], dtype=np.float32)
    proj_out_dim = out_ch * p * p
    noise = _add_linear_3d(
        network, normed, H_dim, proj_out_dim, proj_w, proj_b,
        seq_len=n_img, batch_size=batch_size,
    )
    # Engine output is fp32 by contract (the C++ runtime + Python debug
    # runner bind fp32 host buffers); internal compute stays bf16.
    noise = _to_fp32(network, noise)
    noise.name = "noise_patched"
    network.mark_output(noise)

    b_label = (
        f"1..{max_batch_size}" if use_dynamic_batch else str(batch_size)
    )
    print(
        f"[qwen-image-dit] Building full denoiser engine "
        f"(B={b_label}, n_img={n_img}, n_text={n_text}, "
        f"image_token_shapes={image_token_shapes}, "
        f"blocks={cfg.num_joint_blocks}, hidden={H_dim}, "
        f"heads={cfg.num_attention_heads}, head_dim={head_dim}, "
        f"text_d={txt_d}, in_ch={in_ch}, out_ch={out_ch}, p={p}) "
        f"-> [{b_label}, {n_img}, {proj_out_dim}]",
        file=sys.stderr,
    )
    return _serialize_and_write(builder, network, config, out_path, "qwen_image_dit")


# ---------------------------------------------------------------------------
# Full-denoiser HF weight loader.
#
# Loads every parametric tensor from a diffusers Qwen-Image-2512 transformer/
# directory and packages it together with a QwenImageDiTConfig built from the
# component config.json. The returned dict's keys are the diffusers
# state-dict keys exactly (no renaming) so it can be passed straight into
# build_qwen_image_dit_engine -- which uses _validate_full_weights to assert
# the schema matches the diffusers convention.
# ---------------------------------------------------------------------------


def load_qwen_image_dit_weights(
    transformer_dir: PathLike,
) -> tuple[QwenImageDiTConfig, dict[str, np.ndarray]]:
    """Load full ``QwenImageTransformer2DModel`` weights from a HF ``transformer/`` dir.

    Reads ``transformer/config.json`` to populate :class:`QwenImageDiTConfig`,
    then scans all ``*.safetensors`` shards in the directory and returns
    every tensor as an fp32 ``np.ndarray``. The returned ``cfg`` and
    ``weights`` dict are suitable for direct use with
    :func:`build_qwen_image_dit_engine` — no key remapping is needed
    because the engine builder uses the diffusers state-dict naming
    convention throughout (see :func:`_validate_full_weights`).

    The full transformer dict for Qwen/Qwen-Image-2512 has 1933 parametric
    tensors (verified by listing the safetensors index).
    Returns ALL of them — the full denoiser uses everything (img_in,
    txt_norm, txt_in, time_text_embed, transformer_blocks.*, norm_out,
    proj_out). No filtering is applied because the transformer directory
    only contains the denoiser; sibling components (text encoder, VAE) live
    in their own per-component directories under the diffusers snapshot.

    The Qwen-Image-2512 ``config.json`` is minimal (no ``rope_theta``,
    ``rms_norm_eps``, etc. fields exist); this loader uses the same
    hardcoded defaults that the diffusers source bakes in:
      - ``rope_theta = 10000.0`` (transformer_qwenimage.py:200 default).
      - ``rms_norm_eps = 1e-6`` (QwenImageTransformerBlock default).
      - ``timestep_embed_dim = 256`` (QwenTimestepProjEmbeddings default).
      - ``max_text_tokens = 1024`` (Qwen-Image pipeline default).
      - ``max_image_tokens = 8192`` (Qwen-Image pipeline ``max_image_seq_len``).

    Args:
        transformer_dir: Path to a directory containing ``config.json`` and
            one or more ``*.safetensors`` shards in the HF/diffusers layout.
            This is typically ``<repo>/transformer/`` under a diffusers
            snapshot of Qwen/Qwen-Image-2512.

    Returns:
        Tuple of ``(config, weights)`` where the keys of ``weights`` are the
        diffusers state-dict keys (e.g. ``img_in.weight``,
        ``transformer_blocks.0.attn.to_q.weight``), each cast to fp32
        ``np.ndarray``.

    Raises:
        FileNotFoundError: If ``config.json`` or any ``*.safetensors`` shard
            is missing from ``transformer_dir``.
    """
    import json

    from safetensors import safe_open

    try:
        # Required when safetensors emits a bf16 numpy view that needs
        # ml_dtypes registered before .astype() can decode it. Qwen-Image's
        # transformer shards are bf16-stored, so without this the
        # safe_open(..., framework="numpy") call below cannot return the
        # tensors (numpy has no native bf16 dtype).
        import ml_dtypes  # noqa: F401
    except ImportError:  # pragma: no cover -- best-effort
        pass

    text_dir = Path(transformer_dir)
    config_path = text_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {text_dir}")
    cfg_json = json.loads(config_path.read_text())

    num_attention_heads = int(cfg_json.get("num_attention_heads", 24))
    attention_head_dim = int(cfg_json.get("attention_head_dim", 128))
    hidden_size = num_attention_heads * attention_head_dim

    cfg = QwenImageDiTConfig(
        in_channels=int(cfg_json.get("in_channels", 64)),
        out_channels=int(cfg_json.get("out_channels", 16)),
        patch_size=int(cfg_json.get("patch_size", 2)),
        hidden_size=hidden_size,
        num_joint_blocks=int(cfg_json.get("num_layers", 60)),
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        intermediate_size=hidden_size * 4,  # FeedForward(dim, mult=4) -> 4*hidden.
        text_embed_dim=int(cfg_json.get("joint_attention_dim", 3584)),
        rope_axes_dim=list(cfg_json.get("axes_dims_rope", [16, 56, 56])),
        rope_theta=10000.0,  # hardcoded in diffusers source
        timestep_embed_dim=256,  # hardcoded in QwenTimestepProjEmbeddings
        rms_norm_eps=1e-6,
        layer_norm_eps=1e-6,
        max_image_tokens=8192,
        max_text_tokens=1024,
        guidance_embeds=bool(cfg_json.get("guidance_embeds", False)),
    )

    safetensor_files = sorted(text_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No *.safetensors in {text_dir}")

    weights: dict[str, np.ndarray] = {}
    for sf in safetensor_files:
        with safe_open(str(sf), framework="numpy") as f:
            for key in f.keys():
                arr = f.get_tensor(key)
                # Promote any low-precision dtype (bf16/fp16) to fp32. The
                # builder accepts fp32 only via _as_numpy().
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                weights[key] = arr

    return cfg, weights
