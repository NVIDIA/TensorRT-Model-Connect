"""Cosmos-Predict2 14B Video2World DiT engine builder (DeepSpeed-Ulysses SP).

Sequence-parallel variant of :mod:`cosmos_dit_builder`. Each rank holds
``num_patches / cp_size`` patches but *all* attention heads. Inside every
self-attention block we run two AllToAll collectives that redistribute the
per-rank tensors from "all heads, local seq" to "local heads, all seq" and
back, so the attention math itself is computed on the full sequence with a
subset of heads. All other ops (norms, FFN, cross-attention, output head)
operate on the seq-sharded tensor directly because:

* The block-wise norms / modulation / FFN are pointwise along the sequence
  axis, so they parallelize trivially across ``cp_size`` ranks.
* Cross-attention queries are already seq-sharded; K/V come from
  ``encoder_hidden_states`` which is replicated everywhere (T5 text embedding
  is broadcast pre-engine).
* The final norm_out + proj_out head is pointwise along the sequence axis.

Per-rank engine I/O contract::

    hidden_states         [num_patches / cp_size, hidden_size]            fp32
    encoder_hidden_states [text_seq_len, text_embed_dim]                  fp32  (replicated)
    temb                  [1, hidden_size]                                fp32  (replicated)
    embedded_timestep     [1, hidden_size]                                fp32  (replicated)

    noise_pred            [num_patches / cp_size, out_channels * pt*ph*pw] fp32

The runner (Python wrapper / C++ host) is responsible for the final AllGather
on ``noise_pred`` that reassembles the full denoised tensor before the
unpatchify reshape.

Validity constraints:

* ``num_attention_heads % cp_size == 0`` (40 heads -> cp_size in {1, 2, 4, 8}).
* ``num_patches % cp_size == 0`` (720x1280@49f -> 46800 patches; clean for
  cp_size in {1, 2, 4, 8}).

At ``cp_size == 1`` the builder degenerates into a graph that is structurally
equivalent to :func:`cosmos_dit_builder.build_cosmos_dit_engine` (the
collective helpers are pass-throughs and the per-rank position slice covers
the full sequence), so the engine can be used as a single-rank dry-run gate.

Weight loading is shared with the dense builder via
:func:`cosmos_dit_builder.load_cosmos_dit_weights` -- the engine plan bytes
are identical across ranks; only the per-rank ``cos``/``sin`` RoPE constants
and the I/O shape differ at build time.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from tensorrt_model_connect import trt_compat

from ... import graph_ops
from ...parallel_config import (
    ParallelConfig,
    add_all_to_all,
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)

from . import cosmos_dit_builder
from .cosmos_dit_builder import (
    COSMOS_14B_ADALN_LORA_DIM,
    COSMOS_14B_FFN_DIM,
    COSMOS_14B_HEAD_DIM,
    COSMOS_14B_HIDDEN_SIZE,
    COSMOS_14B_NORM_EPS,
    COSMOS_14B_NUM_HEADS,
    COSMOS_14B_NUM_LAYERS,
    COSMOS_14B_OUT_CHANNELS,
    COSMOS_14B_PATCH_SIZE,
    COSMOS_14B_ROPE_AXES_DIM,
    COSMOS_14B_ROPE_SCALE,
    COSMOS_14B_ROPE_THETA,
    COSMOS_14B_TEXT_EMBED_DIM,
    COSMOS_14B_TEXT_SEQ_LEN,
    COSMOS_14B_VAE_SCALE_SPATIAL,
    COSMOS_14B_VAE_SCALE_TEMPORAL,
    _adaln_final,
    _adaln_zero,
    _block_weights_or_none,
    _rope_table_axis,
    _silu_2d,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# ---------------------------------------------------------------------------
# Per-rank 3-axis RoPE table
#
# The dense builder bakes cos/sin tables of shape [num_patches, head_dim/2]
# from the full (T, H, W) grid. For Ulysses we only need the slice of rows
# this rank's local sequence covers, but the head_dim axis is unchanged
# (heads are split *after* RoPE, by the AllToAll).
#
# Layout reminder: the dense ``_build_3axis_rope_tables`` walks the 3-D grid
# in (T, H, W) order with the H, W axes inside T. Slicing rows
# [rank*L, (rank+1)*L) therefore yields the cos/sin pairs whose linearised
# positions match the local hidden_states shard the rank receives.
# ---------------------------------------------------------------------------


def _build_3axis_rope_tables_per_rank(
    *,
    t_patches: int,
    h_patches: int,
    w_patches: int,
    axes_dim: tuple[int, int, int],
    rope_scale: tuple[float, float, float],
    rope_theta: float,
    cp_size: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-rank ``cos, sin`` of shape ``[num_patches / cp_size, head_dim // 2]``.

    The full 3-axis tables are computed exactly as in the dense builder
    (so cp_size=1 is bit-identical), then sliced along axis 0 to keep only
    the rows ``[rank * L, (rank + 1) * L)`` where ``L = num_patches / cp_size``.
    """
    ad_t, ad_h, ad_w = axes_dim
    cos_t, sin_t = _rope_table_axis(
        num_positions=t_patches, axis_dim=ad_t,
        rope_theta=rope_theta, scale=rope_scale[0])
    cos_h, sin_h = _rope_table_axis(
        num_positions=h_patches, axis_dim=ad_h,
        rope_theta=rope_theta, scale=rope_scale[1])
    cos_w, sin_w = _rope_table_axis(
        num_positions=w_patches, axis_dim=ad_w,
        rope_theta=rope_theta, scale=rope_scale[2])

    cos_t_grid = np.broadcast_to(
        cos_t[:, None, None, :], (t_patches, h_patches, w_patches, ad_t // 2))
    cos_h_grid = np.broadcast_to(
        cos_h[None, :, None, :], (t_patches, h_patches, w_patches, ad_h // 2))
    cos_w_grid = np.broadcast_to(
        cos_w[None, None, :, :], (t_patches, h_patches, w_patches, ad_w // 2))
    sin_t_grid = np.broadcast_to(
        sin_t[:, None, None, :], (t_patches, h_patches, w_patches, ad_t // 2))
    sin_h_grid = np.broadcast_to(
        sin_h[None, :, None, :], (t_patches, h_patches, w_patches, ad_h // 2))
    sin_w_grid = np.broadcast_to(
        sin_w[None, None, :, :], (t_patches, h_patches, w_patches, ad_w // 2))

    num_patches = t_patches * h_patches * w_patches
    cos_full = np.concatenate(
        [cos_t_grid, cos_h_grid, cos_w_grid], axis=-1).reshape(
            num_patches, -1).astype(np.float32)
    sin_full = np.concatenate(
        [sin_t_grid, sin_h_grid, sin_w_grid], axis=-1).reshape(
            num_patches, -1).astype(np.float32)

    if cp_size <= 1:
        return (np.ascontiguousarray(cos_full),
                np.ascontiguousarray(sin_full))

    if num_patches % cp_size != 0:
        raise ValueError(
            f"num_patches={num_patches} must be divisible by cp_size={cp_size}")
    local_len = num_patches // cp_size
    start = rank * local_len
    end = start + local_len
    cos_local = cos_full[start:end, :]
    sin_local = sin_full[start:end, :]
    return (np.ascontiguousarray(cos_local),
            np.ascontiguousarray(sin_local))


# ---------------------------------------------------------------------------
# Self-attention block (Ulysses variant)
# ---------------------------------------------------------------------------


def _self_attention_ulysses(
    network: "trt.INetworkDefinition",
    h1: "trt.ITensor",
    *,
    bw: dict,
    num_heads: int,
    head_dim: int,
    hidden_size: int,
    local_num_patches: int,
    full_num_patches: int,
    cp_size: int,
    cos_const: "trt.ITensor",
    sin_const: "trt.ITensor",
    eps_t: "trt.ITensor",
    tag: str,
) -> "trt.ITensor":
    """One Ulysses self-attention forward.

    Inputs:
        h1: AdaLN-modulated activation, shape ``[L, hidden_size]`` where
            ``L = full_num_patches / cp_size``.
        cos_const/sin_const: per-rank 3-axis RoPE tables of shape
            ``[1, L, head_dim // 2]``.

    Output:
        Pre-residual attention output of shape ``[L, hidden_size]`` (i.e.
        the result of ``to_out.0(softmax(Q K^T / sqrt(d)) V)``) ready for
        the gated residual add.

    AllToAll layout (Q/K/V are row-major ``[Sq, num_heads * head_dim]``):

    * Forward direction (heads scatter / seq gather):
        input  ``[L, num_heads * head_dim]``
            scatter_axis = 1 (hidden / heads axis)
            gather_axis  = 0 (sequence axis)
        output ``[full_num_patches, (num_heads // cp_size) * head_dim]``

    * Backward direction (seq scatter / heads gather), after the local
      attention has been computed on the full sequence with a head subset:
        input  ``[full_num_patches, (num_heads // cp_size) * head_dim]``
            scatter_axis = 0 (sequence axis)
            gather_axis  = 1 (hidden / heads axis)
        output ``[L, num_heads * head_dim]``
    """
    local_num_heads = num_heads // cp_size

    # Q/K/V projections from the seq-sharded input. Shape [L, hidden].
    q = graph_ops.add_matmul_rhs_constant(
        network, h1, hidden_size, hidden_size, bw["attn1.to_q.weight"])
    k = graph_ops.add_matmul_rhs_constant(
        network, h1, hidden_size, hidden_size, bw["attn1.to_k.weight"])
    v = graph_ops.add_matmul_rhs_constant(
        network, h1, hidden_size, hidden_size, bw["attn1.to_v.weight"])

    # Per-head RMSNorm on Q and K (still per-all-40-heads — heads are not
    # split yet; the AllToAll comes *after* RMSNorm + RoPE).
    nq = bw["attn1.norm_q.weight"]
    if nq is not None:
        q = graph_ops.add_rms_norm_per_head(
            network, q, num_heads, head_dim, nq, eps_t,
            sequence_length=local_num_patches)
    nk = bw["attn1.norm_k.weight"]
    if nk is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_heads, head_dim, nk, eps_t,
            sequence_length=local_num_patches)

    # 3-axis RoPE on Q and K using per-rank cos/sin tables (the table rows
    # already encode this rank's local position slice).
    q = graph_ops.add_apply_rope_native_sequence(
        network, q,
        num_heads=num_heads, head_dim=head_dim,
        cos_cache_3d=cos_const, sin_cache_3d=sin_const,
        rotary_embedding_dim=head_dim,
        interleaved=False,
        sequence_length=local_num_patches,
    )
    k = graph_ops.add_apply_rope_native_sequence(
        network, k,
        num_heads=num_heads, head_dim=head_dim,
        cos_cache_3d=cos_const, sin_cache_3d=sin_const,
        rotary_embedding_dim=head_dim,
        interleaved=False,
        sequence_length=local_num_patches,
    )

    # AllToAll #1: scatter heads (axis 1), gather sequence (axis 0).
    # Each rank ends up with the FULL sequence on its local head subset.
    if cp_size > 1:
        q = add_all_to_all(network, q, cp_size, scatter_axis=1, gather_axis=0)
        k = add_all_to_all(network, k, cp_size, scatter_axis=1, gather_axis=0)
        v = add_all_to_all(network, v, cp_size, scatter_axis=1, gather_axis=0)

    # Attention on the FULL sequence with the LOCAL head subset.
    ctx = graph_ops.add_attention_from_rows(
        network, q, k, v,
        num_heads=local_num_heads,
        head_dim=head_dim,
        q_seq=full_num_patches, kv_seq=full_num_patches,
        tag=f"{tag}.attn1")

    # AllToAll #2: scatter sequence (axis 0), gather heads (axis 1).
    # Restore the [L, hidden] row-major layout before the output projection.
    if cp_size > 1:
        ctx = add_all_to_all(
            network, ctx, cp_size, scatter_axis=0, gather_axis=1)

    # Output projection on the seq-sharded, full-head tensor.
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, ctx, hidden_size, hidden_size,
        bw["attn1.to_out.0.weight"])
    return attn_out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_cosmos_dit_ulysses_engine(
    config,
    weights: "WeightDict",
    *,
    video_height: int,
    video_width: int,
    video_num_frames: int,
    precision: str = "fp16",
    verbose: bool = False,
    parallel_config: ParallelConfig,
    hidden_size: int = COSMOS_14B_HIDDEN_SIZE,
    num_heads: int = COSMOS_14B_NUM_HEADS,
    head_dim: int = COSMOS_14B_HEAD_DIM,
    num_layers: int = COSMOS_14B_NUM_LAYERS,
    ffn_dim: int = COSMOS_14B_FFN_DIM,
    out_channels: int = COSMOS_14B_OUT_CHANNELS,
    text_embed_dim: int = COSMOS_14B_TEXT_EMBED_DIM,
    text_seq_len: int = COSMOS_14B_TEXT_SEQ_LEN,
    adaln_lora_dim: int = COSMOS_14B_ADALN_LORA_DIM,
    patch_size: tuple[int, int, int] = COSMOS_14B_PATCH_SIZE,
    rope_axes_dim: tuple[int, int, int] = COSMOS_14B_ROPE_AXES_DIM,
    rope_scale: tuple[float, float, float] = COSMOS_14B_ROPE_SCALE,
    rope_theta: float = COSMOS_14B_ROPE_THETA,
    eps: float = COSMOS_14B_NORM_EPS,
    vae_scale_spatial: int = COSMOS_14B_VAE_SCALE_SPATIAL,
    vae_scale_temporal: int = COSMOS_14B_VAE_SCALE_TEMPORAL,
) -> bytes:
    """Build the Cosmos-Predict2 14B DiT denoiser TRT engine (Ulysses SP).

    Args:
        config: Optional :class:`ModelConfig` for plugin parity (unused by
            the builder itself; engine shape and weight selection are fully
            determined by the explicit arguments below). Accepted for
            compatibility with the dispatch path in ``plugin.build_components``.
        weights: Output of :func:`cosmos_dit_builder.load_cosmos_dit_weights`.
            All ranks load the same weights; the engine plan bytes are
            identical across ranks aside from the per-rank cos/sin RoPE
            constants and the I/O shape.
        video_height, video_width, video_num_frames: Pixel-space target.
        precision: Compute precision tag (currently informational only --
            the dense builder uses fp32 storage for activations; this lane
            mirrors that contract for bit-exact dry-run parity at cp_size=1).
        verbose: TRT builder verbose log level.
        parallel_config: Must be a :class:`ParallelConfig` with
            ``mode='sp_ulysses'`` and a concrete ``rank``. ``cp_size==1`` is
            allowed and produces an engine structurally equivalent to the
            dense builder (the AllToAll calls collapse to pass-through).
        Remaining kwargs override architecture constants (used by the plugin
            to lock per-checkpoint values from ``transformer/config.json``).

    Returns:
        Serialized TRT engine plan as ``bytes``.
    """
    del precision  # Reserved for future fp16/bf16 storage variants.

    parallel = normalize_parallel_config(parallel_config)
    require_tensorrt_11_for_tensor_parallel(
        parallel, feature="Cosmos-Predict2 sequence-parallel (Ulysses) builds")
    if parallel.mode != "sp_ulysses":
        raise ValueError(
            "build_cosmos_dit_ulysses_engine requires parallel.mode='sp_ulysses'; "
            f"got mode={parallel.mode!r}")
    if parallel.rank < 0:
        raise ValueError(
            "build_cosmos_dit_ulysses_engine requires a concrete rank "
            f"(got rank={parallel.rank}); set parallel.for_rank(rank) before "
            "calling the builder.")
    cp_size = int(parallel.cp_size)
    rank = int(parallel.rank)
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}")
    if rank >= cp_size:
        raise ValueError(
            f"rank={rank} must be smaller than cp_size={cp_size}")

    if num_heads * head_dim != hidden_size:
        raise ValueError(
            f"hidden_size={hidden_size} must equal num_heads*head_dim "
            f"({num_heads}*{head_dim}={num_heads * head_dim})")
    if sum(rope_axes_dim) != head_dim:
        raise ValueError(
            f"sum(rope_axes_dim)={sum(rope_axes_dim)} must equal head_dim "
            f"({head_dim}); got {rope_axes_dim}")
    if num_heads % cp_size != 0:
        raise ValueError(
            f"Ulysses requires num_heads ({num_heads}) divisible by cp_size "
            f"({cp_size}); 40 heads support cp_size in {{1, 2, 4, 8}}.")

    pt, ph, pw = patch_size

    # Latent dims after VAE encode (identical to dense builder).
    t_lat = (video_num_frames - 1) // vae_scale_temporal + 1
    h_lat = video_height // vae_scale_spatial
    w_lat = video_width // vae_scale_spatial
    if t_lat % pt != 0 or h_lat % ph != 0 or w_lat % pw != 0:
        raise ValueError(
            f"Latent dims (t={t_lat}, h={h_lat}, w={w_lat}) must be divisible "
            f"by patch_size {patch_size}")
    t_patches = t_lat // pt
    h_patches = h_lat // ph
    w_patches = w_lat // pw
    full_num_patches = t_patches * h_patches * w_patches
    if full_num_patches % cp_size != 0:
        raise ValueError(
            f"Ulysses requires num_patches ({full_num_patches}) divisible by "
            f"cp_size ({cp_size}); rerun with a video shape whose "
            f"patch count is a multiple of {cp_size}.")
    local_num_patches = full_num_patches // cp_size
    patch_dim = out_channels * pt * ph * pw

    # --- Build the TRT network skeleton ---
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 64 << 30)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # --- Inputs (per-rank seq shard for hidden_states / noise_pred,
    #     replicated for everything else) ---
    hidden_inp = network.add_input(
        "hidden_states", trt.float32, (local_num_patches, hidden_size))
    encoder_hidden_inp = network.add_input(
        "encoder_hidden_states", trt.float32, (text_seq_len, text_embed_dim))
    temb_inp = network.add_input(
        "temb", trt.float32, (1, hidden_size))
    embedded_timestep_inp = network.add_input(
        "embedded_timestep", trt.float32, (1, hidden_size))

    # --- Constants ---
    eps_t = graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=np.float32))

    # 3-axis RoPE tables, sliced to the rows this rank owns. cp_size=1 is
    # a no-op slice that yields exactly the dense table.
    cos_local, sin_local = _build_3axis_rope_tables_per_rank(
        t_patches=t_patches,
        h_patches=h_patches,
        w_patches=w_patches,
        axes_dim=rope_axes_dim,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        cp_size=cp_size,
        rank=rank,
    )
    cos_const = graph_ops.add_constant(
        network, (1, local_num_patches, head_dim // 2),
        cos_local.reshape(1, local_num_patches, head_dim // 2))
    sin_const = graph_ops.add_constant(
        network, (1, local_num_patches, head_dim // 2),
        sin_local.reshape(1, local_num_patches, head_dim // 2))

    # Pre-activate temb / embedded_timestep with SiLU (replicated tensors,
    # no comm needed).
    temb_silu = _silu_2d(network, temb_inp)
    embedded_silu = _silu_2d(network, embedded_timestep_inp)

    hidden = hidden_inp

    # --- Per-block forward ---
    for layer_idx in range(num_layers):
        bw = _block_weights_or_none(weights, layer_idx)
        prefix = f"transformer_blocks.{layer_idx}"

        # === 1. Self-attention with AdaLN-Zero (norm1) ===
        h1, gate1 = _adaln_zero(
            network, hidden, temb_silu, embedded_silu,
            hidden_size=hidden_size,
            adaln_lora_dim=adaln_lora_dim,
            eps_tensor=eps_t,
            w_l1=bw["norm1.linear_1.weight"], b_l1=bw["norm1.linear_1.bias"],
            w_l2=bw["norm1.linear_2.weight"], b_l2=bw["norm1.linear_2.bias"],
            w_l3=bw["norm1.linear_3.weight"], b_l3=bw["norm1.linear_3.bias"],
        )
        attn_out = _self_attention_ulysses(
            network, h1,
            bw=bw,
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
            local_num_patches=local_num_patches,
            full_num_patches=full_num_patches,
            cp_size=cp_size,
            cos_const=cos_const,
            sin_const=sin_const,
            eps_t=eps_t,
            tag=prefix,
        )
        gated = network.add_elementwise(
            attn_out, gate1, trt.ElementWiseOperation.PROD).get_output(0)
        hidden = network.add_elementwise(
            hidden, gated, trt.ElementWiseOperation.SUM).get_output(0)

        # === 2. Cross-attention with AdaLN-Zero (norm2) ===
        # Query is seq-sharded ([L, hidden]); K/V come from the replicated
        # encoder_hidden_states ([text_seq_len, text_embed_dim]). The
        # attention math is local — no comm required.
        h2, gate2 = _adaln_zero(
            network, hidden, temb_silu, embedded_silu,
            hidden_size=hidden_size,
            adaln_lora_dim=adaln_lora_dim,
            eps_tensor=eps_t,
            w_l1=bw["norm2.linear_1.weight"], b_l1=bw["norm2.linear_1.bias"],
            w_l2=bw["norm2.linear_2.weight"], b_l2=bw["norm2.linear_2.bias"],
            w_l3=bw["norm2.linear_3.weight"], b_l3=bw["norm2.linear_3.bias"],
        )
        cq = graph_ops.add_matmul_rhs_constant(
            network, h2, hidden_size, hidden_size, bw["attn2.to_q.weight"])
        ck = graph_ops.add_matmul_rhs_constant(
            network, encoder_hidden_inp, text_embed_dim, hidden_size,
            bw["attn2.to_k.weight"])
        cv = graph_ops.add_matmul_rhs_constant(
            network, encoder_hidden_inp, text_embed_dim, hidden_size,
            bw["attn2.to_v.weight"])
        nq2 = bw["attn2.norm_q.weight"]
        if nq2 is not None:
            cq = graph_ops.add_rms_norm_per_head(
                network, cq, num_heads, head_dim, nq2, eps_t,
                sequence_length=local_num_patches)
        cross_ctx = graph_ops.add_attention_from_rows(
            network, cq, ck, cv,
            num_heads=num_heads, head_dim=head_dim,
            q_seq=local_num_patches, kv_seq=text_seq_len,
            tag=f"{prefix}.attn2")
        cross_out = graph_ops.add_matmul_rhs_constant(
            network, cross_ctx, hidden_size, hidden_size,
            bw["attn2.to_out.0.weight"])
        gated2 = network.add_elementwise(
            cross_out, gate2, trt.ElementWiseOperation.PROD).get_output(0)
        hidden = network.add_elementwise(
            hidden, gated2, trt.ElementWiseOperation.SUM).get_output(0)

        # === 3. FFN with AdaLN-Zero (norm3) ===
        # Pointwise along the sequence axis: applies directly to the
        # seq-sharded tensor with no comm.
        h3, gate3 = _adaln_zero(
            network, hidden, temb_silu, embedded_silu,
            hidden_size=hidden_size,
            adaln_lora_dim=adaln_lora_dim,
            eps_tensor=eps_t,
            w_l1=bw["norm3.linear_1.weight"], b_l1=bw["norm3.linear_1.bias"],
            w_l2=bw["norm3.linear_2.weight"], b_l2=bw["norm3.linear_2.bias"],
            w_l3=bw["norm3.linear_3.weight"], b_l3=bw["norm3.linear_3.bias"],
        )
        ffn_h = graph_ops.add_matmul_rhs_constant(
            network, h3, hidden_size, ffn_dim, bw["ff.net.0.proj.weight"])
        if bw["ff.net.0.proj.bias"] is not None:
            ffn_h = graph_ops.add_bias_sum(
                network, ffn_h, ffn_dim, bw["ff.net.0.proj.bias"])
        ffn_h = graph_ops.add_gelu_new(network, ffn_h)
        ffn_h = graph_ops.add_matmul_rhs_constant(
            network, ffn_h, ffn_dim, hidden_size, bw["ff.net.2.weight"])
        if bw["ff.net.2.bias"] is not None:
            ffn_h = graph_ops.add_bias_sum(
                network, ffn_h, hidden_size, bw["ff.net.2.bias"])
        gated3 = network.add_elementwise(
            ffn_h, gate3, trt.ElementWiseOperation.PROD).get_output(0)
        hidden = network.add_elementwise(
            hidden, gated3, trt.ElementWiseOperation.SUM).get_output(0)

    # --- Final norm_out (2 chunks: shift, scale) + proj_out ---
    # Pointwise along the sequence axis -> seq-sharded output, no comm. The
    # caller is responsible for the final AllGather on noise_pred.
    h_final = _adaln_final(
        network, hidden, temb_silu, embedded_silu,
        hidden_size=hidden_size,
        adaln_lora_dim=adaln_lora_dim,
        eps_tensor=eps_t,
        w_l1=weights["norm_out.linear_1.weight"],
        b_l1=weights.get("norm_out.linear_1.bias"),
        w_l2=weights["norm_out.linear_2.weight"],
        b_l2=weights.get("norm_out.linear_2.bias"),
        w_l3=weights["norm_out.linear_3.weight"],
        b_l3=weights.get("norm_out.linear_3.bias"),
    )
    output = graph_ops.add_matmul_rhs_constant(
        network, h_final, hidden_size, patch_dim,
        weights["proj_out.weight"])
    proj_b = weights.get("proj_out.bias")
    if proj_b is not None:
        output = graph_ops.add_bias_sum(network, output, patch_dim, proj_b)

    out_cast = network.add_cast(output, trt.float32).get_output(0)
    out_cast.name = "noise_pred"
    network.mark_output(out_cast)

    print(
        f"[cosmos-dit-ulysses] Building TRT engine (hidden={hidden_size}, "
        f"layers={num_layers}, full_num_patches={full_num_patches}, "
        f"local_num_patches={local_num_patches}, cp_size={cp_size}, "
        f"rank={rank}, t_lat={t_lat}, h_lat={h_lat}, w_lat={w_lat}, "
        f"axes_dim={rope_axes_dim}, rope_scale={rope_scale}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError(
            "TRT engine serialization failed for Cosmos DiT (Ulysses)")
    return bytes(plan)


# ---------------------------------------------------------------------------
# Weight loading is shared with the dense builder.
#
# All ranks load the same weights (Ulysses keeps the projection weights
# replicated; only the per-rank position slice and the I/O shape differ),
# so we re-export ``load_cosmos_dit_weights`` here for convenience.
# ---------------------------------------------------------------------------

load_cosmos_dit_weights = cosmos_dit_builder.load_cosmos_dit_weights
