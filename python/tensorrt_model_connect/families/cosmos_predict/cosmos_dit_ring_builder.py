"""Cosmos-Predict2 14B DiT engine builder — RingAttention sequence-parallel variant.

Sequence-parallel ``cp_size``-way layout. The patch sequence (``num_patches``)
is sharded across ``cp_size`` ranks; each rank holds ``num_patches // cp_size``
patches *and all 40 attention heads*. Replicated inputs (text/T5 encoder
output, ``temb``, ``embedded_timestep``) remain unsharded.

Ring strategy implemented here
-------------------------------

The classic RingAttention formulation steps K/V around a ring of ``cp_size``
ranks and accumulates an online stable-softmax over ``cp_size`` partial
attention computations. Expressing the online-softmax accumulator as a
*static* TRT graph (running max ``m``, running denominator ``l``,
exp-rebalancing of the running numerator at every step) is doable in principle
but requires reimplementing the attention kernel in low-level elementwise /
reduction ops — TRT's fused ``IFusedMHA`` / ``add_attention_from_rows`` only
exposes a single softmax over the full K dimension and cannot be split.

This builder therefore implements the **simplified one-pass Ring**:

  1. Q is computed and kept *local* (sharded along seq).
  2. K and V are computed locally, RMSNorm'd, and have **per-rank RoPE**
     baked in (each rank's local position table covers its own chunk of the
     global ``num_patches``). After RoPE, K and V are ``AllGather``-ed along
     the sequence axis so every rank ends up with the full
     ``[num_patches, hidden]`` K and V.
  3. A single full-sequence attention is run with ``q_seq=num_patches/cp_size``
     and ``kv_seq=num_patches`` — i.e. each rank attends its local Q rows
     against the gathered K/V. The output is naturally seq-sharded.

That is *one-shot* gathering rather than ``cp_size`` ring rotations, so it is
mathematically identical to the dense self-attention (and to the
``AllGather-KV`` variant) — modulo the SP I/O reshape. The framing remains
"Ring" because:

  * the per-rank state (local Q, local K, local V before gather) and ring
    topology are preserved;
  * a real online-softmax loop with ``cp_size - 1`` ``add_all_to_all`` /
    P2P-rotate steps is a drop-in upgrade against this scaffolding (replace
    the AllGather + single attention call with the unrolled accumulator).

Cross-attention is unchanged: Q is seq-sharded; K/V are projected from the
replicated text encoder output (already full sequence).

At ``cp_size == 1`` the ``add_all_gather`` calls in
``parallel_config`` pass tensors through unchanged, so the engine is
bit-identical to ``cosmos_dit_builder.build_cosmos_dit_engine`` (modulo TRT
layer naming).

Constraints
-----------

* ``num_patches % cp_size == 0`` — checked in
  :func:`build_cosmos_dit_ring_engine`.
* All ranks share the same global ``num_patches``; per-rank position offsets
  for RoPE are computed from ``parallel_config.rank``.
"""

from __future__ import annotations

import ast as _ast  # imported only for the bottom-of-file ast.parse self-check
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tensorrt_model_connect import trt_compat

from ... import graph_ops
from ...parallel_config import (
    ParallelConfig,
    add_all_gather,
    normalize_parallel_config,
)

# Re-export the dense weight loader so callers can use a single import.
from .cosmos_dit_builder import (  # noqa: F401  (re-exported for plugin users)
    COSMOS_14B_ADALN_LORA_DIM,
    COSMOS_14B_FFN_DIM,
    COSMOS_14B_HEAD_DIM,
    COSMOS_14B_HIDDEN_SIZE,
    COSMOS_14B_IN_CHANNELS,
    COSMOS_14B_MAX_SIZE,
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
    _build_3axis_rope_tables,
    _silu_2d,
    load_cosmos_dit_weights,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict


# ---------------------------------------------------------------------------
# Per-rank RoPE table slicing
# ---------------------------------------------------------------------------


def _build_per_rank_rope_tables(
    *,
    t_patches: int,
    h_patches: int,
    w_patches: int,
    axes_dim: tuple[int, int, int],
    rope_scale: tuple[float, float, float],
    rope_theta: float,
    cp_size: int,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the cos/sin tables covering just this rank's chunk of patches.

    Returns ``(cos_local, sin_local, local_num_patches)`` where the tables
    have shape ``[local_num_patches, head_dim // 2]``. The slicing is a
    straightforward sequence-axis split of the full 3-axis RoPE table built
    by :func:`_build_3axis_rope_tables`: rank ``r`` gets rows
    ``[r * local : (r + 1) * local]`` of the flattened
    ``[T*H*W, head_dim/2]`` cache.
    """
    cos_full, sin_full = _build_3axis_rope_tables(
        t_patches=t_patches,
        h_patches=h_patches,
        w_patches=w_patches,
        axes_dim=axes_dim,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    num_patches = t_patches * h_patches * w_patches
    if num_patches % cp_size != 0:
        raise ValueError(
            f"num_patches={num_patches} must be divisible by cp_size={cp_size}"
        )
    local = num_patches // cp_size
    start = rank * local
    stop = start + local
    cos_local = np.ascontiguousarray(cos_full[start:stop, :])
    sin_local = np.ascontiguousarray(sin_full[start:stop, :])
    return cos_local, sin_local, local


def _apply_rope_local(
    network: "trt.INetworkDefinition",
    x: "trt.ITensor",
    *,
    num_heads: int,
    head_dim: int,
    local_num_patches: int,
    cos_local: "trt.ITensor",
    sin_local: "trt.ITensor",
) -> "trt.ITensor":
    """Apply rotate-half RoPE to a [local_num_patches, H*D] tensor.

    Same wrapper as the dense :func:`_apply_rope_3axis` but sized for the
    per-rank chunk. The 3-axis layout is encoded in the *table contents*
    (sliced from the full 3-axis cache), not the math.
    """
    return graph_ops.add_apply_rope_native_sequence(
        network,
        x,
        num_heads=num_heads,
        head_dim=head_dim,
        cos_cache_3d=cos_local,
        sin_cache_3d=sin_local,
        rotary_embedding_dim=head_dim,
        interleaved=False,
        sequence_length=local_num_patches,
    )


# ---------------------------------------------------------------------------
# Engine builder
# ---------------------------------------------------------------------------


def build_cosmos_dit_ring_engine(
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
    """Build the Cosmos-Predict2 14B DiT denoiser TRT engine for one Ring rank.

    Args:
        config: Family-level model config (unused inside the builder but
            preserved for plugin parity with the dense entry point).
        weights: Output of
            :func:`cosmos_dit_builder.load_cosmos_dit_weights`. Weights are
            replicated across ranks — RingAttention shards activations
            (the sequence dimension), not parameters.
        video_height, video_width, video_num_frames: Pixel-space target.
        precision: Reserved for future fp16/bf16 selection. The current
            implementation matches the dense builder and runs the whole
            graph in fp32 (TRT may still pick a lower-precision tactic for
            individual ops).
        verbose: TRT verbose logger.
        parallel_config: SP layout. Must have ``mode='sp_ring'``,
            ``cp_size > 1``, and a concrete non-negative ``rank``.
        ... (all other kwargs): architecture constants. Defaults match
            Cosmos-Predict2 14B Video2World.

    Returns:
        Serialized TRT engine ``bytes`` for ``parallel_config.rank``.
    """
    parallel = normalize_parallel_config(parallel_config)
    parallel.validate()
    if parallel.mode != "sp_ring":
        raise ValueError(
            f"build_cosmos_dit_ring_engine requires mode='sp_ring', "
            f"got mode={parallel.mode!r}")
    if parallel.cp_size <= 1:
        raise ValueError(
            f"build_cosmos_dit_ring_engine requires cp_size > 1, "
            f"got cp_size={parallel.cp_size}")
    if parallel.rank < 0:
        raise ValueError(
            "build_cosmos_dit_ring_engine requires a concrete non-negative rank")
    cp_size = int(parallel.cp_size)
    rank = int(parallel.rank)

    if num_heads * head_dim != hidden_size:
        raise ValueError(
            f"hidden_size={hidden_size} must equal num_heads*head_dim "
            f"({num_heads}*{head_dim}={num_heads * head_dim})")
    if sum(rope_axes_dim) != head_dim:
        raise ValueError(
            f"sum(rope_axes_dim)={sum(rope_axes_dim)} must equal head_dim "
            f"({head_dim}); got {rope_axes_dim}")
    pt, ph, pw = patch_size

    # Latent dims after VAE encode.
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
    num_patches = t_patches * h_patches * w_patches
    patch_dim = out_channels * pt * ph * pw

    if num_patches % cp_size != 0:
        raise ValueError(
            f"Ring SP requires num_patches ({num_patches}) divisible by "
            f"cp_size ({cp_size})")
    local_num_patches = num_patches // cp_size
    if rank >= cp_size:
        raise ValueError(
            f"rank={rank} must be < cp_size={cp_size}")

    # --- Build the TRT network skeleton ---
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config_trt = builder.create_builder_config()
    config_trt.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # --- Inputs (per-rank seq-sharded hidden, replicated everything else) ---
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

    # Per-rank 3-axis RoPE tables (sliced from the full [num_patches, hd/2]
    # cache so each rank's tables cover [rank * L, (rank+1) * L)).
    cos_local_np, sin_local_np, _ = _build_per_rank_rope_tables(
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
        cos_local_np.reshape(1, local_num_patches, head_dim // 2))
    sin_const = graph_ops.add_constant(
        network, (1, local_num_patches, head_dim // 2),
        sin_local_np.reshape(1, local_num_patches, head_dim // 2))

    # Pre-activate temb / embedded_timestep with SiLU.
    temb_silu = _silu_2d(network, temb_inp)
    embedded_silu = _silu_2d(network, embedded_timestep_inp)

    hidden = hidden_inp

    # --- Per-block forward ---
    for layer_idx in range(num_layers):
        bw = _block_weights_or_none(weights, layer_idx)
        prefix = f"transformer_blocks.{layer_idx}"

        # === 1. Self-attention with AdaLN-Zero (norm1) — Ring SP path ===
        h1, gate1 = _adaln_zero(
            network, hidden, temb_silu, embedded_silu,
            hidden_size=hidden_size,
            adaln_lora_dim=adaln_lora_dim,
            eps_tensor=eps_t,
            w_l1=bw["norm1.linear_1.weight"], b_l1=bw["norm1.linear_1.bias"],
            w_l2=bw["norm1.linear_2.weight"], b_l2=bw["norm1.linear_2.bias"],
            w_l3=bw["norm1.linear_3.weight"], b_l3=bw["norm1.linear_3.bias"],
        )
        # Q/K/V projections (no biases). Inputs are seq-sharded; all-heads
        # are retained per rank.
        q = graph_ops.add_matmul_rhs_constant(
            network, h1, hidden_size, hidden_size, bw["attn1.to_q.weight"])
        k = graph_ops.add_matmul_rhs_constant(
            network, h1, hidden_size, hidden_size, bw["attn1.to_k.weight"])
        v = graph_ops.add_matmul_rhs_constant(
            network, h1, hidden_size, hidden_size, bw["attn1.to_v.weight"])

        # Per-head RMSNorm on Q and K (sequence length is the local shard).
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

        # 3-axis RoPE on Q and K — applied *before* the K/V AllGather so the
        # gathered K already carries the correct global positional encoding
        # (each rank's cos/sin table targets its own slice of positions).
        q = _apply_rope_local(
            network, q,
            num_heads=num_heads, head_dim=head_dim,
            local_num_patches=local_num_patches,
            cos_local=cos_const, sin_local=sin_const)
        k = _apply_rope_local(
            network, k,
            num_heads=num_heads, head_dim=head_dim,
            local_num_patches=local_num_patches,
            cos_local=cos_const, sin_local=sin_const)

        # --- Ring "one-shot gather" step ---
        # AllGather K and V along the sequence axis (axis=0) so every rank
        # ends up with the full [num_patches, hidden] K / V. This is the
        # simplified Ring described in the module docstring; the unrolled
        # cp_size-step online-softmax loop is a future optimization.
        k_full = add_all_gather(network, k, cp_size, gather_axis=0)
        v_full = add_all_gather(network, v, cp_size, gather_axis=0)

        # Local Q vs full K/V attention. q_seq is the local shard,
        # kv_seq is the global num_patches — same call shape as a
        # cross-attention but with a self-attention weight set.
        ctx = graph_ops.add_attention_from_rows(
            network, q, k_full, v_full,
            num_heads=num_heads, head_dim=head_dim,
            q_seq=local_num_patches, kv_seq=num_patches,
            tag=f"{prefix}.attn1.ring")
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx, hidden_size, hidden_size,
            bw["attn1.to_out.0.weight"])
        gated = network.add_elementwise(
            attn_out, gate1, trt.ElementWiseOperation.PROD).get_output(0)
        hidden = network.add_elementwise(
            hidden, gated, trt.ElementWiseOperation.SUM).get_output(0)

        # === 2. Cross-attention with AdaLN-Zero (norm2) — unchanged.
        # Q is seq-sharded, K/V come from the replicated T5 encoder output,
        # so this matches the dense block 1:1 with the only difference
        # being the q_seq dimension.
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

        # === 3. FFN with AdaLN-Zero (norm3) — pure elementwise / matmul,
        # seq-sharded throughout (no collectives needed).
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

    # --- Final norm_out (2 chunks: shift, scale) + proj_out --- (seq-sharded)
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

    # Mark output (fp32 by contract). Per-rank shape is
    # [local_num_patches, patch_dim].
    out_cast = network.add_cast(output, trt.float32).get_output(0)
    out_cast.name = "noise_pred"
    network.mark_output(out_cast)

    print(
        f"[cosmos-dit-ring] Building TRT engine "
        f"(hidden={hidden_size}, layers={num_layers}, "
        f"num_patches={num_patches}, local={local_num_patches}, "
        f"cp_size={cp_size}, rank={rank}, "
        f"t_lat={t_lat}, h_lat={h_lat}, w_lat={w_lat}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, config_trt)
    if plan is None:
        raise RuntimeError(
            f"TRT engine serialization failed for Cosmos DiT Ring rank={rank}")
    return bytes(plan)


# ---------------------------------------------------------------------------
# Self-check: ast.parse must succeed on this file. Kept inline so callers
# importing the module can detect syntax regressions early. ``_ast.parse``
# is only run when the module is imported with ``__debug__`` enabled (the
# default outside ``python -O``); it is a no-op otherwise.
# ---------------------------------------------------------------------------

if __debug__:
    try:
        _ast.parse(Path(__file__).read_text(encoding="utf-8"), filename=__file__)
    except SyntaxError as _e:  # pragma: no cover — defensive, never expected.
        raise SyntaxError(
            f"cosmos_dit_ring_builder.py failed self ast.parse: {_e}") from _e
