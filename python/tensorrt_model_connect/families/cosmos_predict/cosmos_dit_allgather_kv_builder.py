"""Cosmos-Predict2 14B DiT engine builder — AllGather-KV sequence parallel.

This is the AllGather-KV variant of the sequence-parallel Cosmos-Predict2 14B
denoiser. The patch (sequence) dimension is sharded across ``cp_size`` ranks
and each rank holds **all** attention heads (heads are *not* sharded). The
self-attention block uses a single collective per pass: an AllGather on the
local K and V tensors along the sequence axis, so that every rank reconstructs
the full ``[num_patches, num_heads, head_dim]`` K/V tensors and can compute
attention from its local Q rows against the full keys/values.

Engine I/O contract (per-rank, with ``L = num_patches // cp_size``):

    hidden_states         [L, hidden_size]
    encoder_hidden_states [text_seq_len, text_embed_dim]    (replicated)
    temb                  [1, hidden_size]                  (replicated)
    embedded_timestep     [1, hidden_size]                  (replicated)
    noise_pred            [L, out_channels * pt * ph * pw]  (sharded)

Strategy:

* All Linear projections operate on the local ``L``-row slice (no slicing of
  weights — heads and feature dims are replicated).
* Q/K/V have shape ``[L, num_heads * head_dim]`` after projection.
* Per-head RMSNorm on Q and K — local rows only, no collective.
* 3-axis RoPE is applied to Q and K using a *per-rank* slice of the precomputed
  cos/sin tables (positions ``[rank * L, (rank + 1) * L)``).
* AllGather K and V along sequence axis (gather_axis=0). Each rank now holds
  the full ``[num_patches, num_heads * head_dim]`` K and V tensors. No further
  collective is needed in the block.
* Attention runs with ``q_seq=L`` and ``kv_seq=num_patches`` on each rank,
  producing a local context ``[L, num_heads * head_dim]``.
* ``to_out.0`` projection, gate, and residual stay local.
* Cross-attention is unchanged from the dense builder: text K/V come from the
  replicated ``encoder_hidden_states``, and Q is local.
* The final norm_out + proj_out output head is unchanged (operates on the
  local ``L``-row slice and produces a per-rank ``[L, patch_dim]`` output).

At ``cp_size == 1`` the AllGather is a pass-through (``add_all_gather`` returns
the input tensor unchanged), so the output is bit-identical to the dense
builder.

Memory note: holding the full ``[num_patches, num_heads, head_dim]`` K (and
likewise V) on every rank is the AllGather-KV strategy's main cost. For 14B at
720x1280 @ 49 frames in fp16, ``num_patches = 13 * 90 * 160 = 187200``; per
attention block each rank's K (or V) tensor occupies ``187200 * 40 * 128 * 2``
bytes = ~1.91 GiB.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np

from tensorrt_model_connect import trt_compat

from ... import graph_ops
from ...parallel_config import (
    ParallelConfig,
    add_all_gather,
    normalize_parallel_config,
)

# Re-use the dense builder's architecture constants and small helpers so the
# only divergence is the SP-specific I/O layout and the K/V AllGather.
from .cosmos_dit_builder import (  # noqa: F401  (load_cosmos_dit_weights re-export)
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
    _build_3axis_rope_tables,
    _silu_2d,
    load_cosmos_dit_weights,
)

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...checkpoint_mapper import WeightDict
    from ...config import ModelConfig


# Self-attention AllGather happens along the sequence axis of a row-major
# ``[L, num_heads * head_dim]`` tensor. The sequence dim is axis 0.
_SP_SEQ_GATHER_AXIS = 0


def _apply_rope_3axis_local(
    network: "trt.INetworkDefinition",
    x: "trt.ITensor",
    *,
    num_heads: int,
    head_dim: int,
    local_num_patches: int,
    cos_3d_local: "trt.ITensor",
    sin_3d_local: "trt.ITensor",
) -> "trt.ITensor":
    """Apply per-rank 3-axis RoPE to a ``[L, num_heads * head_dim]`` tensor.

    ``cos_3d_local`` / ``sin_3d_local`` are ``[1, L, head_dim // 2]`` constants
    containing only the rank's local slice of the full RoPE table.
    """
    return graph_ops.add_apply_rope_native_sequence(
        network,
        x,
        num_heads=num_heads,
        head_dim=head_dim,
        cos_cache_3d=cos_3d_local,
        sin_cache_3d=sin_3d_local,
        rotary_embedding_dim=head_dim,
        interleaved=False,
        sequence_length=local_num_patches,
    )


def build_cosmos_dit_allgather_kv_engine(
    config: "ModelConfig | None",
    weights: "WeightDict",
    *,
    video_height: int,
    video_width: int,
    video_num_frames: int,
    precision: str = "fp16",
    verbose: bool = False,
    parallel_config: ParallelConfig,
    # Architecture knobs match the dense builder; defaults are the locked
    # Cosmos-Predict2 14B Video2World values.
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
    """Build the AllGather-KV sequence-parallel Cosmos-Predict2 14B DiT engine.

    The ``parallel_config`` keyword argument is required and must have
    ``mode == "sp_allgather_kv"``. ``cp_size`` may be 1 (in which case the
    engine is bit-identical to the dense builder — the AllGather collective is
    a no-op) or any power-of-two divisor of ``num_patches``.
    """
    del config  # currently unused; signature mirrors the family-wide convention
    del precision  # storage dtype is controlled by ``weights`` already

    parallel = normalize_parallel_config(parallel_config)
    if parallel.mode != "sp_allgather_kv":
        raise ValueError(
            "build_cosmos_dit_allgather_kv_engine requires "
            f"parallel_config.mode='sp_allgather_kv'; got {parallel.mode!r}")
    cp_size = int(parallel.cp_size)
    if cp_size <= 0:
        raise ValueError(
            f"build_cosmos_dit_allgather_kv_engine: cp_size must be positive; "
            f"got {cp_size}")

    if num_heads * head_dim != hidden_size:
        raise ValueError(
            f"hidden_size={hidden_size} must equal num_heads*head_dim "
            f"({num_heads}*{head_dim}={num_heads * head_dim})")
    if sum(rope_axes_dim) != head_dim:
        raise ValueError(
            f"sum(rope_axes_dim)={sum(rope_axes_dim)} must equal head_dim "
            f"({head_dim}); got {rope_axes_dim}")
    pt, ph, pw = patch_size

    # Latent grid after VAE encode → patches.
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
            f"AllGather-KV SP requires num_patches ({num_patches}) divisible "
            f"by cp_size ({cp_size})")
    local_num_patches = num_patches // cp_size

    # Per-rank slice of the sequence: [rank * L, (rank + 1) * L).
    # When the engine is being built rank-agnostically (rank == -1), default
    # to rank 0's slice — the RoPE table content is the only rank-dependent
    # constant baked in. The runtime must materialize a per-rank engine; this
    # mirrors the convention used by TP builders elsewhere in the codebase.
    rank = parallel.rank if parallel.rank >= 0 else 0
    if rank >= cp_size:
        raise ValueError(
            f"rank={rank} must be < cp_size={cp_size}")
    seq_start = rank * local_num_patches

    # --- Build the TRT network skeleton ---
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 << 30)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))

    # --- Inputs (sequence dim sharded as L = num_patches // cp_size) ---
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

    # 3-axis RoPE tables built for the full patch grid, then sliced to the
    # rank's local sequence range. This keeps the math identical to dense at
    # cp_size=1 while producing the correct positions for each rank's local Q
    # rows (and matching local-position K rows pre-AllGather).
    cos_full, sin_full = _build_3axis_rope_tables(
        t_patches=t_patches,
        h_patches=h_patches,
        w_patches=w_patches,
        axes_dim=rope_axes_dim,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    cos_local = cos_full[seq_start:seq_start + local_num_patches]
    sin_local = sin_full[seq_start:seq_start + local_num_patches]
    cos_const = graph_ops.add_constant(
        network, (1, local_num_patches, head_dim // 2),
        cos_local.reshape(1, local_num_patches, head_dim // 2))
    sin_const = graph_ops.add_constant(
        network, (1, local_num_patches, head_dim // 2),
        sin_local.reshape(1, local_num_patches, head_dim // 2))

    # Pre-activate temb / embedded_timestep with SiLU (replicated; same across
    # all ranks).
    temb_silu = _silu_2d(network, temb_inp)
    embedded_silu = _silu_2d(network, embedded_timestep_inp)

    hidden = hidden_inp

    # --- Per-block forward ---
    for layer_idx in range(num_layers):
        bw = _block_weights_or_none(weights, layer_idx)
        prefix = f"transformer_blocks.{layer_idx}"

        # === 1. Self-attention with AdaLN-Zero (norm1) ===
        # AdaLN-Zero modulates per-row, so it stays local: scale/shift/gate are
        # broadcast across the sequence dim.
        h1, gate1 = _adaln_zero(
            network, hidden, temb_silu, embedded_silu,
            hidden_size=hidden_size,
            adaln_lora_dim=adaln_lora_dim,
            eps_tensor=eps_t,
            w_l1=bw["norm1.linear_1.weight"], b_l1=bw["norm1.linear_1.bias"],
            w_l2=bw["norm1.linear_2.weight"], b_l2=bw["norm1.linear_2.bias"],
            w_l3=bw["norm1.linear_3.weight"], b_l3=bw["norm1.linear_3.bias"],
        )
        # Q/K/V projections on the local sequence slice. Weights are
        # replicated (no head sharding under AllGather-KV).
        q = graph_ops.add_matmul_rhs_constant(
            network, h1, hidden_size, hidden_size, bw["attn1.to_q.weight"])
        k = graph_ops.add_matmul_rhs_constant(
            network, h1, hidden_size, hidden_size, bw["attn1.to_k.weight"])
        v = graph_ops.add_matmul_rhs_constant(
            network, h1, hidden_size, hidden_size, bw["attn1.to_v.weight"])
        # Per-head RMSNorm on Q and K (local rows; per-head along feature dim).
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
        # 3-axis RoPE on Q and K using the rank's local slice of the table.
        q = _apply_rope_3axis_local(
            network, q,
            num_heads=num_heads, head_dim=head_dim,
            local_num_patches=local_num_patches,
            cos_3d_local=cos_const, sin_3d_local=sin_const)
        k = _apply_rope_3axis_local(
            network, k,
            num_heads=num_heads, head_dim=head_dim,
            local_num_patches=local_num_patches,
            cos_3d_local=cos_const, sin_3d_local=sin_const)
        # AllGather K and V along the sequence axis (axis 0). After this, both
        # tensors are [num_patches, num_heads * head_dim] on every rank. This
        # is a no-op when cp_size == 1.
        k_full = add_all_gather(
            network, k, cp_size, gather_axis=_SP_SEQ_GATHER_AXIS)
        v_full = add_all_gather(
            network, v, cp_size, gather_axis=_SP_SEQ_GATHER_AXIS)
        # Local-Q against full K/V attention: q_seq=L, kv_seq=num_patches.
        ctx = graph_ops.add_attention_from_rows(
            network, q, k_full, v_full,
            num_heads=num_heads, head_dim=head_dim,
            q_seq=local_num_patches, kv_seq=num_patches,
            tag=f"{prefix}.attn1")
        # to_out.0 projection stays local — [L, hidden_size] in, [L, hidden_size] out.
        attn_out = graph_ops.add_matmul_rhs_constant(
            network, ctx, hidden_size, hidden_size,
            bw["attn1.to_out.0.weight"])
        gated = network.add_elementwise(
            attn_out, gate1, trt.ElementWiseOperation.PROD).get_output(0)
        hidden = network.add_elementwise(
            hidden, gated, trt.ElementWiseOperation.SUM).get_output(0)

        # === 2. Cross-attention with AdaLN-Zero (norm2) ===
        # Unchanged from dense: text K/V come from replicated encoder_hidden,
        # Q is the local sequence slice. No collective needed.
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
        # FFN is point-wise along the sequence dim → local, no collective.
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
    # Output head is point-wise along the sequence dim, so it stays on the
    # local slice. The per-rank output is [L, patch_dim].
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

    # Mark output (fp32 by contract).
    out_cast = network.add_cast(output, trt.float32).get_output(0)
    out_cast.name = "noise_pred"
    network.mark_output(out_cast)

    print(
        f"[cosmos-dit:sp_allgather_kv] Building TRT engine "
        f"(rank={parallel.rank}, cp_size={cp_size}, "
        f"hidden={hidden_size}, layers={num_layers}, "
        f"num_patches={num_patches}, local_num_patches={local_num_patches}, "
        f"t_lat={t_lat}, h_lat={h_lat}, w_lat={w_lat}, "
        f"axes_dim={rope_axes_dim}, rope_scale={rope_scale}) ...",
        file=sys.stderr,
    )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError(
            "TRT engine serialization failed for Cosmos DiT (sp_allgather_kv)")
    return bytes(plan)


# Re-export so callers can do
# ``from cosmos_dit_allgather_kv_builder import load_cosmos_dit_weights``.
__all__ = [
    "build_cosmos_dit_allgather_kv_engine",
    "load_cosmos_dit_weights",
]
