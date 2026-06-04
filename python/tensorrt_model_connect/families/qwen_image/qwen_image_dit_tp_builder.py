"""Tensor-parallel Qwen-Image MMDiT denoiser engine builder.

Builds the same Qwen-Image MMDiT denoiser plan as
``qwen_image_dit_builder.build_qwen_image_dit_engine`` but with per-rank
tensor-parallel sharding of every joint transformer block.

TP policy (mirrors wan_t2v/standard_dit_tp_builder.py and
flux/flux2_dit_tp_builder.py):

* QKV projections (both streams: ``attn.to_q/to_k/to_v`` for the image
  stream and ``attn.add_q_proj/add_k_proj/add_v_proj`` for the text
  stream) are column-parallel: the output dimension is sharded so each
  rank produces a slice of the heads (``num_heads / tp_size`` heads per
  rank).
* Per-head QK-norm weights (``attn.norm_q/norm_k``,
  ``attn.norm_added_q/norm_added_k``) are sharded along the head axis to
  match the local Q/K head count.
* Joint attention runs on the rank-local Q/K/V slices so each rank
  computes only its share of the heads.
* Output projections (``attn.to_out.0``, ``attn.to_add_out``) are
  row-parallel: the rank-local hidden slice is matmul'd with the
  rank-local input slice of the weight, and an all-reduce SUM joins the
  partial sums across ranks.
* MLPs: ``net.0.proj`` is column-parallel, ``net.2`` is row-parallel
  with all-reduce SUM. Activation runs between the two on the rank-local
  slice (``intermediate_size / tp_size``).
* Norms, modulation params (``img_mod.1`` / ``txt_mod.1``),
  ``img_in`` / ``txt_in`` / ``txt_norm``, ``time_text_embed``,
  ``norm_out``, and ``proj_out`` stay replicated (same weights on every
  rank). RoPE cos/sin tables are also replicated.

Constraints:
  * ``num_attention_heads % tp_size == 0`` (so heads divide cleanly).
  * ``intermediate_size % tp_size == 0`` (so MLP shards are integer).

Engine I/O matches the dense builder exactly so the runtime contract is
unchanged for the TP-enabled path: per-rank engines are interchangeable
with the dense engine, and the runtime sums their (post-allreduce)
outputs implicitly through the in-engine NCCL collectives.

This builder intentionally reuses the dense builder's private helpers
(``_add_linear_2d``, ``_add_layernorm_no_affine_3d``, ``_add_modulate``,
``_add_gate_residual``, ``_add_mlp_block``, ``_add_rms_norm_per_head``,
``_add_rope_pair``, ``_add_joint_attention``, RoPE pre-compute,
constants, etc.) so the per-block math stays in lockstep with the dense
path — only the linear/qkv layers diverge for TP sharding.

Trace IDs: ARCH-FAM-001, UD-FAM-QWEN-IMAGE-01 (TP extension).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Union

import numpy as np

from tensorrt_model_connect import trt_compat

from ... import graph_ops
from ...parallel_config import (
    ParallelConfig,
    _slice_first_dim,
    add_all_reduce_sum,
    normalize_parallel_config,
    validate_dit_tp,
)
from . import qwen_image_dit_builder as _dense

trt = trt_compat.get_trt()

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# TP-aware linear primitives.
#
# Each primitive reuses ``_dense._add_linear_2d`` for the actual matmul +
# bias-add so the bf16 storage / fp32-reduction pattern stays identical to
# the dense builder; the only difference is the weight slice we feed in.
# ---------------------------------------------------------------------------


def _hf_w(weights: Mapping[str, "np.ndarray"], key: str) -> np.ndarray:
    """Fetch a weight as contiguous fp32 [out, in] (HF state-dict order)."""
    return np.ascontiguousarray(np.asarray(weights[key], dtype=np.float32))


def _hf_b(weights: Mapping[str, "np.ndarray"], key: str) -> "np.ndarray | None":
    v = weights.get(key)
    if v is None:
        return None
    return np.ascontiguousarray(np.asarray(v, dtype=np.float32))


def _linear_col_parallel_2d(
    network,
    x_2d,
    in_dim: int,
    out_dim: int,
    weights: Mapping[str, "np.ndarray"],
    prefix: str,
    parallel: ParallelConfig,
):
    """Column-parallel Linear on [N, in_dim] -> [N, local_out_dim].

    Slices the HF weight ``[out, in]`` along the leading (output) axis to
    keep only the rank-local output features, then applies the rank-local
    bias slice. Caller is responsible for stitching the local outputs
    back into the original dimension (typically by treating the result as
    local-heads or local-ffn).
    """
    w_full = _hf_w(weights, f"{prefix}.weight")  # [out, in]
    b_full = _hf_b(weights, f"{prefix}.bias")
    if parallel.enabled:
        w_local = _slice_first_dim(w_full, parallel.rank, parallel.tp_size)
        b_local = (
            _slice_first_dim(b_full, parallel.rank, parallel.tp_size)
            if b_full is not None
            else None
        )
    else:
        w_local = w_full
        b_local = b_full
    local_out = w_local.shape[0]
    return _dense._add_linear_2d(network, x_2d, in_dim, local_out, w_local, b_local)


def _linear_row_parallel_2d(
    network,
    x_2d,
    local_in_dim: int,
    out_dim: int,
    weights: Mapping[str, "np.ndarray"],
    prefix: str,
    parallel: ParallelConfig,
):
    """Row-parallel Linear on [N, local_in_dim] -> [N, out_dim] + all-reduce.

    Slices the HF weight ``[out, in]`` along axis=1 (input axis) so that
    each rank only multiplies its local input slice with its local input
    columns of the weight. The result is the partial output sum on this
    rank; we then all-reduce SUM across ranks. The bias is added AFTER
    the all-reduce so it isn't double-counted; it stays replicated.
    """
    w_full = _hf_w(weights, f"{prefix}.weight")  # [out, in]
    b_full = _hf_b(weights, f"{prefix}.bias")
    if parallel.enabled:
        # axis=1 of [out, in] is the input axis -> shard by tp_size.
        parts = np.array_split(w_full, parallel.tp_size, axis=1)
        w_local = np.ascontiguousarray(parts[parallel.rank])
    else:
        w_local = w_full
    # Matmul first WITHOUT bias; add bias after all-reduce so the bias
    # isn't summed tp_size times.
    partial = _dense._add_linear_2d(
        network, x_2d, local_in_dim, out_dim, w_local, None,
    )
    if parallel.enabled:
        partial = add_all_reduce_sum(network, partial, parallel.tp_size)
    if b_full is not None:
        b_const = _dense._add_constant_reduced(
            network, (1, out_dim), b_full,
        )
        partial = network.add_elementwise(
            partial, b_const, trt.ElementWiseOperation.SUM,
        ).get_output(0)
    return partial


def _tp_qknorm_weight(
    weight_fp32: np.ndarray,
    local_num_heads: int,
    head_dim: int,
    parallel: ParallelConfig,
) -> np.ndarray:
    """Return the rank-local per-head norm weight.

    Qwen-Image stores ``attn.norm_q/norm_k`` as a [head_dim] vector that
    is broadcast across all heads (same elementwise scale for every
    head), so per-rank we keep the full vector. If a future variant
    stored a [num_heads * head_dim] vector instead, we'd need to slice
    along the head axis to match the local head count — mirrors what the
    Wan TP DiT helper does.
    """
    if not parallel.enabled:
        return weight_fp32
    if weight_fp32.size == head_dim:
        # Per-head broadcast vector: shared, keep replicated on every
        # rank (matches diffusers QwenImageRMSNorm which has a single
        # ``[head_dim]`` gamma applied across heads).
        return weight_fp32
    # [num_heads * head_dim] layout: shard along the head axis.
    return _slice_first_dim(
        weight_fp32.reshape(-1, head_dim), parallel.rank, parallel.tp_size,
    ).reshape(local_num_heads * head_dim)


# ---------------------------------------------------------------------------
# QKV + qk-norm + RoPE, but column-sharded along the output (head) axis.
#
# This is the TP equivalent of ``_dense._add_qkv_with_norm_and_rope``;
# returns the local-heads Q/K/V each shaped [B=1, S, local_heads * head_dim].
# ---------------------------------------------------------------------------


def _add_qkv_with_norm_and_rope_tp(
    network,
    x_3d,
    weights: Mapping[str, "np.ndarray"],
    *,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    q_key: str,
    k_key: str,
    v_key: str,
    norm_q_key: "str | None",
    norm_k_key: "str | None",
    rms_eps: float,
    cos_2d,
    sin_2d,
    parallel: ParallelConfig,
):
    """Compute QKV + qk-norm + RoPE on the rank-local heads."""
    if parallel.enabled:
        local_num_heads = num_heads // parallel.tp_size
    else:
        local_num_heads = num_heads
    local_dim = local_num_heads * head_dim

    flat = network.add_shuffle(x_3d)
    flat.reshape_dims = (seq_len, hidden_size)
    x_flat = flat.get_output(0)

    q = _linear_col_parallel_2d(
        network, x_flat, hidden_size, num_heads * head_dim, weights, q_key, parallel,
    )
    k = _linear_col_parallel_2d(
        network, x_flat, hidden_size, num_heads * head_dim, weights, k_key, parallel,
    )
    v = _linear_col_parallel_2d(
        network, x_flat, hidden_size, num_heads * head_dim, weights, v_key, parallel,
    )

    def _to_3d(t):
        s = network.add_shuffle(t)
        s.reshape_dims = (1, seq_len, local_dim)
        return s.get_output(0)

    q_3d = _to_3d(q)
    k_3d = _to_3d(k)
    v_3d = _to_3d(v)

    # qk-norm: per-head RMSNorm with broadcast gamma over local heads.
    if norm_q_key is not None and f"{norm_q_key}.weight" in weights:
        gamma_full = np.asarray(weights[f"{norm_q_key}.weight"], dtype=np.float32)
        gamma_local = _tp_qknorm_weight(gamma_full, local_num_heads, head_dim, parallel)
        q_3d = _dense._add_rms_norm_per_head(
            network, q_3d, local_num_heads, head_dim, gamma_local, rms_eps, seq_len,
        )
    if norm_k_key is not None and f"{norm_k_key}.weight" in weights:
        gamma_full = np.asarray(weights[f"{norm_k_key}.weight"], dtype=np.float32)
        gamma_local = _tp_qknorm_weight(gamma_full, local_num_heads, head_dim, parallel)
        k_3d = _dense._add_rms_norm_per_head(
            network, k_3d, local_num_heads, head_dim, gamma_local, rms_eps, seq_len,
        )

    # RoPE on Q and K. cos_2d / sin_2d are [seq_len, head_dim] and apply
    # the same rotation to every head -- they stay replicated.
    q_3d = _dense._add_rope_pair(
        network, q_3d, cos_2d, sin_2d, local_num_heads, head_dim, seq_len,
    )
    k_3d = _dense._add_rope_pair(
        network, k_3d, cos_2d, sin_2d, local_num_heads, head_dim, seq_len,
    )
    return q_3d, k_3d, v_3d


# ---------------------------------------------------------------------------
# MLP block, TP-sharded: net.0.proj column-parallel, net.2 row-parallel.
# ---------------------------------------------------------------------------


def _add_mlp_block_tp(
    network,
    x_3d,
    *,
    hidden_size: int,
    intermediate_size: int,
    weights: Mapping[str, "np.ndarray"],
    prefix: str,
    seq_len: int,
    parallel: ParallelConfig,
):
    """FeedForward with TP: Linear -> GELU(tanh) -> Linear with all-reduce."""
    if parallel.enabled:
        local_intermediate = intermediate_size // parallel.tp_size
    else:
        local_intermediate = intermediate_size

    flat = network.add_shuffle(x_3d)
    flat.reshape_dims = (seq_len, hidden_size)

    h = _linear_col_parallel_2d(
        network, flat.get_output(0), hidden_size, intermediate_size,
        weights, f"{prefix}.net.0.proj", parallel,
    )
    h = graph_ops.add_gelu_tanh(network, h)
    h = _linear_row_parallel_2d(
        network, h, local_intermediate, hidden_size,
        weights, f"{prefix}.net.2", parallel,
    )
    unflat = network.add_shuffle(h)
    unflat.reshape_dims = (1, seq_len, hidden_size)
    return unflat.get_output(0)


# ---------------------------------------------------------------------------
# One TP joint block. Mirrors the math of
# ``_dense._add_joint_block_graph`` but uses the TP-aware QKV/MLP/out-proj
# primitives above.
# ---------------------------------------------------------------------------


def _add_joint_block_graph_tp(
    network,
    *,
    img_3d,
    txt_3d,
    temb_2d,
    cos_img,
    sin_img,
    cos_txt,
    sin_txt,
    weights: Mapping[str, "np.ndarray"],
    weights_prefix: str,
    cfg: "_dense.JointBlockConfig",
    n_img: int,
    n_text: int,
    batch_size: int,
    parallel: ParallelConfig,
):
    """Build one Qwen-Image MMDiT joint block with TP sharding."""
    if batch_size != 1:
        raise NotImplementedError(
            "_add_joint_block_graph_tp currently supports batch_size=1 only"
        )
    H = cfg.num_attention_heads
    D = cfg.attention_head_dim
    dim = cfg.hidden_size
    if parallel.enabled:
        local_num_heads = H // parallel.tp_size
    else:
        local_num_heads = H

    class _PrefixedWeights:
        def __init__(self, base: Mapping[str, "np.ndarray"], prefix: str):
            self._base = base
            self._prefix = prefix

        def __getitem__(self, key: str):
            return self._base[f"{self._prefix}{key}"]

        def get(self, key: str, default=None):
            return self._base.get(f"{self._prefix}{key}", default)

        def __contains__(self, key: str) -> bool:
            return f"{self._prefix}{key}" in self._base

    prefixed = _PrefixedWeights(weights, weights_prefix)

    def _w(key: str) -> np.ndarray:
        return np.asarray(weights[f"{weights_prefix}{key}"], dtype=np.float32)

    def _w_opt(key: str):
        v = weights.get(f"{weights_prefix}{key}")
        return None if v is None else np.asarray(v, dtype=np.float32)

    # ----- AdaLN modulation: replicated linear projection from temb.
    img_mod_w = _w("img_mod.1.weight")
    img_mod_b = _w("img_mod.1.bias")
    txt_mod_w = _w("txt_mod.1.weight")
    txt_mod_b = _w("txt_mod.1.bias")

    temb_silu = graph_ops.add_silu(network, temb_2d)
    img_mod_params = _dense._add_linear_2d(
        network, temb_silu, dim, 6 * dim, img_mod_w, img_mod_b,
    )
    txt_mod_params = _dense._add_linear_2d(
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

    # ----- norm1 + modulate per stream (replicated).
    img_normed = _dense._add_layernorm_no_affine_3d(
        network, img_3d, dim, cfg.layer_norm_eps,
    )
    img_modulated = _dense._add_modulate(
        network, img_normed, img_shift_msa, img_scale_msa, dim,
    )
    txt_normed = _dense._add_layernorm_no_affine_3d(
        network, txt_3d, dim, cfg.layer_norm_eps,
    )
    txt_modulated = _dense._add_modulate(
        network, txt_normed, txt_shift_msa, txt_scale_msa, dim,
    )

    # ----- QKV + qk-norm + RoPE (col-parallel on output / heads).
    img_q, img_k, img_v = _add_qkv_with_norm_and_rope_tp(
        network, img_modulated, prefixed,
        hidden_size=dim, num_heads=H, head_dim=D, seq_len=n_img,
        q_key="attn.to_q", k_key="attn.to_k", v_key="attn.to_v",
        norm_q_key="attn.norm_q", norm_k_key="attn.norm_k",
        rms_eps=cfg.rms_norm_eps, cos_2d=cos_img, sin_2d=sin_img,
        parallel=parallel,
    )
    txt_q, txt_k, txt_v = _add_qkv_with_norm_and_rope_tp(
        network, txt_modulated, prefixed,
        hidden_size=dim, num_heads=H, head_dim=D, seq_len=n_text,
        q_key="attn.add_q_proj", k_key="attn.add_k_proj", v_key="attn.add_v_proj",
        norm_q_key="attn.norm_added_q", norm_k_key="attn.norm_added_k",
        rms_eps=cfg.rms_norm_eps, cos_2d=cos_txt, sin_2d=sin_txt,
        parallel=parallel,
    )

    # ----- joint attention on rank-local heads.
    attn_txt_3d, attn_img_3d = _dense._add_joint_attention(
        network, img_q, img_k, img_v, txt_q, txt_k, txt_v,
        num_heads=local_num_heads, head_dim=D, n_img=n_img, n_txt=n_text,
    )

    # ----- output projections (row-parallel + all-reduce).
    def _out_proj_tp(x_3d, key_suffix: str, seq_len: int):
        flat = network.add_shuffle(x_3d)
        flat.reshape_dims = (seq_len, local_num_heads * D)
        proj = _linear_row_parallel_2d(
            network, flat.get_output(0), local_num_heads * D, dim,
            weights, f"{weights_prefix}{key_suffix}", parallel,
        )
        unflat = network.add_shuffle(proj)
        unflat.reshape_dims = (1, seq_len, dim)
        return unflat.get_output(0)

    img_attn_out = _out_proj_tp(attn_img_3d, "attn.to_out.0", n_img)
    txt_attn_out = _out_proj_tp(attn_txt_3d, "attn.to_add_out", n_text)

    # ----- gated residual (post-attention) -- replicated.
    hs_img = _dense._add_gate_residual(network, img_3d, img_gate_msa, img_attn_out, dim)
    hs_txt = _dense._add_gate_residual(network, txt_3d, txt_gate_msa, txt_attn_out, dim)

    # ----- norm2 + modulate(mod2) + MLP + gated residual.
    img_n2 = _dense._add_layernorm_no_affine_3d(
        network, hs_img, dim, cfg.layer_norm_eps,
    )
    img_mod2_out = _dense._add_modulate(
        network, img_n2, img_shift_mlp, img_scale_mlp, dim,
    )
    img_mlp_out = _add_mlp_block_tp(
        network, img_mod2_out,
        hidden_size=dim, intermediate_size=cfg.intermediate_size,
        weights=weights, prefix=f"{weights_prefix}img_mlp", seq_len=n_img,
        parallel=parallel,
    )
    img_out = _dense._add_gate_residual(network, hs_img, img_gate_mlp, img_mlp_out, dim)

    txt_n2 = _dense._add_layernorm_no_affine_3d(
        network, hs_txt, dim, cfg.layer_norm_eps,
    )
    txt_mod2_out = _dense._add_modulate(
        network, txt_n2, txt_shift_mlp, txt_scale_mlp, dim,
    )
    txt_mlp_out = _add_mlp_block_tp(
        network, txt_mod2_out,
        hidden_size=dim, intermediate_size=cfg.intermediate_size,
        weights=weights, prefix=f"{weights_prefix}txt_mlp", seq_len=n_text,
        parallel=parallel,
    )
    txt_out = _dense._add_gate_residual(network, hs_txt, txt_gate_mlp, txt_mlp_out, dim)
    _ = _w_opt  # silence unused-helper lint; kept symmetric with dense path
    return img_out, txt_out


# ---------------------------------------------------------------------------
# Public TP builder.
#
# Same signature as ``_dense.build_qwen_image_dit_engine`` plus a
# ``parallel_config`` keyword. When parallel_config is disabled the
# output engine is bit-equivalent (or as close as TRT permits given
# layer-ordering) to the dense build.
# ---------------------------------------------------------------------------


def build_qwen_image_dit_engine(
    cfg: "_dense.QwenImageDiTConfig",
    weights: Mapping[str, "np.ndarray"],
    out_path: PathLike,
    *,
    h_lat: int,
    w_lat: int,
    n_text: int,
    image_token_shapes: "list[tuple[int, int]] | None" = None,
    batch_size: int = 1,
    verbose: bool = False,
    parallel_config: "ParallelConfig | None" = None,
) -> Path:
    """Build the full Qwen-Image MMDiT denoiser TRT plan with TP sharding.

    See :func:`qwen_image_dit_builder.build_qwen_image_dit_engine` for the
    base contract. The only TP-specific knob is ``parallel_config``;
    when its ``enabled`` is False this builder is functionally equivalent
    to the dense one.
    """
    if batch_size != 1:
        raise NotImplementedError(
            "build_qwen_image_dit_engine (TP) currently supports batch_size=1 only"
        )
    if cfg.num_attention_heads * cfg.attention_head_dim != cfg.hidden_size:
        raise ValueError(
            f"hidden_size ({cfg.hidden_size}) != num_heads "
            f"({cfg.num_attention_heads}) * head_dim "
            f"({cfg.attention_head_dim})"
        )
    if sum(cfg.rope_axes_dim) != cfg.attention_head_dim:
        raise ValueError(
            f"sum(rope_axes_dim) ({sum(cfg.rope_axes_dim)}) != head_dim "
            f"({cfg.attention_head_dim})"
        )
    if cfg.guidance_embeds:
        raise NotImplementedError(
            "build_qwen_image_dit_engine (TP) does not support guidance_embeds=True"
        )

    parallel = normalize_parallel_config(parallel_config)
    validate_dit_tp(
        dim=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        ffn_dim=cfg.intermediate_size,
        parallel=parallel,
        feature="Qwen-Image MMDiT tensor parallel",
    )

    _dense._validate_full_weights(cfg, weights)

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

    cos_table_np, sin_table_np = _dense._precompute_qwen_rope_tables_for_shapes(
        list(cfg.rope_axes_dim), image_token_shapes, n_text, cfg.rope_theta,
    )
    seq_total = n_img + n_text
    assert cos_table_np.shape == (seq_total, head_dim), (
        f"rope cos shape {cos_table_np.shape} != ({seq_total}, {head_dim})"
    )

    builder, config, network = _dense._make_builder(verbose)

    img_patched = network.add_input(
        "img_patched", trt.float32, (batch_size, n_img, in_ch),
    )
    txt_hidden = network.add_input(
        "txt_hidden", trt.float32, (batch_size, n_text, txt_d),
    )
    timestep = network.add_input("timestep", trt.float32, (batch_size,))

    # img_in / txt_norm / txt_in / time_text_embed all stay replicated --
    # they're small and bake into every rank identically. This mirrors
    # the Wan/FLUX TP policy where embedders/norms are kept replicated.
    img_in_w = np.asarray(weights["img_in.weight"], dtype=np.float32)
    img_in_b = np.asarray(weights["img_in.bias"], dtype=np.float32)
    img_tokens = _dense._add_linear_3d(
        network, img_patched, in_ch, H_dim, img_in_w, img_in_b,
        seq_len=n_img, batch_size=batch_size,
    )

    txt_norm_gamma = np.asarray(weights["txt_norm.weight"], dtype=np.float32)
    txt_normed = _dense._add_rms_norm_last_dim_3d(
        network, txt_hidden, txt_d, txt_norm_gamma, cfg.rms_norm_eps,
    )
    txt_in_w = np.asarray(weights["txt_in.weight"], dtype=np.float32)
    txt_in_b = np.asarray(weights["txt_in.bias"], dtype=np.float32)
    txt_tokens = _dense._add_linear_3d(
        network, txt_normed, txt_d, H_dim, txt_in_w, txt_in_b,
        seq_len=n_text, batch_size=batch_size,
    )

    temb = _dense._add_time_text_embed(
        network, timestep, weights=weights,
        in_dim=cfg.timestep_embed_dim, hidden_size=H_dim,
    )

    cos_const = network.add_constant(
        (seq_total, head_dim), trt.Weights(cos_table_np),
    ).get_output(0)
    sin_const = network.add_constant(
        (seq_total, head_dim), trt.Weights(sin_table_np),
    ).get_output(0)
    cos_img_sl = network.add_slice(
        cos_const, start=(0, 0), shape=(n_img, head_dim), stride=(1, 1),
    )
    sin_img_sl = network.add_slice(
        sin_const, start=(0, 0), shape=(n_img, head_dim), stride=(1, 1),
    )
    cos_txt_sl = network.add_slice(
        cos_const, start=(n_img, 0), shape=(n_text, head_dim), stride=(1, 1),
    )
    sin_txt_sl = network.add_slice(
        sin_const, start=(n_img, 0), shape=(n_text, head_dim), stride=(1, 1),
    )

    jb_cfg = _dense._joint_block_cfg_from(cfg)
    cur_img = img_tokens
    cur_txt = txt_tokens
    for i in range(cfg.num_joint_blocks):
        prefix = f"transformer_blocks.{i}."
        cur_img, cur_txt = _add_joint_block_graph_tp(
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
            parallel=parallel,
        )

    # AdaLayerNormContinuous + proj_out -- replicated on every rank.
    normed = _dense._add_norm_out_3d(
        network, cur_img, temb, weights=weights,
        hidden_size=H_dim, eps=cfg.layer_norm_eps, batch_size=batch_size,
    )
    proj_w = np.asarray(weights["proj_out.weight"], dtype=np.float32)
    proj_b = np.asarray(weights["proj_out.bias"], dtype=np.float32)
    proj_out_dim = out_ch * p * p
    noise = _dense._add_linear_3d(
        network, normed, H_dim, proj_out_dim, proj_w, proj_b,
        seq_len=n_img, batch_size=batch_size,
    )
    noise = _dense._to_fp32(network, noise)
    noise.name = "noise_patched"
    network.mark_output(noise)

    tp_suffix = (
        f", tp={parallel.tp_size}, rank={parallel.rank}"
        if parallel.enabled else ""
    )
    print(
        f"[qwen-image-dit] Building full denoiser engine "
        f"(B={batch_size}, n_img={n_img}, n_text={n_text}, "
        f"image_token_shapes={image_token_shapes}, "
        f"blocks={cfg.num_joint_blocks}, hidden={H_dim}, "
        f"heads={cfg.num_attention_heads}, head_dim={head_dim}, "
        f"text_d={txt_d}, in_ch={in_ch}, out_ch={out_ch}, p={p}"
        f"{tp_suffix}) -> [{batch_size}, {n_img}, {proj_out_dim}]",
        file=sys.stderr,
    )
    return _dense._serialize_and_write(
        builder, network, config, out_path, "qwen_image_dit_tp",
    )
