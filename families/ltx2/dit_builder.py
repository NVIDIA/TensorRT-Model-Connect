# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 joint audio + video DiT (``LTX2VideoTransformer3DModel``) as a TensorRT plan.

One builder serves the single-device plan (``cp_size=1``) and the rank-dynamic
context-parallel plan (``cp_size>1``, one serialized plan shared by every rank).

Engine I/O (``B`` = batch of guidance branches; 1 for the distilled model):
    Inputs:
        video_latent    [B, S, 128]     fp32  packed video latents (full sequence on every rank)
        audio_latent    [B, Sa, 128]    fp32  packed audio latents
        video_context   [B, L, 4096]    bf16  video connector output
        audio_context   [B, L, 2048]    bf16  audio connector output
        timestep        [B]             fp32  sigma * 1000 (also used as ``sigma``, cross timestep)
        stg_keep        [B]             fp32  1 = normal; 0 = spatio-temporal guidance branch
                                              (self-attention context -> V in the STG blocks)
        av_keep         [B]             fp32  1 = normal; 0 = modality-isolated branch
                                              (a2v / v2a residual adds are multiplied by 0)
    Outputs:
        video_velocity  [B, S, 128]     fp32  (full sequence on every rank)
        audio_velocity  [B, Sa, 128]    fp32

Precision: bf16 strongly typed (diffusers runs this model in bf16), with fp32 RMSNorm /
LayerNorm statistics, fp32 split RoPE, fp32 timestep sinusoids and fp32 GELU/SiLU islands.

Context parallelism (``cp_size`` ranks, contiguous token shards of ``S / cp`` rows):
    - every rank derives its index with one tiny REDUCE_SCATTER and gathers its own latent
      rows and RoPE rows;
    - video self-attention: local queries attend all keys/values. Q/K RMSNorm spans all heads
      of a token and RoPE is per token, so K and V are normed and rotated on their own rows and
      then ALL_GATHERed. For two ranks this moves the same bytes as a Ulysses head exchange and
      needs no ALL_TO_ALL. bf16 payloads cross the wire as fp16 (exact inside the fp16 range,
      which covers every activation of the reference run; see ``_ag_tokens``);
    - video -> text and audio -> video (a2v) use local queries against replicated keys;
    - video -> audio (v2a) needs every video key: each rank computes its partial softmax
      statistics (row max, row sum, unnormalized context) over its key shard in fp32, one
      ALL_GATHER (~1 MB) shares them and every rank merges them identically (exact
      log-sum-exp merge, as in flash-decoding);
    - the audio stream (126 tokens) is replicated and computed redundantly on every rank;
    - the video output rows are restored with one fp32 ALL_GATHER.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint
from .graph import Graph, build_plan, make_logger, new_network
from .layers import (
    AttnWeights,
    RopeTables,
    feed_forward,
    from_heads,
    gate_and_project_out,
    gather_rope_rows,
    ltx_attention,
    modulate,
    project_qkv,
    rope_constants,
    split_rope_freqs,
    stg_lerp,
    to_heads,
)
from .parallel import add_collective, local_row_indices

EPS = 1e-6
_FP16_SAFE_BF16_MAX = 65280.0  # largest bf16 value that is finite in fp16


@dataclass(frozen=True)
class DiTConfig:
    in_channels: int
    out_channels: int
    heads: int
    head_dim: int
    audio_in_channels: int
    audio_out_channels: int
    audio_heads: int
    audio_head_dim: int
    cross_attention_dim: int
    audio_cross_attention_dim: int
    layers: int
    cross_attn_mod: bool
    audio_cross_attn_mod: bool
    prompt_adaln: bool
    gated: bool
    audio_gated: bool
    rope_theta: float
    rope_double_precision: bool
    pos_embed_max_pos: int
    audio_pos_embed_max_pos: int
    base_height: int
    base_width: int
    vae_scale_factors: tuple[int, int, int]
    causal_offset: int
    audio_sampling_rate: int
    audio_hop_length: int
    audio_scale_factor: int
    timestep_scale_multiplier: float
    cross_attn_timestep_scale_multiplier: float

    @property
    def dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def audio_dim(self) -> int:
        return self.audio_heads * self.audio_head_dim

    @staticmethod
    def from_dict(c: dict) -> "DiTConfig":
        for key, expected in (("rope_type", "split"), ("qk_norm", "rms_norm_across_heads"),
                              ("patch_size", 1), ("patch_size_t", 1), ("audio_patch_size", 1),
                              ("audio_patch_size_t", 1), ("norm_elementwise_affine", False),
                              ("use_prompt_embeddings", False), ("activation_fn", "gelu-approximate")):
            if c.get(key, expected) != expected:
                raise NotImplementedError(f"LTX-2.5 DiT builder expects {key}={expected!r}, got {c.get(key)!r}")
        return DiTConfig(
            in_channels=int(c["in_channels"]), out_channels=int(c.get("out_channels") or c["in_channels"]),
            heads=int(c["num_attention_heads"]), head_dim=int(c["attention_head_dim"]),
            audio_in_channels=int(c["audio_in_channels"]),
            audio_out_channels=int(c.get("audio_out_channels") or c["audio_in_channels"]),
            audio_heads=int(c["audio_num_attention_heads"]), audio_head_dim=int(c["audio_attention_head_dim"]),
            cross_attention_dim=int(c["cross_attention_dim"]),
            audio_cross_attention_dim=int(c["audio_cross_attention_dim"]),
            layers=int(c["num_layers"]),
            cross_attn_mod=bool(c.get("cross_attn_mod", False)),
            audio_cross_attn_mod=bool(c.get("audio_cross_attn_mod", False)),
            prompt_adaln=bool(c.get("use_prompt_adaln_single", True)),
            gated=bool(c.get("gated_attn", False)), audio_gated=bool(c.get("audio_gated_attn", False)),
            rope_theta=float(c.get("rope_theta", 10000.0)),
            rope_double_precision=bool(c.get("rope_double_precision", True)),
            pos_embed_max_pos=int(c.get("pos_embed_max_pos", 20)),
            audio_pos_embed_max_pos=int(c.get("audio_pos_embed_max_pos", 20)),
            base_height=int(c.get("base_height", 2048)), base_width=int(c.get("base_width", 2048)),
            vae_scale_factors=tuple(int(v) for v in c.get("vae_scale_factors", (8, 32, 32))),
            causal_offset=int(c.get("causal_offset", 1)),
            audio_sampling_rate=int(c.get("audio_sampling_rate", 16000)),
            audio_hop_length=int(c.get("audio_hop_length", 160)),
            audio_scale_factor=int(c.get("audio_scale_factor", 4)),
            timestep_scale_multiplier=float(c.get("timestep_scale_multiplier", 1000)),
            cross_attn_timestep_scale_multiplier=float(c.get("cross_attn_timestep_scale_multiplier", 1000)),
        )


@dataclass(frozen=True)
class DiTShape:
    batch: int
    latent_frames: int
    latent_height: int
    latent_width: int
    audio_frames: int
    text_len: int
    fps: float

    @property
    def video_tokens(self) -> int:
        return self.latent_frames * self.latent_height * self.latent_width


def audio_latent_frames(num_frames: int, fps: float, *, sampling_rate: int = 16000, hop_length: int = 160,
                        temporal_compression: int = 4) -> int:
    """``LTX2Pipeline``: ``round(num_frames / fps * sr / hop / compression)``."""
    return int(round(num_frames / float(fps) * (float(sampling_rate) / float(hop_length) /
                                                float(temporal_compression))))


# ---------------------------------------------------------------------- RoPE grids (diffusers, fp32)


def video_midpoints(cfg: DiTConfig, shape: DiTShape) -> np.ndarray:
    """``prepare_video_coords`` patch midpoints: ``[3, S]`` fp32 (time in seconds, space in pixels)."""
    f32 = np.float32
    gf, gh, gw = np.meshgrid(np.arange(shape.latent_frames, dtype=f32), np.arange(shape.latent_height, dtype=f32),
                             np.arange(shape.latent_width, dtype=f32), indexing="ij")
    start = np.stack([gf, gh, gw], 0).reshape(3, -1)
    end = start + f32(1.0)
    scale = np.asarray(cfg.vae_scale_factors, dtype=f32).reshape(3, 1)
    ps, pe = start * scale, end * scale
    t = f32(cfg.vae_scale_factors[0])
    ps[0] = np.maximum(ps[0] + f32(cfg.causal_offset) - t, f32(0.0)) / f32(shape.fps)
    pe[0] = np.maximum(pe[0] + f32(cfg.causal_offset) - t, f32(0.0)) / f32(shape.fps)
    return ((ps + pe) / f32(2.0)).astype(f32)


def video_grid(cfg: DiTConfig, shape: DiTShape) -> np.ndarray:
    """Video self-attention RoPE positions: midpoints / (max frames, base height, base width): ``[S, 3]``."""
    maxpos = np.asarray([cfg.pos_embed_max_pos, cfg.base_height, cfg.base_width], dtype=np.float32).reshape(3, 1)
    return (video_midpoints(cfg, shape) / maxpos).T.astype(np.float32)


def audio_grid(cfg: DiTConfig, shape: DiTShape, max_pos: int) -> np.ndarray:
    """``prepare_audio_coords`` + midpoint + division: ``[Sa, 1]`` fp32 (seconds / max_pos)."""
    f32 = np.float32
    gf = np.arange(shape.audio_frames, dtype=f32)
    sf = f32(cfg.audio_scale_factor)
    start = np.maximum(gf * sf + f32(cfg.causal_offset) - sf, f32(0.0))
    start = start * f32(cfg.audio_hop_length) / f32(cfg.audio_sampling_rate)
    end = np.maximum((gf + f32(1.0)) * sf + f32(cfg.causal_offset) - sf, f32(0.0))
    end = end * f32(cfg.audio_hop_length) / f32(cfg.audio_sampling_rate)
    mid = (start + end) / f32(2.0)
    return (mid / f32(max_pos)).reshape(-1, 1).astype(f32)


def rope_tables(cfg: DiTConfig, shape: DiTShape) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    vg = video_grid(cfg, shape)
    cross_max = max(cfg.pos_embed_max_pos, cfg.audio_pos_embed_max_pos)
    # cross_attn_rope sees video_coords[:, 0:1] (time only) with max position max(video, audio).
    v_time = (video_midpoints(cfg, shape)[0:1] / np.float32(cross_max)).T.astype(np.float32)
    th, dp = cfg.rope_theta, cfg.rope_double_precision
    return {
        "video": split_rope_freqs(vg, cfg.dim, cfg.heads, th, dp),
        "audio": split_rope_freqs(audio_grid(cfg, shape, cfg.audio_pos_embed_max_pos), cfg.audio_dim,
                                  cfg.audio_heads, th, dp),
        "ca_video": split_rope_freqs(v_time, cfg.audio_cross_attention_dim, cfg.heads, th, dp),
        "ca_audio": split_rope_freqs(audio_grid(cfg, shape, cross_max), cfg.audio_cross_attention_dim,
                                     cfg.audio_heads, th, dp),
    }


# ---------------------------------------------------------------------- AdaLN helpers


def _adaln(g: Graph, ckpt: Checkpoint, prefix: str, t_sin_bf16):
    """``LTX2AdaLayerNormSingle``: returns (mod ``[B, n*D]``, embedded_timestep ``[B, D]``) in bf16."""
    p = f"{prefix}.emb.timestep_embedder"
    e = g.linear(t_sin_bf16, ckpt.get(f"{p}.linear_1.weight"), ckpt.get(f"{p}.linear_1.bias"))
    e = g.linear(g.silu(e), ckpt.get(f"{p}.linear_2.weight"), ckpt.get(f"{p}.linear_2.bias"))
    mod = g.linear(g.silu(e), ckpt.get(f"{prefix}.linear.weight"), ckpt.get(f"{prefix}.linear.bias"))
    return mod, e


def _mod_params(g: Graph, table: np.ndarray, temb, batch: int):
    """``table[None, None] + temb.reshape(B, 1, n, D)`` unbound into n tensors ``[B, 1, D]`` (bf16)."""
    n, d = (int(s) for s in table.shape)
    vals = g.add(g.reshape(temb, (batch, 1, n, d)), g.const(table.reshape(1, 1, n, d), trt.bfloat16))
    return [g.reshape(g.slice(vals, (0, 0, i, 0), (batch, 1, 1, d)), (batch, 1, d)) for i in range(n)]


# ---------------------------------------------------------------------- context-parallel attention


def _ag_tokens(g: Graph, x, cp: int):
    """ALL_GATHER of ``[B, S/cp, D]`` token shards into ``[B, S, D]`` (bf16 carried as fp16)."""
    b, s_loc, d = (int(v) for v in x.shape)
    t = g.transpose(x, (1, 0, 2))  # [S/cp, B, D]: ALL_GATHER concatenates dim 0
    if t.dtype == trt.bfloat16:
        t = g.maximum(g.minimum(t, g.scalar(_FP16_SAFE_BF16_MAX, trt.bfloat16, 3)),
                      g.scalar(-_FP16_SAFE_BF16_MAX, trt.bfloat16, 3))
        out = g.cast(add_collective(g.net, g.cast(t, trt.float16), trt.CollectiveOperation.ALL_GATHER, cp),
                     trt.bfloat16)
    else:
        out = add_collective(g.net, t, trt.CollectiveOperation.ALL_GATHER, cp)
    return g.transpose(out, (1, 0, 2))


def _video_self_attention(g: Graph, aw: AttnWeights, x_n, *, heads: int, rope: RopeTables, cp: int,
                          gated: bool, stg_keep):
    if cp == 1:
        return ltx_attention(g, aw, x_n, x_n, heads=heads, eps=EPS, q_rope=rope, k_rope=rope, gated=gated,
                             stg_keep=stg_keep)
    # Local queries attend the gathered keys/values; Q/K RMSNorm and RoPE are row-local.
    q, k, v = project_qkv(g, aw, x_n, x_n, eps=EPS, q_rope=rope, k_rope=rope)
    kf, vf = _ag_tokens(g, k, cp), _ag_tokens(g, v, cp)
    ctx = from_heads(g, g.attention(to_heads(g, q, heads), to_heads(g, kf, heads), to_heads(g, vf, heads)))
    if stg_keep is not None:
        ctx = stg_lerp(g, v, ctx, stg_keep)
    return gate_and_project_out(g, aw, ctx, x_n, heads, gated=gated)


def _v2a_attention(g: Graph, aw: AttnWeights, a_q, v_kv, *, heads: int, q_rope: RopeTables,
                   k_rope: RopeTables, cp: int, gated: bool):
    """video -> audio cross attention (Q audio, K/V video)."""
    if cp == 1:
        return ltx_attention(g, aw, a_q, v_kv, heads=heads, eps=EPS, q_rope=q_rope, k_rope=k_rope, gated=gated)
    q, k, v = project_qkv(g, aw, a_q, v_kv, eps=EPS, q_rope=q_rope, k_rope=k_rope)
    q4 = g.cast(to_heads(g, q, heads), trt.float32)  # [B, H, Sa, d]
    k4 = g.cast(to_heads(g, k, heads), trt.float32)  # [B, H, S/cp, d]
    v4 = g.cast(to_heads(g, v, heads), trt.float32)
    b, h, sa, d = (int(s) for s in q4.shape)
    q4 = g.mul(q4, g.scalar(1.0 / np.sqrt(d), trt.float32, 4))
    scores = g.net.add_matrix_multiply(q4, trt.MatrixOperation.NONE, k4,
                                       trt.MatrixOperation.TRANSPOSE).get_output(0)  # [B,H,Sa,S/cp]
    m = g.reduce(scores, trt.ReduceOperation.MAX, 3)  # [B,H,Sa,1]
    p = g.unary(g.sub(scores, m), trt.UnaryOperation.EXP)
    l_sum = g.reduce(p, trt.ReduceOperation.SUM, 3)
    num = g.net.add_matrix_multiply(p, trt.MatrixOperation.NONE, v4, trt.MatrixOperation.NONE).get_output(0)
    stats = g.reshape(g.concat([m, l_sum, num], axis=3), (1, b, h, sa, d + 2))
    allst = add_collective(g.net, stats, trt.CollectiveOperation.ALL_GATHER, cp)  # [cp, B, H, Sa, d+2]
    # Merge over a trailing rank axis: slicing / reducing the gathered leading axis directly is
    # mis-compiled by TensorRT-RTX 1.7.1 (wrong values), the transposed form is exact.
    t = g.transpose(allst, (1, 2, 3, 4, 0))  # [B, H, Sa, d+2, cp]
    m_all = g.slice(t, (0, 0, 0, 0, 0), (b, h, sa, 1, cp))
    l_all = g.slice(t, (0, 0, 0, 1, 0), (b, h, sa, 1, cp))
    n_all = g.slice(t, (0, 0, 0, 2, 0), (b, h, sa, d, cp))
    m_max = g.reduce(m_all, trt.ReduceOperation.MAX, 4)  # [B, H, Sa, 1, 1]
    w = g.unary(g.sub(m_all, m_max), trt.UnaryOperation.EXP)
    denom = g.reduce(g.mul(l_all, w), trt.ReduceOperation.SUM, 4)
    numer = g.reduce(g.mul(n_all, w), trt.ReduceOperation.SUM, 4)
    ctx4 = g.reshape(g.div(numer, denom), (b, h, sa, d))
    ctx = from_heads(g, g.cast(ctx4, trt.bfloat16))
    return gate_and_project_out(g, aw, ctx, a_q, heads, gated=gated)


# ---------------------------------------------------------------------- network


def _scalar_like(g: Graph, x, value: float):
    return g.scalar(value, x.dtype, len(x.shape))


def add_dit(g: Graph, ckpt: Checkpoint, cfg: DiTConfig, shape: DiTShape, inputs: dict, *, cp: int = 1,
            stg_blocks: tuple[int, ...] = (), num_layers: int | None = None):
    """Adds the DiT; returns (video_velocity ``[B, S_local, C]`` bf16, audio_velocity ``[B, Sa, C]`` bf16)."""
    B = shape.batch
    S = shape.video_tokens
    if S % cp:
        raise ValueError(f"video tokens {S} are not divisible by context_parallel_size {cp}")
    for name, h in (("video", cfg.heads), ("audio", cfg.audio_heads)):
        if h % cp:
            raise ValueError(f"{name} heads {h} are not divisible by context_parallel_size {cp}")
    s_loc = S // cp
    D, Da = cfg.dim, cfg.audio_dim
    n_layers = cfg.layers if num_layers is None else num_layers

    tables = rope_tables(cfg, shape)
    rope_v = rope_constants(g, *tables["video"])
    rope_ca_v = rope_constants(g, *tables["ca_video"])
    rope_a = rope_constants(g, *tables["audio"])
    rope_ca_a = rope_constants(g, *tables["ca_audio"])

    video_latent = g.cast(inputs["video_latent"], trt.bfloat16)
    if cp > 1:
        rows = local_row_indices(g, cp=cp, local_rows=s_loc)
        video_latent = g.gather(video_latent, rows, 1)
        rope_v = gather_rope_rows(g, rope_v, rows)
        rope_ca_v = gather_rope_rows(g, rope_ca_v, rows)
    audio_latent = g.cast(inputs["audio_latent"], trt.bfloat16)

    x = g.linear(video_latent, ckpt.get("proj_in.weight"), ckpt.get("proj_in.bias"))  # [B, S_loc, D]
    a = g.linear(audio_latent, ckpt.get("audio_proj_in.weight"), ckpt.get("audio_proj_in.bias"))

    t = inputs["timestep"]
    t_sin = g.cast(g.timestep_sinusoid(t), trt.bfloat16)
    gate_factor = cfg.cross_attn_timestep_scale_multiplier / cfg.timestep_scale_multiplier
    t_gate_sin = t_sin if gate_factor == 1.0 else g.cast(
        g.timestep_sinusoid(g.mul(t, g.scalar(gate_factor, trt.float32, 1))), trt.bfloat16)
    temb_v, emb_v = _adaln(g, ckpt, "time_embed", t_sin)
    temb_a, emb_a = _adaln(g, ckpt, "audio_time_embed", t_sin)
    temb_pv = temb_pa = None
    if (cfg.cross_attn_mod or cfg.audio_cross_attn_mod) and cfg.prompt_adaln:
        temb_pv, _ = _adaln(g, ckpt, "prompt_adaln", t_sin)
        temb_pa, _ = _adaln(g, ckpt, "audio_prompt_adaln", t_sin)
    # use_cross_timestep=True with one shared sigma: every cross-modal modulation sees t.
    ca_v, _ = _adaln(g, ckpt, "av_cross_attn_video_scale_shift", t_sin)
    gate_v, _ = _adaln(g, ckpt, "av_cross_attn_video_a2v_gate", t_gate_sin)
    ca_a, _ = _adaln(g, ckpt, "av_cross_attn_audio_scale_shift", t_sin)
    gate_a, _ = _adaln(g, ckpt, "av_cross_attn_audio_v2a_gate", t_gate_sin)

    ctx_v_in = inputs["video_context"]
    ctx_a_in = inputs["audio_context"]
    stg_keep = inputs.get("stg_keep")
    av_keep = g.reshape(g.cast(inputs["av_keep"], trt.bfloat16), (B, 1, 1))

    for i in range(n_layers):
        p = f"transformer_blocks.{i}"
        stg = stg_keep if i in stg_blocks else None
        vm = _mod_params(g, ckpt.get(f"{p}.scale_shift_table"), temb_v, B)
        am = _mod_params(g, ckpt.get(f"{p}.audio_scale_shift_table"), temb_a, B)
        # 1. self-attention
        xn = modulate(g, g.rms_norm(x, None, EPS), vm[1], vm[0])
        x = g.add(x, g.mul(_video_self_attention(g, AttnWeights(ckpt, f"{p}.attn1"), xn, heads=cfg.heads,
                                                 rope=rope_v, cp=cp, gated=cfg.gated, stg_keep=stg), vm[2]))
        an = modulate(g, g.rms_norm(a, None, EPS), am[1], am[0])
        a = g.add(a, g.mul(ltx_attention(g, AttnWeights(ckpt, f"{p}.audio_attn1"), an, an, heads=cfg.audio_heads,
                                         eps=EPS, q_rope=rope_a, k_rope=rope_a, gated=cfg.audio_gated,
                                         stg_keep=stg), am[2]))
        # 2. text cross-attention
        ctx_v, ctx_a = ctx_v_in, ctx_a_in
        if cfg.cross_attn_mod or cfg.audio_cross_attn_mod:
            if temb_pv is not None:
                pv = _mod_params(g, ckpt.get(f"{p}.prompt_scale_shift_table"), temb_pv, B)
                pa = _mod_params(g, ckpt.get(f"{p}.audio_prompt_scale_shift_table"), temb_pa, B)
            else:
                tv = ckpt.get(f"{p}.prompt_scale_shift_table")
                ta = ckpt.get(f"{p}.audio_prompt_scale_shift_table")
                pv = [g.const(tv[j].reshape(1, 1, -1), trt.bfloat16) for j in range(2)]
                pa = [g.const(ta[j].reshape(1, 1, -1), trt.bfloat16) for j in range(2)]
            ctx_v = modulate(g, ctx_v_in, pv[1], pv[0])
            ctx_a = modulate(g, ctx_a_in, pa[1], pa[0])
        xn = g.rms_norm(x, None, EPS)
        if cfg.cross_attn_mod:
            xn = modulate(g, xn, vm[7], vm[6])
        out = ltx_attention(g, AttnWeights(ckpt, f"{p}.attn2"), xn, ctx_v, heads=cfg.heads, eps=EPS, gated=cfg.gated)
        if cfg.cross_attn_mod:
            out = g.mul(out, vm[8])
        x = g.add(x, out)
        an = g.rms_norm(a, None, EPS)
        if cfg.audio_cross_attn_mod:
            an = modulate(g, an, am[7], am[6])
        out = ltx_attention(g, AttnWeights(ckpt, f"{p}.audio_attn2"), an, ctx_a, heads=cfg.audio_heads, eps=EPS,
                            gated=cfg.audio_gated)
        if cfg.audio_cross_attn_mod:
            out = g.mul(out, am[8])
        a = g.add(a, out)
        # 3. audio <-> video cross-attention
        xn = g.rms_norm(x, None, EPS)
        an = g.rms_norm(a, None, EPS)
        vt = ckpt.get(f"{p}.video_a2v_cross_attn_scale_shift_table")
        at = ckpt.get(f"{p}.audio_a2v_cross_attn_scale_shift_table")
        v_a2v_scale, v_a2v_shift, v_v2a_scale, v_v2a_shift = _mod_params(g, vt[:4], ca_v, B)
        a_a2v_scale, a_a2v_shift, a_v2a_scale, a_v2a_shift = _mod_params(g, at[:4], ca_a, B)
        a2v_gate = _mod_params(g, vt[4:], gate_v, B)[0]
        v2a_gate = _mod_params(g, at[4:], gate_a, B)[0]
        a2v = ltx_attention(g, AttnWeights(ckpt, f"{p}.audio_to_video_attn"), modulate(g, xn, v_a2v_scale, v_a2v_shift),
                            modulate(g, an, a_a2v_scale, a_a2v_shift), heads=cfg.audio_heads, eps=EPS,
                            q_rope=rope_ca_v, k_rope=rope_ca_a, gated=cfg.gated)
        v2a = _v2a_attention(g, AttnWeights(ckpt, f"{p}.video_to_audio_attn"),
                             modulate(g, an, a_v2a_scale, a_v2a_shift), modulate(g, xn, v_v2a_scale, v_v2a_shift),
                             heads=cfg.audio_heads, q_rope=rope_ca_a, k_rope=rope_ca_v, cp=cp, gated=cfg.audio_gated)
        x = g.add(x, g.mul(g.mul(a2v_gate, a2v), av_keep))
        a = g.add(a, g.mul(g.mul(v2a_gate, v2a), av_keep))
        # 4. feed-forward
        xn = modulate(g, g.rms_norm(x, None, EPS), vm[4], vm[3])
        x = g.add(x, g.mul(feed_forward(g, ckpt, f"{p}.ff", xn), vm[5]))
        an = modulate(g, g.rms_norm(a, None, EPS), am[4], am[3])
        a = g.add(a, g.mul(feed_forward(g, ckpt, f"{p}.audio_ff", an), am[5]))

    def head(h, emb, table_name, proj, dim):
        sst = ckpt.get(table_name)  # [2, dim]
        vals = g.add(g.reshape(emb, (B, 1, 1, dim)), g.const(sst.reshape(1, 1, 2, dim), trt.bfloat16))
        shift = g.reshape(g.slice(vals, (0, 0, 0, 0), (B, 1, 1, dim)), (B, 1, dim))
        scale = g.reshape(g.slice(vals, (0, 0, 1, 0), (B, 1, 1, dim)), (B, 1, dim))
        y = modulate(g, g.layer_norm(h, EPS), scale, shift)
        return g.linear(y, ckpt.get(f"{proj}.weight"), ckpt.get(f"{proj}.bias"))

    return head(x, emb_v, "scale_shift_table", "proj_out", D), head(a, emb_a, "audio_scale_shift_table",
                                                                     "audio_proj_out", Da)


def _gather_video_rows(g: Graph, y, cp: int, batch: int):
    """fp32 ALL_GATHER of ``[B, S/cp, C]`` token shards into ``[B, S, C]`` (rank-ordered rows)."""
    yf = g.cast(y, trt.float32)
    b, s_loc, c = (int(v) for v in yf.shape)
    if b == 1:
        out = add_collective(g.net, g.reshape(yf, (s_loc, c)), trt.CollectiveOperation.ALL_GATHER, cp)
        return g.reshape(out, (1, s_loc * cp, c))
    t = g.transpose(yf, (1, 0, 2))  # [S/cp, B, C]
    out = add_collective(g.net, t, trt.CollectiveOperation.ALL_GATHER, cp)  # [S, B, C]
    return g.transpose(out, (1, 0, 2))


def build_dit_engine(transformer_dir: str | Path, shape: DiTShape, *, cp_size: int = 1,
                     stg_blocks: tuple[int, ...] = (28,), num_layers: int | None = None,
                     verbose: bool = False) -> bytes:
    ckpt = Checkpoint(transformer_dir)
    cfg = DiTConfig.from_dict(ckpt.config())
    builder, network = new_network(make_logger(verbose))
    g = Graph(network)
    B, S, Sa, L = shape.batch, shape.video_tokens, shape.audio_frames, shape.text_len
    inputs = {
        "video_latent": network.add_input("video_latent", trt.float32, (B, S, cfg.in_channels)),
        "audio_latent": network.add_input("audio_latent", trt.float32, (B, Sa, cfg.audio_in_channels)),
        "video_context": network.add_input("video_context", trt.bfloat16, (B, L, cfg.cross_attention_dim)),
        "audio_context": network.add_input("audio_context", trt.bfloat16, (B, L, cfg.audio_cross_attention_dim)),
        "timestep": network.add_input("timestep", trt.float32, (B,)),
        "stg_keep": network.add_input("stg_keep", trt.float32, (B,)),
        "av_keep": network.add_input("av_keep", trt.float32, (B,)),
    }
    if cfg.cross_attention_dim != cfg.dim or cfg.audio_cross_attention_dim != cfg.audio_dim:
        raise NotImplementedError("LTX-2.5 DiT builder expects the connector widths to match the streams")
    video, audio = add_dit(g, ckpt, cfg, shape, inputs, cp=cp_size, stg_blocks=stg_blocks, num_layers=num_layers)
    if cp_size > 1:
        video = _gather_video_rows(g, video, cp_size, B)
    g.mark_output(video, "video_velocity", trt.float32)
    g.mark_output(audio, "audio_velocity", trt.float32)
    print(f"[ltx2] Building DiT engine (batch={B}, video_tokens={S}, audio_tokens={Sa}, cp={cp_size}, "
          f"layers={num_layers or cfg.layers}) ...", file=sys.stderr)
    return build_plan(builder, network, label="DiT")


def load_dit_config(transformer_dir: str | Path) -> DiTConfig:
    return DiTConfig.from_dict(json.loads((Path(transformer_dir) / "config.json").read_text(encoding="utf-8")))

