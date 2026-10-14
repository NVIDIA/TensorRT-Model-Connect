# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 prompt encoder: Gemma 4 text tower + LTX2TextConnectors as one TensorRT plan.

Engine I/O (one prompt per call, left padded to ``seq_len``):
    Inputs:
        input_ids        [1, L] int32   Gemma token ids (pad id 0 on the left)
        attention_mask   [1, L] int32   1 = token, 0 = padding
    Outputs:
        video_context    [1, L, 4096] bf16   video connector output (``connector_prompt_embeds``)
        audio_context    [1, L, 2048] bf16   audio connector output
        packed           [1, L, 3840*49] bf16 (debug builds only) the stacked Gemma hidden states

Gemma 4 (``Gemma4UnifiedForConditionalGeneration``, text tower only) is computed exactly as
transformers 5.18 does for a text-only, left-padded batch: positions ``arange(L)``, causal +
padding mask (a finite large negative, so padding rows stay finite), sliding layers with
head_dim 256 / GQA and default RoPE (theta 1e4), full layers with head_dim 512, one KV head,
``attention_k_eq_v`` (V = the raw K projection) and proportional partial RoPE (theta 1e6),
q/k RMSNorm with scale, v RMSNorm without scale, attention scale 1.0, sandwich norms and the
per-layer ``layer_scalar``. ``hidden_states`` are the embeddings, every layer output and the
final norm applied to the last layer; the connectors stack them hidden-major / layer-minor.

The connector path is diffusers ``LTX2TextConnectors`` with ``per_modality_projections``:
per-token RMSNorm over the hidden axis of every layer, padding rows zeroed, per-modality
rescale and projection, valid tokens moved to the front with learnable registers in the tail
(an exact index computation replaces the argsort), then the 1D transformer stacks.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint
from .graph import Graph, build_plan, make_logger, new_network
from .layers import (
    AttnWeights,
    connector_rope_tables,
    feed_forward,
    ltx_attention,
    rope_constants,
)

_NEG_MASK = -1.0e9


@dataclass(frozen=True)
class Gemma4TextConfig:
    hidden: int
    intermediate: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    global_head_dim: int
    global_kv_heads: int
    k_eq_v: bool
    layer_types: tuple[str, ...]
    sliding_window: int
    local_theta: float
    global_theta: float
    global_partial: float
    global_rope_type: str
    eps: float
    activation: str

    @staticmethod
    def from_dict(cfg: dict) -> "Gemma4TextConfig":
        tc = cfg.get("text_config", cfg)
        unsupported = []
        if tc.get("enable_moe_block"):
            unsupported.append("MoE blocks")
        if int(tc.get("hidden_size_per_layer_input") or 0):
            unsupported.append("per-layer input embeddings")
        if int(tc.get("num_kv_shared_layers") or 0):
            unsupported.append("KV-shared layers")
        if tc.get("use_double_wide_mlp"):
            unsupported.append("double-wide MLP")
        if tc.get("attention_bias"):
            unsupported.append("attention bias")
        if unsupported:
            raise NotImplementedError("LTX-2.5 Gemma 4 builder does not implement: " + ", ".join(unsupported))
        rope = tc.get("rope_parameters", {})
        local = rope.get("sliding_attention", {})
        glob = rope.get("full_attention", {})
        if local.get("rope_type", "default") != "default":
            raise NotImplementedError(f"sliding RoPE type {local.get('rope_type')}")
        if glob.get("rope_type", "default") not in ("default", "proportional"):
            raise NotImplementedError(f"full-attention RoPE type {glob.get('rope_type')}")
        k_eq_v = bool(tc.get("attention_k_eq_v", False))
        kv_heads = int(tc["num_key_value_heads"])
        global_kv = tc.get("num_global_key_value_heads")
        return Gemma4TextConfig(
            hidden=int(tc["hidden_size"]),
            intermediate=int(tc["intermediate_size"]),
            layers=int(tc["num_hidden_layers"]),
            heads=int(tc["num_attention_heads"]),
            kv_heads=kv_heads,
            head_dim=int(tc["head_dim"]),
            global_head_dim=int(tc.get("global_head_dim") or tc["head_dim"]),
            global_kv_heads=int(global_kv) if (k_eq_v and global_kv) else kv_heads,
            k_eq_v=k_eq_v,
            layer_types=tuple(tc["layer_types"]),
            sliding_window=int(tc.get("sliding_window") or 0),
            local_theta=float(local.get("rope_theta", 10000.0)),
            global_theta=float(glob.get("rope_theta", 10000.0)),
            global_partial=float(glob.get("partial_rotary_factor", 1.0)),
            global_rope_type=str(glob.get("rope_type", "default")),
            eps=float(tc.get("rms_norm_eps", 1e-6)),
            activation=str(tc.get("hidden_activation", "gelu_pytorch_tanh")),
        )


@dataclass(frozen=True)
class ConnectorConfig:
    caption_channels: int
    proj_in_factor: int
    video_heads: int
    video_head_dim: int
    video_layers: int
    video_registers: int
    video_gated: bool
    audio_heads: int
    audio_head_dim: int
    audio_layers: int
    audio_registers: int
    audio_gated: bool
    video_hidden: int
    audio_hidden: int
    rope_base_seq_len: int
    rope_theta: float
    rope_double_precision: bool

    @staticmethod
    def from_dict(cfg: dict) -> "ConnectorConfig":
        if not cfg.get("per_modality_projections", False):
            raise NotImplementedError("LTX-2.5 connectors builder expects per_modality_projections")
        if cfg.get("rope_type", "split") != "split":
            raise NotImplementedError("LTX-2.5 connectors builder expects split RoPE")
        if cfg.get("causal_temporal_positioning", False):
            raise NotImplementedError("causal temporal positioning is not used by LTX-2.5")
        return ConnectorConfig(
            caption_channels=int(cfg["caption_channels"]),
            proj_in_factor=int(cfg["text_proj_in_factor"]),
            video_heads=int(cfg["video_connector_num_attention_heads"]),
            video_head_dim=int(cfg["video_connector_attention_head_dim"]),
            video_layers=int(cfg["video_connector_num_layers"]),
            video_registers=int(cfg.get("video_connector_num_learnable_registers") or 0),
            video_gated=bool(cfg.get("video_gated_attn", False)),
            audio_heads=int(cfg["audio_connector_num_attention_heads"]),
            audio_head_dim=int(cfg["audio_connector_attention_head_dim"]),
            audio_layers=int(cfg["audio_connector_num_layers"]),
            audio_registers=int(cfg.get("audio_connector_num_learnable_registers") or 0),
            audio_gated=bool(cfg.get("audio_gated_attn", False)),
            video_hidden=int(cfg["video_hidden_dim"]),
            audio_hidden=int(cfg["audio_hidden_dim"]),
            rope_base_seq_len=int(cfg.get("connector_rope_base_seq_len", 4096)),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            rope_double_precision=bool(cfg.get("rope_double_precision", True)),
        )


# ---------------------------------------------------------------------- Gemma 4


def gemma_rope_tables(cfg: Gemma4TextConfig, seq_len: int, layer_type: str):
    """cos/sin ``[L, head_dim]`` (rotate-half layout) for one layer type, rounded to bf16 like HF."""
    import ml_dtypes

    if layer_type == "full_attention":
        head_dim = cfg.global_head_dim
        base = cfg.global_theta
        if cfg.global_rope_type == "proportional":
            rope_angles = int(cfg.global_partial * head_dim // 2)
            inv = 1.0 / (base ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float32) / np.float32(head_dim)))
            inv = np.concatenate([inv.astype(np.float32),
                                  np.zeros(head_dim // 2 - rope_angles, dtype=np.float32)])
        else:
            inv = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)))
    else:
        head_dim = cfg.head_dim
        base = cfg.local_theta
        inv = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)))
    inv = inv.astype(np.float32)
    pos = np.arange(seq_len, dtype=np.float32)
    freqs = pos[:, None] * inv[None, :]
    emb = np.concatenate([freqs, freqs], axis=1).astype(np.float32)
    cos = np.cos(emb).astype(ml_dtypes.bfloat16).astype(np.float32)
    sin = np.sin(emb).astype(ml_dtypes.bfloat16).astype(np.float32)
    return cos, sin


def _gemma_rope(g: Graph, x4, cos_t, sin_t):
    """Rotate-half RoPE on ``[1, L, H, D]`` (fp32 math, bf16 tables), result in x.dtype."""
    b, l, h, d = (int(s) for s in x4.shape)
    out_dtype = x4.dtype
    xf = g.cast(x4, trt.float32)
    x1 = g.slice(xf, (0, 0, 0, 0), (b, l, h, d // 2))
    x2 = g.slice(xf, (0, 0, 0, d // 2), (b, l, h, d // 2))
    neg = g.mul(x2, g.scalar(-1.0, trt.float32, 4))
    rot = g.concat([neg, x1], axis=3)
    return g.cast(g.add(g.mul(xf, cos_t), g.mul(rot, sin_t)), out_dtype)


def _repeat_heads(g: Graph, x, kv_heads: int, heads: int):
    """``repeat_kv`` on ``[1, kvH, L, D]`` -> ``[1, H, L, D]``."""
    if kv_heads == heads:
        return x
    rep = heads // kv_heads
    idx = g.const(np.repeat(np.arange(kv_heads, dtype=np.int32), rep), trt.int32)
    return g.gather(x, idx, 1)


def _gemma_mask(g: Graph, mask_i, seq_len: int, window: int | None):
    """Additive ``[1, 1, L, L]`` bf16 mask: causal (and sliding window) x key padding."""
    i = np.arange(seq_len)[:, None]
    j = np.arange(seq_len)[None, :]
    allowed = j <= i
    if window:
        allowed &= (i - j) < window
    allowed_t = g.const(allowed.astype(np.float32).reshape(1, 1, seq_len, seq_len), trt.float32)
    keys = g.reshape(g.cast(mask_i, trt.float32), (1, 1, 1, seq_len))
    keep = g.mul(allowed_t, keys)
    add = g.mul(g.sub(g.scalar(1.0, trt.float32, 4), keep), g.scalar(_NEG_MASK, trt.float32, 4))
    return g.cast(add, trt.bfloat16)


def add_gemma4_text(g: Graph, ckpt: Checkpoint, cfg: Gemma4TextConfig, input_ids, mask_i, *,
                    seq_len: int, prefix: str = "model.language_model", num_layers: int | None = None):
    """Gemma 4 text tower; returns the ``hidden_states`` list (embeddings, layers, final norm)."""
    n_layers = cfg.layers if num_layers is None else num_layers
    hidden = cfg.hidden
    emb = ckpt.get(f"{prefix}.embed_tokens.weight")
    table = g.const(emb, trt.bfloat16)
    ids = g.reshape(input_ids, (seq_len,))
    x = g.reshape(g.gather(table, ids, 0), (1, seq_len, hidden))
    # Gemma4UnifiedTextScaledWordEmbedding: x * embed_scale, the scale rounded to the weight dtype.
    x = g.mul(x, g.scalar(math.sqrt(hidden), trt.bfloat16, 3))

    rope = {}
    masks = {}
    for lt in set(cfg.layer_types[:n_layers]):
        cos, sin = gemma_rope_tables(cfg, seq_len, lt)
        d = cos.shape[1]
        rope[lt] = (g.const(cos.reshape(1, seq_len, 1, d), trt.float32),
                    g.const(sin.reshape(1, seq_len, 1, d), trt.float32))
        window = cfg.sliding_window if lt == "sliding_attention" and cfg.sliding_window < seq_len else None
        key = window or 0
        if key not in masks:
            masks[key] = _gemma_mask(g, mask_i, seq_len, window)
        masks[lt] = masks[key]

    states = [x]
    for li in range(n_layers):
        p = f"{prefix}.layers.{li}"
        lt = cfg.layer_types[li]
        full = lt == "full_attention"
        hd = cfg.global_head_dim if full else cfg.head_dim
        kvh = cfg.global_kv_heads if full else cfg.kv_heads
        alt = cfg.k_eq_v and full

        residual = x
        h = g.rms_norm(x, ckpt.get(f"{p}.input_layernorm.weight", np.float32), cfg.eps)
        q = g.reshape(g.linear(h, ckpt.get(f"{p}.self_attn.q_proj.weight")), (1, seq_len, cfg.heads, hd))
        q = g.rms_norm(q, ckpt.get(f"{p}.self_attn.q_norm.weight", np.float32), cfg.eps)
        q = _gemma_rope(g, q, *rope[lt])
        k_raw = g.reshape(g.linear(h, ckpt.get(f"{p}.self_attn.k_proj.weight")), (1, seq_len, kvh, hd))
        v_raw = k_raw if alt else g.reshape(g.linear(h, ckpt.get(f"{p}.self_attn.v_proj.weight")),
                                            (1, seq_len, kvh, hd))
        k = g.rms_norm(k_raw, ckpt.get(f"{p}.self_attn.k_norm.weight", np.float32), cfg.eps)
        k = _gemma_rope(g, k, *rope[lt])
        v = g.rms_norm(v_raw, None, cfg.eps)
        q4 = g.transpose(q, (0, 2, 1, 3))
        k4 = _repeat_heads(g, g.transpose(k, (0, 2, 1, 3)), kvh, cfg.heads)
        v4 = _repeat_heads(g, g.transpose(v, (0, 2, 1, 3)), kvh, cfg.heads)
        ctx = g.attention(q4, k4, v4, scale=1.0, mask=masks[lt])
        ctx = g.reshape(ctx, (1, seq_len, cfg.heads * hd), first=(0, 2, 1, 3))
        attn = g.linear(ctx, ckpt.get(f"{p}.self_attn.o_proj.weight"))
        attn = g.rms_norm(attn, ckpt.get(f"{p}.post_attention_layernorm.weight", np.float32), cfg.eps)
        x = g.add(residual, attn)

        residual = x
        h = g.rms_norm(x, ckpt.get(f"{p}.pre_feedforward_layernorm.weight", np.float32), cfg.eps)
        gate = g.gelu_tanh(g.linear(h, ckpt.get(f"{p}.mlp.gate_proj.weight")))
        up = g.linear(h, ckpt.get(f"{p}.mlp.up_proj.weight"))
        mlp = g.linear(g.mul(gate, up), ckpt.get(f"{p}.mlp.down_proj.weight"))
        mlp = g.rms_norm(mlp, ckpt.get(f"{p}.post_feedforward_layernorm.weight", np.float32), cfg.eps)
        x = g.add(residual, mlp)
        if ckpt.has(f"{p}.layer_scalar"):
            scalar = ckpt.get(f"{p}.layer_scalar").reshape(1, 1, 1)
            x = g.mul(x, g.const(scalar, trt.bfloat16))
        states.append(x)
    if n_layers == cfg.layers:
        states[-1] = g.rms_norm(x, ckpt.get(f"{prefix}.norm.weight", np.float32), cfg.eps)
    return states


def stack_hidden_states(g: Graph, states):
    """``torch.stack(states, dim=-1)``: ``[1, L, H]`` x N -> ``[1, L, H, N]``."""
    b, l, h = (int(s) for s in states[0].shape)
    return g.concat([g.reshape(s, (b, l, h, 1)) for s in states], axis=3)


# ---------------------------------------------------------------------- connectors


def _front_aligned_rows(g: Graph, x, mask_i, registers: np.ndarray | None, seq_len: int):
    """``LTX2ConnectorTransformer1d`` register replacement for a left-padded prompt.

    Valid tokens (the last n rows) move to rows 0..n-1 in order; rows n.. take the learnable
    registers tiled to the sequence length. Exact replacement of the stable argsort.
    """
    if registers is None:
        return x
    _, _, d = (int(s) for s in x.shape)
    n_f = g.reduce(g.cast(mask_i, trt.float32), trt.ReduceOperation.SUM, 1, keep_dims=False)  # [1]
    n = g.cast(n_f, trt.int32)
    pos = g.const(np.arange(seq_len, dtype=np.int32), trt.int32)  # [L]
    offset = g.sub(g.const(np.array([seq_len], np.int32), trt.int32), n)  # L - n
    src = g.minimum(g.add(pos, offset), g.const(np.array([seq_len - 1], np.int32), trt.int32))
    front = g.gather(x, src, 1)  # [1, L, D]
    is_valid = g.ew(pos, n, trt.ElementWiseOperation.LESS)  # [L]
    reps = seq_len // registers.shape[0]
    tiled = np.tile(registers, (reps, 1)).reshape(1, seq_len, d)
    reg_t = g.const(tiled, x.dtype)
    cond = g.reshape(is_valid, (1, seq_len, 1))
    return g.select(cond, front, reg_t)


def _connector_stack(g: Graph, ckpt: Checkpoint, name: str, x, *, heads: int, head_dim: int,
                     layers: int, gated: bool, rope):
    for li in range(layers):
        p = f"{name}.transformer_blocks.{li}"
        h = g.rms_norm(x, None, 1e-6)
        x = g.add(x, ltx_attention(g, AttnWeights(ckpt, f"{p}.attn1"), h, h, heads=heads, eps=1e-6,
                                   q_rope=rope, k_rope=rope, gated=gated))
        h = g.rms_norm(x, None, 1e-6)
        x = g.add(x, feed_forward(g, ckpt, f"{p}.ff", h))
    return g.rms_norm(x, None, 1e-6)


def add_connectors(g: Graph, ckpt: Checkpoint, cfg: ConnectorConfig, stacked, mask_i, *, seq_len: int):
    """diffusers ``LTX2TextConnectors.forward`` (per-modality projections); returns (video, audio)."""
    b, l, h, n = (int(s) for s in stacked.shape)
    if h != cfg.caption_channels or n != cfg.proj_in_factor:
        raise ValueError(f"packed text states {h}x{n} do not match the connectors "
                         f"({cfg.caption_channels}x{cfg.proj_in_factor})")
    # per_token_rms_norm over the hidden axis of each layer, eps 1e-6, fp32 statistics.
    xf = g.cast(stacked, trt.float32)
    ms = g.reduce(g.mul(xf, xf), trt.ReduceOperation.AVG, 2)
    inv = g.unary(g.unary(g.add(ms, g.scalar(1e-6, trt.float32, 4)), trt.UnaryOperation.SQRT),
                  trt.UnaryOperation.RECIP)
    valid = g.reshape(g.cast(mask_i, trt.float32), (1, seq_len, 1, 1))
    normed = g.mul(g.mul(xf, inv), valid)  # padding rows -> 0
    flat = g.reshape(normed, (1, seq_len, h * n))

    outs = []
    for mod, hidden, heads, head_dim, layers, regs, gated in (
        ("video", cfg.video_hidden, cfg.video_heads, cfg.video_head_dim, cfg.video_layers,
         cfg.video_registers, cfg.video_gated),
        ("audio", cfg.audio_hidden, cfg.audio_heads, cfg.audio_head_dim, cfg.audio_layers,
         cfg.audio_registers, cfg.audio_gated),
    ):
        scale = math.sqrt(hidden / cfg.caption_channels)
        x = g.cast(g.mul(flat, g.scalar(scale, trt.float32, 3)), trt.bfloat16)
        x = g.linear(x, ckpt.get(f"{mod}_text_proj_in.weight"), ckpt.maybe(f"{mod}_text_proj_in.bias"))
        registers = ckpt.maybe(f"{mod}_connector.learnable_registers") if regs else None
        x = _front_aligned_rows(g, x, mask_i, registers, seq_len)
        cos, sin = connector_rope_tables(seq_len, heads * head_dim, heads, base_seq_len=cfg.rope_base_seq_len,
                                         theta=cfg.rope_theta, double_precision=cfg.rope_double_precision)
        rope = rope_constants(g, cos, sin)
        outs.append(_connector_stack(g, ckpt, f"{mod}_connector", x, heads=heads, head_dim=head_dim,
                                     layers=layers, gated=gated, rope=rope))
    return outs[0], outs[1]


# ---------------------------------------------------------------------- engines


def build_text_encoder_engine(model_dir: str | Path, *, seq_len: int = 1024, debug_packed: bool = False,
                              verbose: bool = False, gemma_layers: int | None = None) -> bytes:
    """Gemma 4 + connectors plan for one LTX-2.5 diffusers folder (``text_encoder/``, ``connectors/``)."""
    model_dir = Path(model_dir)
    te_cfg = json.loads((model_dir / "text_encoder" / "config.json").read_text(encoding="utf-8"))
    gcfg = Gemma4TextConfig.from_dict(te_cfg)
    ccfg = ConnectorConfig.from_dict(json.loads((model_dir / "connectors" / "config.json").read_text("utf-8")))
    if ccfg.caption_channels != gcfg.hidden or ccfg.proj_in_factor != gcfg.layers + 1:
        raise ValueError("connectors do not match the Gemma text encoder")
    builder, network = new_network(make_logger(verbose))
    g = Graph(network)
    ids = network.add_input("input_ids", trt.int32, (1, seq_len))
    mask = network.add_input("attention_mask", trt.int32, (1, seq_len))
    te = Checkpoint(model_dir / "text_encoder")
    states = add_gemma4_text(g, te, gcfg, ids, mask, seq_len=seq_len, num_layers=gemma_layers)
    stacked = stack_hidden_states(g, states)
    if debug_packed:
        g.mark_output(g.reshape(stacked, (1, seq_len, gcfg.hidden * len(states))), "packed")
    video, audio = add_connectors(g, Checkpoint(model_dir / "connectors"), ccfg, stacked, mask, seq_len=seq_len)
    g.mark_output(video, "video_context")
    g.mark_output(audio, "audio_context")
    print(f"[ltx2] Building text encoder engine (Gemma 4 {gemma_layers or gcfg.layers} layers + connectors, "
          f"seq_len={seq_len}) ...", file=sys.stderr)
    return build_plan(builder, network, label="text encoder")


def build_connectors_engine(connectors_dir: str | Path, *, seq_len: int, verbose: bool = False) -> bytes:
    """Connectors only, from a packed ``[1, L, H*N]`` bf16 input (parity tests)."""
    ckpt = Checkpoint(connectors_dir)
    ccfg = ConnectorConfig.from_dict(ckpt.config())
    builder, network = new_network(make_logger(verbose))
    g = Graph(network)
    h, n = ccfg.caption_channels, ccfg.proj_in_factor
    packed = network.add_input("packed", trt.bfloat16, (1, seq_len, h * n))
    mask = network.add_input("attention_mask", trt.int32, (1, seq_len))
    stacked = g.reshape(packed, (1, seq_len, h, n))
    video, audio = add_connectors(g, ckpt, ccfg, stacked, mask, seq_len=seq_len)
    g.mark_output(video, "video_context")
    g.mark_output(audio, "audio_context")
    return build_plan(builder, network, label="connectors")


def build_gemma_engine(text_encoder_dir: str | Path, *, seq_len: int, verbose: bool = False) -> bytes:
    """Gemma 4 text tower only, output ``packed`` (parity tests)."""
    ckpt = Checkpoint(text_encoder_dir)
    gcfg = Gemma4TextConfig.from_dict(ckpt.config())
    builder, network = new_network(make_logger(verbose))
    g = Graph(network)
    ids = network.add_input("input_ids", trt.int32, (1, seq_len))
    mask = network.add_input("attention_mask", trt.int32, (1, seq_len))
    states = add_gemma4_text(g, ckpt, gcfg, ids, mask, seq_len=seq_len)
    stacked = stack_hidden_states(g, states)
    g.mark_output(g.reshape(stacked, (1, seq_len, gcfg.hidden * len(states))), "packed")
    return build_plan(builder, network, label="gemma")
