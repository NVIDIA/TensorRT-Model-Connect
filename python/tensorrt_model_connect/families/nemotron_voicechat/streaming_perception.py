# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VoiceChat-owned cache-aware FastConformer TensorRT graph."""

from __future__ import annotations

import math
import sys

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops
from .checkpoint_mapper import WeightDict
from .conformer import (
    _add_conv_norm,
    _add_half_ffn,
    _build_subsampling,
    _compute_causal_enc_seq_len,
    _relative_pe,
)


trt = trt_compat.get_trt()


_STREAMING_TIME_CACHE = 8
_STREAMING_PRE_ENCODE_CACHE = 9
_STREAMING_DROP_PRE_ENCODED = 2


def _streaming_mel_length(
    right_context: int, subsampling_factor: int = 8, *, first_step: bool = False
) -> int:
    if first_step:
        return 1 + subsampling_factor * right_context
    return _STREAMING_PRE_ENCODE_CACHE + subsampling_factor * (right_context + 1)


def _streaming_encoder_frames(right_context: int) -> int:
    return right_context + 1


def _rel_shift_streaming(
    network, x, heads: int, query_len: int, pos_len: int, key_len: int, dtype=np.float32
):
    zeros = graph_ops.add_constant(
        network, (heads, query_len, 1), np.zeros((heads, query_len, 1), dtype=dtype), dtype=dtype
    )
    padded = network.add_concatenation([zeros, x])
    padded.axis = 2
    rs1 = network.add_shuffle(padded.get_output(0))
    rs1.reshape_dims = (heads, pos_len + 1, query_len)
    sl1 = network.add_slice(
        rs1.get_output(0), start=(0, 1, 0), shape=(heads, pos_len, query_len), stride=(1, 1, 1)
    )
    rs2 = network.add_shuffle(sl1.get_output(0))
    rs2.reshape_dims = (heads, query_len, pos_len)
    sl2 = network.add_slice(
        rs2.get_output(0), start=(0, 0, 0), shape=(heads, query_len, key_len), stride=(1, 1, 1)
    )
    return sl2.get_output(0)


def _slice_layer_channel_cache(network, cache, layer: int, cache_len: int, hidden: int):
    sl = network.add_slice(
        cache, start=(layer, 0, 0), shape=(1, cache_len, hidden), stride=(1, 1, 1)
    )
    sh = network.add_shuffle(sl.get_output(0))
    sh.reshape_dims = (cache_len, hidden)
    return sh.get_output(0)


def _slice_layer_time_cache(network, cache, layer: int, hidden: int, time_cache: int):
    sl = network.add_slice(
        cache, start=(layer, 0, 0), shape=(1, hidden, time_cache), stride=(1, 1, 1)
    )
    sh = network.add_shuffle(sl.get_output(0))
    sh.reshape_dims = (hidden, time_cache)
    return sh.get_output(0)


def _add_streaming_rel_pos_attention(
    network,
    hs,
    channel_cache,
    weights,
    pfx,
    hidden,
    heads,
    head_dim,
    query_len,
    cache_len,
    rpe,
    eps,
    enc_mask,
    dtype=np.float32,
):
    key_len = cache_len + query_len
    pos_len = 2 * key_len - 1
    normed = graph_ops.add_layer_norm(
        network,
        hs,
        hidden,
        weights[f"{pfx}.norm_sa"],
        weights[f"{pfx}.norm_sa_b"],
        eps,
        dtype=dtype,
    )

    cat_kv = network.add_concatenation([channel_cache, normed])
    cat_kv.axis = 0
    kv = cat_kv.get_output(0)

    q = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, hidden, weights[f"{pfx}.w_q"], dtype=dtype
        ),
        hidden,
        weights[f"{pfx}.b_q"],
        dtype=dtype,
    )
    k = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, kv, hidden, hidden, weights[f"{pfx}.w_k"], dtype=dtype
        ),
        hidden,
        weights[f"{pfx}.b_k"],
        dtype=dtype,
    )
    v = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, kv, hidden, hidden, weights[f"{pfx}.w_v"], dtype=dtype
        ),
        hidden,
        weights[f"{pfx}.b_v"],
        dtype=dtype,
    )

    qr = network.add_shuffle(q)
    qr.reshape_dims = (query_len, heads, head_dim)
    kr = network.add_shuffle(k)
    kr.reshape_dims = (key_len, heads, head_dim)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (key_len, heads, head_dim)

    bu = graph_ops.add_constant(
        network, (1, heads, head_dim), weights[f"{pfx}.pos_bias_u"], dtype=dtype
    )
    bv = graph_ops.add_constant(
        network, (1, heads, head_dim), weights[f"{pfx}.pos_bias_v"], dtype=dtype
    )
    qu = network.add_elementwise(qr.get_output(0), bu, trt.ElementWiseOperation.SUM).get_output(0)
    qv = network.add_elementwise(qr.get_output(0), bv, trt.ElementWiseOperation.SUM).get_output(0)

    qu_t = network.add_shuffle(qu)
    qu_t.first_transpose = trt.Permutation([1, 0, 2])
    qv_t = network.add_shuffle(qv)
    qv_t.first_transpose = trt.Permutation([1, 0, 2])
    k_t = network.add_shuffle(kr.get_output(0))
    k_t.first_transpose = trt.Permutation([1, 0, 2])
    v_t = network.add_shuffle(vr.get_output(0))
    v_t.first_transpose = trt.Permutation([1, 0, 2])

    cs = network.add_matrix_multiply(
        qu_t.get_output(0),
        trt.MatrixOperation.NONE,
        k_t.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    rp_t = network.add_shuffle(rpe)
    rp_t.first_transpose = trt.Permutation([1, 0, 2])
    ps_raw = network.add_matrix_multiply(
        qv_t.get_output(0),
        trt.MatrixOperation.NONE,
        rp_t.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    ps = _rel_shift_streaming(network, ps_raw, heads, query_len, pos_len, key_len, dtype=dtype)
    total = network.add_elementwise(cs, ps, trt.ElementWiseOperation.SUM).get_output(0)
    scale = graph_ops.add_constant(
        network, (1, 1, 1), np.array([1.0 / math.sqrt(head_dim)], dtype=dtype), dtype=dtype
    )
    scaled = network.add_elementwise(total, scale, trt.ElementWiseOperation.PROD).get_output(0)
    if enc_mask is not None:
        scaled = network.add_elementwise(scaled, enc_mask, trt.ElementWiseOperation.SUM).get_output(
            0
        )

    sm = network.add_softmax(scaled)
    sm.axes = 1 << 2
    ao = network.add_matrix_multiply(
        sm.get_output(0), trt.MatrixOperation.NONE, v_t.get_output(0), trt.MatrixOperation.NONE
    ).get_output(0)
    at = network.add_shuffle(ao)
    at.first_transpose = trt.Permutation([1, 0, 2])
    af = network.add_shuffle(at.get_output(0))
    af.reshape_dims = (query_len, hidden)
    out = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, af.get_output(0), hidden, hidden, weights[f"{pfx}.w_o"], dtype=dtype
        ),
        hidden,
        weights[f"{pfx}.b_o"],
        dtype=dtype,
    )

    cache_tail = network.add_slice(
        channel_cache, start=(query_len, 0), shape=(cache_len - query_len, hidden), stride=(1, 1)
    ).get_output(0)
    next_cache = network.add_concatenation([cache_tail, normed])
    next_cache.axis = 0
    return out, next_cache.get_output(0)


def _add_streaming_conv_module(
    network,
    hs,
    time_cache,
    weights,
    pfx,
    hidden,
    kern,
    query_len,
    eps,
    dtype=np.float32,
):
    normed = graph_ops.add_layer_norm(
        network,
        hs,
        hidden,
        weights[f"{pfx}.norm_conv"],
        weights[f"{pfx}.norm_conv_b"],
        eps,
        dtype=dtype,
    )
    r1 = network.add_shuffle(normed)
    r1.first_transpose = trt.Permutation([1, 0])
    r2 = network.add_shuffle(r1.get_output(0))
    r2.reshape_dims = (1, hidden, query_len)
    x = graph_ops.add_conv1d(
        network,
        r2.get_output(0),
        weight=weights[f"{pfx}.cpw1_w"],
        bias=weights[f"{pfx}.cpw1_b"],
        out_channels=2 * hidden,
        kernel_size=1,
        dtype=dtype,
    )
    xa = network.add_slice(
        x, start=(0, 0, 0), shape=(1, hidden, query_len), stride=(1, 1, 1)
    ).get_output(0)
    xb = network.add_slice(
        x, start=(0, hidden, 0), shape=(1, hidden, query_len), stride=(1, 1, 1)
    ).get_output(0)
    gate = network.add_activation(xb, trt.ActivationType.SIGMOID).get_output(0)
    x = network.add_elementwise(xa, gate, trt.ElementWiseOperation.PROD).get_output(0)

    tc = network.add_shuffle(time_cache)
    tc.reshape_dims = (1, hidden, _STREAMING_TIME_CACHE)
    cached = network.add_concatenation([tc.get_output(0), x])
    cached.axis = 2
    cached_tensor = cached.get_output(0)
    next_time = network.add_slice(
        cached_tensor,
        start=(0, 0, query_len),
        shape=(1, hidden, _STREAMING_TIME_CACHE),
        stride=(1, 1, 1),
    )
    x = graph_ops.add_conv1d(
        network,
        cached_tensor,
        weight=weights[f"{pfx}.cdw_w"],
        bias=weights[f"{pfx}.cdw_b"],
        out_channels=hidden,
        kernel_size=kern,
        groups=hidden,
        dtype=dtype,
    )
    x = _add_conv_norm(network, x, weights, pfx, hidden, query_len, eps, dtype=dtype)
    x = graph_ops.add_activation(network, x, "silu")
    x = graph_ops.add_conv1d(
        network,
        x,
        weight=weights[f"{pfx}.cpw2_w"],
        bias=weights[f"{pfx}.cpw2_b"],
        out_channels=hidden,
        kernel_size=1,
        dtype=dtype,
    )
    r3 = network.add_shuffle(x)
    r3.reshape_dims = (hidden, query_len)
    r4 = network.add_shuffle(r3.get_output(0))
    r4.first_transpose = trt.Permutation([1, 0])

    nt = network.add_shuffle(next_time.get_output(0))
    nt.reshape_dims = (hidden, _STREAMING_TIME_CACHE)
    return r4.get_output(0), nt.get_output(0)


def _add_streaming_conformer_block(
    network,
    hs,
    channel_cache,
    time_cache,
    weights,
    pfx,
    hidden,
    heads,
    head_dim,
    ffn,
    kern,
    query_len,
    cache_len,
    rpe,
    eps,
    enc_mask,
    dtype=np.float32,
):
    ffn1 = _add_half_ffn(network, hs, weights, f"{pfx}.ff1", hidden, ffn, eps, dtype=dtype)
    hs = network.add_elementwise(hs, ffn1, trt.ElementWiseOperation.SUM).get_output(0)
    attn, next_channel = _add_streaming_rel_pos_attention(
        network,
        hs,
        channel_cache,
        weights,
        pfx,
        hidden,
        heads,
        head_dim,
        query_len,
        cache_len,
        rpe,
        eps,
        enc_mask,
        dtype=dtype,
    )
    hs = network.add_elementwise(hs, attn, trt.ElementWiseOperation.SUM).get_output(0)
    conv, next_time = _add_streaming_conv_module(
        network,
        hs,
        time_cache,
        weights,
        pfx,
        hidden,
        kern,
        query_len,
        eps,
        dtype=dtype,
    )
    hs = network.add_elementwise(hs, conv, trt.ElementWiseOperation.SUM).get_output(0)
    ffn2 = _add_half_ffn(network, hs, weights, f"{pfx}.ff2", hidden, ffn, eps, dtype=dtype)
    hs = network.add_elementwise(hs, ffn2, trt.ElementWiseOperation.SUM).get_output(0)
    out = graph_ops.add_layer_norm(
        network,
        hs,
        hidden,
        weights[f"{pfx}.norm_out"],
        weights[f"{pfx}.norm_out_b"],
        eps,
        dtype=dtype,
    )
    return out, next_channel, next_time


def _mark_layer_cache_output(network, tensors, name: str, shape):
    reshaped = []
    for tensor in tensors:
        sh = network.add_shuffle(tensor)
        sh.reshape_dims = (1,) + tuple(shape)
        reshaped.append(sh.get_output(0))
    cat = network.add_concatenation(reshaped)
    cat.axis = 0
    out = cat.get_output(0)
    out.name = name
    network.mark_output(out)


def _build_streaming_encoder(
    weights: WeightDict,
    right_context: int,
    *,
    first_step: bool = False,
    verbose: bool = False,
) -> bytes:
    hidden = int(weights["_hidden"])
    enc_layers = int(weights["_enc_layers"])
    heads = int(weights["_enc_heads"])
    ffn = int(weights["_enc_ffn"])
    mel_bins = int(weights["_mel_bins"])
    kern = int(weights["_kern"])
    sub_ch = int(weights["_sub_ch"])
    head_dim = int(weights["_head_dim"])
    mel_len = _streaming_mel_length(right_context, first_step=first_step)
    pre_encoded = _compute_causal_enc_seq_len(mel_len)
    query_len = _streaming_encoder_frames(right_context)
    drop = pre_encoded - query_len
    expected_drop = 0 if first_step else _STREAMING_DROP_PRE_ENCODED
    if drop != expected_drop:
        raise ValueError(
            f"Unexpected streaming pre-encode drop for right={right_context}, "
            f"first_step={first_step}: {drop}"
        )

    cache_len = int(weights["_streaming_cache_left"])
    key_len = cache_len + query_len
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    if hasattr(config, "clear_flag") and hasattr(trt, "BuilderFlag"):
        config.clear_flag(trt.BuilderFlag.TF32)
    eps = graph_ops.add_constant(network, (1, 1), np.array([1e-5], dtype=np.float32))
    mel = network.add_input("mel_features", trt.float32, (mel_bins, mel_len))
    channel_cache = network.add_input(
        "cache_last_channel", trt.float32, (enc_layers, cache_len, hidden)
    )
    time_cache = network.add_input(
        "cache_last_time", trt.float32, (enc_layers, hidden, _STREAMING_TIME_CACHE)
    )
    enc_mask = network.add_input("encoder_mask", trt.float32, (1, query_len, key_len))
    hs = _build_subsampling(network, mel, weights, sub_ch, hidden, mel_bins, mel_len, pre_encoded)
    hs = network.add_slice(
        hs, start=(drop, 0), shape=(query_len, hidden), stride=(1, 1)
    ).get_output(0)

    rpe_np = _relative_pe(key_len, hidden)
    next_channels = []
    next_times = []
    for i in range(enc_layers):
        pfx = f"el.{i}"
        rel_proj = rpe_np @ weights[f"{pfx}.w_pos"]
        rel_proj = rel_proj.reshape(2 * key_len - 1, heads, head_dim)
        rpe = graph_ops.add_constant(network, (2 * key_len - 1, heads, head_dim), rel_proj)
        layer_channel_cache = _slice_layer_channel_cache(
            network, channel_cache, i, cache_len, hidden
        )
        layer_time_cache = _slice_layer_time_cache(
            network, time_cache, i, hidden, _STREAMING_TIME_CACHE
        )
        hs, next_channel, next_time = _add_streaming_conformer_block(
            network,
            hs,
            layer_channel_cache,
            layer_time_cache,
            weights,
            pfx,
            hidden,
            heads,
            head_dim,
            ffn,
            kern,
            query_len,
            cache_len,
            rpe,
            eps,
            enc_mask,
        )
        next_channels.append(next_channel)
        next_times.append(next_time)

    rnnt_output = hs
    rnnt_output.name = "rnnt_encoder_output"
    network.mark_output(rnnt_output)

    output_dim = int(weights.get("_output_dim", hidden))
    if output_dim != hidden:
        output = graph_ops.add_matmul_rhs_constant(
            network,
            hs,
            hidden,
            output_dim,
            weights["perception_proj"],
        )
        output = graph_ops.add_bias_sum(
            network,
            output,
            output_dim,
            weights["perception_proj_bias"],
        )
    else:
        output = hs
    output.name = "audio_embeddings"
    network.mark_output(output)
    _mark_layer_cache_output(network, next_channels, "cache_last_channel_next", (cache_len, hidden))
    _mark_layer_cache_output(
        network, next_times, "cache_last_time_next", (hidden, _STREAMING_TIME_CACHE)
    )

    if verbose:
        step = "first" if first_step else "next"
        print(
            f"[trtmc build] Building VoiceChat streaming perception {step} "
            f"(right={right_context}, mel={mel_len}, frames={query_len})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(
            f"VoiceChat streaming perception build failed for right={right_context}, "
            f"first_step={first_step}"
        )
    return bytes(plan)
