# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T5 family model -- encoder-decoder seq2seq transformer."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
import numpy as np
from tensorrt_model_connect import trt_compat
from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from ...parallel_config import normalize_parallel_config, require_tensorrt_11_for_tensor_parallel

trt = trt_compat.get_trt()

name = "t5"
runtime_strategy = "t5_text_to_text"


def matches(config) -> bool:
    model_type = getattr(config, "model_type", config)
    model_type = str(model_type)
    return model_type.lower() in ("t5",)


def ensure_tokenizer_json(model_dir, *, previous_error=None):
    from .tokenizer_json import ensure_tokenizer_json

    return ensure_tokenizer_json(model_dir, previous_error=previous_error)


def load_weights(model_dir, config):
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)
    raw = config.raw
    hidden = raw.get("d_model", config.hidden_size)
    num_heads = raw.get("num_heads", config.num_attention_heads)
    d_kv = raw.get("d_kv", hidden // num_heads)
    d_ff = raw.get("d_ff", config.intermediate_size)
    num_layers = raw.get("num_layers", config.num_hidden_layers)
    enc_layers = raw.get("num_encoder_layers", num_layers)
    dec_layers = raw.get("num_decoder_layers", num_layers)
    vocab_size = raw.get("vocab_size", config.vocab_size)
    num_buckets = raw.get("relative_attention_num_buckets", 32)
    max_distance = raw.get("relative_attention_max_distance", 128)
    layer_norm_eps = raw.get("layer_norm_epsilon", 1e-6)

    weights = WeightDict()
    weights["_enc_layers"] = enc_layers
    weights["_dec_layers"] = dec_layers
    weights["_num_heads"] = num_heads
    weights["_d_kv"] = d_kv
    weights["_d_ff"] = d_ff
    weights["_hidden"] = hidden
    weights["_vocab_size"] = vocab_size
    weights["_num_buckets"] = num_buckets
    weights["_max_distance"] = max_distance
    weights["_layer_norm_eps"] = layer_norm_eps

    shared_embed = _load_tensor(readers, "shared.weight").astype(np.float32)
    weights["shared_embedding"] = shared_embed

    enc_rel_bias_key = "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    if _has_tensor(readers, enc_rel_bias_key):
        weights["enc_rel_attn_bias"] = _load_tensor(readers, enc_rel_bias_key).astype(np.float32)

    for i in range(enc_layers):
        hf = f"encoder.block.{i}"
        pfx = f"enc_layer.{i}"
        for proj in ("q", "k", "v", "o"):
            w = _load_tensor(readers, f"{hf}.layer.0.SelfAttention.{proj}.weight")
            weights[f"{pfx}.w_{proj}"] = _transpose_2d(w, f"enc_{proj}")
        weights[f"{pfx}.attn_norm"] = _load_tensor(
            readers, f"{hf}.layer.0.layer_norm.weight"
        ).astype(np.float32)
        weights[f"{pfx}.w_fc1"] = _transpose_2d(
            _load_tensor(readers, f"{hf}.layer.1.DenseReluDense.wi.weight"), "enc_fc1"
        )
        weights[f"{pfx}.w_fc2"] = _transpose_2d(
            _load_tensor(readers, f"{hf}.layer.1.DenseReluDense.wo.weight"), "enc_fc2"
        )
        weights[f"{pfx}.ffn_norm"] = _load_tensor(
            readers, f"{hf}.layer.1.layer_norm.weight"
        ).astype(np.float32)

    weights["enc_final_norm"] = _load_tensor(readers, "encoder.final_layer_norm.weight").astype(
        np.float32
    )

    dec_self_rel_bias_key = "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    if _has_tensor(readers, dec_self_rel_bias_key):
        weights["dec_self_rel_attn_bias"] = _load_tensor(readers, dec_self_rel_bias_key).astype(
            np.float32
        )

    dec_cross_rel_bias_key = (
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"
    )
    if _has_tensor(readers, dec_cross_rel_bias_key):
        weights["dec_cross_rel_attn_bias"] = _load_tensor(readers, dec_cross_rel_bias_key).astype(
            np.float32
        )

    for i in range(dec_layers):
        hf = f"decoder.block.{i}"
        pfx = f"layer.{i}"
        for proj in ("q", "k", "v", "o"):
            w = _load_tensor(readers, f"{hf}.layer.0.SelfAttention.{proj}.weight")
            weights[f"{pfx}.w_{proj}"] = _transpose_2d(w, f"dec_{proj}")
        weights[f"{pfx}.input_norm"] = _load_tensor(
            readers, f"{hf}.layer.0.layer_norm.weight"
        ).astype(np.float32)
        for proj in ("q", "k", "v", "o"):
            w = _load_tensor(readers, f"{hf}.layer.1.EncDecAttention.{proj}.weight")
            weights[f"{pfx}.cross_w_{proj}"] = _transpose_2d(w, f"xattn_{proj}")
        weights[f"{pfx}.cross_attn_norm"] = _load_tensor(
            readers, f"{hf}.layer.1.layer_norm.weight"
        ).astype(np.float32)
        weights[f"{pfx}.w_fc1"] = _transpose_2d(
            _load_tensor(readers, f"{hf}.layer.2.DenseReluDense.wi.weight"), "dec_fc1"
        )
        weights[f"{pfx}.w_fc2"] = _transpose_2d(
            _load_tensor(readers, f"{hf}.layer.2.DenseReluDense.wo.weight"), "dec_fc2"
        )
        weights[f"{pfx}.post_attn_norm"] = _load_tensor(
            readers, f"{hf}.layer.2.layer_norm.weight"
        ).astype(np.float32)

    weights["final_norm"] = _load_tensor(readers, "decoder.final_layer_norm.weight").astype(
        np.float32
    )
    if _has_tensor(readers, "lm_head.weight"):
        weights["w_out"] = _transpose_2d(_load_tensor(readers, "lm_head.weight"), "lm_head")
    else:
        weights["w_out"] = _transpose_2d(shared_embed.copy(), "embedding_tied")
    return weights


def build_engine(
    config,
    weights,
    max_cache_length,
    *,
    verbose=False,
    debug_layer_outputs=False,
    parallel_config=None,
    precision="fp32",
):
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        if precision != "fp32":
            raise NotImplementedError("T5 tensor-parallel decoder builds currently require fp32")
        require_tensorrt_11_for_tensor_parallel(
            parallel, feature="T5 tensor-parallel decoder builds"
        )
        from .decoder_tp_builder import build_t5_tp_decoder_engine

        return build_t5_tp_decoder_engine(
            config,
            weights,
            max_cache_length,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
            parallel_config=parallel,
        )

    weights["_max_cache_length"] = max_cache_length
    dl = weights["_dec_layers"]
    nh = weights["_num_heads"]
    dkv = weights["_d_kv"]
    dff = weights["_d_ff"]
    h = weights["_hidden"]
    vs = weights["_vocab_size"]
    nb = weights["_num_buckets"]
    md = weights["_max_distance"]
    eps = weights["_layer_norm_eps"]
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported T5 precision: {precision}")
    hd = dkv
    asz = nh * hd
    aw = max_cache_length + 1
    msp = max_cache_length
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tc = builder.create_builder_config()
    tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    tc.clear_flag(trt.BuilderFlag.TF32)
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attn_mask = network.add_input("attention_mask", trt.float32, (1, aw))
    ck_in, cv_in = [], []
    for i in range(dl):
        ck_in.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_k", i), trt.float32, (max_cache_length, asz)
            )
        )
        cv_in.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_v", i), trt.float32, (max_cache_length, asz)
            )
        )
    xk_in, xv_in = [], []
    for i in range(dl):
        xk_in.append(
            network.add_input(graph_ops.layer_tensor_name("cross_k", i), trt.float32, (msp, h))
        )
        xv_in.append(
            network.add_input(graph_ops.layer_tensor_name("cross_v", i), trt.float32, (msp, h))
        )
    # Encoder attention mask for cross-attention: [msp] float32, 0.0 valid, -1e9 padding
    enc_mask_input = network.add_input("encoder_mask", trt.float32, (msp,))
    enc_mask_reshape = network.add_shuffle(enc_mask_input)
    enc_mask_reshape.reshape_dims = (1, 1, msp)
    enc_mask_3d = enc_mask_reshape.get_output(0)
    if enc_mask_3d.dtype != work_trt_dtype:
        enc_mask_3d = network.add_cast(enc_mask_3d, work_trt_dtype).get_output(0)
    et = graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=work_np_dtype), dtype=work_np_dtype
    )
    emb = graph_ops.add_constant(network, (vs, h), weights["shared_embedding"], dtype=work_np_dtype)
    hs = network.add_gather(emb, token_id, 0).get_output(0)
    ck_work = [
        network.add_cast(value, work_trt_dtype).get_output(0)
        if value.dtype != work_trt_dtype
        else value
        for value in ck_in
    ]
    cv_work = [
        network.add_cast(value, work_trt_dtype).get_output(0)
        if value.dtype != work_trt_dtype
        else value
        for value in cv_in
    ]
    xk_work = [
        network.add_cast(value, work_trt_dtype).get_output(0)
        if value.dtype != work_trt_dtype
        else value
        for value in xk_in
    ]
    xv_work = [
        network.add_cast(value, work_trt_dtype).get_output(0)
        if value.dtype != work_trt_dtype
        else value
        for value in xv_in
    ]
    attn_mask_work = (
        network.add_cast(attn_mask, work_trt_dtype).get_output(0)
        if attn_mask.dtype != work_trt_dtype
        else attn_mask
    )
    dsrb = weights.get("dec_self_rel_attn_bias")
    dxrb = weights.get("dec_cross_rel_attn_bias")
    if debug_layer_outputs:
        _mark_debug_output(network, hs, "debug_embed")
    pk_out, pv_out = [], []
    for li in range(dl):
        pfx = f"layer.{li}"
        r = _add_t5_decoder_layer(
            network=network,
            hidden=hs,
            cache_k=ck_work[li],
            cache_v=cv_work[li],
            cross_k=xk_work[li],
            cross_v=xv_work[li],
            attention_mask=attn_mask_work,
            position_id=position_id,
            eps_tensor=et,
            weights=weights,
            prefix=pfx,
            hidden_size=h,
            num_heads=nh,
            head_dim=hd,
            attention_size=asz,
            ffn_dim=dff,
            max_cache_length=max_cache_length,
            max_source_positions=msp,
            dec_self_rel_bias=dsrb,
            dec_cross_rel_bias=dxrb,
            num_buckets=nb,
            max_distance=md,
            enc_mask=enc_mask_3d,
            dtype=work_np_dtype,
        )
        hs = r["hidden"]
        pk_out.append(r["present_k"])
        pv_out.append(r["present_v"])
        if debug_layer_outputs:
            _mark_debug_output(network, hs, f"debug_hidden_{li}")
    hs = graph_ops.add_rms_norm(network, hs, h, weights["final_norm"], et, dtype=work_np_dtype)
    logits = graph_ops.add_matmul_rhs_constant(
        network, hs, h, vs, weights["w_out"], dtype=work_np_dtype
    )
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    for i in range(dl):
        present_k = pk_out[i]
        present_v = pv_out[i]
        if present_k.dtype != trt.float32:
            present_k = network.add_cast(present_k, trt.float32).get_output(0)
            present_v = network.add_cast(present_v, trt.float32).get_output(0)
        present_k.name = graph_ops.layer_tensor_name("present_k", i)
        present_v.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(present_k)
        network.mark_output(present_v)
    if verbose:
        print(
            f"[trtmc build] T5 decoder ({dl}L, h={h}, heads={nh}, "
            f"d_kv={dkv}, ffn={dff}, cache={max_cache_length}, "
            f"precision={precision})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, tc)
    if plan is None:
        raise RuntimeError("TRT T5 decoder build failed")
    return bytes(plan)


def build_vision_engine(model_dir, config, weights, *, verbose=False, precision="fp32"):
    return _build_t5_encoder(config, weights, verbose=verbose, precision=precision)


def get_vl_config(config):
    raw = config.raw
    nl = raw.get("num_layers", config.num_hidden_layers)
    return {
        "encoder_layers": raw.get("num_encoder_layers", nl),
        "decoder_layers": raw.get("num_decoder_layers", nl),
        "max_source_positions": raw.get("n_positions", 512),
        "max_target_positions": raw.get("n_positions", 512),
        "has_vision_engine": True,
        "is_encoder_decoder": True,
        "d_kv": raw.get("d_kv", 64),
        "d_ff": raw.get("d_ff", config.intermediate_size),
        "relative_attention_num_buckets": raw.get("relative_attention_num_buckets", 32),
    }


def _build_t5_encoder(config, weights, *, verbose=False, precision="fp32"):
    el = weights["_enc_layers"]
    nh = weights["_num_heads"]
    dkv = weights["_d_kv"]
    dff = weights["_d_ff"]
    h = weights["_hidden"]
    vs = weights["_vocab_size"]
    nb = weights["_num_buckets"]
    md = weights["_max_distance"]
    eps = weights["_layer_norm_eps"]
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported T5 precision: {precision}")
    msl = weights.get("_max_cache_length", config.raw.get("n_positions", 512))
    hd = dkv
    asz = nh * hd
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tc = builder.create_builder_config()
    tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    tc.clear_flag(trt.BuilderFlag.TF32)
    et = graph_ops.add_constant(
        network, (1, 1), np.array([eps], dtype=work_np_dtype), dtype=work_np_dtype
    )
    ids = network.add_input("input_ids", trt.int32, (msl,))
    # Attention mask: [msl] float32, 0.0 for valid, -1e9 for padding
    enc_attn_mask_input = network.add_input("attention_mask", trt.float32, (msl,))
    # Reshape to [1, 1, msl] for broadcasting with [num_heads, msl, msl] scores
    enc_mask_3d = network.add_shuffle(enc_attn_mask_input)
    enc_mask_3d.reshape_dims = (1, 1, msl)
    enc_mask_broadcast = enc_mask_3d.get_output(0)
    if enc_mask_broadcast.dtype != work_trt_dtype:
        enc_mask_broadcast = network.add_cast(enc_mask_broadcast, work_trt_dtype).get_output(0)
    emb = graph_ops.add_constant(network, (vs, h), weights["shared_embedding"], dtype=work_np_dtype)
    hs = network.add_gather(emb, ids, 0).get_output(0)
    rpb = None
    if "enc_rel_attn_bias" in weights:
        bt = weights["enc_rel_attn_bias"]
        bi = graph_ops.make_t5_relative_position_bias(
            num_heads=nh, max_seq_len=msl, num_buckets=nb, max_distance=md
        )
        bv = bt[bi.flatten()].reshape(msl, msl, nh).transpose(2, 0, 1)
        rpb = graph_ops.add_constant(
            network, bv.shape, bv.astype(work_np_dtype), dtype=work_np_dtype
        )
    for li in range(el):
        pfx = f"enc_layer.{li}"
        normed = graph_ops.add_rms_norm(
            network, hs, h, weights[f"{pfx}.attn_norm"], et, dtype=work_np_dtype
        )
        attn = _add_t5_enc_sa(
            network,
            normed,
            weights[f"{pfx}.w_q"],
            weights[f"{pfx}.w_k"],
            weights[f"{pfx}.w_v"],
            weights[f"{pfx}.w_o"],
            h,
            nh,
            hd,
            asz,
            msl,
            rpb,
            enc_mask_broadcast,
            dtype=work_np_dtype,
        )
        hs = network.add_elementwise(hs, attn, trt.ElementWiseOperation.SUM).get_output(0)
        fn = graph_ops.add_rms_norm(
            network, hs, h, weights[f"{pfx}.ffn_norm"], et, dtype=work_np_dtype
        )
        f1 = graph_ops.add_matmul_rhs_constant(
            network, fn, h, dff, weights[f"{pfx}.w_fc1"], dtype=work_np_dtype
        )
        a = network.add_activation(f1, trt.ActivationType.RELU)
        f2 = graph_ops.add_matmul_rhs_constant(
            network, a.get_output(0), dff, h, weights[f"{pfx}.w_fc2"], dtype=work_np_dtype
        )
        hs = network.add_elementwise(hs, f2, trt.ElementWiseOperation.SUM).get_output(0)
    hs = graph_ops.add_rms_norm(network, hs, h, weights["enc_final_norm"], et, dtype=work_np_dtype)
    if hs.dtype != trt.float32:
        hs = network.add_cast(hs, trt.float32).get_output(0)
    hs.name = "encoder_output"
    network.mark_output(hs)
    if verbose:
        print(
            f"[trtmc build] T5 encoder ({el}L, h={h}, heads={nh}, "
            f"seq={msl}, precision={precision})",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, tc)
    if plan is None:
        raise RuntimeError("TRT T5 encoder build failed")
    return bytes(plan)


def _add_t5_enc_sa(
    network,
    hidden,
    wq,
    wk,
    wv,
    wo,
    h,
    nh,
    hd,
    asz,
    sl,
    rpb=None,
    attn_mask=None,
    dtype=np.float32,
):
    q = graph_ops.add_matmul_rhs_constant(network, hidden, h, asz, wq, dtype=dtype)
    k = graph_ops.add_matmul_rhs_constant(network, hidden, h, asz, wk, dtype=dtype)
    v = graph_ops.add_matmul_rhs_constant(network, hidden, h, asz, wv, dtype=dtype)
    mask = None
    if rpb is not None:
        mask = rpb
        if attn_mask is not None:
            mask = network.add_elementwise(
                mask, attn_mask, trt.ElementWiseOperation.SUM
            ).get_output(0)
        mask_4d = network.add_shuffle(mask)
        mask_4d.reshape_dims = (1, nh, sl, sl)
        mask = mask_4d.get_output(0)
    elif attn_mask is not None:
        mask_4d = network.add_shuffle(attn_mask)
        mask_4d.reshape_dims = (1, 1, 1, sl)
        mask = mask_4d.get_output(0)
    cf = graph_ops.add_attention_from_rows(
        network, q, k, v, num_heads=nh, head_dim=hd, q_seq=sl, kv_seq=sl, mask=mask, scale=1.0
    )
    return graph_ops.add_matmul_rhs_constant(network, cf, asz, h, wo, dtype=dtype)


def _add_t5_decoder_layer(
    *,
    network,
    hidden,
    cache_k,
    cache_v,
    cross_k,
    cross_v,
    attention_mask,
    position_id,
    eps_tensor,
    weights,
    prefix,
    hidden_size,
    num_heads,
    head_dim,
    attention_size,
    ffn_dim,
    max_cache_length,
    max_source_positions,
    dec_self_rel_bias,
    dec_cross_rel_bias,
    num_buckets,
    max_distance,
    enc_mask=None,
    dtype=np.float32,
):
    h, nh, hd, asz = hidden_size, num_heads, head_dim, attention_size
    aw = max_cache_length + 1
    msp = max_source_positions
    normed = graph_ops.add_rms_norm(
        network, hidden, h, weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype
    )
    q = graph_ops.add_matmul_rhs_constant(
        network, normed, h, asz, weights[f"{prefix}.w_q"], dtype=dtype
    )
    k = graph_ops.add_matmul_rhs_constant(
        network, normed, h, asz, weights[f"{prefix}.w_k"], dtype=dtype
    )
    v = graph_ops.add_matmul_rhs_constant(
        network, normed, h, asz, weights[f"{prefix}.w_v"], dtype=dtype
    )
    present_k, present_v = k, v
    kr = network.add_shuffle(k)
    kr.reshape_dims = (1, asz)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (1, asz)
    ak = network.add_concatenation([cache_k, kr.get_output(0)])
    ak.axis = 0
    av = network.add_concatenation([cache_v, vr.get_output(0)])
    av.axis = 0
    if dec_self_rel_bias is not None:
        bi = _make_t5_causal_buckets(aw, num_buckets, max_distance)
        bv = dec_self_rel_bias[bi.flatten()].reshape(aw, aw, nh).transpose(2, 0, 1)
        bc = graph_ops.add_constant(network, bv.shape, bv.astype(dtype), dtype=dtype)
        br = network.add_gather(bc, position_id, 1)
        m3 = network.add_shuffle(attention_mask)
        m3.reshape_dims = (1, 1, aw)
        mask_3d = network.add_elementwise(
            br.get_output(0), m3.get_output(0), trt.ElementWiseOperation.SUM
        )
        mask_4d = network.add_shuffle(mask_3d.get_output(0))
        mask_4d.reshape_dims = (1, nh, 1, aw)
        self_mask = mask_4d.get_output(0)
    else:
        self_mask = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    cf = graph_ops.add_attention_from_rows(
        network,
        q,
        ak.get_output(0),
        av.get_output(0),
        num_heads=nh,
        head_dim=hd,
        q_seq=1,
        kv_seq=aw,
        mask=self_mask,
        scale=1.0,
    )
    sa = graph_ops.add_matmul_rhs_constant(
        network, cf, asz, h, weights[f"{prefix}.w_o"], dtype=dtype
    )
    psa = network.add_elementwise(hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)
    cn = graph_ops.add_rms_norm(
        network, psa, h, weights[f"{prefix}.cross_attn_norm"], eps_tensor, dtype=dtype
    )
    cq = graph_ops.add_matmul_rhs_constant(
        network, cn, h, asz, weights[f"{prefix}.cross_w_q"], dtype=dtype
    )
    ckp = graph_ops.add_matmul_rhs_constant(
        network, cross_k, h, asz, weights[f"{prefix}.cross_w_k"], dtype=dtype
    )
    cvp = graph_ops.add_matmul_rhs_constant(
        network, cross_v, h, asz, weights[f"{prefix}.cross_w_v"], dtype=dtype
    )
    if dec_cross_rel_bias is not None:
        xbi = _make_t5_cross_buckets(aw, msp, num_buckets, max_distance)
        xbv = dec_cross_rel_bias[xbi.flatten()].reshape(aw, msp, nh).transpose(2, 0, 1)
        xbc = graph_ops.add_constant(network, xbv.shape, xbv.astype(dtype), dtype=dtype)
        xbr = network.add_gather(xbc, position_id, 1)
        cross_mask = xbr.get_output(0)
        if enc_mask is not None:
            cross_mask = network.add_elementwise(
                cross_mask, enc_mask, trt.ElementWiseOperation.SUM
            ).get_output(0)
        cross_mask_4d = network.add_shuffle(cross_mask)
        cross_mask_4d.reshape_dims = (1, nh, 1, msp)
        cross_mask = cross_mask_4d.get_output(0)
    elif enc_mask is not None:
        cross_mask_4d = network.add_shuffle(enc_mask)
        cross_mask_4d.reshape_dims = (1, 1, 1, msp)
        cross_mask = cross_mask_4d.get_output(0)
    else:
        cross_mask = None
    ccf = graph_ops.add_attention_from_rows(
        network,
        cq,
        ckp,
        cvp,
        num_heads=nh,
        head_dim=hd,
        q_seq=1,
        kv_seq=msp,
        mask=cross_mask,
        scale=1.0,
    )
    ca = graph_ops.add_matmul_rhs_constant(
        network, ccf, asz, h, weights[f"{prefix}.cross_w_o"], dtype=dtype
    )
    pca = network.add_elementwise(psa, ca, trt.ElementWiseOperation.SUM).get_output(0)
    fn = graph_ops.add_rms_norm(
        network, pca, h, weights[f"{prefix}.post_attn_norm"], eps_tensor, dtype=dtype
    )
    f1 = graph_ops.add_matmul_rhs_constant(
        network, fn, h, ffn_dim, weights[f"{prefix}.w_fc1"], dtype=dtype
    )
    act = network.add_activation(f1, trt.ActivationType.RELU)
    f2 = graph_ops.add_matmul_rhs_constant(
        network, act.get_output(0), ffn_dim, h, weights[f"{prefix}.w_fc2"], dtype=dtype
    )
    out = network.add_elementwise(pca, f2, trt.ElementWiseOperation.SUM).get_output(0)
    return {"hidden": out, "present_k": present_k, "present_v": present_v}


def _make_t5_causal_buckets(ml, nb=32, md=128):
    cp = np.arange(ml, dtype=np.int32)[:, None]
    mp = np.arange(ml, dtype=np.int32)[None, :]
    n = np.maximum(-(mp - cp), 0)
    me = nb // 2
    is_s = n < me
    nc = np.maximum(n.astype(np.float32), 1)
    vl = me + (np.log(nc / me) / np.log(md / me) * (nb - me)).astype(np.int32)
    return np.where(is_s, n, np.minimum(vl, nb - 1)).astype(np.int32)


def _make_t5_cross_buckets(ql, kl, nb=32, md=128):
    cp = np.arange(ql, dtype=np.int32)[:, None]
    mp = np.arange(kl, dtype=np.int32)[None, :]
    n = np.maximum(-(mp - cp), 0)
    me = nb // 2
    is_s = n < me
    nc = np.maximum(n.astype(np.float32), 1)
    vl = me + (np.log(nc / me) / np.log(md / me) * (nb - me)).astype(np.int32)
    return np.where(is_s, n, np.minimum(vl, nb - 1)).astype(np.int32)


def _mark_debug_output(network, tensor, name):
    out = tensor
    if out.dtype != trt.float32:
        out = network.add_cast(out, trt.float32).get_output(0)
    out.name = name
    network.mark_output(out)


requires_tokenizer = True
embed_input = False


def _detect_tokenizer_frame(
    source: str, *, revision: str | None = None
) -> tuple[list[int], list[int]] | None:
    try:
        from transformers import AutoTokenizer

        kwargs = {"trust_remote_code": True}
        if revision:
            kwargs["revision"] = revision
        if not Path(source).is_dir():
            kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        default_ids = list(tokenizer.encode("hello"))
        plain_ids = list(tokenizer.encode("hello", add_special_tokens=False))
    except Exception:
        return None
    if default_ids == plain_ids:
        return [], []
    if not plain_ids:
        return default_ids, []
    for start in range(len(default_ids) - len(plain_ids) + 1):
        if default_ids[start : start + len(plain_ids)] == plain_ids:
            return default_ids[:start], default_ids[start + len(plain_ids) :]
    return None


def _apply_generation_config_eos(model_dir: Path, config: dict) -> None:
    path = model_dir / "generation_config.json"
    if not path.is_file():
        return
    generation_config = json.loads(path.read_text(encoding="utf-8"))
    if "eos_token_id" in generation_config:
        config["eos_token_id"] = generation_config["eos_token_id"]


def _build_local_engine(config, weights, max_cache_length, precision, verbose, parallel, options):
    from tensorrt_model_connect.tvm_ffi.graph_build import engine_role, inspection_role

    role = (
        "dual_profile"
        if str(options.get("decoder_engine_layout") or "split") == "dual_profile"
        else "decode"
    )

    def build_role(selected_role: str) -> bytes:
        with engine_role(selected_role):
            return build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                verbose=verbose,
                parallel_config=parallel,
            )

    target_role = inspection_role()
    if target_role is not None:
        build_role(target_role)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")
    return build_role(role), ("dual_profile" if role == "dual_profile" else "single")


def build(model_dir: str, output_path: str, **options) -> None:
    """Build the complete t5 bundle inside its owning family module."""
    from datetime import datetime, timezone

    from tensorrt_model_connect import trt_compat as build_trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import BundleInfo, BundleSection, write_bundle
    from tensorrt_model_connect.parallel_config import (
        normalize_parallel_config,
        rank_engine_section,
        require_tensorrt_11_for_tensor_parallel,
    )

    model_path = Path(model_dir)
    decoder_engine_layout = str(options.get("decoder_engine_layout") or "split")
    if decoder_engine_layout not in {"split", "dual_profile"}:
        raise ValueError(
            "decoder_engine_layout must be 'split' or 'dual_profile', "
            f"got {decoder_engine_layout!r}"
        )
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.cp_enabled:
        raise NotImplementedError("t5 does not support context-parallel builds")
    if options.get("dynamic_kv_cache") or options.get("triattention_stats_path"):
        raise ValueError("t5 does not use a decoder KV-cache runtime")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = False
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))
    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = int(256 if requested_cache_length is None else requested_cache_length)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_plan = None
    if quantize:
        raise ValueError("t5 does not support quantized builds")

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="t5 tensor-parallel builds")

    verbose = bool(options.get("verbose"))
    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                verbose=verbose,
                parallel_config=parallel.for_rank(rank),
            )
            for rank in range(parallel.tp_size)
        }
        sections = [
            BundleSection(rank_engine_section(rank), plan) for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        plan, decoder_layout = _build_local_engine(
            config, weights, max_cache_length, precision, verbose, parallel, options
        )
        sections = [BundleSection("engine_plan", plan)]
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    vision_started = time.monotonic()
    vision_plan = build_vision_engine(
        str(model_path), config, weights, precision=precision, verbose=verbose
    )
    vision_elapsed = time.monotonic() - vision_started
    add_build_timing(timing, "trt_compile_s", vision_elapsed)
    add_build_timing(timing, "trt_compile_vision_engine_s", vision_elapsed)
    write_build_timing(timing)
    if vision_plan is not None:
        sections.append(BundleSection("vision_engine_plan", vision_plan))

    tokenizer_source = str(options.get("tokenizer_source_model_id_or_path") or model_path)
    tokenizer_frame = _detect_tokenizer_frame(
        tokenizer_source,
        revision=(
            str(options["tokenizer_source_revision"])
            if options.get("tokenizer_source_revision")
            else None
        ),
    )
    from tensorrt_model_connect.tokenizer_conversion import (
        ensure_tokenizer_json as ensure_native_tokenizer_json,
    )

    ensure_native_tokenizer_json(model_path, family_ensure=ensure_tokenizer_json)
    if tokenizer_frame is None:
        tokenizer_frame = _detect_tokenizer_frame(str(model_path))
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = build_trt_compat.tensorrt_version() or "unknown"
    version_match = re.search(r"(\d+)\.(\d+)", trt_version)
    trt_abi = f"{version_match.group(1)}.{version_match.group(2)}" if version_match else ""
    try:
        from tensorrt_model_connect.runtime_provider.target import _probe_current_target_with_device

        gpu_name = str(_probe_current_target_with_device()[0]["gpu_name"])
    except Exception:
        gpu_name = ""
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=add_special_tokens,
    )

    source_config = model_path / "config.json"
    runtime_config = (
        json.loads(source_config.read_text(encoding="utf-8"))
        if source_config.is_file()
        else dict(config.raw)
    )
    _apply_generation_config_eos(model_path, runtime_config)
    runtime_config.update(
        {
            "runtime_strategy": runtime_strategy,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": trt_version,
            "precision": precision,
            "tokenizer_add_special_tokens": int(add_special_tokens),
            "decoder_engine_layout": decoder_layout,
        }
    )
    if trt_abi:
        runtime_config["trt_abi"] = trt_abi
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    runtime_config.update(parallel.to_bundle_config_fields())
    if vision_plan is not None:
        runtime_config["has_vision_engine"] = True
    vl_config = get_vl_config(config)
    if vl_config is not None:
        runtime_config.update(vl_config)

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))

    embedded_config = False
    for filename in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.model",
        "preprocessor_config.json",
        "processor_config.json",
    ):
        path = model_path / filename
        if filename == "config.json":
            sections.append(
                BundleSection(filename, json.dumps(runtime_config, indent=2).encode("utf-8"))
            )
            embedded_config = True
        elif path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    if not embedded_config:
        sections.append(
            BundleSection("config.json", json.dumps(runtime_config, indent=2).encode("utf-8"))
        )

    kernel_manifest = []
    for global_name, library in options.get("kernel_artifacts") or ():
        section_name = f"kernel_{global_name.replace('.', '_')}.so"
        sections.append(BundleSection(section_name, Path(library).read_bytes()))
        kernel_manifest.append(
            {"global_name": global_name, "func_name": "run", "section": section_name}
        )
    if kernel_manifest:
        sections.append(
            BundleSection(
                "kernel_manifest.json",
                json.dumps({"kernels": kernel_manifest}).encode("utf-8"),
            )
        )

    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)
