"""Nemotron Speech Streaming family plugin -- FastConformer cache-aware RNNT ASR.

The Hugging Face repo ships a NeMo ``.nemo`` archive. Build-time Python extracts
the checkpoint and builds three TensorRT plans:

  * ``vision_engine_plan``: FastConformer acoustic encoder
  * ``engine_plan``: RNNT prediction network
  * ``joint_engine_plan``: RNNT joint network

The C++ runtime strategy is ``speech_to_text_rnnt`` and performs greedy RNNT
decoding without a Python subprocess.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from ... import graph_ops
from ...checkpoint_mapper import WeightDict, _transpose_2d
from ...config import ModelConfig
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from ..canary import (
    _add_conv_norm,
    _add_half_ffn,
    _build_encoder,
    _build_subsampling,
    _compute_enc_seq_len,
    _compute_causal_enc_seq_len,
    _extract_tokenizer_from_nemo,
    _load_nemo_archive,
    _relative_pe,
    _to_np,
)
from .predictor_tp_builder import build_nemotron_streaming_tp_predictor


trt = trt_compat.get_trt()


def _cfg_int(*values, default: int) -> int:
    for value in values:
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return default


def _cfg_dict(root: dict, *path: str) -> dict:
    cur = root
    for item in path:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(item, {})
    return cur if isinstance(cur, dict) else {}


def _find_tensor(sd: dict, candidates: list[str], label: str):
    for key in candidates:
        if key in sd:
            return sd[key]
    suffix_matches = []
    for key in sd:
        for suffix in candidates:
            if key.endswith(suffix):
                suffix_matches.append(key)
                break
    if len(suffix_matches) == 1:
        return sd[suffix_matches[0]]
    if len(suffix_matches) > 1:
        raise KeyError(f"Ambiguous tensor for {label}: {suffix_matches}")
    raise KeyError(f"Missing tensor for {label}; tried {candidates}")


def _find_joint_linear(sd: dict, prefix: str, label: str):
    candidates = [
        f"{prefix}.joint_net.1.weight",
        f"{prefix}.joint_net.2.weight",
        f"{prefix}.joint_net.0.weight",
    ]
    for key in candidates:
        if key in sd:
            bias_key = key[:-6] + "bias"
            if bias_key not in sd:
                raise KeyError(f"Missing tensor for {label} bias: {bias_key}")
            return sd[key], sd[bias_key]

    suffixes = [".joint_net.1.weight", ".joint_net.2.weight", ".joint_net.0.weight"]
    matches = [key for key in sd if key.startswith(prefix) and any(key.endswith(s) for s in suffixes)]
    if len(matches) == 1:
        key = matches[0]
        bias_key = key[:-6] + "bias"
        if bias_key not in sd:
            raise KeyError(f"Missing tensor for {label} bias: {bias_key}")
        return sd[key], sd[bias_key]
    raise KeyError(f"Missing tensor for {label}; tried {candidates}")


def _add_lstm_cell(network, x, h_prev, c_prev, weights, pfx: str, hidden: int):
    w_ih = graph_ops.add_constant(network, (hidden, 4 * hidden), weights[f"{pfx}.w_ih_t"])
    w_hh = graph_ops.add_constant(network, (hidden, 4 * hidden), weights[f"{pfx}.w_hh_t"])
    bias = graph_ops.add_constant(network, (1, 4 * hidden), weights[f"{pfx}.bias"])

    xw = network.add_matrix_multiply(x, trt.MatrixOperation.NONE, w_ih, trt.MatrixOperation.NONE)
    hw = network.add_matrix_multiply(h_prev, trt.MatrixOperation.NONE, w_hh, trt.MatrixOperation.NONE)
    gates = network.add_elementwise(xw.get_output(0), hw.get_output(0), trt.ElementWiseOperation.SUM)
    gates = network.add_elementwise(gates.get_output(0), bias, trt.ElementWiseOperation.SUM)

    gate_i = network.add_slice(gates.get_output(0), start=(0, 0), shape=(1, hidden), stride=(1, 1))
    gate_f = network.add_slice(gates.get_output(0), start=(0, hidden), shape=(1, hidden), stride=(1, 1))
    gate_g = network.add_slice(
        gates.get_output(0), start=(0, 2 * hidden), shape=(1, hidden), stride=(1, 1))
    gate_o = network.add_slice(
        gates.get_output(0), start=(0, 3 * hidden), shape=(1, hidden), stride=(1, 1))

    i_t = network.add_activation(gate_i.get_output(0), trt.ActivationType.SIGMOID).get_output(0)
    f_t = network.add_activation(gate_f.get_output(0), trt.ActivationType.SIGMOID).get_output(0)
    g_t = network.add_activation(gate_g.get_output(0), trt.ActivationType.TANH).get_output(0)
    o_t = network.add_activation(gate_o.get_output(0), trt.ActivationType.SIGMOID).get_output(0)

    forget = network.add_elementwise(f_t, c_prev, trt.ElementWiseOperation.PROD).get_output(0)
    update = network.add_elementwise(i_t, g_t, trt.ElementWiseOperation.PROD).get_output(0)
    c_new = network.add_elementwise(forget, update, trt.ElementWiseOperation.SUM).get_output(0)
    tanh_c = network.add_activation(c_new, trt.ActivationType.TANH).get_output(0)
    h_new = network.add_elementwise(o_t, tanh_c, trt.ElementWiseOperation.PROD).get_output(0)
    return h_new, c_new


_STREAMING_RIGHT_CONTEXTS = (13, 6, 1, 0)
_STREAMING_LEFT_CONTEXT = 70
_STREAMING_TIME_CACHE = 8
_STREAMING_PRE_ENCODE_CACHE = 9
_STREAMING_DROP_PRE_ENCODED = 2


def _streaming_mel_length(right_context: int, subsampling_factor: int = 8, *,
                          first_step: bool = False) -> int:
    if first_step:
        return 1 + subsampling_factor * right_context
    return _STREAMING_PRE_ENCODE_CACHE + subsampling_factor * (right_context + 1)


def _streaming_encoder_frames(right_context: int) -> int:
    return right_context + 1


def _rel_shift_streaming(network, x, heads: int, query_len: int, pos_len: int, key_len: int):
    zeros = graph_ops.add_constant(
        network, (heads, query_len, 1),
        np.zeros((heads, query_len, 1), dtype=np.float32))
    padded = network.add_concatenation([zeros, x])
    padded.axis = 2
    rs1 = network.add_shuffle(padded.get_output(0))
    rs1.reshape_dims = (heads, pos_len + 1, query_len)
    sl1 = network.add_slice(
        rs1.get_output(0), start=(0, 1, 0), shape=(heads, pos_len, query_len),
        stride=(1, 1, 1))
    rs2 = network.add_shuffle(sl1.get_output(0))
    rs2.reshape_dims = (heads, query_len, pos_len)
    sl2 = network.add_slice(
        rs2.get_output(0), start=(0, 0, 0), shape=(heads, query_len, key_len),
        stride=(1, 1, 1))
    return sl2.get_output(0)


def _slice_layer_channel_cache(network, cache, layer: int, cache_len: int, hidden: int):
    sl = network.add_slice(
        cache, start=(layer, 0, 0), shape=(1, cache_len, hidden), stride=(1, 1, 1))
    sh = network.add_shuffle(sl.get_output(0))
    sh.reshape_dims = (cache_len, hidden)
    return sh.get_output(0)


def _slice_layer_time_cache(network, cache, layer: int, hidden: int, time_cache: int):
    sl = network.add_slice(
        cache, start=(layer, 0, 0), shape=(1, hidden, time_cache), stride=(1, 1, 1))
    sh = network.add_shuffle(sl.get_output(0))
    sh.reshape_dims = (hidden, time_cache)
    return sh.get_output(0)


def _add_streaming_rel_pos_attention(network, hs, channel_cache, weights, pfx,
                                     hidden, heads, head_dim, query_len,
                                     cache_len, rpe, eps, enc_mask):
    key_len = cache_len + query_len
    pos_len = 2 * key_len - 1
    normed = graph_ops.add_layer_norm(
        network, hs, hidden, weights[f"{pfx}.norm_sa"], weights[f"{pfx}.norm_sa_b"], eps)

    cat_kv = network.add_concatenation([channel_cache, normed])
    cat_kv.axis = 0
    kv = cat_kv.get_output(0)

    q = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(network, normed, hidden, hidden, weights[f"{pfx}.w_q"]),
        hidden,
        weights[f"{pfx}.b_q"],
    )
    k = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(network, kv, hidden, hidden, weights[f"{pfx}.w_k"]),
        hidden,
        weights[f"{pfx}.b_k"],
    )
    v = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(network, kv, hidden, hidden, weights[f"{pfx}.w_v"]),
        hidden,
        weights[f"{pfx}.b_v"],
    )

    qr = network.add_shuffle(q)
    qr.reshape_dims = (query_len, heads, head_dim)
    kr = network.add_shuffle(k)
    kr.reshape_dims = (key_len, heads, head_dim)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (key_len, heads, head_dim)

    bu = graph_ops.add_constant(network, (1, heads, head_dim), weights[f"{pfx}.pos_bias_u"])
    bv = graph_ops.add_constant(network, (1, heads, head_dim), weights[f"{pfx}.pos_bias_v"])
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
        qu_t.get_output(0), trt.MatrixOperation.NONE,
        k_t.get_output(0), trt.MatrixOperation.TRANSPOSE).get_output(0)
    rp_t = network.add_shuffle(rpe)
    rp_t.first_transpose = trt.Permutation([1, 0, 2])
    ps_raw = network.add_matrix_multiply(
        qv_t.get_output(0), trt.MatrixOperation.NONE,
        rp_t.get_output(0), trt.MatrixOperation.TRANSPOSE).get_output(0)
    ps = _rel_shift_streaming(network, ps_raw, heads, query_len, pos_len, key_len)
    total = network.add_elementwise(cs, ps, trt.ElementWiseOperation.SUM).get_output(0)
    scale = graph_ops.add_constant(
        network, (1, 1, 1), np.array([1.0 / math.sqrt(head_dim)], dtype=np.float32))
    scaled = network.add_elementwise(total, scale, trt.ElementWiseOperation.PROD).get_output(0)
    if enc_mask is not None:
        scaled = network.add_elementwise(scaled, enc_mask, trt.ElementWiseOperation.SUM).get_output(0)

    sm = network.add_softmax(scaled)
    sm.axes = 1 << 2
    ao = network.add_matrix_multiply(
        sm.get_output(0), trt.MatrixOperation.NONE,
        v_t.get_output(0), trt.MatrixOperation.NONE).get_output(0)
    at = network.add_shuffle(ao)
    at.first_transpose = trt.Permutation([1, 0, 2])
    af = network.add_shuffle(at.get_output(0))
    af.reshape_dims = (query_len, hidden)
    out = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(network, af.get_output(0), hidden, hidden,
                                          weights[f"{pfx}.w_o"]),
        hidden,
        weights[f"{pfx}.b_o"],
    )

    cache_tail = network.add_slice(
        channel_cache, start=(query_len, 0), shape=(cache_len - query_len, hidden),
        stride=(1, 1)).get_output(0)
    next_cache = network.add_concatenation([cache_tail, normed])
    next_cache.axis = 0
    return out, next_cache.get_output(0)


def _add_streaming_conv_module(network, hs, time_cache, weights, pfx, hidden, kern,
                               query_len, eps, conv_norm_type):
    normed = graph_ops.add_layer_norm(
        network, hs, hidden, weights[f"{pfx}.norm_conv"], weights[f"{pfx}.norm_conv_b"], eps)
    r1 = network.add_shuffle(normed)
    r1.first_transpose = trt.Permutation([1, 0])
    r2 = network.add_shuffle(r1.get_output(0))
    r2.reshape_dims = (1, hidden, query_len)
    x = graph_ops.add_conv1d(
        network, r2.get_output(0), weight=weights[f"{pfx}.cpw1_w"],
        bias=weights[f"{pfx}.cpw1_b"], out_channels=2 * hidden, kernel_size=1)
    xa = network.add_slice(x, start=(0, 0, 0), shape=(1, hidden, query_len),
                           stride=(1, 1, 1)).get_output(0)
    xb = network.add_slice(x, start=(0, hidden, 0), shape=(1, hidden, query_len),
                           stride=(1, 1, 1)).get_output(0)
    gate = network.add_activation(xb, trt.ActivationType.SIGMOID).get_output(0)
    x = network.add_elementwise(xa, gate, trt.ElementWiseOperation.PROD).get_output(0)

    tc = network.add_shuffle(time_cache)
    tc.reshape_dims = (1, hidden, _STREAMING_TIME_CACHE)
    cached = network.add_concatenation([tc.get_output(0), x])
    cached.axis = 2
    cached_tensor = cached.get_output(0)
    next_time = network.add_slice(
        cached_tensor, start=(0, 0, query_len), shape=(1, hidden, _STREAMING_TIME_CACHE),
        stride=(1, 1, 1))
    x = graph_ops.add_conv1d(
        network, cached_tensor, weight=weights[f"{pfx}.cdw_w"], bias=weights[f"{pfx}.cdw_b"],
        out_channels=hidden, kernel_size=kern, groups=hidden)
    x = _add_conv_norm(network, x, weights, pfx, hidden, query_len, eps, conv_norm_type)
    x = graph_ops.add_activation(network, x, "silu")
    x = graph_ops.add_conv1d(
        network, x, weight=weights[f"{pfx}.cpw2_w"], bias=weights[f"{pfx}.cpw2_b"],
        out_channels=hidden, kernel_size=1)
    r3 = network.add_shuffle(x)
    r3.reshape_dims = (hidden, query_len)
    r4 = network.add_shuffle(r3.get_output(0))
    r4.first_transpose = trt.Permutation([1, 0])

    nt = network.add_shuffle(next_time.get_output(0))
    nt.reshape_dims = (hidden, _STREAMING_TIME_CACHE)
    return r4.get_output(0), nt.get_output(0)


def _add_streaming_conformer_block(network, hs, channel_cache, time_cache, weights, pfx,
                                   hidden, heads, head_dim, ffn, kern, query_len,
                                   cache_len, rpe, eps, enc_mask, conv_norm_type):
    ffn1 = _add_half_ffn(network, hs, weights, f"{pfx}.ff1", hidden, ffn, eps)
    hs = network.add_elementwise(hs, ffn1, trt.ElementWiseOperation.SUM).get_output(0)
    attn, next_channel = _add_streaming_rel_pos_attention(
        network, hs, channel_cache, weights, pfx, hidden, heads, head_dim, query_len,
        cache_len, rpe, eps, enc_mask)
    hs = network.add_elementwise(hs, attn, trt.ElementWiseOperation.SUM).get_output(0)
    conv, next_time = _add_streaming_conv_module(
        network, hs, time_cache, weights, pfx, hidden, kern, query_len, eps,
        conv_norm_type=conv_norm_type)
    hs = network.add_elementwise(hs, conv, trt.ElementWiseOperation.SUM).get_output(0)
    ffn2 = _add_half_ffn(network, hs, weights, f"{pfx}.ff2", hidden, ffn, eps)
    hs = network.add_elementwise(hs, ffn2, trt.ElementWiseOperation.SUM).get_output(0)
    out = graph_ops.add_layer_norm(
        network, hs, hidden, weights[f"{pfx}.norm_out"], weights[f"{pfx}.norm_out_b"], eps)
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


def _build_streaming_encoder(weights: WeightDict, right_context: int, *,
                             first_step: bool = False, verbose: bool = False) -> bytes:
    hidden = int(weights["_hidden"])
    enc_layers = int(weights["_enc_layers"])
    heads = int(weights["_enc_heads"])
    ffn = int(weights["_enc_ffn"])
    mel_bins = int(weights["_mel_bins"])
    kern = int(weights["_kern"])
    sub_ch = int(weights["_sub_ch"])
    head_dim = int(weights["_head_dim"])
    conv_norm_type = str(weights.get("_conv_norm_type", "batch_norm"))
    mel_len = _streaming_mel_length(right_context, first_step=first_step)
    pre_encoded = _compute_causal_enc_seq_len(mel_len)
    query_len = _streaming_encoder_frames(right_context)
    drop = pre_encoded - query_len
    expected_drop = 0 if first_step else _STREAMING_DROP_PRE_ENCODED
    if drop != expected_drop:
        raise ValueError(
            f"Unexpected streaming pre-encode drop for right={right_context}, "
            f"first_step={first_step}: {drop}")

    cache_len = _STREAMING_LEFT_CONTEXT
    key_len = cache_len + query_len
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    eps = graph_ops.add_constant(network, (1, 1), np.array([1e-5], dtype=np.float32))
    mel = network.add_input("mel_features", trt.float32, (mel_bins, mel_len))
    channel_cache = network.add_input(
        "cache_last_channel", trt.float32, (enc_layers, cache_len, hidden))
    time_cache = network.add_input(
        "cache_last_time", trt.float32, (enc_layers, hidden, _STREAMING_TIME_CACHE))
    enc_mask = network.add_input("encoder_mask", trt.float32, (1, query_len, key_len))

    sub_weights = dict(weights)
    sub_weights["_enc_seq"] = pre_encoded
    hs = _build_subsampling(network, mel, sub_weights, sub_ch, hidden, mel_bins, mel_len)
    hs = network.add_slice(hs, start=(drop, 0), shape=(query_len, hidden),
                           stride=(1, 1)).get_output(0)

    rpe_np = _relative_pe(key_len, hidden)
    next_channels = []
    next_times = []
    for i in range(enc_layers):
        pfx = f"el.{i}"
        rel_proj = rpe_np @ weights[f"{pfx}.w_pos"]
        rel_proj = rel_proj.reshape(2 * key_len - 1, heads, head_dim)
        rpe = graph_ops.add_constant(network, (2 * key_len - 1, heads, head_dim), rel_proj)
        layer_channel_cache = _slice_layer_channel_cache(network, channel_cache, i, cache_len, hidden)
        layer_time_cache = _slice_layer_time_cache(
            network, time_cache, i, hidden, _STREAMING_TIME_CACHE)
        hs, next_channel, next_time = _add_streaming_conformer_block(
            network, hs, layer_channel_cache, layer_time_cache, weights, pfx, hidden,
            heads, head_dim, ffn, kern, query_len, cache_len, rpe, eps, enc_mask,
            conv_norm_type=conv_norm_type)
        next_channels.append(next_channel)
        next_times.append(next_time)

    hs.name = "encoder_output"
    network.mark_output(hs)
    _mark_layer_cache_output(
        network, next_channels, "cache_last_channel_next", (cache_len, hidden))
    _mark_layer_cache_output(
        network, next_times, "cache_last_time_next", (hidden, _STREAMING_TIME_CACHE))

    if verbose:
        step = "first" if first_step else "next"
        print(f"[trtmc build] Building RNNT streaming encoder {step} "
              f"(right={right_context}, mel={mel_len}, frames={query_len})",
              file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(
            f"RNNT streaming encoder build failed for right={right_context}, "
            f"first_step={first_step}")
    return bytes(plan)


class NemotronSpeechStreamingPlugin:
    name = "nemotron_speech_streaming"
    runtime_strategy = "speech_to_text_rnnt"
    _DEFAULT_MEL_LENGTH = 3000

    def __init__(self):
        self._bundle_config: dict = {}
        self._vl_config: dict = {}

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower().replace("-", "_")
        return mt in {
            "nemotron_speech_streaming",
            "nemotron_asr_streaming",
            "nemotron_speech_streaming_rnnt",
            "fastconformer_cacheaware_rnnt",
            "enc_dec_rnnt_bpe",
            "rnnt_bpe",
        }

    def load_weights(self, model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict:
        del precision
        w = WeightDict()
        sd, ncfg = _load_nemo_archive(model_dir)
        if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
            sd = sd["state_dict"]
        _extract_tokenizer_from_nemo(model_dir, Path(model_dir))

        ec = ncfg.get("encoder", {})
        defaults = ncfg.get("model_defaults", {})
        dec_cfg = ncfg.get("decoder", {})
        prednet = dec_cfg.get("prednet", _cfg_dict(dec_cfg, "config_dict", "prednet"))
        joint_cfg = ncfg.get("joint", {})
        jointnet = joint_cfg.get("jointnet", _cfg_dict(joint_cfg, "config_dict", "jointnet"))

        hidden = _cfg_int(ec.get("d_model"), defaults.get("enc_hidden"), default=1024)
        mel_bins = _cfg_int(ncfg.get("preprocessor", {}).get("features"), ec.get("feat_in"), default=128)
        kern = _cfg_int(ec.get("conv_kernel_size"), default=9)
        conv_norm_type = str(ec.get("conv_norm_type", "batch_norm")).lower()
        conv_context_size = str(ec.get("conv_context_size", "symmetric")).lower()
        causal_downsampling = bool(ec.get("causal_downsampling", False))
        enc_heads = _cfg_int(ec.get("n_heads"), default=8)
        enc_ffn = _cfg_int(ec.get("ff_expansion_factor"), default=4) * hidden
        sub_ch = _cfg_int(ec.get("subsampling_conv_channels"), default=256)
        head_dim = hidden // enc_heads

        enc_layers = max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layers.")) + 1
        mel_length = _cfg_int(config.raw.get("mel_length"), ncfg.get("trtmc_mel_length"),
                              default=self._DEFAULT_MEL_LENGTH)
        enc_seq = (_compute_causal_enc_seq_len(mel_length)
                   if causal_downsampling else _compute_enc_seq_len(mel_length))
        att_contexts = ec.get("att_context_size") or [[70, 13]]
        att_context = att_contexts[0] if isinstance(att_contexts, list) and att_contexts else [70, 13]
        att_left = _cfg_int(att_context[0] if len(att_context) > 0 else None, default=70)
        att_right = _cfg_int(att_context[1] if len(att_context) > 1 else None, default=13)

        pred_hidden = _cfg_int(prednet.get("pred_hidden"), defaults.get("pred_hidden"), default=640)
        pred_layers = _cfg_int(prednet.get("pred_rnn_layers"), default=1)
        rnn_hidden = _cfg_int(prednet.get("rnn_hidden_size"), default=pred_hidden)
        if rnn_hidden != pred_hidden:
            raise ValueError(
                "Nemotron Speech Streaming RNNT currently supports predictor LSTMs without "
                f"projection (pred_hidden={pred_hidden}, rnn_hidden_size={rnn_hidden})."
            )
        joint_hidden = _cfg_int(jointnet.get("joint_hidden"), defaults.get("joint_hidden"),
                                default=pred_hidden)
        joint_activation = str(jointnet.get("activation", "relu")).lower()

        # --- Encoder: same FastConformer tensor layout as the native Canary encoder path. ---
        w["_enc_layers"] = enc_layers
        w["_enc_heads"] = enc_heads
        w["_enc_ffn"] = enc_ffn
        w["_hidden"] = hidden
        w["_mel_bins"] = mel_bins
        w["_kern"] = kern
        w["_mel_length"] = mel_length
        w["_enc_seq"] = enc_seq
        w["_sub_ch"] = sub_ch
        w["_head_dim"] = head_dim
        w["_conv_norm_type"] = conv_norm_type
        w["_conv_context_size"] = conv_context_size
        w["_causal_downsampling"] = causal_downsampling
        w["_encoder_attention_mask_2d"] = True

        w["enc_sub_conv0_w"] = _to_np(sd["encoder.pre_encode.conv.0.weight"])
        w["enc_sub_conv0_b"] = _to_np(sd["encoder.pre_encode.conv.0.bias"])
        for s, (di, pi) in enumerate([(2, 3), (5, 6)]):
            w[f"enc_sub_dw{s}_w"] = _to_np(sd[f"encoder.pre_encode.conv.{di}.weight"])
            w[f"enc_sub_dw{s}_b"] = _to_np(sd[f"encoder.pre_encode.conv.{di}.bias"])
            w[f"enc_sub_pw{s}_w"] = _to_np(sd[f"encoder.pre_encode.conv.{pi}.weight"])
            w[f"enc_sub_pw{s}_b"] = _to_np(sd[f"encoder.pre_encode.conv.{pi}.bias"])
        w["enc_sub_out_w"] = _transpose_2d(_to_np(sd["encoder.pre_encode.out.weight"]), "sub")
        w["enc_sub_out_b"] = _to_np(sd["encoder.pre_encode.out.bias"])

        for i in range(enc_layers):
            nk = f"encoder.layers.{i}"
            pk = f"el.{i}"
            for p, n in [("w_q", "linear_q"), ("w_k", "linear_k"), ("w_v", "linear_v"),
                         ("w_o", "linear_out")]:
                w[f"{pk}.{p}"] = _transpose_2d(_to_np(sd[f"{nk}.self_attn.{n}.weight"]), p)
                bk = f"{nk}.self_attn.{n}.bias"
                w[f"{pk}.b_{p[-1]}"] = _to_np(sd[bk]) if bk in sd else np.zeros(hidden, dtype=np.float32)
            w[f"{pk}.pos_bias_u"] = _to_np(sd[f"{nk}.self_attn.pos_bias_u"])
            w[f"{pk}.pos_bias_v"] = _to_np(sd[f"{nk}.self_attn.pos_bias_v"])
            w[f"{pk}.w_pos"] = _transpose_2d(_to_np(sd[f"{nk}.self_attn.linear_pos.weight"]), "pos")
            w[f"{pk}.norm_sa"] = _to_np(sd[f"{nk}.norm_self_att.weight"])
            w[f"{pk}.norm_sa_b"] = _to_np(sd[f"{nk}.norm_self_att.bias"])
            for fn, fk in [("ff1", "feed_forward1"), ("ff2", "feed_forward2")]:
                w[f"{pk}.{fn}.w1"] = _transpose_2d(_to_np(sd[f"{nk}.{fk}.linear1.weight"]), f"{fn}1")
                b1 = f"{nk}.{fk}.linear1.bias"
                w[f"{pk}.{fn}.b1"] = _to_np(sd[b1]) if b1 in sd else np.zeros(enc_ffn, dtype=np.float32)
                w[f"{pk}.{fn}.w2"] = _transpose_2d(_to_np(sd[f"{nk}.{fk}.linear2.weight"]), f"{fn}2")
                b2 = f"{nk}.{fk}.linear2.bias"
                w[f"{pk}.{fn}.b2"] = _to_np(sd[b2]) if b2 in sd else np.zeros(hidden, dtype=np.float32)
                nm = "norm_feed_forward1" if fn == "ff1" else "norm_feed_forward2"
                w[f"{pk}.{fn}.norm"] = _to_np(sd[f"{nk}.{nm}.weight"])
                w[f"{pk}.{fn}.norm_b"] = _to_np(sd[f"{nk}.{nm}.bias"])
            w[f"{pk}.cpw1_w"] = _to_np(sd[f"{nk}.conv.pointwise_conv1.weight"])
            cpw1_b = f"{nk}.conv.pointwise_conv1.bias"
            w[f"{pk}.cpw1_b"] = _to_np(sd[cpw1_b]) if cpw1_b in sd else np.zeros(2 * hidden, dtype=np.float32)
            w[f"{pk}.cdw_w"] = _to_np(sd[f"{nk}.conv.depthwise_conv.weight"])
            cdw_b = f"{nk}.conv.depthwise_conv.bias"
            w[f"{pk}.cdw_b"] = _to_np(sd[cdw_b]) if cdw_b in sd else np.zeros(hidden, dtype=np.float32)
            w[f"{pk}.bn_w"] = _to_np(sd[f"{nk}.conv.batch_norm.weight"])
            w[f"{pk}.bn_b"] = _to_np(sd[f"{nk}.conv.batch_norm.bias"])
            w[f"{pk}.bn_m"] = _to_np(sd[f"{nk}.conv.batch_norm.running_mean"]) if f"{nk}.conv.batch_norm.running_mean" in sd else np.zeros(hidden, dtype=np.float32)
            w[f"{pk}.bn_v"] = _to_np(sd[f"{nk}.conv.batch_norm.running_var"]) if f"{nk}.conv.batch_norm.running_var" in sd else np.ones(hidden, dtype=np.float32)
            w[f"{pk}.cpw2_w"] = _to_np(sd[f"{nk}.conv.pointwise_conv2.weight"])
            cpw2_b = f"{nk}.conv.pointwise_conv2.bias"
            w[f"{pk}.cpw2_b"] = _to_np(sd[cpw2_b]) if cpw2_b in sd else np.zeros(hidden, dtype=np.float32)
            w[f"{pk}.norm_conv"] = _to_np(sd[f"{nk}.norm_conv.weight"])
            w[f"{pk}.norm_conv_b"] = _to_np(sd[f"{nk}.norm_conv.bias"])
            w[f"{pk}.norm_out"] = _to_np(sd[f"{nk}.norm_out.weight"])
            w[f"{pk}.norm_out_b"] = _to_np(sd[f"{nk}.norm_out.bias"])

        rpe = _relative_pe(enc_seq, hidden)
        for i in range(enc_layers):
            proj = rpe @ w[f"el.{i}.w_pos"]
            w[f"el.{i}.rpe_proj"] = proj.reshape(2 * enc_seq - 1, enc_heads, head_dim)

        # --- RNNT predictor. ---
        embed = _to_np(_find_tensor(sd, ["decoder.prediction.embed.weight"], "predictor embedding"))
        vocab_total = int(embed.shape[0])
        blank_id = _cfg_int(dec_cfg.get("blank_idx"), default=vocab_total - 1)
        if blank_id >= vocab_total:
            raise ValueError(
                f"RNNT blank_id={blank_id} is outside predictor embedding rows={vocab_total}; "
                "blank_as_pad=False checkpoints are not supported yet."
            )
        vocab = blank_id
        w["pred_embedding"] = embed
        w["_pred_hidden"] = pred_hidden
        w["_pred_layers"] = pred_layers
        w["_vocab"] = vocab
        w["_vocab_total"] = vocab_total
        w["_blank_id"] = blank_id

        for i in range(pred_layers):
            pfx = f"pred.{i}"
            base_candidates = [
                f"decoder.prediction.dec_rnn.weight_{{kind}}_l{i}",
                f"decoder.prediction.dec_rnn.lstm.weight_{{kind}}_l{i}",
                f"decoder.prediction.dec_rnn.rnn.weight_{{kind}}_l{i}",
                f"decoder.prediction.dec_rnn._rnn.weight_{{kind}}_l{i}",
            ]
            b_candidates = [
                f"decoder.prediction.dec_rnn.bias_{{kind}}_l{i}",
                f"decoder.prediction.dec_rnn.lstm.bias_{{kind}}_l{i}",
                f"decoder.prediction.dec_rnn.rnn.bias_{{kind}}_l{i}",
                f"decoder.prediction.dec_rnn._rnn.bias_{{kind}}_l{i}",
            ]
            w_ih = _to_np(_find_tensor(sd, [c.format(kind="ih") for c in base_candidates],
                                       f"predictor layer {i} weight_ih"))
            w_hh = _to_np(_find_tensor(sd, [c.format(kind="hh") for c in base_candidates],
                                       f"predictor layer {i} weight_hh"))
            b_ih = _to_np(_find_tensor(sd, [c.format(kind="ih") for c in b_candidates],
                                       f"predictor layer {i} bias_ih"))
            b_hh = _to_np(_find_tensor(sd, [c.format(kind="hh") for c in b_candidates],
                                       f"predictor layer {i} bias_hh"))
            if w_ih.shape != (4 * pred_hidden, pred_hidden) or w_hh.shape != (4 * pred_hidden, pred_hidden):
                raise ValueError(
                    f"Unsupported predictor LSTM layer {i} shapes: "
                    f"w_ih={w_ih.shape}, w_hh={w_hh.shape}, expected "
                    f"{(4 * pred_hidden, pred_hidden)}."
                )
            w[f"{pfx}.w_ih_t"] = np.ascontiguousarray(w_ih.T.astype(np.float32))
            w[f"{pfx}.w_hh_t"] = np.ascontiguousarray(w_hh.T.astype(np.float32))
            w[f"{pfx}.bias"] = (b_ih + b_hh).astype(np.float32).reshape(1, -1)

        # --- RNNT joint. ---
        joint_prefix = "joint"
        w["joint_enc_w"] = _transpose_2d(
            _to_np(_find_tensor(sd, [f"{joint_prefix}.enc.weight"], "joint encoder projection")),
            "joint_enc")
        w["joint_enc_b"] = _to_np(_find_tensor(sd, [f"{joint_prefix}.enc.bias"], "joint encoder bias"))
        w["joint_pred_w"] = _transpose_2d(
            _to_np(_find_tensor(sd, [f"{joint_prefix}.pred.weight"], "joint predictor projection")),
            "joint_pred")
        w["joint_pred_b"] = _to_np(_find_tensor(sd, [f"{joint_prefix}.pred.bias"], "joint predictor bias"))
        out_w, out_b = _find_joint_linear(sd, joint_prefix, "joint output")
        w["joint_out_w"] = _transpose_2d(_to_np(out_w), "joint_out")
        w["joint_out_b"] = _to_np(out_b)
        w["_joint_hidden"] = joint_hidden
        w["_joint_activation"] = joint_activation

        config.hidden_size = pred_hidden
        config.vocab_size = vocab_total
        config.num_hidden_layers = pred_layers
        config.num_attention_heads = 1
        config.num_key_value_heads = 1

        self._vl_config = {
            "has_vision_engine": True,
            "num_mel_bins": mel_bins,
            "max_source_positions": enc_seq,
            "encoder_layers": enc_layers,
            "mel_length": mel_length,
            "subsampling_factor": 8,
            "sample_rate": 16000,
        }
        self._bundle_config = {
            "runtime_strategy": self.runtime_strategy,
            "hidden_size": pred_hidden,
            "num_hidden_layers": pred_layers,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": vocab_total,
            "rnnt_encoder_hidden_size": hidden,
            "rnnt_pred_hidden_size": pred_hidden,
            "rnnt_pred_num_layers": pred_layers,
            "rnnt_encoder_layers": enc_layers,
            "rnnt_joint_hidden_size": joint_hidden,
            "rnnt_vocab_size": vocab,
            "rnnt_blank_id": blank_id,
            "rnnt_max_symbols_per_step": _cfg_int(ncfg.get("decoding", {}).get("max_symbols_per_step"),
                                                  default=10),
            "rnnt_causal_downsampling": causal_downsampling,
            "rnnt_att_context_left": att_left,
            "rnnt_att_context_right": att_right,
            "rnnt_streaming_cache_left": _STREAMING_LEFT_CONTEXT,
            "rnnt_streaming_time_cache": _STREAMING_TIME_CACHE,
            "rnnt_streaming_pre_encode_cache": _STREAMING_PRE_ENCODE_CACHE,
            "rnnt_streaming_drop_pre_encoded": _STREAMING_DROP_PRE_ENCODED,
            "rnnt_streaming_right_contexts": list(_STREAMING_RIGHT_CONTEXTS),
        }
        return w

    def build_engine(self, config: ModelConfig, weights: WeightDict, max_cache_length: int,
                     *, precision: str = "fp32", quant_ctx=None, verbose: bool = False,
                     debug_layer_outputs: bool = False, parallel_config=None) -> bytes:
        del config, max_cache_length, precision
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="Nemotron Speech Streaming TP predictor builds")
            if quant_ctx is not None:
                raise ValueError(
                    "Nemotron Speech Streaming tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError(
                    "Nemotron Speech Streaming tensor-parallel builds do not support "
                    "debug_layer_outputs")
            return build_nemotron_streaming_tp_predictor(
                weights, verbose=verbose, parallel_config=parallel)
        return _build_predictor(weights, verbose=verbose)

    def build_vision_engine(self, model_dir: str, config: ModelConfig, weights: WeightDict,
                            *, precision: str = "fp32", verbose: bool = False) -> bytes | None:
        del model_dir, precision
        return _build_encoder(config, weights, verbose=verbose)

    def build_extra_engines(self, config: ModelConfig, weights: WeightDict, max_cache_length: int,
                            *, precision: str = "fp32", verbose: bool = False) -> dict | None:
        del config, max_cache_length, precision
        joint_plan = _build_joint(weights, verbose=verbose)
        extras = {"joint_engine_plan": joint_plan}
        for right_context in _STREAMING_RIGHT_CONTEXTS:
            extras[f"streaming_encoder_plan_ctx{right_context}"] = _build_streaming_encoder(
                weights, right_context, verbose=verbose)
            extras[f"streaming_encoder_first_plan_ctx{right_context}"] = _build_streaming_encoder(
                weights, right_context, first_step=True, verbose=verbose)
        mel = _build_mel_filterbank(weights, verbose=verbose)
        if mel is not None:
            extras["mel_filterbank"] = mel
        return extras

    def get_audio_config(self, config: ModelConfig) -> dict | None:
        del config
        return {
            "mel_n_fft": 512,
            "mel_win_length": 400,
            "mel_hop_length": 160,
            "mel_chunk_length": 30,
            "mel_sampling_rate": 16000,
        }

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        del config
        return self._vl_config or {
            "has_vision_engine": True,
            "num_mel_bins": 128,
            "max_source_positions": 375,
            "encoder_layers": 24,
            "mel_length": self._DEFAULT_MEL_LENGTH,
            "subsampling_factor": 8,
            "sample_rate": 16000,
        }

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict | None:
        del config
        return self._bundle_config or {
            "runtime_strategy": self.runtime_strategy,
            "rnnt_blank_id": 1024,
            "rnnt_vocab_size": 1024,
            "rnnt_pred_hidden_size": 640,
            "rnnt_pred_num_layers": 1,
            "rnnt_encoder_hidden_size": 1024,
            "rnnt_max_symbols_per_step": 10,
            "rnnt_encoder_layers": 24,
            "rnnt_streaming_cache_left": _STREAMING_LEFT_CONTEXT,
            "rnnt_streaming_time_cache": _STREAMING_TIME_CACHE,
            "rnnt_streaming_pre_encode_cache": _STREAMING_PRE_ENCODE_CACHE,
            "rnnt_streaming_drop_pre_encoded": _STREAMING_DROP_PRE_ENCODED,
            "rnnt_streaming_right_contexts": list(_STREAMING_RIGHT_CONTEXTS),
        }


def _build_predictor(weights: WeightDict, *, verbose: bool = False) -> bytes:
    pred_hidden = int(weights["_pred_hidden"])
    pred_layers = int(weights["_pred_layers"])
    vocab_total = int(weights["_vocab_total"])

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)

    token_id = network.add_input("token_id", trt.int32, (1,))
    embedding = graph_ops.add_constant(network, (vocab_total, pred_hidden), weights["pred_embedding"])
    hidden = network.add_gather(embedding, token_id, 0).get_output(0)

    next_h = []
    next_c = []
    for layer in range(pred_layers):
        h_in = network.add_input(f"state_h_{layer}", trt.float32, (1, pred_hidden))
        c_in = network.add_input(f"state_c_{layer}", trt.float32, (1, pred_hidden))
        hidden, c_new = _add_lstm_cell(network, hidden, h_in, c_in, weights, f"pred.{layer}",
                                       pred_hidden)
        next_h.append(hidden)
        next_c.append(c_new)

    pred_output = network.add_identity(hidden).get_output(0)
    pred_output.name = "pred_output"
    network.mark_output(pred_output)
    for layer in range(pred_layers):
        next_h[layer].name = f"next_h_{layer}"
        next_c[layer].name = f"next_c_{layer}"
        network.mark_output(next_h[layer])
        network.mark_output(next_c[layer])

    if verbose:
        print(f"[trtmc build] Building RNNT predictor ({pred_layers}L, h={pred_hidden})",
              file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("RNNT predictor build failed")
    return bytes(plan)


def _build_joint(weights: WeightDict, *, verbose: bool = False) -> bytes:
    enc_hidden = int(weights["_hidden"])
    pred_hidden = int(weights["_pred_hidden"])
    joint_hidden = int(weights["_joint_hidden"])
    vocab_total = int(weights["_vocab_total"])
    activation = str(weights["_joint_activation"]).lower()

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)

    enc = network.add_input("encoder_frame", trt.float32, (1, enc_hidden))
    pred = network.add_input("pred_output", trt.float32, (1, pred_hidden))
    enc_proj = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(network, enc, enc_hidden, joint_hidden,
                                          weights["joint_enc_w"]),
        joint_hidden,
        weights["joint_enc_b"],
    )
    pred_proj = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(network, pred, pred_hidden, joint_hidden,
                                          weights["joint_pred_w"]),
        joint_hidden,
        weights["joint_pred_b"],
    )
    joint = network.add_elementwise(enc_proj, pred_proj, trt.ElementWiseOperation.SUM).get_output(0)
    if activation == "relu":
        joint = network.add_activation(joint, trt.ActivationType.RELU).get_output(0)
    elif activation == "tanh":
        joint = network.add_activation(joint, trt.ActivationType.TANH).get_output(0)
    elif activation == "sigmoid":
        joint = network.add_activation(joint, trt.ActivationType.SIGMOID).get_output(0)
    else:
        raise ValueError(f"Unsupported RNNT joint activation: {activation}")

    logits = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(network, joint, joint_hidden, vocab_total,
                                          weights["joint_out_w"]),
        vocab_total,
        weights["joint_out_b"],
    )
    logits.name = "logits"
    network.mark_output(logits)

    if verbose:
        print(f"[trtmc build] Building RNNT joint (enc={enc_hidden}, pred={pred_hidden}, "
              f"joint={joint_hidden}, vocab={vocab_total})", file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("RNNT joint build failed")
    return bytes(plan)


def _build_mel_filterbank(weights: WeightDict, *, verbose: bool = False) -> bytes | None:
    num_mel_bins = int(weights.get("_mel_bins", 128))
    n_fft = 512
    sampling_rate = 16000
    n_freq_bins = 1 + n_fft // 2
    try:
        import librosa
        filters = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mel_bins,
            fmin=0.0,
            fmax=sampling_rate / 2.0,
            norm="slaney",
        ).T
    except ImportError:
        try:
            from transformers.audio_utils import mel_filter_bank
        except ImportError:
            print("[trtmc build] Warning: mel filterbank builder not available", file=sys.stderr)
            return None
        filters = mel_filter_bank(
            num_frequency_bins=n_freq_bins,
            num_mel_filters=num_mel_bins,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
    filters_flat = np.ascontiguousarray(filters, dtype=np.float32)
    header = np.array([n_freq_bins, num_mel_bins], dtype=np.int32)
    if verbose:
        print(f"[trtmc build] RNNT mel filterbank: {n_freq_bins}x{num_mel_bins}",
              file=sys.stderr)
    return header.tobytes() + filters_flat.tobytes()


plugin = NemotronSpeechStreamingPlugin()
