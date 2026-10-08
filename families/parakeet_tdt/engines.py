# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-local TensorRT graphs ported from PR #1060; native HF input only."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import tensorrt as trt
from . import graph_ops
from .checkpoint import WeightDict, _transpose_2d
from .model_config import ModelConfig
from .fastconformer import _build_encoder, _compute_enc_seq_len, _compute_causal_enc_seq_len, _relative_pe, _to_np


def _load_hf_as_nemo(model_dir: str):
    """Expose the official HF safetensors under the NeMo-style encoder keys.

    The graph builders consume one normalized, family-private layout.  Keeping
    this conversion explicit avoids choosing a topology from a checkpoint name.
    """
    from safetensors.numpy import load_file

    root = Path(model_dir)
    state = dict(load_file(root / "model.safetensors"))
    if any(key.startswith("prompt_kernel.") for key in state):
        raise ValueError("Parakeet TDT native checkpoint does not support prompt_kernel")
    from .config import ParakeetTDTConfig

    cfg = ParakeetTDTConfig.from_dir(root)
    cfg.validate_supported_checkpoint()
    normalized = dict(state)
    aliases = {
        "encoder.pre_encode.conv.0.weight": "encoder.subsampling.layers.0.weight",
        "encoder.pre_encode.conv.0.bias": "encoder.subsampling.layers.0.bias",
        "encoder.pre_encode.conv.2.weight": "encoder.subsampling.layers.2.weight",
        "encoder.pre_encode.conv.2.bias": "encoder.subsampling.layers.2.bias",
        "encoder.pre_encode.conv.3.weight": "encoder.subsampling.layers.3.weight",
        "encoder.pre_encode.conv.3.bias": "encoder.subsampling.layers.3.bias",
        "encoder.pre_encode.conv.5.weight": "encoder.subsampling.layers.5.weight",
        "encoder.pre_encode.conv.5.bias": "encoder.subsampling.layers.5.bias",
        "encoder.pre_encode.conv.6.weight": "encoder.subsampling.layers.6.weight",
        "encoder.pre_encode.conv.6.bias": "encoder.subsampling.layers.6.bias",
        "encoder.pre_encode.out.weight": "encoder.subsampling.linear.weight",
        "encoder.pre_encode.out.bias": "encoder.subsampling.linear.bias",
        "decoder.prediction.embed.weight": "decoder.embedding.weight",
        "joint.enc.weight": "encoder_projector.weight",
        "joint.enc.bias": "encoder_projector.bias",
        "joint.pred.weight": "decoder.decoder_projector.weight",
        "joint.pred.bias": "decoder.decoder_projector.bias",
        "joint.joint_net.2.weight": "joint.head.weight",
        "joint.joint_net.2.bias": "joint.head.bias",
    }
    for layer in range(cfg.decoder_layers):
        for field in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            aliases[f"decoder.prediction.dec_rnn.{field}_l{layer}"] = (
                f"decoder.lstm.{field}_l{layer}")
    for layer in range(cfg.encoder_layers):
        nk = f"encoder.layers.{layer}"
        for nemo, hf in (("linear_q", "q_proj"), ("linear_k", "k_proj"),
                         ("linear_v", "v_proj"), ("linear_out", "o_proj"),
                         ("linear_pos", "relative_k_proj")):
            aliases[f"{nk}.self_attn.{nemo}.weight"] = f"{nk}.self_attn.{hf}.weight"
        aliases[f"{nk}.self_attn.pos_bias_u"] = f"{nk}.self_attn.bias_u"
        aliases[f"{nk}.self_attn.pos_bias_v"] = f"{nk}.self_attn.bias_v"
        aliases[f"{nk}.conv.batch_norm.weight"] = f"{nk}.conv.norm.weight"
        aliases[f"{nk}.conv.batch_norm.bias"] = f"{nk}.conv.norm.bias"
        aliases[f"{nk}.conv.batch_norm.running_mean"] = f"{nk}.conv.norm.running_mean"
        aliases[f"{nk}.conv.batch_norm.running_var"] = f"{nk}.conv.norm.running_var"
    missing = [(dst, src) for dst, src in aliases.items() if src not in state]
    if missing:
        detail = ", ".join(f"{dst} <- {src}" for dst, src in missing[:8])
        raise KeyError(f"HF Parakeet checkpoint is missing normalized tensors: {detail}")
    normalized.update({dst: state[src] for dst, src in aliases.items()})
    nemo_cfg = {
        "encoder": {"d_model": cfg.encoder_hidden_size,
                    "n_layers": cfg.encoder_layers,
                    "n_heads": cfg.encoder_heads,
                    "ff_expansion_factor": cfg.encoder_ffn_size // cfg.encoder_hidden_size,
                    "conv_kernel_size": cfg.encoder_conv_kernel_size,
                    "subsampling_conv_channels": cfg.subsampling_channels,
                    "att_context_size": [[-1, -1]],
                    "conv_norm_type": "batch_norm", "conv_context_size": "symmetric"},
        "preprocessor": {"features": cfg.num_mel_bins},
        "decoder": {"blank_idx": cfg.blank_id,
                    "prednet": {"pred_hidden": cfg.decoder_hidden_size,
                                "pred_rnn_layers": cfg.decoder_layers,
                                "rnn_hidden_size": cfg.decoder_hidden_size}},
        "joint": {"jointnet": {"joint_hidden": cfg.decoder_hidden_size,
                                "activation": cfg.joint_activation}},
        "decoding": {"max_symbols_per_step": cfg.max_symbols_per_step},
        "tdt_durations": list(cfg.durations),
    }
    return normalized, nemo_cfg


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


def _precision_dtypes(precision: str) -> tuple[type[np.generic], object]:
    if precision == "fp16":
        return np.float16, trt.float16
    if precision == "fp32":
        return np.float32, trt.float32
    raise ValueError(
        f"Unsupported Parakeet TDT precision {precision!r}; "
        "expected fp32 or fp16")


def _add_lstm_cell(
        network, x, h_prev, c_prev, weights, pfx: str, hidden: int,
        dtype=np.float32):
    w_ih = graph_ops.add_constant(
        network, (hidden, 4 * hidden), weights[f"{pfx}.w_ih_t"], dtype=dtype)
    w_hh = graph_ops.add_constant(
        network, (hidden, 4 * hidden), weights[f"{pfx}.w_hh_t"], dtype=dtype)
    bias = graph_ops.add_constant(
        network, (1, 4 * hidden), weights[f"{pfx}.bias"], dtype=dtype)

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


def _build_predictor(
        weights: WeightDict, *, precision: str = "fp32",
        verbose: bool = False) -> bytes:
    pred_hidden = int(weights["_pred_hidden"])
    pred_layers = int(weights["_pred_layers"])
    vocab_total = int(weights["_vocab_total"])
    work_np_dtype, work_trt_dtype = _precision_dtypes(precision)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)

    token_id = network.add_input("token_id", trt.int32, (1,))
    embedding = graph_ops.add_constant(
        network, (vocab_total, pred_hidden), weights["pred_embedding"],
        dtype=work_np_dtype)
    hidden = network.add_gather(embedding, token_id, 0).get_output(0)

    next_h = []
    next_c = []
    for layer in range(pred_layers):
        h_in = network.add_input(f"state_h_{layer}", trt.float32, (1, pred_hidden))
        c_in = network.add_input(f"state_c_{layer}", trt.float32, (1, pred_hidden))
        if work_trt_dtype != trt.float32:
            h_in = network.add_cast(h_in, work_trt_dtype).get_output(0)
            c_in = network.add_cast(c_in, work_trt_dtype).get_output(0)
        hidden, c_new = _add_lstm_cell(network, hidden, h_in, c_in, weights, f"pred.{layer}",
                                       pred_hidden, dtype=work_np_dtype)
        next_h.append(hidden)
        next_c.append(c_new)

    # Keep pred_output distinct from the last recurrent state even in FP32;
    # TensorRT output names belong to tensors, so aliasing would rename the
    # same tensor from pred_output to next_h_{last_layer}.
    pred_output = network.add_identity(hidden).get_output(0)
    if pred_output.dtype != trt.float32:
        pred_output = network.add_cast(pred_output, trt.float32).get_output(0)
    pred_output.name = "pred_output"
    network.mark_output(pred_output)
    for layer in range(pred_layers):
        h_output = next_h[layer]
        c_output = next_c[layer]
        if h_output.dtype != trt.float32:
            h_output = network.add_cast(h_output, trt.float32).get_output(0)
            c_output = network.add_cast(c_output, trt.float32).get_output(0)
        h_output.name = f"next_h_{layer}"
        c_output.name = f"next_c_{layer}"
        network.mark_output(h_output)
        network.mark_output(c_output)

    if verbose:
        print(f"[trtmc build] Building TDT predictor ({pred_layers}L, h={pred_hidden})",
              file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TDT predictor build failed")
    return bytes(plan)


def _build_joint(
        weights: WeightDict, *, precision: str = "fp32",
        verbose: bool = False) -> bytes:
    enc_hidden = int(weights["_hidden"])
    pred_hidden = int(weights["_pred_hidden"])
    joint_hidden = int(weights["_joint_hidden"])
    vocab_total = int(weights["_vocab_total"])
    joint_output_size = int(weights["_joint_output_size"])
    duration_count = len(weights["_duration_values"])
    activation = str(weights["_joint_activation"]).lower()
    work_np_dtype, work_trt_dtype = _precision_dtypes(precision)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 1
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)

    enc = network.add_input("encoder_frame", trt.float32, (1, enc_hidden))
    pred = network.add_input("pred_output", trt.float32, (1, pred_hidden))
    if work_trt_dtype != trt.float32:
        enc = network.add_cast(enc, work_trt_dtype).get_output(0)
        pred = network.add_cast(pred, work_trt_dtype).get_output(0)
    enc_proj = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(network, enc, enc_hidden, joint_hidden,
                                          weights["joint_enc_w"],
                                          dtype=work_np_dtype),
        joint_hidden,
        weights["joint_enc_b"],
        dtype=work_np_dtype,
    )
    pred_proj = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(network, pred, pred_hidden, joint_hidden,
                                          weights["joint_pred_w"],
                                          dtype=work_np_dtype),
        joint_hidden,
        weights["joint_pred_b"],
        dtype=work_np_dtype,
    )
    joint = network.add_elementwise(enc_proj, pred_proj, trt.ElementWiseOperation.SUM).get_output(0)
    if activation == "relu":
        joint = network.add_activation(joint, trt.ActivationType.RELU).get_output(0)
    elif activation == "tanh":
        joint = network.add_activation(joint, trt.ActivationType.TANH).get_output(0)
    elif activation == "sigmoid":
        joint = network.add_activation(joint, trt.ActivationType.SIGMOID).get_output(0)
    else:
        raise ValueError(f"Unsupported TDT joint activation: {activation}")

    logits = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(network, joint, joint_hidden, joint_output_size,
                                          weights["joint_out_w"],
                                          dtype=work_np_dtype),
        joint_output_size,
        weights["joint_out_b"],
        dtype=work_np_dtype,
    )
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    token_slice = network.add_slice(logits, (0, 0), (1, vocab_total), (1, 1)).get_output(0)
    duration_slice = network.add_slice(
        logits, (0, vocab_total), (1, duration_count), (1, 1)).get_output(0)
    token_slice.name = "token_logits"
    duration_slice.name = "duration_logits"
    network.mark_output(token_slice)
    network.mark_output(duration_slice)

    if verbose:
        print(f"[trtmc build] Building TDT joint (enc={enc_hidden}, pred={pred_hidden}, "
              f"joint={joint_hidden}, vocab={vocab_total})", file=sys.stderr)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TDT joint build failed")
    return bytes(plan)


def _build_mel_filterbank(weights: WeightDict, *, verbose: bool = False) -> bytes | None:
    num_mel_bins = int(weights.get("_mel_bins", 128))
    n_fft = 512
    sampling_rate = 16000
    n_freq_bins = 1 + n_fft // 2
    import librosa
    filters = librosa.filters.mel(
        sr=sampling_rate, n_fft=n_fft, n_mels=num_mel_bins,
        fmin=0.0, fmax=sampling_rate / 2.0, norm="slaney",
    ).T
    filters_flat = np.ascontiguousarray(filters, dtype=np.float32)
    header = np.array([n_freq_bins, num_mel_bins], dtype=np.int32)
    if verbose:
        print(f"[trtmc build] TDT mel filterbank: {n_freq_bins}x{num_mel_bins}",
              file=sys.stderr)
    return header.tobytes() + filters_flat.tobytes()


def _load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> tuple[WeightDict, dict]:
    del precision
    w = WeightDict()
    sd, ncfg = _load_hf_as_nemo(model_dir)

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
                          default=3000)
    enc_seq = (_compute_causal_enc_seq_len(mel_length)
               if causal_downsampling else _compute_enc_seq_len(mel_length))
    att_contexts = ec.get("att_context_size") or [[-1, -1]]
    att_context = att_contexts[0] if isinstance(att_contexts, list) and att_contexts else [70, 13]
    att_left = _cfg_int(att_context[0] if len(att_context) > 0 else None, default=70)
    att_right = _cfg_int(att_context[1] if len(att_context) > 1 else None, default=13)

    # Streaming knobs are checkpoint-defined. The NeMo config exposes the full
    # list of supported att_context_size pairs; drive the per-right-context
    # engine set from that list.
    streaming_right_contexts = []
    streaming_cache_left = att_left
    w["_streaming_right_contexts"] = streaming_right_contexts
    w["_streaming_cache_left"] = streaming_cache_left

    pred_hidden = _cfg_int(prednet.get("pred_hidden"), defaults.get("pred_hidden"), default=640)
    pred_layers = _cfg_int(prednet.get("pred_rnn_layers"), default=1)
    rnn_hidden = _cfg_int(prednet.get("rnn_hidden_size"), default=pred_hidden)
    if rnn_hidden != pred_hidden:
        raise ValueError(
            "Parakeet TDT TDT currently supports predictor LSTMs without "
            f"projection (pred_hidden={pred_hidden}, rnn_hidden_size={rnn_hidden})."
        )
    joint_hidden = _cfg_int(jointnet.get("joint_hidden"), defaults.get("joint_hidden"),
                            default=pred_hidden)
    joint_activation = str(jointnet.get("activation", "relu")).lower()

    # --- Encoder: same FastConformer tensor layout as the native Parakeet TDT encoder path. ---
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

    # --- TDT predictor. ---
    embed = _to_np(_find_tensor(sd, ["decoder.prediction.embed.weight"], "predictor embedding"))
    vocab_total = int(embed.shape[0])
    blank_id = _cfg_int(dec_cfg.get("blank_idx"), default=vocab_total - 1)
    if blank_id >= vocab_total:
        raise ValueError(
            f"TDT blank_id={blank_id} is outside predictor embedding rows={vocab_total}; "
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

    # --- TDT joint. ---
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
    duration_values = tuple(int(x) for x in ncfg.get("tdt_durations", [0, 1, 2, 3, 4]))
    joint_output_size = int(w["joint_out_b"].shape[0])
    if joint_output_size != vocab_total + len(duration_values):
        raise ValueError(
            f"TDT joint output has {joint_output_size} rows; expected "
            f"{vocab_total} token rows plus {len(duration_values)} duration rows")
    w["_joint_output_size"] = joint_output_size
    w["_duration_values"] = duration_values
    w["_joint_hidden"] = joint_hidden
    w["_joint_activation"] = joint_activation

    config.hidden_size = pred_hidden
    config.vocab_size = vocab_total
    config.num_hidden_layers = pred_layers
    config.num_attention_heads = 1
    config.num_key_value_heads = 1

    frontend = {
        "num_mel_bins": mel_bins,
        "max_source_positions": enc_seq,
        "encoder_layers": enc_layers,
        "mel_length": mel_length,
        "subsampling_factor": 8,
        "sample_rate": 16000,
    }
    runtime = {
        "hidden_size": pred_hidden,
        "num_hidden_layers": pred_layers,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": vocab_total,
        "tdt_encoder_hidden_size": hidden,
        "tdt_pred_hidden_size": pred_hidden,
        "tdt_pred_num_layers": pred_layers,
        "tdt_encoder_layers": enc_layers,
        "tdt_joint_hidden_size": joint_hidden,
        "tdt_vocab_size": vocab,
        "tdt_blank_id": blank_id,
        "tdt_duration_values": list(duration_values),
        "tdt_max_symbols_per_step": _cfg_int(ncfg.get("decoding", {}).get("max_symbols_per_step"),
                                              default=10),
        "tdt_causal_downsampling": causal_downsampling,
        "tdt_att_context_left": att_left,
        "tdt_att_context_right": att_right,
    }
    return w, {**frontend, **runtime}


def compile_engines(model_dir: Path, *, precision: str, verbose: bool):
    config = ModelConfig.from_dir(model_dir)
    weights, runtime = _load_weights(str(model_dir), config, precision=precision)
    mel = _build_mel_filterbank(weights, verbose=verbose)
    if mel is None:
        raise RuntimeError("Parakeet TDT requires a mel filterbank")
    plans = {
        "encoder.plan": _build_encoder(config, weights, precision=precision, verbose=verbose),
        "predictor.plan": _build_predictor(weights, precision=precision, verbose=verbose),
        "joint.plan": _build_joint(weights, precision=precision, verbose=verbose),
        "mel_filterbank": mel,
    }
    runtime.update({
        "tensor_parallel_size": 1, "mel_n_fft": 512, "mel_win_length": 400,
        "mel_hop_length": 160, "mel_chunk_length": 30, "mel_sampling_rate": 16000,
        "mel_preemph": 0.97, "mel_normalize": "per_feature",
    })
    return plans, runtime
