# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parakeet TDT-owned FastConformer encoder graph.

The v3 checkpoint uses 24 Macaron Conformer blocks, 8x depthwise-striding
subsampling, Transformer-XL relative positions, and a 128-bin acoustic input.
Both Hugging Face safetensors and NeMo archive tensors are normalized by the
family plugin before entering these graph helpers.
"""

from __future__ import annotations

import io
import json
import math
import sys
import tarfile
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops


trt = trt_compat.get_trt()


def _to_np(t) -> np.ndarray:
    if hasattr(t, "numpy"):
        return t.detach().cpu().numpy().astype(np.float32)
    return np.asarray(t, dtype=np.float32)


def _load_nemo_archive(path: str):
    import torch
    import yaml

    nemo_path = Path(path)
    if nemo_path.is_dir():
        nemo_files = sorted(nemo_path.glob("*.nemo"))
        if nemo_files:
            nemo_path = nemo_files[0]
        else:
            raise FileNotFoundError(f"No .nemo file found in {path}")
    state_dict = config_dict = None
    with tarfile.open(str(nemo_path), "r") as tar:
        for member in tar.getmembers():
            bn = Path(member.name).name
            if bn == "model_weights.ckpt":
                f = tar.extractfile(member)
                if f:
                    state_dict = torch.load(
                        io.BytesIO(f.read()), map_location="cpu", weights_only=True
                    )
            elif bn == "model_config.yaml":
                f = tar.extractfile(member)
                if f:
                    config_dict = yaml.safe_load(f.read())
    if state_dict is None:
        raise FileNotFoundError(f"model_weights.ckpt not found in {nemo_path}")
    if config_dict is None:
        raise FileNotFoundError(f"model_config.yaml not found in {nemo_path}")
    return state_dict, config_dict


def _extract_tokenizer_from_nemo(nemo_path: str, dest_dir: Path) -> None:
    nemo = Path(nemo_path)
    if nemo.is_dir():
        nemo_files = sorted(nemo.glob("*.nemo"))
        if nemo_files:
            nemo = nemo_files[0]
    with tarfile.open(str(nemo), "r") as tar:
        for member in tar.getmembers():
            bn = Path(member.name).name
            if bn.endswith(".model") and "tokenizer" in bn.lower():
                f = tar.extractfile(member)
                if f:
                    (dest_dir / "tokenizer.model").write_bytes(f.read())
                    break
    # Generate a fast tokenizer.json from the SentencePiece model.
    # Use the tokenizers library directly to avoid HF warnings in stdout.
    tok_model_path = dest_dir / "tokenizer.model"
    tok_json_path = dest_dir / "tokenizer.json"
    if tok_model_path.exists() and not tok_json_path.exists():
        try:
            import sentencepiece as spm

            sp = spm.SentencePieceProcessor()
            sp.Load(str(tok_model_path))
            # Unigram vocab is an ordered list: preserve every SentencePiece id
            # and its learned score, including duplicate piece strings.
            vocab = [
                [sp.IdToPiece(i), float(sp.GetScore(i))]
                for i in range(sp.GetPieceSize())
            ]
            tok_json = {
                "version": "1.0",
                "model": {
                    "type": "Unigram",
                    "unk_id": sp.unk_id(),
                    "vocab": vocab,
                },
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": {"type": "Metaspace", "replacement": "\u2581", "add_prefix_space": True},
            }
            tok_json_path.write_text(json.dumps(tok_json), encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to convert Parakeet SentencePiece tokenizer {tok_model_path}"
            ) from exc

    tok_cfg = dest_dir / "tokenizer_config.json"
    if not tok_cfg.exists():
        tok_cfg.write_text(
            json.dumps(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                },
                indent=2,
            )
        )


def _relative_pe(seq_len: int, d_model: int, max_len: int = 5000) -> np.ndarray:
    """Compute relative PE for Transformer-XL attention.

    Matches NeMo RelPositionalEncoding: builds table with positive and
    negative position encodings, where negative uses sin(-k*d) = -sin(k*d).
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if seq_len > max_len:
        raise ValueError(f"seq_len={seq_len} exceeds relative-position max_len={max_len}")
    pos = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

    pe_pos = np.zeros((max_len, d_model), dtype=np.float32)
    pe_pos[:, 0::2] = np.sin(pos * div)
    pe_pos[:, 1::2] = np.cos(pos * div)
    pe_pos = pe_pos[::-1].copy()

    pe_neg = np.zeros((max_len, d_model), dtype=np.float32)
    pe_neg[:, 0::2] = np.sin(-pos * div)
    pe_neg[:, 1::2] = np.cos(-pos * div)
    pe_neg = pe_neg[1:]

    pe_full = np.concatenate([pe_pos, pe_neg], axis=0)
    start = max_len - seq_len
    end = max_len + seq_len - 1
    return pe_full[start:end]


def _compute_enc_seq_len(mel_length: int) -> int:
    """Encoder time output after 3 CausalConv2D stride-2 stages.

    Time dim uses symmetric padding (left=1, right=1): (t+2-3)//2+1 = (t-1)//2+1
    """
    t = mel_length
    for _ in range(3):
        t = (t + 2 - 3) // 2 + 1
    return t


def _compute_causal_enc_seq_len(mel_length: int) -> int:
    t = mel_length
    for _ in range(3):
        t = t // 2 + 1
    return t


# ---------------------------------------------------------------------------
# Encoder TRT graph helpers
# ---------------------------------------------------------------------------


def _build_subsampling(
    network, mel_input, weights, sub_ch, hidden, num_mel_bins, mel_length, dtype=np.float32
):
    causal_downsampling = bool(weights.get("_causal_downsampling", False))

    def add_subsample_conv(inp, weight, bias, out_channels, *, groups=1):
        if causal_downsampling:
            pad = network.add_padding_nd(inp, pre_padding=(2, 2), post_padding=(1, 1))
            inp = pad.get_output(0)
            padding = (0, 0)
        else:
            padding = (1, 1)
        return graph_ops.add_conv2d(
            network,
            inp,
            weight=weight,
            bias=bias,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=padding,
            groups=groups,
            dtype=dtype,
        )

    # NeMo ConformerEncoder passes audio as [B, T, F] to pre_encode.
    # MaskedConvSequential.forward unsqueezes to [B, 1, T, F].
    # So Conv2d input is [1, 1, mel_length, mel_bins] (time=H, features=W).
    # Our mel_input is [mel_bins, mel_length] = [F, T]. Transpose to [T, F].
    tr_mel = network.add_shuffle(mel_input)
    tr_mel.first_transpose = trt.Permutation([1, 0])  # [F,T] -> [T,F]
    ri = network.add_shuffle(tr_mel.get_output(0))
    ri.reshape_dims = (1, 1, mel_length, num_mel_bins)  # [1, 1, T, F]
    x = ri.get_output(0)
    # Standard Conv2d with symmetric padding + ReLU (NOT SiLU, NOT causal)
    x = add_subsample_conv(x, weights["enc_sub_conv0_w"], weights["enc_sub_conv0_b"], sub_ch)
    x = graph_ops.add_activation(network, x, "relu", dtype=dtype)
    for s in range(2):
        x = add_subsample_conv(
            x, weights[f"enc_sub_dw{s}_w"], weights[f"enc_sub_dw{s}_b"], sub_ch, groups=sub_ch
        )
        x = graph_ops.add_conv2d(
            network,
            x,
            weight=weights[f"enc_sub_pw{s}_w"],
            bias=weights[f"enc_sub_pw{s}_b"],
            out_channels=sub_ch,
            kernel_size=(1, 1),
            dtype=dtype,
        )
        x = graph_ops.add_activation(network, x, "relu", dtype=dtype)
    # After convs: [1, C, T_out, F_out] where T=time, F=features
    time_out = int(weights.get("_enc_seq", _compute_enc_seq_len(mel_length)))
    sub_out_in = int(weights["enc_sub_out_w"].shape[0])
    feat_out = sub_out_in // sub_ch
    # NeMo: x.transpose(1,2).reshape(B,T,-1) on [B,C,T,F]
    # = permute(0,2,1,3) -> [B,T,C,F], reshape -> [T, C*F]
    tr = network.add_shuffle(x)
    tr.first_transpose = trt.Permutation([0, 2, 1, 3])  # [B,C,T,F] -> [B,T,C,F]
    tr.reshape_dims = (time_out, sub_ch * feat_out)  # [T, C*F]
    out = graph_ops.add_matmul_rhs_constant(
        network, tr.get_output(0), sub_ch * feat_out, hidden, weights["enc_sub_out_w"], dtype=dtype
    )
    return graph_ops.add_bias_sum(network, out, hidden, weights["enc_sub_out_b"], dtype=dtype)


def _rel_shift(network, x, H, S, dtype=np.float32):
    zeros = graph_ops.add_constant(
        network, (H, S, 1), np.zeros((H, S, 1), dtype=dtype), dtype=dtype
    )
    padded = network.add_concatenation([zeros, x])
    padded.axis = 2
    rs1 = network.add_shuffle(padded.get_output(0))
    rs1.reshape_dims = (H, 2 * S, S)
    sl1 = network.add_slice(
        rs1.get_output(0), start=(0, 1, 0), shape=(H, 2 * S - 1, S), stride=(1, 1, 1)
    )
    rs2 = network.add_shuffle(sl1.get_output(0))
    rs2.reshape_dims = (H, S, 2 * S - 1)
    sl2 = network.add_slice(rs2.get_output(0), start=(0, 0, 0), shape=(H, S, S), stride=(1, 1, 1))
    return sl2.get_output(0)


def _add_rel_pos_attention(
    network, hs, weights, pfx, hidden, H, D, S, rel_pe_proj, eps, enc_mask=None, dtype=np.float32
):
    normed = graph_ops.add_layer_norm(
        network,
        hs,
        hidden,
        weights[f"{pfx}.norm_sa"],
        weights[f"{pfx}.norm_sa_b"],
        eps,
        dtype=dtype,
    )
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
            network, normed, hidden, hidden, weights[f"{pfx}.w_k"], dtype=dtype
        ),
        hidden,
        weights[f"{pfx}.b_k"],
        dtype=dtype,
    )
    v = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, hidden, weights[f"{pfx}.w_v"], dtype=dtype
        ),
        hidden,
        weights[f"{pfx}.b_v"],
        dtype=dtype,
    )
    qr = network.add_shuffle(q)
    qr.reshape_dims = (S, H, D)
    kr = network.add_shuffle(k)
    kr.reshape_dims = (S, H, D)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (S, H, D)
    bu = graph_ops.add_constant(network, (1, H, D), weights[f"{pfx}.pos_bias_u"], dtype=dtype)
    bv = graph_ops.add_constant(network, (1, H, D), weights[f"{pfx}.pos_bias_v"], dtype=dtype)
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
    rp_t = network.add_shuffle(rel_pe_proj)
    rp_t.first_transpose = trt.Permutation([1, 0, 2])
    ps_raw = network.add_matrix_multiply(
        qv_t.get_output(0),
        trt.MatrixOperation.NONE,
        rp_t.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    ps = _rel_shift(network, ps_raw, H, S, dtype=dtype)
    total = network.add_elementwise(cs, ps, trt.ElementWiseOperation.SUM).get_output(0)
    sc = graph_ops.add_constant(
        network, (1, 1, 1), np.array([1.0 / math.sqrt(D)], dtype=dtype), dtype=dtype
    )
    scaled = network.add_elementwise(total, sc, trt.ElementWiseOperation.PROD).get_output(0)
    # Apply encoder sequence mask: [1, 1, S] added to scores [H, S, S]
    if enc_mask is not None:
        scaled = network.add_elementwise(scaled, enc_mask, trt.ElementWiseOperation.SUM).get_output(
            0
        )
    # Conformer relative-position attention uses a rel-shifted Q*R term in
    # the logits, which native IAttention cannot represent as a plain mask.
    sm = network.add_softmax(scaled)
    sm.axes = 1 << 2
    ao = network.add_matrix_multiply(
        sm.get_output(0), trt.MatrixOperation.NONE, v_t.get_output(0), trt.MatrixOperation.NONE
    ).get_output(0)
    at = network.add_shuffle(ao)
    at.first_transpose = trt.Permutation([1, 0, 2])
    af = network.add_shuffle(at.get_output(0))
    af.reshape_dims = (S, hidden)
    return graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, af.get_output(0), hidden, hidden, weights[f"{pfx}.w_o"], dtype=dtype
        ),
        hidden,
        weights[f"{pfx}.b_o"],
        dtype=dtype,
    )


def _add_causal_depthwise_conv1d(network, x, weights, pfx, hidden, kern, dtype=np.float32):
    pad = kern - 1
    if pad > 0:
        zeros = graph_ops.add_constant(
            network, (1, hidden, pad), np.zeros((1, hidden, pad), dtype=dtype), dtype=dtype
        )
        cat = network.add_concatenation([zeros, x])
        cat.axis = 2
        x = cat.get_output(0)
    return graph_ops.add_conv1d(
        network,
        x,
        weight=weights[f"{pfx}.cdw_w"],
        bias=weights[f"{pfx}.cdw_b"],
        out_channels=hidden,
        kernel_size=kern,
        groups=hidden,
        dtype=dtype,
    )


def _add_conv_norm(network, x, weights, pfx, hidden, S, eps, conv_norm_type, dtype=np.float32):
    if conv_norm_type == "layer_norm":
        r1 = network.add_shuffle(x)
        r1.reshape_dims = (hidden, S)
        r2 = network.add_shuffle(r1.get_output(0))
        r2.first_transpose = trt.Permutation([1, 0])
        normed = graph_ops.add_layer_norm(
            network,
            r2.get_output(0),
            hidden,
            weights[f"{pfx}.bn_w"],
            weights[f"{pfx}.bn_b"],
            eps,
            dtype=dtype,
        )
        r3 = network.add_shuffle(normed)
        r3.first_transpose = trt.Permutation([1, 0])
        r4 = network.add_shuffle(r3.get_output(0))
        r4.reshape_dims = (1, hidden, S)
        return r4.get_output(0)

    bn = network.add_shuffle(x)
    bn.reshape_dims = (1, hidden, 1, S)
    x = graph_ops.add_batch_norm_2d(
        network,
        bn.get_output(0),
        hidden,
        gamma=weights[f"{pfx}.bn_w"],
        beta=weights[f"{pfx}.bn_b"],
        running_mean=weights[f"{pfx}.bn_m"],
        running_var=weights[f"{pfx}.bn_v"],
        dtype=dtype,
    )
    bo = network.add_shuffle(x)
    bo.reshape_dims = (1, hidden, S)
    return bo.get_output(0)


def _add_conv_module(
    network,
    hs,
    weights,
    pfx,
    hidden,
    kern,
    S,
    eps,
    conv_norm_type="batch_norm",
    conv_context_size="symmetric",
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
    r2.reshape_dims = (1, hidden, S)
    x = graph_ops.add_conv1d(
        network,
        r2.get_output(0),
        weight=weights[f"{pfx}.cpw1_w"],
        bias=weights[f"{pfx}.cpw1_b"],
        out_channels=2 * hidden,
        kernel_size=1,
        dtype=dtype,
    )
    xa = network.add_slice(x, start=(0, 0, 0), shape=(1, hidden, S), stride=(1, 1, 1)).get_output(0)
    xb = network.add_slice(
        x, start=(0, hidden, 0), shape=(1, hidden, S), stride=(1, 1, 1)
    ).get_output(0)
    gate = network.add_activation(xb, trt.ActivationType.SIGMOID).get_output(0)
    x = network.add_elementwise(xa, gate, trt.ElementWiseOperation.PROD).get_output(0)
    if conv_context_size == "causal":
        x = _add_causal_depthwise_conv1d(network, x, weights, pfx, hidden, kern, dtype=dtype)
    else:
        x = graph_ops.add_conv1d(
            network,
            x,
            weight=weights[f"{pfx}.cdw_w"],
            bias=weights[f"{pfx}.cdw_b"],
            out_channels=hidden,
            kernel_size=kern,
            padding=kern // 2,
            groups=hidden,
            dtype=dtype,
        )
    x = _add_conv_norm(network, x, weights, pfx, hidden, S, eps, conv_norm_type, dtype=dtype)
    x = graph_ops.add_activation(network, x, "silu", dtype=dtype)
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
    r3.reshape_dims = (hidden, S)
    r4 = network.add_shuffle(r3.get_output(0))
    r4.first_transpose = trt.Permutation([1, 0])
    return r4.get_output(0)


def _add_half_ffn(network, hs, weights, pfx, hidden, ffn, eps, dtype=np.float32):
    normed = graph_ops.add_layer_norm(
        network, hs, hidden, weights[f"{pfx}.norm"], weights[f"{pfx}.norm_b"], eps, dtype=dtype
    )
    fc1 = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, normed, hidden, ffn, weights[f"{pfx}.w1"], dtype=dtype
        ),
        ffn,
        weights[f"{pfx}.b1"],
        dtype=dtype,
    )
    act = graph_ops.add_activation(network, fc1, "silu", dtype=dtype)
    fc2 = graph_ops.add_bias_sum(
        network,
        graph_ops.add_matmul_rhs_constant(
            network, act, ffn, hidden, weights[f"{pfx}.w2"], dtype=dtype
        ),
        hidden,
        weights[f"{pfx}.b2"],
        dtype=dtype,
    )
    half = graph_ops.add_constant(network, (1, 1), np.array([0.5], dtype=dtype), dtype=dtype)
    return network.add_elementwise(fc2, half, trt.ElementWiseOperation.PROD).get_output(0)


def _add_conformer_block(
    network,
    hs,
    weights,
    pfx,
    hidden,
    H,
    D,
    ffn,
    kern,
    S,
    rpe,
    eps,
    enc_mask=None,
    conv_norm_type="batch_norm",
    conv_context_size="symmetric",
    dtype=np.float32,
):
    ffn1 = _add_half_ffn(network, hs, weights, f"{pfx}.ff1", hidden, ffn, eps, dtype=dtype)
    hs = network.add_elementwise(hs, ffn1, trt.ElementWiseOperation.SUM).get_output(0)
    attn = _add_rel_pos_attention(
        network, hs, weights, pfx, hidden, H, D, S, rpe, eps, enc_mask, dtype=dtype
    )
    hs = network.add_elementwise(hs, attn, trt.ElementWiseOperation.SUM).get_output(0)
    conv = _add_conv_module(
        network,
        hs,
        weights,
        pfx,
        hidden,
        kern,
        S,
        eps,
        conv_norm_type=conv_norm_type,
        conv_context_size=conv_context_size,
        dtype=dtype,
    )
    hs = network.add_elementwise(hs, conv, trt.ElementWiseOperation.SUM).get_output(0)
    ffn2 = _add_half_ffn(network, hs, weights, f"{pfx}.ff2", hidden, ffn, eps, dtype=dtype)
    hs = network.add_elementwise(hs, ffn2, trt.ElementWiseOperation.SUM).get_output(0)
    return graph_ops.add_layer_norm(
        network,
        hs,
        hidden,
        weights[f"{pfx}.norm_out"],
        weights[f"{pfx}.norm_out_b"],
        eps,
        dtype=dtype,
    )


def _build_encoder(config, weights, *, precision="fp32", verbose=False):
    el = weights["_enc_layers"]
    eh = weights["_enc_heads"]
    h = weights["_hidden"]
    hd = weights["_head_dim"]
    ef = weights["_enc_ffn"]
    k = weights["_kern"]
    mb = weights["_mel_bins"]
    ml = weights["_mel_length"]
    es = weights["_enc_seq"]
    sc = weights["_sub_ch"]
    conv_norm_type = str(weights.get("_conv_norm_type", "batch_norm")).lower()
    conv_context_size = str(weights.get("_conv_context_size", "symmetric")).lower()
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(
            f"Unsupported streaming encoder precision {precision!r}; expected fp32 or fp16"
        )

    log = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    b = trt.Builder(log)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tc = b.create_builder_config()
    tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    eps = graph_ops.add_constant(
        net, (1, 1), np.array([1e-5], dtype=work_np_dtype), dtype=work_np_dtype
    )
    mel = net.add_input("mel_features", trt.float32, (mb, ml))
    # Encoder attention mask: [1, 1, enc_seq] -- 0.0 valid, -10000.0 padded.
    # Applied additively to self-attention scores before softmax.
    mask_shape = (
        (1, es, es) if bool(weights.get("_encoder_attention_mask_2d", False)) else (1, 1, es)
    )
    enc_mask = net.add_input("encoder_mask", trt.float32, mask_shape)
    if work_trt_dtype != trt.float32:
        mel = net.add_cast(mel, work_trt_dtype).get_output(0)
        enc_mask = net.add_cast(enc_mask, work_trt_dtype).get_output(0)

    hs = _build_subsampling(net, mel, weights, sc, h, mb, ml, dtype=work_np_dtype)
    for li in range(el):
        pfx = f"el.{li}"
        rpe = graph_ops.add_constant(
            net, (2 * es - 1, eh, hd), weights[f"{pfx}.rpe_proj"], dtype=work_np_dtype
        )
        hs = _add_conformer_block(
            net,
            hs,
            weights,
            pfx,
            h,
            eh,
            hd,
            ef,
            k,
            es,
            rpe,
            eps,
            enc_mask,
            conv_norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            dtype=work_np_dtype,
        )

    output = hs
    if output.dtype != trt.float32:
        output = net.add_cast(output, trt.float32).get_output(0)
    output.name = "encoder_output"
    net.mark_output(output)
    if verbose:
        print(
            f"[trtmc build] Building Parakeet TDT encoder ({el}L, h={h}, heads={eh}, seq={es})",
            file=sys.stderr,
        )
    plan = b.build_serialized_network(net, tc)
    if plan is None:
        raise RuntimeError("Parakeet TDT encoder build failed")
    return bytes(plan)
