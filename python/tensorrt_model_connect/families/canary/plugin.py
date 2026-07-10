# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canary family plugin -- FastConformer encoder-decoder ASR (speech-to-text).

nvidia/canary-1b-v2: 978M-param NeMo encoder-decoder ASR model.
  - Encoder: FastConformer (32 layers) with DW-striding subsampling (8x, 256ch),
    Macaron-style conformer blocks, Transformer-XL relative positional encoding
  - Decoder: Pre-LN Transformer (8 layers) with self-attention (KV cache),
    cross-attention to encoder output, ReLU MLP, sinusoidal pos embeddings
  - Format: NeMo .nemo TAR archive (model_weights.ckpt + model_config.yaml)
  - model_type: "canary" / "canary_asr" / "enc_dec_multi_task"

NeMo weight key conventions (note underscore prefixes):
  Encoder: encoder.layers.{i}.self_attn.linear_q.weight/bias
  Decoder: transf_decoder._decoder.layers.{i}.first_sub_layer.query_net.weight/bias
  Embedding: transf_decoder._embedding.token_embedding.weight
  Position: transf_decoder._embedding.position_embedding.pos_enc
  Head: log_softmax.mlp.layer0.weight/bias

Cross-attention: same as Whisper -- cross_k/cross_v are raw encoder output;
per-layer K/V projections baked into the decoder TRT graph.
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

from .config import ModelConfig
from .checkpoint_mapper import WeightDict, _transpose_2d
from . import graph_ops
from . import graph_blocks
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .decoder_tp_builder import build_canary_tp_decoder_engine


trt = trt_compat.get_trt()

_CANARY_V2_LANGUAGES = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
]


def _cfg_int(*values, default: int) -> int:
    for value in values:
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return default


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
                        io.BytesIO(f.read()), map_location="cpu",
                        weights_only=False)
            elif bn == "model_config.yaml":
                f = tar.extractfile(member)
                if f:
                    config_dict = yaml.safe_load(f.read())
    if state_dict is None:
        raise FileNotFoundError(f"model_weights.ckpt not found in {nemo_path}")
    if config_dict is None:
        raise FileNotFoundError(f"model_config.yaml not found in {nemo_path}")
    return state_dict, config_dict


def _extract_tokenizer_from_nemo(
    nemo_path: str, dest_dir: Path, nemo_cfg: dict | None = None,
) -> Path:
    nemo = Path(nemo_path)
    if nemo.is_dir():
        nemo_files = sorted(nemo.glob("*.nemo"))
        if nemo_files:
            nemo = nemo_files[0]
    tokenizer_cfg = (nemo_cfg or {}).get("tokenizer", {})
    configured_name = str(tokenizer_cfg.get("model_path", "")).removeprefix("nemo:")
    with tarfile.open(str(nemo), "r") as tar:
        candidates = [
            member for member in tar.getmembers()
            if Path(member.name).name.endswith(".model")
            and "tokenizer" in Path(member.name).name.lower()
        ]
        selected = next(
            (member for member in candidates
             if Path(member.name).name == Path(configured_name).name),
            candidates[0] if candidates else None,
        )
        if selected is not None:
            f = tar.extractfile(selected)
            if f:
                (dest_dir / "tokenizer.model").write_bytes(f.read())
    # Generate a fast tokenizer.json from the SentencePiece model.
    # Use the tokenizers library directly to avoid HF warnings in stdout.
    tok_model_path = dest_dir / "tokenizer.model"
    tok_json_path = dest_dir / "tokenizer.json"
    if tok_model_path.exists() and not tok_json_path.exists():
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(str(tok_model_path))
            # Build a minimal tokenizer.json compatible with HF fast tokenizer
            vocab = {sp.IdToPiece(i): i for i in range(sp.GetPieceSize())}
            tok_json = {
                "version": "1.0",
                "model": {
                    "type": "Unigram",
                    "unk_id": sp.unk_id(),
                    "vocab": [[piece, 0.0] for piece in vocab],
                },
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": {"type": "Metaspace", "replacement": "\u2581",
                            "add_prefix_space": True},
            }
            tok_json_path.write_text(json.dumps(tok_json))
        except Exception as e:
            raise RuntimeError(
                f"Canary tokenizer.json generation failed for {tok_model_path}: {e}. "
                "Install sentencepiece or provide tokenizer.json."
            ) from e

    tok_cfg = dest_dir / "tokenizer_config.json"
    if not tok_cfg.exists():
        tok_cfg.write_text(json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast",
        }, indent=2))
    return tok_model_path


def _canary_prompt_metadata(nemo_cfg: dict, tokenizer_model: Path) -> dict:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))

    def token_id(piece: str) -> int:
        value = int(processor.piece_to_id(piece))
        if value < 0 or processor.id_to_piece(value) != piece:
            raise ValueError(
                f"Canary tokenizer {tokenizer_model} is missing required token {piece!r}")
        return value

    def has_token(piece: str) -> bool:
        value = int(processor.piece_to_id(piece))
        return value >= 0 and processor.id_to_piece(value) == piece

    defaults = {}
    for prompt in nemo_cfg.get("prompt_defaults", []):
        if prompt.get("role") == "user":
            defaults = dict(prompt.get("slots", {}))
            break
    source = str(defaults.get("source_lang", "<|en|>"))
    target = str(defaults.get("target_lang", source))
    prompt_pieces = [
        "▁", "<|startofcontext|>", "<|startoftranscript|>",
        str(defaults.get("emotion", "<|emo:undefined|>")), source, target,
        str(defaults.get("pnc", "<|pnc|>")),
        str(defaults.get("itn", "<|noitn|>")),
        str(defaults.get("timestamp", "<|notimestamp|>")),
        str(defaults.get("diarize", "<|nodiarize|>")),
    ]

    configured_languages = nemo_cfg.get("supported_languages")
    if configured_languages:
        languages = [str(language) for language in configured_languages]
    else:
        languages = [
            language for language in _CANARY_V2_LANGUAGES
            if has_token(f"<|{language}|>")
        ]
    return {
        "decoder_start_token_ids": [token_id(piece) for piece in prompt_pieces],
        "eot_token_id": token_id("<|endoftext|>"),
        "canary_supported_languages": languages,
        "canary_language_token_ids": [token_id(f"<|{language}|>") for language in languages],
        "canary_source_language_position": 4,
        "canary_target_language_position": 5,
        "canary_punctuation_position": 6,
        "canary_timestamp_position": 8,
        "canary_punctuation_token_id": token_id("<|pnc|>"),
        "canary_no_punctuation_token_id": token_id("<|nopnc|>"),
        "canary_timestamp_token_id": token_id("<|timestamp|>"),
        "canary_no_timestamp_token_id": token_id("<|notimestamp|>"),
        "canary_translation_requires_english": True,
    }


def _sinusoidal_pe(max_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pos = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) *
                 -(math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def _relative_pe(seq_len: int, d_model: int, max_len: int = 5000) -> np.ndarray:
    """Compute relative PE for Transformer-XL attention.

    Matches NeMo RelPositionalEncoding: builds table with positive and
    negative position encodings, where negative uses sin(-k*d) = -sin(k*d).
    """
    pos = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) *
                 -(math.log(10000.0) / d_model))

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

def _build_subsampling(network, mel_input, weights, sub_ch, hidden,
                       num_mel_bins, mel_length, dtype=np.float32):
    causal_downsampling = bool(weights.get("_causal_downsampling", False))

    def add_subsample_conv(inp, weight, bias, out_channels, *, groups=1):
        if causal_downsampling:
            pad = network.add_padding_nd(inp, pre_padding=(2, 2), post_padding=(1, 1))
            inp = pad.get_output(0)
            padding = (0, 0)
        else:
            padding = (1, 1)
        return graph_ops.add_conv2d(
            network, inp, weight=weight, bias=bias, out_channels=out_channels,
            kernel_size=(3, 3), stride=(2, 2), padding=padding, groups=groups,
            dtype=dtype)

    # NeMo ConformerEncoder passes audio as [B, T, F] to pre_encode.
    # MaskedConvSequential.forward unsqueezes to [B, 1, T, F].
    # So Conv2d input is [1, 1, mel_length, mel_bins] (time=H, features=W).
    # Our mel_input is [mel_bins, mel_length] = [F, T]. Transpose to [T, F].
    tr_mel = network.add_shuffle(mel_input)
    tr_mel.first_transpose = trt.Permutation([1, 0])  # [F,T] → [T,F]
    ri = network.add_shuffle(tr_mel.get_output(0))
    ri.reshape_dims = (1, 1, mel_length, num_mel_bins)  # [1, 1, T, F]
    x = ri.get_output(0)
    # Standard Conv2d with symmetric padding + ReLU (NOT SiLU, NOT causal)
    x = add_subsample_conv(
        x, weights["enc_sub_conv0_w"], weights["enc_sub_conv0_b"], sub_ch)
    x = graph_ops.add_activation(network, x, "relu")
    for s in range(2):
        x = add_subsample_conv(
            x, weights[f"enc_sub_dw{s}_w"], weights[f"enc_sub_dw{s}_b"], sub_ch,
            groups=sub_ch)
        x = graph_ops.add_conv2d(network, x,
            weight=weights[f"enc_sub_pw{s}_w"], bias=weights[f"enc_sub_pw{s}_b"],
            out_channels=sub_ch, kernel_size=(1, 1), dtype=dtype)
        x = graph_ops.add_activation(network, x, "relu", dtype=dtype)
    # After convs: [1, C, T_out, F_out] where T=time, F=features
    time_out = int(weights.get("_enc_seq", _compute_enc_seq_len(mel_length)))
    sub_out_in = int(weights["enc_sub_out_w"].shape[0])
    feat_out = sub_out_in // sub_ch
    # NeMo: x.transpose(1,2).reshape(B,T,-1) on [B,C,T,F]
    # = permute(0,2,1,3) → [B,T,C,F], reshape → [T, C*F]
    tr = network.add_shuffle(x)
    tr.first_transpose = trt.Permutation([0, 2, 1, 3])  # [B,C,T,F] → [B,T,C,F]
    tr.reshape_dims = (time_out, sub_ch * feat_out)  # [T, C*F]
    out = graph_ops.add_matmul_rhs_constant(
        network, tr.get_output(0), sub_ch * feat_out, hidden,
        weights["enc_sub_out_w"], dtype=dtype)
    return graph_ops.add_bias_sum(
        network, out, hidden, weights["enc_sub_out_b"], dtype=dtype)


def _rel_shift(network, x, H, S, dtype=np.float32):
    zeros = graph_ops.add_constant(network, (H, S, 1),
        np.zeros((H, S, 1), dtype=dtype), dtype=dtype)
    padded = network.add_concatenation([zeros, x])
    padded.axis = 2
    rs1 = network.add_shuffle(padded.get_output(0))
    rs1.reshape_dims = (H, 2 * S, S)
    sl1 = network.add_slice(rs1.get_output(0),
        start=(0, 1, 0), shape=(H, 2*S-1, S), stride=(1, 1, 1))
    rs2 = network.add_shuffle(sl1.get_output(0))
    rs2.reshape_dims = (H, S, 2*S-1)
    sl2 = network.add_slice(rs2.get_output(0),
        start=(0, 0, 0), shape=(H, S, S), stride=(1, 1, 1))
    return sl2.get_output(0)


def _add_rel_pos_attention(network, hs, weights, pfx, hidden, H, D, S,
                           rel_pe_proj, eps, enc_mask=None, dtype=np.float32):
    normed = graph_ops.add_layer_norm(network, hs, hidden,
        weights[f"{pfx}.norm_sa"], weights[f"{pfx}.norm_sa_b"], eps,
        dtype=dtype)
    q = graph_ops.add_bias_sum(network,
        graph_ops.add_matmul_rhs_constant(network, normed, hidden, hidden,
            weights[f"{pfx}.w_q"], dtype=dtype), hidden,
        weights[f"{pfx}.b_q"], dtype=dtype)
    k = graph_ops.add_bias_sum(network,
        graph_ops.add_matmul_rhs_constant(network, normed, hidden, hidden,
            weights[f"{pfx}.w_k"], dtype=dtype), hidden,
        weights[f"{pfx}.b_k"], dtype=dtype)
    v = graph_ops.add_bias_sum(network,
        graph_ops.add_matmul_rhs_constant(network, normed, hidden, hidden,
            weights[f"{pfx}.w_v"], dtype=dtype), hidden,
        weights[f"{pfx}.b_v"], dtype=dtype)
    qr = network.add_shuffle(q)
    qr.reshape_dims = (S, H, D)
    kr = network.add_shuffle(k)
    kr.reshape_dims = (S, H, D)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (S, H, D)
    bu = graph_ops.add_constant(
        network, (1, H, D), weights[f"{pfx}.pos_bias_u"], dtype=dtype)
    bv = graph_ops.add_constant(
        network, (1, H, D), weights[f"{pfx}.pos_bias_v"], dtype=dtype)
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
    cs = network.add_matrix_multiply(qu_t.get_output(0), trt.MatrixOperation.NONE,
        k_t.get_output(0), trt.MatrixOperation.TRANSPOSE).get_output(0)
    rp_t = network.add_shuffle(rel_pe_proj)
    rp_t.first_transpose = trt.Permutation([1, 0, 2])
    ps_raw = network.add_matrix_multiply(qv_t.get_output(0), trt.MatrixOperation.NONE,
        rp_t.get_output(0), trt.MatrixOperation.TRANSPOSE).get_output(0)
    ps = _rel_shift(network, ps_raw, H, S, dtype=dtype)
    total = network.add_elementwise(cs, ps, trt.ElementWiseOperation.SUM).get_output(0)
    sc = graph_ops.add_constant(
        network, (1, 1, 1), np.array([1.0/math.sqrt(D)], dtype=dtype),
        dtype=dtype)
    scaled = network.add_elementwise(total, sc, trt.ElementWiseOperation.PROD).get_output(0)
    # Apply encoder sequence mask: [1, 1, S] added to scores [H, S, S]
    if enc_mask is not None:
        scaled = network.add_elementwise(
            scaled, enc_mask, trt.ElementWiseOperation.SUM).get_output(0)
    # Conformer relative-position attention uses a rel-shifted Q*R term in
    # the logits, which native IAttention cannot represent as a plain mask.
    sm = network.add_softmax(scaled)
    sm.axes = 1 << 2
    ao = network.add_matrix_multiply(sm.get_output(0), trt.MatrixOperation.NONE,
        v_t.get_output(0), trt.MatrixOperation.NONE).get_output(0)
    at = network.add_shuffle(ao)
    at.first_transpose = trt.Permutation([1, 0, 2])
    af = network.add_shuffle(at.get_output(0))
    af.reshape_dims = (S, hidden)
    return graph_ops.add_bias_sum(network,
        graph_ops.add_matmul_rhs_constant(network, af.get_output(0), hidden, hidden,
            weights[f"{pfx}.w_o"], dtype=dtype), hidden,
        weights[f"{pfx}.b_o"], dtype=dtype)


def _add_causal_depthwise_conv1d(
        network, x, weights, pfx, hidden, kern, dtype=np.float32):
    pad = kern - 1
    if pad > 0:
        zeros = graph_ops.add_constant(
            network, (1, hidden, pad), np.zeros((1, hidden, pad), dtype=dtype),
            dtype=dtype)
        cat = network.add_concatenation([zeros, x])
        cat.axis = 2
        x = cat.get_output(0)
    return graph_ops.add_conv1d(
        network, x,
        weight=weights[f"{pfx}.cdw_w"], bias=weights[f"{pfx}.cdw_b"],
        out_channels=hidden, kernel_size=kern, groups=hidden, dtype=dtype)


def _add_conv_norm(
        network, x, weights, pfx, hidden, S, eps, conv_norm_type,
        dtype=np.float32):
    if conv_norm_type == "layer_norm":
        r1 = network.add_shuffle(x)
        r1.reshape_dims = (hidden, S)
        r2 = network.add_shuffle(r1.get_output(0))
        r2.first_transpose = trt.Permutation([1, 0])
        normed = graph_ops.add_layer_norm(
            network, r2.get_output(0), hidden, weights[f"{pfx}.bn_w"],
            weights[f"{pfx}.bn_b"], eps, dtype=dtype)
        r3 = network.add_shuffle(normed)
        r3.first_transpose = trt.Permutation([1, 0])
        r4 = network.add_shuffle(r3.get_output(0))
        r4.reshape_dims = (1, hidden, S)
        return r4.get_output(0)

    bn = network.add_shuffle(x)
    bn.reshape_dims = (1, hidden, 1, S)
    x = graph_ops.add_batch_norm_2d(
        network, bn.get_output(0), hidden,
        gamma=weights[f"{pfx}.bn_w"], beta=weights[f"{pfx}.bn_b"],
        running_mean=weights[f"{pfx}.bn_m"], running_var=weights[f"{pfx}.bn_v"],
        dtype=dtype)
    bo = network.add_shuffle(x)
    bo.reshape_dims = (1, hidden, S)
    return bo.get_output(0)


def _add_conv_module(network, hs, weights, pfx, hidden, kern, S, eps,
                     conv_norm_type="batch_norm", conv_context_size="symmetric",
                     valid_mask=None, dtype=np.float32):
    normed = graph_ops.add_layer_norm(network, hs, hidden,
        weights[f"{pfx}.norm_conv"], weights[f"{pfx}.norm_conv_b"], eps,
        dtype=dtype)
    r1 = network.add_shuffle(normed)
    r1.first_transpose = trt.Permutation([1, 0])
    r2 = network.add_shuffle(r1.get_output(0))
    r2.reshape_dims = (1, hidden, S)
    conv_in = r2.get_output(0)
    # Zero out padded time positions before the depthwise conv, matching NeMo's
    # ConformerConvolution masked_fill(pad_mask, 0). Without this, the depthwise
    # conv (kernel>1) leaks padded-position activations into valid positions,
    # compounding across layers and corrupting the encoder output for audio
    # shorter than the build-time mel_length. valid_mask is [1, 1, S] with 1.0
    # for valid and 0.0 for padded positions; it broadcasts over channels.
    if valid_mask is not None:
        conv_in = network.add_elementwise(
            conv_in, valid_mask, trt.ElementWiseOperation.PROD).get_output(0)
    x = graph_ops.add_conv1d(network, conv_in,
        weight=weights[f"{pfx}.cpw1_w"], bias=weights[f"{pfx}.cpw1_b"],
        out_channels=2*hidden, kernel_size=1, dtype=dtype)
    xa = network.add_slice(x, start=(0,0,0), shape=(1,hidden,S), stride=(1,1,1)).get_output(0)
    xb = network.add_slice(x, start=(0,hidden,0), shape=(1,hidden,S), stride=(1,1,1)).get_output(0)
    gate = network.add_activation(xb, trt.ActivationType.SIGMOID).get_output(0)
    x = network.add_elementwise(xa, gate, trt.ElementWiseOperation.PROD).get_output(0)
    # Second mask, matching NeMo's ConformerConvolution: pointwise_conv1 has a
    # bias, so padded positions are non-zero again after GLU. Re-zero them
    # before the depthwise conv, otherwise that bias leaks into valid positions.
    if valid_mask is not None:
        x = network.add_elementwise(
            x, valid_mask, trt.ElementWiseOperation.PROD).get_output(0)
    if conv_context_size == "causal":
        x = _add_causal_depthwise_conv1d(
            network, x, weights, pfx, hidden, kern, dtype=dtype)
    else:
        x = graph_ops.add_conv1d(network, x,
            weight=weights[f"{pfx}.cdw_w"], bias=weights[f"{pfx}.cdw_b"],
            out_channels=hidden, kernel_size=kern, padding=kern//2,
            groups=hidden, dtype=dtype)
    x = _add_conv_norm(
        network, x, weights, pfx, hidden, S, eps, conv_norm_type,
        dtype=dtype)
    x = graph_ops.add_activation(network, x, "silu", dtype=dtype)
    x = graph_ops.add_conv1d(network, x,
        weight=weights[f"{pfx}.cpw2_w"], bias=weights[f"{pfx}.cpw2_b"],
        out_channels=hidden, kernel_size=1, dtype=dtype)
    r3 = network.add_shuffle(x)
    r3.reshape_dims = (hidden, S)
    r4 = network.add_shuffle(r3.get_output(0))
    r4.first_transpose = trt.Permutation([1, 0])
    return r4.get_output(0)


def _add_half_ffn(
        network, hs, weights, pfx, hidden, ffn, eps, dtype=np.float32):
    normed = graph_ops.add_layer_norm(network, hs, hidden,
        weights[f"{pfx}.norm"], weights[f"{pfx}.norm_b"], eps, dtype=dtype)
    fc1 = graph_ops.add_bias_sum(network,
        graph_ops.add_matmul_rhs_constant(network, normed, hidden, ffn,
            weights[f"{pfx}.w1"], dtype=dtype), ffn,
        weights[f"{pfx}.b1"], dtype=dtype)
    act = graph_ops.add_activation(network, fc1, "silu", dtype=dtype)
    fc2 = graph_ops.add_bias_sum(network,
        graph_ops.add_matmul_rhs_constant(network, act, ffn, hidden,
            weights[f"{pfx}.w2"], dtype=dtype), hidden,
        weights[f"{pfx}.b2"], dtype=dtype)
    half = graph_ops.add_constant(
        network, (1, 1), np.array([0.5], dtype=dtype), dtype=dtype)
    return network.add_elementwise(fc2, half, trt.ElementWiseOperation.PROD).get_output(0)


def _add_conformer_block(network, hs, weights, pfx, hidden, H, D, ffn,
                         kern, S, rpe, eps, enc_mask=None,
                         conv_norm_type="batch_norm", conv_context_size="symmetric",
                         valid_mask=None, dtype=np.float32):
    ffn1 = _add_half_ffn(
        network, hs, weights, f"{pfx}.ff1", hidden, ffn, eps, dtype=dtype)
    hs = network.add_elementwise(hs, ffn1, trt.ElementWiseOperation.SUM).get_output(0)
    attn = _add_rel_pos_attention(
        network, hs, weights, pfx, hidden, H, D, S, rpe, eps, enc_mask,
        dtype=dtype)
    hs = network.add_elementwise(hs, attn, trt.ElementWiseOperation.SUM).get_output(0)
    conv = _add_conv_module(
        network, hs, weights, pfx, hidden, kern, S, eps,
        conv_norm_type=conv_norm_type, conv_context_size=conv_context_size,
        valid_mask=valid_mask, dtype=dtype)
    hs = network.add_elementwise(hs, conv, trt.ElementWiseOperation.SUM).get_output(0)
    ffn2 = _add_half_ffn(
        network, hs, weights, f"{pfx}.ff2", hidden, ffn, eps, dtype=dtype)
    hs = network.add_elementwise(hs, ffn2, trt.ElementWiseOperation.SUM).get_output(0)
    return graph_ops.add_layer_norm(network, hs, hidden,
        weights[f"{pfx}.norm_out"], weights[f"{pfx}.norm_out_b"], eps,
        dtype=dtype)


# ---------------------------------------------------------------------------
# Decoder TRT graph (follows Whisper pattern)
# ---------------------------------------------------------------------------

def _add_decoder_layer(*, network, hidden, cache_k, cache_v, cross_k,
                       cross_v, attention_mask, eps, weights,
                       pfx, hsz, nheads, hdim, ffn, maxcache, maxsrc,
                       dtype=np.float32):
    aw = maxcache + 1
    # Self-attention
    n = graph_ops.add_layer_norm(network, hidden, hsz,
        weights[f"{pfx}.input_norm"], weights[f"{pfx}.input_norm_b"], eps,
        dtype=dtype)
    q = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(
        network, n, hsz, hsz, weights[f"{pfx}.w_q"], dtype=dtype), hsz,
        weights[f"{pfx}.q_bias"], dtype=dtype)
    k = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(
        network, n, hsz, hsz, weights[f"{pfx}.w_k"], dtype=dtype), hsz,
        weights[f"{pfx}.k_bias"], dtype=dtype)
    v = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(
        network, n, hsz, hsz, weights[f"{pfx}.w_v"], dtype=dtype), hsz,
        weights[f"{pfx}.v_bias"], dtype=dtype)
    pk, pv = k, v
    kr = network.add_shuffle(k)
    kr.reshape_dims = (1, hsz)
    vr = network.add_shuffle(v)
    vr.reshape_dims = (1, hsz)
    ak = network.add_concatenation([cache_k, kr.get_output(0)])
    ak.axis = 0
    av = network.add_concatenation([cache_v, vr.get_output(0)])
    av.axis = 0
    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    cf = graph_ops.add_attention_from_rows(
        network, q, ak.get_output(0), av.get_output(0),
        num_heads=nheads, head_dim=hdim,
        q_seq=1, kv_seq=aw,
        mask=mask_4d)
    sa = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(
        network, cf, hsz, hsz, weights[f"{pfx}.w_o"], dtype=dtype), hsz,
        weights[f"{pfx}.o_bias"], dtype=dtype)
    psa = network.add_elementwise(hidden, sa, trt.ElementWiseOperation.SUM).get_output(0)
    # Cross-attention
    cn = graph_ops.add_layer_norm(network, psa, hsz,
        weights[f"{pfx}.xattn_norm"], weights[f"{pfx}.xattn_norm_b"], eps,
        dtype=dtype)
    cq = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(
        network, cn, hsz, hsz, weights[f"{pfx}.xw_q"], dtype=dtype), hsz,
        weights[f"{pfx}.xb_q"], dtype=dtype)
    ck = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(
        network, cross_k, hsz, hsz, weights[f"{pfx}.xw_k"], dtype=dtype), hsz,
        weights[f"{pfx}.xb_k"], dtype=dtype)
    cv = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(
        network, cross_v, hsz, hsz, weights[f"{pfx}.xw_v"], dtype=dtype), hsz,
        weights[f"{pfx}.xb_v"], dtype=dtype)
    ccf = graph_ops.add_attention_from_rows(
        network, cq, ck, cv,
        num_heads=nheads, head_dim=hdim,
        q_seq=1, kv_seq=maxsrc)
    ca = graph_ops.add_bias_sum(network, graph_ops.add_matmul_rhs_constant(
        network, ccf, hsz, hsz, weights[f"{pfx}.xw_o"], dtype=dtype), hsz,
        weights[f"{pfx}.xb_o"], dtype=dtype)
    pca = network.add_elementwise(psa, ca, trt.ElementWiseOperation.SUM).get_output(0)
    # ReLU MLP
    fn = graph_ops.add_layer_norm(network, pca, hsz,
        weights[f"{pfx}.ffn_norm"], weights[f"{pfx}.ffn_norm_b"], eps,
        dtype=dtype)
    mlp = graph_blocks.add_gelu_fc_mlp(network, fn, weights=weights, prefix=pfx,
        hidden_size=hsz, mlp_size=ffn, activation="relu", dtype=dtype)
    out = network.add_elementwise(pca, mlp, trt.ElementWiseOperation.SUM).get_output(0)
    return {"hidden": out, "present_k": pk, "present_v": pv}


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class CanaryPlugin:
    name = "canary"
    runtime_strategy = "canary_speech_to_text"
    _DEFAULT_MEL_LENGTH = 3000  # 30 seconds at 10 ms hop.

    def __init__(self):
        self._vl_config: dict = {}
        self._prompt_metadata: dict = {}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in ("canary", "canary_asr", "enc_dec_multi_task")

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        w = WeightDict()
        sd, ncfg = _load_nemo_archive(model_dir)
        tokenizer_model = _extract_tokenizer_from_nemo(model_dir, Path(model_dir), ncfg)
        if tokenizer_model.exists():
            self._prompt_metadata = _canary_prompt_metadata(ncfg, tokenizer_model)

        ec = ncfg.get("encoder", {})
        hidden = int(ec.get("d_model", 1024))
        mel_bins = int(ncfg.get("preprocessor", {}).get("features", ec.get("feat_in", 128)))
        kern = int(ec.get("conv_kernel_size", 9))
        enc_heads = int(ec.get("n_heads", 8))
        enc_ffn = int(ec.get("ff_expansion_factor", 4)) * hidden
        sub_ch = int(ec.get("subsampling_conv_channels", 256))
        head_dim = hidden // enc_heads

        dc = ncfg.get("transf_decoder", {}).get("config_dict", {})
        dec_layers = int(dc.get("num_layers", 8))
        dec_heads = int(dc.get("num_attention_heads", 8))
        dec_ffn = int(dc.get("inner_size", 4 * hidden))

        enc_layers = max(int(k.split(".")[2]) for k in sd
                         if k.startswith("encoder.layers.")) + 1
        mel_length = _cfg_int(config.raw.get("mel_length"), ncfg.get("trtmc_mel_length"),
                              default=self._DEFAULT_MEL_LENGTH)
        enc_seq = _compute_enc_seq_len(mel_length)

        te = _to_np(sd["transf_decoder._embedding.token_embedding.weight"])
        vocab = te.shape[0]

        w["_enc_layers"] = enc_layers
        w["_dec_layers"] = dec_layers
        w["_enc_heads"] = enc_heads
        w["_dec_heads"] = dec_heads
        w["_enc_ffn"] = enc_ffn
        w["_dec_ffn"] = dec_ffn
        w["_hidden"] = hidden
        w["_vocab"] = vocab
        w["_mel_bins"] = mel_bins
        w["_kern"] = kern
        w["_mel_length"] = mel_length
        w["_enc_seq"] = enc_seq
        w["_sub_ch"] = sub_ch
        w["_head_dim"] = head_dim

        # --- Subsampling ---
        w["enc_sub_conv0_w"] = _to_np(sd["encoder.pre_encode.conv.0.weight"])
        w["enc_sub_conv0_b"] = _to_np(sd["encoder.pre_encode.conv.0.bias"])
        for s, (di, pi) in enumerate([(2, 3), (5, 6)]):
            w[f"enc_sub_dw{s}_w"] = _to_np(sd[f"encoder.pre_encode.conv.{di}.weight"])
            w[f"enc_sub_dw{s}_b"] = _to_np(sd[f"encoder.pre_encode.conv.{di}.bias"])
            w[f"enc_sub_pw{s}_w"] = _to_np(sd[f"encoder.pre_encode.conv.{pi}.weight"])
            w[f"enc_sub_pw{s}_b"] = _to_np(sd[f"encoder.pre_encode.conv.{pi}.bias"])
        w["enc_sub_out_w"] = _transpose_2d(_to_np(sd["encoder.pre_encode.out.weight"]), "sub")
        w["enc_sub_out_b"] = _to_np(sd["encoder.pre_encode.out.bias"])

        # --- Encoder layers ---
        for i in range(enc_layers):
            nk = f"encoder.layers.{i}"
            pk = f"el.{i}"
            for p, n in [("w_q","linear_q"),("w_k","linear_k"),("w_v","linear_v"),("w_o","linear_out")]:
                w[f"{pk}.{p}"] = _transpose_2d(_to_np(sd[f"{nk}.self_attn.{n}.weight"]), p)
                bk = f"{nk}.self_attn.{n}.bias"
                w[f"{pk}.b_{p[-1]}"] = _to_np(sd[bk]) if bk in sd else np.zeros(hidden, dtype=np.float32)
            w[f"{pk}.pos_bias_u"] = _to_np(sd[f"{nk}.self_attn.pos_bias_u"])
            w[f"{pk}.pos_bias_v"] = _to_np(sd[f"{nk}.self_attn.pos_bias_v"])
            w[f"{pk}.w_pos"] = _transpose_2d(_to_np(sd[f"{nk}.self_attn.linear_pos.weight"]), "pos")
            w[f"{pk}.norm_sa"] = _to_np(sd[f"{nk}.norm_self_att.weight"])
            w[f"{pk}.norm_sa_b"] = _to_np(sd[f"{nk}.norm_self_att.bias"])
            for fn, fk in [("ff1","feed_forward1"),("ff2","feed_forward2")]:
                w[f"{pk}.{fn}.w1"] = _transpose_2d(_to_np(sd[f"{nk}.{fk}.linear1.weight"]), f"{fn}1")
                w[f"{pk}.{fn}.b1"] = _to_np(sd[f"{nk}.{fk}.linear1.bias"]) if f"{nk}.{fk}.linear1.bias" in sd else np.zeros(enc_ffn, dtype=np.float32)
                w[f"{pk}.{fn}.w2"] = _transpose_2d(_to_np(sd[f"{nk}.{fk}.linear2.weight"]), f"{fn}2")
                w[f"{pk}.{fn}.b2"] = _to_np(sd[f"{nk}.{fk}.linear2.bias"]) if f"{nk}.{fk}.linear2.bias" in sd else np.zeros(hidden, dtype=np.float32)
                nm = "norm_feed_forward1" if fn == "ff1" else "norm_feed_forward2"
                w[f"{pk}.{fn}.norm"] = _to_np(sd[f"{nk}.{nm}.weight"])
                w[f"{pk}.{fn}.norm_b"] = _to_np(sd[f"{nk}.{nm}.bias"])
            w[f"{pk}.cpw1_w"] = _to_np(sd[f"{nk}.conv.pointwise_conv1.weight"])
            w[f"{pk}.cpw1_b"] = _to_np(sd[f"{nk}.conv.pointwise_conv1.bias"]) if f"{nk}.conv.pointwise_conv1.bias" in sd else np.zeros(2*hidden, dtype=np.float32)
            w[f"{pk}.cdw_w"] = _to_np(sd[f"{nk}.conv.depthwise_conv.weight"])
            w[f"{pk}.cdw_b"] = _to_np(sd[f"{nk}.conv.depthwise_conv.bias"]) if f"{nk}.conv.depthwise_conv.bias" in sd else np.zeros(hidden, dtype=np.float32)
            w[f"{pk}.bn_w"] = _to_np(sd[f"{nk}.conv.batch_norm.weight"])
            w[f"{pk}.bn_b"] = _to_np(sd[f"{nk}.conv.batch_norm.bias"])
            w[f"{pk}.bn_m"] = _to_np(sd[f"{nk}.conv.batch_norm.running_mean"])
            w[f"{pk}.bn_v"] = _to_np(sd[f"{nk}.conv.batch_norm.running_var"])
            w[f"{pk}.cpw2_w"] = _to_np(sd[f"{nk}.conv.pointwise_conv2.weight"])
            w[f"{pk}.cpw2_b"] = _to_np(sd[f"{nk}.conv.pointwise_conv2.bias"]) if f"{nk}.conv.pointwise_conv2.bias" in sd else np.zeros(hidden, dtype=np.float32)
            w[f"{pk}.norm_conv"] = _to_np(sd[f"{nk}.norm_conv.weight"])
            w[f"{pk}.norm_conv_b"] = _to_np(sd[f"{nk}.norm_conv.bias"])
            w[f"{pk}.norm_out"] = _to_np(sd[f"{nk}.norm_out.weight"])
            w[f"{pk}.norm_out_b"] = _to_np(sd[f"{nk}.norm_out.bias"])

        # Store linear_pos weights for runtime PE computation.
        # The PE is length-dependent (NeMo uses PE[max_len-S:]) so it must
        # be computed at runtime for the actual audio length, not at build time.
        # We store w_pos per layer; the C++ runtime computes PE, projects,
        # and passes the result as an engine input.
        w["_pe_max_len"] = 5000
        # Pre-compute relative PE projections per layer
        rpe = _relative_pe(enc_seq, hidden)
        for i in range(enc_layers):
            proj = rpe @ w[f"el.{i}.w_pos"]
            w[f"el.{i}.rpe_proj"] = proj.reshape(2*enc_seq-1, enc_heads, head_dim)

        # --- Decoder ---
        w["dec_emb"] = te
        pos_key = "transf_decoder._embedding.position_embedding.pos_enc"
        w["dec_pos"] = _to_np(sd[pos_key]) if pos_key in sd else _sinusoidal_pe(1024, hidden)
        w["_max_tgt"] = w["dec_pos"].shape[0]
        w["emb_ln"] = _to_np(sd["transf_decoder._embedding.layer_norm.weight"])
        w["emb_ln_b"] = _to_np(sd["transf_decoder._embedding.layer_norm.bias"])

        for i in range(dec_layers):
            nk = f"transf_decoder._decoder.layers.{i}"
            pk = f"layer.{i}"
            # Self-attention
            w[f"{pk}.w_q"] = _transpose_2d(_to_np(sd[f"{nk}.first_sub_layer.query_net.weight"]), "dq")
            w[f"{pk}.q_bias"] = _to_np(sd[f"{nk}.first_sub_layer.query_net.bias"])
            w[f"{pk}.w_k"] = _transpose_2d(_to_np(sd[f"{nk}.first_sub_layer.key_net.weight"]), "dk")
            w[f"{pk}.k_bias"] = _to_np(sd[f"{nk}.first_sub_layer.key_net.bias"])
            w[f"{pk}.w_v"] = _transpose_2d(_to_np(sd[f"{nk}.first_sub_layer.value_net.weight"]), "dv")
            w[f"{pk}.v_bias"] = _to_np(sd[f"{nk}.first_sub_layer.value_net.bias"])
            w[f"{pk}.w_o"] = _transpose_2d(_to_np(sd[f"{nk}.first_sub_layer.out_projection.weight"]), "do")
            w[f"{pk}.o_bias"] = _to_np(sd[f"{nk}.first_sub_layer.out_projection.bias"])
            w[f"{pk}.input_norm"] = _to_np(sd[f"{nk}.layer_norm_1.weight"])
            w[f"{pk}.input_norm_b"] = _to_np(sd[f"{nk}.layer_norm_1.bias"])
            # Cross-attention
            w[f"{pk}.xw_q"] = _transpose_2d(_to_np(sd[f"{nk}.second_sub_layer.query_net.weight"]), "xq")
            w[f"{pk}.xb_q"] = _to_np(sd[f"{nk}.second_sub_layer.query_net.bias"])
            w[f"{pk}.xw_k"] = _transpose_2d(_to_np(sd[f"{nk}.second_sub_layer.key_net.weight"]), "xk")
            w[f"{pk}.xb_k"] = _to_np(sd[f"{nk}.second_sub_layer.key_net.bias"])
            w[f"{pk}.xw_v"] = _transpose_2d(_to_np(sd[f"{nk}.second_sub_layer.value_net.weight"]), "xv")
            w[f"{pk}.xb_v"] = _to_np(sd[f"{nk}.second_sub_layer.value_net.bias"])
            w[f"{pk}.xw_o"] = _transpose_2d(_to_np(sd[f"{nk}.second_sub_layer.out_projection.weight"]), "xo")
            w[f"{pk}.xb_o"] = _to_np(sd[f"{nk}.second_sub_layer.out_projection.bias"])
            w[f"{pk}.xattn_norm"] = _to_np(sd[f"{nk}.layer_norm_2.weight"])
            w[f"{pk}.xattn_norm_b"] = _to_np(sd[f"{nk}.layer_norm_2.bias"])
            # ReLU FFN
            w[f"{pk}.w_fc1"] = _transpose_2d(_to_np(sd[f"{nk}.third_sub_layer.dense_in.weight"]), "df1")
            w[f"{pk}.fc1_bias"] = _to_np(sd[f"{nk}.third_sub_layer.dense_in.bias"])
            w[f"{pk}.w_fc2"] = _transpose_2d(_to_np(sd[f"{nk}.third_sub_layer.dense_out.weight"]), "df2")
            w[f"{pk}.fc2_bias"] = _to_np(sd[f"{nk}.third_sub_layer.dense_out.bias"])
            w[f"{pk}.ffn_norm"] = _to_np(sd[f"{nk}.layer_norm_3.weight"])
            w[f"{pk}.ffn_norm_b"] = _to_np(sd[f"{nk}.layer_norm_3.bias"])

        w["final_norm"] = _to_np(sd["transf_decoder._decoder.final_layer_norm.weight"])
        w["final_norm_b"] = _to_np(sd["transf_decoder._decoder.final_layer_norm.bias"])
        w["w_out"] = _transpose_2d(_to_np(sd["log_softmax.mlp.layer0.weight"]), "lm")
        w["out_bias"] = _to_np(sd["log_softmax.mlp.layer0.bias"])

        config.hidden_size = hidden
        config.vocab_size = vocab
        config.num_hidden_layers = dec_layers
        config.num_attention_heads = dec_heads

        self._vl_config = {
            "num_mel_bins": mel_bins, "max_source_positions": enc_seq,
            "max_target_positions": w["_max_tgt"],
            "encoder_layers": enc_layers, "decoder_layers": dec_layers,
            "encoder_attention_heads": enc_heads, "decoder_attention_heads": dec_heads,
            "has_vision_engine": True, "mel_length": mel_length,
            "subsampling_factor": 8, "sample_rate": 16000,
        }
        return w

    def build_engine(self, config: ModelConfig, weights: WeightDict,
                     max_cache_length: int, *, precision: str = "fp32",
                     quant_ctx=None, verbose: bool = False,
                     debug_layer_outputs: bool = False,
                     parallel_config=None) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="Canary tensor-parallel decoder builds")
            if quant_ctx is not None:
                raise ValueError("Canary tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError(
                    "Canary tensor-parallel builds do not support debug_layer_outputs")
            return build_canary_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                verbose=verbose,
                parallel_config=parallel,
            )

        dl = weights["_dec_layers"]
        dh = weights["_dec_heads"]
        df = weights["_dec_ffn"]
        h = weights["_hidden"]
        v = weights["_vocab"]
        hd = h // dh
        aw = max_cache_length + 1
        es = weights["_enc_seq"]
        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(
                f"Unsupported Canary precision {precision!r}; expected fp32 or fp16")
        encoder_layers = int(weights["_enc_layers"])
        decoder_io_component = encoder_layers + dl + 1
        use_fp32_decoder_io = (
            precision == "fp16" and
            decoder_io_component in {
                int(layer) for layer in config.raw.get("_fp32_layers", ())
            }
        )
        selected_fp32_layers = {
            int(layer) - encoder_layers
            for layer in config.raw.get("_fp32_layers", ())
            if encoder_layers <= int(layer) < encoder_layers + dl
        } if precision == "fp16" else set()

        log = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        b = trt.Builder(log)
        net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        tc = b.create_builder_config()
        tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        tid = net.add_input("token_id", trt.int32, (1,))
        pid = net.add_input("position_id", trt.int32, (1,))
        amask = net.add_input("attention_mask", trt.float32, (1, aw))
        cki, cvi, xki, xvi = [], [], [], []
        for i in range(dl):
            cki.append(net.add_input(graph_ops.layer_tensor_name("cache_k", i), trt.float32, (max_cache_length, h)))
            cvi.append(net.add_input(graph_ops.layer_tensor_name("cache_v", i), trt.float32, (max_cache_length, h)))
            xki.append(net.add_input(graph_ops.layer_tensor_name("cross_k", i), trt.float32, (es, h)))
            xvi.append(net.add_input(graph_ops.layer_tensor_name("cross_v", i), trt.float32, (es, h)))

        decoder_io_np_dtype = (
            np.float32 if use_fp32_decoder_io else work_np_dtype)
        decoder_io_trt_dtype = (
            trt.float32 if use_fp32_decoder_io else work_trt_dtype)
        emb_table = graph_ops.add_constant(
            net, (v, h), weights["dec_emb"], dtype=decoder_io_np_dtype)
        pos_np = weights["dec_pos"]
        pos_table = graph_ops.add_constant(
            net, pos_np.shape, pos_np, dtype=decoder_io_np_dtype)
        eps = graph_ops.add_constant(
            net, (1, 1), np.array([1e-5], dtype=work_np_dtype),
            dtype=work_np_dtype)
        fp32_eps = None
        if selected_fp32_layers or use_fp32_decoder_io:
            fp32_eps = graph_ops.add_constant(
                net, (1, 1), np.array([1e-5], dtype=np.float32),
                dtype=np.float32)

        hs = network_add_elementwise_sum(net,
            net.add_gather(emb_table, tid, 0).get_output(0),
            net.add_gather(pos_table, pid, 0).get_output(0))
        # Embedding LayerNorm (Canary-specific, Whisper doesn't have this)
        hs = graph_ops.add_layer_norm(net, hs, h,
            weights["emb_ln"], weights["emb_ln_b"],
            fp32_eps if use_fp32_decoder_io else eps,
            dtype=decoder_io_np_dtype)

        pko, pvo = [], []
        for li in range(dl):
            pfx = f"layer.{li}"
            layer_np_dtype = (
                np.float32 if li in selected_fp32_layers else work_np_dtype)
            layer_trt_dtype = (
                trt.float32 if layer_np_dtype == np.float32 else work_trt_dtype)
            if hs.dtype != layer_trt_dtype:
                hs = net.add_cast(hs, layer_trt_dtype).get_output(0)
            layer_cache_k = cki[li]
            layer_cache_v = cvi[li]
            layer_cross_k = xki[li]
            layer_cross_v = xvi[li]
            layer_mask = amask
            if layer_cache_k.dtype != layer_trt_dtype:
                layer_cache_k = net.add_cast(
                    layer_cache_k, layer_trt_dtype).get_output(0)
                layer_cache_v = net.add_cast(
                    layer_cache_v, layer_trt_dtype).get_output(0)
                layer_cross_k = net.add_cast(
                    layer_cross_k, layer_trt_dtype).get_output(0)
                layer_cross_v = net.add_cast(
                    layer_cross_v, layer_trt_dtype).get_output(0)
            if layer_mask.dtype != layer_trt_dtype:
                layer_mask = net.add_cast(
                    layer_mask, layer_trt_dtype).get_output(0)
            r = _add_decoder_layer(
                network=net, hidden=hs,
                cache_k=layer_cache_k, cache_v=layer_cache_v,
                cross_k=layer_cross_k, cross_v=layer_cross_v,
                attention_mask=layer_mask,
                eps=fp32_eps if layer_np_dtype == np.float32 else eps,
                weights=weights, pfx=pfx,
                hsz=h, nheads=dh, hdim=hd, ffn=df,
                maxcache=max_cache_length, maxsrc=es, dtype=layer_np_dtype)
            hs = r["hidden"]
            pko.append(r["present_k"])
            pvo.append(r["present_v"])

        if hs.dtype != decoder_io_trt_dtype:
            hs = net.add_cast(hs, decoder_io_trt_dtype).get_output(0)
        hs = graph_ops.add_layer_norm(
            net, hs, h, weights["final_norm"], weights["final_norm_b"],
            fp32_eps if use_fp32_decoder_io else eps,
            dtype=decoder_io_np_dtype)
        logits = graph_ops.add_bias_sum(net,
            graph_ops.add_matmul_rhs_constant(
                net, hs, h, v, weights["w_out"], dtype=decoder_io_np_dtype),
            v, weights["out_bias"], dtype=decoder_io_np_dtype)
        if logits.dtype != trt.float32:
            logits = net.add_cast(logits, trt.float32).get_output(0)
        logits.name = "logits"
        net.mark_output(logits)
        for i in range(dl):
            if pko[i].dtype != trt.float32:
                pko[i] = net.add_cast(pko[i], trt.float32).get_output(0)
            if pvo[i].dtype != trt.float32:
                pvo[i] = net.add_cast(pvo[i], trt.float32).get_output(0)
            pko[i].name = graph_ops.layer_tensor_name("present_k", i)
            pvo[i].name = graph_ops.layer_tensor_name("present_v", i)
            net.mark_output(pko[i])
            net.mark_output(pvo[i])

        if verbose:
            print(f"[trtmc build] Building Canary decoder ({dl}L, h={h}, heads={dh})", file=sys.stderr)
        plan = b.build_serialized_network(net, tc)
        if plan is None:
            raise RuntimeError("Canary decoder build failed")
        return bytes(plan)

    def build_vision_engine(self, model_dir: str, config: ModelConfig,
                            weights: WeightDict, *, precision: str = "fp32",
                            verbose: bool = False) -> bytes | None:
        return _build_encoder(
            config, weights, precision=precision, verbose=verbose,
            fp32_layers=config.raw.get("_fp32_layers", ()))

    def get_audio_config(self, config: ModelConfig) -> dict | None:
        """NeMo mel spectrogram parameters (differ from Whisper defaults)."""
        return {
            "mel_frontend": "nemo",
            "mel_n_fft": 512,
            "mel_win_length": 400,
            "mel_hop_length": 160,
            "mel_chunk_length": 30,
            "mel_sampling_rate": 16000,
            "mel_preemph": 0.97,
            "mel_normalize": "per_feature",
        }

    def build_extra_engines(self, config: ModelConfig, weights, max_cache_length: int, *, precision: str = "fp32", verbose: bool = False) -> dict | None:
        """Bake the NeMo mel filterbank into the bundle for C++ mel extraction."""
        num_mel_bins = weights.get("_mel_bins", 128)
        n_fft = 512
        sampling_rate = 16000
        n_freq_bins = 1 + n_fft // 2  # 257

        try:
            from transformers.audio_utils import mel_filter_bank
        except ImportError:
            print("[trtmc build] Warning: mel_filter_bank not available", file=sys.stderr)
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
        mel_fb_bytes = header.tobytes() + filters_flat.tobytes()

        if verbose:
            print(f"[trtmc build] Canary mel filterbank: {n_freq_bins}x{num_mel_bins}", file=sys.stderr)

        return {"mel_filterbank": mel_fb_bytes}

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict | None:
        """Override synthetic config with actual values from the NeMo model."""
        overrides = {
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
        }
        overrides.update(self._prompt_metadata)
        return overrides

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        return self._vl_config or {
            "num_mel_bins": 128, "max_source_positions": 375,
            "max_target_positions": 1024, "encoder_layers": 32,
            "decoder_layers": 8, "has_vision_engine": True,
            "mel_length": 3000, "subsampling_factor": 8, "sample_rate": 16000,
        }


def network_add_elementwise_sum(net, a, b):
    return net.add_elementwise(a, b, trt.ElementWiseOperation.SUM).get_output(0)


def _build_encoder(
    config, weights, *, precision="fp32", verbose=False, fp32_layers=(),
):
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
            f"Unsupported Canary precision {precision!r}; expected fp32 or fp16")

    log = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    b = trt.Builder(log)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    tc = b.create_builder_config()
    tc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    eps = graph_ops.add_constant(
        net, (1, 1), np.array([1e-5], dtype=work_np_dtype),
        dtype=work_np_dtype)
    fp32_eps = None
    selected_fp32_layers = set(fp32_layers) if precision == "fp16" else set()
    subsampling_component = el + int(weights["_dec_layers"])
    use_fp32_subsampling = subsampling_component in selected_fp32_layers
    if selected_fp32_layers:
        fp32_eps = graph_ops.add_constant(
            net, (1, 1), np.array([1e-5], dtype=np.float32),
            dtype=np.float32)
    mel = net.add_input("mel_features", trt.float32, (mb, ml))
    # Encoder attention mask: [1, 1, enc_seq] — 0.0 for valid, -10000.0 for padded.
    # Applied additively to self-attention scores before softmax.
    mask_2d = bool(weights.get("_encoder_attention_mask_2d", False))
    mask_shape = (1, es, es) if mask_2d else (1, 1, es)
    enc_mask = net.add_input("encoder_mask", trt.float32, mask_shape)
    if work_trt_dtype != trt.float32 and not use_fp32_subsampling:
        mel = net.add_cast(mel, work_trt_dtype).get_output(0)
    # Derive a multiplicative valid-position mask [1, 1, es] (1.0 valid / 0.0
    # padded) from the additive attention mask (0.0 valid / -10000.0 padded):
    #   valid = clamp(1 + enc_mask * 1e-4, 0, 1)
    # This needs no new engine input or C++ change. Used to zero padded time
    # positions inside each conformer conv module (see _add_conv_module).
    valid_mask = None
    if not mask_2d:
        scale = graph_ops.add_constant(
            net, (1, 1, 1), np.array([1e-4], dtype=np.float32),
            dtype=np.float32)
        one = graph_ops.add_constant(
            net, (1, 1, 1), np.array([1.0], dtype=np.float32),
            dtype=np.float32)
        zero = graph_ops.add_constant(
            net, (1, 1, 1), np.array([0.0], dtype=np.float32),
            dtype=np.float32)
        vm = net.add_elementwise(enc_mask, scale, trt.ElementWiseOperation.PROD).get_output(0)
        vm = net.add_elementwise(vm, one, trt.ElementWiseOperation.SUM).get_output(0)
        vm = net.add_elementwise(vm, zero, trt.ElementWiseOperation.MAX).get_output(0)
        vm = net.add_elementwise(vm, one, trt.ElementWiseOperation.MIN).get_output(0)
        valid_mask = vm

    hs = _build_subsampling(
        net, mel, weights, sc, h, mb, ml,
        dtype=np.float32 if use_fp32_subsampling else work_np_dtype)
    for li in range(el):
        pfx = f"el.{li}"
        layer_np_dtype = (
            np.float32 if li in selected_fp32_layers else work_np_dtype)
        layer_trt_dtype = (
            trt.float32 if layer_np_dtype == np.float32 else work_trt_dtype)
        if hs.dtype != layer_trt_dtype:
            hs = net.add_cast(hs, layer_trt_dtype).get_output(0)

        layer_mask = enc_mask
        layer_valid_mask = valid_mask
        if layer_mask.dtype != layer_trt_dtype:
            layer_mask = net.add_cast(
                layer_mask, layer_trt_dtype).get_output(0)
        if (layer_valid_mask is not None and
                layer_valid_mask.dtype != layer_trt_dtype):
            layer_valid_mask = net.add_cast(
                layer_valid_mask, layer_trt_dtype).get_output(0)

        rpe = graph_ops.add_constant(
            net, (2*es-1, eh, hd), weights[f"{pfx}.rpe_proj"],
            dtype=layer_np_dtype)
        hs = _add_conformer_block(
            net, hs, weights, pfx, h, eh, hd, ef, k, es, rpe,
            fp32_eps if layer_np_dtype == np.float32 else eps,
            layer_mask,
            conv_norm_type=conv_norm_type, conv_context_size=conv_context_size,
            valid_mask=layer_valid_mask, dtype=layer_np_dtype)

    output = hs
    if output.dtype != trt.float32:
        output = net.add_cast(output, trt.float32).get_output(0)
    output.name = "encoder_output"
    net.mark_output(output)
    if verbose:
        print(f"[trtmc build] Building Canary encoder ({el}L, h={h}, heads={eh}, seq={es})", file=sys.stderr)
    plan = b.build_serialized_network(net, tc)
    if plan is None:
        raise RuntimeError("Canary encoder build failed")
    return bytes(plan)


plugin = CanaryPlugin()
