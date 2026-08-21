# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph helpers used by the VoiceChat streaming FastConformer plans."""

from __future__ import annotations

import math

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops


trt = trt_compat.get_trt()


def _relative_pe(seq_len: int, d_model: int, max_len: int = 5000) -> np.ndarray:
    """Compute the NeMo Transformer-XL relative position table."""
    pos = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(math.log(10000.0) / d_model))

    pe_pos = np.zeros((max_len, d_model), dtype=np.float32)
    pe_pos[:, 0::2] = np.sin(pos * div)
    pe_pos[:, 1::2] = np.cos(pos * div)
    pe_pos = pe_pos[::-1].copy()

    pe_neg = np.zeros((max_len, d_model), dtype=np.float32)
    pe_neg[:, 0::2] = np.sin(-pos * div)
    pe_neg[:, 1::2] = np.cos(-pos * div)
    pe_full = np.concatenate([pe_pos, pe_neg[1:]], axis=0)
    return pe_full[max_len - seq_len : max_len + seq_len - 1]


def _compute_causal_enc_seq_len(mel_length: int) -> int:
    for _ in range(3):
        mel_length = mel_length // 2 + 1
    return mel_length


def _build_subsampling(
    network,
    mel_input,
    weights,
    sub_ch,
    hidden,
    num_mel_bins,
    mel_length,
    time_out,
    dtype=np.float32,
):
    """Build the checkpoint's three causal stride-two subsampling stages."""

    def add_subsample_conv(inp, weight, bias, out_channels, *, groups=1):
        padded = network.add_padding_nd(inp, pre_padding=(2, 2), post_padding=(1, 1))
        return graph_ops.add_conv2d(
            network,
            padded.get_output(0),
            weight=weight,
            bias=bias,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            groups=groups,
            dtype=dtype,
        )

    transposed = network.add_shuffle(mel_input)
    transposed.first_transpose = trt.Permutation([1, 0])
    reshaped = network.add_shuffle(transposed.get_output(0))
    reshaped.reshape_dims = (1, 1, mel_length, num_mel_bins)
    x = add_subsample_conv(
        reshaped.get_output(0), weights["enc_sub_conv0_w"], weights["enc_sub_conv0_b"], sub_ch
    )
    x = graph_ops.add_activation(network, x, "relu")
    for stage in range(2):
        x = add_subsample_conv(
            x,
            weights[f"enc_sub_dw{stage}_w"],
            weights[f"enc_sub_dw{stage}_b"],
            sub_ch,
            groups=sub_ch,
        )
        x = graph_ops.add_conv2d(
            network,
            x,
            weight=weights[f"enc_sub_pw{stage}_w"],
            bias=weights[f"enc_sub_pw{stage}_b"],
            out_channels=sub_ch,
            kernel_size=(1, 1),
            dtype=dtype,
        )
        x = graph_ops.add_activation(network, x, "relu")

    sub_out_in = int(weights["enc_sub_out_w"].shape[0])
    feat_out = sub_out_in // sub_ch
    transposed = network.add_shuffle(x)
    transposed.first_transpose = trt.Permutation([0, 2, 1, 3])
    transposed.reshape_dims = (time_out, sub_ch * feat_out)
    output = graph_ops.add_matmul_rhs_constant(
        network,
        transposed.get_output(0),
        sub_ch * feat_out,
        hidden,
        weights["enc_sub_out_w"],
        dtype=dtype,
    )
    return graph_ops.add_bias_sum(network, output, hidden, weights["enc_sub_out_b"], dtype=dtype)


def _add_conv_norm(network, x, weights, pfx, hidden, sequence_length, eps, dtype=np.float32):
    """Apply the checkpoint's per-time-step layer normalization."""
    rows = network.add_shuffle(x)
    rows.reshape_dims = (hidden, sequence_length)
    transposed = network.add_shuffle(rows.get_output(0))
    transposed.first_transpose = trt.Permutation([1, 0])
    normalized = graph_ops.add_layer_norm(
        network,
        transposed.get_output(0),
        hidden,
        weights[f"{pfx}.bn_w"],
        weights[f"{pfx}.bn_b"],
        eps,
        dtype=dtype,
    )
    channels = network.add_shuffle(normalized)
    channels.first_transpose = trt.Permutation([1, 0])
    output = network.add_shuffle(channels.get_output(0))
    output.reshape_dims = (1, hidden, sequence_length)
    return output.get_output(0)


def _add_half_ffn(network, hidden_states, weights, prefix, hidden, ffn, eps, dtype=np.float32):
    normalized = graph_ops.add_layer_norm(
        network,
        hidden_states,
        hidden,
        weights[f"{prefix}.norm"],
        weights[f"{prefix}.norm_b"],
        eps,
        dtype=dtype,
    )
    projected = graph_ops.add_matmul_rhs_constant(
        network, normalized, hidden, ffn, weights[f"{prefix}.w1"], dtype=dtype
    )
    projected = graph_ops.add_bias_sum(
        network, projected, ffn, weights[f"{prefix}.b1"], dtype=dtype
    )
    activated = graph_ops.add_activation(network, projected, "silu")
    projected = graph_ops.add_matmul_rhs_constant(
        network, activated, ffn, hidden, weights[f"{prefix}.w2"], dtype=dtype
    )
    projected = graph_ops.add_bias_sum(
        network, projected, hidden, weights[f"{prefix}.b2"], dtype=dtype
    )
    half = graph_ops.add_constant(network, (1, 1), np.array([0.5], dtype=dtype), dtype=dtype)
    return network.add_elementwise(projected, half, trt.ElementWiseOperation.PROD).get_output(0)
