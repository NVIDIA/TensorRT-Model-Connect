# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT builder for the GLM-ASR audio encoder and its projector.

The graph mirrors ``GlmAsrEncoder`` followed by ``GlmAsrMultiModalProjector``:

    mel [num_mel_bins, mel_length]
      conv1 (k3 s1 p1) + gelu
      conv2 (k3 s2 p1) + gelu            -> [hidden, mel_length // 2]
      transpose                          -> [frames, hidden]
      32 x pre-norm encoder layer, bidirectional attention with rotary q/k
      final LayerNorm
      stack MERGE_FACTOR frames          -> [frames // 4, encoder intermediate]
      linear_1 + gelu + linear_2         -> [frames // 4, decoder hidden]

The encoder attends over a fixed frame count, so the rotary tables are baked in
as constants and the attention runs without a mask.
"""

from __future__ import annotations

import sys

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops
from .config import (
    MERGE_FACTOR,
    audio_encoder_config,
    encoder_output_length,
    projector_config,
)

trt = trt_compat.get_trt()

_AUDIO = "audio.audio_tower."
_PROJ = "projector."


def _work_dtypes(precision: str):
    if precision == "fp16":
        return np.float16, trt.float16
    if precision == "bf16":
        return np.float16, trt.bfloat16
    return np.float32, trt.float32


def _rhs(weights, key: str) -> np.ndarray:
    """HF Linear stores [out, in]; TRT matmul wants [in, out]."""
    return np.ascontiguousarray(weights[key].T)


def _conv_4d(weight: np.ndarray, work_np_dtype) -> np.ndarray:
    """Conv1d [out, in, k] -> Conv2d [out, in, 1, k]."""
    out_channels, in_channels, kernel = weight.shape
    return np.ascontiguousarray(
        weight.reshape(out_channels, in_channels, 1, kernel),
        dtype=work_np_dtype,
    )


def _add_conv_gelu(
    network, source, *, weights, prefix, in_channels, out_channels,
    length_in, length_out, stride, work_np_dtype,
):
    """One Conv1d stage expressed as a 2D convolution, followed by exact GELU."""
    shaped = network.add_shuffle(source)
    shaped.reshape_dims = (1, in_channels, 1, length_in)
    conv = network.add_convolution_nd(
        shaped.get_output(0),
        num_output_maps=out_channels,
        kernel_shape=(1, 3),
        kernel=trt.Weights(_conv_4d(weights[f"{prefix}.weight"], work_np_dtype)),
        bias=trt.Weights(
            np.ascontiguousarray(weights[f"{prefix}.bias"], dtype=work_np_dtype)),
    )
    conv.stride_nd = (1, stride)
    conv.padding_nd = (0, 1)
    squeezed = network.add_shuffle(conv.get_output(0))
    squeezed.reshape_dims = (out_channels, length_out)
    return graph_ops.add_activation(
        network, squeezed.get_output(0), "gelu", dtype=work_np_dtype)


def build_audio_encoder_engine(
    config, weights, *, precision: str = "fp32", verbose: bool = False,
) -> bytes:
    """Serialize the encoder-plus-projector engine for one mel chunk."""
    encoder = audio_encoder_config(config.raw)
    projector = projector_config(config.raw)
    if encoder is None or projector is None:
        raise ValueError("glmasr checkpoint is missing audio or text config")

    hidden = encoder.hidden_size
    heads = encoder.num_attention_heads
    head_dim = encoder.head_dim
    frames = encoder.max_position_embeddings
    mel_length = frames * 2
    if encoder_output_length(mel_length) != frames:
        raise ValueError(
            f"conv geometry yields {encoder_output_length(mel_length)} frames "
            f"for {mel_length} mel frames, but the encoder declares {frames}"
        )
    merged = frames // MERGE_FACTOR

    work_np_dtype, work_trt_dtype = _work_dtypes(precision)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()

    eps = graph_ops.add_constant(
        network, (1, 1),
        np.array([encoder.layer_norm_eps], dtype=work_np_dtype),
        dtype=work_np_dtype,
    )
    # The mel input stays FP32 whatever the build precision. The runtime copies
    # host tensors into engine buffers byte for byte, so a half input would
    # silently reinterpret FP32 mel values; keeping this edge FP32 makes the
    # C++ contract independent of precision at a negligible transfer cost.
    mel = network.add_input(
        "mel_features", trt.float32, (encoder.num_mel_bins, mel_length))
    if work_trt_dtype != trt.float32:
        mel = network.add_cast(mel, work_trt_dtype).get_output(0)

    hs = _add_conv_gelu(
        network, mel, weights=weights, prefix=f"{_AUDIO}conv1",
        in_channels=encoder.num_mel_bins, out_channels=hidden,
        length_in=mel_length, length_out=mel_length, stride=1,
        work_np_dtype=work_np_dtype,
    )
    hs = _add_conv_gelu(
        network, hs, weights=weights, prefix=f"{_AUDIO}conv2",
        in_channels=hidden, out_channels=hidden,
        length_in=mel_length, length_out=frames, stride=2,
        work_np_dtype=work_np_dtype,
    )

    transposed = network.add_shuffle(hs)
    transposed.first_transpose = trt.Permutation([1, 0])
    hs = transposed.get_output(0)

    rotary_dim = encoder.rotary_dim
    cos_table, sin_table = graph_ops.make_encoder_rope_half_tables(
        frames, rotary_dim, encoder.rope_theta)

    def _rope(rows, num_rope_heads):
        return graph_ops.add_apply_rope_rows(
            network, rows, frames, num_rope_heads, head_dim, rotary_dim,
            cos_table, sin_table, dtype=work_np_dtype,
        )

    def _project(source, key, in_features, out_features, *, bias=True):
        out = graph_ops.add_matmul_rhs_constant(
            network, source, in_features, out_features,
            _rhs(weights, f"{key}.weight"), dtype=work_np_dtype,
        )
        if not bias:
            return out
        return graph_ops.add_bias_sum(
            network, out, out_features,
            weights[f"{key}.bias"], dtype=work_np_dtype,
        )

    attention_size = heads * head_dim
    for layer_idx in range(encoder.num_hidden_layers):
        layer = f"{_AUDIO}layers.{layer_idx}"

        normed = graph_ops.add_layer_norm(
            network, hs, hidden,
            weights[f"{layer}.input_layernorm.weight"],
            weights[f"{layer}.input_layernorm.bias"],
            eps, dtype=work_np_dtype,
        )

        # q/v/o carry biases; k_proj does not, matching GlmAsrAttention.
        q = _rope(_project(
            normed, f"{layer}.self_attn.q_proj", hidden, attention_size), heads)
        k = _rope(_project(
            normed, f"{layer}.self_attn.k_proj", hidden,
            encoder.kv_attention_size, bias=False), encoder.num_key_value_heads)
        v = _project(
            normed, f"{layer}.self_attn.v_proj", hidden,
            encoder.kv_attention_size)

        attended = graph_ops.add_attention_from_rows(
            network, q, k, v,
            num_heads=heads, head_dim=head_dim,
            num_kv_heads=encoder.num_key_value_heads,
            q_seq=frames, kv_seq=frames,
            causal=False, scale=float(head_dim) ** -0.5,
            tag=f"glmasr_encoder_layer_{layer_idx}",
        )
        projected = _project(
            attended, f"{layer}.self_attn.o_proj", attention_size, hidden)
        hs = network.add_elementwise(
            hs, projected, trt.ElementWiseOperation.SUM).get_output(0)

        normed = graph_ops.add_layer_norm(
            network, hs, hidden,
            weights[f"{layer}.post_attention_layernorm.weight"],
            weights[f"{layer}.post_attention_layernorm.bias"],
            eps, dtype=work_np_dtype,
        )
        ffn = _project(
            normed, f"{layer}.mlp.fc1", hidden, encoder.intermediate_size)
        ffn = graph_ops.add_activation(
            network, ffn, encoder.hidden_act, dtype=work_np_dtype)
        ffn = _project(
            ffn, f"{layer}.mlp.fc2", encoder.intermediate_size, hidden)
        hs = network.add_elementwise(
            hs, ffn, trt.ElementWiseOperation.SUM).get_output(0)

    hs = graph_ops.add_layer_norm(
        network, hs, hidden,
        weights[f"{_AUDIO}norm.weight"], weights[f"{_AUDIO}norm.bias"],
        eps, dtype=work_np_dtype,
    )

    # Stack MERGE_FACTOR consecutive frames into one projector input row.
    stacked = network.add_shuffle(hs)
    stacked.reshape_dims = (merged, projector.in_features)
    hs = stacked.get_output(0)

    hs = _project(
        hs, f"{_PROJ}linear_1",
        projector.in_features, projector.hidden_features)
    hs = graph_ops.add_activation(
        network, hs, projector.hidden_act, dtype=work_np_dtype)
    hs = _project(
        hs, f"{_PROJ}linear_2",
        projector.hidden_features, projector.out_features)

    # The decoder consumes these embeddings directly, so keep them in FP32.
    if work_trt_dtype != trt.float32:
        hs = network.add_cast(hs, trt.float32).get_output(0)
    hs.name = "audio_embeddings"
    network.mark_output(hs)

    if verbose:
        print(
            f"[trtmc build] Building GLM-ASR audio encoder "
            f"({encoder.num_hidden_layers}L, h={hidden}, heads={heads}, "
            f"mel={encoder.num_mel_bins}, frames={frames}, "
            f"embeddings={merged}, precision={precision})",
            file=sys.stderr,
        )

    serialized = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError("failed to build the GLM-ASR audio encoder engine")
    return bytes(serialized)
