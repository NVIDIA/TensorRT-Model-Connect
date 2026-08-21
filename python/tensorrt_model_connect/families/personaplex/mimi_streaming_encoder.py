# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT builder for the stateful PersonaPlex Mimi encoder."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops
from .checkpoint_mapper import WeightDict
from .mimi_weights import _load_mimi_weights
from .utils import BuilderContextFactory, with_builder_context


trt = trt_compat.get_trt() if trt_compat.is_available() else None

_PERSONAPLEX_MIMI_CHUNK_SAMPLES = 1_920
_PERSONAPLEX_MIMI_CODEBOOKS = 8
_PERSONAPLEX_MIMI_ATTENTION_CONTEXT = 250


def _add_mimi_streaming_conv1d(
    network,
    inp,
    weight,
    bias,
    out_channels,
    kernel_size,
    *,
    stride=1,
    state_name,
    replicate_initial_state_when_position_zero=None,
    dtype=np.float32,
):
    """Apply one official Mimi streaming-convolution step.

    Moshi retains ``kernel_size - stride`` input samples between calls.  The
    state is explicit in the TensorRT graph so a fixed 1920-sample engine can
    preserve the same operation shapes as the official streaming codec.
    """
    state_length = int(kernel_size) - int(stride)
    if state_length <= 0:
        return (
            graph_ops.add_conv1d(
                network,
                inp,
                weight,
                bias,
                out_channels,
                kernel_size,
                stride=stride,
                padding=0,
                dtype=dtype,
            ),
            None,
        )

    input_channels = int(weight.shape[1])
    state = network.add_input(
        state_name,
        trt.float32,
        (1, input_channels, state_length),
    )
    if replicate_initial_state_when_position_zero is not None:
        first_input = network.add_slice(
            inp,
            start=(0, 0, 0),
            shape=(1, input_channels, 1),
            stride=(1, 1, 1),
        ).get_output(0)
        replicated_initial_state = network.add_concatenation(
            [first_input] * state_length
        )
        replicated_initial_state.axis = 2

        first_position = network.add_slice(
            replicate_initial_state_when_position_zero,
            start=(0,),
            shape=(1,),
            stride=(1,),
        ).get_output(0)
        zero_position = graph_ops.add_constant(
            network,
            (1,),
            np.zeros((1,), dtype=np.int32),
            dtype=np.int32,
        )
        use_replicated_state = network.add_elementwise(
            first_position,
            zero_position,
            trt.ElementWiseOperation.EQUAL,
        ).get_output(0)
        condition_shape = network.add_shuffle(use_replicated_state)
        condition_shape.reshape_dims = (1, 1, 1)
        state = network.add_select(
            condition_shape.get_output(0),
            replicated_initial_state.get_output(0),
            state,
        ).get_output(0)
    joined_layer = network.add_concatenation([state, inp])
    joined_layer.axis = 2
    joined = joined_layer.get_output(0)

    output = graph_ops.add_conv1d(
        network,
        joined,
        weight,
        bias,
        out_channels,
        kernel_size,
        stride=stride,
        padding=0,
        dtype=dtype,
    )

    joined_length = int(joined.shape[2])
    next_state_layer = network.add_slice(
        joined,
        start=(0, 0, joined_length - state_length),
        shape=(1, input_channels, state_length),
        stride=(1, 1, 1),
    )
    next_state = next_state_layer.get_output(0)
    next_state.name = f"{state_name}_out"
    network.mark_output(next_state)
    return output, next_state


def _add_mimi_streaming_transformer_layer(
    network,
    hidden,
    *,
    layer_idx,
    hidden_size,
    num_heads,
    head_dim,
    intermediate_size,
    norm_eps,
    layer_weights,
    cos_cache,
    sin_cache,
    position_ids,
    cache_indices,
    attention_mask,
    dtype=np.float32,
):
    """Apply one two-token Mimi transformer step with explicit KV state."""
    # The codec frontend emits 25 Hz features: two rows per 12.5 Hz frame.
    current_tokens = 2
    cache_capacity = _PERSONAPLEX_MIMI_ATTENTION_CONTEXT

    eps_const = graph_ops.add_constant(
        network, (1, 1), np.array([[norm_eps]], dtype=dtype), dtype=dtype
    )

    residual = hidden
    normalized = graph_ops.add_layer_norm(
        network,
        hidden,
        hidden_size,
        layer_weights["input_layernorm.weight"],
        layer_weights["input_layernorm.bias"],
        eps_const,
        dtype=dtype,
    )

    q = graph_ops.add_matmul_rhs_constant(
        network,
        normalized,
        hidden_size,
        hidden_size,
        layer_weights["self_attn.q_proj.weight"],
        dtype=dtype,
    )
    k = graph_ops.add_matmul_rhs_constant(
        network,
        normalized,
        hidden_size,
        hidden_size,
        layer_weights["self_attn.k_proj.weight"],
        dtype=dtype,
    )
    v = graph_ops.add_matmul_rhs_constant(
        network,
        normalized,
        hidden_size,
        hidden_size,
        layer_weights["self_attn.v_proj.weight"],
        dtype=dtype,
    )
    q = graph_ops.add_apply_rope_native(
        network,
        q,
        num_heads,
        head_dim,
        cos_cache,
        sin_cache,
        position_ids,
        head_dim,
        interleaved=True,
        sequence_length=current_tokens,
    )
    k = graph_ops.add_apply_rope_native(
        network,
        k,
        num_heads,
        head_dim,
        cos_cache,
        sin_cache,
        position_ids,
        head_dim,
        interleaved=True,
        sequence_length=current_tokens,
    )

    cache_k_name = f"mimi_cache_k_{layer_idx}"
    cache_v_name = f"mimi_cache_v_{layer_idx}"
    cache_k = network.add_input(cache_k_name, trt.float32, (cache_capacity, hidden_size))
    cache_v = network.add_input(cache_v_name, trt.float32, (cache_capacity, hidden_size))
    all_k = network.add_scatter(
        cache_k,
        cache_indices,
        k,
        trt.ScatterMode.ND,
    ).get_output(0)
    all_v = network.add_scatter(
        cache_v,
        cache_indices,
        v,
        trt.ScatterMode.ND,
    ).get_output(0)

    context = graph_ops.add_attention_from_rows(
        network,
        q,
        all_k,
        all_v,
        num_heads=num_heads,
        head_dim=head_dim,
        q_seq=current_tokens,
        kv_seq=cache_capacity,
        causal=False,
        mask=attention_mask,
    )
    attention_out = graph_ops.add_matmul_rhs_constant(
        network,
        context,
        hidden_size,
        hidden_size,
        layer_weights["self_attn.o_proj.weight"],
        dtype=dtype,
    )
    attention_scale = graph_ops.add_constant(
        network,
        (1, hidden_size),
        layer_weights["self_attn_layer_scale.scale"].reshape(1, -1),
        dtype=dtype,
    )
    attention_out = network.add_elementwise(
        attention_out, attention_scale, trt.ElementWiseOperation.PROD
    ).get_output(0)
    hidden = network.add_elementwise(
        residual, attention_out, trt.ElementWiseOperation.SUM
    ).get_output(0)

    residual = hidden
    normalized = graph_ops.add_layer_norm(
        network,
        hidden,
        hidden_size,
        layer_weights["post_attention_layernorm.weight"],
        layer_weights["post_attention_layernorm.bias"],
        eps_const,
        dtype=dtype,
    )
    mlp = graph_ops.add_matmul_rhs_constant(
        network,
        normalized,
        hidden_size,
        intermediate_size,
        layer_weights["mlp.fc1.weight"],
        dtype=dtype,
    )
    mlp = graph_ops.add_gelu_erf(network, mlp, dtype=dtype)
    mlp = graph_ops.add_matmul_rhs_constant(
        network,
        mlp,
        intermediate_size,
        hidden_size,
        layer_weights["mlp.fc2.weight"],
        dtype=dtype,
    )
    mlp_scale = graph_ops.add_constant(
        network,
        (1, hidden_size),
        layer_weights["mlp_layer_scale.scale"].reshape(1, -1),
        dtype=dtype,
    )
    mlp = network.add_elementwise(mlp, mlp_scale, trt.ElementWiseOperation.PROD).get_output(0)
    hidden = network.add_elementwise(residual, mlp, trt.ElementWiseOperation.SUM).get_output(0)

    for name, cache in ((cache_k_name, all_k), (cache_v_name, all_v)):
        cache.name = f"{name}_out"
        network.mark_output(cache)

    return hidden


@with_builder_context(
    explicit_batch=True,
    disable_tf32=True,
    builder_optimization_level=0,
    max_num_tactics=1,
)
def _build_mimi_streaming_encoder_engine(
    weights: WeightDict,
    *,
    precision: str = "fp32",
    verbose: bool = False,
    max_frames: int = 512,
    num_output_codebooks: int = _PERSONAPLEX_MIMI_CODEBOOKS,
    model_dir: str | Path | None = None,
    _builder_context_factory: BuilderContextFactory,
) -> bytes | None:
    """Build the stateful 1920-sample Mimi encoder used by PersonaPlex."""
    if trt is None or graph_ops is None:
        if verbose:
            print(
                "[trtmc build] Skipping streaming Mimi encoder (TRT not available)",
                file=sys.stderr,
            )
        return None
    if precision != "fp32":
        raise ValueError(
            "PersonaPlex streaming Mimi encoder requires FP32 to match the official codec"
        )

    print(
        "[trtmc build] Building streaming Mimi encoder TRT engine ...",
        file=sys.stderr,
    )
    mimi_w, mimi_cfg = _load_mimi_weights(model_dir)
    dtype = np.float32

    hidden_size = mimi_cfg["hidden_size"]
    num_heads = mimi_cfg["num_attention_heads"]
    head_dim = mimi_cfg["head_dim"]
    intermediate_size = mimi_cfg["intermediate_size"]
    num_layers = mimi_cfg["num_hidden_layers"]
    norm_eps = mimi_cfg["norm_eps"]
    codebook_dim = mimi_cfg["codebook_dim"]
    codebook_size = mimi_cfg["codebook_size"]
    rope_theta = mimi_cfg.get("rope_theta", 10000.0)
    compress = mimi_cfg.get("compress", 2)
    upsampling_ratios = mimi_cfg["upsampling_ratios"]
    num_filters = mimi_cfg["num_filters"]
    kernel_size = mimi_cfg["kernel_size"]
    residual_kernel_size = mimi_cfg.get("residual_kernel_size", 3)
    last_kernel_size = mimi_cfg.get("last_kernel_size", 3)
    max_frames = max(1, int(max_frames))
    num_output_codebooks = int(num_output_codebooks)
    if not 1 <= num_output_codebooks <= 32:
        raise ValueError(
            f"Mimi output codebooks must be between 1 and 32, got {num_output_codebooks}"
        )

    builder_context = _builder_context_factory()
    builder = builder_context.builder
    network = builder_context.network
    config = builder_context.config

    audio_input = network.add_input(
        "audio_input",
        trt.float32,
        (1, 1, _PERSONAPLEX_MIMI_CHUNK_SAMPLES),
    )
    channels = [num_filters * (2**index) for index in range(len(upsampling_ratios) + 1)]
    x, _ = _add_mimi_streaming_conv1d(
        network,
        audio_input,
        mimi_w["encoder.layers.0.conv.weight"],
        mimi_w["encoder.layers.0.conv.bias"],
        channels[0],
        kernel_size,
        state_name="mimi_conv_state_0",
        dtype=dtype,
    )

    encoder_layers = (
        (1, 3, upsampling_ratios[3]),
        (4, 6, upsampling_ratios[2]),
        (7, 9, upsampling_ratios[1]),
        (10, 12, upsampling_ratios[0]),
    )
    for block_index, (residual_index, downsample_index, ratio) in enumerate(encoder_layers):
        input_channels = channels[block_index]
        output_channels = channels[block_index + 1]
        residual = x
        branch = graph_ops.add_elu(network, x)
        branch, _ = _add_mimi_streaming_conv1d(
            network,
            branch,
            mimi_w[f"encoder.layers.{residual_index}.block.1.conv.weight"],
            mimi_w[f"encoder.layers.{residual_index}.block.1.conv.bias"],
            input_channels // compress,
            residual_kernel_size,
            state_name=f"mimi_residual_state_{block_index}",
            dtype=dtype,
        )
        branch = graph_ops.add_elu(network, branch)
        branch = graph_ops.add_conv1d(
            network,
            branch,
            mimi_w[f"encoder.layers.{residual_index}.block.3.conv.weight"],
            mimi_w[f"encoder.layers.{residual_index}.block.3.conv.bias"],
            input_channels,
            1,
            stride=1,
            padding=0,
            dtype=dtype,
        )
        x = network.add_elementwise(residual, branch, trt.ElementWiseOperation.SUM).get_output(0)
        x = graph_ops.add_elu(network, x)
        x, _ = _add_mimi_streaming_conv1d(
            network,
            x,
            mimi_w[f"encoder.layers.{downsample_index}.conv.weight"],
            mimi_w[f"encoder.layers.{downsample_index}.conv.bias"],
            output_channels,
            2 * ratio,
            stride=ratio,
            state_name=f"mimi_downsample_state_{block_index}",
            dtype=dtype,
        )

    x = graph_ops.add_elu(network, x)
    x, _ = _add_mimi_streaming_conv1d(
        network,
        x,
        mimi_w["encoder.layers.14.conv.weight"],
        mimi_w["encoder.layers.14.conv.bias"],
        hidden_size,
        last_kernel_size,
        state_name="mimi_conv_state_output",
        dtype=dtype,
    )

    encoder_tokens = int(x.shape[2])
    if encoder_tokens != 2:
        raise ValueError(
            "PersonaPlex streaming Mimi frontend must emit two tokens per "
            f"chunk, got {encoder_tokens}"
        )
    to_rows = network.add_shuffle(x)
    to_rows.first_transpose = trt.Permutation([0, 2, 1])
    to_rows.reshape_dims = (encoder_tokens, hidden_size)
    x = to_rows.get_output(0)

    max_positions = max_frames * encoder_tokens
    cos_values = graph_ops.make_rope_table_half_dim(
        max_positions,
        head_dim,
        rope_theta,
        True,
        interleaved=True,
    )
    sin_values = graph_ops.make_rope_table_half_dim(
        max_positions,
        head_dim,
        rope_theta,
        False,
        interleaved=True,
    )
    cos_cache = graph_ops.add_constant(network, cos_values.shape, cos_values, dtype=dtype)
    sin_cache = graph_ops.add_constant(network, sin_values.shape, sin_values, dtype=dtype)
    position_ids = network.add_input("mimi_position_ids", trt.int32, (encoder_tokens,))
    cache_indices = network.add_input(
        "mimi_cache_indices",
        trt.int32,
        (encoder_tokens, 1),
    )
    attention_mask = network.add_input(
        "mimi_attention_mask",
        trt.float32,
        (
            1,
            1,
            encoder_tokens,
            _PERSONAPLEX_MIMI_ATTENTION_CONTEXT,
        ),
    )

    for layer_index in range(num_layers):
        prefix = f"encoder_transformer.layers.{layer_index}"
        layer_weights = {
            "input_layernorm.weight": mimi_w[f"{prefix}.input_layernorm.weight"],
            "input_layernorm.bias": mimi_w[f"{prefix}.input_layernorm.bias"],
            "self_attn.q_proj.weight": mimi_w[f"{prefix}.self_attn.q_proj.weight"].T.copy(),
            "self_attn.k_proj.weight": mimi_w[f"{prefix}.self_attn.k_proj.weight"].T.copy(),
            "self_attn.v_proj.weight": mimi_w[f"{prefix}.self_attn.v_proj.weight"].T.copy(),
            "self_attn.o_proj.weight": mimi_w[f"{prefix}.self_attn.o_proj.weight"].T.copy(),
            "self_attn_layer_scale.scale": mimi_w[f"{prefix}.self_attn_layer_scale.scale"],
            "post_attention_layernorm.weight": mimi_w[f"{prefix}.post_attention_layernorm.weight"],
            "post_attention_layernorm.bias": mimi_w[f"{prefix}.post_attention_layernorm.bias"],
            "mlp.fc1.weight": mimi_w[f"{prefix}.mlp.fc1.weight"].T.copy(),
            "mlp.fc2.weight": mimi_w[f"{prefix}.mlp.fc2.weight"].T.copy(),
            "mlp_layer_scale.scale": mimi_w[f"{prefix}.mlp_layer_scale.scale"],
        }
        x = _add_mimi_streaming_transformer_layer(
            network,
            x,
            layer_idx=layer_index,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            norm_eps=norm_eps,
            layer_weights=layer_weights,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            position_ids=position_ids,
            cache_indices=cache_indices,
            attention_mask=attention_mask,
            dtype=dtype,
        )

    to_channels = network.add_shuffle(x)
    to_channels.reshape_dims = (1, encoder_tokens, hidden_size)
    to_channels.second_transpose = trt.Permutation([0, 2, 1])
    x = to_channels.get_output(0)
    x, _ = _add_mimi_streaming_conv1d(
        network,
        x,
        mimi_w["downsample.conv.weight"],
        None,
        hidden_size,
        4,
        stride=compress,
        state_name="mimi_quantizer_downsample_state",
        replicate_initial_state_when_position_zero=position_ids,
        dtype=dtype,
    )
    output_frames = int(x.shape[2])
    if output_frames != 1:
        raise ValueError(
            "PersonaPlex streaming Mimi quantizer must emit one frame per "
            f"chunk, got {output_frames}"
        )
    semantic_projection = graph_ops.add_conv1d(
        network,
        x,
        mimi_w["quantizer.semantic_residual_vector_quantizer.input_proj.weight"],
        None,
        codebook_dim,
        1,
        dtype=dtype,
    )
    semantic_rows = network.add_shuffle(semantic_projection)
    semantic_rows.first_transpose = trt.Permutation([0, 2, 1])
    semantic_rows.reshape_dims = (output_frames, codebook_dim)
    semantic_rows = semantic_rows.get_output(0)
    semantic_codebook = mimi_w[
        "quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embedding"
    ]
    semantic_codebook_constant = graph_ops.add_constant(
        network,
        (codebook_dim, codebook_size),
        semantic_codebook.T.copy(),
        dtype=dtype,
    )
    semantic_similarity = network.add_matrix_multiply(
        semantic_rows,
        trt.MatrixOperation.NONE,
        semantic_codebook_constant,
        trt.MatrixOperation.NONE,
    ).get_output(0)
    semantic_norms = graph_ops.add_constant(
        network,
        (1, codebook_size),
        -0.5 * np.sum(semantic_codebook**2, axis=1, keepdims=True).T,
        dtype=dtype,
    )
    semantic_similarity = network.add_elementwise(
        semantic_similarity,
        semantic_norms,
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    semantic_topk = network.add_topk(semantic_similarity, trt.TopKOperation.MAX, 1, 1 << 1)
    all_indices = [semantic_topk.get_output(1)]

    acoustic_projection = graph_ops.add_conv1d(
        network,
        x,
        mimi_w["quantizer.acoustic_residual_vector_quantizer.input_proj.weight"],
        None,
        codebook_dim,
        1,
        dtype=dtype,
    )
    acoustic_rows = network.add_shuffle(acoustic_projection)
    acoustic_rows.first_transpose = trt.Permutation([0, 2, 1])
    acoustic_rows.reshape_dims = (output_frames, codebook_dim)
    acoustic_residual = acoustic_rows.get_output(0)
    for codebook_index in range(num_output_codebooks - 1):
        codebook = mimi_w[
            "quantizer.acoustic_residual_vector_quantizer.layers."
            f"{codebook_index}.codebook.embedding"
        ]
        codebook_constant = graph_ops.add_constant(
            network,
            (codebook_dim, codebook_size),
            codebook.T.copy(),
            dtype=dtype,
        )
        similarity = network.add_matrix_multiply(
            acoustic_residual,
            trt.MatrixOperation.NONE,
            codebook_constant,
            trt.MatrixOperation.NONE,
        ).get_output(0)
        norms = graph_ops.add_constant(
            network,
            (1, codebook_size),
            -0.5 * np.sum(codebook**2, axis=1, keepdims=True).T,
            dtype=dtype,
        )
        similarity = network.add_elementwise(
            similarity, norms, trt.ElementWiseOperation.SUM
        ).get_output(0)
        topk = network.add_topk(similarity, trt.TopKOperation.MAX, 1, 1 << 1)
        index = topk.get_output(1)
        all_indices.append(index)

        flat_index = network.add_shuffle(index)
        flat_index.reshape_dims = (output_frames,)
        row_codebook = graph_ops.add_constant(
            network,
            (codebook_size, codebook_dim),
            codebook.copy(),
            dtype=dtype,
        )
        selected = network.add_gather(row_codebook, flat_index.get_output(0), axis=0).get_output(0)
        acoustic_residual = network.add_elementwise(
            acoustic_residual, selected, trt.ElementWiseOperation.SUB
        ).get_output(0)

    stacked_layer = network.add_concatenation(all_indices)
    stacked_layer.axis = 1
    transposed = network.add_shuffle(stacked_layer.get_output(0))
    transposed.first_transpose = trt.Permutation([1, 0])
    codec_tokens = network.add_cast(transposed.get_output(0), trt.float32).get_output(0)
    codec_tokens.name = "codec_tokens"
    network.mark_output(codec_tokens)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        print(
            "[trtmc build] ERROR: streaming Mimi encoder engine build failed",
            file=sys.stderr,
        )
        return None
    plan_bytes = bytes(plan)
    print(
        "[trtmc build] Streaming Mimi encoder engine built "
        f"({len(plan_bytes) / (1024 * 1024):.1f} MB)",
        file=sys.stderr,
    )
    return plan_bytes
