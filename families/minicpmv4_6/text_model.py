# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V 4.6 language model: the Qwen3.5 hybrid decoder as TensorRT graphs.

The text tower is a ``qwen3_5_text`` decoder: Gated DeltaNet layers
(``linear_attention``) interleaved with gated full self-attention layers, a
SwiGLU MLP in every layer, (1 + weight) RMSNorm and tied input/output
embeddings. Positions are one-dimensional: the checkpoint's provider passes
plain sequence positions, so every rotary axis carries the same index.

Two engines share the weights and the runtime state contract:

``decoder``  one token per launch.
``prefill``  1..``PREFILL_MAX_ROWS`` prompt rows per launch. Each row takes
             either its token embedding or one supplied vision feature row
             (``visual_embedding`` selected by ``use_visual_embedding``), so a
             prompt that interleaves text and image tokens is prefilled in
             chunks without a host copy of the embedding table.

Both engines read and write the DeltaNet convolution and recurrent state and
the attention KV cache under the same tensor names; the recurrence and its
state stay in FP32 at every precision, as in the reference implementation.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import deltanet_rows
from . import graph_blocks
from . import graph_ops
from .checkpoint_mapper import WeightDict, _has_tensor, _load_tensor, decay_rates
from .config import TextConfig
from .rope import rope_positions, rope_tables

# Prefill rows per launch; the chunked delta rule pads its solve to this size.
PREFILL_MAX_ROWS = deltanet_rows.SOLVE_ROWS
PREFILL_OPT_ROWS = 32


def _projection(network, x, in_features, out_features, weights, key, *, dtype):
    """Projection matmul with a constant ``[in, out]`` right-hand side."""
    return graph_ops.add_matmul_rhs_constant(
        network, x, in_features, out_features, weights[key], dtype=dtype
    )


def _normalized_heads(network: trt.INetworkDefinition, tensor: trt.ITensor, reduce_axis: int) -> trt.ITensor:
    """L2-normalized Q or K heads in FP32: the reference's l2norm output enters its FP32 recurrence unrounded."""
    if tensor.dtype != trt.float32:
        tensor = network.add_cast(tensor, trt.float32).get_output(0)
    return graph_ops.add_l2_norm(network, tensor, reduce_axis, eps=1e-6, dtype=np.float32)


def _prepare_runtime_inputs(
    network,
    work_trt_dtype,
    attention_mask,
    conv_state_inputs,
    ssm_state_inputs,
    cache_k_inputs,
    cache_v_inputs,
):
    """Cast storage tensors while preserving DeltaNet recurrence in FP32."""
    if work_trt_dtype == trt.float32:
        return (
            attention_mask,
            conv_state_inputs,
            ssm_state_inputs,
            cache_k_inputs,
            cache_v_inputs,
        )

    def cast_all(tensors):
        return [network.add_cast(tensor, work_trt_dtype).get_output(0) for tensor in tensors]

    return (
        network.add_cast(attention_mask, work_trt_dtype).get_output(0),
        cast_all(conv_state_inputs),
        # The reference keeps the DeltaNet recurrent state in FP32. Never round
        # this persistent input before the per-token recurrence.
        ssm_state_inputs,
        cast_all(cache_k_inputs),
        cast_all(cache_v_inputs),
    )


def _transposed(readers, key: str, rows: int, columns: int) -> np.ndarray:
    """Read one ``[out, in]`` checkpoint projection as an FP32 ``[in, out]`` array."""
    return np.ascontiguousarray(_load_tensor(readers, key, (rows, columns)).T)


def _vector(readers, key: str, size: int) -> np.ndarray:
    return np.ascontiguousarray(_load_tensor(readers, key, (size,)), dtype=np.float32)


def load_text_weights(readers, config: TextConfig) -> WeightDict:
    """Map the ``model.language_model.*`` tensors to the decoder graph weights."""
    hidden = config.hidden_size
    vocab = config.vocab_size
    attn_size = config.num_attention_heads * config.head_dim
    kv_size = config.num_key_value_heads * config.head_dim
    d_inner = config.linear_num_value_heads * config.linear_value_head_dim
    qk_dim = config.linear_num_key_heads * config.linear_key_head_dim
    conv_dim = 2 * qk_dim + d_inner
    root = "model.language_model"

    weights = WeightDict()
    embedding = _load_tensor(readers, f"{root}.embed_tokens.weight", (vocab, hidden))
    weights["embedding"] = np.ascontiguousarray(embedding, dtype=np.float32)
    # The checkpoint ties the LM head to the input embedding.
    if _has_tensor(readers, "lm_head.weight"):
        raise ValueError("MiniCPM-V 4.6 ties lm_head to the input embedding; found a separate lm_head")
    weights["w_lm_head"] = np.ascontiguousarray(embedding.T, dtype=np.float32)

    for index, layer_type in enumerate(config.layer_types):
        prefix = f"layer.{index}"
        source = f"{root}.layers.{index}"
        # Qwen3.5 RMSNorm scales by (1 + weight).
        weights[f"{prefix}.input_norm"] = 1.0 + _vector(readers, f"{source}.input_layernorm.weight", hidden)
        weights[f"{prefix}.post_attn_norm"] = 1.0 + _vector(
            readers, f"{source}.post_attention_layernorm.weight", hidden
        )
        if layer_type == "deltanet":
            linear = f"{source}.linear_attn"
            heads = config.linear_num_value_heads
            head_dim = config.linear_value_head_dim
            weights[f"{prefix}.deltanet_in_proj_qkv"] = _transposed(
                readers, f"{linear}.in_proj_qkv.weight", conv_dim, hidden
            )
            weights[f"{prefix}.deltanet_z_proj"] = _transposed(readers, f"{linear}.in_proj_z.weight", d_inner, hidden)
            weights[f"{prefix}.deltanet_a_proj"] = _transposed(readers, f"{linear}.in_proj_a.weight", heads, hidden)
            weights[f"{prefix}.deltanet_b_proj"] = _transposed(readers, f"{linear}.in_proj_b.weight", heads, hidden)
            # Decay uses -exp(A_log); the reference evaluates it in FP32.
            weights[f"{prefix}.A"] = decay_rates(_vector(readers, f"{linear}.A_log", heads), f"{linear}.A_log")
            weights[f"{prefix}.dt_bias"] = _vector(readers, f"{linear}.dt_bias", heads)
            conv = _load_tensor(readers, f"{linear}.conv1d.weight", (conv_dim, 1, config.linear_conv_kernel_dim))
            weights[f"{prefix}.conv1d_weight"] = np.ascontiguousarray(
                conv.reshape(conv_dim, config.linear_conv_kernel_dim), dtype=np.float32
            )
            if _has_tensor(readers, f"{linear}.conv1d.bias"):
                raise ValueError("MiniCPM-V 4.6 DeltaNet convolutions have no bias")
            weights[f"{prefix}.conv1d_bias"] = np.zeros(conv_dim, dtype=np.float32)
            # Gated RMSNorm weight is per head dimension; the graph applies it per head.
            weights[f"{prefix}.deltanet_norm"] = np.tile(_vector(readers, f"{linear}.norm.weight", head_dim), heads)
            weights[f"{prefix}.deltanet_out_proj"] = _transposed(
                readers, f"{linear}.out_proj.weight", hidden, d_inner
            )
        else:
            attention = f"{source}.self_attn"
            heads = config.num_attention_heads
            head_dim = config.head_dim
            # q_proj interleaves query and gate per head: [head][query | gate].
            query_gate = _transposed(readers, f"{attention}.q_proj.weight", 2 * attn_size, hidden)
            rows = np.arange(2 * attn_size).reshape(heads, 2 * head_dim)
            weights[f"{prefix}.w_q"] = np.ascontiguousarray(query_gate[:, rows[:, :head_dim].reshape(-1)])
            weights[f"{prefix}.w_gate_attn"] = np.ascontiguousarray(query_gate[:, rows[:, head_dim:].reshape(-1)])
            weights[f"{prefix}.w_k"] = _transposed(readers, f"{attention}.k_proj.weight", kv_size, hidden)
            weights[f"{prefix}.w_v"] = _transposed(readers, f"{attention}.v_proj.weight", kv_size, hidden)
            weights[f"{prefix}.w_o"] = _transposed(readers, f"{attention}.o_proj.weight", hidden, attn_size)
            weights[f"{prefix}.q_norm"] = np.tile(
                1.0 + _vector(readers, f"{attention}.q_norm.weight", head_dim), heads
            )
            weights[f"{prefix}.k_norm"] = np.tile(
                1.0 + _vector(readers, f"{attention}.k_norm.weight", head_dim), config.num_key_value_heads
            )
        mlp = f"{source}.mlp"
        weights[f"{prefix}.w_gate"] = _transposed(readers, f"{mlp}.gate_proj.weight", config.intermediate_size, hidden)
        weights[f"{prefix}.w_up"] = _transposed(readers, f"{mlp}.up_proj.weight", config.intermediate_size, hidden)
        weights[f"{prefix}.w_down"] = _transposed(readers, f"{mlp}.down_proj.weight", hidden, config.intermediate_size)

    weights["final_norm"] = 1.0 + _vector(readers, f"{root}.norm.weight", hidden)
    return weights


def text_runtime_config(config: TextConfig) -> dict:
    """Decoder geometry shared with the native runtime (``runtime.json``)."""
    return {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": len(config.layer_types),
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "layer_types": list(config.layer_types),
        "num_mamba_layers": config.layer_types.count("deltanet"),
        "num_attention_layers": config.layer_types.count("attention"),
        "d_inner": config.linear_num_value_heads * config.linear_value_head_dim,
        "mamba_d_state": config.linear_value_head_dim,
        "mamba_d_conv": config.linear_conv_kernel_dim,
        "mamba_nheads": config.linear_num_value_heads,
        "mamba_head_dim": config.linear_value_head_dim,
        "conv_dim": 2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim,
    }


def build_text_engine(
    config: TextConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str,
    prefill_rows: int | None = None,
    verbose: bool = False,
) -> bytes:
    """Build the one-token decoder, or the prefill engine when ``prefill_rows`` is set.

    The prefill engine runs ``S`` rows with one profile ``1..prefill_rows``:
    DeltaNet as one chunked delta rule and attention over the cache plus the
    chunk's causal prefix. Its logits come from the last row; present K/V
    cover all rows and the recurrent outputs hold the state after the last row.
    """
    if prefill_rows is not None and not 1 <= prefill_rows <= PREFILL_MAX_ROWS:
        raise ValueError(f"MiniCPM-V 4.6 prefill rows must be in 1..{PREFILL_MAX_ROWS}")
    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "fp32":
        work_np_dtype, work_trt_dtype = np.float32, trt.float32
    else:
        raise ValueError(f"Unsupported MiniCPM-V 4.6 precision {precision!r}; expected fp32 or fp16")

    hidden = config.hidden_size
    vocab = config.vocab_size
    layer_types = config.layer_types
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    attn_size = num_heads * head_dim
    kv_attention_size = num_kv_heads * head_dim
    d_inner = config.linear_num_value_heads * config.linear_value_head_dim
    deltanet_num_heads = config.linear_num_value_heads
    deltanet_num_kv_heads = config.linear_num_key_heads
    deltanet_head_dim = config.linear_value_head_dim
    d_conv = config.linear_conv_kernel_dim
    conv_dim = 2 * deltanet_num_kv_heads * config.linear_key_head_dim + d_inner
    mlp_size = config.intermediate_size
    num_mamba = layer_types.count("deltanet")
    num_attn = layer_types.count("attention")
    attention_window = rope_positions(max_cache_length)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.builder_optimization_level = 1
    # FP32 math stays FP32 (the recurrence, norms, FP32 builds), as in the reference.
    trt_config.clear_flag(trt.BuilderFlag.TF32)

    # --- Inputs ---
    rows = None if prefill_rows else 1
    row_dim = -1 if prefill_rows else 1
    token_id = network.add_input("token_id", trt.int32, (row_dim,))
    position_id = network.add_input("position_id", trt.int32, (row_dim,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (row_dim, -1 if prefill_rows else attention_window)
    )
    visual = selector = None
    if prefill_rows:
        visual = network.add_input("visual_embedding", work_trt_dtype, (-1, hidden))
        selector = network.add_input("use_visual_embedding", trt.int32, (-1, 1))
        max_rows = min(prefill_rows, max_cache_length)
        opt_rows = min(PREFILL_OPT_ROWS, max_rows)
        profile = builder.create_optimization_profile()
        for name in ("token_id", "position_id"):
            profile.set_shape(name, (1,), (opt_rows,), (max_rows,))
        profile.set_shape("visual_embedding", (1, hidden), (opt_rows, hidden), (max_rows, hidden))
        profile.set_shape("use_visual_embedding", (1, 1), (opt_rows, 1), (max_rows, 1))
        profile.set_shape(
            "attention_mask",
            (1, max_cache_length + 1),
            (opt_rows, max_cache_length + opt_rows),
            (max_rows, max_cache_length + max_rows),
        )
        trt_config.add_optimization_profile(profile)

    # DeltaNet state inputs (conv + ssm per DeltaNet layer)
    conv_state_inputs = []
    ssm_state_inputs = []
    for index in range(num_mamba):
        conv_state_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("conv_state", index), trt.float32, (conv_dim, d_conv)
        ))
        ssm_state_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("ssm_state", index),
            trt.float32,
            (deltanet_num_heads, deltanet_head_dim, deltanet_head_dim),
        ))

    # Attention KV cache inputs, stored at the compute precision.
    cache_k_inputs = []
    cache_v_inputs = []
    for index in range(num_attn):
        cache_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_k", index), work_trt_dtype, (max_cache_length, kv_attention_size)
        ))
        cache_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_v", index), work_trt_dtype, (max_cache_length, kv_attention_size)
        ))

    (
        attention_mask,
        conv_state_inputs,
        ssm_state_inputs,
        cache_k_inputs,
        cache_v_inputs,
    ) = _prepare_runtime_inputs(
        network,
        work_trt_dtype,
        attention_mask,
        conv_state_inputs,
        ssm_state_inputs,
        cache_k_inputs,
        cache_v_inputs,
    )

    # --- Shared constants ---
    embedding_table = graph_ops.add_constant(network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype
    )
    cos_half, sin_half = rope_tables(config, attention_window)
    cos_half_tensor = graph_ops.add_constant(network, cos_half.shape, cos_half, dtype=work_np_dtype)
    sin_half_tensor = graph_ops.add_constant(network, sin_half.shape, sin_half, dtype=work_np_dtype)

    # --- Embedding ---
    hidden_state = network.add_gather(embedding_table, token_id, 0).get_output(0)
    if prefill_rows:
        use_visual = network.add_cast(selector, trt.bool).get_output(0)
        hidden_state = network.add_select(use_visual, visual, hidden_state).get_output(0)

    # --- Layer stack ---
    present_conv_outputs = []
    present_ssm_outputs = []
    present_k_outputs = []
    present_v_outputs = []
    mamba_counter = 0
    attn_counter = 0
    for layer_index, layer_type in enumerate(layer_types):
        prefix = f"layer.{layer_index}"
        if layer_type == "deltanet":
            result = _add_deltanet_layer(
                network=network,
                hidden=hidden_state,
                conv_state_in=conv_state_inputs[mamba_counter],
                # The reference casts the DeltaNet recurrence and its persistent
                # state to FP32 even when the projections run in FP16.
                ssm_state_in=ssm_state_inputs[mamba_counter],
                eps_tensor=eps_tensor,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                d_inner=d_inner,
                d_conv=d_conv,
                conv_dim=conv_dim,
                num_heads=deltanet_num_heads,
                num_kv_heads=deltanet_num_kv_heads,
                head_dim=deltanet_head_dim,
                mlp_size=mlp_size,
                norm_eps=config.rms_norm_eps,
                dtype=work_np_dtype,
                sequence_length=rows,
            )
            hidden_state = result["hidden"]
            present_conv_outputs.append(result["present_conv"])
            present_ssm_outputs.append(result["present_ssm"])
            mamba_counter += 1
        else:
            result = _add_full_attention_layer(
                network=network,
                hidden=hidden_state,
                cache_k=cache_k_inputs[attn_counter],
                cache_v=cache_v_inputs[attn_counter],
                attention_mask=attention_mask,
                position_id=position_id,
                cos_half_tensor=cos_half_tensor,
                sin_half_tensor=sin_half_tensor,
                eps_tensor=eps_tensor,
                weights=weights,
                prefix=prefix,
                hidden_size=hidden,
                attn_size=attn_size,
                kv_attention_size=kv_attention_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_embedding_dim=config.rotary_dim,
                max_cache_length=max_cache_length,
                mlp_size=mlp_size,
                dtype=work_np_dtype,
                sequence_length=rows,
            )
            hidden_state = result["hidden"]
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])
            attn_counter += 1

    if prefill_rows:
        # Only the last row can affect the next token.
        last = network.add_elementwise(
            deltanet_rows.row_count(network, hidden_state),
            graph_ops.add_constant(network, (1,), np.array([1], dtype=np.int64), dtype=np.int64),
            trt.ElementWiseOperation.SUB,
        ).get_output(0)
        hidden_state = deltanet_rows.dynamic_slice(network, hidden_state, (last, 0), (1, hidden))

    # --- Final norm and tied LM head ---
    hidden_state = graph_ops.add_rms_norm(
        network, hidden_state, hidden, weights["final_norm"], eps_tensor, dtype=work_np_dtype
    )
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, weights["w_lm_head"], dtype=work_np_dtype
    )
    if logits.dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    # --- Present state outputs ---
    for index in range(num_mamba):
        conv = present_conv_outputs[index]
        ssm = present_ssm_outputs[index]
        if conv.dtype != trt.float32:
            conv = network.add_cast(conv, trt.float32).get_output(0)
        if ssm.dtype != trt.float32:
            ssm = network.add_cast(ssm, trt.float32).get_output(0)
        conv.name = graph_ops.layer_tensor_name("present_conv", index)
        ssm.name = graph_ops.layer_tensor_name("present_ssm", index)
        network.mark_output(conv)
        network.mark_output(ssm)
    for index in range(num_attn):
        key = present_k_outputs[index]
        value = present_v_outputs[index]
        if key.dtype != work_trt_dtype:
            key = network.add_cast(key, work_trt_dtype).get_output(0)
        if value.dtype != work_trt_dtype:
            value = network.add_cast(value, work_trt_dtype).get_output(0)
        key.name = graph_ops.layer_tensor_name("present_k", index)
        value.name = graph_ops.layer_tensor_name("present_v", index)
        network.mark_output(key)
        network.mark_output(value)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("MiniCPM-V 4.6 text engine build failed")
    return bytes(plan)


def _add_deltanet_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    conv_state_in: trt.ITensor,
    ssm_state_in: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    d_inner: int,
    d_conv: int,
    conv_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    mlp_size: int,
    norm_eps: float,
    dtype: np.dtype = np.float32,
    sequence_length: int | None = 1,
) -> dict[str, trt.ITensor]:
    """Add one Gated DeltaNet layer over one row, or ``S`` rows when ``sequence_length`` is None.

    DeltaNet uses delta-rule linear attention with:
      - Conv1d on QKV projection output
      - L2-normalized Q,K with Compact GQA/MQA K/V
      - Per-head decay (A_log + softplus(a + dt_bias))
      - Per-head beta (write strength via sigmoid)
      - Delta rule state update: S' = decay*S + outer(k, (v - S@k)*beta)
      - Gated output: norm(S'@q) * silu(z) * norm_weight

    Returns: {hidden, present_conv, present_ssm}
    """
    # ===== 1. RMSNorm =====
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype
    )

    # ===== 2. Input projections =====
    # QKV combined: [1, hidden] -> [1, conv_dim]
    qkv = _projection(
        network, normed, hidden_size, conv_dim, weights, f"{prefix}.deltanet_in_proj_qkv", dtype=dtype
    )

    # Gate (z): [1, hidden] -> [1, d_inner]
    z = _projection(
        network, normed, hidden_size, d_inner, weights, f"{prefix}.deltanet_z_proj", dtype=dtype
    )

    # Decay projection (a): [1, hidden] -> [1, num_heads]
    a_raw = _projection(
        network, normed, hidden_size, num_heads, weights, f"{prefix}.deltanet_a_proj", dtype=dtype
    )

    # Beta projection (b): [1, hidden] -> [1, num_heads]
    b_raw = _projection(
        network, normed, hidden_size, num_heads, weights, f"{prefix}.deltanet_b_proj", dtype=dtype
    )

    mixer = _add_deltanet_mixer_rows if sequence_length is None else _add_deltanet_mixer_step
    mixer_output, present_conv, present_ssm = mixer(
        network,
        qkv=qkv,
        a_raw=a_raw,
        b_raw=b_raw,
        conv_state_in=conv_state_in,
        ssm_state_in=ssm_state_in,
        weights=weights,
        prefix=prefix,
        d_inner=d_inner,
        d_conv=d_conv,
        conv_dim=conv_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
    )

    # ===== 10. Gated RMSNorm per-head: weight * norm(output) * silu(z) =====
    # The reference recurrent kernel returns the attention output in the model
    # storage dtype before Qwen3_5RMSNormGated casts it back to FP32.
    recurrent_output = mixer_output
    rows = 1 if sequence_length is not None else -1
    head_shape = (num_heads, head_dim) if sequence_length is not None else (-1, num_heads, head_dim)
    if recurrent_output.dtype != hidden.dtype:
        recurrent_output = network.add_cast(recurrent_output, hidden.dtype).get_output(0)

    # HF norm operates per head_v_dim: reshape to [num_heads, head_dim], norm, reshape back
    deltanet_norm_w = weights[f"{prefix}.deltanet_norm"]

    # Reshape output and z to [num_heads, head_dim] for per-head norm
    output_heads = network.add_shuffle(recurrent_output)
    output_heads.reshape_dims = head_shape
    norm_input = output_heads.get_output(0)
    norm_output_dtype = norm_input.dtype
    if dtype != np.float32:
        norm_input = network.add_cast(norm_input, trt.float32).get_output(0)

    # Per-head RMSNorm with the configured epsilon, as Qwen3_5RMSNormGated
    normalized = graph_ops.add_rms_norm_last_axis(network, norm_input, norm_eps)

    # Reshape back and apply weight
    norm_flat = network.add_shuffle(normalized)
    norm_flat.reshape_dims = (rows, d_inner)
    gamma_t = graph_ops.add_constant(network, (1, d_inner), deltanet_norm_w, dtype=np.float32)
    normed_output = network.add_elementwise(
        norm_flat.get_output(0), gamma_t, trt.ElementWiseOperation.PROD
    )
    normed_output_tensor = normed_output.get_output(0)
    if normed_output_tensor.dtype != norm_output_dtype:
        normed_output_tensor = network.add_cast(normed_output_tensor, norm_output_dtype).get_output(
            0
        )

    # Gate: multiply by silu(z)
    z_activated = graph_ops.add_activation(network, z, "silu", dtype=dtype)
    gated = network.add_elementwise(
        normed_output_tensor, z_activated, trt.ElementWiseOperation.PROD
    )

    # ===== 11. Output projection + residual =====
    out = _projection(
        network, gated.get_output(0), d_inner, hidden_size, weights, f"{prefix}.deltanet_out_proj", dtype=dtype
    )

    residual = network.add_elementwise(hidden, out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual.get_output(0)

    # ===== 12. Post-attention norm + SwiGLU MLP + residual =====
    post_normed = graph_ops.add_rms_norm(
        network,
        hidden_after_attn,
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        eps_tensor,
        dtype=dtype,
    )

    mlp_out = graph_blocks.add_swiglu_mlp(
        network,
        post_normed,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        dtype=dtype,
    )

    mlp_residual = network.add_elementwise(hidden_after_attn, mlp_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": mlp_residual.get_output(0),
        "present_conv": present_conv,
        "present_ssm": present_ssm,
    }


def _add_deltanet_mixer_step(
    network: trt.INetworkDefinition,
    *,
    qkv: trt.ITensor,
    a_raw: trt.ITensor,
    b_raw: trt.ITensor,
    conv_state_in: trt.ITensor,
    ssm_state_in: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    d_inner: int,
    d_conv: int,
    conv_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: np.dtype,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    """Conv step and delta-rule update for one token: ``(output [1, d_inner], conv, ssm)``."""
    qk_dim = num_kv_heads * head_dim  # Q and K dimension before Compact GQA/MQA K/V

    # ===== 3. Conv1d step on QKV =====
    # conv_state_in: [conv_dim, d_conv]
    # qkv: [1, conv_dim] -> [conv_dim, 1]
    qkv_col = network.add_shuffle(qkv)
    qkv_col.reshape_dims = (conv_dim, 1)

    if d_conv > 1:
        slice_layer = network.add_slice(
            conv_state_in, start=(0, 1), shape=(conv_dim, d_conv - 1), stride=(1, 1)
        )
        new_conv_state = network.add_concatenation(
            [slice_layer.get_output(0), qkv_col.get_output(0)]
        )
        new_conv_state.axis = 1
        present_conv = new_conv_state.get_output(0)
    else:
        present_conv = qkv_col.get_output(0)

    conv_w = graph_ops.add_constant(
        network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"], dtype=dtype
    )
    conv_prod = network.add_elementwise(present_conv, conv_w, trt.ElementWiseOperation.PROD)
    conv_sum = network.add_reduce(
        conv_prod.get_output(0), trt.ReduceOperation.SUM, 1 << 1, keep_dims=True
    )
    conv_flat = network.add_shuffle(conv_sum.get_output(0))
    conv_flat.reshape_dims = (1, conv_dim)
    conv_out = graph_ops.add_bias_sum(
        network, conv_flat.get_output(0), conv_dim, weights[f"{prefix}.conv1d_bias"], dtype=dtype
    )
    qkv_activated = graph_ops.add_activation(network, conv_out, "silu", dtype=dtype)

    # ===== 4. Split Q, K, V from activated output =====
    offset = 0
    q_slice = network.add_slice(qkv_activated, start=(0, offset), shape=(1, qk_dim), stride=(1, 1))
    q_raw_t = q_slice.get_output(0)
    offset += qk_dim

    k_slice = network.add_slice(qkv_activated, start=(0, offset), shape=(1, qk_dim), stride=(1, 1))
    k_raw_t = k_slice.get_output(0)
    offset += qk_dim

    v_slice = network.add_slice(qkv_activated, start=(0, offset), shape=(1, d_inner), stride=(1, 1))
    v_raw = v_slice.get_output(0)

    # ===== 5. L2-normalize Q and K =====
    # Reshape to [num_kv_heads, head_dim], normalize per-head, reshape back
    q_heads_in = network.add_shuffle(q_raw_t)
    q_heads_in.reshape_dims = (num_kv_heads, head_dim)
    q_normed = _normalized_heads(network, q_heads_in.get_output(0), 1)

    k_heads_in = network.add_shuffle(k_raw_t)
    k_heads_in.reshape_dims = (num_kv_heads, head_dim)
    k_normed = _normalized_heads(network, k_heads_in.get_output(0), 1)

    # ===== 6. keep compact Q,K from num_kv_heads -> num_heads =====
    heads_per_group = num_heads // num_kv_heads

    if heads_per_group > 1:
        # Q: [num_kv_heads, head_dim] -> [num_kv_heads, 1, head_dim] ->
        #    tile -> [num_kv_heads, heads_per_group, head_dim] ->
        #    [num_heads, head_dim]
        q_3d = network.add_shuffle(q_normed)
        q_3d.reshape_dims = (num_kv_heads, 1, head_dim)
        tile_ones = graph_ops.add_constant(
            network,
            (1, heads_per_group, 1),
            np.ones((1, heads_per_group, 1), dtype=np.float32),
            dtype=np.float32,
        )
        q_tiled = network.add_elementwise(
            q_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD
        )
        q_expanded_s = network.add_shuffle(q_tiled.get_output(0))
        q_expanded_s.reshape_dims = (num_heads, head_dim)
        q_expanded = q_expanded_s.get_output(0)

        k_3d = network.add_shuffle(k_normed)
        k_3d.reshape_dims = (num_kv_heads, 1, head_dim)
        k_tiled = network.add_elementwise(
            k_3d.get_output(0), tile_ones, trt.ElementWiseOperation.PROD
        )
        k_t_s = network.add_shuffle(k_tiled.get_output(0))
        k_t_s.reshape_dims = (num_heads, head_dim)
        k_t = k_t_s.get_output(0)
    else:
        q_expanded = q_normed
        k_t = k_normed

    # V: [1, d_inner] -> [num_heads, head_dim]
    v_heads = network.add_shuffle(v_raw)
    v_heads.reshape_dims = (num_heads, head_dim)
    v_t = v_heads.get_output(0)

    # ===== 7. Compute decay: -exp(A_log) * softplus(a + dt_bias) per head =====
    # Transformers performs the decay and recurrent rule in FP32 even when
    # the model projections use FP16.  Keeping these tensors in the model
    # storage dtype quantizes the persistent state again on every token.
    recurrent_dtype = trt.float32

    def recurrent_cast(tensor: trt.ITensor) -> trt.ITensor:
        if tensor.dtype == recurrent_dtype:
            return tensor
        return network.add_cast(tensor, recurrent_dtype).get_output(0)

    # A: [num_heads] (precomputed as -exp(A_log))
    A_const = graph_ops.add_constant(
        network, (1, num_heads), weights[f"{prefix}.A"], dtype=np.float32
    )

    # dt_bias: [num_heads]
    dt_bias_const = graph_ops.add_constant(
        network, (1, num_heads), weights[f"{prefix}.dt_bias"], dtype=np.float32
    )
    a_biased = network.add_elementwise(
        recurrent_cast(a_raw), dt_bias_const, trt.ElementWiseOperation.SUM
    )

    # softplus(a + dt_bias), without overflow for large arguments
    a_softplus = graph_ops.add_softplus(network, a_biased.get_output(0))

    # decay = A * softplus(...) per head: [1, num_heads]
    decay_flat = network.add_elementwise(A_const, a_softplus, trt.ElementWiseOperation.PROD)
    # exp(decay) for the state update: [1, num_heads] -> [num_heads, 1, 1]
    decay_reshaped = network.add_shuffle(decay_flat.get_output(0))
    decay_reshaped.reshape_dims = (num_heads, 1, 1)
    decay_exp = network.add_unary(decay_reshaped.get_output(0), trt.UnaryOperation.EXP)

    # ===== 8. Compute beta: sigmoid(b) per head =====
    # b_raw: [1, num_heads]
    beta = network.add_activation(b_raw, trt.ActivationType.SIGMOID)
    # [1, num_heads] -> [num_heads, 1]
    beta_reshaped = network.add_shuffle(recurrent_cast(beta.get_output(0)))
    beta_reshaped.reshape_dims = (num_heads, 1)

    # ===== 9. Delta rule state update =====
    # HF state layout: [H, K_dim, V_dim]
    # ssm_state_in: [num_heads, head_dim, head_dim]  (K on axis -2, V on axis -1)
    # k: [num_heads, head_dim], q: [num_heads, head_dim], v: [num_heads, head_dim]

    # 9a. Decay state first: state = state * exp(g)
    decayed_state = network.add_elementwise(
        decay_exp.get_output(0), recurrent_cast(ssm_state_in), trt.ElementWiseOperation.PROD
    )

    # 9b. kv_mem = state^T @ k: read old value for this key
    # [H, V, K] @ [H, K, 1] = [H, V, 1]  (transpose state to swap K/V axes)
    k_recurrent = recurrent_cast(k_t)
    v_recurrent = recurrent_cast(v_t)
    q_recurrent = recurrent_cast(q_expanded)

    k_col = network.add_shuffle(k_recurrent)
    k_col.reshape_dims = (num_heads, head_dim, 1)
    kv_old_3d = network.add_matrix_multiply(
        decayed_state.get_output(0),
        trt.MatrixOperation.TRANSPOSE,
        k_col.get_output(0),
        trt.MatrixOperation.NONE,
    )
    kv_old = network.add_shuffle(kv_old_3d.get_output(0))
    kv_old.reshape_dims = (num_heads, head_dim)

    # 9c. delta = (v - kv_mem) * beta
    v_minus_old = network.add_elementwise(
        v_recurrent, kv_old.get_output(0), trt.ElementWiseOperation.SUB
    )
    v_delta = network.add_elementwise(
        v_minus_old.get_output(0), beta_reshaped.get_output(0), trt.ElementWiseOperation.PROD
    )

    # 9d. state_new = decayed_state + outer(k, delta)
    # outer: k[:, :, None] * delta[:, None, :] = [H, K, 1] @ [H, 1, V] = [H, K, V]
    k_col2 = network.add_shuffle(k_recurrent)
    k_col2.reshape_dims = (num_heads, head_dim, 1)
    v_delta_row = network.add_shuffle(v_delta.get_output(0))
    v_delta_row.reshape_dims = (num_heads, 1, head_dim)
    outer_prod = network.add_matrix_multiply(
        k_col2.get_output(0),
        trt.MatrixOperation.NONE,
        v_delta_row.get_output(0),
        trt.MatrixOperation.NONE,
    )

    new_state = network.add_elementwise(
        decayed_state.get_output(0), outer_prod.get_output(0), trt.ElementWiseOperation.SUM
    )
    present_ssm = new_state.get_output(0)

    # 9e. output = state_new^T @ (q * scale)
    # HF applies: query *= 1/sqrt(k_dim)
    q_scale = graph_ops.add_constant(
        network, (1, 1), np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32), dtype=np.float32
    )
    q_scaled = network.add_elementwise(q_recurrent, q_scale, trt.ElementWiseOperation.PROD)
    # [H, V, K] @ [H, K, 1] = [H, V, 1]
    q_col = network.add_shuffle(q_scaled.get_output(0))
    q_col.reshape_dims = (num_heads, head_dim, 1)
    output_3d = network.add_matrix_multiply(
        present_ssm, trt.MatrixOperation.TRANSPOSE, q_col.get_output(0), trt.MatrixOperation.NONE
    )
    output_flat = network.add_shuffle(output_3d.get_output(0))
    output_flat.reshape_dims = (1, d_inner)


    return output_flat.get_output(0), present_conv, present_ssm


def _add_deltanet_mixer_rows(
    network: trt.INetworkDefinition,
    *,
    qkv: trt.ITensor,
    a_raw: trt.ITensor,
    b_raw: trt.ITensor,
    conv_state_in: trt.ITensor,
    ssm_state_in: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    d_inner: int,
    d_conv: int,
    conv_dim: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: np.dtype,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    """Conv and delta rule over ``S`` rows in one launch: ``(output [S, d_inner], conv, ssm)``.

    Same arithmetic as :func:`_add_deltanet_mixer_step` applied token by token;
    the recurrence runs as one chunk (see :mod:`.deltanet_rows`).
    """
    qk_dim = num_kv_heads * head_dim

    def shuffle(tensor, reshape=None, transpose=None):
        layer = network.add_shuffle(tensor)
        if transpose is not None:
            layer.first_transpose = transpose
        if reshape is not None:
            layer.reshape_dims = reshape
        return layer.get_output(0)

    def fp32(tensor):
        return tensor if tensor.dtype == trt.float32 else network.add_cast(tensor, trt.float32).get_output(0)

    conv_w = graph_ops.add_constant(
        network, (conv_dim, d_conv), weights[f"{prefix}.conv1d_weight"], dtype=dtype
    )
    conv_columns, present_conv = deltanet_rows.add_short_conv_rows(
        network,
        columns=shuffle(qkv, transpose=(1, 0)),
        conv_state=conv_state_in,
        weight=conv_w,
        channels=conv_dim,
        kernel=d_conv,
    )
    conv_out = graph_ops.add_bias_sum(
        network, shuffle(conv_columns, transpose=(1, 0)), conv_dim,
        weights[f"{prefix}.conv1d_bias"], dtype=dtype,
    )
    activated = graph_ops.add_activation(network, conv_out, "silu", dtype=dtype)
    rows = deltanet_rows.row_count(network, activated)

    def columns(offset, width):
        return deltanet_rows.dynamic_slice(network, activated, (0, offset), (rows, width))

    q_normed = _normalized_heads(network, shuffle(columns(0, qk_dim), reshape=(-1, num_kv_heads, head_dim)), 2)
    k_normed = _normalized_heads(network, shuffle(columns(qk_dim, qk_dim), reshape=(-1, num_kv_heads, head_dim)), 2)
    heads_per_group = num_heads // num_kv_heads

    def expand_heads(tensor):
        # [S, kv_heads, D] -> [S, heads, D], head = kv_head * heads_per_group + g.
        if heads_per_group == 1:
            return tensor
        tile = graph_ops.add_constant(
            network, (1, 1, heads_per_group, 1),
            np.ones((1, 1, heads_per_group, 1), dtype=np.float32), dtype=np.float32,
        )
        tiled = network.add_elementwise(
            shuffle(tensor, reshape=(-1, num_kv_heads, 1, head_dim)), tile,
            trt.ElementWiseOperation.PROD,
        ).get_output(0)
        return shuffle(tiled, reshape=(-1, num_heads, head_dim))

    def head_major(tensor):
        return shuffle(fp32(tensor), transpose=(1, 0, 2))  # [H, S, D]

    q_scale = graph_ops.add_constant(
        network, (1, 1, 1), np.array([[[1.0 / np.sqrt(head_dim)]]], dtype=np.float32),
        dtype=np.float32,
    )
    q_heads = network.add_elementwise(
        head_major(expand_heads(q_normed)), q_scale, trt.ElementWiseOperation.PROD
    ).get_output(0)
    k_heads = head_major(expand_heads(k_normed))
    v_heads = head_major(shuffle(columns(2 * qk_dim, d_inner), reshape=(-1, num_heads, head_dim)))

    # Decay g = A * softplus(a + dt_bias) and beta = sigmoid(b), FP32, [H, S].
    a_const = graph_ops.add_constant(network, (1, num_heads), weights[f"{prefix}.A"], dtype=np.float32)
    dt_bias = graph_ops.add_constant(
        network, (1, num_heads), weights[f"{prefix}.dt_bias"], dtype=np.float32
    )
    biased = network.add_elementwise(fp32(a_raw), dt_bias, trt.ElementWiseOperation.SUM).get_output(0)
    softplus = graph_ops.add_softplus(network, biased)
    decay = network.add_elementwise(a_const, softplus, trt.ElementWiseOperation.PROD).get_output(0)
    beta = fp32(network.add_activation(b_raw, trt.ActivationType.SIGMOID).get_output(0))

    outputs, present_ssm = deltanet_rows.add_chunked_delta_rule(
        network,
        q=q_heads,
        k=k_heads,
        v=v_heads,
        g=shuffle(decay, transpose=(1, 0)),
        beta=shuffle(beta, transpose=(1, 0)),
        state=fp32(ssm_state_in),
        num_heads=num_heads,
        head_dim=head_dim,
    )
    return shuffle(outputs, transpose=(1, 0, 2), reshape=(-1, d_inner)), present_conv, present_ssm



def _add_full_attention_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attn_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_embedding_dim: int,
    max_cache_length: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
    sequence_length: int | None = 1,
) -> dict[str, trt.ITensor]:
    """Add one full self-attention layer with output gating.

    Qwen3.5 full attention has:
      - QK-norm with (1+weight) centering
      - Partial RoPE (25% of dims)
      - Output gating: context * sigmoid(gate) BEFORE o_proj
      - SwiGLU MLP after attention

    ``sequence_length`` is the query row count, or None for ``S`` runtime rows
    attending over the cache plus their own causal prefix.

    Returns: {hidden, present_k, present_v}
    """
    rows = -1 if sequence_length is None else sequence_length
    kv_rows = None if sequence_length is None else max_cache_length + sequence_length

    # Pre-attention norm
    normed = graph_ops.add_rms_norm(
        network, hidden, hidden_size, weights[f"{prefix}.input_norm"], eps_tensor, dtype=dtype
    )

    # QKV projections
    q = _projection(
        network, normed, hidden_size, attn_size, weights, f"{prefix}.w_q", dtype=dtype
    )
    k = _projection(
        network, normed, hidden_size, kv_attention_size, weights, f"{prefix}.w_k", dtype=dtype
    )
    v = _projection(
        network, normed, hidden_size, kv_attention_size, weights, f"{prefix}.w_v", dtype=dtype
    )

    # Per-head QK norm
    q_norm = weights.get(f"{prefix}.q_norm")
    if q_norm is not None:
        q = graph_ops.add_rms_norm_per_head(
            network, q, num_heads, head_dim, q_norm, eps_tensor, dtype=dtype,
            sequence_length=sequence_length,
        )
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_kv_heads, head_dim, k_norm, eps_tensor, dtype=dtype,
            sequence_length=sequence_length,
        )

    # Native RoPE
    q = graph_ops.add_apply_rope_native(
        network,
        q,
        num_heads,
        head_dim,
        cos_half_tensor,
        sin_half_tensor,
        position_id,
        rotary_embedding_dim,
        sequence_length=sequence_length,
    )
    k = graph_ops.add_apply_rope_native(
        network,
        k,
        num_kv_heads,
        head_dim,
        cos_half_tensor,
        sin_half_tensor,
        position_id,
        rotary_embedding_dim,
        sequence_length=sequence_length,
    )

    present_k, present_v = k, v

    # Reshape K, V for concatenation
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (rows, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (rows, kv_attention_size)

    # Concatenate with cache
    all_k = network.add_concatenation([cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v_reshape.get_output(0)])
    all_v.axis = 0

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    context_flat = graph_ops.add_attention_from_rows(
        network,
        q,
        all_k.get_output(0),
        all_v.get_output(0),
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        q_seq=sequence_length,
        kv_seq=kv_rows,
        mask=mask_4d,
    )

    # Gate: applied BEFORE o_proj (HF order)
    gate_attn_w = weights.get(f"{prefix}.w_gate_attn")
    attn_out = context_flat
    if gate_attn_w is not None:
        gate = _projection(
            network,
            normed,
            hidden_size,
            attn_size,
            weights,
            f"{prefix}.w_gate_attn",
            dtype=dtype,
        )
        gate_sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
        gated = network.add_elementwise(
            attn_out, gate_sigmoid.get_output(0), trt.ElementWiseOperation.PROD
        )
        attn_out = gated.get_output(0)

    # Output projection (AFTER gate)
    attn_out = _projection(
        network, attn_out, attn_size, hidden_size, weights, f"{prefix}.w_o", dtype=dtype
    )

    residual = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
    hidden_after_attn = residual.get_output(0)

    # Post-attention norm + SwiGLU MLP + residual
    post_normed = graph_ops.add_rms_norm(
        network,
        hidden_after_attn,
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        eps_tensor,
        dtype=dtype,
    )

    mlp_out = graph_blocks.add_swiglu_mlp(
        network,
        post_normed,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        dtype=dtype,
    )

    mlp_residual = network.add_elementwise(hidden_after_attn, mlp_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": mlp_residual.get_output(0),
        "present_k": present_k,
        "present_v": present_v,
    }
