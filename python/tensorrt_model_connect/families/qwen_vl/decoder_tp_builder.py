# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Qwen-VL text decoder builder.

This mirrors the single-device Qwen-VL decoder paths while adding only tensor
parallel projection sharding and distributed ALL_REDUCE joins. It keeps the VL
``input_embed`` override used by Qwen2.5-VL and optionally injects Qwen3-VL
DeepStack features after attention residuals.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_blocks
from . import graph_ops
from ...parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
    shard_standard_decoder_weights,
)

trt = trt_compat.get_trt()
_NATIVE_BUILDER_WORKSPACE_BYTES = 16 << 30

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig


def _mark_debug_output(network: trt.INetworkDefinition, tensor: trt.ITensor, name: str) -> None:
    cast = network.add_cast(tensor, trt.float32)
    out = cast.get_output(0)
    out.name = name
    network.mark_output(out)


def _ensure_tp_metadata(config: "ModelConfig", weights: "WeightDict") -> "WeightDict":
    if "_kv_attention_size" in weights:
        return weights
    copied = type(weights)(weights)
    copied["_kv_attention_size"] = int(weights.get("_attention_size", config.attention_size))
    return copied


def _apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype = np.float32,
    eps: float | None = None,
) -> trt.ITensor:
    return graph_blocks.apply_norm(
        network, inp, hidden_size, gamma, beta, eps_tensor, norm_type,
        dtype=dtype, eps=eps)


def build_qwen_vl_tp_decoder_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "bf16",
    quant_ctx=None,
    norm_type: str = "rmsnorm",
    mlp_type: str = "swiglu",
    position_type: str = "rope",
    activation: str = "silu",
    partial_rotary_factor: float = 1.0,
    parallel_residual: bool = False,
    scale_attn_weights: bool = True,
    embed_input: bool = True,
    deepstack_num_levels: int = 0,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    """Build one rank-local Qwen-VL text decoder engine."""
    if precision != "bf16" or quant_ctx is not None or debug_layer_outputs:
        raise ValueError("native Qwen-VL TP requires BF16 without quant/debug")
    if mlp_type != "swiglu":
        raise NotImplementedError("Qwen-VL tensor-parallel builds support SwiGLU MLPs only")

    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError("Qwen-VL tensor-parallel builder requires an enabled parallel config")
    if parallel.rank < 0:
        raise ValueError("Qwen-VL tensor-parallel engine build requires a concrete rank")

    weights = _ensure_tp_metadata(config, weights)
    weights = shard_standard_decoder_weights(config, weights, parallel)

    attention_size = int(weights.get("_attention_size", config.attention_size))
    mlp_size = int(weights.get("_mlp_size", config.intermediate_size))
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads // parallel.tp_size
    num_kv_heads = config.num_key_value_heads // parallel.tp_size
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    from .build_routing import native_mrope_settings

    mrope_section, mrope_interleaved = native_mrope_settings(config)
    active_inv_freq = graph_ops.make_native_active_rope_inv_freq(
        head_dim, config.rope_theta, partial_rotary_factor)

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, _NATIVE_BUILDER_WORKSPACE_BYTES)
    multi_device_preview = getattr(
        getattr(trt, "PreviewFeature", object()), "MULTI_DEVICE_ENGINE", None)
    if multi_device_preview is not None:
        trt_config.set_preview_feature(multi_device_preview, True)

    if precision == "fp16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == "bf16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.bfloat16
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32

    token_id = network.add_input("token_id", trt.int32, (-1,))
    _position_id = network.add_input("position_id", trt.int32, (-1,))
    mrope_position_ids = network.add_input(
        "mrope_position_ids", trt.int32, (3, -1))
    cache_write_indices = network.add_input(
        "cache_write_indices", trt.int32, (1,))
    key_value_lengths = network.add_input(
        "key_value_lengths", trt.int32, (1,))

    input_embed_tensor = None
    use_input_embed_tensor = None
    if embed_input:
        input_embed_tensor = network.add_input("input_embed", trt.float32, (-1, hidden))
        use_input_embed_tensor = network.add_input("use_input_embed", trt.float32, (-1, 1))

    deepstack_embed_inputs: list[trt.ITensor] = []
    deepstack_active_tensor = None
    if deepstack_num_levels > 0:
        for level in range(deepstack_num_levels):
            deepstack_embed_inputs.append(network.add_input(
                f"deepstack_embed_{level}", trt.float32, (-1, hidden)))
        deepstack_active_tensor = network.add_input(
            "deepstack_active", trt.float32, (-1, 1))

    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        cache_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            work_trt_dtype,
            (1, num_kv_heads, max_cache_length, head_dim),
        ))
        cache_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            work_trt_dtype,
            (1, num_kv_heads, max_cache_length, head_dim),
        ))

    # TP bundles store one rank-local engine section per rank. Keep the
    # existing bundle layout while giving that engine distinct prefill and
    # decode execution contexts via two optimization profiles.
    max_prefill_length = max(1, min(max_cache_length, 32768))
    opt_prefill_length = max(1, min(max_prefill_length, 128))

    def _add_profile(min_sq: int, opt_sq: int, max_sq: int) -> None:
        profile = builder.create_optimization_profile()
        profile.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        profile.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        profile.set_shape(
            "mrope_position_ids",
            (3, min_sq), (3, opt_sq), (3, max_sq),
        )
        if embed_input:
            profile.set_shape(
                "input_embed",
                (min_sq, hidden), (opt_sq, hidden), (max_sq, hidden),
            )
            profile.set_shape(
                "use_input_embed", (min_sq, 1), (opt_sq, 1), (max_sq, 1),
            )
        if deepstack_num_levels > 0:
            for level in range(deepstack_num_levels):
                profile.set_shape(
                    f"deepstack_embed_{level}",
                    (min_sq, hidden), (opt_sq, hidden), (max_sq, hidden),
                )
            profile.set_shape(
                "deepstack_active", (min_sq, 1), (opt_sq, 1), (max_sq, 1),
            )
        trt_config.add_optimization_profile(profile)

    _add_profile(1, opt_prefill_length, max_prefill_length)
    _add_profile(1, 1, 1)

    def _cast_work_dtype(tensor: trt.ITensor) -> trt.ITensor:
        if tensor.dtype == work_trt_dtype:
            return tensor
        return network.add_cast(tensor, work_trt_dtype).get_output(0)

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)

    cos_half_tensor = None
    sin_half_tensor = None
    rotary_embedding_dim = int(head_dim * partial_rotary_factor)
    if position_type == "rope":
        graph_ops.validate_native_rope_dim(rotary_embedding_dim)
        cos_half_tensor, sin_half_tensor = graph_ops.add_active_mrope_cache(
            network,
            mrope_position_ids,
            active_inv_freq,
            mrope_section,
            work_trt_dtype,
            mrope_interleaved=mrope_interleaved,
        )
    else:
        raise NotImplementedError("Qwen-VL tensor-parallel builds require RoPE")

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype),
        dtype=work_np_dtype)
    attn_scale = (1.0 / np.sqrt(max(head_dim, 1))) if scale_attn_weights else 1.0

    token_embed = network.add_gather(embedding_table, token_id, 0).get_output(0)
    if embed_input and input_embed_tensor is not None and use_input_embed_tensor is not None:
        token_embed_for_math = _cast_work_dtype(token_embed)
        flag_for_math = use_input_embed_tensor
        if work_trt_dtype != trt.float32:
            flag_for_math = network.add_cast(flag_for_math, work_trt_dtype).get_output(0)
        one_const = graph_ops.add_constant(
            network, (1, 1), np.array([1.0], dtype=work_np_dtype),
            dtype=work_np_dtype)
        one_const = _cast_work_dtype(one_const)
        inv_flag = network.add_elementwise(
            one_const, flag_for_math, trt.ElementWiseOperation.SUB).get_output(0)
        tok_part = network.add_elementwise(
            inv_flag, token_embed_for_math, trt.ElementWiseOperation.PROD).get_output(0)
        input_embed_for_math = input_embed_tensor
        if input_embed_for_math.dtype != work_trt_dtype:
            input_embed_for_math = network.add_cast(
                input_embed_for_math, work_trt_dtype).get_output(0)
        embed_part = network.add_elementwise(
            flag_for_math, input_embed_for_math, trt.ElementWiseOperation.PROD).get_output(0)
        hidden_state = network.add_elementwise(
            tok_part, embed_part, trt.ElementWiseOperation.SUM).get_output(0)
    else:
        hidden_state = token_embed

    if hidden_state.dtype != work_trt_dtype:
        hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_tp_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            attention_scale=attn_scale,
            eps_tensor=eps_tensor,
            eps=config.rms_norm_eps,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            attention_size=attention_size,
            kv_attention_size=kv_attention_size,
            mlp_size=mlp_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            norm_type=norm_type,
            activation=activation,
            parallel_residual=parallel_residual,
            dtype=work_np_dtype,
            quant_ctx=quant_ctx,
            cos_half_tensor=cos_half_tensor,
            sin_half_tensor=sin_half_tensor,
            rotary_embedding_dim=rotary_embedding_dim,
            tp_size=parallel.tp_size,
            deepstack_embed=(
                deepstack_embed_inputs[layer_idx]
                if layer_idx < len(deepstack_embed_inputs)
                else None
            ),
            deepstack_active=deepstack_active_tensor,
            cache_write_indices=cache_write_indices,
            key_value_lengths=key_value_lengths,
            sequence_length=None,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])
        if debug_layer_outputs:
            _mark_debug_output(network, result["post_attn"], f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(
            network, hidden_state, hidden, final_norm,
            weights.get("final_norm_beta"), eps_tensor, norm_type,
            dtype=work_np_dtype, eps=config.rms_norm_eps)

    # Only the final prompt row is needed for sampling. Dynamic slicing keeps
    # the output contract [1, vocab] identical for both TP profiles.
    hidden_shape = network.add_shape(hidden_state).get_output(0)
    one_hidden = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    last_start = network.add_elementwise(
        hidden_shape, one_hidden, trt.ElementWiseOperation.SUB).get_output(0)
    last_size = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64), dtype=np.int64)
    last_slice = network.add_slice(
        hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
    last_slice.set_input(1, last_start)
    last_slice.set_input(2, last_size)
    last_hidden = last_slice.get_output(0)

    out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
    logits = graph_ops.add_matmul_rhs_constant(
        network, last_hidden, hidden, out_vocab, weights["w_out"], dtype=work_np_dtype)
    lm_bias = weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(network, logits, out_vocab, lm_bias, dtype=work_np_dtype)
    else:
        logits = graph_ops.add_bias_sum(
            network, logits, out_vocab, np.zeros(out_vocab, dtype=work_np_dtype),
            dtype=work_np_dtype)
    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for i in range(num_layers):
        present_k_outputs[i].name = graph_ops.layer_tensor_name("present_k", i)
        present_v_outputs[i].name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(present_k_outputs[i])
        network.mark_output(present_v_outputs[i])

    if verbose:
        print(
            f"[trtmc build] Building Qwen-VL TP rank "
            f"{parallel.rank}/{parallel.tp_size} ({num_layers}L, h={hidden}, "
            f"local_heads={num_heads}, local_attn={attention_size}, "
            f"local_mlp={mlp_size}, cache={max_cache_length}, "
            f"prefill_chunk={max_prefill_length}, precision={precision}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Qwen-VL tensor-parallel decoder build failed")
    return bytes(plan)


def _add_tp_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_scale: float | None,
    eps_tensor: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    mlp_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    norm_type: str,
    activation: str,
    parallel_residual: bool,
    dtype: np.dtype,
    quant_ctx,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    rotary_embedding_dim: int,
    tp_size: int,
    cache_write_indices: trt.ITensor,
    key_value_lengths: trt.ITensor,
    deepstack_embed: trt.ITensor | None = None,
    deepstack_active: trt.ITensor | None = None,
    eps: float | None = None,
    sequence_length: int | None = 1,
) -> dict[str, trt.ITensor]:
    attn = graph_blocks.add_attention_block(
        network, hidden, cache_k, cache_v,
        weights=weights, prefix=prefix,
        hidden_size=hidden_size, attention_size=attention_size,
        kv_attention_size=kv_attention_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        attention_scale=attention_scale,
        eps_tensor=eps_tensor, eps=eps,
        norm_type=norm_type,
        dtype=dtype,
        quant_ctx=quant_ctx,
        layer_prefix=prefix,
        cos_half_tensor=cos_half_tensor,
        sin_half_tensor=sin_half_tensor,
        rotary_embedding_dim=rotary_embedding_dim,
        cache_write_indices=cache_write_indices,
        key_value_lengths=key_value_lengths,
        sequence_length=sequence_length,
    )
    attn_out = add_all_reduce_sum(network, attn["attn_out"], tp_size)

    if parallel_residual:
        post_attn_norm_w = weights.get(f"{prefix}.post_attn_norm")
        if post_attn_norm_w is not None:
            norm2 = _apply_norm(
                network, hidden, hidden_size,
                post_attn_norm_w,
                weights.get(f"{prefix}.post_attn_norm_beta"),
                eps_tensor, norm_type, dtype=dtype, eps=eps)
        else:
            norm2 = attn["normed"]
    else:
        residual1 = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM).get_output(0)
        if deepstack_embed is not None and deepstack_active is not None:
            active = deepstack_active
            deepstack_for_math = deepstack_embed
            if deepstack_for_math.dtype != residual1.dtype:
                deepstack_for_math = network.add_cast(
                    deepstack_for_math, residual1.dtype).get_output(0)
            if active.dtype != deepstack_for_math.dtype:
                active = network.add_cast(active, deepstack_for_math.dtype).get_output(0)
            zero = graph_ops.add_constant(
                network, (1, 1), np.zeros((1, 1), dtype=np.float32),
                dtype=np.float32)
            if zero.dtype != deepstack_for_math.dtype:
                zero = network.add_cast(zero, deepstack_for_math.dtype).get_output(0)
            enabled = network.add_elementwise(
                active, zero, trt.ElementWiseOperation.GREATER).get_output(0)
            ds_scaled = network.add_select(
                enabled, deepstack_for_math, zero).get_output(0)
            residual1 = network.add_elementwise(
                residual1, ds_scaled, trt.ElementWiseOperation.SUM).get_output(0)
        norm2 = _apply_norm(
            network, residual1, hidden_size,
            weights[f"{prefix}.post_attn_norm"],
            weights.get(f"{prefix}.post_attn_norm_beta"),
            eps_tensor, norm_type, dtype=dtype, eps=eps)

    mlp_out = graph_blocks.add_swiglu_mlp(
        network, norm2, weights=weights, prefix=prefix,
        hidden_size=hidden_size, mlp_size=mlp_size, dtype=dtype,
        quant_ctx=quant_ctx, layer_prefix=prefix)
    mlp_out = add_all_reduce_sum(network, mlp_out, tp_size)

    if parallel_residual:
        sum_attn = network.add_elementwise(
            hidden, attn_out, trt.ElementWiseOperation.SUM).get_output(0)
        residual2 = network.add_elementwise(
            sum_attn, mlp_out, trt.ElementWiseOperation.SUM).get_output(0)
        post_attn_tensor = sum_attn
    else:
        residual2 = network.add_elementwise(
            residual1, mlp_out, trt.ElementWiseOperation.SUM).get_output(0)
        post_attn_tensor = residual1

    return {
        "hidden": residual2,
        "post_attn": post_attn_tensor,
        "present_k": attn["present_k"],
        "present_v": attn["present_v"],
    }
