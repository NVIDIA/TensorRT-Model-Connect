# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel DeepSeek-OCR text decoder builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import graph_ops
from .prefill_config import sequence_prefill_profile_lengths
from ...parallel_config import add_all_reduce_sum, normalize_parallel_config
from .norm_utils import _apply_norm

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from ...parallel_config import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_deepseek_ocr_tp(
    config: "ModelConfig",
    weights: "WeightDict",
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError(
            "DeepSeek-OCR tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    if int(config.num_attention_heads) % tp != 0:
        raise ValueError(
            "DeepSeek-OCR tensor parallel requires num_attention_heads "
            f"divisible by tp_size ({config.num_attention_heads} vs {tp})")
    if int(config.num_key_value_heads) % tp != 0:
        raise ValueError(
            "DeepSeek-OCR tensor parallel requires num_key_value_heads "
            f"divisible by tp_size ({config.num_key_value_heads} vs {tp})")

    checks = {
        "attention_size": int(weights.get(
            "_attention_size", config.attention_size)),
        "kv_attention_size": int(weights.get(
            "_kv_attention_size", config.num_key_value_heads * config.head_dim)),
        "dense_intermediate_size": int(config.intermediate_size),
        "moe_intermediate_size": int(weights["_moe_intermediate_size"]),
        "shared_intermediate_size": int(weights["_shared_intermediate_size"]),
    }
    for name, value in checks.items():
        if value % tp != 0:
            raise ValueError(
                "DeepSeek-OCR tensor parallel requires "
                f"{name} divisible by tp_size ({value} vs {tp})")


def shard_deepseek_ocr_weights(
    config: "ModelConfig",
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local DeepSeek-OCR decoder weights."""
    _validate_deepseek_ocr_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue

        if key.endswith((".w_q", ".w_k", ".w_v", ".w_gate", ".w_up")):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".w_o", ".w_down")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        else:
            out[key] = value

    out["_attention_size"] = (
        int(weights["_attention_size"]) // parallel.tp_size)
    out["_kv_attention_size"] = (
        int(weights["_kv_attention_size"]) // parallel.tp_size)
    out["_moe_intermediate_size"] = (
        int(weights["_moe_intermediate_size"]) // parallel.tp_size)
    out["_shared_intermediate_size"] = (
        int(weights["_shared_intermediate_size"]) // parallel.tp_size)
    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def _add_swiglu_local(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    *,
    dtype=np.float16,
) -> trt.ITensor:
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_gate, dtype=dtype)
    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_up, dtype=dtype)
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    return graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), intermediate_size, hidden_size,
        w_down, dtype=dtype)


def _add_swiglu_tp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    tp_size: int,
    *,
    dtype=np.float16,
) -> trt.ITensor:
    local = _add_swiglu_local(
        network, inp, hidden_size, intermediate_size,
        w_gate, w_up, w_down, dtype=dtype)
    return add_all_reduce_sum(network, local, tp_size)


def _add_moe_tp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    n_routed_experts: int,
    moe_intermediate: int,
    num_experts_per_tok: int,
    shared_intermediate: int,
    tp_size: int,
    *,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    dtype=np.float16,
) -> trt.ITensor:
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, n_routed_experts,
        weights[f"{prefix}.router"], dtype=dtype)
    softmax = network.add_softmax(router_logits)
    softmax.axes = 1 << 1
    topk = network.add_topk(
        softmax.get_output(0), trt.TopKOperation.MAX,
        num_experts_per_tok, 1 << 1)
    top_values = topk.get_output(0)
    top_indices = topk.get_output(1)

    if norm_topk_prob:
        denominator = network.add_reduce(
            top_values, trt.ReduceOperation.SUM, 1 << 1,
            keep_dims=True).get_output(0)
        selected_weights = network.add_elementwise(
            top_values, denominator,
            trt.ElementWiseOperation.DIV).get_output(0)
    elif routed_scaling_factor != 1.0:
        scale = graph_ops.add_constant(
            network, (1, 1),
            np.array([[routed_scaling_factor]], dtype=np.float16),
            dtype=np.float16)
        if scale.dtype != top_values.dtype:
            scale = network.add_cast(scale, top_values.dtype).get_output(0)
        selected_weights = network.add_elementwise(
            top_values, scale,
            trt.ElementWiseOperation.PROD).get_output(0)
    else:
        selected_weights = top_values

    routed_local = None
    for expert_idx in range(n_routed_experts):
        expert_output = _add_swiglu_local(
            network, inp, hidden_size, moe_intermediate,
            weights[f"{prefix}.expert.{expert_idx}.w_gate"],
            weights[f"{prefix}.expert.{expert_idx}.w_up"],
            weights[f"{prefix}.expert.{expert_idx}.w_down"],
            dtype=dtype)
        expert_index = graph_ops.add_constant(
            network, (1, 1), np.array([[expert_idx]], dtype=np.int32),
            dtype=np.int32)
        selected = network.add_elementwise(
            top_indices, expert_index,
            trt.ElementWiseOperation.EQUAL).get_output(0)
        selected = network.add_cast(
            selected, selected_weights.dtype).get_output(0)
        token_weights = network.add_elementwise(
            selected_weights, selected,
            trt.ElementWiseOperation.PROD).get_output(0)
        expert_weight = network.add_reduce(
            token_weights, trt.ReduceOperation.SUM, 1 << 1,
            keep_dims=True).get_output(0)
        scaled = network.add_elementwise(
            expert_output, expert_weight,
            trt.ElementWiseOperation.PROD).get_output(0)
        if routed_local is None:
            routed_local = scaled
        else:
            routed_local = network.add_elementwise(
                routed_local, scaled,
                trt.ElementWiseOperation.SUM).get_output(0)

    if routed_local is None:
        raise ValueError("DeepSeek-OCR requires at least one routed expert")
    shared_local = _add_swiglu_local(
        network, inp, hidden_size, shared_intermediate,
        weights[f"{prefix}.shared.w_gate"],
        weights[f"{prefix}.shared.w_up"],
        weights[f"{prefix}.shared.w_down"],
        dtype=dtype)
    local_total = network.add_elementwise(
        routed_local, shared_local,
        trt.ElementWiseOperation.SUM).get_output(0)
    return add_all_reduce_sum(network, local_total, tp_size)


def _add_decoder_layer_tp(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    cache_write_indices: trt.ITensor,
    key_value_lengths: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    eps_tensor: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_moe_layer: bool,
    n_routed_experts: int,
    num_experts_per_tok: int,
    moe_intermediate: int,
    shared_intermediate: int,
    dense_intermediate: int,
    tp_size: int,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    dtype=np.float16,
) -> dict[str, trt.ITensor]:
    norm1 = _apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], None, eps_tensor,
        "rmsnorm", dtype=dtype)
    q = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, attention_size,
        weights[f"{prefix}.w_q"], dtype=dtype)
    k = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_k"], dtype=dtype)
    v = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_v"], dtype=dtype)
    q = graph_ops.add_apply_rope_native(
        network, q, num_heads, head_dim,
        cos_half_tensor, sin_half_tensor, None, head_dim,
        sequence_length=None)
    k = graph_ops.add_apply_rope_native(
        network, k, num_kv_heads, head_dim,
        cos_half_tensor, sin_half_tensor, None, head_dim,
        sequence_length=None)

    native_attention = graph_ops.add_native_kv_cache_attention_from_rows(
        network, q, k, v, cache_k, cache_v,
        cache_write_indices, key_value_lengths,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, q_seq=None, tag=f"{prefix}.attn")
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, native_attention["context"],
        attention_size, hidden_size, weights[f"{prefix}.w_o"],
        dtype=dtype)
    attn_out = add_all_reduce_sum(network, attn_out, tp_size)

    residual1 = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM).get_output(0)
    norm2 = _apply_norm(
        network, residual1, hidden_size,
        weights[f"{prefix}.post_attn_norm"], None, eps_tensor,
        "rmsnorm", dtype=dtype)

    if is_moe_layer:
        mlp_out = _add_moe_tp(
            network, norm2, weights, prefix,
            hidden_size, n_routed_experts, moe_intermediate,
            num_experts_per_tok, shared_intermediate, tp_size,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor, dtype=dtype)
    else:
        mlp_out = _add_swiglu_tp(
            network, norm2, hidden_size, dense_intermediate,
            weights[f"{prefix}.w_gate"],
            weights[f"{prefix}.w_up"],
            weights[f"{prefix}.w_down"], tp_size, dtype=dtype)

    hidden_out = network.add_elementwise(
        residual1, mlp_out,
        trt.ElementWiseOperation.SUM).get_output(0)
    return {
        "hidden": hidden_out,
        "present_k": native_attention["present_k"],
        "present_v": native_attention["present_v"],
    }


def build_deepseek_ocr_tp_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "bf16",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
    profile_mode: str = "decode",
) -> bytes:
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_deepseek_ocr_tp_engine requires tensor_parallel mode")
    if precision != "bf16" or quant_ctx is not None:
        raise ValueError(
            "DeepSeek-OCR TP supports only unquantized BF16 native KV")
    if profile_mode not in ("prefill", "decode"):
        raise ValueError("profile_mode must be 'prefill' or 'decode'")

    rank_weights = shard_deepseek_ocr_weights(
        config, weights, parallel=parallel)
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads) // parallel.tp_size
    num_kv_heads = int(config.num_key_value_heads) // parallel.tp_size
    attention_size = int(rank_weights["_attention_size"])
    kv_attention_size = int(rank_weights["_kv_attention_size"])
    head_dim = attention_size // num_heads
    opt_prefill_length, max_prefill_length = (
        sequence_prefill_profile_lengths(max_cache_length))

    n_routed_experts = int(rank_weights["_n_routed_experts"])
    num_experts_per_tok = int(rank_weights["_num_experts_per_tok"])
    first_k_dense_replace = int(rank_weights["_first_k_dense_replace"])
    moe_intermediate = int(rank_weights["_moe_intermediate_size"])
    shared_intermediate = int(rank_weights["_shared_intermediate_size"])
    dense_intermediate = int(config.intermediate_size) // parallel.tp_size
    norm_topk_prob = bool(rank_weights.get("_norm_topk_prob", False))
    routed_scaling_factor = float(
        rank_weights.get("_routed_scaling_factor", 1.0))
    work_np_dtype = np.float16
    work_trt_dtype = trt.bfloat16

    logger = trt.Logger(
        trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, 1 << 30)
    multi_device_preview = getattr(
        trt.PreviewFeature, "MULTIDEVICE_RUNTIME_10_16", None)
    if multi_device_preview is not None:
        trt_config.set_preview_feature(multi_device_preview, True)

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    cache_write_indices = network.add_input(
        "cache_write_indices", trt.int32, (1,))
    key_value_lengths = network.add_input(
        "key_value_lengths", trt.int32, (1,))
    input_embed = network.add_input(
        "input_embed", trt.float32, (-1, hidden))
    use_input_embed = network.add_input(
        "use_input_embed", trt.float32, (-1, 1))

    cache_shape = (1, num_kv_heads, max_cache_length, head_dim)
    cache_k_inputs = []
    cache_v_inputs = []
    for layer_idx in range(num_layers):
        cache_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_k", layer_idx),
            work_trt_dtype, cache_shape))
        cache_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_v", layer_idx),
            work_trt_dtype, cache_shape))

    profile = builder.create_optimization_profile()
    if profile_mode == "prefill":
        profile.set_shape(
            "token_id", (1,), (opt_prefill_length,),
            (max_prefill_length,))
        profile.set_shape(
            "position_id", (1,), (opt_prefill_length,),
            (max_prefill_length,))
        profile.set_shape(
            "input_embed", (1, hidden),
            (opt_prefill_length, hidden),
            (max_prefill_length, hidden))
        profile.set_shape(
            "use_input_embed", (1, 1),
            (opt_prefill_length, 1),
            (max_prefill_length, 1))
    else:
        profile.set_shape("token_id", (1,), (1,), (1,))
        profile.set_shape("position_id", (1,), (1,), (1,))
        profile.set_shape(
            "input_embed", (1, hidden), (1, hidden), (1, hidden))
        profile.set_shape(
            "use_input_embed", (1, 1), (1, 1), (1, 1))
    trt_config.add_optimization_profile(profile)

    def const_in_work_dtype(shape, values):
        tensor = graph_ops.add_constant(
            network, shape, values, dtype=work_np_dtype)
        if tensor.dtype != work_trt_dtype:
            tensor = network.add_cast(
                tensor, work_trt_dtype).get_output(0)
        return tensor

    embedding_table = const_in_work_dtype(
        (vocab, hidden), rank_weights["embedding"])
    inv_freq = graph_ops.make_native_active_rope_inv_freq(
        head_dim, config.rope_theta)
    cos_half_tensor, sin_half_tensor = graph_ops.add_active_rope_cache(
        network, position_id, inv_freq, work_trt_dtype)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1),
        np.array([config.rms_norm_eps], dtype=np.float32))

    token_embed = network.add_gather(
        embedding_table, token_id, 0).get_output(0)
    input_embed = network.add_cast(
        input_embed, work_trt_dtype).get_output(0)
    use_input_embed = network.add_cast(
        use_input_embed, work_trt_dtype).get_output(0)
    one = const_in_work_dtype(
        (1, 1), np.array([1.0], dtype=work_np_dtype))
    inverse_selector = network.add_elementwise(
        one, use_input_embed,
        trt.ElementWiseOperation.SUB).get_output(0)
    token_part = network.add_elementwise(
        inverse_selector, token_embed,
        trt.ElementWiseOperation.PROD).get_output(0)
    embed_part = network.add_elementwise(
        use_input_embed, input_embed,
        trt.ElementWiseOperation.PROD).get_output(0)
    hidden_state = network.add_elementwise(
        token_part, embed_part,
        trt.ElementWiseOperation.SUM).get_output(0)

    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_decoder_layer_tp(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            cache_write_indices=cache_write_indices,
            key_value_lengths=key_value_lengths,
            position_id=position_id,
            cos_half_tensor=cos_half_tensor,
            sin_half_tensor=sin_half_tensor,
            eps_tensor=eps_tensor,
            weights=rank_weights,
            prefix=prefix,
            hidden_size=hidden,
            attention_size=attention_size,
            kv_attention_size=kv_attention_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            is_moe_layer=layer_idx >= first_k_dense_replace,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate=moe_intermediate,
            shared_intermediate=shared_intermediate,
            dense_intermediate=dense_intermediate,
            tp_size=parallel.tp_size,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            dtype=work_np_dtype,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])

    final_norm = rank_weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(
            network, hidden_state, hidden, final_norm,
            None, eps_tensor, "rmsnorm", dtype=work_np_dtype)

    hidden_shape = network.add_shape(hidden_state).get_output(0)
    one_hidden = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64),
        dtype=np.int64)
    last_start = network.add_elementwise(
        hidden_shape, one_hidden,
        trt.ElementWiseOperation.SUB).get_output(0)
    last_size = graph_ops.add_constant(
        network, (2,), np.array([1, hidden], dtype=np.int64),
        dtype=np.int64)
    last_slice = network.add_slice(
        hidden_state, start=(0, 0), shape=(0, 0), stride=(1, 1))
    last_slice.set_input(1, last_start)
    last_slice.set_input(2, last_size)

    logits = graph_ops.add_matmul_rhs_constant(
        network, last_slice.get_output(0), hidden, vocab,
        rank_weights["w_out"], dtype=work_np_dtype)
    logits = graph_ops.add_bias_sum(
        network, logits, vocab, np.zeros(vocab, dtype=np.float32),
        dtype=work_np_dtype)
    logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for layer_idx, (present_k, present_v) in enumerate(
            zip(present_k_outputs, present_v_outputs)):
        present_k.name = graph_ops.layer_tensor_name(
            "present_k", layer_idx)
        present_v.name = graph_ops.layer_tensor_name(
            "present_v", layer_idx)
        network.mark_output(present_k)
        network.mark_output(present_v)

    if verbose:
        print(
            "[trtmc build] Building DeepSeek-OCR TP native KV "
            f"{profile_mode} engine (rank={parallel.rank}/"
            f"{parallel.tp_size}, layers={num_layers}, hidden={hidden}, "
            f"heads={num_heads}, kv_heads={num_kv_heads}, "
            f"capacity={max_cache_length}) ...",
            file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("DeepSeek-OCR TP TensorRT engine build failed")
    return bytes(plan)
