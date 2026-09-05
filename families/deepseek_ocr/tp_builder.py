# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel DeepSeek-OCR text decoder builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_blocks, graph_ops
from .prefill_config import sequence_prefill_profile_lengths
from .parallel import add_all_reduce_sum, normalize_parallel_config
from .default_decoder import _apply_norm


if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from .config import ModelConfig
    from .parallel import ParallelConfig


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
        elif key.endswith(".experts.w_down"):
            out[key] = np.ascontiguousarray(
                np.array_split(value, parallel.tp_size, axis=-2)[parallel.rank])
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
) -> trt.ITensor:
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_gate)
    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_up)
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    return graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), intermediate_size, hidden_size, w_down)


def _add_swiglu_tp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    tp_size: int,
) -> trt.ITensor:
    local = _add_swiglu_local(
        network, inp, hidden_size, intermediate_size, w_gate, w_up, w_down)
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
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
) -> trt.ITensor:
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, n_routed_experts,
        weights[f"{prefix}.router"])
    sm = network.add_softmax(router_logits)
    sm.axes = 1 << 1
    topk = network.add_topk(
        sm.get_output(0), trt.TopKOperation.MAX,
        num_experts_per_tok, 1 << 1)
    top_values = topk.get_output(0)
    top_indices = topk.get_output(1)

    if norm_topk_prob:
        sum_val = network.add_reduce(
            top_values, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
        scaled_weights = network.add_elementwise(
            top_values, sum_val.get_output(0), trt.ElementWiseOperation.DIV
        ).get_output(0)
    elif routed_scaling_factor != 1.0:
        scale_c = graph_ops.add_constant(
            network, (1, 1),
            np.array([[routed_scaling_factor]], dtype=np.float32))
        scaled_weights = network.add_elementwise(
            top_values, scale_c, trt.ElementWiseOperation.PROD).get_output(0)
    else:
        scaled_weights = top_values

    routed_local = graph_blocks.add_routed_swiglu_experts(
        network,
        inp,
        top_indices,
        scaled_weights,
        hidden_size=hidden_size,
        top_k=num_experts_per_tok,
        w_gate=weights[f"{prefix}.experts.w_gate"],
        w_up=weights[f"{prefix}.experts.w_up"],
        w_down=weights[f"{prefix}.experts.w_down"],
    )

    shared_local = _add_swiglu_local(
        network, inp, hidden_size, shared_intermediate,
        weights[f"{prefix}.shared.w_gate"],
        weights[f"{prefix}.shared.w_up"],
        weights[f"{prefix}.shared.w_down"],
    )
    local_total = network.add_elementwise(
        routed_local, shared_local, trt.ElementWiseOperation.SUM)
    return add_all_reduce_sum(network, local_total.get_output(0), tp_size)


def _add_decoder_layer_tp(
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
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_length: int,
    is_moe_layer: bool,
    n_routed_experts: int,
    num_experts_per_tok: int,
    moe_intermediate: int,
    shared_intermediate: int,
    dense_intermediate: int,
    tp_size: int,
    norm_topk_prob: bool = False,
    routed_scaling_factor: float = 1.0,
    sequence_length: int | None = 1,
) -> dict[str, trt.ITensor]:
    norm1 = _apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"], None, eps_tensor, "rmsnorm")

    q = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, attention_size,
        weights[f"{prefix}.w_q"])
    k = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_k"])
    v = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_v"])

    q = graph_ops.add_apply_rope_native(
        network, q, num_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, head_dim, sequence_length=sequence_length)
    k = graph_ops.add_apply_rope_native(
        network, k, num_kv_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, head_dim, sequence_length=sequence_length)

    present_k = k
    present_v = v

    current_k = k
    current_v = v
    if sequence_length is not None:
        k_reshape = network.add_shuffle(k)
        k_reshape.reshape_dims = (sequence_length, kv_attention_size)
        current_k = k_reshape.get_output(0)
        v_reshape = network.add_shuffle(v)
        v_reshape.reshape_dims = (sequence_length, kv_attention_size)
        current_v = v_reshape.get_output(0)

    all_k = network.add_concatenation([cache_k, current_k])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, current_v])
    all_v.axis = 0

    mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
    context_flat = graph_ops.add_attention_from_rows(
        network, q, all_k.get_output(0), all_v.get_output(0),
        num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads,
        q_seq=sequence_length,
        kv_seq=None if sequence_length is None else max_cache_length + 1,
        mask=mask_4d)

    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context_flat, attention_size, hidden_size,
        weights[f"{prefix}.w_o"])
    attn_out = add_all_reduce_sum(network, attn_out, tp_size)

    residual1 = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)
    norm2 = _apply_norm(
        network, residual1.get_output(0), hidden_size,
        weights[f"{prefix}.post_attn_norm"], None, eps_tensor, "rmsnorm")

    if is_moe_layer:
        mlp_out = _add_moe_tp(
            network, norm2, weights, prefix,
            hidden_size, n_routed_experts, moe_intermediate,
            num_experts_per_tok, shared_intermediate, tp_size,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor)
    else:
        mlp_out = _add_swiglu_tp(
            network, norm2, hidden_size, dense_intermediate,
            weights[f"{prefix}.w_gate"],
            weights[f"{prefix}.w_up"],
            weights[f"{prefix}.w_down"],
            tp_size)

    residual2 = network.add_elementwise(
        residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)
    return {
        "hidden": residual2.get_output(0),
        "present_k": present_k,
        "present_v": present_v,
    }


def build_deepseek_ocr_tp_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    parallel_config=None,
) -> bytes:
    del precision, quant_ctx
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "build_deepseek_ocr_tp_engine requires tensor_parallel mode with "
            "tp_size > 1")

    image_prefill_tokens = 145
    if max_cache_length <= image_prefill_tokens:
        print(
            "[trtmc build] WARNING: DeepSeek-OCR-2 uses 145 image prefill "
            f"tokens. max_cache_length={max_cache_length} is too small and "
            "can cause prompt echo / repeated skip-like tokens. Use "
            "--max-cache-length 4096.",
            file=sys.stderr,
        )
    elif max_cache_length < 4096:
        print(
            "[trtmc build] NOTE: DeepSeek-OCR-2 is more stable with "
            "--max-cache-length 4096.",
            file=sys.stderr,
        )

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
    opt_prefill_length, max_prefill_length = sequence_prefill_profile_lengths(max_cache_length)

    n_routed_experts = int(rank_weights["_n_routed_experts"])
    num_experts_per_tok = int(rank_weights["_num_experts_per_tok"])
    first_k_dense_replace = int(rank_weights["_first_k_dense_replace"])
    moe_intermediate = int(rank_weights["_moe_intermediate_size"])
    shared_intermediate = int(rank_weights["_shared_intermediate_size"])
    dense_intermediate = int(config.intermediate_size) // parallel.tp_size
    norm_topk_prob = bool(rank_weights.get("_norm_topk_prob", False))
    routed_scaling_factor = float(rank_weights.get("_routed_scaling_factor", 1.0))

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    multi_device_preview = getattr(
        trt.PreviewFeature, "MULTIDEVICE_RUNTIME_10_16", None)
    if multi_device_preview is not None:
        trt_config.set_preview_feature(multi_device_preview, True)

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (-1, -1))
    input_embed_tensor = network.add_input(
        "input_embed", trt.float32, (-1, hidden))
    use_input_embed_tensor = network.add_input(
        "use_input_embed", trt.float32, (-1, 1))

    cache_k_inputs = []
    cache_v_inputs = []
    for layer_idx in range(num_layers):
        cache_k_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_k", layer_idx),
            trt.float32, (max_cache_length, kv_attention_size)))
        cache_v_inputs.append(network.add_input(
            graph_ops.layer_tensor_name("cache_v", layer_idx),
            trt.float32, (max_cache_length, kv_attention_size)))

    def _add_profile(opt_sq: int, max_sq: int, *, fixed: bool = False) -> None:
        profile = builder.create_optimization_profile()
        min_sq = opt_sq if fixed else 1
        profile.set_shape("token_id", (min_sq,), (opt_sq,), (max_sq,))
        profile.set_shape("position_id", (min_sq,), (opt_sq,), (max_sq,))
        profile.set_shape(
            "attention_mask",
            (min_sq, max_cache_length + min_sq),
            (opt_sq, max_cache_length + opt_sq),
            (max_sq, max_cache_length + max_sq))
        profile.set_shape(
            "input_embed", (min_sq, hidden), (opt_sq, hidden), (max_sq, hidden))
        profile.set_shape(
            "use_input_embed", (min_sq, 1), (opt_sq, 1), (max_sq, 1))
        trt_config.add_optimization_profile(profile)

    _add_profile(opt_prefill_length, max_prefill_length)
    _add_profile(1, 1, fixed=True)

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), rank_weights["embedding"])
    graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
    cos_half_np = graph_ops.make_rope_table_half_dim(
        max_cache_length * 2, head_dim, config.rope_theta, True)
    sin_half_np = graph_ops.make_rope_table_half_dim(
        max_cache_length * 2, head_dim, config.rope_theta, False)
    cos_half_tensor = graph_ops.add_constant(network, cos_half_np.shape, cos_half_np)
    sin_half_tensor = graph_ops.add_constant(network, sin_half_np.shape, sin_half_np)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))

    gather = network.add_gather(embedding_table, token_id, 0)
    token_embed = gather.get_output(0)
    one_const = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=np.float32))
    inv_flag = network.add_elementwise(
        one_const, use_input_embed_tensor, trt.ElementWiseOperation.SUB)
    tok_part = network.add_elementwise(
        inv_flag.get_output(0), token_embed, trt.ElementWiseOperation.PROD)
    embed_part = network.add_elementwise(
        use_input_embed_tensor, input_embed_tensor,
        trt.ElementWiseOperation.PROD)
    hidden_sum = network.add_elementwise(
        tok_part.get_output(0), embed_part.get_output(0),
        trt.ElementWiseOperation.SUM)
    hidden_state = hidden_sum.get_output(0)

    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_decoder_layer_tp(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            attention_mask=attention_mask,
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
            max_cache_length=max_cache_length,
            is_moe_layer=layer_idx >= first_k_dense_replace,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate=moe_intermediate,
            shared_intermediate=shared_intermediate,
            dense_intermediate=dense_intermediate,
            tp_size=parallel.tp_size,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
            sequence_length=None,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])

    final_norm = rank_weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(
            network, hidden_state, hidden, final_norm,
            None, eps_tensor, "rmsnorm")

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

    logits = graph_ops.add_matmul_rhs_constant(
        network, last_hidden, hidden, vocab, rank_weights["w_out"])
    logits = graph_ops.add_bias_sum(
        network, logits, vocab, np.zeros(vocab, dtype=np.float32))
    logits.name = "logits"
    network.mark_output(logits)

    for layer_idx in range(num_layers):
        pk = present_k_outputs[layer_idx]
        pv = present_v_outputs[layer_idx]
        pk.name = graph_ops.layer_tensor_name("present_k", layer_idx)
        pv.name = graph_ops.layer_tensor_name("present_v", layer_idx)
        network.mark_output(pk)
        network.mark_output(pv)

    if verbose:
        print(
            "[trtmc build] Building DeepSeek-OCR TP TRT engine "
            f"(rank={parallel.rank}/{parallel.tp_size}, {num_layers} layers, "
            f"hidden={hidden}, attn={attention_size}, heads={num_heads}, "
            f"kv_heads={num_kv_heads}, experts={n_routed_experts}, "
            f"top_k={num_experts_per_tok}, moe_inter={moe_intermediate}, "
            f"shared_inter={shared_intermediate}, "
            f"dense_inter={dense_intermediate}, cache={max_cache_length}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(plan)
