"""Tensor-parallel Phi-MoE builder."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from . import model as graph_blocks, model as graph_ops
from ....parallel_config import add_all_reduce_sum, normalize_parallel_config
from .model import _apply_norm, _mark_debug_output

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ..weights import WeightDict
    from ..config import ModelConfig
    from ....parallel_config import ParallelConfig


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=-1)[rank])


def _slice_first_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    return np.ascontiguousarray(np.array_split(arr, tp_size, axis=0)[rank])


def _validate_phi_moe_tp(
    config: "ModelConfig",
    weights: "WeightDict",
    parallel: "ParallelConfig",
) -> None:
    parallel.validate()
    if not parallel.enabled:
        return
    if parallel.rank < 0:
        raise ValueError("Phi-MoE tensor-parallel build requires a concrete rank")

    tp = parallel.tp_size
    if int(config.num_attention_heads) % tp != 0:
        raise ValueError(
            "Phi-MoE tensor parallel requires num_attention_heads divisible by "
            f"tp_size ({config.num_attention_heads} vs {tp})"
        )
    if int(config.num_key_value_heads) % tp != 0:
        raise ValueError(
            "Phi-MoE tensor parallel requires num_key_value_heads divisible by "
            f"tp_size ({config.num_key_value_heads} vs {tp})"
        )
    if int(weights.get("_num_experts_per_tok", 2)) != 2:
        raise ValueError("Phi-MoE tensor parallel currently supports top_k=2")

    attention_size = int(weights.get("_attention_size", config.attention_size))
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights,
        num_kv_heads=int(config.num_key_value_heads),
        head_dim=attention_size // int(config.num_attention_heads),
    )
    moe_intermediate = int(weights["_moe_intermediate_size"])
    checks = {
        "attention_size": attention_size,
        "kv_attention_size": kv_attention_size,
        "moe_intermediate_size": moe_intermediate,
    }
    for name, value in checks.items():
        if value % tp != 0:
            raise ValueError(
                f"Phi-MoE tensor parallel requires {name} divisible by tp_size ({value} vs {tp})"
            )


def shard_phi_moe_weights(
    config: "ModelConfig",
    weights: "WeightDict",
    *,
    parallel: "ParallelConfig",
) -> "WeightDict":
    """Return rank-local Phi-MoE weights for the TP builder."""
    _validate_phi_moe_tp(config, weights, parallel)
    if not parallel.enabled:
        return weights

    out = type(weights)()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue
        if key.endswith((".w_q", ".w_k", ".w_v", ".w_gate", ".w_up")):
            out[key] = _slice_last_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".q_bias", ".k_bias", ".v_bias")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        elif key.endswith((".w_o", ".w_down")):
            out[key] = _slice_first_dim(value, parallel.rank, parallel.tp_size)
        else:
            out[key] = value

    out["_attention_size"] = int(weights["_attention_size"]) // parallel.tp_size
    out["_moe_intermediate_size"] = int(weights["_moe_intermediate_size"]) // parallel.tp_size
    out["_tensor_parallel_size"] = parallel.tp_size
    out["_tensor_parallel_rank"] = parallel.rank
    return out


def _add_bias_if_present(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    dim: int,
    bias: np.ndarray | None,
) -> trt.ITensor:
    if bias is None:
        return tensor
    return graph_ops.add_bias_sum(network, tensor, dim, bias)


def _add_swiglu_expert_tp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> trt.ITensor:
    gate = graph_ops.add_matmul_rhs_constant(network, inp, hidden_size, intermediate_size, w_gate)
    up = graph_ops.add_matmul_rhs_constant(network, inp, hidden_size, intermediate_size, w_up)
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    return graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), intermediate_size, hidden_size, w_down
    )


def _sparsemixer_weight(
    network: trt.INetworkDefinition,
    scores: trt.ITensor,
    num_experts: int,
    jitter_eps: float,
    original_scores: trt.ITensor | None = None,
) -> tuple[trt.ITensor, trt.ITensor]:
    if original_scores is None:
        original_scores = scores

    topk1 = network.add_topk(scores, trt.TopKOperation.MAX, 1, 1 << 1)
    max_val = topk1.get_output(0)
    max_ind = topk1.get_output(1)

    abs_scores = network.add_unary(original_scores, trt.UnaryOperation.ABS)
    factor = network.add_elementwise(
        abs_scores.get_output(0), max_val, trt.ElementWiseOperation.MAX
    )

    diff = network.add_elementwise(max_val, original_scores, trt.ElementWiseOperation.SUB)
    ratio = network.add_elementwise(
        diff.get_output(0), factor.get_output(0), trt.ElementWiseOperation.DIV
    )

    threshold = graph_ops.add_constant(
        network, (1, 1), np.array([2.0 * jitter_eps], dtype=np.float32)
    )
    mask_float = network.add_elementwise(
        ratio.get_output(0), threshold, trt.ElementWiseOperation.GREATER
    )

    neginf = graph_ops.add_constant(network, (1, 1), np.array([-1e9], dtype=np.float32))
    mask_f = network.add_cast(mask_float.get_output(0), trt.float32)
    penalty = network.add_elementwise(mask_f.get_output(0), neginf, trt.ElementWiseOperation.PROD)
    masked = network.add_elementwise(scores, penalty.get_output(0), trt.ElementWiseOperation.SUM)

    sm = network.add_softmax(masked.get_output(0))
    sm.axes = 1 << 1

    idx_flat = network.add_shuffle(max_ind)
    idx_flat.reshape_dims = (1,)
    weight = network.add_gather(sm.get_output(0), idx_flat.get_output(0), 1)
    return weight.get_output(0), max_ind


def _add_phi_moe_tp_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    num_experts: int,
    moe_intermediate: int,
    tp_size: int,
    jitter_eps: float,
) -> trt.ITensor:
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, num_experts, weights[f"{prefix}.router"]
    )

    weight_1, idx_1 = _sparsemixer_weight(network, router_logits, num_experts, jitter_eps)

    idx_1_flat = network.add_shuffle(idx_1)
    idx_1_flat.reshape_dims = (1,)
    range_const = graph_ops.add_constant(
        network, (1, num_experts), np.arange(num_experts, dtype=np.float32).reshape(1, -1)
    )
    idx_1_broadcast = network.add_shuffle(idx_1_flat.get_output(0))
    idx_1_broadcast.reshape_dims = (1, 1)
    idx_1_f = network.add_cast(idx_1_broadcast.get_output(0), trt.float32)
    eq = network.add_elementwise(range_const, idx_1_f.get_output(0), trt.ElementWiseOperation.EQUAL)
    eq_f = network.add_cast(eq.get_output(0), trt.float32)
    neginf_mask = graph_ops.add_constant(network, (1, 1), np.array([-1e9], dtype=np.float32))
    penalty = network.add_elementwise(
        eq_f.get_output(0), neginf_mask, trt.ElementWiseOperation.PROD
    )
    scores_2 = network.add_elementwise(
        router_logits, penalty.get_output(0), trt.ElementWiseOperation.SUM
    )

    weight_2, idx_2 = _sparsemixer_weight(
        network, scores_2.get_output(0), num_experts, jitter_eps, original_scores=router_logits
    )

    expert_outputs = []
    for expert_idx in range(num_experts):
        expert_outputs.append(
            _add_swiglu_expert_tp(
                network,
                inp,
                hidden_size,
                moe_intermediate,
                weights[f"{prefix}.expert.{expert_idx}.w_gate"],
                weights[f"{prefix}.expert.{expert_idx}.w_up"],
                weights[f"{prefix}.expert.{expert_idx}.w_down"],
            )
        )

    stacked = network.add_concatenation(expert_outputs)
    stacked.axis = 0
    stacked_out = stacked.get_output(0)

    idx_1_scalar = network.add_shuffle(idx_1)
    idx_1_scalar.reshape_dims = (1,)
    expert_1_out = network.add_gather(stacked_out, idx_1_scalar.get_output(0), 0)
    scaled_1 = network.add_elementwise(
        expert_1_out.get_output(0), weight_1, trt.ElementWiseOperation.PROD
    )

    idx_2_scalar = network.add_shuffle(idx_2)
    idx_2_scalar.reshape_dims = (1,)
    expert_2_out = network.add_gather(stacked_out, idx_2_scalar.get_output(0), 0)
    scaled_2 = network.add_elementwise(
        expert_2_out.get_output(0), weight_2, trt.ElementWiseOperation.PROD
    )

    summed = network.add_elementwise(
        scaled_1.get_output(0), scaled_2.get_output(0), trt.ElementWiseOperation.SUM
    )
    return add_all_reduce_sum(network, summed.get_output(0), tp_size)


def _add_phi_moe_tp_decoder_layer(
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
    num_experts: int,
    moe_intermediate: int,
    tp_size: int,
    jitter_eps: float,
) -> dict[str, trt.ITensor]:
    attention_window = max_cache_length + 1
    norm1 = _apply_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        weights.get(f"{prefix}.input_norm_beta"),
        eps_tensor,
        "layernorm",
    )

    q = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, attention_size, weights[f"{prefix}.w_q"]
    )
    q = _add_bias_if_present(network, q, attention_size, weights.get(f"{prefix}.q_bias"))
    k = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, kv_attention_size, weights[f"{prefix}.w_k"]
    )
    k = _add_bias_if_present(network, k, kv_attention_size, weights.get(f"{prefix}.k_bias"))
    v = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, kv_attention_size, weights[f"{prefix}.w_v"]
    )
    v = _add_bias_if_present(network, v, kv_attention_size, weights.get(f"{prefix}.v_bias"))

    q = graph_ops.add_apply_rope_native(
        network, q, num_heads, head_dim, cos_half_tensor, sin_half_tensor, position_id, head_dim
    )
    k = graph_ops.add_apply_rope_native(
        network, k, num_kv_heads, head_dim, cos_half_tensor, sin_half_tensor, position_id, head_dim
    )

    present_k = k
    present_v = v
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)
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
        q_seq=1,
        kv_seq=attention_window,
        mask=mask_4d,
    )

    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context_flat, attention_size, hidden_size, weights[f"{prefix}.w_o"]
    )
    attn_out = add_all_reduce_sum(network, attn_out, tp_size)
    attn_out = _add_bias_if_present(network, attn_out, hidden_size, weights.get(f"{prefix}.o_bias"))

    residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)
    norm2 = _apply_norm(
        network,
        residual1.get_output(0),
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights.get(f"{prefix}.post_attn_norm_beta"),
        eps_tensor,
        "layernorm",
    )

    moe_out = _add_phi_moe_tp_block(
        network,
        norm2,
        weights,
        prefix,
        hidden_size,
        num_experts,
        moe_intermediate,
        tp_size,
        jitter_eps,
    )
    residual2 = network.add_elementwise(
        residual1.get_output(0), moe_out, trt.ElementWiseOperation.SUM
    )

    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": present_k,
        "present_v": present_v,
    }


def build_phi_moe_tp_engine(
    config: "ModelConfig",
    weights: "WeightDict",
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    del precision, quant_ctx
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError("build_phi_moe_tp_engine requires tensor_parallel mode with tp_size > 1")

    rank_weights = shard_phi_moe_weights(config, weights, parallel=parallel)
    attention_size = int(rank_weights["_attention_size"])
    num_experts = int(rank_weights["_num_experts"])
    moe_intermediate = int(rank_weights["_moe_intermediate_size"])
    hidden = int(config.hidden_size)
    vocab = int(config.vocab_size)
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads) // parallel.tp_size
    num_kv_heads = int(config.num_key_value_heads) // parallel.tp_size
    head_dim = attention_size // num_heads
    kv_attention_size = num_kv_heads * head_dim
    attention_window = max_cache_length + 1
    jitter_eps = float(config.raw.get("router_jitter_noise", 0.01))

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))
    cache_k_inputs = []
    cache_v_inputs = []
    for layer_idx in range(num_layers):
        cache_k_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_k", layer_idx),
                trt.float32,
                (max_cache_length, kv_attention_size),
            )
        )
        cache_v_inputs.append(
            network.add_input(
                graph_ops.layer_tensor_name("cache_v", layer_idx),
                trt.float32,
                (max_cache_length, kv_attention_size),
            )
        )

    embedding_table = graph_ops.add_constant(network, (vocab, hidden), rank_weights["embedding"])
    graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
    cos_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, True
    )
    sin_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, False
    )
    cos_half_tensor = graph_ops.add_constant(network, cos_half_np.shape, cos_half_np)
    sin_half_tensor = graph_ops.add_constant(network, sin_half_np.shape, sin_half_np)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32)
    )

    hidden_state = network.add_gather(embedding_table, token_id, 0).get_output(0)
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        result = _add_phi_moe_tp_decoder_layer(
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
            num_experts=num_experts,
            moe_intermediate=moe_intermediate,
            tp_size=parallel.tp_size,
            jitter_eps=jitter_eps,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])
        if debug_layer_outputs:
            _mark_debug_output(network, result["post_attn"], f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    final_norm = rank_weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(
            network,
            hidden_state,
            hidden,
            final_norm,
            rank_weights.get("final_norm_beta"),
            eps_tensor,
            "layernorm",
        )

    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, rank_weights["w_out"]
    )
    lm_bias = rank_weights.get("lm_head_bias")
    if lm_bias is not None:
        logits = graph_ops.add_bias_sum(network, logits, vocab, lm_bias)
    else:
        logits = graph_ops.add_bias_sum(network, logits, vocab, np.zeros(vocab, dtype=np.float32))
    logits.name = "logits"
    network.mark_output(logits)

    for layer_idx in range(num_layers):
        present_k_outputs[layer_idx].name = graph_ops.layer_tensor_name("present_k", layer_idx)
        present_v_outputs[layer_idx].name = graph_ops.layer_tensor_name("present_v", layer_idx)
        network.mark_output(present_k_outputs[layer_idx])
        network.mark_output(present_v_outputs[layer_idx])

    if verbose:
        print(
            "[trtmc build] Phi-MoE TP engine "
            f"(rank={parallel.rank}/{parallel.tp_size}, {num_layers}L, "
            f"local_attn={attention_size}, local_moe={moe_intermediate})",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT Phi-MoE TP engine build failed")
    return bytes(plan)
