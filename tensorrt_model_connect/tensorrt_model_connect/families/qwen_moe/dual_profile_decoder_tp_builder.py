"""Tensor-parallel MoE decoder builder for Qwen3-MoE.

Sibling file to ``plugin.py`` (which holds the dense MoE build path —
this builder does not modify the dense path). The TP design is
**TP-within-experts**: all ranks run the same routing and the same set
of routed experts, but each expert's MLP is column/row-sharded so the
moe_intermediate dimension is split across ranks. A TRT 11.0+
distributed ALL_REDUCE collective joins the partial outputs after the
packed expert down-projection and after the attention output
projection.

Shape contract (rank-local):
  * num_heads, num_kv_heads     // tp_size
  * moe_intermediate            // tp_size  (per-expert hidden)
  * shared_expert_intermediate  // tp_size  (Qwen2.5-MoE shared MLP)
  * dense_intermediate          // tp_size  (mlp_only_layers)
  * router weight               replicated (no TP on routing)
  * gate weight (shared expert) replicated (sigmoid gate is scalar-ish)

Scope: covers Qwen3-MoE (pure routed, no shared expert) and Qwen2.5-MoE
(shared expert) — the same variants the dense ``plugin.py`` build
supports. Quantization is rejected (TP+quant is a follow-up).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
from tensorrt_model_connect import trt_compat

from ... import graph_ops
from ... import graph_blocks
from ...parallel_config import (
    add_all_reduce_sum,
    normalize_parallel_config,
)
from .standard_decoder_builder import _apply_norm, _mark_debug_output

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from ...config import ModelConfig
    from ...checkpoint_mapper import WeightDict


# ---------------------------------------------------------------------------
# Weight sharder
# ---------------------------------------------------------------------------


def _slice_last_dim(arr: np.ndarray, rank: int, tp_size: int) -> np.ndarray:
    parts = np.array_split(arr, tp_size, axis=-1)
    return np.ascontiguousarray(parts[rank])


def _slice_axis(arr: np.ndarray, axis: int, rank: int, tp_size: int) -> np.ndarray:
    parts = np.array_split(arr, tp_size, axis=axis)
    return np.ascontiguousarray(parts[rank])


def _shard_qwen_moe_weights(
    weights: "WeightDict",
    parallel,
) -> "WeightDict":
    """Return rank-local Qwen-MoE weights for TP-within-experts.

    Sharding policy:
      * .w_q / .w_k / .w_v / .q_bias / .k_bias / .v_bias  -> last-dim (col)
      * .w_o                                              -> first-dim (row)
      * .experts.w_gate / .experts.w_up
        shape [num_experts, hidden, moe_inter]            -> axis 2 (col)
      * .experts.w_down
        shape [num_experts, moe_inter, hidden]            -> axis 1 (row)
      * .shared_expert.w_gate / w_up                      -> last-dim (col)
      * .shared_expert.w_down                             -> first-dim (row)
      * .mlp.w_gate / w_up (dense MLP layers)             -> last-dim (col)
      * .mlp.w_down (dense MLP layers)                    -> first-dim (row)
      * .router, .shared_expert_gate                      -> replicated
      * q_norm, k_norm (per-head, size <= head_dim sets)  -> per-head shard

    Embeddings, layernorms, LM head are kept replicated.
    """
    from ...checkpoint_mapper import WeightDict

    if not parallel.enabled:
        return weights
    rank = parallel.rank
    tp = parallel.tp_size

    out = WeightDict()
    for key, value in weights.items():
        if not isinstance(value, np.ndarray):
            out[key] = value
            continue
        if key.endswith((".w_q", ".w_k", ".w_v",
                         ".q_bias", ".k_bias", ".v_bias")):
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith(".w_o"):
            out[key] = _slice_axis(value, 0, rank, tp)
        elif key.endswith((".experts.w_gate", ".experts.w_up")):
            # Shape [num_experts, hidden, moe_inter] — shard last dim.
            out[key] = _slice_axis(value, 2, rank, tp)
        elif key.endswith(".experts.w_down"):
            # Shape [num_experts, moe_inter, hidden] — shard axis 1.
            out[key] = _slice_axis(value, 1, rank, tp)
        elif key.endswith((".shared_expert.w_gate", ".shared_expert.w_up",
                           ".mlp.w_gate", ".mlp.w_up",
                           ".w_gate", ".w_up")):
            out[key] = _slice_last_dim(value, rank, tp)
        elif key.endswith((".shared_expert.w_down", ".mlp.w_down", ".w_down")):
            out[key] = _slice_axis(value, 0, rank, tp)
        else:
            # Replicated: router, shared_expert_gate, embedding, norms,
            # LM head, q_norm, k_norm.
            out[key] = value

    # Update size metadata if present.
    if "_attention_size" in weights:
        out["_attention_size"] = int(weights["_attention_size"]) // tp
    if "_kv_attention_size" in weights:
        out["_kv_attention_size"] = int(weights["_kv_attention_size"]) // tp
    if "_moe_intermediate_size" in weights:
        out["_moe_intermediate_size"] = int(weights["_moe_intermediate_size"]) // tp
    if "_shared_expert_intermediate_size" in weights:
        v = int(weights["_shared_expert_intermediate_size"])
        out["_shared_expert_intermediate_size"] = (v // tp) if v > 0 else 0
    if "_dense_intermediate_size" in weights:
        v = int(weights["_dense_intermediate_size"])
        out["_dense_intermediate_size"] = (v // tp) if v > 0 else 0
    out["_tensor_parallel_size"] = tp
    out["_tensor_parallel_rank"] = rank
    return out


def _validate_moe_tp(weights: "WeightDict", parallel, config) -> None:
    """Reject configs whose dims don't divide cleanly into tp_size."""
    if not parallel.enabled:
        return
    tp = parallel.tp_size
    # Head divisibility (same constraint dense TP enforces).
    if config.num_attention_heads % tp != 0:
        raise ValueError(
            f"Qwen-MoE TP requires num_attention_heads="
            f"{config.num_attention_heads} divisible by tp_size={tp}")
    if config.num_key_value_heads < tp:
        raise ValueError(
            f"Qwen-MoE TP requires num_key_value_heads="
            f"{config.num_key_value_heads} >= tp_size={tp}. "
            f"KV-head replication for small kv-head counts is not implemented "
            f"yet; use a model with num_kv_heads >= tp_size (e.g. "
            f"Qwen3-MoE-30B-A3B has 4 kv_heads, supports TP=2 and TP=4).")
    if config.num_key_value_heads % tp != 0:
        raise ValueError(
            f"Qwen-MoE TP requires num_key_value_heads="
            f"{config.num_key_value_heads} divisible by tp_size={tp}")
    # MoE / dense intermediate dims must also divide.
    for key in ("_moe_intermediate_size",
                "_shared_expert_intermediate_size",
                "_dense_intermediate_size"):
        v = int(weights.get(key, 0))
        if v > 0 and v % tp != 0:
            raise ValueError(
                f"Qwen-MoE TP requires {key}={v} divisible by tp_size={tp}")


# ---------------------------------------------------------------------------
# MoE block helpers (TP-aware)
# ---------------------------------------------------------------------------


def _add_tp_swiglu_expert(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    *,
    tp_size: int,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Shared / dense SwiGLU MLP with column/row sharding + ALL_REDUCE.

    ``intermediate_size`` is the rank-local (already-sharded) inner dim.
    """
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_gate, dtype=dtype)
    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_up, dtype=dtype)
    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)
    down = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), intermediate_size, hidden_size,
        w_down, dtype=dtype)
    return add_all_reduce_sum(network, down, tp_size)


def _add_tp_packed_swiglu_experts(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    *,
    tp_size: int,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Packed expert MLPs with TP sharding + ALL_REDUCE.

    ``w_gate`` / ``w_up`` are rank-local with shape
    [num_experts, hidden, moe_inter/tp_size]. ``w_down`` is
    [num_experts, moe_inter/tp_size, hidden]. Returns
    [num_experts, 1, hidden] (full, ALL_REDUCE-joined).
    """
    num_experts, _, intermediate_size = w_gate.shape

    inp_3d = network.add_shuffle(inp)
    inp_3d.reshape_dims = (1, 1, hidden_size)

    expert_scale = graph_ops.add_constant(
        network, (num_experts, 1, 1),
        np.ones((num_experts, 1, 1), dtype=dtype), dtype=dtype)
    batched_inp = network.add_elementwise(
        inp_3d.get_output(0), expert_scale,
        trt.ElementWiseOperation.PROD).get_output(0)

    gate_w = graph_ops.add_constant(network, w_gate.shape, w_gate, dtype=dtype)
    up_w = graph_ops.add_constant(network, w_up.shape, w_up, dtype=dtype)
    down_w = graph_ops.add_constant(network, w_down.shape, w_down, dtype=dtype)

    gate = network.add_matrix_multiply(
        batched_inp, trt.MatrixOperation.NONE,
        gate_w, trt.MatrixOperation.NONE)
    up = network.add_matrix_multiply(
        batched_inp, trt.MatrixOperation.NONE,
        up_w, trt.MatrixOperation.NONE)

    sigmoid = network.add_activation(
        gate.get_output(0), trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate.get_output(0), sigmoid.get_output(0),
        trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up.get_output(0),
        trt.ElementWiseOperation.PROD)
    down = network.add_matrix_multiply(
        gated.get_output(0), trt.MatrixOperation.NONE,
        down_w, trt.MatrixOperation.NONE)

    # Partial sums across the moe_inter axis split — ALL_REDUCE to combine.
    return add_all_reduce_sum(network, down.get_output(0), tp_size)


def _add_tp_qwen3_moe_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: "WeightDict",
    prefix: str,
    hidden_size: int,
    num_experts: int,
    moe_intermediate: int,
    shared_expert_intermediate: int,
    top_k: int,
    *,
    tp_size: int,
    has_shared_expert: bool = True,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """TP-aware Qwen MoE block. ``moe_intermediate`` is rank-local."""
    # Router (replicated, runs the same on all ranks).
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, num_experts,
        weights[f"{prefix}.router"], dtype=dtype)
    sm = network.add_softmax(router_logits)
    sm.axes = 1 << 1
    topk = network.add_topk(
        sm.get_output(0), trt.TopKOperation.MAX, top_k, 1 << 1)
    top_values = topk.get_output(0)
    top_indices = topk.get_output(1)
    sum_val = network.add_reduce(
        top_values, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
    norm_weights = network.add_elementwise(
        top_values, sum_val.get_output(0),
        trt.ElementWiseOperation.DIV)

    # Packed experts with TP sharding + ALL_REDUCE inside.
    expert_outputs = _add_tp_packed_swiglu_experts(
        network, inp, hidden_size,
        weights[f"{prefix}.experts.w_gate"],
        weights[f"{prefix}.experts.w_up"],
        weights[f"{prefix}.experts.w_down"],
        tp_size=tp_size, dtype=dtype)

    # Gather selected experts, scale by router weights, sum.
    routed_result = None
    for k in range(top_k):
        idx_slice = network.add_slice(
            top_indices, start=(0, k), shape=(1, 1), stride=(1, 1))
        idx_flat = network.add_shuffle(idx_slice.get_output(0))
        idx_flat.reshape_dims = (1,)
        w_slice = network.add_slice(
            norm_weights.get_output(0),
            start=(0, k), shape=(1, 1), stride=(1, 1))
        w_reshape = network.add_shuffle(w_slice.get_output(0))
        w_reshape.reshape_dims = (1, 1, 1)
        expert_out = network.add_gather(
            expert_outputs, idx_flat.get_output(0), 0)
        scaled_expert = network.add_elementwise(
            expert_out.get_output(0), w_reshape.get_output(0),
            trt.ElementWiseOperation.PROD)
        scaled_flat = network.add_shuffle(scaled_expert.get_output(0))
        scaled_flat.reshape_dims = (1, hidden_size)
        if routed_result is None:
            routed_result = scaled_flat.get_output(0)
        else:
            sum_layer = network.add_elementwise(
                routed_result, scaled_flat.get_output(0),
                trt.ElementWiseOperation.SUM)
            routed_result = sum_layer.get_output(0)

    if not has_shared_expert:
        return routed_result

    # Shared expert (Qwen2.5-MoE) — TP-sharded SwiGLU.
    shared_out = _add_tp_swiglu_expert(
        network, inp, hidden_size, shared_expert_intermediate,
        weights[f"{prefix}.shared_expert.w_gate"],
        weights[f"{prefix}.shared_expert.w_up"],
        weights[f"{prefix}.shared_expert.w_down"],
        tp_size=tp_size, dtype=dtype)

    shared_gate_w = weights.get(f"{prefix}.shared_expert_gate")
    if shared_gate_w is not None:
        gate_score = graph_ops.add_matmul_rhs_constant(
            network, inp, hidden_size, 1, shared_gate_w.reshape(-1, 1),
            dtype=dtype)
        gate_sigmoid = network.add_activation(
            gate_score, trt.ActivationType.SIGMOID)
        shared_gated = network.add_elementwise(
            shared_out, gate_sigmoid.get_output(0),
            trt.ElementWiseOperation.PROD)
        shared_final = shared_gated.get_output(0)
    else:
        shared_final = shared_out

    combined = network.add_elementwise(
        routed_result, shared_final, trt.ElementWiseOperation.SUM)
    return combined.get_output(0)


# ---------------------------------------------------------------------------
# Decoder layer (attention + MoE) — TP-aware
# ---------------------------------------------------------------------------


def _add_tp_qwen3_moe_decoder_layer(
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
    is_dense: bool,
    num_experts: int,
    moe_intermediate: int,
    shared_expert_intermediate: int,
    dense_intermediate: int,
    top_k: int,
    tp_size: int,
    has_shared_expert: bool = True,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """One Qwen-MoE decoder layer with TP attention + TP MoE.

    Reuses ``graph_blocks.add_attention_block`` (which is TP-shape-agnostic
    — it operates on whatever num_heads/kv it's given), then inserts
    ALL_REDUCE on the attention output (row-parallel join after W_O).
    The MoE block does its own ALL_REDUCE internally.
    """
    attn = graph_blocks.add_attention_block(
        network, hidden, cache_k, cache_v, attention_mask, position_id,
        weights=weights, prefix=prefix,
        hidden_size=hidden_size, attention_size=attention_size,
        kv_attention_size=kv_attention_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        max_cache_length=max_cache_length,
        eps_tensor=eps_tensor,
        norm_type="rmsnorm", position_type="rope",
        dtype=dtype,
        cos_half_tensor=cos_half_tensor,
        sin_half_tensor=sin_half_tensor,
        rotary_embedding_dim=head_dim,
    )
    # ALL_REDUCE after attention W_O (row-parallel join).
    attn_out = add_all_reduce_sum(network, attn["attn_out"], tp_size)

    residual1 = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)

    norm2 = _apply_norm(
        network, residual1.get_output(0), hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        None, eps_tensor, "rmsnorm", dtype=dtype)

    if is_dense:
        # Dense SwiGLU MLP with TP shard + ALL_REDUCE.
        mlp_out = _add_tp_swiglu_expert(
            network, norm2, hidden_size, dense_intermediate,
            weights[f"{prefix}.mlp.w_gate"],
            weights[f"{prefix}.mlp.w_up"],
            weights[f"{prefix}.mlp.w_down"],
            tp_size=tp_size, dtype=dtype)
    else:
        mlp_out = _add_tp_qwen3_moe_block(
            network, norm2, weights, prefix,
            hidden_size, num_experts, moe_intermediate,
            shared_expert_intermediate, top_k,
            tp_size=tp_size,
            has_shared_expert=has_shared_expert,
            dtype=dtype)

    residual2 = network.add_elementwise(
        residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": attn["present_k"],
        "present_v": attn["present_v"],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_dual_profile_tp_moe_decoder_engine(
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
    """Build a rank-local Qwen-MoE TRT engine with TP-within-experts.

    Mirrors ``plugin.py::Qwen3MoePlugin.build_engine`` but with TP-shape
    decisions baked in. Quantization is rejected.
    """
    parallel = normalize_parallel_config(parallel_config)
    if not parallel.enabled:
        raise ValueError(
            "dual_profile_decoder_tp_builder requires parallel.mode=tensor_parallel "
            "and tp_size > 1")
    if quant_ctx is not None:
        raise ValueError(
            "Tensor-parallel Qwen-MoE builds do not support quantization yet")

    _validate_moe_tp(weights, parallel, config)
    weights = _shard_qwen_moe_weights(weights, parallel)

    attention_size: int = weights.get("_attention_size", config.attention_size)
    num_experts: int = weights["_num_experts"]
    moe_intermediate: int = weights["_moe_intermediate_size"]
    shared_expert_intermediate: int = weights["_shared_expert_intermediate_size"]
    dense_intermediate: int = weights["_dense_intermediate_size"]
    top_k: int = weights["_num_experts_per_tok"]
    mlp_only_layers: list[int] = weights.get("_mlp_only_layers", [])
    mlp_only_set = set(mlp_only_layers)
    has_shared_expert: bool = weights.get("_has_shared_expert", True)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads // parallel.tp_size
    num_kv_heads = config.num_key_value_heads // parallel.tp_size
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
    attention_window = max_cache_length + 1

    logger = trt.Logger(
        trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if precision == "fp16":
        work_np_dtype, work_trt_dtype = np.float16, trt.float16
    elif precision == "bf16":
        work_np_dtype, work_trt_dtype = np.float16, trt.bfloat16
    else:
        work_np_dtype, work_trt_dtype = np.float32, trt.float32

    # Inputs (single-token decode profile, matching dense build).
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input(
        "attention_mask", trt.float32, (1, attention_window))

    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            work_trt_dtype, (max_cache_length, kv_attention_size))
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            work_trt_dtype, (max_cache_length, kv_attention_size))
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    if work_trt_dtype != trt.float32:
        attention_mask = network.add_cast(
            attention_mask, work_trt_dtype).get_output(0)

    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype)

    graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
    cos_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, True)
    sin_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, False)
    cos_half_tensor = graph_ops.add_constant(
        network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype)
    sin_half_tensor = graph_ops.add_constant(
        network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype)
    eps_tensor = graph_ops.add_constant(
        network, (1, 1),
        np.array([config.rms_norm_eps], dtype=work_np_dtype),
        dtype=work_np_dtype)

    gather = network.add_gather(embedding_table, token_id, 0)
    hidden_state = gather.get_output(0)
    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    present_k_outputs = []
    present_v_outputs = []
    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        is_dense = layer_idx in mlp_only_set
        result = _add_tp_qwen3_moe_decoder_layer(
            network=network,
            hidden=hidden_state,
            cache_k=cache_k_inputs[layer_idx],
            cache_v=cache_v_inputs[layer_idx],
            attention_mask=attention_mask,
            position_id=position_id,
            cos_half_tensor=cos_half_tensor,
            sin_half_tensor=sin_half_tensor,
            eps_tensor=eps_tensor,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden,
            attention_size=attention_size,
            kv_attention_size=kv_attention_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_cache_length=max_cache_length,
            is_dense=is_dense,
            num_experts=num_experts,
            moe_intermediate=moe_intermediate,
            shared_expert_intermediate=shared_expert_intermediate,
            dense_intermediate=dense_intermediate,
            top_k=top_k,
            tp_size=parallel.tp_size,
            has_shared_expert=has_shared_expert,
            dtype=work_np_dtype,
        )
        hidden_state = result["hidden"]
        present_k_outputs.append(result["present_k"])
        present_v_outputs.append(result["present_v"])
        if debug_layer_outputs:
            _mark_debug_output(network, result["post_attn"],
                               f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state,
                               f"debug_hidden_{layer_idx}")

    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(
            network, hidden_state, hidden, final_norm,
            None, eps_tensor, "rmsnorm", dtype=work_np_dtype)

    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, weights["w_out"],
        dtype=work_np_dtype)
    b_out = np.zeros(vocab, dtype=work_np_dtype)
    logits = graph_ops.add_bias_sum(
        network, logits, vocab, b_out, dtype=work_np_dtype)
    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)

    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    if verbose:
        print(f"[trtmc-build] Building Qwen-MoE TP TRT engine "
              f"(layers={num_layers}, hidden={hidden}, attn={attention_size}, "
              f"experts={num_experts}, top_k={top_k}, "
              f"moe_inter={moe_intermediate}, "
              f"shared_inter={shared_expert_intermediate}, "
              f"dense_inter={dense_intermediate}, "
              f"shared_expert={has_shared_expert}, "
              f"cache={max_cache_length}, precision={precision}, "
              f"tp={parallel.tp_size}, rank={parallel.rank}) ...",
              file=sys.stderr)

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("Qwen-MoE tensor-parallel engine build failed")
    return bytes(plan)
