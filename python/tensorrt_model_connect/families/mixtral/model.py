# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixtral family model — Mixture of Experts with standard top-k softmax routing.

Experimental support for Mixtral models (Mistral AI) with block-sparse MoE layers.

Mixtral uses standard decoder attention (RMSNorm + RoPE + GQA, no biases)
but replaces the SwiGLU MLP with a router + N expert MLPs. The router uses
standard top-k softmax to select top-2 experts per token. Selected expert
weights are renormalized to sum to 1.0.

Key differences from Phi-MoE:
  - RMSNorm (no biases) instead of LayerNorm
  - No biases on attention projections
  - Standard top-k softmax routing (not SparseMixer)
  - Separate Q/K/V/O projections with GQA

Weight key mapping:
  HF: model.layers.{i}.block_sparse_moe.gate.weight         -> router [num_experts, hidden]
  HF: model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight -> expert gate [inter, hidden]
  HF: model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight -> expert up   [inter, hidden]
  HF: model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight -> expert down [hidden, inter]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from . import graph_blocks
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .standard_decoder_builder import _apply_norm, _mark_debug_output


trt = trt_compat.get_trt()

name = "mixtral"
runtime_strategy = "mixtral_decoder_moe"
runtime_capabilities = {"decoder_kv"}


def matches(config: object) -> bool:
    """Return whether this module owns the parsed model config."""
    model_type = str(getattr(config, "model_type", config))
    return model_type.lower() == "mixtral"


def load_weights(model_dir: str, config: ModelConfig, *, precision: str = "fp32") -> WeightDict:
    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_experts = config.raw.get("num_local_experts", 8)
    intermediate_size = config.intermediate_size

    weights = WeightDict()

    # Embedding
    embedding = _load_tensor(readers, "model.embed_tokens.weight")
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
    )
    weights["embedding"] = embedding.astype(np.float32)

    attention_size = 0

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hf_prefix = f"model.layers.{layer_idx}"

        # RMSNorm weights (no biases)
        input_norm = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
        weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)

        post_norm = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
        weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

        # Q/K/V/O projections (separate, no biases)
        q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
        k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
        v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
        o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")

        if attention_size == 0:
            attention_size = q_raw.shape[0]

        q_t = _transpose_2d(q_raw, "q_proj")
        k_t = _transpose_2d(k_raw, "k_proj")
        v_t = _transpose_2d(v_raw, "v_proj")
        o_t = _transpose_2d(o_raw, "o_proj")
        del q_raw, k_raw, v_raw, o_raw

        weights[f"{prefix}.w_q"] = q_t
        weights[f"{prefix}.w_k"] = k_t
        weights[f"{prefix}.w_v"] = v_t
        weights[f"{prefix}.w_o"] = o_t

        # Router weight
        router_raw = _load_tensor(readers, f"{hf_prefix}.block_sparse_moe.gate.weight")
        weights[f"{prefix}.router"] = _transpose_2d(router_raw, "router")
        del router_raw

        # Per-expert weights
        for e in range(num_experts):
            exp_prefix = f"{hf_prefix}.block_sparse_moe.experts.{e}"
            w1_raw = _load_tensor(readers, f"{exp_prefix}.w1.weight")
            w3_raw = _load_tensor(readers, f"{exp_prefix}.w3.weight")
            w2_raw = _load_tensor(readers, f"{exp_prefix}.w2.weight")

            weights[f"{prefix}.expert.{e}.w_gate"] = _transpose_2d(w1_raw, f"expert_{e}_gate")
            weights[f"{prefix}.expert.{e}.w_up"] = _transpose_2d(w3_raw, f"expert_{e}_up")
            weights[f"{prefix}.expert.{e}.w_down"] = _transpose_2d(w2_raw, f"expert_{e}_down")
            del w1_raw, w3_raw, w2_raw

    # Final norm
    final_norm_key = "model.norm.weight"
    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)

    # LM head
    lm_head_key = "lm_head.weight"
    if _has_tensor(readers, lm_head_key):
        weights["w_out"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
    else:
        weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

    weights["_attention_size"] = attention_size  # type: ignore[assignment]
    weights["_num_experts"] = num_experts  # type: ignore[assignment]
    weights["_moe_intermediate_size"] = intermediate_size  # type: ignore[assignment]
    weights["_num_experts_per_tok"] = config.raw.get("num_experts_per_tok", 2)  # type: ignore[assignment]

    return weights


def build_engine(
    config: ModelConfig,
    weights: WeightDict,
    max_cache_length: int,
    *,
    precision: str = "fp32",
    quant_ctx=None,
    verbose: bool = False,
    debug_layer_outputs: bool = False,
    parallel_config=None,
) -> bytes:
    parallel = normalize_parallel_config(parallel_config)
    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="Mixtral tensor-parallel builds")
        from .tp_builder import build_mixtral_tp_engine

        return build_mixtral_tp_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
            parallel_config=parallel,
        )

    attention_size: int = weights.get("_attention_size", config.attention_size)
    num_experts: int = weights["_num_experts"]
    moe_intermediate: int = weights["_moe_intermediate_size"]
    top_k: int = weights["_num_experts_per_tok"]
    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = attention_size // num_heads
    kv_attention_size = graph_blocks.infer_kv_attention_size(
        weights, num_kv_heads=num_kv_heads, head_dim=head_dim
    )
    attention_window = max_cache_length + 1

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    trt_config = builder.create_builder_config()
    trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    if precision == "fp16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.float16
    elif precision == "bf16":
        work_np_dtype = np.float16
        work_trt_dtype = trt.bfloat16
    else:
        work_np_dtype = np.float32
        work_trt_dtype = trt.float32

    requested_fp32_layers = frozenset(int(layer) for layer in config.raw.get("_fp32_layers", ()))
    invalid_fp32_layers = sorted(
        layer for layer in requested_fp32_layers if layer < 0 or layer >= num_layers
    )
    if invalid_fp32_layers:
        raise ValueError(f"fp32_layers contains out-of-range indices: {invalid_fp32_layers}")

    # -----------------------------------------------------------
    # Inputs
    # -----------------------------------------------------------
    token_id = network.add_input("token_id", trt.int32, (1,))
    position_id = network.add_input("position_id", trt.int32, (1,))
    attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))

    cache_k_inputs = []
    cache_v_inputs = []
    for i in range(num_layers):
        ck = network.add_input(
            graph_ops.layer_tensor_name("cache_k", i),
            work_trt_dtype,
            (max_cache_length, kv_attention_size),
        )
        cv = network.add_input(
            graph_ops.layer_tensor_name("cache_v", i),
            work_trt_dtype,
            (max_cache_length, kv_attention_size),
        )
        cache_k_inputs.append(ck)
        cache_v_inputs.append(cv)

    if work_trt_dtype != trt.float32:
        attention_mask = network.add_cast(attention_mask, work_trt_dtype).get_output(0)

    # -----------------------------------------------------------
    # Shared constants
    # -----------------------------------------------------------
    embedding_table = graph_ops.add_constant(
        network, (vocab, hidden), weights["embedding"], dtype=work_np_dtype
    )

    graph_ops.validate_native_rope_dim(head_dim, field_name="head_dim")
    cos_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, True
    )
    sin_half_np = graph_ops.make_rope_table_half_dim(
        attention_window, head_dim, config.rope_theta, False
    )
    cos_half_tensor = graph_ops.add_constant(
        network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype
    )
    sin_half_tensor = graph_ops.add_constant(
        network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype
    )

    eps_tensor = graph_ops.add_constant(
        network, (1, 1), np.array([config.rms_norm_eps], dtype=work_np_dtype), dtype=work_np_dtype
    )
    # -----------------------------------------------------------
    # Embedding lookup
    # -----------------------------------------------------------
    gather = network.add_gather(embedding_table, token_id, 0)
    hidden_state = gather.get_output(0)

    if debug_layer_outputs:
        _mark_debug_output(network, hidden_state, "debug_embed")

    # -----------------------------------------------------------
    # Decoder layers
    # -----------------------------------------------------------
    present_k_outputs = []
    present_v_outputs = []

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        layer_is_fp32 = precision == "fp16" and layer_idx in requested_fp32_layers
        layer_np_dtype = np.float32 if layer_is_fp32 else work_np_dtype
        layer_trt_dtype = trt.float32 if layer_is_fp32 else work_trt_dtype

        def layer_cast(tensor):
            if tensor.dtype == layer_trt_dtype:
                return tensor
            return network.add_cast(tensor, layer_trt_dtype).get_output(0)

        result = _add_mixtral_decoder_layer(
            network=network,
            hidden=layer_cast(hidden_state),
            cache_k=layer_cast(cache_k_inputs[layer_idx]),
            cache_v=layer_cast(cache_v_inputs[layer_idx]),
            attention_mask=layer_cast(attention_mask),
            position_id=position_id,
            cos_half_tensor=layer_cast(cos_half_tensor),
            sin_half_tensor=layer_cast(sin_half_tensor),
            eps_tensor=layer_cast(eps_tensor),
            weights=weights,
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
            top_k=top_k,
            dtype=layer_np_dtype,
        )

        hidden_state = result["hidden"]
        present_k = result["present_k"]
        present_v = result["present_v"]
        if layer_is_fp32:
            hidden_state = network.add_cast(hidden_state, work_trt_dtype).get_output(0)
            present_k = network.add_cast(present_k, work_trt_dtype).get_output(0)
            present_v = network.add_cast(present_v, work_trt_dtype).get_output(0)
        present_k_outputs.append(present_k)
        present_v_outputs.append(present_v)

        if debug_layer_outputs:
            _mark_debug_output(network, result["post_attn"], f"debug_post_attn_{layer_idx}")
            _mark_debug_output(network, hidden_state, f"debug_hidden_{layer_idx}")

    # -----------------------------------------------------------
    # Final norm
    # -----------------------------------------------------------
    final_norm = weights.get("final_norm")
    if final_norm is not None and len(final_norm) > 0:
        hidden_state = _apply_norm(
            network,
            hidden_state,
            hidden,
            final_norm,
            None,
            eps_tensor,
            "rmsnorm",
            dtype=work_np_dtype,
        )

    # -----------------------------------------------------------
    # LM head (logits)
    # -----------------------------------------------------------
    logits = graph_ops.add_matmul_rhs_constant(
        network, hidden_state, hidden, vocab, weights["w_out"], dtype=work_np_dtype
    )
    b_out = np.zeros(vocab, dtype=work_np_dtype)
    logits = graph_ops.add_bias_sum(network, logits, vocab, b_out, dtype=work_np_dtype)

    if work_trt_dtype != trt.float32:
        logits = network.add_cast(logits, trt.float32).get_output(0)

    logits.name = "logits"
    network.mark_output(logits)

    # -----------------------------------------------------------
    # Present K/V outputs
    # -----------------------------------------------------------
    for i in range(num_layers):
        pk = present_k_outputs[i]
        pv = present_v_outputs[i]
        pk.name = graph_ops.layer_tensor_name("present_k", i)
        pv.name = graph_ops.layer_tensor_name("present_v", i)
        network.mark_output(pk)
        network.mark_output(pv)

    # -----------------------------------------------------------
    # Build engine
    # -----------------------------------------------------------
    if verbose:
        print(
            f"[trtmc build] Building Mixtral MoE TRT engine "
            f"({num_layers} layers, hidden={hidden}, "
            f"attn={attention_size}, experts={num_experts}, "
            f"top_k={top_k}, inter={moe_intermediate}, "
            f"cache={max_cache_length}) ...",
            file=sys.stderr,
        )

    plan = builder.build_serialized_network(network, trt_config)
    if plan is None:
        raise RuntimeError("TensorRT engine build failed")

    return bytes(plan)


def _add_swiglu_expert(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Compute a single SwiGLU expert: down(silu(gate(x)) * up(x))."""
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_gate, dtype=dtype
    )
    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_up, dtype=dtype
    )

    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(swish.get_output(0), up, trt.ElementWiseOperation.PROD)

    down = graph_ops.add_matmul_rhs_constant(
        network, gated.get_output(0), intermediate_size, hidden_size, w_down, dtype=dtype
    )
    return down


def _add_mixtral_moe_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_experts: int,
    moe_intermediate: int,
    top_k: int = 2,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add Mixture of Experts block with standard top-k softmax routing.

    Steps:
      1. Router logits: inp @ router_weight -> [1, num_experts]
      2. Softmax -> [1, num_experts]
      3. TopK -> top_k indices and values
      4. Renormalize selected weights to sum to 1.0
      5. Compute ALL expert SwiGLU outputs -> [num_experts, hidden]
      6. Gather selected experts, scale, and sum -> [1, hidden]
    """
    # 1. Router logits. HF's linear projection returns the model dtype; the
    # FP32 routing boundary starts after that output has been quantized.
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, num_experts, weights[f"{prefix}.router"], dtype=dtype
    )

    # HF Mixtral deliberately performs the numerically sensitive routing
    # operations in FP32, even when the model weights and hidden states are
    # FP16/BF16. Keeping softmax/top-k in the model dtype can change the
    # selected expert for close router scores, making generation depend on the
    # tactic selected for an otherwise equivalent engine build.
    routing_logits = router_logits
    if routing_logits.dtype != trt.float32:
        routing_logits = network.add_cast(routing_logits, trt.float32).get_output(0)

    # 2. Softmax over router logits
    sm = network.add_softmax(routing_logits)
    sm.axes = 1 << 1

    # 3. TopK selection
    topk = network.add_topk(sm.get_output(0), trt.TopKOperation.MAX, top_k, 1 << 1)
    top_values = topk.get_output(0)  # [1, top_k]
    top_indices = topk.get_output(1)  # [1, top_k]

    # 4. Renormalize: values / sum(values)
    sum_val = network.add_reduce(top_values, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
    norm_weights = network.add_elementwise(
        top_values, sum_val.get_output(0), trt.ElementWiseOperation.DIV
    )  # [1, top_k]

    # 5. Compute ALL expert outputs and stack
    expert_outputs = []
    for e in range(num_experts):
        exp_out = _add_swiglu_expert(
            network,
            inp,
            hidden_size,
            moe_intermediate,
            weights[f"{prefix}.expert.{e}.w_gate"],
            weights[f"{prefix}.expert.{e}.w_up"],
            weights[f"{prefix}.expert.{e}.w_down"],
            dtype=dtype,
        )
        expert_outputs.append(exp_out)

    stacked = network.add_concatenation(expert_outputs)
    stacked.axis = 0
    stacked_out = stacked.get_output(0)  # [num_experts, hidden_size]

    # 6. Gather and scale each selected expert, then sum
    # Extract individual expert indices and weights via slicing
    result = None
    for k in range(top_k):
        # Extract index k: [1, 1]
        idx_slice = network.add_slice(top_indices, start=(0, k), shape=(1, 1), stride=(1, 1))
        idx_flat = network.add_shuffle(idx_slice.get_output(0))
        idx_flat.reshape_dims = (1,)

        # Extract weight k: [1, 1]
        w_slice = network.add_slice(
            norm_weights.get_output(0), start=(0, k), shape=(1, 1), stride=(1, 1)
        )

        # Gather expert output
        expert_out = network.add_gather(stacked_out, idx_flat.get_output(0), 0)

        # HF multiplies the model-dtype expert output by the FP32 routing
        # weight, then converts that contribution back to the hidden-state
        # dtype before accumulating it into the output tensor.
        expert_value = expert_out.get_output(0)
        if expert_value.dtype != trt.float32:
            expert_value = network.add_cast(expert_value, trt.float32).get_output(0)

        # Scale
        scaled_expert = network.add_elementwise(
            expert_value, w_slice.get_output(0), trt.ElementWiseOperation.PROD
        )
        scaled_value = scaled_expert.get_output(0)
        if scaled_value.dtype != inp.dtype:
            scaled_value = network.add_cast(scaled_value, inp.dtype).get_output(0)

        if result is None:
            result = scaled_value
        else:
            sum_layer = network.add_elementwise(
                result, scaled_value, trt.ElementWiseOperation.SUM
            )
            result = sum_layer.get_output(0)

    return result


def _add_mixtral_decoder_layer(
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
    attention_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_length: int,
    num_experts: int,
    moe_intermediate: int,
    top_k: int = 2,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add one decoder layer with Mixtral MoE MLP. Attention is standard."""
    attention_window = max_cache_length + 1

    # Pre-attention RMSNorm
    norm1 = _apply_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        None,
        eps_tensor,
        "rmsnorm",
        dtype=dtype,
    )

    # QKV projections (no biases)
    q = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, attention_size, weights[f"{prefix}.w_q"], dtype=dtype
    )
    k = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, kv_attention_size, weights[f"{prefix}.w_k"], dtype=dtype
    )
    v = graph_ops.add_matmul_rhs_constant(
        network, norm1, hidden_size, kv_attention_size, weights[f"{prefix}.w_v"], dtype=dtype
    )

    q = graph_ops.add_apply_rope_native(
        network, q, num_heads, head_dim, cos_half_tensor, sin_half_tensor, position_id, head_dim
    )
    k = graph_ops.add_apply_rope_native(
        network, k, num_kv_heads, head_dim, cos_half_tensor, sin_half_tensor, position_id, head_dim
    )

    # Save present K/V
    present_k = k
    present_v = v

    # Reshape current K, V for concatenation
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)

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
        q_seq=1,
        kv_seq=attention_window,
        mask=mask_4d,
    )

    # Output projection (no bias)
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context_flat, attention_size, hidden_size, weights[f"{prefix}.w_o"], dtype=dtype
    )

    # Residual connection
    residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)

    # Post-attention RMSNorm
    norm2 = _apply_norm(
        network,
        residual1.get_output(0),
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        None,
        eps_tensor,
        "rmsnorm",
        dtype=dtype,
    )

    # MoE block
    moe_out = _add_mixtral_moe_block(
        network,
        norm2,
        weights,
        prefix,
        hidden_size,
        num_experts,
        moe_intermediate,
        top_k,
        dtype=dtype,
    )

    # Residual connection
    residual2 = network.add_elementwise(
        residual1.get_output(0), moe_out, trt.ElementWiseOperation.SUM
    )

    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": present_k,
        "present_v": present_v,
    }


def supports_split_decoder_roles(config: ModelConfig) -> bool:
    return False


def _dynamic_kv_profile_rows(
    max_cache_length: int,
    kv_budget: int,
    *,
    bucket_rows: int = 32,
    preferred_rows: list[int] | None = None,
) -> list[int]:
    if max_cache_length < 1:
        return [1]
    start = ((max(kv_budget, 1) + bucket_rows - 1) // bucket_rows) * bucket_rows
    start = max(bucket_rows, min(start, max_cache_length))
    rows: list[int] = []

    def add_row(value: int) -> None:
        rounded = (
            (min(max(value, 1), max_cache_length) + bucket_rows - 1) // bucket_rows
        ) * bucket_rows
        rounded = max(bucket_rows, min(rounded, max_cache_length))
        if rounded not in rows:
            rows.append(rounded)

    for value in preferred_rows or ():
        add_row(value)
    row = start
    while row < max_cache_length:
        add_row(row)
        row = (
            (min(max(row + bucket_rows, row * 2), max_cache_length) + bucket_rows - 1)
            // bucket_rows
        ) * bucket_rows
    add_row(max_cache_length)
    return sorted(rows)


def _sanitize_dynamic_kv_profile_rows(
    rows: list[int] | None,
    max_cache_length: int,
) -> list[int] | None:
    if rows is None:
        return None
    sanitized = sorted({max(1, min(int(value), max_cache_length)) for value in rows})
    if not sanitized:
        raise ValueError("dynamic_kv_profile_rows_override must contain at least one row")
    return sanitized


def build(model_dir: str, output_path: str, **options) -> None:
    """Build a complete mixtral bundle from checkpoint to serialized artifact."""
    import json
    import time
    from dataclasses import replace
    from datetime import datetime, timezone
    from pathlib import Path

    from tensorrt_model_connect import trt_compat
    from tensorrt_model_connect.build_timing import (
        add_build_timing,
        new_build_timing,
        write_build_timing,
    )
    from tensorrt_model_connect.bundle_writer import (
        BundleInfo,
        BundleSection,
        gpu_name,
        tensorrt_abi,
        tensorrt_version,
        write_bundle,
    )
    from tensorrt_model_connect.parallel_config import (
        normalize_parallel_config,
        rank_engine_section,
        require_tensorrt_11_for_tensor_parallel,
    )

    model_path = Path(model_dir)
    decoder_engine_layout = str(options.get("decoder_engine_layout") or "split")
    if decoder_engine_layout not in {"split", "dual_profile"}:
        raise ValueError(
            "decoder_engine_layout must be 'split' or 'dual_profile', "
            f"got {decoder_engine_layout!r}"
        )
    parallel = normalize_parallel_config(options.get("parallel_config"))
    if parallel.cp_enabled:
        raise NotImplementedError("mixtral does not support context-parallel builds")

    config = ModelConfig.from_dir(model_path)
    config.raw["_model_dir"] = str(model_path)
    config.raw["_decoder_engine_layout"] = decoder_engine_layout
    config.raw["_fp32_layers"] = sorted(set(options.get("fp32_layers") or ()))
    config.raw["_family_build_options"] = dict(options.get("family_build_options") or {})
    config.raw["_parallel_build_enabled"] = bool(parallel.enabled)
    config.raw["_rtx_build_requested"] = bool(options.get("rtx"))
    config.raw["_runtime_dynamic_kv_requested"] = bool(
        options.get("dynamic_kv_cache") or options.get("triattention_stats_path")
    )
    config.raw["_quantized_build_requested"] = bool(options.get("quantize"))
    config.raw["_disable_dual_profile_decoder"] = True

    precision = str(options.get("precision") or "fp32").lower()
    config.raw["_resolved_build_precision"] = precision
    requested_cache_length = options.get("max_cache_length")
    max_cache_length = int(256) if requested_cache_length is None else int(requested_cache_length)
    if max_cache_length < 1:
        raise ValueError("max_cache_length must be >= 1")

    timing = new_build_timing(options.get("build_timing_path"))
    timing["model_dir"] = str(model_path)
    timing["output_path"] = str(output_path)
    started = time.monotonic()
    write_build_timing(timing)

    weights_started = time.monotonic()
    weights = load_weights(str(model_path), config, precision=precision)
    add_build_timing(timing, "weights_loading_s", time.monotonic() - weights_started)
    write_build_timing(timing)

    quantize = options.get("quantize")
    quant_ctx = None
    quant_plan = None
    if quantize:
        from tensorrt_model_connect.quantization import QuantPlan, build_quant_context
        from . import graph_ops

        quant_plan = QuantPlan.from_build_args(
            precision=precision,
            quantize=str(quantize),
            quant_scales=options.get("quant_scales"),
            quant_calibration_samples=int(options.get("quant_calibration_samples") or 512),
        )
        quant_method = str(
            config.raw.get("quantization_config", {}).get("quant_method", "")
        ).lower()
        if quant_plan.scale_source == "modelopt" and quant_method in {
            "awq",
            "gptq",
            "compressed-tensors",
            "compressed_tensors",
        }:
            quant_plan = replace(quant_plan, scale_source="prequantized")
        quant_ctx = build_quant_context(
            format_name=quant_plan.quant_format,
            model_dir=str(model_path),
            config=config,
            scales_json=options.get("quant_scales"),
            num_calibration_samples=int(options.get("quant_calibration_samples") or 512),
            quant_plan=quant_plan,
            graph_ops=graph_ops,
        )

    dynamic_kv_cache = bool(options.get("dynamic_kv_cache"))
    dynamic_kv_budget = 1
    triattention_config = None
    triattention_section = None
    if options.get("triattention_stats_path"):
        from tensorrt_model_connect.triattention_export import (
            TriAttentionBundleConfig,
            export_triattention_stats_section,
        )

        recent_window = int(options.get("triattention_recent_window", 128))
        divide_length = int(options.get("triattention_divide_length", 128))
        score_aggregation = str(options.get("triattention_score_aggregation", "mean"))
        if recent_window < 0:
            raise ValueError("TriAttention recent_window must be >= 0")
        if divide_length < 1:
            raise ValueError("TriAttention divide_length must be >= 1")
        if score_aggregation not in {"mean", "max"}:
            raise ValueError("TriAttention score_aggregation must be 'mean' or 'max'")
        requested_budget = options.get("triattention_kv_budget")
        dynamic_kv_budget = max_cache_length if requested_budget is None else int(requested_budget)
        if not 1 <= dynamic_kv_budget <= max_cache_length:
            raise ValueError("TriAttention kv_budget must fit max_cache_length")
        triattention_config = TriAttentionBundleConfig(
            kv_budget=dynamic_kv_budget,
            divide_length=divide_length,
            recent_window=recent_window,
            score_aggregation=score_aggregation,
            count_prompt_tokens=bool(options.get("triattention_count_prompt_tokens", True)),
            protect_prefill=bool(options.get("triattention_protect_prefill", True)),
            disable_mlr=bool(options.get("triattention_disable_mlr", False)),
            disable_trig=bool(options.get("triattention_disable_trig", False)),
        )
        triattention_section = export_triattention_stats_section(
            str(options["triattention_stats_path"]),
            config=config,
        )
        dynamic_kv_cache = True

    if dynamic_kv_cache:
        rows = _sanitize_dynamic_kv_profile_rows(
            options.get("dynamic_kv_profile_rows_override"),
            max_cache_length,
        )
        if rows is None:
            preferred_rows = (
                [max(32, dynamic_kv_budget // 2)]
                if triattention_config is not None and dynamic_kv_budget >= 4096
                else None
            )
            rows = _dynamic_kv_profile_rows(
                max_cache_length,
                dynamic_kv_budget,
                preferred_rows=preferred_rows,
            )
        config.raw["dynamic_kv_cache"] = True
        config.raw["_dynamic_kv_opt_length"] = max_cache_length
        config.raw["_dynamic_kv_profile_rows"] = rows

    if parallel.enabled:
        require_tensorrt_11_for_tensor_parallel(parallel, feature="mixtral tensor-parallel builds")

        if quant_ctx is not None:
            raise ValueError("mixtral tensor-parallel builds do not support quantization")
        if dynamic_kv_cache:
            raise NotImplementedError(
                "Tensor-parallel mixtral builds do not support dynamic KV cache or TriAttention"
            )

    verbose = bool(options.get("verbose"))

    def build_role(role: str, *, rank_parallel=parallel) -> bytes:
        from tensorrt_model_connect.tvm_ffi.graph_build import engine_role

        previous = config.raw.get("_decoder_engine_role")
        config.raw["_decoder_engine_role"] = role
        try:
            with engine_role(role):
                return build_engine(
                    config,
                    weights,
                    max_cache_length,
                    precision=precision,
                    quant_ctx=quant_ctx,
                    verbose=verbose,
                    parallel_config=rank_parallel,
                )
        finally:
            if previous is None:
                config.raw.pop("_decoder_engine_role", None)
            else:
                config.raw["_decoder_engine_role"] = previous

    from tensorrt_model_connect.tvm_ffi.graph_build import inspection_role

    inspection = inspection_role()
    if inspection is not None:
        build_role(inspection)
        raise RuntimeError("graph inspection did not reach TensorRT serialization")

    compile_started = time.monotonic()
    if parallel.enabled:
        plans = {
            rank: build_role("dual_profile", rank_parallel=parallel.for_rank(rank))
            for rank in range(parallel.tp_size)
        }
        sections = [
            BundleSection(rank_engine_section(rank), plan) for rank, plan in sorted(plans.items())
        ]
        decoder_layout = "dual_profile"
    else:
        split = (
            decoder_engine_layout == "split"
            and not dynamic_kv_cache
            and supports_split_decoder_roles(config)
        )
        if split:
            previous_active = config.raw.get("_active_split_decoder_build")
            config.raw["_active_split_decoder_build"] = True
            try:
                quant_label = str(quantize or "noquant")

                def build_split_role(role: str) -> bytes:
                    scope = (
                        f"split-{config.model_type}-h{config.hidden_size}"
                        f"-l{config.num_hidden_layers}-{precision}-{quant_label}-{role}"
                    )
                    with trt_compat.scoped_timing_cache(scope):
                        return build_role(role)

                prefill_started = time.monotonic()
                prefill_plan = build_split_role("prefill")
                add_build_timing(
                    timing,
                    "trt_compile_prefill_engine_s",
                    time.monotonic() - prefill_started,
                )
                decode_started = time.monotonic()
                plan = build_split_role("decode")
                add_build_timing(
                    timing,
                    "trt_compile_decode_engine_s",
                    time.monotonic() - decode_started,
                )
            finally:
                if previous_active is None:
                    config.raw.pop("_active_split_decoder_build", None)
                else:
                    config.raw["_active_split_decoder_build"] = previous_active
            sections = [
                BundleSection("engine_plan", plan),
                BundleSection("prefill_engine_plan", prefill_plan),
            ]
            decoder_layout = "split"
        else:
            role = "dual_profile" if decoder_engine_layout == "dual_profile" else "decode"
            plan = build_role(role)
            sections = [BundleSection("engine_plan", plan)]
            decoder_layout = "dual_profile" if role == "dual_profile" else "single"
    compile_elapsed = time.monotonic() - compile_started
    add_build_timing(timing, "trt_compile_s", compile_elapsed)
    add_build_timing(timing, "trt_compile_main_engine_s", compile_elapsed)
    write_build_timing(timing)

    if triattention_config is not None and triattention_section is not None:
        sections.append(BundleSection(triattention_config.stats_section, triattention_section))

    from tensorrt_model_connect.tokenizer_conversion import (
        prepare_tokenizer_special_frame,
    )

    tokenizer_frame = prepare_tokenizer_special_frame(
        model_path,
        source_model_id_or_path=options.get("tokenizer_source_model_id_or_path"),
        source_revision=options.get("tokenizer_source_revision"),
    )
    prefix_ids, suffix_ids = tokenizer_frame or ([], [])
    add_special_tokens = bool(prefix_ids or suffix_ids)

    trt_version = tensorrt_version()
    trt_abi = tensorrt_abi(trt_version)
    info = BundleInfo(
        model_id=model_path.name,
        model_type=config.model_type,
        family=name,
        trt_version=trt_version,
        trt_abi=trt_abi,
        gpu_name=gpu_name(),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_cache_length=max_cache_length,
        runtime_strategy=runtime_strategy,
        precision=precision,
        quantization=(quant_plan.quant_format if quant_plan else "none"),
        tokenizer_add_special_tokens=add_special_tokens,
    )

    source_config = model_path / "config.json"
    runtime_config = (
        json.loads(source_config.read_text(encoding="utf-8"))
        if source_config.is_file()
        else {key: value for key, value in config.raw.items() if not str(key).startswith("_")}
    )
    generation_config = model_path / "generation_config.json"
    if generation_config.is_file():
        generation = json.loads(generation_config.read_text(encoding="utf-8"))
        if "eos_token_id" in generation:
            runtime_config["eos_token_id"] = generation["eos_token_id"]
    runtime_config.update(
        {
            "runtime_strategy": runtime_strategy,
            "engine_backend": "trt_rtx" if options.get("rtx") else "trt",
            "trt_version": trt_version,
            "precision": precision,
            "tokenizer_add_special_tokens": int(add_special_tokens),
            "decoder_engine_layout": decoder_layout,
        }
    )
    if trt_abi:
        runtime_config["trt_abi"] = trt_abi
    if options.get("fp32_layers"):
        runtime_config["fp32_layers"] = sorted(set(options["fp32_layers"]))
    if tokenizer_frame is not None:
        runtime_config["tokenizer_special_prefix_ids"] = prefix_ids
        runtime_config["tokenizer_special_suffix_ids"] = suffix_ids
    if quant_plan is not None:
        runtime_config["quantization"] = quant_plan.as_config_dict()
    if dynamic_kv_cache:
        runtime_config["dynamic_kv_cache"] = True
        runtime_config["dynamic_kv_profile_rows"] = config.raw["_dynamic_kv_profile_rows"]
    if triattention_config is not None:
        runtime_config["triattention"] = triattention_config.to_dict()
    runtime_config.update(parallel.to_bundle_config_fields())

    tokenizer_override = None
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "vocab.txt",
        "special_tokens_map.json",
        "tokenizer.model",
    ):
        path = model_path / filename
        if filename == "tokenizer.json" and tokenizer_override is not None:
            sections.append(BundleSection(filename, tokenizer_override))
        elif path.is_file():
            sections.append(BundleSection(filename, path.read_bytes()))
    sections.append(
        BundleSection(
            "config.json",
            json.dumps(runtime_config, indent=2).encode("utf-8"),
        )
    )

    from tensorrt_model_connect.tvm_ffi.graph_build import kernel_slots_section

    slot_section = kernel_slots_section()
    if slot_section is not None:
        sections.append(BundleSection("kernel_slots.json", slot_section))
    kernel_manifest = []
    for global_name, library in options.get("kernel_artifacts") or ():
        section_name = f"kernel_{global_name.replace('.', '_')}.so"
        sections.append(BundleSection(section_name, Path(library).read_bytes()))
        kernel_manifest.append(
            {"global_name": global_name, "func_name": "run", "section": section_name}
        )
    if kernel_manifest:
        sections.append(
            BundleSection(
                "kernel_manifest.json",
                json.dumps({"kernels": kernel_manifest}).encode("utf-8"),
            )
        )

    write_started = time.monotonic()
    write_bundle(output_path, info, sections)
    add_build_timing(timing, "bundle_write_s", time.monotonic() - write_started)
    timing["total_s"] = time.monotonic() - started
    write_build_timing(timing)
