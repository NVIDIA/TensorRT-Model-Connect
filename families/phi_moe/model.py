# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-MoE family plugin — Mixture of Experts with SparseMixer routing.

Phi-MoE uses the standard decoder attention (RoPE + GQA) but replaces the
SwiGLU MLP with a router + N expert MLPs. The router uses SparseMixer
(not standard top-k softmax) to select top-2 experts per token. Each
expert's weight is computed from an independent masked softmax over all
logits, so the weights do NOT sum to 1.0.

Key differences from standard Phi-3:
  - LayerNorm (with bias) instead of RMSNorm
  - Separate Q/K/V/O projections (not fused) with biases
  - MoE block: router + 16 experts, each a SwiGLU MLP
  - lm_head has bias

Weight key mapping:
  HF: model.layers.{i}.block_sparse_moe.gate.weight         -> router [num_experts, hidden]
  HF: model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight -> expert gate [inter, hidden]
  HF: model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight -> expert up   [inter, hidden]
  HF: model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight -> expert down [hidden, inter]
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt

from .config import ModelConfig
from .parallel import ParallelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from . import graph_ops
from . import graph_blocks
from .parallel import normalize_parallel_config
from .default_decoder import _apply_norm, _mark_debug_output


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _PhiMoEModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        """Load Phi-MoE weights: standard attention + per-expert MLP weights."""
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_experts = config.raw.get("num_local_experts", 16)
        intermediate_size = config.intermediate_size  # per-expert intermediate

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

            # LayerNorm weights + biases
            input_norm = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)

            input_norm_bias_key = f"{hf_prefix}.input_layernorm.bias"
            if _has_tensor(readers, input_norm_bias_key):
                weights[f"{prefix}.input_norm_beta"] = _load_tensor(
                    readers, input_norm_bias_key
                ).astype(np.float32)

            post_norm = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            post_norm_bias_key = f"{hf_prefix}.post_attention_layernorm.bias"
            if _has_tensor(readers, post_norm_bias_key):
                weights[f"{prefix}.post_attn_norm_beta"] = _load_tensor(
                    readers, post_norm_bias_key
                ).astype(np.float32)

            # Q/K/V/O projections (separate, not fused) with biases
            q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")

            if attention_size == 0:
                attention_size = q_raw.shape[0]

            # Transpose [out, in] -> [in, out]
            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")
            o_t = _transpose_2d(o_raw, "o_proj")
            del q_raw, k_raw, v_raw, o_raw

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t

            # Attention biases
            for proj, tag in [
                ("q_proj", "q_bias"),
                ("k_proj", "k_bias"),
                ("v_proj", "v_bias"),
                ("o_proj", "o_bias"),
            ]:
                bias_key = f"{hf_prefix}.self_attn.{proj}.bias"
                if _has_tensor(readers, bias_key):
                    raw = _load_tensor(readers, bias_key).astype(np.float32)
                    weights[f"{prefix}.{tag}"] = raw

            # Router weight
            router_raw = _load_tensor(readers, f"{hf_prefix}.block_sparse_moe.gate.weight")
            # Shape: [num_experts, hidden] — transpose to [hidden, num_experts]
            weights[f"{prefix}.router"] = _transpose_2d(router_raw, "router")
            del router_raw

            # Per-expert weights
            for e in range(num_experts):
                exp_prefix = f"{hf_prefix}.block_sparse_moe.experts.{e}"
                # w1 = gate projection [intermediate, hidden]
                w1_raw = _load_tensor(readers, f"{exp_prefix}.w1.weight")
                # w3 = up projection [intermediate, hidden]
                w3_raw = _load_tensor(readers, f"{exp_prefix}.w3.weight")
                # w2 = down projection [hidden, intermediate]
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

        final_norm_bias_key = "model.norm.bias"
        if _has_tensor(readers, final_norm_bias_key):
            weights["final_norm_beta"] = _load_tensor(readers, final_norm_bias_key).astype(
                np.float32
            )

        # LM head (weight + bias)
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

        lm_head_bias_key = "lm_head.bias"
        if _has_tensor(readers, lm_head_bias_key):
            weights["lm_head_bias"] = _load_tensor(readers, lm_head_bias_key).astype(np.float32)

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_num_experts"] = num_experts  # type: ignore[assignment]
        weights["_moe_intermediate_size"] = intermediate_size  # type: ignore[assignment]
        weights["_num_experts_per_tok"] = config.raw.get("num_experts_per_tok", 2)  # type: ignore[assignment]

        return weights

    def build_engine(
        self,
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
        """Build TRT engine with MoE layers.

        The attention is standard (reuses _add_decoder_layer logic), but the MLP
        is replaced with MoE routing + expert dispatch.
        """
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            parallel.validate()
            from .tp_builder import build_phi_moe_tp_engine

            return build_phi_moe_tp_engine(
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
        norm_type = "layernorm"
        jitter_eps = config.raw.get("router_jitter_noise", 0.01)
        if precision == "fp16":
            work_np_dtype, work_trt_dtype = np.float16, trt.float16
        elif precision == "fp32":
            work_np_dtype, work_trt_dtype = np.float32, trt.float32
        else:
            raise ValueError(f"Unsupported Phi-MoE precision: {precision}")

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()

        # -----------------------------------------------------------
        # Inputs
        # -----------------------------------------------------------
        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input("attention_mask", trt.float32, (1, attention_window))
        attention_mask_work = attention_mask
        if work_trt_dtype != trt.float32:
            attention_mask_work = network.add_cast(attention_mask, work_trt_dtype).get_output(0)

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
            network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32)
        )
        # -----------------------------------------------------------
        # Embedding lookup
        # -----------------------------------------------------------
        gather = network.add_gather(embedding_table, token_id, 0)
        hidden_state = gather.get_output(0)  # [1, hidden]

        if debug_layer_outputs:
            _mark_debug_output(network, hidden_state, "debug_embed")

        # -----------------------------------------------------------
        # Decoder layers
        # -----------------------------------------------------------
        present_k_outputs = []
        present_v_outputs = []

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"

            result = _add_moe_decoder_layer(
                network=network,
                hidden=hidden_state,
                cache_k=cache_k_inputs[layer_idx],
                cache_v=cache_v_inputs[layer_idx],
                attention_mask=attention_mask_work,
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
                num_experts=num_experts,
                moe_intermediate=moe_intermediate,
                top_k=top_k,
                jitter_eps=jitter_eps,
                norm_type=norm_type,
                dtype=work_np_dtype,
                work_trt_dtype=work_trt_dtype,
            )

            hidden_state = result["hidden"]
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])

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
                weights.get("final_norm_beta"),
                eps_tensor,
                norm_type,
                dtype=work_np_dtype,
            )

        # -----------------------------------------------------------
        # LM head (logits)
        # -----------------------------------------------------------
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_out"], dtype=work_np_dtype
        )

        # LM head bias
        lm_bias = weights.get("lm_head_bias")
        if lm_bias is not None:
            logits = graph_ops.add_bias_sum(network, logits, vocab, lm_bias, dtype=work_np_dtype)
        else:
            b_out = np.zeros(vocab, dtype=work_np_dtype)
            logits = graph_ops.add_bias_sum(network, logits, vocab, b_out, dtype=work_np_dtype)
        if logits.dtype != trt.float32:
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
                f"[trtmc build] Building MoE TRT engine ({num_layers} layers, "
                f"hidden={hidden}, attn={attention_size}, "
                f"experts={num_experts}, top_k={top_k}, "
                f"inter={moe_intermediate}, "
                f"cache={max_cache_length}, precision={precision}) ...",
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


def _sparsemixer_weight(
    network: trt.INetworkDefinition,
    scores: trt.ITensor,
    num_experts: int,
    jitter_eps: float,
    original_scores: trt.ITensor | None = None,
    dtype: np.dtype = np.float32,
    work_trt_dtype=None,
) -> tuple[trt.ITensor, trt.ITensor]:
    """Compute one expert selection via SparseMixer (inference mode).

    Replicates the HF ``sparsemixer()`` function for a single expert:
      1. max_val, max_ind = max(scores)
      2. factor = clamp(|original_scores|, min=max_val)
      3. mask = ((max_val - original_scores) / factor) > (2 * jitter_eps)
      4. masked = where(mask, -inf, scores)
      5. weight = softmax(masked)[max_ind]

    HF uses the ORIGINAL unmasked scores for factor and threshold even
    for the second expert selection.  The ``original_scores`` parameter
    carries the unmasked router logits.

    Args:
        scores: [1, num_experts] router logits (may contain -inf for
                previously selected experts).
        num_experts: Number of experts.
        jitter_eps: Router jitter epsilon from config.
        original_scores: Original unmasked router logits for factor/threshold.
                If None, uses scores (first expert selection).

    Returns:
        (weight, index) where weight uses the graph compute dtype and index is
        [1, 1] int32.
    """
    if original_scores is None:
        original_scores = scores
    if work_trt_dtype is None:
        work_trt_dtype = trt.float32

    # max_val [1, 1], max_ind [1, 1]  — from (potentially masked) scores
    topk1 = network.add_topk(scores, trt.TopKOperation.MAX, 1, 1 << 1)
    max_val = topk1.get_output(0)  # [1, 1]
    max_ind = topk1.get_output(1)  # [1, 1]

    # factor = clamp(|original_scores|, min=max_val)
    # HF uses original (unmasked) scores for abs(), not the masked scores.
    abs_scores = network.add_unary(original_scores, trt.UnaryOperation.ABS)
    factor = network.add_elementwise(
        abs_scores.get_output(0), max_val, trt.ElementWiseOperation.MAX
    )

    # (max_val - original_scores) / factor
    # HF uses original scores here too.
    diff = network.add_elementwise(max_val, original_scores, trt.ElementWiseOperation.SUB)
    ratio = network.add_elementwise(
        diff.get_output(0), factor.get_output(0), trt.ElementWiseOperation.DIV
    )

    # > 2 * jitter_eps  (boolean mask)
    threshold = graph_ops.add_constant(
        network, (1, 1), np.array([2.0 * jitter_eps], dtype=dtype), dtype=dtype
    )
    mask_float = network.add_elementwise(
        ratio.get_output(0), threshold, trt.ElementWiseOperation.GREATER
    )  # bool tensor

    # where(mask, -inf, scores)  ->  scores + mask * (-inf - scores)
    # Simpler: mask * -1e9 + (1 - mask) * 0 added to scores
    # Actually: just add mask * -1e9 to scores, where mask=1 for masked positions
    neginf = graph_ops.add_constant(
        network, (1, 1), np.array([np.finfo(dtype).min], dtype=dtype), dtype=dtype
    )
    # Cast bool mask to float
    mask_f = network.add_cast(mask_float.get_output(0), work_trt_dtype)
    penalty = network.add_elementwise(mask_f.get_output(0), neginf, trt.ElementWiseOperation.PROD)
    masked = network.add_elementwise(scores, penalty.get_output(0), trt.ElementWiseOperation.SUM)

    # softmax over masked logits
    sm = network.add_softmax(masked.get_output(0))
    sm.axes = 1 << 1
    sm_out = sm.get_output(0)  # [1, num_experts]

    # Gather the weight at max_ind: reshape max_ind to scalar
    idx_flat = network.add_shuffle(max_ind)
    idx_flat.reshape_dims = (1,)
    weight = network.add_gather(sm_out, idx_flat.get_output(0), 1)
    # weight shape: [1, 1]

    return weight.get_output(0), max_ind


def _add_moe_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_experts: int,
    moe_intermediate: int,
    top_k: int,
    jitter_eps: float = 0.01,
    dtype: np.dtype = np.float32,
    work_trt_dtype=None,
) -> trt.ITensor:
    """Add Mixture of Experts block with SparseMixer routing (top-2).

    Dense implementation: computes all expert outputs, then selects top-2
    via SparseMixer routing weights. The SparseMixer algorithm computes
    each expert's weight from an independent softmax (weights do NOT
    sum to 1.0).

    Steps:
      1. Router logits: inp @ router_weight -> [1, num_experts]
      2. SparseMixer expert 1: masked softmax -> weight_1, index_1
      3. Scatter -inf at index_1, SparseMixer expert 2 -> weight_2, index_2
      4. Compute all expert SwiGLU outputs -> [num_experts, hidden]
      5. Gather selected experts and apply weights
      6. Weighted sum -> [1, hidden]
    """
    if work_trt_dtype is None:
        work_trt_dtype = trt.float32

    # 1. Router logits
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, num_experts, weights[f"{prefix}.router"], dtype=dtype
    )  # [1, num_experts]

    # 2. SparseMixer expert 1 selection
    weight_1, idx_1 = _sparsemixer_weight(
        network, router_logits, num_experts, jitter_eps, dtype=dtype, work_trt_dtype=work_trt_dtype
    )
    # weight_1: [1, 1], idx_1: [1, 1]

    # 3. Mask out expert 1 for second selection
    # Create one-hot of idx_1: [1, num_experts]
    idx_1_flat = network.add_shuffle(idx_1)
    idx_1_flat.reshape_dims = (1,)
    range_const = graph_ops.add_constant(
        network, (1, num_experts), np.arange(num_experts, dtype=dtype).reshape(1, -1), dtype=dtype
    )
    idx_1_broadcast = network.add_shuffle(idx_1_flat.get_output(0))
    idx_1_broadcast.reshape_dims = (1, 1)
    # Cast idx to float for comparison
    idx_1_f = network.add_cast(idx_1_broadcast.get_output(0), work_trt_dtype)
    # one_hot_mask: 1 where expert == idx_1, 0 elsewhere
    eq = network.add_elementwise(range_const, idx_1_f.get_output(0), trt.ElementWiseOperation.EQUAL)
    eq_f = network.add_cast(eq.get_output(0), work_trt_dtype)
    # Subtract large value at expert 1 position
    neginf_mask = graph_ops.add_constant(
        network, (1, 1), np.array([np.finfo(dtype).min], dtype=dtype), dtype=dtype
    )
    penalty = network.add_elementwise(
        eq_f.get_output(0), neginf_mask, trt.ElementWiseOperation.PROD
    )
    scores_2 = network.add_elementwise(
        router_logits, penalty.get_output(0), trt.ElementWiseOperation.SUM
    )

    # 4. SparseMixer expert 2 selection — pass original router_logits
    # for the factor/threshold computation (HF uses unmasked scores).
    weight_2, idx_2 = _sparsemixer_weight(
        network,
        scores_2.get_output(0),
        num_experts,
        jitter_eps,
        original_scores=router_logits,
        dtype=dtype,
        work_trt_dtype=work_trt_dtype,
    )

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
        )  # [1, hidden_size]
        expert_outputs.append(exp_out)

    # Stack: [num_experts, hidden_size]
    stacked = network.add_concatenation(expert_outputs)
    stacked.axis = 0
    stacked_out = stacked.get_output(0)  # [num_experts, hidden_size]

    # 6. Gather expert 1 output and scale
    idx_1_scalar = network.add_shuffle(idx_1)
    idx_1_scalar.reshape_dims = (1,)
    expert_1_out = network.add_gather(stacked_out, idx_1_scalar.get_output(0), 0)
    # expert_1_out: [1, hidden_size]
    scaled_1 = network.add_elementwise(
        expert_1_out.get_output(0), weight_1, trt.ElementWiseOperation.PROD
    )

    # Gather expert 2 output and scale
    idx_2_scalar = network.add_shuffle(idx_2)
    idx_2_scalar.reshape_dims = (1,)
    expert_2_out = network.add_gather(stacked_out, idx_2_scalar.get_output(0), 0)
    scaled_2 = network.add_elementwise(
        expert_2_out.get_output(0), weight_2, trt.ElementWiseOperation.PROD
    )

    # Sum: weighted expert 1 + weighted expert 2
    moe_out = network.add_elementwise(
        scaled_1.get_output(0), scaled_2.get_output(0), trt.ElementWiseOperation.SUM
    )

    return moe_out.get_output(0)  # [1, hidden_size]


def _add_moe_decoder_layer(
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
    top_k: int,
    jitter_eps: float = 0.01,
    norm_type: str = "layernorm",
    dtype: np.dtype = np.float32,
    work_trt_dtype=None,
) -> dict[str, trt.ITensor]:
    """Add one decoder layer with MoE MLP. Attention is standard."""

    # Attention block (pre-norm -> QKV -> RoPE -> cache -> attn -> out proj)
    attn = graph_blocks.add_attention_block(
        network,
        hidden,
        cache_k,
        cache_v,
        attention_mask,
        position_id,
        weights=weights,
        prefix=prefix,
        hidden_size=hidden_size,
        attention_size=attention_size,
        kv_attention_size=kv_attention_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_cache_length=max_cache_length,
        eps_tensor=eps_tensor,
        norm_type=norm_type,
        position_type="rope",
        cos_half_tensor=cos_half_tensor,
        sin_half_tensor=sin_half_tensor,
        rotary_embedding_dim=head_dim,
        dtype=dtype,
    )
    attn_out = attn["attn_out"]

    # Residual connection
    residual1 = network.add_elementwise(hidden, attn_out, trt.ElementWiseOperation.SUM)

    # Post-attention norm
    norm2 = _apply_norm(
        network,
        residual1.get_output(0),
        hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        weights.get(f"{prefix}.post_attn_norm_beta"),
        eps_tensor,
        norm_type,
        dtype=dtype,
    )

    # MoE block (replaces standard MLP)
    moe_out = _add_moe_block(
        network,
        norm2,
        weights,
        prefix,
        hidden_size,
        num_experts,
        moe_intermediate,
        top_k,
        jitter_eps=jitter_eps,
        dtype=dtype,
        work_trt_dtype=work_trt_dtype,
    )

    # Residual connection
    residual2 = network.add_elementwise(
        residual1.get_output(0), moe_out, trt.ElementWiseOperation.SUM
    )

    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": attn["present_k"],
        "present_v": attn["present_v"],
    }


_BUNDLE_FILES = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


def _chat_template(model_dir: Path) -> bytes:
    value = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8")).get(
        "chat_template"
    )
    if not isinstance(value, str) or not value:
        raise ValueError("Phi-MoE tokenizer_config.json requires chat_template")
    return value.encode("utf-8")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _runtime_config(model_dir: Path, config: ModelConfig, model: _PhiMoEModel, **updates) -> dict:
    runtime = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }
    generation_path = model_dir / "generation_config.json"
    if generation_path.is_file():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("generation_config.json must contain one JSON object")
        if "eos_token_id" in generation:
            runtime["eos_token_id"] = generation["eos_token_id"]
    runtime.update(updates)
    return runtime


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Phi-MoE bundle through family-owned code."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("phi_moe does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("phi_moe does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("phi_moe does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("phi_moe does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("phi_moe does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("phi_moe supports only task=text_generation")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "phimoe":
        raise ValueError(f"Phi-MoE does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Phi-MoE precision must be fp32, fp16, or bf16")
    model = _PhiMoEModel()
    default_length = min(config.max_position_embeddings, 256)
    max_sequence_length = _positive_int(
        request.max_sequence_length or default_length, "max_sequence_length"
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Phi-MoE max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Phi-MoE quantization requires a family-owned qualified path")
    if request.fp32_layers:
        raise NotImplementedError("Phi-MoE does not expose mixed-precision layer selection")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    config.raw["_quantized_build_requested"] = False
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="phi_moe", task=request.task, backend=request.backend)
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            plan = model.build_engine(
                config,
                weights,
                max_sequence_length,
                precision=precision,
                quant_ctx=None,
                verbose=bool(request.verbose),
                debug_layer_outputs=False,
                parallel_config=parallel.for_rank(rank),
            )
            writer.add_bytes(f"engine.rank{rank}.plan", plan)
        layout = "dual_profile"
    else:
        plan = model.build_engine(
            config,
            weights,
            max_sequence_length,
            precision=precision,
            quant_ctx=None,
            verbose=bool(request.verbose),
            debug_layer_outputs=False,
            parallel_config=parallel,
        )
        writer.add_bytes("engine.plan", plan)
        layout = "single"
    writer.add_json(
        "runtime.json",
        _runtime_config(
            model_dir,
            config,
            model,
            precision=precision,
            max_cache_length=max_sequence_length,
            decoder_engine_layout=layout,
            tensor_parallel_size=parallel.tp_size,
            tensor_parallel_mode="tensor_parallel" if parallel.enabled else "single",
        ),
    )
    writer.add_bytes("chat_template.jinja", _chat_template(model_dir))
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
