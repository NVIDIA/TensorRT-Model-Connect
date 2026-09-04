# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixtral family plugin — Mixture of Experts with standard top-k softmax routing.

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


class _MixtralModel:
    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
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
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            parallel.validate()
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

        if precision == "fp16":
            work_np_dtype = np.float16
            work_trt_dtype = trt.float16
        elif precision == "bf16":
            work_np_dtype = np.float16
            work_trt_dtype = trt.bfloat16
        else:
            work_np_dtype = np.float32
            work_trt_dtype = trt.float32

        requested_fp32_layers = frozenset(
            int(layer) for layer in config.raw.get("_fp32_layers", ())
        )
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
            network,
            (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype,
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
    # FP16/BF16.  Keeping softmax/top-k in the model dtype can change the
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
            sum_layer = network.add_elementwise(result, scaled_value, trt.ElementWiseOperation.SUM)
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


_BUNDLE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.model",
)


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


def _runtime_config(model_dir: Path, config: ModelConfig, model: _MixtralModel, **updates) -> dict:
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
    """Build one Mixtral bundle through family-owned code."""
    if request.image_height is not None:
        raise NotImplementedError("mixtral does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("mixtral does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("mixtral does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("mixtral does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_generation":
        raise ValueError("mixtral supports only task=text_generation")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() != "mixtral":
        raise ValueError(f"Mixtral does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Mixtral precision must be fp32, fp16, or bf16")
    model = _MixtralModel()
    default_length = min(config.max_position_embeddings, 256)
    max_sequence_length = _positive_int(
        request.max_sequence_length or default_length, "max_sequence_length"
    )
    if max_sequence_length > config.max_position_embeddings:
        raise ValueError("Mixtral max_sequence_length exceeds checkpoint context capacity")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("Mixtral quantization requires a family-owned qualified path")
    if request.fp32_layers:
        raise NotImplementedError("Mixtral does not expose mixed-precision layer selection")
    parallel = ParallelConfig(
        tp_size=_positive_int(request.tensor_parallel_size, "tensor_parallel_size")
    )
    parallel.validate()
    config.raw["_model_dir"] = str(model_dir)
    config.raw["_resolved_build_precision"] = precision
    config.raw["_parallel_build_enabled"] = parallel.enabled
    config.raw["_quantized_build_requested"] = False
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="mixtral", task=request.task, backend="trt")
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
    for filename in _BUNDLE_FILES:
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
