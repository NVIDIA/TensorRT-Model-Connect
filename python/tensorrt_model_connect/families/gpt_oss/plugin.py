# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS family plugin — OpenAI GPT-OSS MoE with packed expert weights.

GPT-OSS-20B is a 21B-param sparse MoE (32 experts, 4 active per token).

Key characteristics:
  - RMSNorm (no norm biases)
  - Biases on all attention projections (Q/K/V/O)
  - GQA: 64 Q heads, 8 KV heads, head_dim=64
  - Router: topk on raw logits, then softmax over selected values only
  - Packed expert weights [num_experts, in_dim, out_dim] with biases
  - Custom gated activation (NOT standard SwiGLU):
      gate, up are INTERLEAVED in gate_up_proj (even/odd columns)
      glu = clamp(gate, max=7) * sigmoid(clamp(gate, max=7) * 1.702)
      output = (clamp(up, -7, 7) + 1) * glu

MXFP4 weights are auto-dequantized by HF AutoModelForCausalLM.

Weight key mapping (per layer, after dequant):
  HF: model.layers.{i}.self_attn.{q,k,v,o}_proj.weight/bias
  HF: model.layers.{i}.mlp.router.weight/bias              -> router + bias
  HF: model.layers.{i}.mlp.experts.gate_up_proj             -> packed [E, H, 2I]
  HF: model.layers.{i}.mlp.experts.gate_up_proj_bias        -> packed [E, 2I]
  HF: model.layers.{i}.mlp.experts.down_proj                -> packed [E, I, H]
  HF: model.layers.{i}.mlp.experts.down_proj_bias           -> packed [E, H]
"""

from __future__ import annotations

import gc
import sys

import numpy as np
from tensorrt_model_connect import trt_compat

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _transpose_2d,
)
from . import graph_ops
from . import graph_blocks
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .standard_decoder_builder import _apply_norm, _mark_debug_output
from .utils import make_rope_half_tables


trt = trt_compat.get_trt()

class GptOssPlugin:
    name = "gpt_oss"
    runtime_strategy = "gpt_oss_decoder_moe"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "gpt_oss"

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load GPT-OSS weights via AutoModelForCausalLM (handles MXFP4 dequant).

        The model uses packed expert tensors [num_experts, ...] which must be
        unpacked into per-expert weight/bias arrays for the TRT engine builder.
        """
        import torch
        from transformers import AutoModelForCausalLM

        print("[trtmc build] Loading GPT-OSS model (MXFP4 auto-dequant) ...",
              file=sys.stderr, flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        state = {
            k: v.float().cpu().numpy()
            for k, v in model.state_dict().items()
        }
        del model
        gc.collect()

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_experts = config.raw.get("num_local_experts", 32)

        weights = WeightDict()

        # Embedding
        embedding = state["model.embed_tokens.weight"]
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        attention_size = 0
        moe_intermediate = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf = f"model.layers.{layer_idx}"

            # RMSNorm (no biases)
            weights[f"{prefix}.input_norm"] = state[
                f"{hf}.input_layernorm.weight"].astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = state[
                f"{hf}.post_attention_layernorm.weight"].astype(np.float32)

            # --- Attention projections (with biases) ---
            q_raw = state[f"{hf}.self_attn.q_proj.weight"]
            k_raw = state[f"{hf}.self_attn.k_proj.weight"]
            v_raw = state[f"{hf}.self_attn.v_proj.weight"]
            o_raw = state[f"{hf}.self_attn.o_proj.weight"]

            if attention_size == 0:
                attention_size = q_raw.shape[0]

            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")
            o_t = _transpose_2d(o_raw, "o_proj")

            # Keep compact GQA/MQA K/V

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t

            # Attention biases
            weights[f"{prefix}.q_bias"] = state[
                f"{hf}.self_attn.q_proj.bias"].astype(np.float32)
            weights[f"{prefix}.o_bias"] = state[
                f"{hf}.self_attn.o_proj.bias"].astype(np.float32)

            weights[f"{prefix}.k_bias"] = state[
                f"{hf}.self_attn.k_proj.bias"].astype(np.float32)
            weights[f"{prefix}.v_bias"] = state[
                f"{hf}.self_attn.v_proj.bias"].astype(np.float32)

            # Attention sinks (per-head learned parameter for softmax normalization)
            sinks_key = f"{hf}.self_attn.sinks"
            if sinks_key in state:
                weights[f"{prefix}.sinks"] = state[sinks_key].astype(
                    np.float32)

            # --- Router ---
            router_w = state[f"{hf}.mlp.router.weight"]  # [num_experts, hidden]
            weights[f"{prefix}.router"] = _transpose_2d(
                router_w, "router")
            weights[f"{prefix}.router_bias"] = state[
                f"{hf}.mlp.router.bias"].astype(np.float32)

            # --- Packed expert weights ---
            gate_up = state[f"{hf}.mlp.experts.gate_up_proj"]
            gate_up_bias = state[f"{hf}.mlp.experts.gate_up_proj_bias"]
            down = state[f"{hf}.mlp.experts.down_proj"]
            down_bias = state[f"{hf}.mlp.experts.down_proj_bias"]

            # gate_up_proj is [E, hidden, 2*inter] with INTERLEAVED
            # gate/up columns: gate=even indices, up=odd indices.
            # De-interleave into separate gate [hidden, inter] and
            # up [hidden, inter] per expert.
            per_expert_inter = gate_up.shape[-1] // 2
            for e_idx in range(num_experts):
                gu = gate_up[e_idx]  # [hidden, 2*inter]
                # Interleaved: gate = columns 0,2,4,...  up = columns 1,3,5,...
                weights[f"{prefix}.expert.{e_idx}.w_gate"] = (
                    np.ascontiguousarray(
                        gu[:, ::2], dtype=np.float32))
                weights[f"{prefix}.expert.{e_idx}.w_up"] = (
                    np.ascontiguousarray(
                        gu[:, 1::2], dtype=np.float32))
                weights[f"{prefix}.expert.{e_idx}.w_down"] = (
                    np.ascontiguousarray(
                        down[e_idx], dtype=np.float32))

            if moe_intermediate == 0:
                moe_intermediate = per_expert_inter

            # Per-expert biases (also interleaved for gate_up)
            for e_idx in range(num_experts):
                gu_b = gate_up_bias[e_idx]  # [2*inter]
                weights[f"{prefix}.expert.{e_idx}.gate_bias"] = (
                    gu_b[::2].astype(np.float32))
                weights[f"{prefix}.expert.{e_idx}.up_bias"] = (
                    gu_b[1::2].astype(np.float32))
                weights[f"{prefix}.expert.{e_idx}.down_bias"] = (
                    down_bias[e_idx].astype(np.float32))

        # Final norm
        final_key = "model.norm.weight"
        if final_key in state:
            weights["final_norm"] = state[final_key].astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_key = "lm_head.weight"
        if lm_key in state:
            weights["w_out"] = _transpose_2d(state[lm_key], "lm_head")
        else:
            weights["w_out"] = _transpose_2d(
                embedding.copy(), "embedding_tied")

        lm_bias_key = "lm_head.bias"
        if lm_bias_key in state:
            weights["lm_head_bias"] = state[lm_bias_key].astype(np.float32)

        # Metadata
        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_num_experts"] = num_experts  # type: ignore[assignment]
        weights["_moe_intermediate_size"] = moe_intermediate  # type: ignore[assignment]
        weights["_num_experts_per_tok"] = config.raw.get(
            "num_experts_per_tok", 4)  # type: ignore[assignment]

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="GPT-OSS tensor-parallel builds")
            from .tp_builder import build_gpt_oss_tp_engine
            return build_gpt_oss_tp_engine(
                config, weights, max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                verbose=verbose,
                debug_layer_outputs=debug_layer_outputs,
                parallel_config=parallel,
            )

        attention_size: int = weights.get(
            "_attention_size", config.attention_size)
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
            weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
        attention_window = max_cache_length + 1

        logger = trt.Logger(
            trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 1 << 30)

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
            int(layer) for layer in config.raw.get("_fp32_layers", ()))
        invalid_fp32_layers = sorted(
            layer for layer in requested_fp32_layers
            if layer < 0 or layer >= num_layers)
        if invalid_fp32_layers:
            raise ValueError(
                "fp32_layers contains out-of-range indices: "
                f"{invalid_fp32_layers}")

        # -----------------------------------------------------------
        # Inputs
        # -----------------------------------------------------------
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

        # GPT-OSS alternates sliding_attention (windowed) and full_attention
        # layers. The runtime mask only encodes causal validity, so build a
        # second mask that additionally hides cache columns older than the
        # sliding window and feed it to the sliding layers.
        layer_types = list(config.raw.get("layer_types") or [])
        sliding_window = int(config.raw.get("sliding_window") or 0)
        sliding_attention_mask = None
        if sliding_window > 0 and "sliding_attention" in layer_types:
            sliding_attention_mask = graph_ops.add_sliding_window_mask(
                network, attention_mask, position_id,
                attention_window, sliding_window)

        # -----------------------------------------------------------
        # Shared constants
        # -----------------------------------------------------------
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"],
            dtype=work_np_dtype)

        # GPT-OSS uses YaRN RoPE scaling. Hub checkpoints serialize it under
        # rope_scaling; transformers 5.x configs use rope_parameters. The
        # shared helper accepts both and applies HF-exact YaRN semantics.
        cos_half_np, sin_half_np = make_rope_half_tables(
            config, attention_window, head_dim)

        cos_half_tensor = graph_ops.add_constant(
            network, cos_half_np.shape, cos_half_np, dtype=work_np_dtype)
        sin_half_tensor = graph_ops.add_constant(
            network, sin_half_np.shape, sin_half_np, dtype=work_np_dtype)

        eps_tensor = graph_ops.add_constant(
            network, (1, 1),
            np.array([config.rms_norm_eps], dtype=work_np_dtype),
            dtype=work_np_dtype)
        attn_scale = 1.0 / np.sqrt(max(head_dim, 1))
        attn_scale_tensor = graph_ops.add_constant(
            network, (1, 1, 1),
            np.array([attn_scale], dtype=work_np_dtype),
            dtype=work_np_dtype)

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
            layer_is_fp32 = (
                precision == "fp16" and layer_idx in requested_fp32_layers)
            layer_np_dtype = np.float32 if layer_is_fp32 else work_np_dtype
            layer_trt_dtype = trt.float32 if layer_is_fp32 else work_trt_dtype

            def layer_cast(tensor):
                if tensor.dtype == layer_trt_dtype:
                    return tensor
                return network.add_cast(tensor, layer_trt_dtype).get_output(0)

            layer_type = (
                layer_types[layer_idx] if layer_idx < len(layer_types)
                else "full_attention")
            layer_mask = (
                sliding_attention_mask
                if (sliding_attention_mask is not None
                    and layer_type == "sliding_attention")
                else attention_mask)

            result = _add_gpt_oss_decoder_layer(
                network=network,
                hidden=layer_cast(hidden_state),
                cache_k=layer_cast(cache_k_inputs[layer_idx]),
                cache_v=layer_cast(cache_v_inputs[layer_idx]),
                attention_mask=layer_cast(layer_mask),
                position_id=position_id,
                cos_half_tensor=layer_cast(cos_half_tensor),
                sin_half_tensor=layer_cast(sin_half_tensor),
                attn_scale_tensor=layer_cast(attn_scale_tensor),
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
                hidden_state = network.add_cast(
                    hidden_state, work_trt_dtype).get_output(0)
                present_k = network.add_cast(
                    present_k, work_trt_dtype).get_output(0)
                present_v = network.add_cast(
                    present_v, work_trt_dtype).get_output(0)
            present_k_outputs.append(present_k)
            present_v_outputs.append(present_v)

            if debug_layer_outputs:
                _mark_debug_output(
                    network, result["post_attn"],
                    f"debug_post_attn_{layer_idx}")
                _mark_debug_output(
                    network, hidden_state,
                    f"debug_hidden_{layer_idx}")

        # -----------------------------------------------------------
        # Final norm
        # -----------------------------------------------------------
        final_norm = weights.get("final_norm")
        if final_norm is not None and len(final_norm) > 0:
            hidden_state = _apply_norm(
                network, hidden_state, hidden, final_norm,
                None, eps_tensor, "rmsnorm", dtype=work_np_dtype)

        # -----------------------------------------------------------
        # LM head (logits)
        # -----------------------------------------------------------
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, vocab, weights["w_out"],
            dtype=work_np_dtype)

        lm_bias = weights.get("lm_head_bias")
        if lm_bias is not None:
            logits = graph_ops.add_bias_sum(
                network, logits, vocab, lm_bias, dtype=work_np_dtype)
        else:
            b_out = np.zeros(vocab, dtype=work_np_dtype)
            logits = graph_ops.add_bias_sum(
                network, logits, vocab, b_out, dtype=work_np_dtype)

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
            print(f"[trtmc build] Building GPT-OSS MoE TRT engine "
                  f"({num_layers} layers, hidden={hidden}, "
                  f"attn={attention_size}, experts={num_experts}, "
                  f"top_k={top_k}, inter={moe_intermediate}, "
                  f"cache={max_cache_length}) ...", file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        return bytes(plan)


def _add_gpt_oss_expert(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    intermediate_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    gate_bias: np.ndarray | None = None,
    up_bias: np.ndarray | None = None,
    down_bias: np.ndarray | None = None,
    alpha: float = 1.702,
    limit: float = 7.0,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GPT-OSS gated expert activation (NOT standard SwiGLU).

    HF implementation:
      gate, up = gate_up[..., ::2], gate_up[..., 1::2]  (de-interleaved at load time)
      gate = clamp(gate, max=limit)
      up = clamp(up, min=-limit, max=limit)
      glu = gate * sigmoid(gate * alpha)
      output = (up + 1) * glu
      result = output @ down_proj + down_bias
    """
    gate = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_gate, dtype=dtype)
    if gate_bias is not None:
        gate = graph_ops.add_bias_sum(
            network, gate, intermediate_size, gate_bias, dtype=dtype)

    up = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, intermediate_size, w_up, dtype=dtype)
    if up_bias is not None:
        up = graph_ops.add_bias_sum(
            network, up, intermediate_size, up_bias, dtype=dtype)

    # Clamp gate to max=limit
    limit_const = graph_ops.add_constant(
        network, (1, 1), np.array([limit], dtype=dtype), dtype=dtype)
    gate = network.add_elementwise(
        gate, limit_const, trt.ElementWiseOperation.MIN).get_output(0)

    # Clamp up to [-limit, limit]
    neg_limit_const = graph_ops.add_constant(
        network, (1, 1), np.array([-limit], dtype=dtype), dtype=dtype)
    up = network.add_elementwise(
        up, limit_const, trt.ElementWiseOperation.MIN).get_output(0)
    up = network.add_elementwise(
        up, neg_limit_const, trt.ElementWiseOperation.MAX).get_output(0)

    # glu = gate * sigmoid(gate * alpha)
    alpha_const = graph_ops.add_constant(
        network, (1, 1), np.array([alpha], dtype=dtype), dtype=dtype)
    gate_scaled = network.add_elementwise(
        gate, alpha_const, trt.ElementWiseOperation.PROD).get_output(0)
    sigmoid = network.add_activation(
        gate_scaled, trt.ActivationType.SIGMOID).get_output(0)
    glu = network.add_elementwise(
        gate, sigmoid, trt.ElementWiseOperation.PROD).get_output(0)

    # output = (up + 1) * glu
    one_const = graph_ops.add_constant(
        network, (1, 1), np.array([1.0], dtype=dtype), dtype=dtype)
    up_plus_one = network.add_elementwise(
        up, one_const, trt.ElementWiseOperation.SUM).get_output(0)
    gated = network.add_elementwise(
        up_plus_one, glu, trt.ElementWiseOperation.PROD).get_output(0)

    # down projection + bias
    down_out = graph_ops.add_matmul_rhs_constant(
        network, gated, intermediate_size, hidden_size, w_down, dtype=dtype)
    if down_bias is not None:
        down_out = graph_ops.add_bias_sum(
            network, down_out, hidden_size, down_bias, dtype=dtype)
    return down_out


def _add_gpt_oss_moe_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_experts: int,
    moe_intermediate: int,
    top_k: int = 4,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """GPT-OSS MoE block: topk on raw logits, softmax over selected only.

    Router logic (matches HF GptOssTopKRouter):
      1. Router logits = inp @ router_w + bias  (raw, no softmax yet)
      2. TopK on RAW logits -> top_k values + indices
      3. Softmax ONLY over the selected top_k values (sums to 1)

    Then: compute all expert outputs, gather selected, scale, sum.
    """
    # 1. Router logits + bias (raw)
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, num_experts,
        weights[f"{prefix}.router"], dtype=dtype)
    router_bias = weights.get(f"{prefix}.router_bias")
    if router_bias is not None:
        router_logits = graph_ops.add_bias_sum(
            network, router_logits, num_experts, router_bias, dtype=dtype)

    # 2. TopK on RAW logits (not softmax)
    topk = network.add_topk(
        router_logits, trt.TopKOperation.MAX, top_k, 1 << 1)
    top_values = topk.get_output(0)   # [1, top_k]
    top_indices = topk.get_output(1)  # [1, top_k]

    # 3. Softmax ONLY over the selected top-k values
    sm = network.add_softmax(top_values)
    sm.axes = 1 << 1
    routing_weights = sm.get_output(0)  # [1, top_k], sums to 1

    # 4. Compute ALL expert outputs and stack
    expert_outputs = []
    for e in range(num_experts):
        exp_out = _add_gpt_oss_expert(
            network, inp, hidden_size, moe_intermediate,
            weights[f"{prefix}.expert.{e}.w_gate"],
            weights[f"{prefix}.expert.{e}.w_up"],
            weights[f"{prefix}.expert.{e}.w_down"],
            weights.get(f"{prefix}.expert.{e}.gate_bias"),
            weights.get(f"{prefix}.expert.{e}.up_bias"),
            weights.get(f"{prefix}.expert.{e}.down_bias"),
            dtype=dtype,
        )
        expert_outputs.append(exp_out)

    stacked = network.add_concatenation(expert_outputs)
    stacked.axis = 0
    stacked_out = stacked.get_output(0)  # [num_experts, hidden_size]

    # 5. Gather selected experts, scale, and sum
    result = None
    for k_idx in range(top_k):
        idx_slice = network.add_slice(
            top_indices, start=(0, k_idx), shape=(1, 1), stride=(1, 1))
        idx_flat = network.add_shuffle(idx_slice.get_output(0))
        idx_flat.reshape_dims = (1,)

        w_slice = network.add_slice(
            routing_weights,
            start=(0, k_idx), shape=(1, 1), stride=(1, 1))

        expert_out = network.add_gather(
            stacked_out, idx_flat.get_output(0), 0)

        scaled = network.add_elementwise(
            expert_out.get_output(0), w_slice.get_output(0),
            trt.ElementWiseOperation.PROD)

        if result is None:
            result = scaled.get_output(0)
        else:
            sum_layer = network.add_elementwise(
                result, scaled.get_output(0),
                trt.ElementWiseOperation.SUM)
            result = sum_layer.get_output(0)

    return result


def _add_gpt_oss_attention(
    network: trt.INetworkDefinition,
    normed: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_length: int,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    attn_scale_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """GPT-OSS attention with attention sinks.

    Like standard attention but with learned per-head sink logits concatenated
    to attention scores before softmax, then dropped after. This causes the
    softmax to "leak" probability mass to the sink, normalizing the real
    attention weights differently.

    HF logic:
      combined = cat([attn_weights, sinks.expand(...)], dim=-1)
      combined = combined - combined.max(dim=-1, keepdim=True)
      probs = softmax(combined, dim=-1)
      scores = probs[..., :-1]  # drop the sink column
      output = scores @ V
    """
    attention_window = max_cache_length + 1

    # QKV projections with biases
    q = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attention_size,
        weights[f"{prefix}.w_q"], dtype=dtype)
    k = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_k"], dtype=dtype)
    v = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size,
        weights[f"{prefix}.w_v"], dtype=dtype)

    q_bias = weights.get(f"{prefix}.q_bias")
    if q_bias is not None:
        q = graph_ops.add_bias_sum(
            network, q, attention_size, q_bias, dtype=dtype)
    k_bias = weights.get(f"{prefix}.k_bias")
    if k_bias is not None:
        k = graph_ops.add_bias_sum(
            network, k, kv_attention_size, k_bias, dtype=dtype)
    v_bias = weights.get(f"{prefix}.v_bias")
    if v_bias is not None:
        v = graph_ops.add_bias_sum(
            network, v, kv_attention_size, v_bias, dtype=dtype)

    # Native RoPE
    q = graph_ops.add_apply_rope_native(
        network, q, num_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, head_dim)
    k = graph_ops.add_apply_rope_native(
        network, k, num_kv_heads, head_dim, cos_half_tensor, sin_half_tensor,
        position_id, head_dim)

    # Save present K/V
    present_k = k
    present_v = v

    # Reshape and concat with cache
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)

    all_k = network.add_concatenation(
        [cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation(
        [cache_v, v_reshape.get_output(0)])
    all_v.axis = 0

    # Native attention is valid when no learned attention sink logits are
    # present. Sink logits append an extra softmax column and must stay manual.
    sinks = weights.get(f"{prefix}.sinks")
    if sinks is None:
        mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)
        context_flat = graph_ops.add_attention_from_rows(
            network, q, all_k.get_output(0), all_v.get_output(0),
            num_heads=num_heads, head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            q_seq=1, kv_seq=attention_window,
            mask=mask_4d)
    else:
        # Multi-head reshape
        q_heads = network.add_shuffle(q)
        q_heads.reshape_dims = (num_heads, 1, head_dim)

        k_heads = network.add_shuffle(all_k.get_output(0))
        k_heads.reshape_dims = (attention_window, num_kv_heads, head_dim)
        v_heads = network.add_shuffle(all_v.get_output(0))
        v_heads.reshape_dims = (attention_window, num_kv_heads, head_dim)

        k_heads.second_transpose = trt.Permutation([1, 0, 2])
        v_heads.second_transpose = trt.Permutation([1, 0, 2])
        k_heads_t = k_heads.get_output(0)
        v_heads_t = v_heads.get_output(0)
        if num_kv_heads != num_heads:
            group_size = num_heads // num_kv_heads
            k_slices = []
            v_slices = []
            for kvh in range(num_kv_heads):
                ks = network.add_slice(
                    k_heads_t, start=(kvh, 0, 0),
                    shape=(1, attention_window, head_dim),
                    stride=(1, 1, 1))
                vs = network.add_slice(
                    v_heads_t, start=(kvh, 0, 0),
                    shape=(1, attention_window, head_dim),
                    stride=(1, 1, 1))
                k_slices.extend([ks.get_output(0)] * group_size)
                v_slices.extend([vs.get_output(0)] * group_size)
            k_expand = network.add_concatenation(k_slices)
            k_expand.axis = 0
            v_expand = network.add_concatenation(v_slices)
            v_expand.axis = 0
            k_heads_t = k_expand.get_output(0)
            v_heads_t = v_expand.get_output(0)

        # Attention scores: Q @ K^T * scale
        score = network.add_matrix_multiply(
            q_heads.get_output(0), trt.MatrixOperation.NONE,
            k_heads_t, trt.MatrixOperation.TRANSPOSE)
        scaled = network.add_elementwise(
            score.get_output(0), attn_scale_tensor,
            trt.ElementWiseOperation.PROD)

        # Apply causal mask: [num_heads, 1, attention_window]
        mask3d = network.add_shuffle(attention_mask)
        mask3d.reshape_dims = (1, 1, attention_window)
        masked = network.add_elementwise(
            scaled.get_output(0), mask3d.get_output(0),
            trt.ElementWiseOperation.SUM)
        # masked: [num_heads, 1, attention_window]

        # --- Attention sinks ---
        # sinks: [num_heads] -> reshape to [num_heads, 1, 1]
        sinks_const = graph_ops.add_constant(
            network, (num_heads, 1, 1), sinks.reshape(num_heads, 1, 1),
            dtype=dtype)
        # Concatenate sink column to attention logits: [H, 1, W] + [H, 1, 1] -> [H, 1, W+1]
        combined = network.add_concatenation(
            [masked.get_output(0), sinks_const])
        combined.axis = 2
        combined_out = combined.get_output(0)  # [num_heads, 1, attention_window + 1]

        # Subtract max for numerical stability (matches HF)
        max_val = network.add_reduce(
            combined_out, trt.ReduceOperation.MAX, 1 << 2, keep_dims=True)
        stable = network.add_elementwise(
            combined_out, max_val.get_output(0),
            trt.ElementWiseOperation.SUB)

        # Softmax over the extended dimension
        softmax = network.add_softmax(stable.get_output(0))
        softmax.axes = 1 << 2

        # Drop the sink column: slice [H, 1, :attention_window] from [H, 1, W+1]
        scores = network.add_slice(
            softmax.get_output(0),
            start=(0, 0, 0),
            shape=(num_heads, 1, attention_window),
            stride=(1, 1, 1))
        attn_probs = scores.get_output(0)

        # Context: attn_probs @ V
        context_heads = network.add_matrix_multiply(
            attn_probs, trt.MatrixOperation.NONE,
            v_heads_t, trt.MatrixOperation.NONE)

        context_flat = network.add_shuffle(context_heads.get_output(0))
        context_flat.reshape_dims = (1, attention_size)

    # Output projection with bias
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context_flat.get_output(0),
        attention_size, hidden_size,
        weights[f"{prefix}.w_o"], dtype=dtype)
    o_bias = weights.get(f"{prefix}.o_bias")
    if o_bias is not None:
        attn_out = graph_ops.add_bias_sum(
            network, attn_out, hidden_size, o_bias, dtype=dtype)

    return {
        "attn_out": attn_out,
        "present_k": present_k,
        "present_v": present_v,
    }


def _add_gpt_oss_decoder_layer(
    *,
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    attn_scale_tensor: trt.ITensor,
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
    top_k: int = 4,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """One GPT-OSS decoder layer: attention with sinks + MoE."""

    # Pre-attention RMSNorm
    normed = _apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"],
        None, eps_tensor, "rmsnorm", dtype=dtype)

    # Attention with sinks
    attn = _add_gpt_oss_attention(
        network, normed, cache_k, cache_v, attention_mask, position_id,
        weights=weights, prefix=prefix,
        hidden_size=hidden_size, attention_size=attention_size,
        kv_attention_size=kv_attention_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        max_cache_length=max_cache_length,
        cos_half_tensor=cos_half_tensor, sin_half_tensor=sin_half_tensor,
        attn_scale_tensor=attn_scale_tensor,
        dtype=dtype,
    )

    # Residual connection
    residual1 = network.add_elementwise(
        hidden, attn["attn_out"], trt.ElementWiseOperation.SUM)

    # Post-attention RMSNorm
    norm2 = _apply_norm(
        network, residual1.get_output(0), hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        None, eps_tensor, "rmsnorm", dtype=dtype)

    # MoE block
    moe_out = _add_gpt_oss_moe_block(
        network, norm2, weights, prefix,
        hidden_size, num_experts, moe_intermediate, top_k, dtype=dtype)

    # Residual connection
    residual2 = network.add_elementwise(
        residual1.get_output(0), moe_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": attn["present_k"],
        "present_v": attn["present_v"],
    }


plugin = GptOssPlugin()
