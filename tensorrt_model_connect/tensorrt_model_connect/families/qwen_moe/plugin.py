"""Qwen3 MoE family plugin — Mixture of Experts with optional shared experts.

Supports two Qwen MoE variants:

  **Qwen2.5-MoE** (model_type=qwen2_moe): shared expert on every MoE layer,
  gated by a learned sigmoid gate (shared_expert_gate).

  **Qwen3-MoE** (model_type=qwen3_moe): pure routed MoE with no shared expert,
  per-head QK RMSNorm on Q and K projections before RoPE.

Common details:
  - Standard top-k softmax routing with renormalization (norm_topk_prob=True)
  - Some layers use dense MLP instead of MoE (controlled by mlp_only_layers)
  - No biases on attention or norm projections
  - Separate Q/K/V/O projections with GQA

Weight key mapping:
  HF: model.layers.{i}.mlp.gate.weight                       -> router [num_experts, hidden]
  HF: model.layers.{i}.mlp.experts.{e}.gate_proj.weight      -> expert gate [moe_inter, hidden]
  HF: model.layers.{i}.mlp.experts.{e}.up_proj.weight        -> expert up   [moe_inter, hidden]
  HF: model.layers.{i}.mlp.experts.{e}.down_proj.weight      -> expert down [hidden, moe_inter]
  HF: model.layers.{i}.mlp.shared_expert.gate_proj.weight    -> shared expert gate (Qwen2.5 only)
  HF: model.layers.{i}.mlp.shared_expert.up_proj.weight      -> shared expert up   (Qwen2.5 only)
  HF: model.layers.{i}.mlp.shared_expert.down_proj.weight    -> shared expert down  (Qwen2.5 only)
  HF: model.layers.{i}.mlp.shared_expert_gate.weight         -> shared expert gate sigmoid [1, hidden] (Qwen2.5 only)
  HF: model.layers.{i}.self_attn.q_norm.weight               -> per-head Q RMSNorm (Qwen3 only)
  HF: model.layers.{i}.self_attn.k_norm.weight               -> per-head K RMSNorm (Qwen3 only)
  HF: model.layers.{i}.mlp.gate_proj/up_proj/down_proj       -> dense MLP (mlp_only_layers)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorrt_model_connect import trt_compat

from ...config import ModelConfig
from ...checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _target_np_dtype,
    _transpose_2d,
)
from ... import graph_ops
from ... import graph_blocks
from .standard_decoder_builder import _apply_norm, _mark_debug_output


trt = trt_compat.get_trt()

class Qwen3MoePlugin:
    name = "qwen_moe"
    runtime_strategy = "decoder_moe"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in ("qwen3_moe", "qwen2_moe")

    def load_weights(
        self, model_dir: str, config: ModelConfig, *,
        precision: str = "fp32",
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)
        weight_dtype = _target_np_dtype(precision)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        raw = config.raw
        num_experts = raw.get("num_experts", 128)
        num_experts_per_tok = raw.get("num_experts_per_tok", 8)
        moe_intermediate_size = raw.get("moe_intermediate_size", 2560)
        shared_expert_intermediate_size = raw.get(
            "shared_expert_intermediate_size", 0)
        dense_intermediate_size = config.intermediate_size
        mlp_only_layers = set(raw.get("mlp_only_layers", []))

        # Detect whether the model has shared experts by probing layer 0
        # (Qwen3-MoE does not, Qwen2.5-MoE does)
        has_shared_expert = _has_tensor(
            readers, "model.layers.0.mlp.shared_expert.gate_proj.weight")
        if has_shared_expert and shared_expert_intermediate_size == 0:
            # Infer from actual weight shape
            shared_expert_intermediate_size = _load_tensor(
                readers,
                "model.layers.0.mlp.shared_expert.gate_proj.weight"
            ).shape[0]

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(weight_dtype)

        attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # RMSNorm weights (no biases)
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)

            post_norm = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # Q/K/V/O projections (separate, no biases)
            q_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(
                readers, f"{hf_prefix}.self_attn.o_proj.weight")

            if attention_size == 0:
                attention_size = q_raw.shape[0]

            q_t = _transpose_2d(q_raw, "q_proj", precision=precision)
            k_t = _transpose_2d(k_raw, "k_proj", precision=precision)
            v_t = _transpose_2d(v_raw, "v_proj", precision=precision)
            o_t = _transpose_2d(o_raw, "o_proj", precision=precision)
            del q_raw, k_raw, v_raw, o_raw

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t

            # Per-head Q/K RMSNorm (Qwen3 MoE)
            # HF stores [head_dim] weights shared across all heads;
            # graph_ops.add_rms_norm_per_head expects [num_heads * head_dim].
            # K is keep compacted to num_heads before this point, so both
            # Q and K norm tile to num_heads.
            q_norm_key = f"{hf_prefix}.self_attn.q_norm.weight"
            if _has_tensor(readers, q_norm_key):
                qn = _load_tensor(readers, q_norm_key).astype(np.float32)
                weights[f"{prefix}.q_norm"] = np.tile(qn, num_heads)
            k_norm_key = f"{hf_prefix}.self_attn.k_norm.weight"
            if _has_tensor(readers, k_norm_key):
                kn = _load_tensor(readers, k_norm_key).astype(np.float32)
                weights[f"{prefix}.k_norm"] = np.tile(kn, num_kv_heads)

            is_dense = layer_idx in mlp_only_layers

            if is_dense:
                # Dense SwiGLU MLP
                gate_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.gate_proj.weight")
                up_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.up_proj.weight")
                down_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.down_proj.weight")

                weights[f"{prefix}.w_gate"] = _transpose_2d(
                    gate_raw, "gate_proj", precision=precision)
                weights[f"{prefix}.w_up"] = _transpose_2d(
                    up_raw, "up_proj", precision=precision)
                weights[f"{prefix}.w_down"] = _transpose_2d(
                    down_raw, "down_proj", precision=precision)
                del gate_raw, up_raw, down_raw
            else:
                # MoE layer: router + per-expert + shared expert
                router_raw = _load_tensor(
                    readers, f"{hf_prefix}.mlp.gate.weight")
                weights[f"{prefix}.router"] = _transpose_2d(
                    router_raw, "router", precision=precision)
                del router_raw

                # Pack expert weights per layer so the TRT graph can use a
                # handful of batched matmuls instead of one branch per expert.
                expert_w_gate = np.empty(
                    (num_experts, hidden, moe_intermediate_size),
                    dtype=weight_dtype,
                )
                expert_w_up = np.empty(
                    (num_experts, hidden, moe_intermediate_size),
                    dtype=weight_dtype,
                )
                expert_w_down = np.empty(
                    (num_experts, moe_intermediate_size, hidden),
                    dtype=weight_dtype,
                )

                for e in range(num_experts):
                    exp_hf = f"{hf_prefix}.mlp.experts.{e}"
                    gate_raw = _load_tensor(
                        readers, f"{exp_hf}.gate_proj.weight")
                    up_raw = _load_tensor(
                        readers, f"{exp_hf}.up_proj.weight")
                    down_raw = _load_tensor(
                        readers, f"{exp_hf}.down_proj.weight")

                    expert_w_gate[e] = _transpose_2d(
                        gate_raw, f"expert_{e}_gate", precision=precision)
                    expert_w_up[e] = _transpose_2d(
                        up_raw, f"expert_{e}_up", precision=precision)
                    expert_w_down[e] = _transpose_2d(
                        down_raw, f"expert_{e}_down", precision=precision)
                    del gate_raw, up_raw, down_raw

                weights[f"{prefix}.experts.w_gate"] = expert_w_gate
                weights[f"{prefix}.experts.w_up"] = expert_w_up
                weights[f"{prefix}.experts.w_down"] = expert_w_down

                # Shared expert weights (Qwen2.5-MoE only)
                if has_shared_expert:
                    shared_hf = f"{hf_prefix}.mlp.shared_expert"
                    s_gate_raw = _load_tensor(
                        readers, f"{shared_hf}.gate_proj.weight")
                    s_up_raw = _load_tensor(
                        readers, f"{shared_hf}.up_proj.weight")
                    s_down_raw = _load_tensor(
                        readers, f"{shared_hf}.down_proj.weight")

                    weights[f"{prefix}.shared_expert.w_gate"] = _transpose_2d(
                        s_gate_raw, "shared_gate", precision=precision)
                    weights[f"{prefix}.shared_expert.w_up"] = _transpose_2d(
                        s_up_raw, "shared_up", precision=precision)
                    weights[f"{prefix}.shared_expert.w_down"] = _transpose_2d(
                        s_down_raw, "shared_down", precision=precision)
                    del s_gate_raw, s_up_raw, s_down_raw

                    # Shared expert gate (sigmoid gating weight)
                    shared_gate_key = (
                        f"{hf_prefix}.mlp.shared_expert_gate.weight")
                    if _has_tensor(readers, shared_gate_key):
                        sg_raw = _load_tensor(readers, shared_gate_key)
                        weights[f"{prefix}.shared_expert_gate"] = (
                            sg_raw.astype(weight_dtype))
                        del sg_raw

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head",
                precision=precision)
        else:
            weights["w_out"] = _transpose_2d(
                embedding.copy(), "embedding_tied", precision=precision)

        # Metadata for engine builder
        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_num_experts"] = num_experts  # type: ignore[assignment]
        weights["_num_experts_per_tok"] = num_experts_per_tok  # type: ignore[assignment]
        weights["_moe_intermediate_size"] = moe_intermediate_size  # type: ignore[assignment]
        weights["_shared_expert_intermediate_size"] = shared_expert_intermediate_size  # type: ignore[assignment]
        weights["_dense_intermediate_size"] = dense_intermediate_size  # type: ignore[assignment]
        weights["_mlp_only_layers"] = sorted(mlp_only_layers)  # type: ignore[assignment]
        weights["_has_shared_expert"] = has_shared_expert  # type: ignore[assignment]

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False, parallel_config=None,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        from ...parallel_config import normalize_parallel_config
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            # Local import keeps the dense build path free of TP code.
            from .dual_profile_decoder_tp_builder import (
                build_dual_profile_tp_moe_decoder_engine,
            )
            return build_dual_profile_tp_moe_decoder_engine(
                config, weights, max_cache_length, precision=precision,
                quant_ctx=quant_ctx, verbose=verbose,
                debug_layer_outputs=debug_layer_outputs,
                parallel_config=parallel)
        attention_size: int = weights.get(
            "_attention_size", config.attention_size)
        num_experts: int = weights["_num_experts"]
        moe_intermediate: int = weights["_moe_intermediate_size"]
        shared_expert_intermediate: int = weights[
            "_shared_expert_intermediate_size"]
        dense_intermediate: int = weights["_dense_intermediate_size"]
        top_k: int = weights["_num_experts_per_tok"]
        mlp_only_layers: list[int] = weights.get("_mlp_only_layers", [])
        mlp_only_set = set(mlp_only_layers)
        has_shared_expert: bool = weights.get("_has_shared_expert", True)

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

        # -----------------------------------------------------------
        # Shared constants
        # -----------------------------------------------------------
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"],
            dtype=work_np_dtype)

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
            is_dense = layer_idx in mlp_only_set

            result = _add_qwen3_moe_decoder_layer(
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
                has_shared_expert=has_shared_expert,
                dtype=work_np_dtype,
            )

            hidden_state = result["hidden"]
            present_k_outputs.append(result["present_k"])
            present_v_outputs.append(result["present_v"])

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
            print(f"[trtmc build] Building Qwen MoE TRT engine "
                  f"({num_layers} layers, hidden={hidden}, "
                  f"attn={attention_size}, experts={num_experts}, "
                  f"top_k={top_k}, moe_inter={moe_intermediate}, "
                  f"shared_expert={has_shared_expert}, "
                  f"shared_inter={shared_expert_intermediate}, "
                  f"dense_inter={dense_intermediate}, "
                  f"dense_layers={sorted(mlp_only_set)}, "
                  f"cache={max_cache_length}, precision={precision}) ...",
                  file=sys.stderr)

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
    *,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Compute a single SwiGLU expert: down(silu(gate(x)) * up(x))."""
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
        network, gated.get_output(0), intermediate_size, hidden_size, w_down,
        dtype=dtype)
    return down


def _add_packed_swiglu_experts(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Compute all expert outputs with three batched matmuls.

    Returns a tensor of shape [num_experts, 1, hidden_size].
    """
    num_experts, _, intermediate_size = w_gate.shape

    inp_3d = network.add_shuffle(inp)
    inp_3d.reshape_dims = (1, 1, hidden_size)

    expert_scale = graph_ops.add_constant(
        network, (num_experts, 1, 1),
        np.ones((num_experts, 1, 1), dtype=dtype),
        dtype=dtype,
    )
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

    sigmoid = network.add_activation(gate.get_output(0), trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate.get_output(0), sigmoid.get_output(0),
        trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up.get_output(0),
        trt.ElementWiseOperation.PROD)

    down = network.add_matrix_multiply(
        gated.get_output(0), trt.MatrixOperation.NONE,
        down_w, trt.MatrixOperation.NONE)
    return down.get_output(0)


def _add_qwen3_moe_block(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    num_experts: int,
    moe_intermediate: int,
    shared_expert_intermediate: int,
    top_k: int,
    has_shared_expert: bool = True,
    dtype: np.dtype = np.float32,
) -> trt.ITensor:
    """Add Qwen MoE block with top-k softmax routing and optional shared expert.

    Steps:
      1. Router logits -> softmax -> top-k -> renormalize
      2. Compute all routed expert outputs, gather top-k, weighted sum
      3. (If has_shared_expert) Compute shared expert output (always active)
      4. (If has_shared_expert) Gate shared expert with sigmoid
      5. Final = routed_output [+ gated_shared_output]
    """
    # 1. Router logits
    router_logits = graph_ops.add_matmul_rhs_constant(
        network, inp, hidden_size, num_experts,
        weights[f"{prefix}.router"], dtype=dtype)

    # 2. Softmax over router logits
    sm = network.add_softmax(router_logits)
    sm.axes = 1 << 1

    # 3. TopK selection
    topk = network.add_topk(
        sm.get_output(0), trt.TopKOperation.MAX, top_k, 1 << 1)
    top_values = topk.get_output(0)   # [1, top_k]
    top_indices = topk.get_output(1)  # [1, top_k]

    # 4. Renormalize: values / sum(values)
    sum_val = network.add_reduce(
        top_values, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
    norm_weights = network.add_elementwise(
        top_values, sum_val.get_output(0),
        trt.ElementWiseOperation.DIV)  # [1, top_k]

    # 5. Compute all expert outputs with three packed batched matmuls.
    expert_outputs = _add_packed_swiglu_experts(
        network, inp, hidden_size,
        weights[f"{prefix}.experts.w_gate"],
        weights[f"{prefix}.experts.w_up"],
        weights[f"{prefix}.experts.w_down"],
        dtype=dtype,
    )

    # 6. Gather selected experts, scale, and sum
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
        # Pure routed MoE (Qwen3-MoE): no shared expert
        return routed_result

    # 7. Shared expert output (always active, Qwen2.5-MoE)
    shared_out = _add_swiglu_expert(
        network, inp, hidden_size, shared_expert_intermediate,
        weights[f"{prefix}.shared_expert.w_gate"],
        weights[f"{prefix}.shared_expert.w_up"],
        weights[f"{prefix}.shared_expert.w_down"],
        dtype=dtype,
    )

    # 8. Gate shared expert with sigmoid
    shared_gate_w = weights.get(f"{prefix}.shared_expert_gate")
    if shared_gate_w is not None:
        # shared_expert_gate weight shape: [1, hidden] — compute gate score
        # gate = sigmoid(inp @ shared_expert_gate^T) where inp is [1, hidden]
        # shared_gate_w stored as raw [1, hidden], use as matmul constant
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

    # 9. Combine: routed_output + gated_shared_output
    combined = network.add_elementwise(
        routed_result, shared_final, trt.ElementWiseOperation.SUM)

    return combined.get_output(0)


def _add_qwen3_moe_decoder_layer(
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
    is_dense: bool,
    num_experts: int,
    moe_intermediate: int,
    shared_expert_intermediate: int,
    dense_intermediate: int,
    top_k: int,
    has_shared_expert: bool = True,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add one Qwen MoE decoder layer: attention + (dense MLP or MoE)."""

    # Attention block (pre-norm -> QKV -> RoPE -> cache -> attn -> out proj)
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
    attn_out = attn["attn_out"]

    # Residual connection
    residual1 = network.add_elementwise(
        hidden, attn_out, trt.ElementWiseOperation.SUM)

    # Post-attention RMSNorm
    norm2 = _apply_norm(
        network, residual1.get_output(0), hidden_size,
        weights[f"{prefix}.post_attn_norm"],
        None, eps_tensor, "rmsnorm", dtype=dtype)

    # MLP: either dense SwiGLU or MoE (with optional shared expert)
    if is_dense:
        mlp_out = graph_blocks.add_swiglu_mlp(
            network, norm2,
            weights=weights,
            prefix=prefix,
            hidden_size=hidden_size,
            mlp_size=dense_intermediate,
            dtype=dtype)
    else:
        mlp_out = _add_qwen3_moe_block(
            network, norm2, weights, prefix,
            hidden_size, num_experts, moe_intermediate,
            shared_expert_intermediate, top_k,
            has_shared_expert=has_shared_expert,
            dtype=dtype)

    # Residual connection
    residual2 = network.add_elementwise(
        residual1.get_output(0), mlp_out, trt.ElementWiseOperation.SUM)

    return {
        "hidden": residual2.get_output(0),
        "post_attn": residual1.get_output(0),
        "present_k": attn["present_k"],
        "present_v": attn["present_v"],
    }


plugin = Qwen3MoePlugin()
