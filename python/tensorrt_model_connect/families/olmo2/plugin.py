# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OLMo-2 family plugin -- post-norm decoder with QK normalization.

OLMo-2 (allenai/OLMo-2-0425-1B) uses:
  - Post-norm residual layout: norm is applied to attn/MLP output BEFORE
    the residual addition (unlike LLaMA pre-norm).
  - QK normalization (RMSNorm on Q and K per-head before RoPE)
  - SwiGLU MLP (gate_proj / up_proj / down_proj)
  - RoPE position embeddings
  - Untied word embeddings (has separate lm_head)
  - No input_layernorm; uses post_attention_layernorm + post_feedforward_layernorm

Layer pattern:
  attn_out = self_attn(hidden)            # QK norm inside
  normed_attn = post_attention_layernorm(attn_out)
  residual1 = hidden + normed_attn
  mlp_out = mlp(residual1)
  normed_mlp = post_feedforward_layernorm(mlp_out)
  hidden = residual1 + normed_mlp
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)


class Olmo2Plugin:
    name = "olmo2"

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "olmo2"

    runtime_strategy = "olmo2_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} !== ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        mlp_size = 0
        attention_size = 0
        kv_attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # OLMo-2 norms: post_attention_layernorm and post_feedforward_layernorm
            post_attn_norm = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.post_attn_norm"] = post_attn_norm.astype(np.float32)

            post_ff_norm = _load_tensor(
                readers, f"{hf_prefix}.post_feedforward_layernorm.weight")
            weights[f"{prefix}.post_ff_norm"] = post_ff_norm.astype(np.float32)

            # Q/K/V/O projections
            q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
            o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")

            q_hidden = q_raw.shape[0]
            if attention_size == 0:
                attention_size = q_hidden

            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")
            o_t = _transpose_2d(o_raw, "o_proj")

            # Compact GQA/MQA K/V

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t
            weights[f"{prefix}.w_o"] = o_t
            if kv_attention_size == 0:
                kv_attention_size = k_t.shape[1]

            # QK normalization -- OLMo-2 q_norm/k_norm are already
            # full-size (num_heads * head_dim), NOT per-head like Qwen3.
            # Load directly without _repeat_head_norm.
            q_norm_key = f"{hf_prefix}.self_attn.q_norm.weight"
            k_norm_key = f"{hf_prefix}.self_attn.k_norm.weight"
            if _has_tensor(readers, q_norm_key):
                weights[f"{prefix}.q_norm"] = _load_tensor(
                    readers, q_norm_key).astype(np.float32)
            if _has_tensor(readers, k_norm_key):
                weights[f"{prefix}.k_norm"] = _load_tensor(
                    readers, k_norm_key).astype(np.float32)

            # MLP
            gate_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_proj.weight")
            up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.up_proj.weight")
            down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")
            if mlp_size == 0:
                mlp_size = gate_raw.shape[0]

            weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj")
            weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
            weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")

        # Final norm
        weights["final_norm"] = _load_tensor(
            readers, "model.norm.weight").astype(np.float32)

        # LM head (untied)
        weights["w_out"] = _transpose_2d(
            _load_tensor(readers, "lm_head.weight"), "lm_head")

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = kv_attention_size  # type: ignore[assignment]
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        """Build TRT engine with OLMo-2 post-norm residual layout."""
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="OLMo2 tensor-parallel builds")
            from .tp_builder import build_olmo2_tp_engine
            return build_olmo2_tp_engine(
                config, weights, max_cache_length,
                verbose=verbose,
                parallel_config=parallel)

        import sys
        from tensorrt_model_connect import trt_compat
        trt = trt_compat.get_trt()
        from . import graph_ops
        from . import graph_blocks

        attention_size: int = weights.get("_attention_size", config.attention_size)
        mlp_size: int = weights.get("_mlp_size", config.intermediate_size)
        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = attention_size // num_heads
        kv_attention_size = graph_blocks.infer_kv_attention_size(
            weights, num_kv_heads=num_kv_heads, head_dim=head_dim)
        attention_window = max_cache_length + 1

        logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        trt_config = builder.create_builder_config()
        trt_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        trt_config.clear_flag(trt.BuilderFlag.TF32)

        # Inputs
        token_id = network.add_input("token_id", trt.int32, (1,))
        position_id = network.add_input("position_id", trt.int32, (1,))
        attention_mask = network.add_input(
            "attention_mask", trt.float32, (1, attention_window))

        cache_k_inputs = []
        cache_v_inputs = []
        for i in range(num_layers):
            ck = network.add_input(
                graph_ops.layer_tensor_name("cache_k", i),
                trt.float32, (max_cache_length, kv_attention_size))
            cv = network.add_input(
                graph_ops.layer_tensor_name("cache_v", i),
                trt.float32, (max_cache_length, kv_attention_size))
            cache_k_inputs.append(ck)
            cache_v_inputs.append(cv)

        # Constants
        embedding_table = graph_ops.add_constant(
            network, (vocab, hidden), weights["embedding"])

        cos_table_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, True)
        sin_table_np = graph_ops.make_rope_table_half_dim(
            attention_window, head_dim, config.rope_theta, False)

        cos_tensor = graph_ops.add_constant(
            network, cos_table_np.shape, cos_table_np)
        sin_tensor = graph_ops.add_constant(
            network, sin_table_np.shape, sin_table_np)

        eps_tensor = graph_ops.add_constant(
            network, (1, 1), np.array([config.rms_norm_eps], dtype=np.float32))

        # Embedding lookup
        gather = network.add_gather(embedding_table, token_id, 0)
        hidden_state = gather.get_output(0)

        # Decoder layers
        present_k_outputs = []
        present_v_outputs = []

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"

            # ---- Attention (no pre-norm, QK norm inside) ----
            q = graph_ops.add_matmul_rhs_constant(
                network, hidden_state, hidden, attention_size,
                weights[f"{prefix}.w_q"])
            k = graph_ops.add_matmul_rhs_constant(
                network, hidden_state, hidden, kv_attention_size,
                weights[f"{prefix}.w_k"])
            v = graph_ops.add_matmul_rhs_constant(
                network, hidden_state, hidden, kv_attention_size,
                weights[f"{prefix}.w_v"])

            # QK RMSNorm (full-dim, NOT per-head -- OLMo-2 applies norm
            # over the entire num_heads*head_dim dimension before reshape)
            q_norm_w = weights.get(f"{prefix}.q_norm")
            if q_norm_w is not None:
                q = graph_ops.add_rms_norm(
                    network, q, attention_size, q_norm_w, eps_tensor)
            k_norm_w = weights.get(f"{prefix}.k_norm")
            if k_norm_w is not None:
                k = graph_ops.add_rms_norm(
                    network, k, kv_attention_size, k_norm_w, eps_tensor)

            # RoPE
            q = graph_ops.add_apply_rope_native(
                network, q, num_heads, head_dim,
                cos_tensor, sin_tensor, position_id, head_dim)
            k = graph_ops.add_apply_rope_native(
                network, k, num_kv_heads, head_dim,
                cos_tensor, sin_tensor, position_id, head_dim)

            # Save present K/V
            present_k = k
            present_v = v

            # Cache concat
            k_reshape = network.add_shuffle(k)
            k_reshape.reshape_dims = (1, kv_attention_size)
            v_reshape = network.add_shuffle(v)
            v_reshape.reshape_dims = (1, kv_attention_size)

            all_k = network.add_concatenation(
                [cache_k_inputs[layer_idx], k_reshape.get_output(0)])
            all_k.axis = 0
            all_v = network.add_concatenation(
                [cache_v_inputs[layer_idx], v_reshape.get_output(0)])
            all_v.axis = 0

            mask_reshape = network.add_shuffle(attention_mask)
            mask_reshape.reshape_dims = (1, 1, 1, attention_window)

            context_flat = graph_ops.add_attention_from_rows(
                network, q, all_k.get_output(0), all_v.get_output(0),
                num_heads=num_heads, head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                q_seq=1, kv_seq=attention_window,
                mask=mask_reshape.get_output(0))

            # Output projection
            attn_out = graph_ops.add_matmul_rhs_constant(
                network, context_flat,
                attention_size, hidden, weights[f"{prefix}.w_o"])

            # ---- Post-attention norm ----
            normed_attn = graph_ops.add_rms_norm(
                network, attn_out, hidden,
                weights[f"{prefix}.post_attn_norm"], eps_tensor)
            residual1 = network.add_elementwise(
                hidden_state, normed_attn,
                trt.ElementWiseOperation.SUM)
            post_attn_state = residual1.get_output(0)

            # ---- MLP (SwiGLU, no pre-norm) ----
            mlp_out = graph_blocks.add_swiglu_mlp(
                network, post_attn_state, weights=weights, prefix=prefix,
                hidden_size=hidden, mlp_size=mlp_size)

            # ---- Post-feedforward norm ----
            normed_mlp = graph_ops.add_rms_norm(
                network, mlp_out, hidden,
                weights[f"{prefix}.post_ff_norm"], eps_tensor)
            residual2 = network.add_elementwise(
                post_attn_state, normed_mlp,
                trt.ElementWiseOperation.SUM)
            hidden_state = residual2.get_output(0)

            present_k_outputs.append(present_k)
            present_v_outputs.append(present_v)

        # Final norm
        hidden_state = graph_ops.add_rms_norm(
            network, hidden_state, hidden,
            weights["final_norm"], eps_tensor)

        # LM head
        out_vocab = weights["w_out"].shape[1] if isinstance(weights["w_out"], np.ndarray) else vocab
        logits = graph_ops.add_matmul_rhs_constant(
            network, hidden_state, hidden, out_vocab, weights["w_out"])
        b_out = np.zeros(out_vocab, dtype=np.float32)
        logits = graph_ops.add_bias_sum(network, logits, out_vocab, b_out)

        logits.name = "logits"
        network.mark_output(logits)

        # Present K/V outputs
        for i in range(num_layers):
            pk = present_k_outputs[i]
            pv = present_v_outputs[i]
            pk.name = graph_ops.layer_tensor_name("present_k", i)
            pv.name = graph_ops.layer_tensor_name("present_v", i)
            network.mark_output(pk)
            network.mark_output(pv)

        # Build engine
        if verbose:
            print(f"[trtmc build] Building TRT engine ({num_layers} layers, "
                  f"hidden={hidden}, attn={attention_size}, mlp={mlp_size}, "
                  f"cache={max_cache_length}) ...", file=sys.stderr)

        plan = builder.build_serialized_network(network, trt_config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        return bytes(plan)


plugin = Olmo2Plugin()
