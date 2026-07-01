# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-4 family plugin — handles fused gate_up_proj splitting."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import ModelConfig
from .weights import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
    _target_np_dtype,
)
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .model.parallel import build_dual_profile_tp_decoder_engine
from .model.model import build_standard_decoder_engine


class GlmPlugin:
    name = "glm"
    runtime_strategy = "glm_decoder_kv_cache"
    runtime_capabilities = {"decoder_kv"}

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "glm"

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "fp32",
    ) -> WeightDict:
        """Load GLM-4 weights, splitting fused gate_up_proj."""
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        kv_attention_size = config.num_key_value_heads * config.head_dim
        target_dtype = _target_np_dtype(precision)

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(target_dtype)

        def _load_layer(layer_idx: int) -> tuple[int, WeightDict, int, int]:
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"
            layer = WeightDict()

            # Norms (1D, no transpose)
            input_norm = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
            post_norm = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
            layer[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            layer[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # ---- Separate Q/K/V projections ----
            q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
            k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
            v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")

            # Transpose [out, in] -> [in, out]
            q_t = _transpose_2d(q_raw, "q_proj", precision=precision)
            k_t = _transpose_2d(k_raw, "k_proj", precision=precision)
            v_t = _transpose_2d(v_raw, "v_proj", precision=precision)

            # Keep compact GQA/MQA K/V

            layer[f"{prefix}.w_q"] = q_t
            layer[f"{prefix}.w_k"] = k_t
            layer[f"{prefix}.w_v"] = v_t

            # Q/K/V biases (GLM-4 has biases on Q, K, V but NOT O)
            q_bias_key = f"{hf_prefix}.self_attn.q_proj.bias"
            k_bias_key = f"{hf_prefix}.self_attn.k_proj.bias"
            v_bias_key = f"{hf_prefix}.self_attn.v_proj.bias"
            if _has_tensor(readers, q_bias_key):
                layer[f"{prefix}.q_bias"] = _load_tensor(readers, q_bias_key).astype(target_dtype)
            if _has_tensor(readers, k_bias_key):
                layer[f"{prefix}.k_bias"] = _load_tensor(readers, k_bias_key).astype(target_dtype)
            if _has_tensor(readers, v_bias_key):
                layer[f"{prefix}.v_bias"] = _load_tensor(readers, v_bias_key).astype(target_dtype)

            # Output projection (no bias in GLM-4)
            o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")
            layer[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj", precision=precision)

            # ---- Fused gate_up projection ----
            # Shape: [2 * intermediate_size, hidden]
            gate_up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_up_proj.weight")
            intermediate = gate_up_raw.shape[0] // 2

            gate_raw = gate_up_raw[:intermediate, :]
            up_raw = gate_up_raw[intermediate:, :]
            del gate_up_raw

            layer[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj", precision=precision)
            layer[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj", precision=precision)
            del gate_raw, up_raw

            # Down projection
            down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")
            layer[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj", precision=precision)

            return layer_idx, layer, q_raw.shape[0], intermediate

        layer_results: list[tuple[int, WeightDict, int, int] | None] = [None] * num_layers
        max_workers = min(8, max(1, os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_load_layer, i) for i in range(num_layers)]
            for future in as_completed(futures):
                layer_idx, layer, attention_size, mlp_size = future.result()
                layer_results[layer_idx] = (layer_idx, layer, attention_size, mlp_size)

        attention_size = 0
        mlp_size = 0
        for result in layer_results:
            if result is None:
                continue
            _layer_idx, layer, layer_attention_size, layer_mlp_size = result
            weights.update(layer)
            if attention_size == 0:
                attention_size = layer_attention_size
            if mlp_size == 0:
                mlp_size = layer_mlp_size

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head", precision=precision
            )
        else:
            weights["w_out"] = _transpose_2d(
                embedding.copy(), "embedding_tied", precision=precision
            )

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_kv_attention_size"] = kv_attention_size  # type: ignore[assignment]
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]

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
        # GLM-4 uses partial RoPE (default 0.5) with interleaved layout.
        partial_rotary_factor = config.raw.get("partial_rotary_factor", 0.5)
        interleaved_rope = config.model_type.lower() == self.name
        parallel = normalize_parallel_config(parallel_config)

        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(parallel, feature="GLM tensor-parallel builds")
            if quant_ctx is not None:
                raise ValueError("GLM tensor-parallel builds do not support quantization")
            if debug_layer_outputs:
                raise ValueError("GLM tensor-parallel builds do not support debug_layer_outputs")
            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                partial_rotary_factor=partial_rotary_factor,
                interleaved_rope=interleaved_rope,
                verbose=verbose,
                parallel_config=parallel,
            )

        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            partial_rotary_factor=partial_rotary_factor,
            interleaved_rope=interleaved_rope,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )


plugin = GlmPlugin()
