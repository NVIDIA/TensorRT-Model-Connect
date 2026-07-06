# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-4-multimodal family plugin — vision-adapted text decoder.

Phi-4-multimodal stores base weights under `*.base_layer.weight` (LoRA adapters
are in `*.lora_A.*` / `*.lora_B.*`). Vision inference uses the merged vision
adapter on every decoder projection.
The text decoder is Phi-3 architecture with partial_rotary_factor=0.75.
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


def _load_vision_adapted_weight(
    readers, base_key: str, config: ModelConfig,
) -> np.ndarray:
    """Return a base projection with the checkpoint's vision LoRA merged."""
    base = _load_tensor(readers, base_key).astype(np.float32)
    if not base_key.endswith(".base_layer.weight"):
        return base

    projection_prefix = base_key.removesuffix(".base_layer.weight")
    lora_a_key = f"{projection_prefix}.lora_A.vision.weight"
    lora_b_key = f"{projection_prefix}.lora_B.vision.weight"
    if not (_has_tensor(readers, lora_a_key) and _has_tensor(readers, lora_b_key)):
        return base

    lora_a = _load_tensor(readers, lora_a_key).astype(np.float32)
    lora_b = _load_tensor(readers, lora_b_key).astype(np.float32)
    vision_lora = config.raw.get("vision_lora", {})
    rank = int(vision_lora.get("r", lora_a.shape[0]))
    alpha = float(vision_lora.get("lora_alpha", rank))
    if rank <= 0 or lora_a.shape[0] != rank or lora_b.shape[1] != rank:
        raise ValueError(
            f"Invalid Phi-4 vision LoRA shapes for {projection_prefix}: "
            f"A={lora_a.shape}, B={lora_b.shape}, configured rank={rank}")
    return base + (lora_b @ lora_a) * (alpha / rank)


class Phi4MultimodalPlugin:
    name = "phi4_multimodal"
    runtime_strategy = "phi4_multimodal_vision_language"
    embed_input = True

    def matches(self, model_type: str) -> bool:
        mt = model_type.lower()
        return mt in ("phi4mm", "phi4_multimodal")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
    ) -> WeightDict:
        """Load Phi-4-multimodal weights with the vision LoRA merged."""
        model_dir_path = Path(model_dir)
        readers = _open_safetensors(model_dir_path)

        hidden = config.hidden_size
        vocab = config.vocab_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        weights = WeightDict()

        # Embedding
        embedding = _load_tensor(readers, "model.embed_tokens.weight")
        assert embedding.shape == (vocab, hidden), (
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
        weights["embedding"] = embedding.astype(np.float32)

        mlp_size = 0
        attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # Norms (1D, no transpose, no LoRA)
            input_norm = _load_tensor(
                readers, f"{hf_prefix}.input_layernorm.weight")
            post_norm = _load_tensor(
                readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # ---- Fused QKV projection (base_layer) ----
            # Shape: [q_dim + 2*kv_dim, hidden]
            qkv_raw = _load_vision_adapted_weight(
                readers,
                f"{hf_prefix}.self_attn.qkv_proj.base_layer.weight",
                config)
            total_qkv = qkv_raw.shape[0]
            expected_qkv = q_dim + 2 * kv_dim
            assert total_qkv == expected_qkv, (
                f"Layer {layer_idx} qkv_proj rows {total_qkv} != "
                f"expected {expected_qkv} (q={q_dim}, kv={kv_dim})")

            q_raw = qkv_raw[:q_dim, :]
            k_raw = qkv_raw[q_dim:q_dim + kv_dim, :]
            v_raw = qkv_raw[q_dim + kv_dim:, :]
            del qkv_raw

            if attention_size == 0:
                attention_size = q_dim

            # Transpose [out, in] -> [in, out]
            q_t = _transpose_2d(q_raw, "q_proj")
            k_t = _transpose_2d(k_raw, "k_proj")
            v_t = _transpose_2d(v_raw, "v_proj")
            del q_raw, k_raw, v_raw

            weights[f"{prefix}.w_q"] = q_t
            weights[f"{prefix}.w_k"] = k_t
            weights[f"{prefix}.w_v"] = v_t

            # Output projection (base_layer)
            o_raw = _load_vision_adapted_weight(
                readers,
                f"{hf_prefix}.self_attn.o_proj.base_layer.weight",
                config)
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")
            del o_raw

            # ---- Fused gate_up projection (base_layer) ----
            # Shape: [2 * intermediate_size, hidden]
            gate_up_raw = _load_vision_adapted_weight(
                readers,
                f"{hf_prefix}.mlp.gate_up_proj.base_layer.weight",
                config)
            intermediate = gate_up_raw.shape[0] // 2
            if mlp_size == 0:
                mlp_size = intermediate

            gate_raw = gate_up_raw[:intermediate, :]
            up_raw = gate_up_raw[intermediate:, :]
            del gate_up_raw

            weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate_proj")
            weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up_proj")
            del gate_raw, up_raw

            # Down projection (base_layer)
            down_raw = _load_vision_adapted_weight(
                readers,
                f"{hf_prefix}.mlp.down_proj.base_layer.weight",
                config)
            weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")
            del down_raw

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(
                readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head (tied embeddings — no lm_head.weight in this model)
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(
                _load_tensor(readers, lm_head_key), "lm_head")
        else:
            weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

        weights["_attention_size"] = attention_size  # type: ignore[assignment]
        weights["_mlp_size"] = mlp_size  # type: ignore[assignment]
        # TensorRT 11's fused IAttention compiler rejects the 768+ cache shape
        # required by the canonical Dynamic-HD prompt. Keep the same equation
        # using explicit attention primitives with FP32 score accumulation.
        weights["_explicit_attention"] = True

        return weights

    def build_engine(
        self, config: ModelConfig, weights: WeightDict,
        max_cache_length: int, *, precision: str = "fp32",
        quant_ctx=None, verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        from .standard_decoder_builder import build_standard_decoder_engine

        partial_rotary = config.raw.get("partial_rotary_factor", 1.0)
        return build_standard_decoder_engine(
            config, weights, max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx, partial_rotary_factor=partial_rotary,
            embed_input=True,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs)

    def build_vision_engine(
        self, model_dir: str, config: ModelConfig, weights: WeightDict,
        *, precision: str = "fp32", verbose: bool = False,
    ) -> bytes:
        from .phi4mm_vision_builder import build_phi4mm_vision_engine

        del config, weights
        return build_phi4mm_vision_engine(
            _load_vision_weights(model_dir), precision=precision,
            verbose=verbose)

    def get_vl_config(self, config: ModelConfig) -> dict:
        return {
            "image_token_id": 200010,
            "fixed_image_size": 448,
            "patch_size": 14,
            "num_image_pad_tokens": 721,
            "vision_output_dim": config.hidden_size,
            "preprocessor_type": "phi4_hd_chw",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "interpolation": "bilinear",
            "vl_prompt_template": (
                "<|user|>{image_pads}{prompt}<|end|><|assistant|>"),
            "image_token_str": "<|endoftext10|>",
        }


def _load_vision_weights(model_dir: str) -> WeightDict:
    """Load and canonicalize the checkpoint's image tower weights."""
    readers = _open_safetensors(Path(model_dir))
    checkpoint_prefix = "model.embed_tokens_extend.image_embed."
    weights = WeightDict()
    for reader in readers:
        for key in reader.keys():
            if key.startswith(checkpoint_prefix):
                weights[key.removeprefix(checkpoint_prefix)] = _load_tensor(
                    [reader], key)
    if not weights:
        raise RuntimeError("Phi-4 checkpoint contains no image tower weights")
    return weights


plugin = Phi4MultimodalPlugin()
