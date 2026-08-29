# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocateAnything family plugin.

LocateAnything-3B is a custom-code vision-language model:
  - Vision: MoonViT patch encoder.
  - Projector: LayerNorm + two-layer MLP (``mlp1``).
  - Text: Qwen2.5 decoder under ``language_model.model.*``.

This plugin wires a fixed single-image TRT MC contract: 448x448 image input,
14x14 MoonViT patches, 32x32 patch grid, 2x2 merge, and 256 image tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .checkpoint_mapper import (
    WeightDict,
    _has_tensor,
    _load_tensor,
    _open_safetensors,
    _transpose_2d,
)
from .config import ModelConfig
from ...parallel_config import (
    normalize_parallel_config,
    require_tensorrt_11_for_tensor_parallel,
)
from .default_decoder import build_standard_decoder_engine

if TYPE_CHECKING:
    from ...quantization.context import QuantContext


_DEFAULT_FIXED_IMAGE_SIZE = 448


class LocateAnythingPlugin:
    name = "locateanything"
    runtime_strategy = "locateanything_vision_language"
    embed_input = True

    def matches(self, model_type: str) -> bool:
        return model_type.lower() == "locateanything"

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        return _load_locateanything_text_weights(model_dir, config)

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx: "QuantContext | None" = None,
        verbose: bool = False,
        debug_layer_outputs: bool = False,
        parallel_config=None,
    ) -> bytes:
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            require_tensorrt_11_for_tensor_parallel(
                parallel, feature="LocateAnything tensor-parallel builds")
            if debug_layer_outputs:
                raise ValueError(
                    "LocateAnything tensor-parallel builds do not support "
                    "debug layer outputs")
            from .decoder_tp_builder import build_qwen_vl_tp_decoder_engine

            return build_qwen_vl_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                embed_input=True,
                deepstack_num_levels=0,
                verbose=verbose,
                debug_layer_outputs=debug_layer_outputs,
                parallel_config=parallel,
            )

        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            embed_input=True,
            verbose=verbose,
            debug_layer_outputs=debug_layer_outputs,
        )

    def build_vision_engine(
        self,
        model_dir: str,
        config: ModelConfig,
        weights: WeightDict,
        *,
        precision: str = "fp32",
        verbose: bool = False,
    ) -> bytes | None:
        from .vision_builder import build_locateanything_vision_engine

        return build_locateanything_vision_engine(
            model_dir, config, fixed_image_size=_DEFAULT_FIXED_IMAGE_SIZE,
            verbose=verbose)

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        vision_config = config.raw.get("vision_config")
        if not isinstance(vision_config, dict):
            return None

        patch_size = _first_int(vision_config.get("patch_size", 14), 14)
        merge_kernel = vision_config.get("merge_kernel_size", [2, 2])
        if isinstance(merge_kernel, (list, tuple)) and len(merge_kernel) >= 2:
            merge_h = _first_int(merge_kernel[0], 2)
            merge_w = _first_int(merge_kernel[1], 2)
        else:
            merge_h = merge_w = 2

        fixed_image_size = _DEFAULT_FIXED_IMAGE_SIZE
        grid_h = fixed_image_size // patch_size
        grid_w = fixed_image_size // patch_size
        num_image_tokens = (grid_h * grid_w) // max(merge_h * merge_w, 1)

        return {
            "image_token_id": config.raw.get(
                "image_token_index", config.raw.get("image_token_id", 151665)),
            "fixed_image_size": fixed_image_size,
            "patch_size": patch_size,
            "merge_size": merge_h,
            "num_image_pad_tokens": num_image_tokens,
            "vision_output_dim": config.hidden_size,
            "preprocessor_type": "patchify_chw",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "temporal_patch_size": 1,
            "interpolation": "bicubic",
            "vl_prompt_template": (
                "<|im_start|>system\n"
                "You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "<img>{image_pads}</img>{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "image_token_str": "<IMG_CONTEXT>",
            "locateanything_fixed_vision_grid_h": grid_h,
            "locateanything_fixed_vision_grid_w": grid_w,
            "box_start_token_id": config.raw.get("box_start_token_id", 151668),
            "box_end_token_id": config.raw.get("box_end_token_id", 151669),
            "coord_start_token_id": config.raw.get("coord_start_token_id", 151677),
            "coord_end_token_id": config.raw.get("coord_end_token_id", 152677),
            "ref_start_token_id": config.raw.get("ref_start_token_id", 151672),
            "ref_end_token_id": config.raw.get("ref_end_token_id", 151673),
            "none_token_id": config.raw.get("none_token_id", 4064),
        }

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict | None:
        return {
            "model_type": "locateanything",
            "runtime_strategy": "locateanything_vision_language",
            "embed_input": True,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
        }


def _first_int(value: object, default: int) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_locateanything_text_weights(
    model_dir: str,
    config: ModelConfig,
) -> WeightDict:
    """Load the Qwen text decoder from LocateAnything safetensors."""
    from pathlib import Path

    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    weights = WeightDict()

    embed_key = _first_existing_tensor(
        readers,
        [
            "language_model.model.embed_tokens.weight",
            "model.language_model.model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ],
        "text token embedding",
    )
    embedding = _load_tensor(readers, embed_key)
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})")
    weights["embedding"] = embedding.astype(np.float32)

    layer_prefix = _detect_layer_prefix(readers)
    attention_size = 0
    kv_attention_size = 0
    mlp_size = 0

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hf_prefix = f"{layer_prefix}.{layer_idx}"

        weights[f"{prefix}.input_norm"] = _load_tensor(
            readers, f"{hf_prefix}.input_layernorm.weight").astype(np.float32)
        weights[f"{prefix}.post_attn_norm"] = _load_tensor(
            readers, f"{hf_prefix}.post_attention_layernorm.weight").astype(np.float32)

        q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
        k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
        v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
        o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")

        if attention_size == 0:
            attention_size = q_raw.shape[0]
        if kv_attention_size == 0:
            kv_attention_size = k_raw.shape[0]

        weights[f"{prefix}.w_q"] = _transpose_2d(q_raw, "q_proj")
        weights[f"{prefix}.w_k"] = _transpose_2d(k_raw, "k_proj")
        weights[f"{prefix}.w_v"] = _transpose_2d(v_raw, "v_proj")
        weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")

        for proj_name, weight_key in [
            ("q_bias", "self_attn.q_proj.bias"),
            ("k_bias", "self_attn.k_proj.bias"),
            ("v_bias", "self_attn.v_proj.bias"),
        ]:
            full_key = f"{hf_prefix}.{weight_key}"
            if _has_tensor(readers, full_key):
                weights[f"{prefix}.{proj_name}"] = _load_tensor(
                    readers, full_key).astype(np.float32)

        gate_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_proj.weight")
        up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.up_proj.weight")
        down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")

        if mlp_size == 0:
            mlp_size = gate_raw.shape[0]

        weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate")
        weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up")
        weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down")

    final_norm_key = _first_existing_tensor(
        readers,
        [
            f"{layer_prefix.rsplit('.layers', 1)[0]}.norm.weight",
            "language_model.model.norm.weight",
            "model.norm.weight",
        ],
        "text final norm",
        required=False,
    )
    if final_norm_key is not None:
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)

    lm_head_key = _first_existing_tensor(
        readers,
        [
            "language_model.lm_head.weight",
            "lm_head.weight",
            "model.lm_head.weight",
        ],
        "LM head",
        required=False,
    )
    if lm_head_key is not None:
        weights["w_out"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
    else:
        weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

    weights["_attention_size"] = attention_size
    weights["_kv_attention_size"] = kv_attention_size
    weights["_mlp_size"] = mlp_size
    return weights


def _first_existing_tensor(
    readers,
    names: list[str],
    description: str,
    *,
    required: bool = True,
) -> str | None:
    for name in names:
        if _has_tensor(readers, name):
            return name
    if required:
        raise RuntimeError(f"Cannot find LocateAnything {description} weights")
    return None


def _detect_layer_prefix(readers) -> str:
    for prefix in [
        "language_model.model.layers",
        "model.language_model.model.layers",
        "model.layers",
    ]:
        if _has_tensor(readers, f"{prefix}.0.input_layernorm.weight"):
            return prefix
    raise RuntimeError("Cannot find LocateAnything text decoder layer weights")


plugin = LocateAnythingPlugin()
