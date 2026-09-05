# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-4-multimodal family plugin — vision-adapted text decoder.

Phi-4-multimodal stores base weights under `*.base_layer.weight` (LoRA adapters
are in `*.lora_A.*` / `*.lora_B.*`). Vision inference uses the merged vision
adapter on every decoder projection.
The text decoder is Phi-3 architecture with partial_rotary_factor=0.75.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    readers,
    base_key: str,
    config: ModelConfig,
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
            f"A={lora_a.shape}, B={lora_b.shape}, configured rank={rank}"
        )
    return base + (lora_b @ lora_a) * (alpha / rank)


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _Phi4MultimodalModel:
    embed_input = True

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
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
            f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
        )
        weights["embedding"] = embedding.astype(np.float32)

        mlp_size = 0
        attention_size = 0

        for layer_idx in range(num_layers):
            prefix = f"layer.{layer_idx}"
            hf_prefix = f"model.layers.{layer_idx}"

            # Norms (1D, no transpose, no LoRA)
            input_norm = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
            post_norm = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
            weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
            weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

            # ---- Fused QKV projection (base_layer) ----
            # Shape: [q_dim + 2*kv_dim, hidden]
            qkv_raw = _load_vision_adapted_weight(
                readers, f"{hf_prefix}.self_attn.qkv_proj.base_layer.weight", config
            )
            total_qkv = qkv_raw.shape[0]
            expected_qkv = q_dim + 2 * kv_dim
            assert total_qkv == expected_qkv, (
                f"Layer {layer_idx} qkv_proj rows {total_qkv} != "
                f"expected {expected_qkv} (q={q_dim}, kv={kv_dim})"
            )

            q_raw = qkv_raw[:q_dim, :]
            k_raw = qkv_raw[q_dim : q_dim + kv_dim, :]
            v_raw = qkv_raw[q_dim + kv_dim :, :]
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
                readers, f"{hf_prefix}.self_attn.o_proj.base_layer.weight", config
            )
            weights[f"{prefix}.w_o"] = _transpose_2d(o_raw, "o_proj")
            del o_raw

            # ---- Fused gate_up projection (base_layer) ----
            # Shape: [2 * intermediate_size, hidden]
            gate_up_raw = _load_vision_adapted_weight(
                readers, f"{hf_prefix}.mlp.gate_up_proj.base_layer.weight", config
            )
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
                readers, f"{hf_prefix}.mlp.down_proj.base_layer.weight", config
            )
            weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down_proj")
            del down_raw

        # Final norm
        final_norm_key = "model.norm.weight"
        if _has_tensor(readers, final_norm_key):
            weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
        else:
            weights["final_norm"] = np.ones(hidden, dtype=np.float32)

        # LM head (tied embeddings — no lm_head.weight in this model)
        lm_head_key = "lm_head.weight"
        if _has_tensor(readers, lm_head_key):
            weights["w_out"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
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
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
        debug_layer_outputs: bool = False,
    ) -> bytes:
        from .default_decoder import build_standard_decoder_engine

        partial_rotary = config.raw.get("partial_rotary_factor", 1.0)
        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            quant_ctx=quant_ctx,
            partial_rotary_factor=partial_rotary,
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
    ) -> bytes:
        from .phi4mm_vision_builder import build_phi4mm_vision_engine

        del config, weights
        return build_phi4mm_vision_engine(
            _load_vision_weights(model_dir), precision=precision, verbose=verbose
        )

    def get_vl_config(self, config: ModelConfig) -> dict:
        return {
            "image_token_id": 200010,
            "fixed_image_size": 448,
            "patch_size": 14,
            "merge_size": 2,
            "temporal_patch_size": 1,
            "num_image_pad_tokens": 721,
            "vision_output_dim": config.hidden_size,
            "preprocessor_type": "phi4_hd_chw",
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "interpolation": "bilinear",
            "vl_prompt_template": ("<|user|>{image_pads}{prompt}<|end|><|assistant|>"),
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
                weights[key.removeprefix(checkpoint_prefix)] = _load_tensor(readers, key)
    if not weights:
        raise RuntimeError("Phi-4 checkpoint contains no image tower weights")
    return weights


def _tokenizer_runtime_contract(model_dir: Path) -> dict[str, object]:
    """Resolve this family's exact native-tokenizer framing."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_fast=True,
    )
    default_ids = list(tokenizer.encode("hello"))
    plain_ids = list(tokenizer.encode("hello", add_special_tokens=False))
    if default_ids == plain_ids:
        prefix_ids, suffix_ids = [], []
    elif not plain_ids:
        prefix_ids, suffix_ids = default_ids, []
    else:
        frame = next(
            (
                start
                for start in range(len(default_ids) - len(plain_ids) + 1)
                if default_ids[start : start + len(plain_ids)] == plain_ids
            ),
            None,
        )
        if frame is None:
            raise RuntimeError("tokenizer special-token framing is not a prefix/suffix")
        prefix_ids = default_ids[:frame]
        suffix_ids = default_ids[frame + len(plain_ids) :]
    return {
        "tokenizer_add_special_tokens": False,
        "tokenizer_prefix_ids": prefix_ids,
        "tokenizer_suffix_ids": suffix_ids,
    }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Phi-4 Multimodal vision-language bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("phi4_multimodal does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("phi4_multimodal does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("phi4_multimodal does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("phi4_multimodal does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("phi4_multimodal does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "vision_language_generation":
        raise ValueError("phi4_multimodal supports only task=vision_language_generation")
    if (
        request.tensor_parallel_size != 1
        or request.quantization not in {None, "none"}
        or request.fp32_layers
    ):
        raise NotImplementedError(
            "Phi-4 Multimodal supports only single-device non-quantized builds"
        )
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() not in {"phi4mm", "phi4_multimodal"}:
        raise ValueError(f"Phi-4 Multimodal does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    max_length = int(request.max_sequence_length or min(config.max_position_embeddings, 256))
    config.raw["_model_dir"] = str(model_dir)
    model = _Phi4MultimodalModel()
    weights = model.load_weights(str(model_dir), config)
    config.raw["_decoder_engine_role"] = "prefill"
    prefill = model.build_engine(
        config,
        weights,
        max_length,
        precision=precision,
        quant_ctx=None,
        verbose=request.verbose,
    )
    config.raw["_decoder_engine_role"] = "decode"
    decode = model.build_engine(
        config,
        weights,
        max_length,
        precision=precision,
        quant_ctx=None,
        verbose=request.verbose,
    )
    config.raw.pop("_decoder_engine_role", None)
    vision = model.build_vision_engine(
        str(model_dir), config, weights, precision=precision, verbose=request.verbose
    )
    if vision is None:
        raise RuntimeError("Phi-4 Multimodal vision build returned no engine")
    vl = model.get_vl_config(config) or {}
    runtime = {
        "tensor_parallel_size": 1,
        "num_layers": config.num_hidden_layers,
        "max_cache_length": max_length,
        "vocab_size": config.vocab_size,
        "id_bos": config.bos_token_id,
        "id_eos": config.eos_token_id,
        "image_token_id": int(vl.get("image_token_id", -1)),
        "vision_output_dim": int(vl.get("vision_output_dim", config.hidden_size)),
        "prefill_max_length": int(vl.get("prefill_max_length", max_length)),
        "io_map": {
            "cache_k_pattern": "cache_k_{i}",
            "cache_v_pattern": "cache_v_{i}",
            "present_k_pattern": "present_k_{i}",
            "present_v_pattern": "present_v_{i}",
        },
    }
    runtime.update(vl)
    writer.set_header(family="phi4_multimodal", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", decode)
    writer.add_bytes("prefill.plan", prefill)
    writer.add_bytes("vision.plan", vision)
    runtime.update(_tokenizer_runtime_contract(model_dir))
    writer.add_json("runtime.json", runtime)
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.model",
    ):
        path = model_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
