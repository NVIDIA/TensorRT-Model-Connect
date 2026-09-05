# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InternVL3 family plugin — vision-language model.

InternVL3-8B-hf architecture:
  - Vision: InternViT-300M-448px (ViT with learned positions, GELU FFN,
    LayerNorm, layer scaling, absolute position embeddings)
  - Projector: LayerNorm + 2-layer MLP (linear_1 + GELU + linear_2)
    with pixel-shuffle downsampling (downsample_ratio=0.5)
  - Text: Qwen2 backbone (standard decoder with RoPE, RMSNorm, SwiGLU, Q/K/V biases)

Detection: model_type == "internvl"
Weight prefix: vision_tower.*, multi_modal_projector.*, language_model.*
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING


import numpy as np

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    _open_safetensors,
    _load_tensor,
    _has_tensor,
    _transpose_2d,
)
from .parallel import ParallelConfig, normalize_parallel_config
from .default_decoder import build_standard_decoder_engine

if TYPE_CHECKING:
    pass

_DEFAULT_FIXED_IMAGE_SIZE = 448


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _InternVLModel:
    embed_input = True

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
    ) -> WeightDict:
        """Load text decoder weights (Qwen2 pattern).

        InternVL3-8B-hf stores text decoder weights under model.language_model.*
        prefix. Falls back to standard model.layers.* if not found.
        """
        return _load_internvl_text_weights(model_dir, config)

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
        """Build text decoder engine (Qwen2 architecture with embed_input for VL)."""
        parallel = normalize_parallel_config(parallel_config)
        if parallel.enabled:
            parallel.validate()
            if debug_layer_outputs:
                raise ValueError(
                    "InternVL tensor-parallel builds do not support debug layer outputs"
                )
            from .tp_builder import build_dual_profile_tp_decoder_engine

            return build_dual_profile_tp_decoder_engine(
                config,
                weights,
                max_cache_length,
                precision=precision,
                quant_ctx=quant_ctx,
                norm_type="rmsnorm",
                mlp_type="swiglu",
                position_type="rope",
                activation="silu",
                embed_input=True,
                verbose=verbose,
                parallel_config=parallel,
            )

        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            verbose=verbose,
            quant_ctx=quant_ctx,
            embed_input=True,
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
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            return None

        vision_weights = _load_vision_and_projector_weights(model_dir, config)

        from .internvit_vision_builder import build_internvit_vision_engine

        return build_internvit_vision_engine(
            config.raw,
            vision_config,
            vision_weights,
            fixed_image_size=_DEFAULT_FIXED_IMAGE_SIZE,
            verbose=verbose,
        )

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            return None

        patch_size_raw = vision_config.get("patch_size", 14)
        patch_size = (
            patch_size_raw[0] if isinstance(patch_size_raw, (list, tuple)) else patch_size_raw
        )
        fixed_image_size = _DEFAULT_FIXED_IMAGE_SIZE
        downsample_ratio = config.raw.get("downsample_ratio", 0.5)

        grid_h = fixed_image_size // patch_size
        grid_w = fixed_image_size // patch_size
        num_patches = grid_h * grid_w

        # Pixel-shuffle downsampling reduces token count
        scale = int(1.0 / downsample_ratio)
        num_output_tokens = num_patches // (scale * scale)

        image_token_id = config.raw.get("image_token_id", 151667)
        image_seq_length = config.raw.get("image_seq_length", num_output_tokens)

        return {
            "image_token_id": image_token_id,
            "fixed_image_size": fixed_image_size,
            "patch_size": patch_size,
            "merge_size": 2,
            "temporal_patch_size": 1,
            "num_image_pad_tokens": image_seq_length,
            "vision_output_dim": config.hidden_size,
            "preprocessor_type": "simple_chw",
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "interpolation": "bicubic",
            "vl_prompt_template": (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                "{image_pads}\n"
                "{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "image_token_str": "<IMG_CONTEXT>",
        }

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict[str, int]:
        """Expose InternVL's nested text decoder contract at bundle scope."""
        return {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "bos_token_id": config.bos_token_id,
        }


# ---------------------------------------------------------------------------
# Text decoder weight loading
# ---------------------------------------------------------------------------


def _load_internvl_text_weights(
    model_dir: str,
    config: ModelConfig,
) -> WeightDict:
    """Load InternVL3 text decoder weights.

    InternVL3-8B-hf uses model.language_model.model.layers.{i}.* prefix.
    Falls back to model.layers.{i}.* if language_model prefix not found.
    The text decoder is standard Qwen2 architecture.
    """
    from pathlib import Path

    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    hidden = config.hidden_size
    vocab = config.vocab_size
    num_layers = config.num_hidden_layers
    weights = WeightDict()

    # Detect prefix: try language_model.model first
    embed_key = "language_model.model.embed_tokens.weight"
    if not _has_tensor(readers, embed_key):
        embed_key = "model.language_model.model.embed_tokens.weight"
    if not _has_tensor(readers, embed_key):
        embed_key = "model.embed_tokens.weight"

    embedding = _load_tensor(readers, embed_key)
    assert embedding.shape == (vocab, hidden), (
        f"Embedding shape {embedding.shape} != ({vocab}, {hidden})"
    )
    weights["embedding"] = embedding.astype(np.float32)

    # Determine layer prefix
    test_key = "language_model.model.layers.0.input_layernorm.weight"
    if _has_tensor(readers, test_key):
        layer_prefix = "language_model.model.layers"
    elif _has_tensor(readers, "model.language_model.model.layers.0.input_layernorm.weight"):
        layer_prefix = "model.language_model.model.layers"
    elif _has_tensor(readers, "model.layers.0.input_layernorm.weight"):
        layer_prefix = "model.layers"
    else:
        raise RuntimeError("Cannot find text decoder layer weights")

    attention_size = 0
    kv_attention_size = 0
    mlp_size = 0

    for layer_idx in range(num_layers):
        prefix = f"layer.{layer_idx}"
        hf_prefix = f"{layer_prefix}.{layer_idx}"

        # Norms
        input_norm = _load_tensor(readers, f"{hf_prefix}.input_layernorm.weight")
        post_norm = _load_tensor(readers, f"{hf_prefix}.post_attention_layernorm.weight")
        weights[f"{prefix}.input_norm"] = input_norm.astype(np.float32)
        weights[f"{prefix}.post_attn_norm"] = post_norm.astype(np.float32)

        # Q/K/V/O projections
        q_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.q_proj.weight")
        k_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.k_proj.weight")
        v_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.v_proj.weight")
        o_raw = _load_tensor(readers, f"{hf_prefix}.self_attn.o_proj.weight")

        q_hidden = q_raw.shape[0]
        if attention_size == 0:
            attention_size = q_hidden
        if kv_attention_size == 0:
            kv_attention_size = k_raw.shape[0]

        q_t = _transpose_2d(q_raw, "q_proj")
        k_t = _transpose_2d(k_raw, "k_proj")
        v_t = _transpose_2d(v_raw, "v_proj")
        o_t = _transpose_2d(o_raw, "o_proj")

        weights[f"{prefix}.w_q"] = q_t
        weights[f"{prefix}.w_k"] = k_t
        weights[f"{prefix}.w_v"] = v_t
        weights[f"{prefix}.w_o"] = o_t

        # Optional QKV biases (Qwen2 has q/k biases)
        for proj_name, weight_key in [
            ("q_bias", "self_attn.q_proj.bias"),
            ("k_bias", "self_attn.k_proj.bias"),
            ("v_bias", "self_attn.v_proj.bias"),
        ]:
            full_key = f"{hf_prefix}.{weight_key}"
            if _has_tensor(readers, full_key):
                raw = _load_tensor(readers, full_key).astype(np.float32)
                weights[f"{prefix}.{proj_name}"] = raw

        # SwiGLU MLP
        gate_raw = _load_tensor(readers, f"{hf_prefix}.mlp.gate_proj.weight")
        up_raw = _load_tensor(readers, f"{hf_prefix}.mlp.up_proj.weight")
        down_raw = _load_tensor(readers, f"{hf_prefix}.mlp.down_proj.weight")

        if mlp_size == 0:
            mlp_size = gate_raw.shape[0]

        weights[f"{prefix}.w_gate"] = _transpose_2d(gate_raw, "gate")
        weights[f"{prefix}.w_up"] = _transpose_2d(up_raw, "up")
        weights[f"{prefix}.w_down"] = _transpose_2d(down_raw, "down")

    # Final norm
    final_norm_key = f"{layer_prefix.rsplit('.layers', 1)[0]}.norm.weight"
    alt_final_norm_key = "language_model.model.norm.weight"
    if _has_tensor(readers, final_norm_key):
        weights["final_norm"] = _load_tensor(readers, final_norm_key).astype(np.float32)
    elif _has_tensor(readers, alt_final_norm_key):
        weights["final_norm"] = _load_tensor(readers, alt_final_norm_key).astype(np.float32)
    elif _has_tensor(readers, "model.norm.weight"):
        weights["final_norm"] = _load_tensor(readers, "model.norm.weight").astype(np.float32)
    else:
        weights["final_norm"] = np.ones(hidden, dtype=np.float32)

    # LM head
    lm_head_key = "language_model.lm_head.weight"
    if not _has_tensor(readers, lm_head_key):
        lm_head_key = "lm_head.weight"
    if _has_tensor(readers, lm_head_key):
        weights["w_out"] = _transpose_2d(_load_tensor(readers, lm_head_key), "lm_head")
    else:
        weights["w_out"] = _transpose_2d(embedding.copy(), "embedding_tied")

    weights["_attention_size"] = attention_size
    weights["_kv_attention_size"] = kv_attention_size
    weights["_mlp_size"] = mlp_size

    return weights


# ---------------------------------------------------------------------------
# Vision + projector weight loading
# ---------------------------------------------------------------------------


def _load_vision_and_projector_weights(
    model_dir: str,
    config: ModelConfig,
) -> WeightDict:
    """Load vision encoder + MLP projector weights."""
    from pathlib import Path

    model_dir_path = Path(model_dir)
    readers = _open_safetensors(model_dir_path)

    weights = WeightDict()
    for reader in readers:
        for key in reader.keys():
            if (
                key.startswith("vision_tower.")
                or key.startswith("multi_modal_projector.")
                or key.startswith("visual.")
                or key.startswith("mlp1.")
            ):
                weights[key] = _load_tensor(readers, key)

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
    """Build one InternVL vision-language bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("internvl does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("internvl does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("internvl does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("internvl does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("internvl does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "vision_language_generation":
        raise ValueError("internvl supports only task=vision_language_generation")
    if request.quantization not in {None, "none"} or request.fp32_layers:
        raise NotImplementedError("InternVL supports only non-quantized uniform-precision builds")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    if str(config.model_type).lower() not in {"internvl_chat", "internvl3", "internvl"}:
        raise ValueError(f"InternVL does not support model_type={config.model_type!r}")
    precision = str(request.precision).lower()
    max_length = int(request.max_sequence_length or min(config.max_position_embeddings, 256))
    parallel = ParallelConfig(tp_size=int(request.tensor_parallel_size))
    parallel.validate()
    config.raw["_model_dir"] = str(model_dir)
    model = _InternVLModel()
    weights = model.load_weights(str(model_dir), config)
    writer.set_header(family="internvl", task=request.task, backend=request.backend)
    if parallel.enabled:
        for rank in range(parallel.tp_size):
            writer.add_bytes(
                f"engine.rank{rank}.plan",
                model.build_engine(
                    config,
                    weights,
                    max_length,
                    precision=precision,
                    quant_ctx=None,
                    verbose=request.verbose,
                    parallel_config=parallel.for_rank(rank),
                ),
            )
    else:
        config.raw["_decoder_engine_role"] = "prefill"
        prefill = model.build_engine(
            config,
            weights,
            max_length,
            precision=precision,
            quant_ctx=None,
            verbose=request.verbose,
            parallel_config=parallel,
        )
        config.raw["_decoder_engine_role"] = "decode"
        decode = model.build_engine(
            config,
            weights,
            max_length,
            precision=precision,
            quant_ctx=None,
            verbose=request.verbose,
            parallel_config=parallel,
        )
        config.raw.pop("_decoder_engine_role", None)
        writer.add_bytes("engine.plan", decode)
        writer.add_bytes("prefill.plan", prefill)
    vision = model.build_vision_engine(
        str(model_dir), config, weights, precision=precision, verbose=request.verbose
    )
    if vision is None:
        raise RuntimeError("InternVL vision build returned no engine")
    vl = model.get_vl_config(config) or {}
    runtime = {
        "tensor_parallel_size": parallel.tp_size,
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
