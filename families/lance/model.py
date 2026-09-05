# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lance family plugin — ByteDance ``bytedance-research/Lance`` unified model.

Scope (Stage 1): the **understanding** path only — ``x2t_image`` and
``x2t_video``. Lance's understanding sub-model is a Qwen2.5-VL ViT vision
encoder feeding a Lance text decoder, which maps onto the existing
``lance_vision_language`` runtime strategy.

Lance is a Mixture-of-Transformer-Experts model: every decoder layer carries a
second ``*_moe_gen`` parameter set, plus ``llm2vae`` / ``vae2llm`` /
``time_embedder`` / ``latent_pos_embed`` tensors. Those drive flow-matching
image/video **generation** and are intentionally NOT consumed here:
``load_standard_weights`` only reads the unsuffixed understanding-expert keys
(``self_attn.q_proj``, ``mlp.*``, ``input_layernorm`` …), so the generation
expert is dropped automatically. Generation/editing is a later stage that needs
a new runtime strategy and is out of scope for this plugin.

Architecture (confirmed against ``modeling/lance/qwen2_navit.py``): the
understanding decoder is GQA (16/2) with **QKV bias** (Qwen2 style) **and**
per-head **QK-norm** over ``head_dim`` (``qk_norm_und``) + SwiGLU + standard
RoPE; ViT is the standard Qwen2.5-VL encoder shipped with bare ``blocks.*`` /
``merger.*`` / ``patch_embed.*`` names (we re-add the ``visual.`` prefix the
shared vision builder expects). The shared decoder builder applies QKV-bias and
QK-norm conditionally when the weights are present.

Numerical validation: the TRT decoder matches an independent eager reference
exactly (per-layer and logits), and end-to-end ``trtmc run`` at **bf16** is
verified correct ("White car driving on the street." / "White"). Reduced
precision relies on the #184 builder fix (now in main): for embed bundles
``input_embed`` is bound as fp32 and cast inside the graph, and ``build_engine``
forwards ``precision`` so bf16/fp16 build true reduced-precision engines.

Checkpoint layout: this builder consumes a flat Lance understanding checkpoint
with ``config.json``, ``model.safetensors``, tokenizer files, and
``vision/model.safetensors``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pathlib import Path

import numpy as np
from safetensors import safe_open

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    load_standard_weights,
)

# Reuse the Qwen-VL vision encoder shape. The decoder builder is local so the
# Lance family does not depend on another family's text-builder package.
from .default_decoder import build_standard_decoder_engine
from .qwen_vl_vision_builder import build_qwen_vl_vision_engine

# Standard Qwen2.5-VL ViT input size; the runtime resizes images to this.
_DEFAULT_FIXED_IMAGE_SIZE = 448
# Lance LLM weights live under this prefix. The generation expert (``*_moe_gen``)
# and the VAE/time-embedder/latent-pos tensors are deliberately not requested.
_LLM_PREFIX = "language_model.model"
_LM_HEAD_KEY = "language_model.lm_head.weight"


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _LanceModel:
    # During VL prefill the decoder consumes ViT features as input_embed in
    # place of the image-pad token embeddings.
    embed_input = True

    def load_weights(self, model_dir: str, config: ModelConfig) -> WeightDict:
        # Reads only the understanding-expert weights; *_moe_gen and the
        # generation-only tensors are never requested and thus ignored.
        return load_standard_weights(
            model_dir,
            config,
            model_prefix=_LLM_PREFIX,
            lm_head_key=_LM_HEAD_KEY,
        )

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
        return build_standard_decoder_engine(
            config,
            weights,
            max_cache_length,
            precision=precision,
            verbose=verbose,
            quant_ctx=quant_ctx,
            embed_input=True,
            round_rope_inv_freq_to_bf16=(precision == "bf16"),
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
        vision_weights = _load_lance_vision_weights(model_dir)
        return build_qwen_vl_vision_engine(
            vision_config,
            vision_weights,
            fixed_image_size=_DEFAULT_FIXED_IMAGE_SIZE,
            verbose=verbose,
        )

    def get_vl_config(self, config: ModelConfig) -> dict | None:
        vision_config = config.raw.get("vision_config")
        if vision_config is None:
            return None

        patch_size = vision_config.get("patch_size", 14)
        merge_size = vision_config.get("spatial_merge_size", 2)
        fixed = _DEFAULT_FIXED_IMAGE_SIZE
        num_patches = (fixed // patch_size) ** 2
        num_merged = num_patches // (merge_size * merge_size)

        return {
            # Lance's pinned x2t_image reference intentionally routes image
            # features through Qwen2.5-VL's video placeholder.
            "image_token_id": config.raw.get("video_token_id", 151656),
            "fixed_image_size": fixed,
            "patch_size": patch_size,
            "merge_size": merge_size,
            "temporal_patch_size": 2,
            "num_image_pad_tokens": num_merged,
            "vision_output_dim": config.hidden_size,
            "preprocessor_type": "merge_group_chw",
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "interpolation": "bicubic",
            "vl_prompt_template": (
                "<|im_start|>system\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "<|vision_start|>{image_pads}<|vision_end|>"
                "{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "image_token_str": "<|video_pad|>",
        }


def _load_lance_vision_weights(model_dir: str) -> WeightDict:
    """Load the ViT directly from the official Lance repository layout."""
    vit_dir = Path(model_dir) / "Qwen2.5-VL-ViT"
    checkpoint = vit_dir / "vit.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Lance ViT weights not found at {checkpoint}")
    reader = safe_open(str(checkpoint), framework="numpy")
    weights = WeightDict()
    for key in reader.keys():
        weights[f"visual.{key}"] = np.asarray(reader.get_tensor(key), dtype=np.float32)
    return weights


def _tokenizer_runtime_contract(model_dir: Path) -> dict[str, object]:
    """Resolve this family's exact native-tokenizer framing."""

    from transformers import AutoConfig, AutoTokenizer

    config_path = model_dir / "llm_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Lance tokenizer config not found at {config_path}")
    tokenizer_config = AutoConfig.from_pretrained(
        str(config_path),
        trust_remote_code=True,
    )
    if tokenizer_config.model_type != "qwen2_5_vl":
        raise ValueError(
            "Lance tokenizer requires model_type='qwen2_5_vl' in Lance_3B/llm_config.json"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        config=tokenizer_config,
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
    """Build one Lance vision-language bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("lance does not support dynamic_kv_cache")

    if request.image_height is not None:
        raise NotImplementedError("lance does not support image_height")

    if request.image_width is not None:
        raise NotImplementedError("lance does not support image_width")

    if request.video_num_frames is not None:
        raise NotImplementedError("lance does not support video_num_frames")

    if request.max_batch_size != 1:
        raise NotImplementedError("lance does not support max_batch_size")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "vision_language_generation":
        raise ValueError("lance supports only task=vision_language_generation")
    if (
        request.tensor_parallel_size != 1
        or request.quantization not in {None, "none"}
        or request.fp32_layers
    ):
        raise NotImplementedError("Lance supports only single-device non-quantized builds")
    model_dir = Path(request.model_dir)
    llm_dir = model_dir / "Lance_3B"
    llm_config = llm_dir / "llm_config.json"
    vision_checkpoint = model_dir / "Qwen2.5-VL-ViT" / "vit.safetensors"
    if not llm_config.is_file() or not (llm_dir / "model.safetensors").is_file():
        raise FileNotFoundError("Lance checkpoint is missing Lance_3B model files")
    if not vision_checkpoint.is_file():
        raise FileNotFoundError("Lance checkpoint is missing Qwen2.5-VL-ViT/vit.safetensors")
    config = ModelConfig.from_json(llm_config.read_text(encoding="utf-8"))
    config.model_type = "lance"
    config.raw["model_type"] = "lance"
    precision = str(request.precision).lower()
    max_length = int(request.max_sequence_length or min(config.max_position_embeddings, 256))
    config.raw["_model_dir"] = str(llm_dir)
    model = _LanceModel()
    weights = model.load_weights(str(llm_dir), config)
    config.raw["_decoder_engine_role"] = "prefill"
    prefill = model.build_engine(
        config,
        weights,
        max_length,
        precision=precision,
        quant_ctx=None,
        verbose=request.verbose,
        parallel_config=None,
    )
    config.raw["_decoder_engine_role"] = "decode"
    decode = model.build_engine(
        config,
        weights,
        max_length,
        precision=precision,
        quant_ctx=None,
        verbose=request.verbose,
        parallel_config=None,
    )
    config.raw.pop("_decoder_engine_role", None)
    vision = model.build_vision_engine(
        str(model_dir), config, weights, precision=precision, verbose=request.verbose
    )
    if vision is None:
        raise RuntimeError("Lance vision build returned no engine")
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
    writer.set_header(family="lance", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", decode)
    writer.add_bytes("prefill.plan", prefill)
    writer.add_bytes("vision.plan", vision)
    runtime.update(_tokenizer_runtime_contract(llm_dir))
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
        path = llm_dir / filename
        if path.is_file():
            writer.add_bytes(filename, path.read_bytes())
