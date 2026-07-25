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

Checkpoint layout: the Lance HF repo is not a flat HF checkpoint (it nests
``Lance_3B/llm_config.json`` and a separate ``Qwen2.5-VL-ViT/`` dir). Run
``python -m tensorrt_model_connect.families.lance.prepare_model`` to stage a
directory this plugin can build:
``config.json`` (model_type=lance), ``model.safetensors``, the tokenizer files,
and the ViT at ``vision/model.safetensors``.
"""

from __future__ import annotations

from pathlib import Path

from .config import ModelConfig
from .checkpoint_mapper import (
    WeightDict,
    load_standard_weights,
    _open_safetensors,
    _load_tensor,
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


class LancePlugin:
    name = "lance"
    runtime_strategy = "lance_vision_language"
    # During VL prefill the decoder consumes ViT features as input_embed in
    # place of the image-pad token embeddings.
    embed_input = True

    def matches(self, model_type: str) -> bool:
        # Lance shares model_type "qwen2_5_vl" with real Qwen2.5-VL, so
        # The family-owned prepare_model tool stamps model_type="lance" to
        # route the checkpoint here instead of the qwen_vl plugin.
        return model_type.lower() == "lance"

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
            "image_token_id": config.raw.get("image_token_id", 151655),
            "fixed_image_size": fixed,
            "num_image_pad_tokens": num_merged,
            "vision_output_dim": config.hidden_size,
            "preprocessor_type": "merge_group_chw",
            "vl_prompt_template": (
                "<|im_start|>user\n"
                "<|vision_start|>{image_pads}<|vision_end|>\n"
                "{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "image_token_str": "<|image_pad|>",
        }


def _load_lance_vision_weights(model_dir: str) -> WeightDict:
    """Load the Qwen2.5-VL ViT weights, adding the ``visual.`` prefix the shared
    vision builder expects. The staged ViT lives at ``<model_dir>/vision/``."""
    vit_dir = Path(model_dir) / "vision"
    if not (vit_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"Lance ViT weights not found at {vit_dir}/model.safetensors. "
            "Run python -m tensorrt_model_connect.families.lance.prepare_model "
            "to stage the model."
        )
    readers = _open_safetensors(vit_dir)
    weights = WeightDict()
    for reader in readers:
        for key in reader.keys():
            weights[f"visual.{key}"] = _load_tensor([reader], key)
    return weights


plugin = LancePlugin()
