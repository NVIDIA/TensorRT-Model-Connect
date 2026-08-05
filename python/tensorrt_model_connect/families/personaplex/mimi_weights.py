# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint-owned Mimi weight loading for PersonaPlex."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np


_PERSONAPLEX_MIMI_FILENAME = "tokenizer-e351c8d8-checkpoint125.safetensors"


def _translate_personaplex_mimi_weights(
    source: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map the Moshi-native PersonaPlex Mimi checkpoint to TRT Mimi names."""
    mapped: dict[str, np.ndarray] = {}

    for key, value in source.items():
        target = None
        if key.startswith("encoder.model."):
            target = key.replace("encoder.model.", "encoder.layers.", 1)
            target = target.replace(".conv.conv.", ".conv.")
        elif key.startswith("decoder.model."):
            target = key.replace("decoder.model.", "decoder.layers.", 1)
            target = target.replace(".conv.conv.", ".conv.")
            target = target.replace(".convtr.convtr.", ".conv.")
        elif key == "downsample.conv.conv.conv.weight":
            target = "downsample.conv.weight"
        elif key == "upsample.convtr.convtr.convtr.weight":
            target = "upsample.conv.weight"
        elif key.startswith("quantizer.rvq_first."):
            target = key.replace(
                "quantizer.rvq_first.",
                "quantizer.semantic_residual_vector_quantizer.", 1)
        elif key.startswith("quantizer.rvq_rest."):
            target = key.replace(
                "quantizer.rvq_rest.",
                "quantizer.acoustic_residual_vector_quantizer.", 1)

        if target is not None:
            target = target.replace(".vq.layers.", ".layers.")
            target = target.replace(
                "._codebook.embedding_sum", ".codebook.embed_sum")
            target = target.replace(
                "._codebook.cluster_usage", ".codebook.cluster_usage")
            target = target.replace(
                "._codebook._initialized", ".codebook.initialized")
            mapped[target] = value

    layer_pattern = re.compile(
        r"^(encoder|decoder)_transformer\.transformer\.layers\.(\d+)\.")
    layer_prefixes = {
        match.group(0)[:-1]
        for key in source
        if (match := layer_pattern.match(key)) is not None
    }
    for source_prefix in sorted(layer_prefixes):
        target_prefix = source_prefix.replace(
            ".transformer.layers.", ".layers.")
        direct_suffixes = {
            "norm1.weight": "input_layernorm.weight",
            "norm1.bias": "input_layernorm.bias",
            "norm2.weight": "post_attention_layernorm.weight",
            "norm2.bias": "post_attention_layernorm.bias",
            "linear1.weight": "mlp.fc1.weight",
            "linear2.weight": "mlp.fc2.weight",
            "layer_scale_1.scale": "self_attn_layer_scale.scale",
            "layer_scale_2.scale": "mlp_layer_scale.scale",
            "self_attn.out_proj.weight": "self_attn.o_proj.weight",
        }
        for source_suffix, target_suffix in direct_suffixes.items():
            mapped[f"{target_prefix}.{target_suffix}"] = source[
                f"{source_prefix}.{source_suffix}"]

        fused_qkv = source[f"{source_prefix}.self_attn.in_proj_weight"]
        q_proj, k_proj, v_proj = np.split(fused_qkv, 3, axis=0)
        mapped[f"{target_prefix}.self_attn.q_proj.weight"] = q_proj
        mapped[f"{target_prefix}.self_attn.k_proj.weight"] = k_proj
        mapped[f"{target_prefix}.self_attn.v_proj.weight"] = v_proj

    return mapped


def _load_mimi_weights(model_dir: str | Path | None = None):
    """Load the checkpoint-owned Mimi codec.

    Returns a dict mapping weight name -> numpy array.
    Codebook embeddings are computed as embed_sum / cluster_usage.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    import json

    if model_dir is not None:
        candidate = Path(model_dir) / _PERSONAPLEX_MIMI_FILENAME
        if not candidate.is_file():
            raise FileNotFoundError(
                "PersonaPlex requires its checkpoint-owned Mimi codec: "
                f"{candidate}"
            )
        sf_path = str(candidate)
    else:
        sf_path = hf_hub_download("kyutai/mimi", "model.safetensors")

    cfg_path = hf_hub_download("kyutai/mimi", "config.json")
    with open(cfg_path) as f:
        mimi_cfg = json.load(f)

    mimi_weights = {}
    with safe_open(sf_path, framework="pt") as sf:
        for key in sf.keys():
            mimi_weights[key] = sf.get_tensor(key).numpy().astype(np.float32)
    if Path(sf_path).name == _PERSONAPLEX_MIMI_FILENAME:
        mimi_weights = _translate_personaplex_mimi_weights(mimi_weights)

    # Compute actual codebook embeddings from embed_sum / cluster_usage
    for prefix in ["quantizer.semantic_residual_vector_quantizer",
                    "quantizer.acoustic_residual_vector_quantizer"]:
        i = 0
        while f"{prefix}.layers.{i}.codebook.embed_sum" in mimi_weights:
            embed_sum = mimi_weights[f"{prefix}.layers.{i}.codebook.embed_sum"]
            usage = mimi_weights[f"{prefix}.layers.{i}.codebook.cluster_usage"]
            codebook = embed_sum / np.maximum(usage[:, None], 1e-8)
            mimi_weights[f"{prefix}.layers.{i}.codebook.embedding"] = codebook
            i += 1

    return mimi_weights, mimi_cfg
