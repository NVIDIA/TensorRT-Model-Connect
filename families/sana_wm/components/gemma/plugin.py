# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma family plugin — applies +1.0 to RMSNorm gamma and sqrt(hidden) embed scale."""

from __future__ import annotations

import math

from .checkpoint_mapper import WeightDict, load_standard_weights
from .config import ModelConfig


class _GemmaWeights:

    def matches(self, model_type: str) -> bool:
        return model_type.lower().startswith("gemma")

    def load_weights(
        self, model_dir: str, config: ModelConfig,
        *, precision: str = "fp32",
    ) -> WeightDict:
        load_kwargs: dict[str, str] = {}
        if config.model_type == "gemma3" and isinstance(
            config.raw.get("text_config"), dict
        ):
            load_kwargs = {
                "model_prefix": "language_model.model",
                "lm_head_key": "language_model.lm_head.weight",
            }
        weights = load_standard_weights(
            model_dir,
            config,
            precision=precision,
            **load_kwargs,
        )

        # Fix 1: Gemma uses (1 + gamma) * normalized instead of gamma * normalized.
        for layer_idx in range(config.num_hidden_layers):
            prefix = f"layer.{layer_idx}"
            weights[f"{prefix}.input_norm"] = weights[f"{prefix}.input_norm"] + 1.0
            weights[f"{prefix}.post_attn_norm"] = weights[f"{prefix}.post_attn_norm"] + 1.0
            if f"{prefix}.q_norm" in weights:
                weights[f"{prefix}.q_norm"] = weights[f"{prefix}.q_norm"] + 1.0
            if f"{prefix}.k_norm" in weights:
                weights[f"{prefix}.k_norm"] = weights[f"{prefix}.k_norm"] + 1.0
            if f"{prefix}.pre_ff_norm" in weights:
                weights[f"{prefix}.pre_ff_norm"] = weights[f"{prefix}.pre_ff_norm"] + 1.0
            if f"{prefix}.post_ff_norm" in weights:
                weights[f"{prefix}.post_ff_norm"] = weights[f"{prefix}.post_ff_norm"] + 1.0
        weights["final_norm"] = weights["final_norm"] + 1.0

        # Gemma scales gathered embeddings in the model dtype at runtime.
        weights["_embedding_scale"] = math.sqrt(config.hidden_size)

        return weights

plugin = _GemmaWeights()
