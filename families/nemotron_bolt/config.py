# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize the prepared Puzzle target and its LiLiCorr companion, not Nano Omni."""

import json
from pathlib import Path

from .support import is_bolt


def validate_inputs(target: Path, draft: Path | None) -> dict:
    cfg = json.loads((target / "config.json").read_text())
    if not is_bolt(cfg):
        raise ValueError("Expected the documented multimodal Bolt-74B Puzzle checkpoint")
    groups = cfg.get("quantization_config", {}).get("quantized_layers", {})
    prefix = "language_model.backbone.layers.1.mixer."
    for name, fmt in [
        ("language_model.lm_head", "Q8_0"),
        (prefix + "fc1_latent_proj", "Q8_0"),
        (prefix + "experts.0.up_proj", "IQ2_XXS"),
        ("language_model.backbone.layers.0.mixer.in_proj", "W4A16_NVFP4"),
    ]:
        if groups.get(name, {}).get("quant_algo") != fmt:
            raise ValueError(f"Bolt requires source format {fmt} for {name}")
    if draft is not None:
        companion = json.loads((draft / "config.json").read_text())
        lattice = companion.get("dflash_config", {})
        if not (
            companion.get("architectures") == ["LiLiCorrDraftModel"]
            and companion.get("block_size") == 8
            and companion.get("hidden_size") == cfg["llm_config"]["hidden_size"]
            and companion.get("vocab_size") == cfg["llm_config"]["vocab_size"]
            and lattice.get("projector_type") == "lilicorr"
            and lattice.get("lilicorr_enabled") is True
            and lattice.get("lilicorr_candidate_topk") == 8
            and companion.get("num_hidden_layers") == 5
            and companion.get("num_attention_heads") == 32
            and companion.get("num_key_value_heads") == 2
            and companion.get("head_dim") == 128
            and companion.get("rope_theta") == 10000
            and companion.get("rope_scaling") is None
            and lattice.get("swa_window_size") == 1024
            and lattice.get("causal") is False
            and lattice.get("mask_token_id") == 990
            and lattice.get("conv_kernel_size") == 2
            and lattice.get("conv_group_size") == 16
            and lattice.get("target_layer_ids") == [1, 18, 35, 51, 68, 85]
        ):
            raise ValueError("Expected the published eight-slot Bolt LiLiCorr draft")
    return cfg
