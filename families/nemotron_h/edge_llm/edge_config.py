# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native hybrid source topology gates; Edge still constructs every graph."""

from __future__ import annotations

import math

_PATTERN = {"M": "mamba", "-": "mlp", "*": "attention", "E": "moe"}


def layers(raw: dict) -> list[str] | None:
    """Require complete consistent explicit hybrid topology, never filter unknowns."""
    forms = []
    for name in ("layers_block_type", "layer_types"):
        if name in raw:
            value = raw[name]
            if not isinstance(value, list) or any(x not in _PATTERN.values() for x in value):
                return None
            forms.append(value)
    if raw.get("hybrid_override_pattern"):
        pattern = raw["hybrid_override_pattern"]
        if not isinstance(pattern, str) or any(x not in _PATTERN for x in pattern):
            return None
        forms.append([_PATTERN[x] for x in pattern])
    if not forms or not forms[0] or any(x != forms[0] for x in forms):
        return None
    return forms[0] if len(forms[0]) == raw.get("num_hidden_layers") else None


def topology_matches(raw: dict, precision: str) -> bool:
    """Admit only pinned dense/SSD or non-gated NVFP4 MoE semantics."""
    kinds = layers(raw)
    if (raw.get("model_type") != "nemotron_h"
            or raw.get("architectures") != ["NemotronHForCausalLM"] or kinds is None
            or "mamba" not in kinds
            or any(key in raw for key in ("text_config", "vision_config", "audio_config", "thinker_config",
                                          "eagle_config", "dflash_config", "jetspec_config", "dspark_config"))
            or raw.get("mamba_hidden_act", "silu") != "silu"
            or raw.get("mlp_hidden_act", "relu2") != "relu2"
            or raw.get("hidden_act", "relu2") != "relu2"
            or raw.get("mamba_ssm_cache_dtype", "float32") != "float32"
            or any(raw.get(key, False) for key in ("attention_bias", "mamba_proj_bias", "mlp_bias"))
            or raw.get("rope_scaling") is not None or raw.get("hybrid_uses_rope", False)
            or raw.get("norm_before_gate", False) or raw.get("conv_kernel", 4) != 4):
        return False
    if type(raw.get("bos_token_id", -1)) is not int or raw.get("bos_token_id", -1) < -1:
        return False
    keys = ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "head_dim", "mamba_num_heads", "mamba_head_dim", "n_groups", "ssm_state_size",
            "max_position_embeddings", "vocab_size")
    values = [raw.get(key) for key in keys]
    if any(type(value) is not int or value <= 0 for value in values):
        return False
    hidden, _, heads, kv, head, mheads, mhead, groups, state, _, _ = values
    if (heads % kv or head % 8 or mheads % groups or state not in {64, 128}
            or mhead not in {64, 80, 128} or mhead == 80 and state != 128
            or raw.get("conv_dim", mheads * mhead + 2 * groups * state)
            != mheads * mhead + 2 * groups * state):
        return False
    if "mlp" in kinds and (type(raw.get("intermediate_size")) is not int or raw["intermediate_size"] <= 0):
        return False
    if "moe" not in kinds:
        return not raw.get("n_routed_experts", 0)
    names = ("n_routed_experts", "num_experts_per_tok", "n_group", "topk_group",
             "moe_intermediate_size", "moe_shared_expert_intermediate_size")
    if any(type(raw.get(key)) is not int or raw[key] <= 0 for key in names):
        return False
    experts, topk, router_groups, selected_groups, intermediate, shared = [raw[key] for key in names]
    scale = raw.get("routed_scaling_factor", 1.0)
    latent = raw.get("moe_latent_size", hidden)
    latent = hidden if latent is None else latent
    return (precision == "nvfp4" and experts <= 512 and topk <= experts
            and experts % router_groups == 0 and selected_groups <= router_groups
            and topk <= selected_groups * (experts // router_groups)
            and raw.get("n_shared_experts", 1) == 1 and intermediate % 16 == shared % 16 == 0
            and type(latent) is int and latent > 0 and latent % 16 == 0
            and isinstance(scale, (int, float)) and not isinstance(scale, bool)
            and math.isfinite(scale) and scale > 0)
