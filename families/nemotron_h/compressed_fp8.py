# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recognize the Lightning MoE static W8A8 checkpoint contract, not arbitrary compression."""

import re
from pathlib import Path


def compressed_fp8_exclusions(model_dir: Path, raw: dict) -> tuple[str, ...]:
    """Return literal unquantized modules; reject ambiguous metadata before building."""
    q = raw.get("quantization_config")
    if (raw.get("model_type") != "nemotron_h" or raw.get("architectures") != ["NemotronHForCausalLM"]
            or type(raw.get("n_routed_experts")) is not int or raw["n_routed_experts"] <= 0
            or not isinstance(q, dict) or q.get("quant_method") != "compressed-tensors"
            or q.get("format") != "float-quantized" or q.get("quantization_status") != "compressed"
            or "kv_cache_scheme" not in q or q["kv_cache_scheme"] is not None
            or q.get("sparsity_config") or q.get("transform_config")
            or any((model_dir / name).exists() for name in (
                "hf_quant_config.json", "quantize_config.json", "quant_config.json"))):
        raise ValueError("Unsupported or conflicting Nemotron-H compressed-tensors checkpoint")
    groups = q.get("config_groups")
    if not isinstance(groups, dict) or len(groups) != 1:
        raise ValueError("Nemotron-H compressed FP8 requires one static W8A8 linear group")
    group = next(iter(groups.values()))
    if (not isinstance(group, dict) or group.get("format") != "float-quantized"
            or group.get("targets") != ["Linear"] or group.get("output_activations") is not None):
        raise ValueError("Unsupported Nemotron-H compressed FP8 module policy")
    for role in ("weights", "input_activations"):
        policy = group.get(role)
        if (not isinstance(policy, dict) or policy.get("type") != "float" or policy.get("num_bits") != 8
                or policy.get("strategy") != "tensor" or policy.get("symmetric") is not True
                or policy.get("dynamic") is not False or policy.get("group_size") is not None
                or policy.get("block_structure") is not None):
            raise ValueError("Nemotron-H compressed FP8 requires symmetric static per-tensor weights and activations")
    excluded = q.get("ignore")
    if not isinstance(excluded, list) or not all(
        isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.]+", name) for name in excluded
    ):
        raise ValueError("Nemotron-H compressed FP8 requires literal module exclusions")
    return tuple(excluded)
