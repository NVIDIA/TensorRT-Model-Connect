# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact family-owned source policy admission; never convert or calibrate weights."""

from __future__ import annotations

import json
from pathlib import Path


def descriptor(value: dict) -> tuple | None:
    """Normalize known ModelOpt policy fields, rejecting unknown packed layouts."""
    if not isinstance(value, dict):
        return None
    q = value.get("quantization", value)
    if not isinstance(q, dict):
        return None
    if q.get("quant_method", "modelopt").lower() not in {"modelopt", ""}:
        return None
    algorithm = str(q.get("quant_algo", "")).upper()
    if algorithm == "MIXED_PRECISION":
        layers = q.get("quantized_layers")
        if not isinstance(layers, dict) or not layers:
            return None
        algorithms = set()
        for name, policy in layers.items():
            if not isinstance(name, str) or not name or not isinstance(policy, dict):
                return None
            algo = policy.get("quant_algo")
            if algo not in {"FP8", "NVFP4", "W4A16_NVFP4"}:
                return None
            group = policy.get("group_size", 1 if algo == "FP8" else 16)
            if type(group) is not int or group not in ({1, 16} if algo == "FP8" else {16}):
                return None
            if set(policy) - {"quant_algo", "group_size"}:
                return None
            algorithms.add(algo)
        if algorithms not in ({"FP8", "NVFP4"}, {"FP8", "W4A16_NVFP4"}):
            return None
        kv = q.get("kv_cache_quant_algo")
        scheme = q.get("kv_cache_scheme")
        if scheme is not None:
            if scheme != {"dynamic": False, "num_bits": 8, "type": "float"}:
                return None
            if kv not in {None, "FP8"}:
                return None
            kv = "FP8"
        if kv != "FP8":
            return None
        # The pinned direct builder consumes the sidecar per-layer map. Keep
        # its exact algorithms/scales; do not reinterpret ModelOpt config_groups.
        return "nvfp4", kv, json.dumps(layers, sort_keys=True), "mixed"
    precision = {"FP8": "fp8", "NVFP4": "nvfp4", "W4A16_NVFP4": "nvfp4"}.get(algorithm)
    if precision is None:
        return None
    group = q.get("group_size", 1 if precision == "fp8" else 16)
    # FP8 projections use scalar scaling, not grouped packing. ModelOpt emits
    # either1 (parser default) or16 (export metadata); neither changes FP8 layout.
    if type(group) is not int or group not in ({1, 16} if precision == "fp8" else {16}):
        return None
    if any(key in q for key in ("config_groups", "quantized_layers", "kv_cache_scheme")):
        return None
    if any(key not in {"quant_algo", "group_size", "kv_cache_quant_algo", "exclude_modules",
                       "ignore", "quant_method"} for key in q):
        return None
    excluded = q.get("exclude_modules", q.get("ignore", []))
    if (not isinstance(excluded, list) or any(not isinstance(x, str) for x in excluded)
            or ("ignore" in q and "exclude_modules" in q and set(q["ignore"]) != set(excluded))):
        return None
    kv = q.get("kv_cache_quant_algo")
    if kv not in {None, "FP8"}:
        return None
    return precision, kv, tuple(sorted(set(excluded)))


def source_quantization(request, raw: dict) -> str | None:
    """Preserve consistent original/FP8/NVFP4 source, or return native admission."""
    try:
        policies = []
        embedded = raw.get("quantization_config")
        if embedded is not None:
            policies.append(descriptor(embedded))
        for name in ("hf_quant_config.json", "quantize_config.json", "quant_config.json"):
            path = Path(request.model_dir) / name
            if path.exists():
                value = json.loads(path.read_text(encoding="utf-8"))
                policies.append(descriptor(value))
        if policies:
            if any(value is None or value != policies[0] for value in policies):
                return None
            # Upstream only consumes hf_quant_config or embedded metadata.
            if embedded is None and not (Path(request.model_dir) / "hf_quant_config.json").is_file():
                return None
            if policies[0][-1] == "mixed" and (
                embedded is None or not (Path(request.model_dir) / "hf_quant_config.json").is_file()
            ):
                return None
            precision = policies[0][0]
        else:
            precision = "fp16"
        requested = request.quantization
        if requested is not None and requested != ("none" if precision == "fp16" else precision):
            return None
        return precision
    except (OSError, ValueError, TypeError, AttributeError):
        return None
