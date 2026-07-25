# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build-time contract for dynamically bound Qwen-VL LoRA weights.

The TensorRT graph consumes one fixed-shape A/B pair for every selected
projection.  Adapter loading is deliberately outside the graph builder: the
runtime owns the device buffers and binds their addresses to these inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass




PEFT_TO_WEIGHT_NAME = {
    "q_proj": "w_q",
    "k_proj": "w_k",
    "v_proj": "w_v",
    "o_proj": "w_o",
    "gate_proj": "w_gate",
    "up_proj": "w_up",
    "down_proj": "w_down",
}

DEFAULT_TARGET_MODULES = tuple(PEFT_TO_WEIGHT_NAME)
_MAX_SUPPORTED_RANK = 256
_PEFT_WEIGHT_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\."
    r"(?:self_attn|mlp)\."
    r"(?P<module>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
    r"lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$"
)


def _parse_target_modules(raw: object) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_TARGET_MODULES
    if not isinstance(raw, str):
        raise ValueError("qwen_vl_lora.target_modules must be a comma-separated string")

    modules: list[str] = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in PEFT_TO_WEIGHT_NAME:
            supported = ", ".join(DEFAULT_TARGET_MODULES)
            raise ValueError(
                f"Unsupported Qwen-VL LoRA target module {name!r}; supported: {supported}")
        if name not in modules:
            modules.append(name)
    if not modules:
        raise ValueError("qwen_vl_lora.target_modules must select at least one module")
    return tuple(modules)


@dataclass(frozen=True)
class DynamicLoraConfig:
    """Validated engine-build settings for dynamic LoRA binding."""

    enabled: bool = False
    max_rank: int = 0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES

    @classmethod
    def from_model_config(cls, model_config) -> "DynamicLoraConfig":
        family_options = model_config.raw.get("_family_build_options", {})
        if not isinstance(family_options, dict):
            return cls()
        raw = family_options.get("qwen_vl_lora", {})
        if not isinstance(raw, dict):
            raise ValueError("qwen_vl_lora build options must be an object")

        enabled = bool(raw.get("enabled", False))
        max_rank = int(raw.get("max_rank", 0) or 0)
        target_modules = _parse_target_modules(raw.get("target_modules"))
        if not enabled:
            return cls(target_modules=target_modules)
        if max_rank <= 0 or max_rank > _MAX_SUPPORTED_RANK:
            raise ValueError(
                "qwen_vl_lora.max_rank must be between 1 and "
                f"{_MAX_SUPPORTED_RANK} when dynamic LoRA is enabled")
        return cls(enabled=True, max_rank=max_rank, target_modules=target_modules)

    @property
    def canonical_targets(self) -> frozenset[str]:
        return frozenset(PEFT_TO_WEIGHT_NAME[name] for name in self.target_modules)

    def targets_weight(self, weight_name: str) -> bool:
        return self.enabled and weight_name.rsplit(".", 1)[-1] in self.canonical_targets

    def input_names(self, weight_name: str) -> tuple[str, str]:
        """Return stable TensorRT input names for a canonical projection name."""
        stem = weight_name.replace(".", "_")
        return f"lora_a_{stem}", f"lora_b_{stem}"

    def bundle_config(self) -> dict[str, object]:
        return {
            "lora_dynamic_binding": self.enabled,
            "lora_max_rank": self.max_rank,
            "lora_target_modules": list(self.target_modules) if self.enabled else [],
            "lora_scale_in_b": self.enabled,
        }
