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
from pathlib import Path

import numpy as np

from .lora_peft import (
    PeftLoraConfig,
    load_peft_lora_artifact,
)


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


@dataclass(frozen=True)
class PreparedLoraAdapter:
    """PEFT adapter tensors normalized for the TensorRT binding contract."""

    adapter_id: str
    tensors: dict[str, np.ndarray]
    rank: int
    scale: float


def prepare_peft_adapter(
    adapter_id: str,
    adapter_config: dict,
    peft_tensors: dict[str, np.ndarray],
    *,
    max_rank: int,
    dtype: np.dtype = np.float16,
) -> PreparedLoraAdapter:
    """Convert PEFT A/B tensors into padded, scale-folded runtime inputs."""
    if not adapter_id:
        raise ValueError("adapter_id must not be empty")
    peft_config = PeftLoraConfig.from_dict(adapter_config)
    rank = peft_config.rank
    scale = peft_config.scale
    raw_targets = peft_config.target_modules or DEFAULT_TARGET_MODULES
    targets = _parse_target_modules(",".join(raw_targets))
    if max_rank <= 0 or rank > max_rank:
        raise ValueError(
            f"Adapter rank {rank} exceeds engine max_rank {max_rank}")

    pairs: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for name, value in peft_tensors.items():
        match = _PEFT_WEIGHT_RE.search(name)
        if match is None:
            continue
        module = match.group("module")
        if module not in targets:
            continue
        key = (int(match.group("layer")), module)
        pairs.setdefault(key, {})[match.group("side")] = np.asarray(value)

    if not pairs:
        raise ValueError("No supported Qwen-VL LoRA A/B tensors were found")

    prepared: dict[str, np.ndarray] = {}
    target_dtype = np.dtype(dtype)
    for (layer, module), sides in sorted(pairs.items()):
        if set(sides) != {"A", "B"}:
            missing = "B" if "A" in sides else "A"
            raise ValueError(f"Missing LoRA {missing} tensor for layer {layer} {module}")
        a = sides["A"]
        b = sides["B"]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"LoRA tensors for layer {layer} {module} must be rank-2")
        if a.shape[0] != rank or b.shape[1] != rank or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for layer {layer} {module}: "
                f"A={a.shape}, B={b.shape}, config rank={rank}")

        # PEFT: A=[rank,in], B=[out,rank]. TensorRT: A=[in,max_rank],
        # B=[max_rank,out]. Unused rank columns/rows remain zero.
        runtime_a = np.zeros((a.shape[1], max_rank), dtype=target_dtype)
        runtime_b = np.zeros((max_rank, b.shape[0]), dtype=target_dtype)
        runtime_a[:, :rank] = a.T.astype(target_dtype, copy=False)
        runtime_b[:rank, :] = (b.T * scale).astype(target_dtype, copy=False)

        canonical = f"layer.{layer}.{PEFT_TO_WEIGHT_NAME[module]}"
        names = DynamicLoraConfig(enabled=True, max_rank=max_rank).input_names(canonical)
        prepared[names[0]] = np.ascontiguousarray(runtime_a)
        prepared[names[1]] = np.ascontiguousarray(runtime_b)

    return PreparedLoraAdapter(
        adapter_id=adapter_id,
        tensors=prepared,
        rank=rank,
        scale=scale,
    )


def load_peft_adapter(
    adapter_id: str,
    adapter_dir: str | Path,
    *,
    max_rank: int,
    dtype: np.dtype = np.float16,
) -> PreparedLoraAdapter:
    """Load and normalize a standard PEFT safetensors adapter directory."""
    artifact = load_peft_lora_artifact(adapter_dir)
    return prepare_peft_adapter(
        adapter_id,
        {
            "peft_type": "LORA",
            "r": artifact.config.rank,
            "lora_alpha": artifact.config.alpha,
            "target_modules": list(artifact.config.target_modules),
        },
        dict(artifact.tensors),
        max_rank=max_rank,
        dtype=dtype,
    )
