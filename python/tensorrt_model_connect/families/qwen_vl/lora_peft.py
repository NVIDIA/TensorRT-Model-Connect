# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-VL loading and validation for standard PEFT LoRA artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


def _enabled(config: Mapping[str, object], key: str) -> bool:
    value = config.get(key, False)
    return value is not None and bool(value)


def _nonempty(config: Mapping[str, object], key: str) -> bool:
    value = config.get(key)
    return value is not None and bool(value)


@dataclass(frozen=True)
class PeftLoraConfig:
    """Standard PEFT fields that do not depend on a model family."""

    rank: int
    alpha: float
    target_modules: tuple[str, ...]

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PeftLoraConfig":
        if str(raw.get("peft_type", "LORA")) != "LORA":
            raise ValueError("PEFT adapter peft_type must be LORA")
        if any(_enabled(raw, key) for key in ("use_dora", "use_rslora", "use_qalora")):
            raise NotImplementedError("DoRA, rsLoRA, and QALoRA adapters are not supported")
        if _enabled(raw, "fan_in_fan_out"):
            raise NotImplementedError("fan_in_fan_out LoRA weights are not supported")
        if str(raw.get("bias", "none")) != "none" or _enabled(raw, "lora_bias"):
            raise NotImplementedError("LoRA bias adaptation is not supported")
        if _nonempty(raw, "modules_to_save"):
            raise NotImplementedError("PEFT modules_to_save are not supported")
        if _nonempty(raw, "rank_pattern") or _nonempty(raw, "alpha_pattern"):
            raise NotImplementedError("Per-module LoRA rank/alpha patterns are not supported yet")

        rank = int(raw.get("r", 0) or 0)
        if rank <= 0:
            raise ValueError("PEFT adapter_config.json must define a positive rank 'r'")
        alpha = float(raw.get("lora_alpha", rank))

        raw_targets = raw.get("target_modules")
        if raw_targets is None:
            targets: tuple[str, ...] = ()
        elif isinstance(raw_targets, str):
            targets = (raw_targets,)
        elif isinstance(raw_targets, (list, tuple)):
            targets = tuple(str(item) for item in raw_targets)
        else:
            raise ValueError("PEFT target_modules must be a string or list of strings")
        if any(not target for target in targets):
            raise ValueError("PEFT target_modules must not contain empty names")
        targets = tuple(dict.fromkeys(targets))
        return cls(rank=rank, alpha=alpha, target_modules=targets)


@dataclass(frozen=True)
class PeftLoraArtifact:
    """A validated PEFT config plus its family-unmapped tensors."""

    config: PeftLoraConfig
    tensors: Mapping[str, np.ndarray]


def load_peft_lora_artifact(adapter_dir: str | Path) -> PeftLoraArtifact:
    """Load the standard PEFT JSON and safetensors filenames from a directory."""
    root = Path(adapter_dir)
    config_path = root / "adapter_config.json"
    weights_path = root / "adapter_model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter config: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter weights: {weights_path}")

    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load PEFT adapters") from exc

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("PEFT adapter_config.json must contain a JSON object")
    return PeftLoraArtifact(
        config=PeftLoraConfig.from_dict(raw_config),
        tensors=load_file(str(weights_path)),
    )
