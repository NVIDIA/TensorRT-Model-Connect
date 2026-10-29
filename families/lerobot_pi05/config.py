# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and validate a LeRobot Pi 0.5 checkpoint directory.

The supported boundary is deliberately narrow. Anything outside it fails here,
with the offending field named, instead of building a graph that silently
disagrees with the LeRobot reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


EMPTY_CAMERA_PREFIX = "observation.images.empty_camera_"
SUPPORTED_NORMALIZATION = ("IDENTITY", "MEAN_STD", "QUANTILES")

# PaliGemma Gemma-2B prefix and Gemma-300M action expert, as LeRobot defines them.
PREFIX_VARIANT = "gemma_2b"
EXPERT_VARIANT = "gemma_300m"
IMAGE_RESOLUTION = (224, 224)
PATCH_SIZE = 14
IMAGE_TOKENS = (IMAGE_RESOLUTION[0] // PATCH_SIZE) ** 2  # SigLIP tokens per camera

_REQUIRED_EXACT = {
    "type": "pi05",
    "n_obs_steps": 1,
    "paligemma_variant": PREFIX_VARIANT,
    "action_expert_variant": EXPERT_VARIANT,
    "image_resolution": list(IMAGE_RESOLUTION),
}


@dataclass(frozen=True)
class Normalization:
    """One feature's normalization. ``offset``/``scale`` are None when the checkpoint has no stats.

    LeRobot skips a feature whose statistics are absent, so ``None`` means the value passes
    through unchanged regardless of the configured mode. Otherwise, per dimension:

    - ``MEAN_STD``: ``offset=mean``, ``scale=std``; normalize ``(x - mean) / (std + eps)``,
      unnormalize ``x * std + mean``.
    - ``QUANTILES``: ``offset=q01``, ``scale=q99 - q01`` (``eps`` where that is exactly zero);
      normalize ``2 (x - q01) / scale - 1``, unnormalize ``(x + 1) scale / 2 + q01``.
    """

    mode: str
    eps: float
    offset: tuple[float, ...] | None = None
    scale: tuple[float, ...] | None = None

    @property
    def is_identity(self) -> bool:
        return self.offset is None or self.mode == "IDENTITY"


@dataclass(frozen=True)
class Pi05Config:
    cameras: tuple[str, ...]
    empty_cameras: int
    camera_height: int
    camera_width: int
    state_dim: int
    action_dim: int
    chunk_size: int
    n_action_steps: int
    max_state_dim: int
    max_action_dim: int
    num_inference_steps: int
    tokenizer_max_length: int
    tokenizer_name: str
    min_period: float
    max_period: float
    state_normalization: Normalization
    action_normalization: Normalization

    @property
    def camera_slots(self) -> int:
        return len(self.cameras) + self.empty_cameras


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot Pi 0.5 checkpoint is missing {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _check_boundary(raw: dict[str, Any]) -> None:
    mismatches = {
        key: (raw.get(key), wanted)
        for key, wanted in _REQUIRED_EXACT.items()
        if raw.get(key) != wanted
    }
    rtc = raw.get("rtc_config")
    if isinstance(rtc, dict) and rtc.get("enabled"):
        mismatches["rtc_config.enabled"] = (True, False)
    if raw.get("use_relative_actions"):
        mismatches["use_relative_actions"] = (True, False)
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in sorted(mismatches.items())
        )
        raise ValueError(f"Unsupported LeRobot Pi 0.5 policy contract: {details}")


def _features(raw: dict[str, Any]) -> tuple[list[tuple[str, list[int]]], int, int]:
    inputs = raw.get("input_features") or {}
    outputs = raw.get("output_features") or {}
    # Padded placeholder cameras are listed by some checkpoints and implied by
    # empty_cameras in others; they are never real inputs either way.
    cameras = [
        (name, list(spec["shape"]))
        for name, spec in inputs.items()
        if spec.get("type") == "VISUAL" and not name.startswith(EMPTY_CAMERA_PREFIX)
    ]
    state = [spec for spec in inputs.values() if spec.get("type") == "STATE"]
    action = [spec for spec in outputs.values() if spec.get("type") == "ACTION"]
    if not cameras:
        raise ValueError("LeRobot Pi 0.5 checkpoint declares no camera input")
    if len(state) != 1 or len(action) != 1:
        raise ValueError("LeRobot Pi 0.5 needs exactly one STATE input and one ACTION output")
    return cameras, int(state[0]["shape"][0]), int(action[0]["shape"][0])


def _normalization(
    model_dir: Path, processor: str, registry_name: str, feature_key: str, feature_type: str
) -> Normalization:
    document = _read_json(model_dir / processor)
    steps = [s for s in document.get("steps", []) if s.get("registry_name") == registry_name]
    if len(steps) != 1:
        raise ValueError(f"{processor} must contain exactly one {registry_name} step")
    step = steps[0]
    settings = step.get("config", {})
    mode = str(settings.get("norm_map", {}).get(feature_type, "IDENTITY")).upper()
    eps = float(settings.get("eps", 1e-8))
    state_file = step.get("state_file")
    if not state_file:
        # LeRobot skips normalization for a feature with no statistics at all.
        return Normalization(mode=mode, eps=eps)
    if mode not in SUPPORTED_NORMALIZATION:
        raise NotImplementedError(
            f"LeRobot Pi 0.5 normalization mode {mode} for {feature_type} is not supported "
            f"(supported: {', '.join(SUPPORTED_NORMALIZATION)})"
        )
    path = model_dir / state_file
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot Pi 0.5 checkpoint is missing {state_file}")
    names_for_mode = {"MEAN_STD": ("mean", "std"), "QUANTILES": ("q01", "q99")}
    with safe_open(str(path), framework="numpy") as reader:
        names = set(reader.keys())
        wanted = tuple(f"{feature_key}.{stat}" for stat in names_for_mode.get(mode, ()))
        if mode == "IDENTITY":
            return Normalization(mode=mode, eps=eps)
        if any(name not in names for name in wanted):
            raise KeyError(f"{state_file} has no {wanted[0]} / {wanted[1]} statistics")
        first, second = (np.asarray(reader.get_tensor(name), dtype=np.float32).reshape(-1) for name in wanted)
    if mode == "MEAN_STD":
        return Normalization(mode=mode, eps=eps, offset=tuple(map(float, first)), scale=tuple(map(float, second)))
    spread = (second - first).astype(np.float32)
    spread = np.where(spread == 0, np.float32(eps), spread)
    return Normalization(mode=mode, eps=eps, offset=tuple(map(float, first)), scale=tuple(map(float, spread)))


def _tokenizer_name(model_dir: Path) -> str:
    document = _read_json(model_dir / "policy_preprocessor.json")
    steps = [s for s in document.get("steps", []) if s.get("registry_name") == "tokenizer_processor"]
    if len(steps) != 1:
        raise ValueError("policy_preprocessor.json must contain exactly one tokenizer_processor step")
    config = steps[0].get("config", {})
    if config.get("padding_side", "right") != "right" or config.get("padding") != "max_length":
        raise NotImplementedError("LeRobot Pi 0.5 requires right-padded max_length tokenization")
    return str(config.get("tokenizer_name", ""))


def load_config(model_dir: str | Path) -> Pi05Config:
    model_dir = Path(model_dir)
    raw = _read_json(model_dir / "config.json")
    _check_boundary(raw)

    cameras, state_dim, action_dim = _features(raw)
    shapes = {tuple(shape) for _, shape in cameras}
    if len(shapes) != 1:
        raise NotImplementedError("LeRobot Pi 0.5 cameras must share one image shape")
    channels, height, width = next(iter(shapes))
    if channels != 3:
        raise NotImplementedError("LeRobot Pi 0.5 requires 3-channel camera images")
    if height != width:
        # The reference pads a non-square image to a square; that is not implemented.
        raise NotImplementedError("LeRobot Pi 0.5 requires square camera images")

    max_state_dim = int(raw["max_state_dim"])
    max_action_dim = int(raw["max_action_dim"])
    if state_dim > max_state_dim or action_dim > max_action_dim:
        raise ValueError("LeRobot Pi 0.5 state/action exceeds the padded model dimension")

    return Pi05Config(
        cameras=tuple(name for name, _ in cameras),
        empty_cameras=int(raw.get("empty_cameras", 0)),
        camera_height=int(height),
        camera_width=int(width),
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=int(raw["chunk_size"]),
        n_action_steps=int(raw.get("n_action_steps", raw["chunk_size"])),
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        num_inference_steps=int(raw["num_inference_steps"]),
        tokenizer_max_length=int(raw["tokenizer_max_length"]),
        tokenizer_name=_tokenizer_name(model_dir),
        min_period=float(raw["min_period"]),
        max_period=float(raw["max_period"]),
        state_normalization=_normalization(
            model_dir, "policy_preprocessor.json", "normalizer_processor", "observation.state", "STATE"
        ),
        action_normalization=_normalization(
            model_dir, "policy_postprocessor.json", "unnormalizer_processor", "action", "ACTION"
        ),
    )
