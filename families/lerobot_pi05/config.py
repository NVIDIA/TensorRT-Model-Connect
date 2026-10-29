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
STATE_KEY = "observation.state"
ACTION_KEY = "action"
# LeRobot's statistics per normalization mode: (offset, upper or scale) entry names.
_STATISTICS = {
    "MEAN_STD": ("mean", "std"),
    "MIN_MAX": ("min", "max"),
    "QUANTILES": ("q01", "q99"),
    "QUANTILE10": ("q10", "q90"),
}
SUPPORTED_NORMALIZATION = ("IDENTITY", *_STATISTICS)
# Processor steps whose effect this family reproduces, or that are no-ops for inference.
_PREPROCESSOR_STEPS = {
    "rename_observations_processor",  # inputs already use the checkpoint's own keys
    "to_batch_processor",
    "relative_actions_processor",  # accepted only when disabled
    "normalizer_processor",
    "pi05_prepare_state_tokenizer_processor_step",
    "tokenizer_processor",
    "device_processor",
}
_POSTPROCESSOR_STEPS = {
    "unnormalizer_processor",
    "absolute_actions_processor",  # accepted only when disabled
    "device_processor",
}
_DISABLED_ONLY = {"relative_actions_processor", "absolute_actions_processor"}

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
    """One feature's normalization, as LeRobot's NormalizerProcessorStep applies it.

    ``offset``/``scale`` are None when LeRobot would pass the value through: identity mode,
    a feature the processor does not normalize, or no statistics for it. Otherwise, per
    dimension and in float32 with LeRobot's operation order:

    - ``MEAN_STD``: ``offset=mean``, ``scale=std``; ``(x - mean) / (std + eps)`` and
      ``x * std + mean``.
    - ``QUANTILES`` / ``QUANTILE10`` / ``MIN_MAX``: ``offset`` is the lower statistic (q01, q10
      or min) and ``scale`` the spread to the upper one, ``eps`` where it is exactly zero;
      ``2 (x - offset) / scale - 1``, and back with ``(x + 1) scale / 2 + offset`` (quantiles)
      or ``(x + 1) / 2 scale + offset`` (min/max).
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
    # Dataset min/max of each action dimension, when the checkpoint carries them.
    action_range: tuple[tuple[float, ...], tuple[float, ...]] | None = None

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
    state = [(name, spec) for name, spec in inputs.items() if spec.get("type") == "STATE"]
    action = [(name, spec) for name, spec in outputs.items() if spec.get("type") == "ACTION"]
    if not cameras:
        raise ValueError("LeRobot Pi 0.5 checkpoint declares no camera input")
    # LeRobot's pi05 prompt step reads the state from this exact key.
    if [name for name, _ in state] != [STATE_KEY] or [name for name, _ in action] != [ACTION_KEY]:
        raise ValueError(f"LeRobot Pi 0.5 needs exactly one {STATE_KEY} input and one {ACTION_KEY} output")
    state_shape, action_shape = list(state[0][1]["shape"]), list(action[0][1]["shape"])
    if len(state_shape) != 1 or len(action_shape) != 1:
        raise ValueError("LeRobot Pi 0.5 state and action must be vectors")
    return cameras, int(state_shape[0]), int(action_shape[0])


def _steps(model_dir: Path, processor: str, allowed: set[str]) -> list[dict[str, Any]]:
    """The processor's steps, after refusing any whose effect this family does not reproduce."""
    steps = _read_json(model_dir / processor).get("steps", [])
    for step in steps:
        name = str(step.get("registry_name", ""))
        known = name in allowed or name.startswith("to_batch_processor")
        if not known:
            raise NotImplementedError(f"LeRobot Pi 0.5 does not support the {name!r} step in {processor}")
        if name in _DISABLED_ONLY and step.get("config", {}).get("enabled", False):
            raise NotImplementedError(f"LeRobot Pi 0.5 does not support an enabled {name!r} step")
    return steps


def _one_step(steps: list[dict[str, Any]], processor: str, name: str) -> tuple[int, dict[str, Any]]:
    found = [(index, step) for index, step in enumerate(steps) if step.get("registry_name") == name]
    if len(found) != 1:
        raise ValueError(f"{processor} must contain exactly one {name} step")
    return found[0]


def _statistics(model_dir: Path, state_file: str | None, feature_key: str) -> dict[str, np.ndarray]:
    """``{stat: values}`` for one feature, empty when the checkpoint has none for it."""
    if not state_file:
        return {}
    path = model_dir / state_file
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot Pi 0.5 checkpoint is missing {state_file}")
    prefix = feature_key + "."
    with safe_open(str(path), framework="numpy") as reader:
        return {
            name[len(prefix):]: np.asarray(reader.get_tensor(name), dtype=np.float32).reshape(-1)
            for name in reader.keys()
            if name.startswith(prefix)
        }


def _normalization(model_dir: Path, step: dict[str, Any], feature_key: str, observation: bool) -> Normalization:
    """Mirror ``_NormalizationMixin`` for one feature of one (un)normalizer step."""
    settings = step.get("config", {})
    norm_map = {str(k).upper(): str(v).upper() for k, v in settings.get("norm_map", {}).items()}
    eps = float(settings.get("eps", 1e-8))
    if observation:
        # Observations are normalized only when the step lists them, and only those kept by
        # normalize_observation_keys; the listed feature type selects the mode.
        feature = settings.get("features", {}).get(feature_key)
        selected = settings.get("normalize_observation_keys")
        if feature is None or feature.get("type") == "ACTION" or (
            selected is not None and feature_key not in selected
        ):
            return Normalization(mode="IDENTITY", eps=eps)
        feature_type = str(feature.get("type")).upper()
    else:
        # Actions are always (un)normalized by the ACTION mode, whatever the step lists.
        feature_type = "ACTION"
    mode = norm_map.get(feature_type, "IDENTITY")
    stats = _statistics(model_dir, step.get("state_file"), feature_key)
    if mode == "IDENTITY" or not stats:
        # LeRobot passes a feature through when it has no statistics at all.
        return Normalization(mode=mode, eps=eps)
    if mode not in _STATISTICS:
        raise NotImplementedError(
            f"LeRobot Pi 0.5 normalization mode {mode} for {feature_key} is not supported "
            f"(supported: {', '.join(SUPPORTED_NORMALIZATION)})"
        )
    low, high = _STATISTICS[mode]
    if low not in stats or high not in stats:
        raise ValueError(f"{mode} normalization of {feature_key} needs its {low} and {high} statistics")
    if mode == "MEAN_STD":
        offset, scale = stats["mean"], stats["std"]
    else:
        offset = stats[low]
        spread = (stats[high] - stats[low]).astype(np.float32)
        scale = np.where(spread == 0, np.float32(eps), spread).astype(np.float32)
    return Normalization(
        mode=mode, eps=eps, offset=tuple(map(float, offset)), scale=tuple(map(float, scale))
    )


def _action_range(model_dir: Path, step: dict[str, Any]):
    stats = _statistics(model_dir, step.get("state_file"), ACTION_KEY)
    if "min" not in stats or "max" not in stats:
        return None
    return tuple(map(float, stats["min"])), tuple(map(float, stats["max"]))


def _tokenizer_name(steps: list[dict[str, Any]], tokenizer_max_length: int) -> str:
    _, step = _one_step(steps, "policy_preprocessor.json", "tokenizer_processor")
    config = step.get("config", {})
    if config.get("padding_side", "right") != "right" or config.get("padding") != "max_length":
        raise NotImplementedError("LeRobot Pi 0.5 requires right-padded max_length tokenization")
    if int(config.get("max_length", tokenizer_max_length)) != tokenizer_max_length:
        raise ValueError("LeRobot Pi 0.5 tokenizer max_length differs from tokenizer_max_length")
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

    pre = _steps(model_dir, "policy_preprocessor.json", _PREPROCESSOR_STEPS)
    post = _steps(model_dir, "policy_postprocessor.json", _POSTPROCESSOR_STEPS)
    normalizer_index, normalizer = _one_step(pre, "policy_preprocessor.json", "normalizer_processor")
    prompt_index, _ = _one_step(pre, "policy_preprocessor.json", "pi05_prepare_state_tokenizer_processor_step")
    if normalizer_index > prompt_index:
        # The prompt discretizes the state, so it must see the normalized value.
        raise NotImplementedError("LeRobot Pi 0.5 requires the state to be normalized before the prompt step")
    _, unnormalizer = _one_step(post, "policy_postprocessor.json", "unnormalizer_processor")

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
        tokenizer_name=_tokenizer_name(pre, int(raw["tokenizer_max_length"])),
        min_period=float(raw["min_period"]),
        max_period=float(raw["max_period"]),
        state_normalization=_normalization(model_dir, normalizer, STATE_KEY, observation=True),
        action_normalization=_normalization(model_dir, unnormalizer, ACTION_KEY, observation=False),
        action_range=_action_range(model_dir, unnormalizer),
    )
