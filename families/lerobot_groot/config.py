# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and validate a LeRobot GR00T N1.7 checkpoint directory.

The supported boundary is the processor pipeline LeRobot 0.6.1 saves for an N1.7
policy with absolute actions: anything outside it fails here, with the offending
field named, instead of building a graph that silently disagrees with LeRobot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_PREFIX = "observation.images."
BACKBONE = "nvidia/Cosmos-Reason2-2B"
NATIVE_ACTION_HORIZON = 40
MAX_DIM = 132
LIBERO_DECODE = "libero"

_PREPROCESSOR_STEPS = (
    "rename_observations_processor",
    "to_batch_processor",
    "groot_n1_7_pack_inputs_v1",
    "groot_n1_7_vlm_encode_v1",
    "device_processor",
)
_POSTPROCESSOR_STEPS = ("groot_action_unpack_unnormalize_v2", "device_processor")

_REQUIRED_EXACT = {
    "type": "groot",
    "n_obs_steps": 1,
    "max_state_dim": MAX_DIM,
    "max_action_dim": MAX_DIM,
}
# Settings that change the model or its inputs in ways this family does not reproduce.
_REQUIRED_FALSE = ("use_relative_actions", "use_peft")


@dataclass(frozen=True)
class ImageTransform:
    """The evaluation-time crop LeRobot applies with ``use_albumentations``: OpenCV ``INTER_AREA``
    resize of the shortest edge, a floored centre crop of ``crop_fraction``, resize back."""

    shortest_edge: int
    crop_fraction: float
    target_height: int
    target_width: int


@dataclass(frozen=True)
class GrootConfig:
    cameras: tuple[str, ...]  # In the order the model sees them.
    camera_height: int
    camera_width: int
    state_dim: int
    action_dim: int
    chunk_size: int  # Actions returned per call: the leading steps of the native horizon.
    action_horizon: int
    max_state_dim: int
    max_action_dim: int
    num_inference_steps: int
    embodiment_tag: str
    embodiment_id: int
    formalize_language: bool
    image: ImageTransform
    state_min: tuple[float, ...]
    state_max: tuple[float, ...]
    clip_state: bool
    action_min: tuple[float, ...]
    action_max: tuple[float, ...]
    clip_action: bool
    libero_gripper: bool  # Turn the last action dimension into LIBERO's -1/+1 gripper command.


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot GR00T checkpoint is missing {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _unsupported(details: dict[str, tuple[Any, Any]]) -> None:
    if details:
        text = ", ".join(f"{key}={actual!r} (expected {wanted!r})" for key, (actual, wanted) in sorted(details.items()))
        raise ValueError(f"Unsupported LeRobot GR00T policy contract: {text}")


def _check_boundary(raw: dict[str, Any]) -> None:
    mismatches = {key: (raw.get(key), wanted) for key, wanted in _REQUIRED_EXACT.items() if raw.get(key) != wanted}
    mismatches.update({key: (raw.get(key), False) for key in _REQUIRED_FALSE if raw.get(key)})
    if raw.get("action_decode_transform") not in (None, LIBERO_DECODE):
        mismatches["action_decode_transform"] = (raw.get("action_decode_transform"), f"None or {LIBERO_DECODE!r}")
    _unsupported(mismatches)


def _steps(model_dir: Path, processor: str, expected: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """The processor's steps by registry name, after checking they are exactly ``expected``."""
    steps = _read_json(model_dir / processor).get("steps", [])
    names = tuple(str(step.get("registry_name", "")) for step in steps)
    if names != expected:
        raise NotImplementedError(f"LeRobot GR00T {processor} has steps {list(names)}; supported: {list(expected)}")
    return {name: step for name, step in zip(names, steps)}


def _statistics(model_dir: Path, step: dict[str, Any], feature_key: str) -> dict[str, np.ndarray]:
    state_file = step.get("state_file")
    if not state_file:
        raise ValueError(f"LeRobot GR00T step {step.get('registry_name')} carries no statistics")
    path = model_dir / state_file
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot GR00T checkpoint is missing {state_file}")
    prefix = feature_key + "."
    with safe_open(str(path), framework="numpy") as reader:
        stats = {name[len(prefix):]: np.asarray(reader.get_tensor(name), dtype=np.float32).reshape(-1)
                 for name in reader.keys() if name.startswith(prefix)}
    if "min" not in stats or "max" not in stats:
        raise ValueError(f"LeRobot GR00T {state_file} has no min/max statistics for {feature_key}")
    return stats


def _features(raw: dict[str, Any]) -> tuple[list[str], tuple[int, int], int, int]:
    inputs, outputs = raw.get("input_features") or {}, raw.get("output_features") or {}
    cameras = {name: list(spec["shape"]) for name, spec in inputs.items() if spec.get("type") == "VISUAL"}
    state = [name for name, spec in inputs.items() if spec.get("type") == "STATE"]
    action = [name for name, spec in outputs.items() if spec.get("type") == "ACTION"]
    if not cameras or any(not name.startswith(IMAGE_PREFIX) for name in cameras):
        raise ValueError(f"LeRobot GR00T needs camera inputs named {IMAGE_PREFIX}<camera>")
    if state != [STATE_KEY] or action != [ACTION_KEY]:
        raise ValueError(f"LeRobot GR00T needs exactly one {STATE_KEY} input and one {ACTION_KEY} output")
    shapes = {tuple(shape) for shape in cameras.values()}
    if len(shapes) != 1:
        raise NotImplementedError("LeRobot GR00T cameras must share one image shape")
    height, width, channels = next(iter(shapes))  # LeRobot records GR00T cameras as HWC.
    if channels != 3:
        raise NotImplementedError("LeRobot GR00T requires 3-channel camera images")
    state_shape, action_shape = inputs[STATE_KEY]["shape"], outputs[ACTION_KEY]["shape"]
    if len(state_shape) != 1 or len(action_shape) != 1:
        raise ValueError("LeRobot GR00T state and action must be vectors")
    return list(cameras), (int(height), int(width)), int(state_shape[0]), int(action_shape[0])


def _camera_order(cameras: list[str], keys: Any) -> tuple[str, ...]:
    """The pack step's order: sorted names, or the checkpoint's ``video_modality_keys``."""
    if keys is None:
        return tuple(sorted(cameras))
    ordered = tuple(IMAGE_PREFIX + str(key) for key in keys)
    if sorted(ordered) != sorted(cameras):
        raise NotImplementedError(f"LeRobot GR00T video_modality_keys {keys} must name exactly the cameras {cameras}")
    return ordered


def _pack(step: dict[str, Any]) -> dict[str, Any]:
    config = step.get("config", {})
    expected = {"state_horizon": 1, "action_horizon": NATIVE_ACTION_HORIZON, "video_horizon": None,
                "max_state_dim": MAX_DIM, "max_action_dim": MAX_DIM, "language_key": "task", "normalize_min_max": True}
    mismatches = {f"groot_n1_7_pack_inputs_v1.{k}": (config.get(k), v) for k, v in expected.items() if config.get(k) != v}
    for key in ("raw_stats", "modality_config"):
        if config.get(key):
            # Grouped (per-modality or relative) statistics change normalization; not reproduced.
            mismatches[f"groot_n1_7_pack_inputs_v1.{key}"] = ("non-empty", "empty")
    _unsupported(mismatches)
    tag = config.get("embodiment_tag")
    mapping = config.get("embodiment_mapping") or {}
    if tag not in mapping:
        raise ValueError(f"LeRobot GR00T embodiment_tag {tag!r} is not in the checkpoint's embodiment_mapping")
    return config


def _image(step: dict[str, Any], height: int, width: int) -> ImageTransform:
    config = step.get("config", {})
    if config.get("model_name") != BACKBONE:
        raise NotImplementedError(f"LeRobot GR00T backbone {config.get('model_name')!r} is not supported (only {BACKBONE})")
    if not config.get("use_albumentations") or config.get("letter_box_transform"):
        raise NotImplementedError("LeRobot GR00T supports the use_albumentations crop without letter boxing only")
    target = config.get("image_target_size")
    if not target or len(target) != 2:
        raise ValueError("LeRobot GR00T needs image_target_size")
    target_h, target_w = int(target[0]), int(target[1])
    shortest = int(config.get("shortest_image_edge") or target_h)
    fraction = config.get("crop_fraction")
    if fraction is None and config.get("image_crop_size"):
        fraction = config["image_crop_size"][0] / float(target_h)
    fraction = float(fraction) if fraction is not None else 1.0
    if (height, width) != (target_h, target_w) or height != width or shortest != target_h:
        # Non-square cameras or a resize to another size are not implemented.
        raise NotImplementedError(
            f"LeRobot GR00T supports square cameras already at image_target_size {target}, got {height}x{width}")
    return ImageTransform(shortest_edge=shortest, crop_fraction=fraction, target_height=target_h, target_width=target_w)


def load_config(model_dir: str | Path) -> GrootConfig:
    model_dir = Path(model_dir)
    raw = _read_json(model_dir / "config.json")
    _check_boundary(raw)
    cameras, (height, width), state_dim, action_dim = _features(raw)

    pre = _steps(model_dir, "policy_preprocessor.json", _PREPROCESSOR_STEPS)
    post = _steps(model_dir, "policy_postprocessor.json", _POSTPROCESSOR_STEPS)
    if pre["rename_observations_processor"].get("config", {}).get("rename_map"):
        raise NotImplementedError("LeRobot GR00T supports an empty rename_map only")
    pack = _pack(pre["groot_n1_7_pack_inputs_v1"])
    image = _image(pre["groot_n1_7_vlm_encode_v1"], height, width)
    unpack_step = post["groot_action_unpack_unnormalize_v2"]
    unpack = unpack_step.get("config", {})
    if not unpack.get("normalize_min_max") or int(unpack.get("env_action_dim", 0)) != action_dim:
        raise NotImplementedError("LeRobot GR00T action decoding must unnormalize min/max to the action size")

    chunk = min(int(raw["chunk_size"]), int(raw.get("n_action_steps", raw["chunk_size"])), NATIVE_ACTION_HORIZON)
    if state_dim > MAX_DIM or action_dim > MAX_DIM or chunk < 1:
        raise ValueError("LeRobot GR00T state/action size or chunk size is outside the model")
    state = _statistics(model_dir, pre["groot_n1_7_pack_inputs_v1"], STATE_KEY)
    action = _statistics(model_dir, unpack_step, ACTION_KEY)
    if state["min"].size != state_dim or action["min"].size != action_dim:
        raise ValueError("LeRobot GR00T statistics do not match the state/action sizes")
    libero = raw.get("action_decode_transform") == LIBERO_DECODE
    if libero and action_dim < 7:
        raise ValueError("LeRobot GR00T LIBERO action decoding needs at least 7 action dimensions")

    return GrootConfig(
        cameras=_camera_order(cameras, pack.get("video_modality_keys")),
        camera_height=height,
        camera_width=width,
        state_dim=state_dim,
        action_dim=action_dim,
        chunk_size=chunk,
        action_horizon=NATIVE_ACTION_HORIZON,
        max_state_dim=MAX_DIM,
        max_action_dim=MAX_DIM,
        num_inference_steps=int(raw["num_inference_timesteps"]),
        embodiment_tag=str(pack["embodiment_tag"]),
        embodiment_id=int(pack["embodiment_mapping"][pack["embodiment_tag"]]),
        formalize_language=bool(pack.get("formalize_language", True)),
        image=image,
        state_min=tuple(map(float, state["min"])),
        state_max=tuple(map(float, state["max"])),
        clip_state=bool(pack.get("clip_outliers", True)),
        action_min=tuple(map(float, action["min"])),
        action_max=tuple(map(float, action["max"])),
        clip_action=bool(unpack.get("clip_normalized_action", False)),
        # LeRobot derives the gripper transform from action_decode_transform, not the saved step flag.
        libero_gripper=libero,
    )
