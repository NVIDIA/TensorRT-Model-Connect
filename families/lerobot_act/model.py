# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LeRobot Action Chunking Transformer family builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checkpoint import load_checkpoint


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


ACTION_MIN = [
    -0.07363107800483704,
    -0.9587380290031433,
    0.6826214790344238,
    -0.20248547196388245,
    -0.8375535607337952,
    -0.3374757766723633,
    0.15309308469295502,
    -0.3405437469482422,
    -1.0400390625,
    0.4693981409072876,
    -1.4450099468231201,
    -1.0154953002929688,
    -1.3621749877929688,
    0.1409180760383606,
]
ACTION_MAX = [
    0.04141748324036598,
    -0.10431069880723953,
    1.2471264600753784,
    0.012271846644580364,
    -0.26384469866752625,
    0.13038836419582367,
    1.1414905786514282,
    0.33133986592292786,
    0.2791845202445984,
    1.2931458950042725,
    0.25003886222839355,
    0.6120583415031433,
    1.2210487127304077,
    1.2004203796386719,
]


def _read_config(model_dir: str | Path) -> dict[str, Any] | None:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if str(raw.get("type", "")).lower() != "act":
        return None
    return raw


def _validate_initial_policy(raw: dict[str, Any]) -> None:
    inputs = raw.get("input_features") or {}
    outputs = raw.get("output_features") or {}
    expected = {
        "n_obs_steps": 1,
        "chunk_size": 100,
        "n_action_steps": 100,
        "vision_backbone": "resnet18",
        "dim_model": 512,
        "n_heads": 8,
        "dim_feedforward": 3200,
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        "pre_norm": False,
        "use_vae": True,
        "latent_dim": 32,
    }
    mismatches = {
        key: (raw.get(key), value) for key, value in expected.items() if raw.get(key) != value
    }
    if (inputs.get("observation.images.top") or {}).get("shape") != [3, 480, 640]:
        mismatches["observation.images.top"] = (
            (inputs.get("observation.images.top") or {}).get("shape"),
            [3, 480, 640],
        )
    if (inputs.get("observation.state") or {}).get("shape") != [14]:
        mismatches["observation.state"] = (
            (inputs.get("observation.state") or {}).get("shape"),
            [14],
        )
    if (outputs.get("action") or {}).get("shape") != [14]:
        mismatches["action"] = ((outputs.get("action") or {}).get("shape"), [14])
    if raw.get("temporal_ensemble_coeff") is not None:
        mismatches["temporal_ensemble_coeff"] = (raw.get("temporal_ensemble_coeff"), None)
    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {wanted!r})"
            for key, (actual, wanted) in sorted(mismatches.items())
        )
        raise ValueError(f"Unsupported LeRobot ACT policy contract: {details}")


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build the one qualified LeRobot ACT policy contract."""
    if request.task != "robot_control":
        raise ValueError("lerobot_act supports only task=robot_control")
    if request.precision.lower() != "fp32":
        raise ValueError("LeRobot ACT supports only fp32")
    if request.max_sequence_length is not None:
        raise ValueError("LeRobot ACT does not accept max_sequence_length")
    if request.image_height not in {None, 480} or request.image_width not in {None, 640}:
        raise ValueError("LeRobot ACT requires a 640x480 observation image")
    if request.video_num_frames is not None or request.max_batch_size != 1:
        raise ValueError("LeRobot ACT supports one image observation per request")
    if request.tensor_parallel_size != 1 or request.context_parallel_size != 1:
        raise ValueError("LeRobot ACT supports only one device")
    if request.quantization not in {None, "none"} or request.fp32_layers:
        raise ValueError("LeRobot ACT does not support quantization or mixed precision")

    model_dir = Path(request.model_dir)
    raw = _read_config(model_dir)
    if raw is None:
        raise ValueError("LeRobot ACT checkpoint must provide config.json with type=act")
    _validate_initial_policy(raw)

    from .builder import build_act_engine

    plan = build_act_engine(
        raw,
        load_checkpoint(model_dir),
        precision="fp32",
        verbose=request.verbose,
    )
    writer.set_header(family="lerobot_act", task=request.task, backend="trt")
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "image_height": 480,
            "image_width": 640,
            "image_channels": 3,
            "state_dim": 14,
            "action_dim": 14,
            "chunk_size": 100,
            "action_min": ACTION_MIN,
            "action_max": ACTION_MAX,
        },
    )
