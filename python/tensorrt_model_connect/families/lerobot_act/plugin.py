# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned LeRobot Action Chunking Transformer family plugin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint


POLICY_ID = "lerobot/act_aloha_sim_transfer_cube_human"
POLICY_REVISION = "ba73b2766f1371cdc133ca4efb97eb090d744625"
LEROBOT_REVISION = "3c0a209f9fac4d2a57617e686a7f2a2309144ba2"
DATASET_ID = "lerobot/aloha_sim_transfer_cube_human"
DATASET_REVISION = "6a43d500f101255823a9d2b9dc244eeb01a2cd31"
GYM_ALOHA_VERSION = "0.1.1"
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


def config_from_dir(model_dir: str | Path) -> dict[str, Any] | None:
    raw = _read_config(model_dir)
    if raw is None or not (Path(model_dir) / "model.safetensors").is_file():
        return None
    _validate_initial_policy(raw)
    return {
        **raw,
        "model_type": "lerobot_act",
        "architectures": ["ACTPolicy"],
        "runtime_strategy": "lerobot_act_action_chunk",
        "hidden_size": int(raw["dim_model"]),
        "intermediate_size": int(raw["dim_feedforward"]),
        "num_hidden_layers": int(raw["n_encoder_layers"]),
        "num_attention_heads": int(raw["n_heads"]),
        "num_key_value_heads": int(raw["n_heads"]),
        "max_position_embeddings": 302,
        "requires_tokenizer": False,
    }


class LeRobotActPlugin:
    name = "lerobot_act"
    runtime_strategy = "lerobot_act_action_chunk"
    requires_tokenizer = False
    default_build_precision = "fp32"

    def matches(self, model_type: str) -> bool:
        normalized = (model_type or "").lower().replace("-", "_")
        return normalized in {"act", "act_policy", "actpolicy", "lerobot_act"}

    def matches_config(self, config: Any) -> bool:
        if isinstance(config, str):
            return self.matches(config)
        raw = getattr(config, "raw", {}) or {}
        return str(raw.get("type", "")).lower() == "act" or self.matches(
            str(getattr(config, "model_type", ""))
        )

    def load_weights(
        self,
        model_dir: str,
        config: Any,
        *,
        precision: str = "fp32",
    ) -> dict:
        if precision != "fp32":
            raise ValueError("The pinned LeRobot ACT accuracy contract supports fp32 builds only")
        _validate_initial_policy(config.raw)
        return load_checkpoint(model_dir)

    def build_engine(
        self,
        config: Any,
        weights: dict,
        max_cache_length: int,
        *,
        precision: str = "fp32",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        del max_cache_length
        if quant_ctx is not None:
            raise ValueError("LeRobot ACT does not support quantized builds")
        from .builder import build_act_engine

        return build_act_engine(config.raw, weights, precision=precision, verbose=verbose)

    def get_bundle_config_overrides(self, config: Any) -> dict[str, Any]:
        _validate_initial_policy(config.raw)
        return {
            "policy_id": POLICY_ID,
            "policy_revision": POLICY_REVISION,
            "lerobot_revision": LEROBOT_REVISION,
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "gym_aloha_version": GYM_ALOHA_VERSION,
            "task_environment": "AlohaTransferCube-v0",
            "control_frequency_hz": 50,
            "observation_image_key": "observation.images.top",
            "observation_image_height": 480,
            "observation_image_width": 640,
            "observation_image_channels": 3,
            "observation_state_key": "observation.state",
            "observation_state_dim": 14,
            "action_key": "action",
            "action_dim": 14,
            "action_chunk_size": 100,
            "action_steps": 100,
            "temporal_context_steps": 1,
            "action_training_min": ACTION_MIN,
            "action_training_max": ACTION_MAX,
            "robotics_safety_validated": False,
        }


plugin = LeRobotActPlugin()
