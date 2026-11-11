# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recognize released dense D3 geometries and explicit decision checkpoint identity."""
import json
from pathlib import Path
from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata

PROFILES = json.loads(Path(__file__).with_name("profiles.json").read_text())


def profile_name(config):
    text, vision = config.get("text_config", {}), config.get("vision_config", {})
    if not isinstance(text, dict) or not isinstance(vision, dict):
        return None
    for name, profile in PROFILES.items():
        if (all(text.get(k) == v for k, v in profile["text"].items())
                and all(vision.get(k) == v for k, v in profile["vision"].items())):
            return name
    return None


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    c = metadata.config
    text = c.get("text_config", {})
    if (metadata.model_type != "qwen3_5" or metadata.architectures != ("Qwen3_5Model",)
            or not {"decision_config.json", "readout.safetensors"} <= set(metadata.files)
            or not isinstance(text, dict)
            or profile_name(c) is None
            or not isinstance(c.get("vision_config"), dict)):
        return None
    return FamilySupport(("structured_decision",), "structured_decision", "bf16")
