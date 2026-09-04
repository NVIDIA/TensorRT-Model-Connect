# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V precomputed FP8 scales."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .model_config import WAN22_TI2V_5B, select_generation_profile


_SCALE_PATH = Path(__file__).with_name("data") / "wan22-ti2v-5b-921dbaf3-fp8-scales.json"


def load_precomputed_fp8_scales(config) -> dict[str, dict[str, float]]:
    """Load the family-owned scale map for its qualified full profile."""
    if select_generation_profile(config.raw) != WAN22_TI2V_5B:
        raise ValueError("Wan2.2 FP8 supports only the 1280x704, 121-frame profile")
    scales = json.loads(_SCALE_PATH.read_text(encoding="utf-8"))
    expected = {
        name
        for index in range(WAN22_TI2V_5B.num_layers)
        for name in (
            f"blocks.{index}.ffn.net.0.proj",
            f"blocks.{index}.ffn.net.2",
            f"blocks.{index}.attn2.to_q",
            f"blocks.{index}.attn2.to_out.0",
        )
    }
    if set(scales) != expected:
        raise ValueError("Wan2.2 FP8 scale map does not match the denoiser graph")
    for name, entry in scales.items():
        if set(entry) != {"input_scale", "weight_scale"}:
            raise ValueError(f"Wan2.2 FP8 scale entry is invalid: {name}")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
            for value in entry.values()
        ):
            raise ValueError(f"Wan2.2 FP8 scale entry is invalid: {name}")
    return scales
