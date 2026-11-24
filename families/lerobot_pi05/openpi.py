# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The openpi policies are separate from the LeRobot processor bundled with the same weights."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from . import config


@dataclass(frozen=True)
class Policy:
    state_dim: int
    action_dim: int
    horizon: int
    cameras: tuple[str, ...]
    stats_subdir: str
    stats_generation: str


POLICIES = {
    "pi05_libero": Policy(8, 7, 10, ("observation/image", "observation/wrist_image"),
                          "pi05_libero/assets/physical-intelligence/libero", "1757354341826612"),
    "pi05_droid": Policy(8, 8, 15, ("observation/exterior_image_1_left", "observation/wrist_image_left"),
                         "pi05_droid/assets/droid", "1757354377575323"),
    "pi05_aloha": Policy(14, 14, 50, ("cam_high", "cam_left_wrist", "cam_right_wrist"),
                         "pi05_base/assets/trossen", "1757354310095901"),
}


def _quantiles(stats: dict, feature: str, width: int) -> config.Normalization:
    # DROID stores 32 padded statistics in the training asset. Only robot coordinates are used.
    lo, hi = (np.asarray(stats[feature][k], dtype=np.float32) for k in ("q01", "q99"))
    if (lo.ndim != 1 or hi.shape != lo.shape or lo.size < width
            or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi < lo)):
        raise ValueError(f"Invalid openpi quantiles for {feature} ({width} coordinates required)")
    lo, hi = lo[:width], hi[:width]
    # openpi adds epsilon even for non-degenerate intervals, in both directions.
    return config.Normalization("QUANTILES", 1e-6, tuple(map(float, lo)),
                                tuple(map(float, hi - lo + np.float32(1e-6))))


def load_policy(model_dir: Path, name: str) -> config.Pi05Config:
    if name not in POLICIES:
        raise ValueError(f"Unsupported openpi policy: {name}")
    policy = POLICIES[name]
    raw = json.loads((model_dir / "config.json").read_text())
    # The mirror's vision-language/DiT topology is authoritative, but not its robot processor.
    config._check_boundary(raw)
    required = {"n_action_steps": policy.horizon, "max_state_dim": 32, "max_action_dim": 32,
                "tokenizer_max_length": 200}
    for key, value in required.items():
        if raw.get(key) != value:
            raise ValueError(f"{name} requires {key}={value}, got {raw.get(key)!r}")
    if raw.get("num_inference_steps") != 10:
        raise ValueError(f"{name} requires the documented ten denoise steps")
    stats = json.loads((model_dir / "assets/norm_stats.json").read_text())["norm_stats"]
    return config.Pi05Config(
        cameras=policy.cameras, empty_cameras=0, camera_height=224, camera_width=224,
        state_dim=policy.state_dim, action_dim=policy.action_dim, chunk_size=policy.horizon,
        n_action_steps=policy.horizon, max_state_dim=32, max_action_dim=32, num_inference_steps=10,
        tokenizer_max_length=200, tokenizer_name="google/paligemma-3b-pt-224",
        min_period=float(raw["min_period"]), max_period=float(raw["max_period"]),
        state_normalization=_quantiles(stats, "state", policy.state_dim),
        action_normalization=_quantiles(stats, "actions", policy.action_dim), openpi_policy=name)
