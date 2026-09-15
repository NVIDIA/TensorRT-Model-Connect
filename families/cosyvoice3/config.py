# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the published CosyVoice3 configuration without executing HyperPyYAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_ID = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
MODEL_REVISION = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
SOURCE_REVISION = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"

@dataclass(frozen=True)
class FlowConfig:
    dim: int = 1024
    depth: int = 22
    heads: int = 16
    head_dim: int = 64
    ff_mult: int = 2
    mel_dim: int = 80
    spk_dim: int = 80
    time_dim: int = 256
    conv_kernel: int = 31
    conv_groups: int = 16

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim != self.heads * self.head_dim:
            raise ValueError("dim must equal heads * head_dim")
        if self.head_dim % 2 or self.time_dim < 4 or self.time_dim % 2:
            raise ValueError("Rotary and time embedding dimensions must be even")
        if self.dim % self.conv_groups:
            raise ValueError("dim must be divisible by conv_groups")


def read_config(model_dir: str | Path) -> FlowConfig:
    """Accept only the published architecture; never instantiate YAML objects.

    BaseLoader treats even !new/!apply/!ref as plain data. In particular, the
    upstream YAML's torch/numpy/random seed constructors are never executed.
    config.json is not an architecture source for this checkpoint.
    """
    import yaml

    path = Path(model_dir) / "cosyvoice3.yaml"
    if path.stat().st_size > 1_000_000:
        raise ValueError("Unexpectedly large cosyvoice3.yaml")
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    expected = {
        "sample_rate": "24000", "llm_input_size": "896",
        "llm_output_size": "896", "spk_embed_dim": "192",
        "token_frame_rate": "25", "token_mel_ratio": "2",
    }
    if not isinstance(raw, dict):
        raise ValueError("cosyvoice3.yaml must contain a mapping")
    for key, value in expected.items():
        if raw.get(key) != value:
            raise ValueError(f"Unsupported CosyVoice3 {key}: {raw.get(key)!r}")
    try:
        flow = raw["flow"]
        llm = raw["llm"]
        est = flow["decoder"]["estimator"]
        cfg = FlowConfig(
            dim=int(est["dim"]), depth=int(est["depth"]),
            heads=int(est["heads"]), head_dim=int(est["dim_head"]),
            ff_mult=int(est["ff_mult"]), mel_dim=int(est["mel_dim"]),
            spk_dim=int(est["spk_dim"]),
        )
        valid = (cfg == FlowConfig() and est["mu_dim"] == "80"
                 and est["out_channels"] == "80"
                 and flow["vocab_size"] == "6561"
                 and llm["speech_token_size"] == "6561"
                 and est.get("long_skip_connection", "false").lower() == "false")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Invalid or unsupported CosyVoice3 flow configuration") from exc
    if not valid:
        raise ValueError("Only the Fun-CosyVoice3-0.5B-2512 flow architecture is supported")
    return cfg


@dataclass(frozen=True)
class ShapeProfile:
    min_frames: int = 4
    opt_frames: int = 64
    max_frames: int = 256

    def __post_init__(self):
        values = (self.min_frames, self.opt_frames, self.max_frames)
        if any(type(n) is not int for n in values) or not 1 <= values[0] <= values[1] <= values[2] <= 15000:
            raise ValueError("Require 1 <= min_frames <= opt_frames <= max_frames <= 15000")

    def validate_frames(self, frames: int) -> None:
        if not self.min_frames <= frames <= self.max_frames:
            raise ValueError(f"Frame count {frames} outside [{self.min_frames}, {self.max_frames}]")
