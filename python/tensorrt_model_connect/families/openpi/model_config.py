# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned architecture and deployment profiles for OpenPI flow policies.

The values in this module are intentionally explicit.  Checkpoint conversion
must not infer a model variant from a path name: doing so can silently select
the wrong action horizon or state-tokenization contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping


OPENPI_UPSTREAM_REPOSITORY = "https://github.com/Physical-Intelligence/openpi"
OPENPI_UPSTREAM_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
OPENPI_MODEL_TYPE = "openpi_pi05_flow"


@dataclass(frozen=True)
class VisionConfig:
    image_size: int = 224
    patch_size: int = 14
    width: int = 1152
    depth: int = 27
    mlp_dim: int = 4304
    num_heads: int = 16
    output_width: int = 2048
    num_image_slots: int = 3

    @property
    def tokens_per_image(self) -> int:
        return (self.image_size // self.patch_size) ** 2


@dataclass(frozen=True)
class GemmaConfig:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int

    @property
    def attention_width(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim


@dataclass(frozen=True)
class OpenPIProfile:
    """Complete, named policy contract used for conversion and deployment."""

    name: str
    checkpoint_uri: str
    asset_id: str
    action_horizon: int
    external_state_dim: int
    external_action_dim: int
    discrete_state_input: bool
    camera_names: tuple[str, str, str] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    camera_mask: tuple[bool, bool, bool] = (True, True, False)
    action_dim: int = 32
    max_token_length: int = 200
    denoise_steps: int = 10
    dtype: str = "bfloat16"
    use_quantile_normalization: bool = True
    vocab_size: int = 257_152
    rms_norm_epsilon: float = 1e-6
    rope_theta: float = 10_000.0
    upstream_commit: str = OPENPI_UPSTREAM_COMMIT
    vision: VisionConfig = VisionConfig()
    prefix: GemmaConfig = GemmaConfig(
        width=2048,
        depth=18,
        mlp_dim=16_384,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    )
    action_expert: GemmaConfig = GemmaConfig(
        width=1024,
        depth=18,
        mlp_dim=4096,
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
    )

    def __post_init__(self) -> None:
        if self.upstream_commit != OPENPI_UPSTREAM_COMMIT:
            raise ValueError(
                "OpenPI profile must use the audited upstream commit "
                f"{OPENPI_UPSTREAM_COMMIT}; got {self.upstream_commit}"
            )
        if len(self.camera_names) != self.vision.num_image_slots:
            raise ValueError("camera_names must match the number of image slots")
        if len(self.camera_mask) != self.vision.num_image_slots:
            raise ValueError("camera_mask must match the number of image slots")
        if self.action_dim < self.external_action_dim:
            raise ValueError("internal action_dim cannot be smaller than external_action_dim")
        if self.action_dim < self.external_state_dim:
            raise ValueError("internal action_dim cannot be smaller than external_state_dim")
        if self.action_horizon <= 0 or self.denoise_steps <= 0:
            raise ValueError("action_horizon and denoise_steps must be positive")
        if self.prefix.depth != self.action_expert.depth:
            raise ValueError("prefix and action experts must have the same depth")
        if (
            self.prefix.num_heads != self.action_expert.num_heads
            or self.prefix.num_kv_heads != self.action_expert.num_kv_heads
            or self.prefix.head_dim != self.action_expert.head_dim
        ):
            raise ValueError("prefix and action experts must share attention geometry")

    @property
    def prefix_length(self) -> int:
        return self.vision.num_image_slots * self.vision.tokens_per_image + self.max_token_length

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROFILES: Mapping[str, OpenPIProfile] = MappingProxyType(
    {
        "pi05_droid": OpenPIProfile(
            name="pi05_droid",
            checkpoint_uri="gs://openpi-assets/checkpoints/pi05_droid",
            asset_id="droid",
            action_horizon=15,
            external_state_dim=8,
            external_action_dim=8,
            discrete_state_input=True,
        ),
    }
)


def profile_names() -> tuple[str, ...]:
    return tuple(_PROFILES)


def get_profile(name: str) -> OpenPIProfile:
    """Return an explicit supported profile; path-based inference is forbidden."""

    try:
        return _PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(profile_names())
        raise ValueError(
            f"unsupported OpenPI profile {name!r}; expected one of: {choices}"
        ) from exc


def config_from_dir(model_dir: str) -> dict[str, Any] | None:
    """Read a prepared OpenPI config without importing conversion dependencies."""

    import json
    from pathlib import Path

    config_path = Path(model_dir) / "openpi_config.json"
    if not config_path.is_file():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    profile = get_profile(str(payload.get("profile", "")))
    if payload.get("upstream_commit") != OPENPI_UPSTREAM_COMMIT:
        raise ValueError(
            f"prepared OpenPI config has an unaudited upstream commit: {payload.get('upstream_commit')!r}"
        )
    return {
        "model_type": OPENPI_MODEL_TYPE,
        "architectures": ["OpenPI05FlowPolicy"],
        "vocab_size": profile.vocab_size,
        "hidden_size": profile.prefix.width,
        "intermediate_size": profile.prefix.mlp_dim,
        "num_hidden_layers": profile.prefix.depth,
        "num_attention_heads": profile.prefix.num_heads,
        "num_key_value_heads": profile.prefix.num_kv_heads,
        "head_dim": profile.prefix.head_dim,
        "max_position_embeddings": profile.prefix_length + profile.action_horizon,
        "rms_norm_eps": profile.rms_norm_epsilon,
        "rope_theta": profile.rope_theta,
        "openpi": profile.to_dict(),
        "openpi_profile": profile.name,
        "openpi_upstream_commit": payload.get("upstream_commit"),
        "openpi_checkpoint_uri": payload.get("checkpoint_uri"),
        "openpi_checkpoint_identity_sha256": payload.get("checkpoint_identity_sha256"),
        "openpi_weights_file": payload.get("weights", "model.safetensors"),
        "openpi_tokenizer_file": payload.get("tokenizer", "tokenizer.model"),
        "openpi_tokenizer_sha256": payload.get("tokenizer_sha256"),
        "openpi_tokenizer_source_sha256": payload.get("tokenizer_source_sha256"),
        "openpi_tokenizer_export": payload.get("tokenizer_export"),
        "openpi_normalization_file": payload.get("normalization", "preprocessor_config.json"),
        "openpi_normalization_sha256": payload.get("normalization_sha256"),
        "openpi_conversion_manifest": payload.get(
            "conversion_manifest", "openpi_conversion_manifest.json"
        ),
        "openpi_conversion_manifest_sha256": payload.get("conversion_manifest_sha256"),
    }
