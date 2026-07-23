# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-Connect family plugin for pinned OpenPI π0.5 policies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .model_config import (
    OPENPI_MODEL_TYPE,
    OPENPI_UPSTREAM_COMMIT,
    OPENPI_UPSTREAM_REPOSITORY,
    OpenPIProfile,
    get_profile,
)
from .numerics import round_to_bfloat16_float32
from .prefill_builder import (
    build_prefill_engine,
    required_prefill_weight_shapes,
)


class ModelConfig(Protocol):
    raw: dict[str, Any]


WeightDict = dict[str, Any]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str, *, reject_zero: bool = False) -> str:
    digest = str(value or "")
    if (
        len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        or (reject_zero and digest == "0" * 64)
    ):
        raise ValueError(f"OpenPI prepared config lacks a valid {label} SHA-256")
    return digest


def _require_manifest_artifact(
    manifest: dict[str, Any],
    name: str,
    *,
    expected_file: str,
    actual_sha256: str,
) -> None:
    artifacts = manifest.get("artifacts")
    artifact = artifacts.get(name) if isinstance(artifacts, dict) else None
    if not isinstance(artifact, dict):
        raise ValueError(f"OpenPI conversion manifest is missing its {name} artifact")
    if artifact.get("file") != expected_file:
        raise ValueError(f"OpenPI manifest {name} path does not match the prepared config")
    manifest_sha256 = _require_sha256(artifact.get("sha256"), f"manifest {name}")
    if manifest_sha256 != actual_sha256:
        raise ValueError(f"OpenPI manifest {name} SHA-256 does not match the staged asset")
    asset_sha256 = artifact.get("asset_sha256")
    if (
        asset_sha256 is not None
        and _require_sha256(asset_sha256, f"manifest {name} asset") != actual_sha256
    ):
        raise ValueError(f"OpenPI manifest {name} asset SHA-256 does not match the staged asset")


def _resolve_owned_file(model_dir: str | Path, relative: str, label: str) -> Path:
    root = Path(model_dir).resolve()
    if not relative:
        raise ValueError(f"OpenPI {label} path is missing from the prepared config")
    relative_path = Path(relative)
    candidate = root / relative_path
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"OpenPI {label} escapes the prepared model directory")
    if not candidate.is_file() or candidate.stat().st_size == 0:
        raise FileNotFoundError(f"OpenPI {label} is missing or empty: {candidate}")
    return candidate


def _profile_from_config(config: ModelConfig) -> OpenPIProfile:
    raw = config.raw or {}
    profile_name = raw.get("openpi_profile")
    if not profile_name and isinstance(raw.get("openpi"), dict):
        profile_name = raw["openpi"].get("name")
    profile = get_profile(str(profile_name or ""))
    if raw.get("openpi_upstream_commit") != OPENPI_UPSTREAM_COMMIT:
        raise ValueError("OpenPI build config is not bound to the audited upstream commit")
    return profile


def _validated_prepared_weight_path(model_dir: str | Path, config: ModelConfig) -> Path:
    """Validate the prepared snapshot and return its converted weights.

    The source config remains compatible with the original nested OpenPI asset
    paths.  ``tokenizer.model`` and ``preprocessor_config.json`` are mandatory
    aliases because the existing shared bundle writer embeds those standard
    root filenames without an OpenPI-specific hook.
    """

    raw = config.raw or {}
    profile = _profile_from_config(config)
    weights_relative = str(raw.get("openpi_weights_file") or "model.safetensors")
    weights_path = _resolve_owned_file(model_dir, weights_relative, "converted weights")
    tokenizer_relative = str(raw.get("openpi_tokenizer_file") or "tokenizer.model")
    tokenizer_path = _resolve_owned_file(model_dir, tokenizer_relative, "flattened tokenizer")
    normalization_relative = str(raw.get("openpi_normalization_file") or "preprocessor_config.json")
    normalization_path = _resolve_owned_file(
        model_dir, normalization_relative, "normalization statistics"
    )
    manifest_relative = str(
        raw.get("openpi_conversion_manifest") or "openpi_conversion_manifest.json"
    )
    manifest_path = _resolve_owned_file(model_dir, manifest_relative, "conversion manifest")
    manifest_payload = manifest_path.read_bytes()
    manifest_sha256 = _sha256_bytes(manifest_payload)
    expected_manifest_sha256 = _require_sha256(
        raw.get("openpi_conversion_manifest_sha256"), "conversion manifest"
    )
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("OpenPI conversion manifest SHA-256 does not match the prepared config")
    manifest = _strict_json_object(manifest_payload, "conversion manifest")
    upstream = manifest.get("upstream")
    if (
        not isinstance(upstream, dict)
        or upstream.get("repository") != OPENPI_UPSTREAM_REPOSITORY
        or upstream.get("commit") != OPENPI_UPSTREAM_COMMIT
    ):
        raise ValueError(
            "OpenPI conversion manifest has an unaudited upstream repository or commit"
        )
    if manifest.get("profile") != profile.name:
        raise ValueError("OpenPI conversion manifest profile does not match the build config")

    checkpoint_sha256 = _require_sha256(
        raw.get("openpi_checkpoint_identity_sha256"),
        "checkpoint identity",
        reject_zero=True,
    )
    source_checkpoint = manifest.get("source_checkpoint")
    manifest_checkpoint_sha256 = (
        source_checkpoint.get("identity_sha256") if isinstance(source_checkpoint, dict) else None
    )
    if manifest_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("OpenPI manifest checkpoint identity does not match the prepared config")

    weights_sha256 = _sha256_file(weights_path)
    _require_manifest_artifact(
        manifest,
        "weights",
        expected_file=weights_relative,
        actual_sha256=weights_sha256,
    )

    tokenizer = tokenizer_path.read_bytes()
    if not tokenizer.startswith(b"TRTMCBPE"):
        raise ValueError("OpenPI flattened tokenizer has an invalid binary header")
    tokenizer_sha256 = _sha256_bytes(tokenizer)
    expected_tokenizer_sha256 = _require_sha256(
        raw.get("openpi_tokenizer_sha256"),
        "tokenizer",
    )
    if tokenizer_sha256 != expected_tokenizer_sha256:
        raise ValueError("OpenPI tokenizer SHA-256 does not match the prepared config")
    _require_manifest_artifact(
        manifest,
        "tokenizer",
        expected_file=tokenizer_relative,
        actual_sha256=tokenizer_sha256,
    )

    normalization = normalization_path.read_bytes()
    _strict_json_object(normalization, "normalization statistics")
    normalization_sha256 = _sha256_bytes(normalization)
    configured_normalization_sha256 = raw.get("openpi_normalization_sha256")
    if configured_normalization_sha256 is not None and (
        _require_sha256(
            configured_normalization_sha256,
            "normalization statistics",
        )
        != normalization_sha256
    ):
        raise ValueError("OpenPI normalization SHA-256 does not match the prepared config")
    _require_manifest_artifact(
        manifest,
        "normalization",
        expected_file=normalization_relative,
        actual_sha256=normalization_sha256,
    )

    standard_tokenizer = _resolve_owned_file(
        model_dir, "tokenizer.model", "standard tokenizer bundle asset"
    )
    if _sha256_file(standard_tokenizer) != tokenizer_sha256:
        raise ValueError("OpenPI tokenizer.model does not match the converted tokenizer asset")
    standard_preprocessor = _resolve_owned_file(
        model_dir,
        "preprocessor_config.json",
        "standard normalization bundle asset",
    )
    if _sha256_file(standard_preprocessor) != normalization_sha256:
        raise ValueError("OpenPI preprocessor_config.json does not match the normalization asset")

    # The final config binds only payloads that are actually bundled. Source
    # checkpoint and conversion-manifest identity remain build-time checks.
    raw["openpi_tokenizer_sha256"] = tokenizer_sha256
    raw["openpi_normalization_sha256"] = normalization_sha256
    return weights_path


def _load_prepared_weights(model_dir: str | Path, config: ModelConfig) -> WeightDict:
    path = _validated_prepared_weight_path(model_dir, config)
    try:
        from safetensors.numpy import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to build an OpenPI plan") from exc
    loaded = load_file(str(path))
    result = WeightDict()
    for name, value in loaded.items():
        array = np.asarray(value)
        if array.dtype.kind != "f" and array.dtype.name != "bfloat16":
            raise ValueError(f"OpenPI converted weight {name!r} is not floating point")
        result[name] = np.ascontiguousarray(array)
    return result


def _validate_weight_inventory(weights: WeightDict, profile: OpenPIProfile) -> None:
    from .action_expert_builder import required_action_weight_shapes

    expected = {
        **required_prefill_weight_shapes(profile),
        **required_action_weight_shapes(profile),
    }
    actual = {name for name, value in weights.items() if isinstance(value, np.ndarray)}
    missing = sorted(set(expected) - actual)
    unexpected = sorted(actual - set(expected))
    shape_errors = [
        f"{name}: expected {shape}, got {tuple(np.asarray(weights[name]).shape)}"
        for name, shape in expected.items()
        if name in weights and tuple(np.asarray(weights[name]).shape) != shape
    ]
    if missing or unexpected or shape_errors:
        details = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unexpected:
            details.append("unexpected=" + ", ".join(unexpected))
        if shape_errors:
            details.append("shape=" + "; ".join(shape_errors))
        raise ValueError("OpenPI prepared weight inventory mismatch: " + " | ".join(details))


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"OpenPI {label} is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ValueError(f"OpenPI {label} must contain a JSON object")
    return document


def _require_bf16_precision(precision: str) -> None:
    if precision != "bf16":
        raise ValueError(f"OpenPI builds support only precision='bf16'; got {precision!r}")


class OpenPIPlugin:
    name = "openpi"
    runtime_strategy = "openpi_vla"
    runtime_capabilities = ("robot_action_generation",)
    requires_tokenizer = False

    def matches(self, model_type: str) -> bool:
        return (model_type or "").lower().replace("-", "_") in {
            OPENPI_MODEL_TYPE,
            "openpi",
            "pi05_droid",
        }

    def load_weights(
        self,
        model_dir: str,
        config: ModelConfig,
        *,
        precision: str = "bf16",
    ) -> WeightDict:
        _require_bf16_precision(precision)
        profile = _profile_from_config(config)
        weights = _load_prepared_weights(model_dir, config)
        _validate_weight_inventory(weights, profile)
        # Pinned upstream calls restore_params(..., dtype=jnp.bfloat16), so
        # even parameters used at FP32 boundaries carry BF16-rounded values.
        # TensorRT constants must start from the same values.
        for name, value in weights.items():
            weights[name] = round_to_bfloat16_float32(value)
        return weights

    def build_engine(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "bf16",
        quant_ctx=None,
        verbose: bool = False,
    ) -> bytes:
        _require_bf16_precision(precision)
        del max_cache_length, quant_ctx
        plan = build_prefill_engine(
            weights,
            _profile_from_config(config),
            precision=precision,
            verbose=verbose,
        )
        config.raw["openpi_prefill_engine_sha256"] = _sha256_bytes(plan)
        return plan

    def build_extra_engines(
        self,
        config: ModelConfig,
        weights: WeightDict,
        max_cache_length: int,
        *,
        precision: str = "bf16",
        verbose: bool = False,
        build_timing: dict | None = None,
    ) -> dict[str, bytes]:
        _require_bf16_precision(precision)
        del max_cache_length, build_timing
        from .action_expert_builder import build_action_expert_engine

        plan = build_action_expert_engine(
            _profile_from_config(config),
            weights,
            precision=precision,
            verbose=verbose,
        )
        config.raw["openpi_action_engine_sha256"] = _sha256_bytes(plan)
        return {"openpi_action_step_engine_plan": plan}

    def get_bundle_config_overrides(self, config: ModelConfig) -> dict[str, Any]:
        profile = _profile_from_config(config)
        raw = config.raw or {}
        return {
            "engine_backend": "trt",
            "runtime_strategy": self.runtime_strategy,
            "task_strategy": "robot_action_generation",
            "user_contract": "robot_action_chunk",
            "model_type": OPENPI_MODEL_TYPE,
            "openpi_profile": profile.name,
            "openpi_upstream_commit": OPENPI_UPSTREAM_COMMIT,
            "openpi_action_horizon": profile.action_horizon,
            "openpi_internal_action_dim": profile.action_dim,
            "openpi_external_action_dim": profile.external_action_dim,
            "openpi_external_state_dim": profile.external_state_dim,
            "openpi_prefix_length": profile.prefix_length,
            "openpi_max_token_length": profile.max_token_length,
            "openpi_num_layers": profile.prefix.depth,
            "openpi_num_heads": profile.prefix.num_heads,
            "openpi_num_kv_heads": profile.prefix.num_kv_heads,
            "openpi_head_dim": profile.prefix.head_dim,
            "openpi_denoise_steps": profile.denoise_steps,
            "openpi_discrete_state_input": profile.discrete_state_input,
            "openpi_camera_names": list(profile.camera_names),
            "openpi_camera_mask": list(profile.camera_mask),
            "openpi_batch_size": 1,
            "openpi_runtime_contract": "native_cpp_device_resident_flow",
            "openpi_parameter_dtype": "bfloat16",
            "openpi_tokenizer_sha256": _require_sha256(
                raw.get("openpi_tokenizer_sha256"),
                "tokenizer",
                reject_zero=True,
            ),
            "openpi_normalization_sha256": _require_sha256(
                raw.get("openpi_normalization_sha256"),
                "normalization statistics",
                reject_zero=True,
            ),
            "openpi_prefill_engine_sha256": _require_sha256(
                raw.get("openpi_prefill_engine_sha256"),
                "prefill engine",
                reject_zero=True,
            ),
            "openpi_action_engine_sha256": _require_sha256(
                raw.get("openpi_action_engine_sha256"),
                "action engine",
                reject_zero=True,
            ),
        }


plugin = OpenPIPlugin()
