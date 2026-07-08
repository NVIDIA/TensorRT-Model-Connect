# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest loader — load, validate, and normalize E2E model manifests.

Reads unified per-model JSON manifests from tests/e2e/models/ and model-owned
tests/e2e/models/<family>/manifests/ directories. Each manifest describes one
buildable model and contains one or more normalized E2E testcases.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path, PurePosixPath
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from .contracts import (
    CILane,
    E2ECase,
    E2EModel,
    OracleLevel,
    PreflightRequirement,
    StageSpec,
)
from .python_profiles import PROFILE_PHASES, normalize_execution_profiles
from .runtime_strategy_metadata import runtime_strategy_task_strategy

logger = logging.getLogger(__name__)

# Default manifest directory (relative to project root)
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "e2e" / "models"

_THRESHOLD_SIDECAR_FIELDS = frozenset(
    {
        "threshold_overrides",
        "logit_atol",
        "layer_atol",
        "min_pixel_agreement",
        "min_pixel_mean",
        "max_pixel_mean",
        "min_pixel_std",
        "reference_min_pixel_std_for_ratio",
        "min_reference_std_ratio",
        "speech_min_token_match",
        "speech_min_frame_exact",
        "speech_min_rms",
    }
)

_MODEL_ASSET_FIELDS = frozenset(
    {
        "test_image",
        "test_input_audio",
        "speech_reference_tokens",
        "golden_snapshot_path",
        "edit_condition_image",
        "fp8_scales",
        "elf_replay_artifact",
        "upstream_replay_artifact",
    }
)


def _model_test_dir_from_manifest_path(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "manifests":
        return manifest_path.parent.parent
    return manifest_path.parent


def _resolve_model_asset_path(value: str, model_test_dir: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return value

    posix = PurePosixPath(value)
    if posix.parts[:3] == ("tests", "e2e", "data"):
        return str(model_test_dir / "data" / posix.name)
    if posix.parts and posix.parts[0] == "data":
        return str(model_test_dir / Path(*posix.parts))

    candidate = model_test_dir / "data" / value
    if candidate.is_file():
        return str(candidate)
    return value


def _resolve_model_asset_paths(value: Any, model_test_dir: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            item_key: _resolve_model_asset_paths(item_value, model_test_dir, item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_resolve_model_asset_paths(item, model_test_dir, key) for item in value]
    if isinstance(value, str) and key in _MODEL_ASSET_FIELDS:
        return _resolve_model_asset_path(value, model_test_dir)
    return value


def _resolve_preflight_asset_paths(manifest: dict[str, Any], model_test_dir: Path) -> None:
    requirements = manifest.get("preflight_requirements")
    if not isinstance(requirements, list):
        return
    for requirement in requirements:
        if not isinstance(requirement, dict) or requirement.get("kind") != "asset_exists":
            continue
        args = requirement.get("args")
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if isinstance(path, str):
            args["path"] = _resolve_model_asset_path(path, model_test_dir)


def _read_model_index(index_path: Path) -> dict[str, Any]:
    text = index_path.read_text(encoding="utf-8")
    if tomllib is not None:
        return tomllib.loads(text)

    entries: list[str] = []
    in_array = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if not in_array:
            if not line.startswith("test_manifests"):
                continue
            _, value = line.split("=", 1)
            line = value.strip()
            if not line.startswith("["):
                raise ValueError(f"{index_path}: test_manifests must be an array")
            line = line[1:]
            in_array = True
        if in_array:
            if "]" in line:
                line = line.split("]", 1)[0]
                in_array = False
            entries.extend(re.findall(r'"([^"]+)"', line))
    return {"test_manifests": entries}


def _manifest_paths_from_model_index(index_path: Path) -> list[Path]:
    """Return manifest paths declared by tests/e2e/models/<family>/MODEL.toml."""
    raw = _read_model_index(index_path)
    manifest_entries = raw.get("test_manifests", [])
    if not isinstance(manifest_entries, list):
        raise TypeError(f"{index_path}: test_manifests must be a list")

    paths: list[Path] = []
    for entry in manifest_entries:
        if not isinstance(entry, str):
            raise TypeError(f"{index_path}: test_manifests entries must be strings")
        rel = PurePosixPath(entry.replace("\\", "/"))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"{index_path}: invalid manifest path {entry!r}")
        manifest_path = index_path.parent / Path(*rel.parts)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{index_path}: missing manifest {entry!r}")
        paths.append(manifest_path)
    return paths


def _threshold_sidecar_path(
    manifest_path: Path,
    testcase_name: str | None = None,
) -> Path:
    """Return the model-local threshold sidecar path for a testcase."""
    filename = f"{testcase_name}.json" if testcase_name else manifest_path.name
    if manifest_path.parent.name == "manifests":
        return manifest_path.parent.parent / "thresholds" / filename
    return manifest_path.parent / "thresholds" / filename


def _load_threshold_sidecar(
    manifest_path: Path,
    testcase_name: str | None = None,
) -> dict[str, Any]:
    """Load tests/e2e/models/<family>/thresholds/<case>.json if present."""
    sidecar_path = _threshold_sidecar_path(manifest_path, testcase_name)
    if not sidecar_path.is_file():
        return {}

    try:
        raw = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sidecar_path}: invalid threshold JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise TypeError(f"{sidecar_path}: threshold sidecar must be an object")

    unknown = sorted(set(raw) - _THRESHOLD_SIDECAR_FIELDS)
    if unknown:
        raise ValueError(f"{sidecar_path}: unsupported threshold sidecar field(s): {unknown}")

    overrides = raw.get("threshold_overrides", {})
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError(f"{sidecar_path}: threshold_overrides must be an object")

    for key, value in raw.items():
        if key == "threshold_overrides":
            for metric in value:
                if not isinstance(metric, str):
                    raise TypeError(f"{sidecar_path}: threshold_overrides keys must be strings")
        elif not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{sidecar_path}: {key} must be numeric")

    return raw


def _merge_threshold_sidecar(
    raw: dict[str, Any],
    manifest_path: Path,
    testcase_name: str | None = None,
) -> dict[str, Any]:
    sidecar = _load_threshold_sidecar(manifest_path, testcase_name)
    if not sidecar:
        return raw

    merged = raw.copy()
    sidecar_overrides = sidecar.get("threshold_overrides", {})
    raw_overrides = merged.get("threshold_overrides", {})
    if raw_overrides and not isinstance(raw_overrides, dict):
        raise TypeError(f"{manifest_path}: threshold_overrides must be an object")

    merged.update({key: value for key, value in sidecar.items() if key != "threshold_overrides"})
    if sidecar_overrides:
        merged["threshold_overrides"] = {
            **raw_overrides,
            **sidecar_overrides,
        }
    return merged


def iter_manifest_paths(models_dir: str | Path | None = None) -> list[Path]:
    """Return E2E manifest paths from indexed, flat, and nested layouts."""
    if models_dir is None:
        models_dir = _DEFAULT_MODELS_DIR

    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []

    paths = set(models_dir.glob("*.json"))
    direct_index = models_dir / "MODEL.toml"
    if direct_index.is_file():
        paths.update(_manifest_paths_from_model_index(direct_index))
    else:
        paths.update(models_dir.glob("manifests/*.json"))

    indexed_model_dirs: set[Path] = set()
    for index_path in sorted(models_dir.glob("*/MODEL.toml")):
        indexed_model_dirs.add(index_path.parent)
        paths.update(_manifest_paths_from_model_index(index_path))
    for manifest_path in models_dir.glob("*/manifests/*.json"):
        if manifest_path.parent.parent not in indexed_model_dirs:
            paths.add(manifest_path)
    return sorted(paths)


def find_manifest_path(
    manifest_name: str,
    models_dir: str | Path | None = None,
) -> Path | None:
    """Find a manifest by file name or case name across supported layouts."""
    target = manifest_name
    if not target.endswith(".json"):
        target = f"{target}.json"

    for manifest_path in iter_manifest_paths(models_dir):
        if manifest_path.name == target or manifest_path.stem == manifest_name:
            return manifest_path
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        testcases = raw.get("testcases", []) if isinstance(raw, dict) else []
        if any(
            isinstance(testcase, dict) and testcase.get("name") == manifest_name
            for testcase in testcases
        ):
            return manifest_path
    return None


# ---------------------------------------------------------------------------
# v1 -> v2 field inference
# ---------------------------------------------------------------------------


def _infer_task_strategy(manifest: dict) -> str:
    """Return the model-owned task_strategy field when declared."""
    if "task_strategy" in manifest:
        return str(manifest["task_strategy"])
    runtime_strategy = str(manifest.get("runtime_strategy") or "")
    if not runtime_strategy:
        return ""
    return runtime_strategy_task_strategy(runtime_strategy) or runtime_strategy


def _model_e2e_defaults(manifest_path: Path, task_strategy: str) -> dict[str, Any]:
    """Return family-owned E2E defaults for the manifest's task strategy."""
    index_path = _model_test_dir_from_manifest_path(manifest_path) / "MODEL.toml"
    if not index_path.is_file():
        return {}
    raw = _read_model_index(index_path)
    defaults = raw.get("e2e_defaults", {})
    if not isinstance(defaults, dict):
        return {}
    task_defaults = defaults.get(task_strategy, {})
    return task_defaults if isinstance(task_defaults, dict) else {}


def _infer_reference_backend(manifest: dict, defaults: dict[str, Any]) -> str:
    """Return the model-owned reference_backend."""
    if "reference_backend" in manifest:
        return manifest["reference_backend"]
    value = defaults.get("reference_backend", "")
    return str(value) if value else ""


def _infer_oracle_level(manifest: dict, defaults: dict[str, Any]) -> str:
    """Return the model-owned oracle_level."""
    if "oracle_level" in manifest:
        return manifest["oracle_level"]
    value = defaults.get("oracle_level", "")
    return str(value) if value else OracleLevel.L2_INTERNAL_REFERENCE.value


def _infer_reference_family(manifest: dict) -> str:
    """Return the model-owned reference_family field when declared."""
    return str(manifest.get("reference_family", "") or "")


def _infer_user_contract(manifest: dict, reference_family: str) -> str:
    """Return the model-owned user_contract field when declared."""
    return str(manifest.get("user_contract", "") or "")


def _infer_ci_lane(manifest: dict) -> str:
    """Infer ci_lane from manifest or default to acceptance."""
    if "ci_lane" in manifest:
        return manifest["ci_lane"]
    return CILane.ACCEPTANCE.value


def _preflight_requirements(raw_requirements: Any) -> list[PreflightRequirement]:
    if not isinstance(raw_requirements, list):
        return []
    return [
        PreflightRequirement(
            kind=req["kind"],
            args=req.get("args", {}),
            gating=req.get("gating", True),
        )
        for req in raw_requirements
        if isinstance(req, dict) and "kind" in req
    ]


def _build_preflight(manifest: dict, defaults: dict[str, Any]) -> list[PreflightRequirement]:
    """Build preflight requirements from manifest or infer defaults."""
    if "preflight_requirements" in manifest:
        return _preflight_requirements(manifest["preflight_requirements"])

    # Auto-generate preflight from manifest fields
    reqs: list[PreflightRequirement] = _preflight_requirements(
        defaults.get("preflight_requirements", [])
    )

    # All models need the binary
    reqs.append(PreflightRequirement(kind="binary_exists", args={}, gating=True))

    asset_fields = defaults.get("preflight_asset_fields", [])
    if "preflight_asset_fields" in manifest:
        asset_fields = manifest["preflight_asset_fields"]
    if isinstance(asset_fields, list):
        for field in asset_fields:
            if isinstance(field, str) and manifest.get(field):
                reqs.append(
                    PreflightRequirement(
                        kind="asset_exists",
                        args={"path": manifest[field]},
                        gating=True,
                    )
                )

    # Gated models require either authentication or a complete-enough local
    # snapshot that Hugging Face can resolve in offline mode.  Isolated model
    # proofs deliberately do not receive secrets or networking, so a warmed
    # cache is the authorization-independent path used there.
    if manifest.get("gated"):
        reqs.append(
            PreflightRequirement(
                kind="hf_auth_token_present",
                args={"hf_id": str(manifest.get("hf_id") or "")},
                gating=True,
            )
        )
    elif manifest.get("trust_remote_code"):
        reqs.append(
            PreflightRequirement(
                kind="hf_auth_token_present",
                args={"hf_id": str(manifest.get("hf_id") or "")},
                gating=False,
            )
        )

    return reqs


def _stage_specs(raw_stages: Any) -> list[StageSpec]:
    if not isinstance(raw_stages, list):
        return []
    return [
        StageSpec(
            name=s["name"],
            required=s.get("required", True),
            runner_override=s.get("runner_override"),
            comparator_override=s.get("comparator_override"),
            artifact_type=s.get("artifact_type", ""),
            comparison_mode=s.get("comparison_mode", ""),
            ci_lanes=s.get("ci_lanes", [CILane.ACCEPTANCE.value]),
        )
        for s in raw_stages
        if isinstance(s, dict) and "name" in s
    ]


def _build_stages(manifest: dict, defaults: dict[str, Any]) -> list[StageSpec]:
    """Build stage specs from manifest or infer defaults."""
    if "stages" in manifest:
        return _stage_specs(manifest["stages"])

    stages = _stage_specs(defaults.get("stages", []))
    if stages:
        return stages
    return [StageSpec(name="full_generation", required=True)]


def _build_inputs(manifest: dict, defaults: dict[str, Any]) -> dict:
    """Extract input specification from manifest."""
    inputs: dict[str, Any] = {}

    explicit_inputs = manifest.get("inputs")
    if isinstance(explicit_inputs, dict):
        inputs.update(explicit_inputs)

    # Text prompt
    prompt = manifest.get("prompt") or manifest.get("test_prompt", "")
    if prompt:
        inputs["prompt"] = prompt

    input_fields = defaults.get("input_fields", [])
    if "input_fields" in manifest:
        input_fields = manifest["input_fields"]
    if isinstance(input_fields, list):
        for mapping in input_fields:
            if not isinstance(mapping, dict):
                continue
            input_name = mapping.get("input")
            manifest_field = mapping.get("manifest")
            required_manifest = mapping.get("required_manifest")
            if (
                not isinstance(input_name, str)
                or not isinstance(manifest_field, str)
                or input_name in inputs
            ):
                continue
            if isinstance(required_manifest, str) and required_manifest not in manifest:
                continue
            if isinstance(required_manifest, list) and any(
                not isinstance(field, str) or field not in manifest for field in required_manifest
            ):
                continue
            if manifest_field in manifest:
                inputs[input_name] = manifest[manifest_field]
            elif "default" in mapping:
                inputs[input_name] = mapping["default"]

    # Max new tokens
    inputs["max_new_tokens"] = manifest.get("max_new_tokens", 30)
    inputs["max_cache_length"] = manifest.get(
        "max_cache_length",
        manifest.get("build_args", {}).get("max_cache_length", 256),
    )

    # Generation parameters (optional, default to each runtime's configured value)
    for key in ("temperature", "top_p", "top_k", "min_p", "seed"):
        if key in manifest:
            inputs[key] = manifest[key]

    return inputs


def _build_threshold_overrides(manifest: dict) -> dict:
    """Extract threshold overrides from manifest."""
    overrides: dict[str, Any] = {}

    if "threshold_overrides" in manifest:
        overrides.update(manifest["threshold_overrides"])

    # v1 compat: map specific fields to threshold overrides
    if "logit_atol" in manifest:
        overrides["logit_atol"] = manifest["logit_atol"]
    if "layer_atol" in manifest:
        overrides["layer_atol"] = manifest["layer_atol"]
    if "min_pixel_agreement" in manifest:
        overrides["min_pixel_agreement"] = manifest["min_pixel_agreement"]
    if "min_pixel_mean" in manifest:
        overrides["min_pixel_mean"] = manifest["min_pixel_mean"]
    if "max_pixel_mean" in manifest:
        overrides["max_pixel_mean"] = manifest["max_pixel_mean"]
    if "min_pixel_std" in manifest:
        overrides["min_pixel_std"] = manifest["min_pixel_std"]
    if "reference_min_pixel_std_for_ratio" in manifest:
        overrides["reference_min_pixel_std_for_ratio"] = manifest[
            "reference_min_pixel_std_for_ratio"
        ]
    if "min_reference_std_ratio" in manifest:
        overrides["min_reference_std_ratio"] = manifest["min_reference_std_ratio"]
    if "speech_min_token_match" in manifest:
        overrides["speech_min_token_match"] = manifest["speech_min_token_match"]
    if "speech_min_frame_exact" in manifest:
        overrides["speech_min_frame_exact"] = manifest["speech_min_frame_exact"]
    if "speech_min_rms" in manifest:
        overrides["speech_min_rms"] = manifest["speech_min_rms"]
    if "num_expected_masks" in manifest:
        overrides["num_expected_masks"] = manifest["num_expected_masks"]

    return overrides


def _build_determinism(manifest: dict) -> dict:
    """Extract determinism settings from manifest."""
    if "determinism" in manifest:
        return manifest["determinism"]

    return {
        "seed": manifest.get("seed", 42),
        "reruns": manifest.get("determinism_reruns", 0),
    }


def _build_execution_profiles(
    manifest: dict,
    *,
    reference_backend: str = "",
) -> dict[str, str]:
    """Extract execution profile selections from the manifest."""
    return normalize_execution_profiles(
        manifest.get("execution_profiles"),
        family=str(manifest.get("family", "") or ""),
        runtime_strategy=str(manifest.get("runtime_strategy", "") or ""),
        reference_backend=reference_backend or str(manifest.get("reference_backend", "") or ""),
    )


def _build_metadata(manifest: dict, defaults: dict[str, Any]) -> dict:
    """Collect all non-standard fields into metadata."""
    standard_fields = {
        "name",
        "hf_id",
        "model_id",
        "bundle",
        "family",
        "runtime_strategy",
        "task_strategy",
        "reference_backend",
        "oracle_level",
        "prompt",
        "test_prompt",
        "max_new_tokens",
        "max_cache_length",
        "precision",
        "reference_precision",
        "fp32_layers",
        "quantization",
        "logit_atol",
        "layer_atol",
        "trust_remote_code",
        "skip",
        "skip_comparison",
        "test_image",
        "test_input_audio",
        "speech_reference_tokens",
        "speech_test_max_frames",
        "speech_min_token_match",
        "speech_min_frame_exact",
        "speech_min_rms",
        "language",
        "reference_transcript_file",
        "streaming",
        "point_x",
        "point_y",
        "num_expected_masks",
        "min_pixel_agreement",
        "min_pixel_mean",
        "max_pixel_mean",
        "min_pixel_std",
        "reference_min_pixel_std_for_ratio",
        "min_reference_std_ratio",
        "video_num_frames",
        "video_height",
        "video_width",
        "num_inference_steps",
        "build_args",
        "build_cli_args",
        "preflight_requirements",
        "preflight_asset_fields",
        "input_fields",
        "stages",
        "comparison_profile",
        "threshold_overrides",
        "determinism",
        "inputs",
        "metadata",
        "reference_family",
        "user_contract",
        "ci_lane",
        "execution_profiles",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "negative_prompt",
        "cfg_scale",
        "height",
        "width",
        "image_height",
        "image_width",
    }

    meta = manifest.get("metadata", {}).copy()
    for k, v in manifest.items():
        if k not in standard_fields:
            meta[k] = v

    # Explicitly propagate trust_remote_code to metadata so reference runners
    # can access it via case.metadata["trust_remote_code"].
    if "trust_remote_code" in manifest:
        meta["trust_remote_code"] = manifest["trust_remote_code"]

    # Propagate precision to metadata so the orchestrator can pass it to
    # the trtmc build CLI when building bundles.
    if "precision" in manifest:
        meta["precision"] = manifest["precision"]
    if "reference_precision" in manifest:
        meta["reference_precision"] = manifest["reference_precision"]
    if "fp32_layers" in manifest:
        meta["fp32_layers"] = manifest["fp32_layers"]

    if "quantization" in manifest:
        meta["quantization"] = manifest["quantization"]

    # Propagate build_args so the orchestrator can select the correct backend.
    if "build_args" in manifest:
        meta["build_args"] = manifest["build_args"]

    if "build_cli_args" in defaults:
        meta["build_cli_args"] = defaults["build_cli_args"]
    if "build_cli_args" in manifest:
        meta["build_cli_args"] = manifest["build_cli_args"]

    # Speech-to-text multilingual: language tag (e.g. "es-ES") routes the
    # nemotron-3.5 prompt_kernel via case.metadata["language"]. Reference
    # transcript file path lets ASR comparators load a checked-in golden when
    # no live reference backend produces a transcript.
    if "language" in manifest:
        meta["language"] = manifest["language"]
    if "reference_transcript_file" in manifest:
        meta["reference_transcript_file"] = manifest["reference_transcript_file"]

    # Streaming-mode ASR config (cache-aware streaming-only checkpoints like
    # nemotron-3.5 require chunked decode; the non-streaming `trtmc transcribe`
    # path produces garbage for them). Block shape:
    #   {"enabled": true, "chunk_ms": 1120, "att_context_size": [56, 13]}
    if "streaming" in manifest:
        meta["streaming"] = manifest["streaming"]

    return meta


def _convert_skip_to_known_limitation(manifest: dict) -> dict | None:
    """Convert v1 skip field to known_limitations entry."""
    skip_reason = manifest.get("skip")
    if skip_reason:
        return {
            "reason": skip_reason,
            "source": "v1_skip_migration",
        }
    return None


# ---------------------------------------------------------------------------
# Manifest schema validation
# ---------------------------------------------------------------------------

_LEGACY_RUNTIME_STRATEGY_ALIASES = frozenset(
    {
        "diffusion",
        "text_to_audio",
    }
)
_KNOWN_RUNTIME_STRATEGIES_CACHE: frozenset[str] | None = None


def _runtime_model_manifests_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "runtime" / "models"


def _read_runtime_model_manifest(path: Path) -> dict[str, Any]:
    if tomllib is not None:
        with path.open("rb") as f:
            return tomllib.load(f)

    parsed: dict[str, Any] = {}
    text = path.read_text(encoding="utf-8")
    single = re.search(r'(?m)^\s*runtime_strategy\s*=\s*"([^"]+)"', text)
    if single:
        parsed["runtime_strategy"] = single.group(1)
    multi = re.search(r"(?ms)^\s*runtime_strategies\s*=\s*\[([^\]]*)\]", text)
    if multi:
        parsed["runtime_strategies"] = re.findall(r'"([^"]+)"', multi.group(1))
    return parsed


def _known_runtime_strategies() -> frozenset[str]:
    global _KNOWN_RUNTIME_STRATEGIES_CACHE
    if _KNOWN_RUNTIME_STRATEGIES_CACHE is not None:
        return _KNOWN_RUNTIME_STRATEGIES_CACHE

    strategies = set(_LEGACY_RUNTIME_STRATEGY_ALIASES)
    for manifest in sorted(_runtime_model_manifests_dir().glob("*/MODEL.toml")):
        try:
            raw = _read_runtime_model_manifest(manifest)
        except Exception as exc:
            logger.warning("Failed to read runtime model manifest %s: %s", manifest, exc)
            continue
        values = raw.get("runtime_strategies")
        if values is None:
            values = [raw.get("runtime_strategy")]
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        strategies.update(value for value in values if isinstance(value, str) and value)

    _KNOWN_RUNTIME_STRATEGIES_CACHE = frozenset(strategies)
    return _KNOWN_RUNTIME_STRATEGIES_CACHE


def _validate_manifest(raw: dict, path: str) -> None:
    """Validate manifest schema before loading.

    Raises:
        ValueError: If required fields are missing.
        TypeError: If typed fields have wrong types.

    Warns on unknown runtime_strategy values.
    """
    # 1. 'name' is always required
    if "name" not in raw:
        raise ValueError(f"Manifest {path!r} is missing required field 'name'")

    # 2. When not skipped, hf_id and family are required
    if not raw.get("skip"):
        if "hf_id" not in raw and "model_id" not in raw:
            raise ValueError(
                f"Manifest {path!r} (name={raw['name']!r}) is missing "
                f"required field 'hf_id' (and no 'skip' is set)"
            )
        if "family" not in raw:
            raise ValueError(
                f"Manifest {path!r} (name={raw['name']!r}) is missing "
                f"required field 'family' (and no 'skip' is set)"
            )
        if "runtime_strategy" not in raw:
            raise ValueError(
                f"Manifest {path!r} (name={raw['name']!r}) is missing "
                f"required field 'runtime_strategy' (and no 'skip' is set)"
            )

    # 3. Type checks for int fields
    for field_name in ("max_new_tokens", "max_cache_length"):
        if field_name in raw:
            val = raw[field_name]
            if not isinstance(val, int) or isinstance(val, bool):
                raise TypeError(
                    f"Manifest {path!r}: '{field_name}' must be int, "
                    f"got {type(val).__name__} ({val!r})"
                )

    # 4. Warn on unknown runtime_strategy
    rs = raw.get("runtime_strategy")
    known_runtime_strategies = _known_runtime_strategies()
    if rs is not None and rs not in known_runtime_strategies:
        warnings.warn(
            f"Manifest {path!r} (name={raw.get('name')!r}): unknown "
            f"runtime_strategy {rs!r}. Known values: "
            f"{sorted(known_runtime_strategies)}",
            stacklevel=2,
        )

    execution_profiles = raw.get("execution_profiles")
    if execution_profiles is not None:
        if not isinstance(execution_profiles, dict):
            raise TypeError(
                f"Manifest {path!r}: 'execution_profiles' must be an object, "
                f"got {type(execution_profiles).__name__}"
            )
        for phase, profile in execution_profiles.items():
            if phase not in PROFILE_PHASES:
                raise ValueError(
                    f"Manifest {path!r}: execution_profiles contains unsupported "
                    f"phase {phase!r}; expected one of {PROFILE_PHASES}"
                )
            if not isinstance(profile, str) or not profile.strip():
                raise TypeError(
                    f"Manifest {path!r}: execution_profiles[{phase!r}] must be a non-empty string"
                )

    for precision_field in ("precision", "reference_precision"):
        precision = raw.get(precision_field)
        if precision is not None and precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(
                f"Manifest {path!r}: {precision_field} must be one of 'fp32', 'fp16', or 'bf16'"
            )

    fp32_layers = raw.get("fp32_layers")
    if fp32_layers is not None:
        if not isinstance(fp32_layers, list) or any(
            not isinstance(layer, int) or isinstance(layer, bool) or layer < 0
            for layer in fp32_layers
        ):
            raise TypeError(
                f"Manifest {path!r}: fp32_layers must be a list of non-negative integer indices"
            )
        if len(fp32_layers) != len(set(fp32_layers)):
            raise ValueError(f"Manifest {path!r}: fp32_layers must not contain duplicates")

    build_timeout_s = raw.get("build_timeout_s")
    if build_timeout_s is not None and (
        not isinstance(build_timeout_s, int)
        or isinstance(build_timeout_s, bool)
        or build_timeout_s <= 0
    ):
        raise TypeError(
            f"Manifest {path!r}: build_timeout_s must be a positive integer"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_MODEL_ONLY_FIELDS = frozenset(
    {
        "hf_id",
        "model_id",
        "bundle",
        "family",
        "runtime_strategy",
        "task_strategy",
        "max_cache_length",
        "precision",
        "fp32_layers",
        "quantization",
        "fp8_scales",
        "trust_remote_code",
        "build_args",
        "build_env",
        "build_timeout_s",
        "e2e_parallel_resource",
        "e2e_size",
        "distributed_runtime",
    }
)


def _load_case(raw: dict[str, Any], path: Path, model_name: str) -> E2ECase:
    testcase_name = str(raw.get("name", "") or "")
    raw = _merge_threshold_sidecar(raw, path, testcase_name)
    model_test_dir = _model_test_dir_from_manifest_path(path)
    raw = _resolve_model_asset_paths(raw, model_test_dir)
    _resolve_preflight_asset_paths(raw, model_test_dir)

    _validate_manifest(raw, str(path))

    task_strategy = _infer_task_strategy(raw)
    e2e_defaults = _model_e2e_defaults(path, task_strategy)
    reference_backend = _infer_reference_backend(raw, e2e_defaults)
    oracle_level = _infer_oracle_level(raw, e2e_defaults)

    # Infer reference family, user contract, and CI lane
    reference_family = _infer_reference_family(raw)
    user_contract = _infer_user_contract(raw, reference_family)
    ci_lane = _infer_ci_lane(raw)

    # Handle skip -> known_limitations migration
    known_limitation = _convert_skip_to_known_limitation(raw)
    metadata = _build_metadata(raw, e2e_defaults)
    metadata["model_name"] = model_name
    metadata["manifest_path"] = str(path)
    metadata["model_test_dir"] = str(model_test_dir)
    if known_limitation:
        metadata["known_limitations"] = [known_limitation]
        metadata["skip_reason"] = raw["skip"]

    # Partial skip: run TRT but skip HF reference/comparison.
    if raw.get("skip_comparison"):
        reason = raw["skip_comparison"]
        if isinstance(reason, bool):
            reason = "comparison skipped"
        metadata["skip_comparison_reason"] = reason

    return E2ECase(
        name=raw["name"],
        hf_id=raw.get("hf_id", raw.get("model_id", "")),
        family=raw.get("family", ""),
        runtime_strategy=str(raw.get("runtime_strategy") or ""),
        task_strategy=task_strategy,
        reference_backend=reference_backend,
        oracle_level=oracle_level,
        reference_family=reference_family,
        user_contract=user_contract,
        ci_lane=ci_lane,
        bundle=raw.get("bundle", f"{raw['name']}.trtfb"),
        inputs=_build_inputs(raw, e2e_defaults),
        preflight=_build_preflight(raw, e2e_defaults),
        stages=_build_stages(raw, e2e_defaults),
        comparison_profile=raw.get("comparison_profile", "default"),
        threshold_overrides=_build_threshold_overrides(raw),
        determinism=_build_determinism(raw),
        execution_profiles=_build_execution_profiles(
            raw,
            reference_backend=reference_backend,
        ),
        metadata=metadata,
    )


def load_model_manifest(manifest_path: str | Path) -> E2EModel:
    """Load one buildable model and all testcases declared by its manifest."""
    path = Path(manifest_path)
    with path.open(encoding="utf-8") as manifest_file:
        raw = json.load(manifest_file)
    if not isinstance(raw, dict):
        raise TypeError(f"Manifest {str(path)!r} must contain an object")

    model_name = raw.get("name")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"Manifest {str(path)!r} is missing required field 'name'")
    testcases = raw.get("testcases")
    if not isinstance(testcases, list) or not testcases:
        raise ValueError(
            f"Manifest {str(path)!r} (name={model_name!r}) must contain a "
            "non-empty 'testcases' array"
        )

    model_defaults = {key: value for key, value in raw.items() if key != "testcases"}
    loaded_cases: list[E2ECase] = []
    seen_case_names: set[str] = set()
    for index, testcase in enumerate(testcases):
        if not isinstance(testcase, dict):
            raise TypeError(f"Manifest {str(path)!r}: testcases[{index}] must be an object")
        misplaced = sorted(_MODEL_ONLY_FIELDS.intersection(testcase))
        if misplaced:
            raise ValueError(
                f"Manifest {str(path)!r}: testcases[{index}] contains "
                f"model-level field(s): {misplaced}"
            )
        case_name = testcase.get("name")
        if not isinstance(case_name, str) or not case_name:
            raise ValueError(
                f"Manifest {str(path)!r}: testcases[{index}] is missing a non-empty 'name'"
            )
        if case_name in seen_case_names:
            raise ValueError(f"Manifest {str(path)!r}: duplicate testcase name {case_name!r}")
        seen_case_names.add(case_name)
        case_raw = {**model_defaults, **testcase, "name": case_name}
        loaded_cases.append(_load_case(case_raw, path, model_name))

    build_case = next(
        (case for case in loaded_cases if case.name == model_name),
        loaded_cases[0],
    )
    return E2EModel(
        name=model_name,
        hf_id=build_case.hf_id,
        family=build_case.family,
        bundle=build_case.bundle,
        testcases=loaded_cases,
        manifest_path=str(path),
    )


def load_manifest(manifest_path: str | Path) -> E2ECase:
    """Load the canonical build case from one unified model manifest."""
    return load_model_manifest(manifest_path).build_case


def load_all_model_manifests(
    models_dir: str | Path | None = None,
    task_strategy_filter: str | None = None,
) -> list[E2EModel]:
    """Load buildable model manifests with their selected child testcases."""
    if models_dir is None:
        models_dir = _DEFAULT_MODELS_DIR

    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        logger.warning("Models directory not found: %s", models_dir)
        return []

    models: list[E2EModel] = []
    model_names: dict[str, Path] = {}
    bundle_names: dict[str, Path] = {}
    testcase_names: dict[str, Path] = {}
    for manifest_path in iter_manifest_paths(models_dir):
        try:
            model = load_model_manifest(manifest_path)
        except Exception as exc:
            logger.error("Failed to load manifest %s: %s", manifest_path, exc)
            continue

        if model.name in model_names:
            raise ValueError(
                f"Duplicate model name {model.name!r}: "
                f"{model_names[model.name]} and {manifest_path}"
            )
        if model.bundle in bundle_names:
            raise ValueError(
                f"Duplicate model bundle {model.bundle!r}: "
                f"{bundle_names[model.bundle]} and {manifest_path}"
            )
        for case in model.testcases:
            if case.name in testcase_names:
                raise ValueError(
                    f"Duplicate testcase name {case.name!r}: "
                    f"{testcase_names[case.name]} and {manifest_path}"
                )
            testcase_names[case.name] = manifest_path

        model_names[model.name] = manifest_path
        bundle_names[model.bundle] = manifest_path
        if task_strategy_filter:
            filtered_cases = [
                case for case in model.testcases if case.task_strategy == task_strategy_filter
            ]
            if not filtered_cases:
                continue
            model.testcases = filtered_cases
        models.append(model)

    return sorted(models, key=lambda model: model.name)


def load_all_manifests(
    models_dir: str | Path | None = None,
    task_strategy_filter: str | None = None,
) -> list[E2ECase]:
    """Load all model manifests from a directory.

    Args:
        models_dir: Directory containing flat *.json manifests or
            <family>/manifests/*.json manifests. Defaults to tests/e2e/models/.
        task_strategy_filter: If set, only return cases matching this
            task_strategy.

    Returns:
        List of E2ECase instances, sorted by name.
    """
    models = load_all_model_manifests(models_dir, task_strategy_filter)
    return sorted(
        (case for model in models for case in model.testcases),
        key=lambda case: case.name,
    )


def get_case_names(
    models_dir: str | Path | None = None,
    task_strategy_filter: str | None = None,
) -> list[str]:
    """Return child testcase names for compatibility callers."""
    cases = load_all_manifests(models_dir, task_strategy_filter)
    return [c.name for c in cases]


def get_case_by_name(
    name: str,
    models_dir: str | Path | None = None,
) -> E2ECase | None:
    """Look up a single case by name."""
    cases = load_all_manifests(models_dir)
    for c in cases:
        if c.name == name:
            return c
    return None


def get_model_by_name(
    name: str,
    models_dir: str | Path | None = None,
) -> E2EModel | None:
    """Look up one buildable model by its manifest name."""
    for model in load_all_model_manifests(models_dir):
        if model.name == name:
            return model
    return None
