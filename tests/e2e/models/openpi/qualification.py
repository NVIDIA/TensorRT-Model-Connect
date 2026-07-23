# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the pinned contracts and replay artifacts used by OpenPI E2E tests."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


UPSTREAM_REPOSITORY = "https://github.com/Physical-Intelligence/openpi.git"
UPSTREAM_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
SCHEMA_VERSION = 1

_ROOT = Path(__file__).resolve().parent
_CONTRACTS = {
    "pi05_droid": _ROOT / "contracts" / "pi05_droid.json",
}
_THRESHOLDS = {
    "pi05_droid": _ROOT / "thresholds" / "pi05-droid.json",
}
_DTYPE_BYTES = {
    "bool": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


class OpenPIQualificationError(ValueError):
    """Raised when a contract or replay artifact is invalid."""


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(key)
        result[key] = value
    return result


def strict_json_load(path: str | Path) -> dict[str, Any]:
    """Load a JSON object while rejecting duplicate keys and non-finite numbers."""

    source = Path(path)

    def reject_constant(value: str) -> None:
        raise OpenPIQualificationError(f"{source}: non-finite JSON number {value!r}")

    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except _DuplicateKeyError as error:
        raise OpenPIQualificationError(
            f"{source}: duplicate JSON object key {str(error)!r}"
        ) from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OpenPIQualificationError(f"Unable to read strict JSON {source}: {error}") from error
    if not isinstance(value, dict):
        raise OpenPIQualificationError(f"{source}: top-level JSON value must be an object")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OpenPIQualificationError(f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    required: set[str],
    *,
    optional: set[str] = frozenset(),
    label: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise OpenPIQualificationError(f"{label} is missing required fields: {missing}")
    unexpected = sorted(set(value) - required - optional)
    if unexpected:
        raise OpenPIQualificationError(f"{label} contains unexpected fields: {unexpected}")


def _require_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise OpenPIQualificationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise OpenPIQualificationError(f"{label} must be >= {minimum}, got {value}")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value == "0" * 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise OpenPIQualificationError(f"{label} must be a non-zero lowercase SHA-256")
    return value


def _safe_relative_path(value: Any, *, label: str, prefix: str | None = None) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise OpenPIQualificationError(f"{label} must be a non-empty string")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != value
        or (prefix is not None and relative.parts[:1] != (prefix,))
    ):
        suffix = f" under {prefix}/" if prefix else ""
        raise OpenPIQualificationError(f"{label} must be a canonical relative path{suffix}")
    return relative


def _hf_cache_blob_root(root: Path) -> Path | None:
    """Return the sibling blob store for a canonical HF snapshot path."""

    for snapshots_dir in root.parents:
        if snapshots_dir.name != "snapshots":
            continue
        relative = root.relative_to(snapshots_dir)
        revision = relative.parts[0] if relative.parts else ""
        if (
            len(revision) == 40
            and all(char in "0123456789abcdef" for char in revision)
            and snapshots_dir.parent.name.startswith("models--")
        ):
            blobs = snapshots_dir.parent / "blobs"
            return blobs.resolve(strict=True) if blobs.is_dir() else None
    return None


def _resolve_relative_file(root: Path, relative: PurePosixPath, *, label: str) -> Path:
    """Resolve a local file or a standard HF snapshot symlink without traversal."""

    try:
        resolved_root = root.resolve(strict=True)
        resolved = root.joinpath(*relative.parts).resolve(strict=True)
    except OSError as error:
        raise OpenPIQualificationError(f"{label} is missing") from error
    if not resolved.is_file():
        raise OpenPIQualificationError(f"{label} is not a file")
    if resolved_root in resolved.parents:
        return resolved
    blob_root = _hf_cache_blob_root(resolved_root)
    if blob_root is not None and blob_root in resolved.parents:
        return resolved
    raise OpenPIQualificationError(f"{label} escapes its artifact directory")


def load_contract(profile_name: str) -> dict[str, Any]:
    try:
        path = _CONTRACTS[profile_name]
    except KeyError as error:
        raise OpenPIQualificationError(f"Unsupported OpenPI profile {profile_name!r}") from error
    contract = strict_json_load(path)
    validate_contract(contract)
    return contract


def load_thresholds(profile_name: str) -> dict[str, Any]:
    try:
        path = _THRESHOLDS[profile_name]
    except KeyError as error:
        raise OpenPIQualificationError(f"Unsupported OpenPI profile {profile_name!r}") from error
    document = strict_json_load(path)
    _require_exact_keys(document, {"threshold_overrides"}, label="OpenPI thresholds")
    overrides = _object(document["threshold_overrides"], label="threshold_overrides")
    for name, value in overrides.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise OpenPIQualificationError(
                "threshold_overrides must contain numeric values keyed by strings"
            )
    return dict(overrides)


def validate_contract(contract: Mapping[str, Any]) -> None:
    _require_exact_keys(
        contract,
        {
            "schema_version",
            "profile_name",
            "variant",
            "upstream",
            "precision",
            "images",
            "prompt",
            "state",
            "actions",
            "prefix",
            "flow",
            "runtime",
        },
        label="OpenPI contract",
    )
    profile = contract["profile_name"]
    if profile not in _CONTRACTS:
        raise OpenPIQualificationError(f"Unknown contract profile {profile!r}")
    if contract["schema_version"] != SCHEMA_VERSION or contract["variant"] != "pi05_flow":
        raise OpenPIQualificationError("OpenPI contract must declare schema v1 and pi05_flow")

    upstream = _object(contract["upstream"], label="OpenPI contract upstream")
    _require_exact_keys(
        upstream,
        {"repository", "commit", "checkpoint_uri", "checkpoint_digest_policy"},
        label="OpenPI contract upstream",
    )
    expected_checkpoint = f"gs://openpi-assets/checkpoints/{profile}"
    if upstream != {
        "repository": UPSTREAM_REPOSITORY,
        "commit": UPSTREAM_COMMIT,
        "checkpoint_uri": expected_checkpoint,
        "checkpoint_digest_policy": "required_in_reference_artifact",
    }:
        raise OpenPIQualificationError("OpenPI contract does not use the pinned upstream")

    precision = _object(contract["precision"], label="OpenPI contract precision")
    if precision != {"network": "bfloat16", "sensitive_accumulation": "float32"}:
        raise OpenPIQualificationError("OpenPI precision contract was weakened")

    images = _object(contract["images"], label="OpenPI contract images")
    if images != {
        "order": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        "validity": [True, True, False],
        "shape_hwc": [224, 224, 3],
        "tokens_per_image": 256,
        "embedding_width": 2048,
    }:
        raise OpenPIQualificationError("OpenPI image contract does not match upstream")

    expected_actions = {"horizon": 15, "external_dim": 8}
    actions = _object(contract["actions"], label="OpenPI contract actions")
    if actions != {
        **expected_actions,
        "internal_dim": 32,
        "normalization": "quantile_q01_q99",
    }:
        raise OpenPIQualificationError(f"{profile} action contract does not match upstream")

    state = _object(contract["state"], label="OpenPI contract state")
    if state != {
        "external_dim": 8,
        "internal_dim": 32,
        "normalization": "quantile_q01_q99",
    }:
        raise OpenPIQualificationError("OpenPI state contract does not match upstream")

    prompt = _object(contract["prompt"], label="OpenPI contract prompt")
    if prompt != {
        "max_tokens": 200,
        "discrete_state_input": True,
        "vocab_size": 257152,
    }:
        raise OpenPIQualificationError("OpenPI prompt contract does not match upstream")

    prefix = _object(contract["prefix"], label="OpenPI contract prefix")
    if prefix != {
        "max_physical_tokens": 968,
        "layers": 18,
        "query_heads": 8,
        "kv_heads": 1,
        "head_dim": 256,
    }:
        raise OpenPIQualificationError("OpenPI prefix contract does not match upstream")

    flow = _object(contract["flow"], label="OpenPI contract flow")
    if flow != {
        "steps": 10,
        "initial_t": 1.0,
        "dt": -0.1,
        "external_noise_required_for_parity": True,
        "timestep_injection": "adaptive_rms_norm",
    }:
        raise OpenPIQualificationError("OpenPI flow contract does not match upstream")

    runtime = _object(contract["runtime"], label="OpenPI contract runtime")
    if runtime != {
        "language": "c++",
        "engine_api": "tensorrt",
        "onnx_allowed": False,
        "python_allowed": False,
        "additional_frameworks_allowed": False,
    }:
        raise OpenPIQualificationError("OpenPI runtime purity contract was weakened")


def _expected_tensor_contract(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    horizon = int(_object(contract["actions"], label="actions")["horizon"])
    external_dim = int(_object(contract["actions"], label="actions")["external_dim"])
    prefix = _object(contract["prefix"], label="prefix")
    flow_shape = [1, horizon, 32]
    expected: dict[str, dict[str, Any]] = {
        "initial_noise": {
            "stage": "preprocess",
            "role": "input",
            "dtype": "float32",
            "shape": flow_shape,
        },
        "token_ids": {
            "stage": "preprocess",
            "role": "intermediate",
            "dtype": "int32",
            "shape": [1, 200],
        },
        "token_mask": {
            "stage": "preprocess",
            "role": "intermediate",
            "dtype": "bool",
            "shape": [1, 200],
        },
        "preprocessed_images": {
            "stage": "preprocess",
            "role": "intermediate",
            "dtype": "float32",
            "shape": [1, 3, 224, 224, 3],
        },
        "image_mask": {
            "stage": "preprocess",
            "role": "intermediate",
            "dtype": "bool",
            "shape": [1, 3],
        },
        "normalized_state": {
            "stage": "preprocess",
            "role": "intermediate",
            "dtype": "float32",
            "shape": [1, 32],
        },
        "vision_tokens": {
            "stage": "vision",
            "role": "intermediate",
            "dtype": "bfloat16",
            "shape": [1, 3, 256, 2048],
        },
        "prefix_kv_cache": {
            "stage": "prefix",
            "role": "intermediate",
            "dtype": "bfloat16",
            "shape": [
                int(prefix["layers"]),
                2,
                1,
                int(prefix["max_physical_tokens"]),
                int(prefix["kv_heads"]),
                int(prefix["head_dim"]),
            ],
        },
        "normalized_actions": {
            "stage": "postprocess",
            "role": "output",
            "dtype": "float32",
            "shape": flow_shape,
        },
        "physical_actions": {
            "stage": "postprocess",
            "role": "output",
            "dtype": "float32",
            "shape": [1, horizon, external_dim],
        },
    }
    for step in range(11):
        expected[f"flow_state_{step:02d}"] = {
            "stage": "flow",
            "role": "intermediate",
            "dtype": "float32",
            "shape": flow_shape,
        }
    for step in range(10):
        expected[f"velocity_{step:02d}"] = {
            "stage": "flow",
            "role": "intermediate",
            "dtype": "float32",
            "shape": flow_shape,
        }
    return expected


def _tensor_byte_length(dtype: str, shape: Sequence[int]) -> int:
    try:
        width = _DTYPE_BYTES[dtype]
    except KeyError as error:
        raise OpenPIQualificationError(f"Unsupported tensor dtype {dtype!r}") from error
    elements = 1
    for index, dimension in enumerate(shape):
        elements *= _require_int(dimension, label=f"shape[{index}]", minimum=1)
    return elements * width


def _validate_tensor_descriptor(
    name: str,
    descriptor: Mapping[str, Any],
    *,
    artifact_root: Path,
    verify_payloads: bool,
) -> None:
    _require_exact_keys(
        descriptor,
        {"path", "stage", "role", "dtype", "shape", "byte_length", "sha256"},
        label=f"tensor {name!r}",
    )
    relative = _safe_relative_path(
        descriptor["path"], label=f"tensor {name!r}.path", prefix="tensors"
    )
    shape = descriptor["shape"]
    if not isinstance(shape, list) or not shape:
        raise OpenPIQualificationError(f"tensor {name!r}.shape must be a non-empty array")
    expected_bytes = _tensor_byte_length(str(descriptor["dtype"]), shape)
    byte_length = _require_int(
        descriptor["byte_length"], label=f"tensor {name!r}.byte_length", minimum=1
    )
    if byte_length != expected_bytes:
        raise OpenPIQualificationError(
            f"tensor {name!r} byte length {byte_length} does not match dtype/shape {expected_bytes}"
        )
    expected_sha = _require_sha256(descriptor["sha256"], label=f"tensor {name!r}.sha256")
    if not verify_payloads:
        return

    payload = _resolve_relative_file(
        artifact_root,
        relative,
        label=f"tensor {name!r} payload",
    )
    if payload.stat().st_size != byte_length:
        raise OpenPIQualificationError(
            f"tensor {name!r} payload size does not match its descriptor"
        )
    if sha256_file(payload) != expected_sha:
        raise OpenPIQualificationError(f"tensor {name!r} payload SHA-256 mismatch")


def validate_reference_artifact(
    artifact_path: str | Path,
    *,
    verify_payloads: bool = True,
) -> dict[str, Any]:
    path = Path(artifact_path)
    artifact = strict_json_load(path)
    _require_exact_keys(
        artifact,
        {
            "schema_version",
            "artifact_type",
            "profile_name",
            "contract_sha256",
            "upstream",
            "case",
            "fixed_external_noise",
            "tensors",
        },
        optional={"exporter"},
        label="OpenPI reference artifact",
    )
    if (
        artifact["schema_version"] != SCHEMA_VERSION
        or artifact["artifact_type"] != "openpi_pinned_reference"
    ):
        raise OpenPIQualificationError("Not an OpenPI pinned reference artifact v1")

    profile = str(artifact["profile_name"])
    contract = load_contract(profile)
    contract_path = _CONTRACTS[profile]
    if artifact["contract_sha256"] != sha256_file(contract_path):
        raise OpenPIQualificationError(
            "Reference artifact is not bound to the current profile contract"
        )

    upstream = _object(artifact["upstream"], label="reference upstream")
    _require_exact_keys(
        upstream,
        {"repository", "commit", "checkpoint", "tokenizer", "normalization"},
        label="reference upstream",
    )
    if upstream["repository"] != UPSTREAM_REPOSITORY or upstream["commit"] != UPSTREAM_COMMIT:
        raise OpenPIQualificationError("Reference artifact does not use the pinned OpenPI source")
    for asset_name in ("checkpoint", "tokenizer", "normalization"):
        asset = _object(upstream[asset_name], label=f"upstream.{asset_name}")
        _require_exact_keys(asset, {"uri", "sha256"}, label=f"upstream.{asset_name}")
        if not isinstance(asset["uri"], str) or not asset["uri"]:
            raise OpenPIQualificationError(f"upstream.{asset_name}.uri must be non-empty")
        _require_sha256(asset["sha256"], label=f"upstream.{asset_name}.sha256")
    checkpoint = _object(upstream["checkpoint"], label="upstream.checkpoint")
    if (
        checkpoint["uri"]
        != _object(contract["upstream"], label="contract upstream")["checkpoint_uri"]
    ):
        raise OpenPIQualificationError(
            "Reference checkpoint URI does not match the profile contract"
        )

    case = _object(artifact["case"], label="reference case")
    _require_exact_keys(case, {"id", "prompt"}, optional={"description"}, label="reference case")
    if not isinstance(case["id"], str) or not case["id"]:
        raise OpenPIQualificationError("Reference case id must be non-empty")
    if not isinstance(case["prompt"], str):
        raise OpenPIQualificationError("Reference prompt must be a string")

    if artifact["fixed_external_noise"] != {
        "tensor": "initial_noise",
        "provided_by_caller": True,
        "rng_backend": "external",
    }:
        raise OpenPIQualificationError("Reference artifact must use caller-supplied initial_noise")

    tensors = _object(artifact["tensors"], label="reference tensors")
    expected_tensors = _expected_tensor_contract(contract)
    missing = sorted(set(expected_tensors) - set(tensors))
    if missing:
        raise OpenPIQualificationError(
            f"Reference artifact is missing stagewise tensors: {missing}"
        )
    paths: set[str] = set()
    for name, descriptor_value in tensors.items():
        if not isinstance(name, str) or not name:
            raise OpenPIQualificationError("Reference tensor names must be non-empty strings")
        descriptor = _object(descriptor_value, label=f"tensor {name!r}")
        _validate_tensor_descriptor(
            name,
            descriptor,
            artifact_root=path.parent,
            verify_payloads=verify_payloads,
        )
        descriptor_path = str(descriptor["path"])
        if descriptor_path in paths:
            raise OpenPIQualificationError(f"Reference tensor path {descriptor_path!r} is reused")
        paths.add(descriptor_path)
        expected = expected_tensors.get(name)
        if expected is not None:
            for field in ("stage", "role", "dtype", "shape"):
                if descriptor[field] != expected[field]:
                    raise OpenPIQualificationError(
                        f"tensor {name!r}.{field}={descriptor[field]!r}; "
                        f"expected {expected[field]!r}"
                    )
    return artifact


def materialize_reference_artifact(
    capture_spec_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Copy a complete upstream capture into a hashed replay artifact."""

    spec_path = Path(capture_spec_path)
    spec = strict_json_load(spec_path)
    _require_exact_keys(
        spec,
        {"schema_version", "artifact_type", "profile_name", "upstream", "case", "tensors"},
        optional={"exporter"},
        label="OpenPI capture specification",
    )
    if spec["schema_version"] != SCHEMA_VERSION or (
        spec["artifact_type"] != "openpi_reference_capture_spec"
    ):
        raise OpenPIQualificationError("Capture specification must declare OpenPI schema v1")
    profile = str(spec["profile_name"])
    contract = load_contract(profile)
    expected = _expected_tensor_contract(contract)
    tensors = _object(spec["tensors"], label="capture tensors")
    missing = sorted(set(expected) - set(tensors))
    if missing:
        raise OpenPIQualificationError(f"Capture is missing stagewise tensors: {missing}")

    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise OpenPIQualificationError(f"Reference output directory is not empty: {destination}")
    tensor_dir = destination / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    output_tensors: dict[str, Any] = {}
    safe_name_chars = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    for name in sorted(tensors):
        if not isinstance(name, str) or not name or not set(name) <= safe_name_chars:
            raise OpenPIQualificationError(f"Capture tensor name is unsafe: {name!r}")
        capture = _object(tensors[name], label=f"capture tensor {name!r}")
        _require_exact_keys(
            capture,
            {"source", "stage", "role", "dtype", "shape"},
            label=f"capture tensor {name!r}",
        )
        source_text = capture["source"]
        if not isinstance(source_text, str) or not source_text:
            raise OpenPIQualificationError(f"Capture tensor {name!r}.source must be non-empty")
        source = Path(source_text)
        if not source.is_absolute():
            source = spec_path.parent / source
        if not source.is_file():
            raise OpenPIQualificationError(f"Capture tensor {name!r} is missing: {source}")
        shape = capture["shape"]
        if not isinstance(shape, list) or not shape:
            raise OpenPIQualificationError(
                f"Capture tensor {name!r}.shape must be a non-empty array"
            )
        expected_bytes = _tensor_byte_length(str(capture["dtype"]), shape)
        if source.stat().st_size != expected_bytes:
            raise OpenPIQualificationError(
                f"Capture tensor {name!r} has {source.stat().st_size} bytes; "
                f"expected {expected_bytes}"
            )
        target = tensor_dir / f"{name}.bin"
        shutil.copyfile(source, target)
        output_tensors[name] = {
            "path": f"tensors/{name}.bin",
            "stage": capture["stage"],
            "role": capture["role"],
            "dtype": capture["dtype"],
            "shape": shape,
            "byte_length": expected_bytes,
            "sha256": sha256_file(target),
        }

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "openpi_pinned_reference",
        "profile_name": profile,
        "contract_sha256": sha256_file(_CONTRACTS[profile]),
        "upstream": spec["upstream"],
        "case": spec["case"],
        "fixed_external_noise": {
            "tensor": "initial_noise",
            "provided_by_caller": True,
            "rng_backend": "external",
        },
        "tensors": output_tensors,
    }
    if "exporter" in spec:
        artifact["exporter"] = spec["exporter"]
    artifact_path = destination / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        validate_reference_artifact(artifact_path)
    except Exception:
        artifact_path.unlink(missing_ok=True)
        raise
    return artifact_path


def materialize_reference_set(
    artifact_paths: Sequence[str | Path],
    output_path: str | Path,
) -> Path:
    """Create a content-addressed index over independently replayable cases."""

    target = Path(output_path)
    if target.exists():
        raise OpenPIQualificationError(f"Reference-set output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    root = target.parent.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    profile: str | None = None
    seen_ids: set[str] = set()
    for artifact_value in artifact_paths:
        artifact_path = Path(artifact_value).resolve(strict=True)
        if root not in artifact_path.parents:
            raise OpenPIQualificationError(
                f"Reference case must be stored beneath the set directory: {artifact_path}"
            )
        artifact = validate_reference_artifact(artifact_path)
        artifact_profile = str(artifact["profile_name"])
        if profile is None:
            profile = artifact_profile
        elif profile != artifact_profile:
            raise OpenPIQualificationError("A reference set cannot mix OpenPI profiles")
        case_id = str(_object(artifact["case"], label="reference case")["id"])
        if case_id in seen_ids:
            raise OpenPIQualificationError(f"Duplicate reference case id {case_id!r}")
        seen_ids.add(case_id)
        entries.append(
            {
                "id": case_id,
                "path": artifact_path.relative_to(root).as_posix(),
                "sha256": sha256_file(artifact_path),
            }
        )
    if profile is None:
        raise OpenPIQualificationError("Reference set requires at least one case artifact")
    entries.sort(key=lambda item: item["id"])
    document = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "openpi_pinned_reference_set",
        "profile_name": profile,
        "contract_sha256": sha256_file(_CONTRACTS[profile]),
        "case_count": len(entries),
        "cases": entries,
    }
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        validate_reference_set(target)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def validate_reference_set(
    set_path: str | Path,
    *,
    verify_payloads: bool = True,
    minimum_cases: int = 1,
) -> dict[str, Any]:
    path = Path(set_path)
    document = strict_json_load(path)
    _require_exact_keys(
        document,
        {
            "schema_version",
            "artifact_type",
            "profile_name",
            "contract_sha256",
            "case_count",
            "cases",
        },
        label="OpenPI reference set",
    )
    if document["schema_version"] != SCHEMA_VERSION or (
        document["artifact_type"] != "openpi_pinned_reference_set"
    ):
        raise OpenPIQualificationError("Not an OpenPI pinned reference set v1")
    profile = str(document["profile_name"])
    load_contract(profile)
    if document["contract_sha256"] != sha256_file(_CONTRACTS[profile]):
        raise OpenPIQualificationError("Reference set is not bound to the current profile contract")
    cases = document["cases"]
    if not isinstance(cases, list):
        raise OpenPIQualificationError("Reference set cases must be an array")
    case_count = _require_int(document["case_count"], label="reference set case_count", minimum=1)
    required_count = _require_int(minimum_cases, label="minimum_cases", minimum=1)
    if case_count != len(cases):
        raise OpenPIQualificationError("Reference set case_count does not match its case array")
    if case_count < required_count:
        raise OpenPIQualificationError(
            f"Reference set contains {case_count} cases; qualification requires {required_count}"
        )

    root = path.parent.resolve(strict=True)
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, entry_value in enumerate(cases):
        entry = _object(entry_value, label=f"reference set cases[{index}]")
        _require_exact_keys(entry, {"id", "path", "sha256"}, label=f"reference set cases[{index}]")
        case_id = entry["id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen_ids:
            raise OpenPIQualificationError(f"Invalid or duplicate reference case id {case_id!r}")
        relative = _safe_relative_path(entry["path"], label=f"reference set cases[{index}].path")
        relative_text = relative.as_posix()
        if relative_text in seen_paths:
            raise OpenPIQualificationError(
                f"Unsafe or duplicate reference case path {relative_text!r}"
            )
        logical_artifact_path = root.joinpath(*relative.parts)
        artifact_path = _resolve_relative_file(
            root,
            relative,
            label=f"Reference case artifact {relative_text}",
        )
        expected_sha = _require_sha256(
            entry["sha256"], label=f"reference set cases[{index}].sha256"
        )
        if sha256_file(artifact_path) != expected_sha:
            raise OpenPIQualificationError(f"Reference case SHA-256 mismatch: {relative_text}")
        artifact = validate_reference_artifact(
            logical_artifact_path,
            verify_payloads=verify_payloads,
        )
        artifact_case = _object(artifact["case"], label="reference case")
        if artifact["profile_name"] != profile or artifact_case["id"] != case_id:
            raise OpenPIQualificationError(f"Reference case metadata mismatch: {relative_text}")
        seen_ids.add(case_id)
        seen_paths.add(relative_text)
    return document
