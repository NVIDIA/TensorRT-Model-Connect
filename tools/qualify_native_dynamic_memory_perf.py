#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed static-versus-dynamic native-runtime performance gate.

The four input JSON files use ``trtmc.benchmark-worker-result/v1`` plus two
qualification-only objects:

``qualification_provenance``
    Identifies the exact source snapshot, bundle, request, build, model
    revision, target, precision, toolchain, and benchmark environment.  A
    result must explicitly say that its bundle was freshly built for this
    qualification and was not reused.

``runtime_memory_receipt``
    Supplies TensorRT-derived serialized-plan and resident-weight accounting.
    Missing or unavailable accounting fails closed; bundle size is measured
    directly from the two bundle paths supplied on the command line.

``generation_workload`` and ``tokenizer_contract``
    Prove a fixed-length greedy AR measurement, repeatability within each
    case, and identical prompt-tokenization inputs.  Generated token IDs are
    preserved as a diagnostic rather than a hard static/dynamic equality gate:
    two logit rows may satisfy the correctness tolerance yet select different
    tokens near a numerical tie, while retaining the same decoder shape work.

This tool deliberately does not discover artifacts from a repository-specific
directory.  Every evidence input is explicit, so an old artifact tree cannot
be selected implicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Mapping, Sequence


RESULT_SCHEMA = "trtmc.benchmark-worker-result/v1"
REPORT_SCHEMA = "trtmc.native-dynamic-memory-perf-qualification/v1"
BUILD_SCHEMA = "trtmc.native-dynamic-memory-perf-build/v2"
GENERATION_WORKLOAD_SCHEMA = (
    "trtmc.native-dynamic-memory-generation-workload/v1"
)
TOKENIZER_CONTRACT_SCHEMA = (
    "trtmc.native-dynamic-memory-tokenizer-contract/v1"
)
MAX_DYNAMIC_TO_STATIC_SIZE_RATIO = 1.05
MIN_DYNAMIC_TO_STATIC_DECODE_RATIO = 0.95
MAX_DYNAMIC_TO_STATIC_PREFILL_RATIO = 1.10
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40,64}$")
_SHARED_PROVENANCE_FIELDS = (
    "git_head",
    "source_state_sha256",
    "prebuild_source_state_sha256",
    "postbuild_source_state_sha256",
    "model_revision",
    "precision",
    "target",
    "toolchain_sha256",
    "benchmark_environment_sha256",
)
_SOURCE_STATE_BOUNDARY_NAMES = (
    "build_pre",
    "build_post",
    "benchmark_pre",
    "benchmark_post",
)
_SOURCE_STATE_BOUNDARY_FIELDS = (
    "git_head",
    "source_state_sha256",
    "git_dirty",
    "exact_head_gate_satisfied",
)
_RECEIPT_FIELDS = (
    "serialized_plan_bytes",
    "resident_weight_bytes",
    "resident_weight_copy_count",
    "weight_streaming_active",
)
_MODULE_RESIDENCY_RECEIPT_FIELDS = (
    "receipt_schema_version",
    "contract_version",
    "module_residency_reserve_bytes",
    "module_residency_reserve_profile_limit",
    "module_residency_plan_set_sha256",
    "module_residency_evidence_sha256",
    "module_residency_cuda_module_loading_mode",
)
_MEASUREMENT_SOURCES = {
    "serialized_plan_bytes": "bundle_engine_section_sizes",
    "resident_weight_bytes": (
        "tensorrt_total_weights_size_weight_streaming_disabled"
    ),
    "resident_weight_copy_count": "deduplicated_tensorrt_engine_identity",
}
_RUNTIME_PLAN_FIELDS = (
    "schema",
    "device",
    "role",
    "hq",
    "hkv",
    "d",
    "C",
    "Sq",
    "T",
    "stats",
    "heur",
    "plan",
    "workspace_bytes",
    "cudnn_version",
)
_RUNTIME_PLAN_INTEGER_FIELDS = (
    "schema",
    "device",
    "hq",
    "hkv",
    "d",
    "C",
    "Sq",
    "T",
    "workspace_bytes",
    "cudnn_version",
)
_RUNTIME_PLAN_IDENTITY_FIELDS = (
    "device",
    "role",
    "hq",
    "hkv",
    "d",
    "C",
    "Sq",
    "T",
)
_RUNTIME_STACK_FIELDS = (
    "schema",
    "sm",
    "tensorrt",
    "cuda_runtime",
    "cudnn_backend",
    "cudnn_frontend_revision",
    "nvrtc",
    "driver",
)
_RUNTIME_LIBRARIES_FIELDS = (
    "directory",
    "live_nvrtc_version",
    "nvrtc",
    "nvrtc_builtins",
)
_RUNTIME_LIBRARY_FILE_FIELDS = (
    "path",
    "basename",
    "sha256",
    "size_bytes",
)
_BUILD_RUNTIME_KV_PLUGIN_FIELDS = (
    "path",
    "device",
    "inode",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
    "sha256",
)
_RUNTIME_TRTMC_FIELDS = (
    "model_id",
    "model_family",
    "core",
    "trt_backend",
    "runtime_kv_plugin",
    "model",
)
_MAPPED_DSO_IDENTITY_FIELDS = (
    "path",
    "device",
    "inode",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
    "sha256",
)
_CUDA_CACHE_FIELDS = (
    "path",
    "path_source",
    "cuda_cache_path_env",
    "cuda_cache_disable_env",
    "enabled",
    "initial_state",
    "worker_started_ns",
    "worker_finished_ns",
    "before",
    "after",
)
_CUDA_CACHE_SNAPSHOT_FIELDS = (
    "captured_at_ns",
    "exists",
    "is_directory",
    "entry_count",
    "file_count",
    "total_bytes",
    "metadata_sha256",
)
_GENERATION_WORKLOAD_FIELDS = (
    "schema_version",
    "kind",
    "structural_identity",
    "structural_identity_sha256",
    "measured_generated_token_ids",
    "measured_generated_token_ids_sha256",
    "token_stream_repeatable_within_case",
)
_STRUCTURAL_IDENTITY_FIELDS = (
    "operation",
    "prompt_sha256",
    "prompt_utf8_bytes",
    "generation",
    "measurement",
)
_TOKENIZER_CONTRACT_FIELDS = (
    "schema_version",
    "tokenizer_json_sha256",
    "tokenizer_json_bytes",
    "tokenizer_add_special_tokens",
    "tokenizer_special_prefix_ids",
    "tokenizer_special_suffix_ids",
)


class QualificationError(RuntimeError):
    """An input is not sufficient to make a performance claim."""


@dataclass(frozen=True)
class CaseEvidence:
    label: str
    path: Path
    result_sha256: str
    model_id: str
    iterations: int
    warmup: int
    prefill_ms: tuple[float, ...]
    decode_ms: tuple[float, ...]
    output_tokens: tuple[int, ...]
    provenance: Mapping[str, Any]
    receipt: Mapping[str, Any]
    runtime_memory_contract: Mapping[str, Any] | None
    module_residency_receipt: Mapping[str, Any] | None
    runtime_attention_plans: tuple[Mapping[str, Any], ...]
    runtime_stack: Mapping[str, Any] | None
    runtime_libraries: Mapping[str, Any] | None
    runtime_trtmc_libraries: Mapping[str, Any]
    mapped_dso_identities: tuple[Mapping[str, Any], ...]
    build_runtime_kv_plugin: Mapping[str, Any] | None
    build_manifest: Mapping[str, Any]
    build_receipt: Mapping[str, Any]
    cuda_jit_cache: Mapping[str, Any]
    generation_workload: Mapping[str, Any]
    tokenizer_contract: Mapping[str, Any]

    @property
    def mean_prefill_ms(self) -> float:
        return statistics.fmean(self.prefill_ms)

    @property
    def decode_tokens_per_second(self) -> float:
        return sum(self.output_tokens) * 1000.0 / sum(self.decode_ms)

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "result_sha256": self.result_sha256,
            "model_id": self.model_id,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "mean_prefill_ms": self.mean_prefill_ms,
            "decode_tokens_per_second": self.decode_tokens_per_second,
            "total_output_tokens": sum(self.output_tokens),
            "qualification_provenance": dict(self.provenance),
            "runtime_memory_receipt": {
                field: self.receipt[field] for field in _RECEIPT_FIELDS
            },
            "bundle_runtime_memory_contract": (
                dict(self.runtime_memory_contract)
                if self.runtime_memory_contract is not None
                else None
            ),
            "runtime_module_residency_receipt": (
                dict(self.module_residency_receipt)
                if self.module_residency_receipt is not None
                else None
            ),
            "runtime_attention_plans": [
                dict(plan) for plan in self.runtime_attention_plans
            ],
            "runtime_stack": (
                dict(self.runtime_stack)
                if self.runtime_stack is not None
                else None
            ),
            "runtime_libraries": (
                dict(self.runtime_libraries)
                if self.runtime_libraries is not None
                else None
            ),
            "runtime_trtmc_libraries": dict(
                self.runtime_trtmc_libraries
            ),
            "mapped_dso_identities": [
                dict(identity)
                for identity in self.mapped_dso_identities
            ],
            "build_runtime_kv_plugin": (
                dict(self.build_runtime_kv_plugin)
                if self.build_runtime_kv_plugin is not None
                else None
            ),
            "build_manifest": dict(self.build_manifest),
            "cuda_jit_cache": dict(self.cuda_jit_cache),
            "generation_workload": dict(self.generation_workload),
            "tokenizer_contract": dict(self.tokenizer_contract),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise QualificationError(f"{where} must be a JSON object")
    return value


def _required(mapping: Mapping[str, Any], field: str, where: str) -> Any:
    if field not in mapping:
        raise QualificationError(f"missing field: {where}.{field}")
    return mapping[field]


def _nonempty_string(mapping: Mapping[str, Any], field: str, where: str) -> str:
    value = _required(mapping, field, where)
    if not isinstance(value, str) or not value.strip():
        raise QualificationError(f"{where}.{field} must be a non-empty string")
    return value


def _sha_field(
    mapping: Mapping[str, Any],
    field: str,
    where: str,
    *,
    git: bool = False,
) -> str:
    value = _nonempty_string(mapping, field, where)
    pattern = _GIT_SHA if git else _SHA256
    if pattern.fullmatch(value) is None:
        kind = "Git object ID" if git else "lowercase SHA-256"
        raise QualificationError(f"{where}.{field} must be a {kind}")
    return value


def _positive_int(mapping: Mapping[str, Any], field: str, where: str) -> int:
    value = _required(mapping, field, where)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QualificationError(f"{where}.{field} must be a positive integer")
    return value


def _boolean(mapping: Mapping[str, Any], field: str, where: str) -> bool:
    value = _required(mapping, field, where)
    if not isinstance(value, bool):
        raise QualificationError(f"{where}.{field} must be boolean")
    return value


def _validate_source_state_boundaries(
    value: Any, *, where: str
) -> Mapping[str, Mapping[str, Any]]:
    boundaries = _object(value, where)
    _exact_fields(boundaries, _SOURCE_STATE_BOUNDARY_NAMES, where)
    validated: dict[str, Mapping[str, Any]] = {}
    for name in _SOURCE_STATE_BOUNDARY_NAMES:
        boundary_where = f"{where}.{name}"
        boundary = _object(
            _required(boundaries, name, where), boundary_where
        )
        _exact_fields(boundary, _SOURCE_STATE_BOUNDARY_FIELDS, boundary_where)
        _sha_field(boundary, "git_head", boundary_where, git=True)
        _sha_field(boundary, "source_state_sha256", boundary_where)
        _boolean(boundary, "git_dirty", boundary_where)
        _boolean(
            boundary,
            "exact_head_gate_satisfied",
            boundary_where,
        )
        validated[name] = boundary
    return validated


def _nonnegative_int(
    mapping: Mapping[str, Any], field: str, where: str
) -> int:
    value = _required(mapping, field, where)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualificationError(
            f"{where}.{field} must be a non-negative integer"
        )
    return value


def _finite_positive(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{where} must be a finite positive number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise QualificationError(f"{where} must be a finite positive number")
    return converted


def _exact_fields(
    mapping: Mapping[str, Any], expected: Sequence[str], where: str
) -> None:
    expected_fields = set(expected)
    missing = sorted(expected_fields - mapping.keys())
    extra = sorted(mapping.keys() - expected_fields)
    if missing or extra:
        raise QualificationError(
            f"{where} fields must match capture schema: "
            f"missing={missing!r}, extra={extra!r}"
        )


def _validate_runtime_attention_plans(
    value: Any, *, where: str, artifact_role: str
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise QualificationError(f"{where} must be a JSON array")
    if artifact_role == "exact-head-static-split":
        if value:
            raise QualificationError(
                f"{where} must be [] for a static baseline"
            )
        return ()
    if not value:
        raise QualificationError(
            f"{where} must contain at least one dynamic LSE plan"
        )

    identities: list[tuple[Any, ...]] = []
    rows: list[Mapping[str, Any]] = []
    for index, raw_row in enumerate(value):
        row_where = f"{where}[{index}]"
        row = _object(raw_row, row_where)
        _exact_fields(row, _RUNTIME_PLAN_FIELDS, row_where)
        for field in _RUNTIME_PLAN_INTEGER_FIELDS:
            _nonnegative_int(row, field, row_where)
        if row["schema"] != 1:
            raise QualificationError(f"{row_where}.schema must be 1")
        if row["role"] not in ("history", "current"):
            raise QualificationError(
                f"{row_where}.role must be 'history' or 'current'"
            )
        for field in ("hq", "hkv", "d", "C", "Sq", "T", "cudnn_version"):
            if row[field] <= 0:
                raise QualificationError(
                    f"{row_where}.{field} must be a positive integer"
                )
        if row["stats"] != "lse":
            raise QualificationError(f"{row_where}.stats must be 'lse'")
        for field in ("heur", "plan"):
            _nonempty_string(row, field, row_where)
        identity = tuple(row[field] for field in _RUNTIME_PLAN_IDENTITY_FIELDS)
        if identity in identities:
            raise QualificationError(
                f"{where} contains duplicate capture identity {identity!r}"
            )
        identities.append(identity)
        rows.append(row)

    if identities != sorted(identities):
        raise QualificationError(
            f"{where} must use capture's canonical identity ordering"
        )
    if len({row["device"] for row in rows}) != 1:
        raise QualificationError(
            f"{where} must contain one CUDA device identity"
        )
    if len({row["cudnn_version"] for row in rows}) != 1:
        raise QualificationError(
            f"{where} must contain one encoded cuDNN version"
        )
    return tuple(rows)


def _encoded_cudnn_version(version: str, where: str) -> int:
    parts = version.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise QualificationError(
            f"{where} must contain major.minor.patch"
        )
    major, minor, patch = (int(part) for part in parts)
    return major * 10000 + minor * 100 + patch


def _validate_runtime_stack(
    value: Any, *, where: str, artifact_role: str
) -> Mapping[str, Any] | None:
    if artifact_role == "exact-head-static-split":
        if value is not None:
            raise QualificationError(
                f"{where} must be null for a static baseline"
            )
        return None

    stack = _object(value, where)
    _exact_fields(stack, _RUNTIME_STACK_FIELDS, where)
    schema = _required(stack, "schema", where)
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        raise QualificationError(f"{where}.schema must be 1")
    for field in _RUNTIME_STACK_FIELDS[1:]:
        text = _nonempty_string(stack, field, where)
        if text == "unavailable":
            raise QualificationError(f"{where}.{field} is unavailable")
    if re.fullmatch(r"sm[0-9]+", stack["sm"]) is None:
        raise QualificationError(f"{where}.sm must use the smNNN form")
    if re.fullmatch(
        r"[0-9a-f]{40}", stack["cudnn_frontend_revision"]
    ) is None:
        raise QualificationError(
            f"{where}.cudnn_frontend_revision must be a full Git SHA"
        )
    for field in ("tensorrt", "cuda_runtime", "cudnn_backend", "nvrtc"):
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", stack[field]) is None:
            raise QualificationError(
                f"{where}.{field} must be a dotted numeric version"
            )
    _encoded_cudnn_version(stack["cudnn_backend"], f"{where}.cudnn_backend")
    return stack


def _validate_runtime_libraries(
    value: Any,
    *,
    where: str,
    artifact_role: str,
    runtime_stack: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if artifact_role == "exact-head-static-split":
        if value is not None:
            raise QualificationError(
                f"{where} must be null for a static baseline"
            )
        return None
    if runtime_stack is None:
        raise QualificationError(f"{where} has no live runtime stack")

    libraries = _object(value, where)
    _exact_fields(libraries, _RUNTIME_LIBRARIES_FIELDS, where)
    directory = Path(_nonempty_string(libraries, "directory", where))
    if not directory.is_absolute():
        raise QualificationError(f"{where}.directory must be absolute")
    live_version = _nonempty_string(
        libraries, "live_nvrtc_version", where
    )
    if live_version != runtime_stack["nvrtc"]:
        raise QualificationError(
            f"{where}.live_nvrtc_version disagrees with runtime_stack.nvrtc"
        )

    for label in ("nvrtc", "nvrtc_builtins"):
        file_where = f"{where}.{label}"
        evidence = _object(
            _required(libraries, label, where), file_where
        )
        _exact_fields(evidence, _RUNTIME_LIBRARY_FILE_FIELDS, file_where)
        path = Path(_nonempty_string(evidence, "path", file_where))
        basename = _nonempty_string(evidence, "basename", file_where)
        if not path.is_absolute() or path.parent != directory:
            raise QualificationError(
                f"{file_where}.path must be inside {directory}"
            )
        if path.name != basename:
            raise QualificationError(
                f"{file_where}.basename disagrees with path"
            )
        expected_pattern = (
            r"libnvrtc\.so\.13(?:\.[0-9]+)*"
            if label == "nvrtc"
            else r"libnvrtc-builtins\.so\.13(?:\.[0-9]+)*"
        )
        if re.fullmatch(expected_pattern, basename) is None:
            raise QualificationError(
                f"{file_where}.basename is not a CUDA 13 library"
            )
        size = _positive_int(evidence, "size_bytes", file_where)
        digest = _sha_field(evidence, "sha256", file_where)
        try:
            stat = path.stat()
        except OSError as exc:
            raise QualificationError(
                f"{file_where}.path is not readable: {exc}"
            ) from exc
        if not path.is_file() or stat.st_size != size or _sha256(path) != digest:
            raise QualificationError(
                f"{file_where} does not match the captured file"
            )
        minor = str(live_version).split(".")
        if len(minor) < 2 or f".so.13.{minor[1]}" not in basename:
            raise QualificationError(
                f"{file_where}.basename disagrees with NVRTC {live_version}"
            )
    return libraries


def _validate_build_runtime_kv_plugin(
    value: Any,
    *,
    where: str,
    artifact_role: str,
) -> Mapping[str, Any] | None:
    if artifact_role == "exact-head-static-split":
        if value is not None:
            raise QualificationError(
                f"{where} must be null for the static baseline"
            )
        return None
    if artifact_role != "native-dynamic":
        raise QualificationError(f"{where} has an unsupported artifact role")

    return _validate_binary_identity(value, where=where)


def _validate_binary_identity(
    value: Any,
    *,
    where: str,
    expected_path: Path | None = None,
) -> Mapping[str, Any]:
    identity = _object(value, where)
    _exact_fields(identity, _BUILD_RUNTIME_KV_PLUGIN_FIELDS, where)
    path = Path(_nonempty_string(identity, "path", where))
    if not path.is_absolute():
        raise QualificationError(f"{where}.path must be absolute")
    try:
        canonical = path.resolve(strict=True)
        metadata = canonical.stat()
    except OSError as exc:
        raise QualificationError(f"{where}.path is not readable: {exc}") from exc
    if canonical != path:
        raise QualificationError(f"{where}.path must be canonical")
    if expected_path is not None and canonical != expected_path.resolve():
        raise QualificationError(f"{where}.path does not match expected file")
    expected = {
        "path": str(canonical),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "size_bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "ctime_ns": metadata.st_ctime_ns,
        "sha256": _sha256(canonical),
    }
    if dict(identity) != expected:
        raise QualificationError(
            f"{where} does not match the captured binary identity"
        )
    return identity


def _validate_mapped_dso_identities(
    value: Any,
    *,
    where: str,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise QualificationError(f"{where} must be a JSON array")
    rows: list[Mapping[str, Any]] = []
    for index, raw in enumerate(value):
        row = _object(raw, f"{where}[{index}]")
        _exact_fields(
            row,
            _MAPPED_DSO_IDENTITY_FIELDS,
            f"{where}[{index}]",
        )
        rows.append(
            _validate_binary_identity(
                row,
                where=f"{where}[{index}]",
            )
        )
    expected = sorted(
        rows,
        key=lambda row: (
            str(row["path"]),
            int(row["device"]),
            int(row["inode"]),
        ),
    )
    if rows != expected:
        raise QualificationError(
            f"{where} must use canonical path/inode ordering"
        )
    if len(
        {
            (row["path"], row["device"], row["inode"])
            for row in rows
        }
    ) != len(rows):
        raise QualificationError(f"{where} contains duplicate mapped DSOs")
    return tuple(rows)


def _load_boundary_module() -> Any:
    path = Path(__file__).with_name("qualify_native_dynamic_memory.py")
    spec = importlib.util.spec_from_file_location(
        "_trtmc_dynamic_memory_perf_boundary_replay",
        path,
    )
    if spec is None or spec.loader is None:
        raise QualificationError(
            f"cannot load sealed-bundle validator: {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _replay_sealed_runtime_memory_contract(
    bundle: Path,
) -> Mapping[str, Any]:
    """Independently hash the split plans and validate sealed contract v2."""

    boundary = _load_boundary_module()
    try:
        header = boundary._read_bundle_header(bundle)
        spec = boundary._resolve_spec(header)
        contract = boundary._sealed_profile_sweep_contract(
            bundle,
            header,
            expected_model_id=spec.model_id,
            expected_context_limit=spec.context_limit,
            expected_profile_limits=tuple(spec.buckets),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise QualificationError(
            "final performance gate could not independently replay the "
            f"sealed v2 bundle contract: {exc}"
        ) from exc
    return contract


def _replay_module_residency_receipt(
    receipt: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    live_runtime_stack: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Independently bind one live receipt to plan hashes and stack tuple."""

    calibration = _object(
        _required(
            contract,
            "module_residency_calibration",
            "sealed runtime_memory contract",
        ),
        "sealed runtime_memory contract.module_residency_calibration",
    )
    qualified_stack = _object(
        _required(
            contract,
            "qualified_runtime_stack",
            "sealed runtime_memory contract",
        ),
        "sealed runtime_memory contract.qualified_runtime_stack",
    )
    observed_stack = {
        field: live_runtime_stack.get(field)
        for field in qualified_stack
    }
    if observed_stack != dict(qualified_stack):
        raise QualificationError(
            "live runtime stack does not match the independently reopened "
            "bundle calibration"
        )
    capacity = _positive_int(
        receipt,
        "runtime_kv_capacity_tokens",
        "runtime_memory_receipt",
    )
    raw_reserves = _required(
        calibration,
        "profile_reserves",
        "sealed runtime_memory contract.module_residency_calibration",
    )
    if not isinstance(raw_reserves, list):
        raise QualificationError(
            "sealed module-residency profile reserves must be an array"
        )
    selected = next(
        (
            row
            for row in raw_reserves
            if isinstance(row, Mapping)
            and type(row.get("covering_profile_limit")) is int
            and int(row["covering_profile_limit"]) >= capacity
        ),
        None,
    )
    if selected is None:
        raise QualificationError(
            f"sealed module-residency calibration does not cover R={capacity}"
        )
    expected = {
        "receipt_schema_version": 4,
        "contract_version": 2,
        "module_residency_reserve_bytes": selected[
            "cumulative_reserve_bytes"
        ],
        "module_residency_reserve_profile_limit": selected[
            "covering_profile_limit"
        ],
        "module_residency_plan_set_sha256": calibration[
            "plan_set_sha256"
        ],
        "module_residency_evidence_sha256": calibration[
            "evidence_sha256"
        ],
        "module_residency_cuda_module_loading_mode": calibration[
            "cuda_module_loading_mode"
        ],
    }
    mismatches = {
        field: {"expected": value, "actual": receipt.get(field)}
        for field, value in expected.items()
        if receipt.get(field) != value
    }
    if mismatches:
        raise QualificationError(
            "runtime receipt does not match the independently reopened "
            f"module-residency calibration: {mismatches}"
        )
    return expected


def _validate_runtime_trtmc_libraries(
    value: Any,
    *,
    where: str,
    model_id: str,
) -> Mapping[str, Any]:
    libraries = _object(value, where)
    _exact_fields(libraries, _RUNTIME_TRTMC_FIELDS, where)
    if libraries.get("model_id") != model_id:
        raise QualificationError(f"{where}.model_id disagrees with result")
    expected_family = {
        "Qwen/Qwen3-0.6B": "qwen",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "llama",
    }.get(model_id)
    if expected_family is None or libraries.get("model_family") != expected_family:
        raise QualificationError(
            f"{where}.model_family is not qualified for {model_id!r}"
        )
    for field in ("core", "trt_backend", "runtime_kv_plugin", "model"):
        _validate_binary_identity(
            _required(libraries, field, where),
            where=f"{where}.{field}",
        )
    if Path(libraries["core"]["path"]).name != "libtrtmc_core.so":
        raise QualificationError(f"{where}.core is not libtrtmc_core.so")
    if (
        re.fullmatch(
            r"libtrtmc_backend_trt(?:_[0-9]+_[0-9]+)?\.so",
            Path(libraries["trt_backend"]["path"]).name,
        )
        is None
    ):
        raise QualificationError(f"{where}.trt_backend has an invalid basename")
    expected_model_dso = f"libtrtmc_model_{expected_family}.so"
    if Path(libraries["model"]["path"]).name != expected_model_dso:
        raise QualificationError(
            f"{where}.model must be {expected_model_dso}"
        )
    return libraries


def _validate_cache_snapshot(
    value: Any, *, where: str
) -> Mapping[str, Any]:
    snapshot = _object(value, where)
    _exact_fields(snapshot, _CUDA_CACHE_SNAPSHOT_FIELDS, where)
    captured_at_ns = _positive_int(snapshot, "captured_at_ns", where)
    del captured_at_ns
    exists = _required(snapshot, "exists", where)
    is_directory = _required(snapshot, "is_directory", where)
    for field, field_value in (
        ("exists", exists),
        ("is_directory", is_directory),
    ):
        if not isinstance(field_value, bool):
            raise QualificationError(f"{where}.{field} must be boolean")
    entry_count = _nonnegative_int(snapshot, "entry_count", where)
    file_count = _nonnegative_int(snapshot, "file_count", where)
    total_bytes = _nonnegative_int(snapshot, "total_bytes", where)
    _sha_field(snapshot, "metadata_sha256", where)
    if file_count > entry_count:
        raise QualificationError(
            f"{where}.file_count cannot exceed entry_count"
        )
    if exists is not is_directory:
        raise QualificationError(
            f"{where}.exists and is_directory must agree"
        )
    if not exists and (entry_count != 0 or file_count != 0 or total_bytes != 0):
        raise QualificationError(
            f"{where} must have zero counts when the cache path is absent"
        )
    return snapshot


def _validate_cuda_jit_cache(value: Any, *, where: str) -> Mapping[str, Any]:
    cache = _object(value, where)
    _exact_fields(cache, _CUDA_CACHE_FIELDS, where)
    path = _nonempty_string(cache, "path", where)
    if not Path(path).is_absolute():
        raise QualificationError(f"{where}.path must be absolute")
    source = _nonempty_string(cache, "path_source", where)
    if source not in ("CUDA_CACHE_PATH", "cuda_default"):
        raise QualificationError(
            f"{where}.path_source must identify capture's path source"
        )
    path_env = _required(cache, "cuda_cache_path_env", where)
    disable_env = _required(cache, "cuda_cache_disable_env", where)
    for field, field_value in (
        ("cuda_cache_path_env", path_env),
        ("cuda_cache_disable_env", disable_env),
    ):
        if field_value is not None and not isinstance(field_value, str):
            raise QualificationError(
                f"{where}.{field} must be a string or null"
            )
    if source == "CUDA_CACHE_PATH" and not path_env:
        raise QualificationError(
            f"{where}.cuda_cache_path_env must name the configured path"
        )
    if source == "cuda_default" and path_env:
        raise QualificationError(
            f"{where}.cuda_cache_path_env must be empty for the default path"
        )
    enabled = _required(cache, "enabled", where)
    if not isinstance(enabled, bool):
        raise QualificationError(f"{where}.enabled must be boolean")
    if enabled != (disable_env != "1"):
        raise QualificationError(
            f"{where}.enabled disagrees with CUDA_CACHE_DISABLE"
        )
    initial_state = _nonempty_string(cache, "initial_state", where)
    if initial_state not in ("cold", "warm", "disabled"):
        raise QualificationError(
            f"{where}.initial_state must be cold, warm, or disabled"
        )
    started_ns = _positive_int(cache, "worker_started_ns", where)
    finished_ns = _positive_int(cache, "worker_finished_ns", where)
    before = _validate_cache_snapshot(
        _required(cache, "before", where), where=f"{where}.before"
    )
    after = _validate_cache_snapshot(
        _required(cache, "after", where), where=f"{where}.after"
    )
    expected_state = (
        "disabled"
        if not enabled
        else ("warm" if before["file_count"] > 0 else "cold")
    )
    if initial_state != expected_state:
        raise QualificationError(
            f"{where}.initial_state must be {expected_state!r}"
        )
    if not (
        before["captured_at_ns"]
        <= started_ns
        <= finished_ns
        <= after["captured_at_ns"]
    ):
        raise QualificationError(
            f"{where} timestamps must enclose the worker execution"
        )
    return cache


def _token_id_stream(value: Any, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise QualificationError(f"{where} must be a non-empty token array")
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        for token in value
    ):
        raise QualificationError(
            f"{where} must contain non-negative integer token IDs"
        )
    return tuple(value)


def _validate_generation_workload(
    value: Any,
    *,
    where: str,
    iterations: int,
    warmup: int,
    output_tokens: tuple[int, ...],
) -> Mapping[str, Any]:
    workload = _object(value, where)
    _exact_fields(workload, _GENERATION_WORKLOAD_FIELDS, where)
    if workload["schema_version"] != GENERATION_WORKLOAD_SCHEMA:
        raise QualificationError(
            f"{where}.schema_version must be {GENERATION_WORKLOAD_SCHEMA!r}"
        )
    if workload["kind"] != "fixed_length_greedy_ar":
        raise QualificationError(
            f"{where}.kind must be 'fixed_length_greedy_ar'"
        )
    structural = _object(
        workload["structural_identity"], f"{where}.structural_identity"
    )
    _exact_fields(
        structural,
        _STRUCTURAL_IDENTITY_FIELDS,
        f"{where}.structural_identity",
    )
    if structural["operation"] != "generate":
        raise QualificationError(
            f"{where}.structural_identity.operation must be 'generate'"
        )
    _sha_field(
        structural,
        "prompt_sha256",
        f"{where}.structural_identity",
    )
    _positive_int(
        structural,
        "prompt_utf8_bytes",
        f"{where}.structural_identity",
    )
    generation = _object(
        structural["generation"],
        f"{where}.structural_identity.generation",
    )
    expected_generation_fields = {
        "generation_mode",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
        "num_samples",
        "eos_token_id",
        "use_chat_template",
        "stop_on_boxed_answer",
        "capture_generated_token_ids",
        "max_new_tokens",
    }
    _exact_fields(
        generation,
        tuple(expected_generation_fields),
        f"{where}.structural_identity.generation",
    )
    max_new_tokens = _positive_int(
        generation,
        "max_new_tokens",
        f"{where}.structural_identity.generation",
    )
    if generation != {
        "generation_mode": "ar",
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "num_samples": 1,
        "eos_token_id": 2_147_483_647,
        "use_chat_template": False,
        "stop_on_boxed_answer": False,
        "capture_generated_token_ids": True,
        "max_new_tokens": max_new_tokens,
    }:
        raise QualificationError(
            f"{where}.structural_identity.generation is not the captured "
            "fixed-length greedy AR contract"
        )
    measurement = _object(
        structural["measurement"],
        f"{where}.structural_identity.measurement",
    )
    if (
        measurement.get("iterations") != iterations
        or measurement.get("warmup") != warmup
    ):
        raise QualificationError(
            f"{where}.structural_identity.measurement disagrees with worker "
            "iterations/warmup"
        )
    structural_sha = _sha_field(
        workload, "structural_identity_sha256", where
    )
    if structural_sha != _canonical_sha(structural):
        raise QualificationError(
            f"{where}.structural_identity_sha256 does not match its object"
        )

    raw_streams = workload["measured_generated_token_ids"]
    if not isinstance(raw_streams, list) or len(raw_streams) != iterations:
        raise QualificationError(
            f"{where}.measured_generated_token_ids must contain "
            f"{iterations} streams"
        )
    streams = tuple(
        _token_id_stream(
            stream,
            where=f"{where}.measured_generated_token_ids[{index}]",
        )
        for index, stream in enumerate(raw_streams)
    )
    if tuple(len(stream) for stream in streams) != output_tokens or any(
        len(stream) != max_new_tokens for stream in streams
    ):
        raise QualificationError(
            f"{where}.measured_generated_token_ids disagrees with fixed "
            "output-token counts"
        )
    stream_sha = _sha_field(
        workload, "measured_generated_token_ids_sha256", where
    )
    if stream_sha != _canonical_sha(raw_streams):
        raise QualificationError(
            f"{where}.measured_generated_token_ids_sha256 does not match "
            "its arrays"
        )
    repeatable = workload["token_stream_repeatable_within_case"]
    if not isinstance(repeatable, bool):
        raise QualificationError(
            f"{where}.token_stream_repeatable_within_case must be boolean"
        )
    actual_repeatable = all(stream == streams[0] for stream in streams[1:])
    if repeatable is not actual_repeatable or not repeatable:
        raise QualificationError(
            f"{where} does not prove a repeatable greedy token stream"
        )
    return workload


def _validate_tokenizer_contract(
    value: Any, *, where: str
) -> Mapping[str, Any]:
    contract = _object(value, where)
    _exact_fields(contract, _TOKENIZER_CONTRACT_FIELDS, where)
    if contract["schema_version"] != TOKENIZER_CONTRACT_SCHEMA:
        raise QualificationError(
            f"{where}.schema_version must be {TOKENIZER_CONTRACT_SCHEMA!r}"
        )
    _sha_field(contract, "tokenizer_json_sha256", where)
    _positive_int(contract, "tokenizer_json_bytes", where)
    if not isinstance(contract["tokenizer_add_special_tokens"], bool):
        raise QualificationError(
            f"{where}.tokenizer_add_special_tokens must be boolean"
        )
    for field in (
        "tokenizer_special_prefix_ids",
        "tokenizer_special_suffix_ids",
    ):
        raw = contract[field]
        if not isinstance(raw, list) or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            for token in raw
        ):
            raise QualificationError(
                f"{where}.{field} must contain non-negative integer token IDs"
            )
    return contract


def _validate_evidence_hash(
    *,
    value: Any,
    provenance: Mapping[str, Any],
    field: str,
    provenance_where: str,
) -> None:
    actual = _sha_field(provenance, field, provenance_where)
    expected = _canonical_sha(value)
    if actual != expected:
        raise QualificationError(
            f"{provenance_where}.{field} does not match captured evidence"
        )


def _load_capture_module() -> Any:
    path = Path(__file__).with_name("capture_native_dynamic_memory_perf.py")
    spec = importlib.util.spec_from_file_location(
        "_trtmc_dynamic_memory_perf_capture_replay", path
    )
    if spec is None or spec.loader is None:
        raise QualificationError(f"cannot load performance receipt validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reopen_file_hash(
    evidence: Mapping[str, Any],
    *,
    path_field: str,
    sha_field: str,
    where: str,
) -> Path:
    raw_path = _nonempty_string(evidence, path_field, where)
    path = Path(raw_path)
    if not path.is_absolute():
        raise QualificationError(f"{where}.{path_field} must be absolute")
    try:
        canonical = path.resolve(strict=True)
    except OSError as exc:
        raise QualificationError(
            f"{where}.{path_field} is not readable: {exc}"
        ) from exc
    if canonical != path or not canonical.is_file():
        raise QualificationError(
            f"{where}.{path_field} must be a canonical file"
        )
    digest = _sha_field(evidence, sha_field, where)
    if _sha256(canonical) != digest:
        raise QualificationError(
            f"{where}.{sha_field} no longer matches {path_field}"
        )
    return canonical


def _validate_qualification_evidence(
    result: Mapping[str, Any],
    *,
    label: str,
    expected_role: str,
    expected_bundle: Path,
    model_id: str,
    provenance: Mapping[str, Any],
    runtime_trtmc_libraries: Mapping[str, Any],
    mapped_dso_identities: tuple[Mapping[str, Any], ...],
    build_runtime_kv_plugin: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    where = f"{label}.qualification_evidence"
    evidence = _object(_required(result, "qualification_evidence", label), where)
    receipt_path = _reopen_file_hash(
        evidence,
        path_field="build_receipt",
        sha_field="build_receipt_sha256",
        where=where,
    )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(
            f"{where}.build_receipt cannot be parsed: {exc}"
        ) from exc
    receipt = _object(receipt, f"{where}.build_receipt")
    if receipt.get("schema_version") != BUILD_SCHEMA:
        raise QualificationError(
            f"{where}.build_receipt must use {BUILD_SCHEMA}"
        )
    if receipt.get("model_id") != model_id:
        raise QualificationError(
            f"{where}.build_receipt model_id disagrees with result"
        )
    source_state = _object(
        _required(evidence, "source_state_pre", where),
        f"{where}.source_state_pre",
    )
    toolchain = _object(
        _required(evidence, "toolchain", where), f"{where}.toolchain"
    )
    _exact_fields(
        toolchain,
        (
            "worker",
            "plugin_library",
            "runtime_trtmc_libraries",
            "build_manifest",
            "capture_tool",
            "capture_tool_sha256",
        ),
        f"{where}.toolchain",
    )
    worker_identity = _validate_binary_identity(
        _required(toolchain, "worker", f"{where}.toolchain"),
        where=f"{where}.toolchain.worker",
    )
    plugin_identity = _validate_binary_identity(
        _required(toolchain, "plugin_library", f"{where}.toolchain"),
        where=f"{where}.toolchain.plugin_library",
    )
    if toolchain.get("runtime_trtmc_libraries") != runtime_trtmc_libraries:
        raise QualificationError(
            f"{where}.toolchain runtime TRTMC DSOs disagree with result"
        )
    build_manifest = _object(
        _required(evidence, "build_manifest", where),
        f"{where}.build_manifest",
    )
    if toolchain.get("build_manifest") != build_manifest:
        raise QualificationError(
            f"{where}.toolchain build manifest binding disagrees"
        )
    if receipt.get("build_manifest") != build_manifest:
        raise QualificationError(
            f"{where}.build_receipt build manifest binding disagrees"
        )
    if evidence.get("runtime_trtmc_libraries") != runtime_trtmc_libraries:
        raise QualificationError(
            f"{where}.runtime_trtmc_libraries disagree with result"
        )
    if evidence.get("mapped_dso_identities") != list(
        mapped_dso_identities
    ):
        raise QualificationError(
            f"{where}.mapped_dso_identities disagree with result"
        )
    if evidence.get("build_runtime_kv_plugin") != build_runtime_kv_plugin:
        raise QualificationError(
            f"{where}.build_runtime_kv_plugin disagrees with result"
        )
    if provenance.get("toolchain_sha256") != _canonical_sha(toolchain):
        raise QualificationError(
            f"{label}.qualification_provenance.toolchain_sha256 is invalid"
        )
    _reopen_file_hash(
        toolchain,
        path_field="capture_tool",
        sha_field="capture_tool_sha256",
        where=f"{where}.toolchain",
    )
    environment = _object(
        _required(evidence, "environment", where),
        f"{where}.environment",
    )
    if provenance.get("benchmark_environment_sha256") != _canonical_sha(
        environment
    ):
        raise QualificationError(
            f"{label}.qualification_provenance benchmark environment hash "
            "is invalid"
        )
    if provenance.get("build_manifest_sha256") != _canonical_sha(
        build_manifest
    ):
        raise QualificationError(
            f"{label}.qualification_provenance.build_manifest_sha256 is invalid"
        )
    for field in ("worker_stdout", "worker_stderr", "request_file"):
        _reopen_file_hash(
            evidence,
            path_field=field,
            sha_field=f"{field}_sha256",
            where=where,
        )
    worker_command = _required(evidence, "worker_command", where)
    if (
        not isinstance(worker_command, list)
        or len(worker_command) != 5
        or worker_command[0] != worker_identity["path"]
        or worker_command[1] != "--request"
        or worker_command[3] != "--output"
        or evidence.get("worker_command_sha256")
        != _canonical_sha(worker_command)
    ):
        raise QualificationError(f"{where}.worker_command is not canonical")
    if Path(worker_command[2]).resolve() != Path(evidence["request_file"]):
        raise QualificationError(
            f"{where}.worker_command request does not match request_file"
        )

    capture = _load_capture_module()
    try:
        _, manifest_artifacts = capture._validate_build_receipt(
            receipt,
            bundle=expected_bundle.resolve(),
            role=expected_role,
            source_state=source_state,
            plugin_library=Path(plugin_identity["path"]),
        )
    except Exception as exc:
        if exc.__class__.__name__ != "CaptureError":
            raise
        raise QualificationError(
            f"{where}.build_receipt replay failed: {exc}"
        ) from exc

    runtime_receipt = _object(
        _required(result, "runtime_memory_receipt", label),
        f"{label}.runtime_memory_receipt",
    )
    if evidence.get("runtime_memory_receipt") != runtime_receipt:
        raise QualificationError(
            f"{where}.runtime_memory_receipt disagrees with result"
        )
    if provenance.get("runtime_memory_receipt_sha256") != _canonical_sha(
        runtime_receipt
    ):
        raise QualificationError(
            f"{label}.qualification_provenance."
            "runtime_memory_receipt_sha256 is invalid"
        )
    comparison_sequence_limit = _positive_int(
        evidence,
        "comparison_sequence_limit",
        where,
    )
    if expected_role == "native-dynamic":
        try:
            capture._validate_complete_schema_v4_receipt(
                runtime_receipt,
                comparison_sequence_limit=comparison_sequence_limit,
            )
        except Exception as exc:
            if exc.__class__.__name__ != "CaptureError":
                raise
            raise QualificationError(
                f"{where} complete schema-v4 receipt replay failed: {exc}"
            ) from exc

    runtime_memory_contract = result.get(
        "bundle_runtime_memory_contract"
    )
    module_residency_receipt = result.get(
        "runtime_module_residency_receipt"
    )
    if expected_role == "native-dynamic":
        runtime_memory_contract = _object(
            runtime_memory_contract,
            f"{label}.bundle_runtime_memory_contract",
        )
        module_residency_receipt = _object(
            module_residency_receipt,
            f"{label}.runtime_module_residency_receipt",
        )
        if (
            evidence.get("bundle_runtime_memory_contract")
            != runtime_memory_contract
            or evidence.get("runtime_module_residency_receipt")
            != module_residency_receipt
        ):
            raise QualificationError(
                f"{where} module-residency evidence disagrees with result"
            )
        if provenance.get(
            "bundle_runtime_memory_contract_sha256"
        ) != _canonical_sha(runtime_memory_contract):
            raise QualificationError(
                f"{label}.qualification_provenance."
                "bundle_runtime_memory_contract_sha256 is invalid"
            )
        if provenance.get(
            "runtime_module_residency_receipt_sha256"
        ) != _canonical_sha(module_residency_receipt):
            raise QualificationError(
                f"{label}.qualification_provenance."
                "runtime_module_residency_receipt_sha256 is invalid"
            )
        try:
            reopened_contract = _replay_sealed_runtime_memory_contract(
                expected_bundle
            )
            replayed_receipt = _replay_module_residency_receipt(
                runtime_receipt,
                contract=reopened_contract,
                live_runtime_stack=_object(
                    _required(result, "runtime_stack", label),
                    f"{label}.runtime_stack",
                ),
            )
        except Exception as exc:
            if exc.__class__.__name__ not in {
                "CaptureError",
                "ValueError",
                "RuntimeError",
            }:
                raise
            raise QualificationError(
                f"{where} module-residency replay failed: {exc}"
            ) from exc
        if (
            dict(runtime_memory_contract) != reopened_contract
            or dict(module_residency_receipt) != replayed_receipt
        ):
            raise QualificationError(
                f"{where} module-residency objects do not match the "
                "reopened bundle and runtime receipt"
            )
    elif (
        runtime_memory_contract is not None
        or module_residency_receipt is not None
        or evidence.get("bundle_runtime_memory_contract") is not None
        or evidence.get("runtime_module_residency_receipt") is not None
        or provenance.get("bundle_runtime_memory_contract_sha256")
        is not None
        or provenance.get(
            "runtime_module_residency_receipt_sha256"
        )
        is not None
    ):
        raise QualificationError(
            f"{where} static baseline unexpectedly claims a dynamic "
            "module-residency contract"
        )
    model_key = (
        "model_qwen"
        if runtime_trtmc_libraries["model_family"] == "qwen"
        else "model_llama"
    )
    for manifest_key, runtime_key in (
        ("benchmark_worker", None),
        ("core", "core"),
        ("trt_backend", "trt_backend"),
        ("runtime_kv_plugin", "runtime_kv_plugin"),
        (model_key, "model"),
    ):
        actual = (
            worker_identity
            if runtime_key is None
            else runtime_trtmc_libraries[runtime_key]
        )
        if manifest_artifacts.get(manifest_key) != actual:
            raise QualificationError(
                f"{where} {manifest_key} does not match build manifest"
            )
    if runtime_trtmc_libraries["runtime_kv_plugin"] != plugin_identity:
        raise QualificationError(
            f"{where} worker plugin differs from toolchain plugin"
        )
    if (
        expected_role == "native-dynamic"
        and build_runtime_kv_plugin != plugin_identity
    ):
        raise QualificationError(
            f"{where} build and worker runtime-KV plugins differ"
        )
    return evidence, receipt


def _read_case(
    path: Path,
    label: str,
    expected_role: str,
    expected_bundle: Path,
) -> CaseEvidence:
    resolved = path.expanduser().resolve()
    try:
        raw = resolved.read_bytes()
        result = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationError(f"{label}: cannot read worker result: {exc}") from exc
    result = _object(result, label)
    if _required(result, "schema_version", label) != RESULT_SCHEMA:
        raise QualificationError(
            f"{label}.schema_version must be {RESULT_SCHEMA!r}"
        )
    if _required(result, "status", label) != "completed":
        raise QualificationError(f"{label}.status must be 'completed'")
    if _required(result, "operation", label) != "generate":
        raise QualificationError(f"{label}.operation must be 'generate'")
    if (
        _required(result, "timing_scope", label)
        != "public_pipeline_call_wall"
    ):
        raise QualificationError(
            f"{label}.timing_scope must be 'public_pipeline_call_wall'"
        )
    model_id = _nonempty_string(result, "model_id", label)
    iterations = _positive_int(result, "iterations", label)
    warmup = _required(result, "warmup", label)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise QualificationError(f"{label}.warmup must be a non-negative integer")

    observations = _required(result, "observations", label)
    if not isinstance(observations, list) or len(observations) != iterations:
        raise QualificationError(
            f"{label}.observations must contain exactly {iterations} rows"
        )
    prefill: list[float] = []
    decode: list[float] = []
    output_tokens: list[int] = []
    for index, raw_observation in enumerate(observations):
        where = f"{label}.observations[{index}]"
        observation = _object(raw_observation, where)
        prefill.append(
            _finite_positive(_required(observation, "prefill_ms", where),
                             f"{where}.prefill_ms")
        )
        decode.append(
            _finite_positive(_required(observation, "decode_ms", where),
                             f"{where}.decode_ms")
        )
        output_tokens.append(_positive_int(observation, "output_tokens", where))

    provenance_where = f"{label}.qualification_provenance"
    provenance = _object(
        _required(result, "qualification_provenance", label),
        provenance_where,
    )
    for field in (
        "source_state_sha256",
        "source_state_pre_sha256",
        "source_state_post_sha256",
        "prebuild_source_state_sha256",
        "postbuild_source_state_sha256",
        "bundle_sha256",
        "request_sha256",
        "toolchain_sha256",
        "benchmark_environment_sha256",
        "runtime_attention_plans_sha256",
        "runtime_stack_sha256",
        "runtime_libraries_sha256",
        "runtime_trtmc_libraries_sha256",
        "mapped_dso_identities_sha256",
        "build_runtime_kv_plugin_sha256",
        "build_manifest_sha256",
        "cuda_jit_cache_sha256",
        "generation_workload_sha256",
        "tokenizer_contract_sha256",
        "runtime_memory_receipt_sha256",
    ):
        _sha_field(provenance, field, provenance_where)
    if expected_role == "native-dynamic":
        for field in (
            "bundle_runtime_memory_contract_sha256",
            "runtime_module_residency_receipt_sha256",
        ):
            _sha_field(provenance, field, provenance_where)
    _sha_field(provenance, "git_head", provenance_where, git=True)
    _boolean(provenance, "source_state_unchanged", provenance_where)
    _validate_source_state_boundaries(
        _required(provenance, "source_state_boundaries", provenance_where),
        where=f"{provenance_where}.source_state_boundaries",
    )
    for field in (
        "model_revision",
        "precision",
        "target",
        "bundle_build_id",
        "artifact_role",
    ):
        _nonempty_string(provenance, field, provenance_where)
    if provenance["artifact_role"] != expected_role:
        raise QualificationError(
            f"{provenance_where}.artifact_role must be {expected_role!r}"
        )
    for field in ("fresh_build", "artifact_reused"):
        _boolean(provenance, field, provenance_where)

    receipt_where = f"{label}.runtime_memory_receipt"
    receipt = _object(
        _required(result, "runtime_memory_receipt", label), receipt_where
    )
    for field in _RECEIPT_FIELDS[:3]:
        _positive_int(receipt, field, receipt_where)
    streaming = _required(receipt, "weight_streaming_active", receipt_where)
    if not isinstance(streaming, bool):
        raise QualificationError(
            f"{receipt_where}.weight_streaming_active must be boolean"
        )
    sources_where = f"{receipt_where}.measurement_sources"
    sources = _object(
        _required(receipt, "measurement_sources", receipt_where),
        sources_where,
    )
    for field, expected_source in _MEASUREMENT_SOURCES.items():
        source = _nonempty_string(sources, field, sources_where)
        if source != expected_source:
            raise QualificationError(
                f"{sources_where}.{field} must be {expected_source!r}"
            )

    plans_value = _required(result, "runtime_attention_plans", label)
    runtime_attention_plans = _validate_runtime_attention_plans(
        plans_value,
        where=f"{label}.runtime_attention_plans",
        artifact_role=expected_role,
    )
    stack_value = _required(result, "runtime_stack", label)
    runtime_stack = _validate_runtime_stack(
        stack_value,
        where=f"{label}.runtime_stack",
        artifact_role=expected_role,
    )
    libraries_value = _required(result, "runtime_libraries", label)
    runtime_libraries = _validate_runtime_libraries(
        libraries_value,
        where=f"{label}.runtime_libraries",
        artifact_role=expected_role,
        runtime_stack=runtime_stack,
    )
    runtime_trtmc_value = _required(
        result, "runtime_trtmc_libraries", label
    )
    runtime_trtmc_libraries = _validate_runtime_trtmc_libraries(
        runtime_trtmc_value,
        where=f"{label}.runtime_trtmc_libraries",
        model_id=model_id,
    )
    mapped_dso_value = _required(
        result,
        "mapped_dso_identities",
        label,
    )
    mapped_dso_identities = _validate_mapped_dso_identities(
        mapped_dso_value,
        where=f"{label}.mapped_dso_identities",
    )
    runtime_identity_rows = tuple(
        runtime_trtmc_libraries[field]
        for field in ("core", "trt_backend", "runtime_kv_plugin", "model")
    )
    if any(identity not in mapped_dso_identities for identity in runtime_identity_rows):
        raise QualificationError(
            f"{label}.mapped_dso_identities omits a mapped TRTMC runtime DSO"
        )
    if runtime_libraries is not None:
        for library_name in ("nvrtc", "nvrtc_builtins"):
            runtime_library = _object(
                runtime_libraries[library_name],
                f"{label}.runtime_libraries.{library_name}",
            )
            matches = [
                identity
                for identity in mapped_dso_identities
                if (
                    identity["path"] == runtime_library["path"]
                    and identity["size_bytes"]
                    == runtime_library["size_bytes"]
                    and identity["sha256"] == runtime_library["sha256"]
                )
            ]
            if len(matches) != 1:
                raise QualificationError(
                    f"{label}.mapped_dso_identities does not bind the live "
                    f"{library_name} DSO"
                )
    build_plugin_value = _required(
        result,
        "build_runtime_kv_plugin",
        label,
    )
    build_runtime_kv_plugin = _validate_build_runtime_kv_plugin(
        build_plugin_value,
        where=f"{label}.build_runtime_kv_plugin",
        artifact_role=expected_role,
    )
    cache_value = _required(result, "cuda_jit_cache", label)
    cuda_jit_cache = _validate_cuda_jit_cache(
        cache_value, where=f"{label}.cuda_jit_cache"
    )
    workload_value = _required(result, "generation_workload", label)
    generation_workload = _validate_generation_workload(
        workload_value,
        where=f"{label}.generation_workload",
        iterations=iterations,
        warmup=warmup,
        output_tokens=tuple(output_tokens),
    )
    tokenizer_value = _required(result, "tokenizer_contract", label)
    tokenizer_contract = _validate_tokenizer_contract(
        tokenizer_value,
        where=f"{label}.tokenizer_contract",
    )
    for value, field in (
        (plans_value, "runtime_attention_plans_sha256"),
        (stack_value, "runtime_stack_sha256"),
        (libraries_value, "runtime_libraries_sha256"),
        (
            runtime_trtmc_value,
            "runtime_trtmc_libraries_sha256",
        ),
        (
            mapped_dso_value,
            "mapped_dso_identities_sha256",
        ),
        (
            build_plugin_value,
            "build_runtime_kv_plugin_sha256",
        ),
        (cache_value, "cuda_jit_cache_sha256"),
        (workload_value, "generation_workload_sha256"),
        (tokenizer_value, "tokenizer_contract_sha256"),
    ):
        _validate_evidence_hash(
            value=value,
            provenance=provenance,
            field=field,
            provenance_where=provenance_where,
        )
    if runtime_stack is not None:
        expected_cudnn = _encoded_cudnn_version(
            runtime_stack["cudnn_backend"],
            f"{label}.runtime_stack.cudnn_backend",
        )
        if any(
            plan["cudnn_version"] != expected_cudnn
            for plan in runtime_attention_plans
        ):
            raise QualificationError(
                f"{label}.runtime_attention_plans cudnn_version disagrees "
                "with runtime_stack.cudnn_backend"
            )

    qualification_evidence, build_receipt = (
        _validate_qualification_evidence(
            result,
            label=label,
            expected_role=expected_role,
            expected_bundle=expected_bundle,
            model_id=model_id,
            provenance=provenance,
            runtime_trtmc_libraries=runtime_trtmc_libraries,
            mapped_dso_identities=mapped_dso_identities,
            build_runtime_kv_plugin=build_runtime_kv_plugin,
        )
    )
    raw_runtime_memory_contract = result.get(
        "bundle_runtime_memory_contract"
    )
    raw_module_residency_receipt = result.get(
        "runtime_module_residency_receipt"
    )
    runtime_memory_contract = (
        _object(
            raw_runtime_memory_contract,
            f"{label}.bundle_runtime_memory_contract",
        )
        if raw_runtime_memory_contract is not None
        else None
    )
    module_residency_receipt = (
        _object(
            raw_module_residency_receipt,
            f"{label}.runtime_module_residency_receipt",
        )
        if raw_module_residency_receipt is not None
        else None
    )
    build_manifest = _object(
        _required(qualification_evidence, "build_manifest", f"{label}.qualification_evidence"),
        f"{label}.qualification_evidence.build_manifest",
    )

    return CaseEvidence(
        label=label,
        path=resolved,
        result_sha256=hashlib.sha256(raw).hexdigest(),
        model_id=model_id,
        iterations=iterations,
        warmup=warmup,
        prefill_ms=tuple(prefill),
        decode_ms=tuple(decode),
        output_tokens=tuple(output_tokens),
        provenance=provenance,
        receipt=receipt,
        runtime_memory_contract=runtime_memory_contract,
        module_residency_receipt=module_residency_receipt,
        runtime_attention_plans=runtime_attention_plans,
        runtime_stack=runtime_stack,
        runtime_libraries=runtime_libraries,
        runtime_trtmc_libraries=runtime_trtmc_libraries,
        mapped_dso_identities=mapped_dso_identities,
        build_runtime_kv_plugin=build_runtime_kv_plugin,
        build_manifest=build_manifest,
        build_receipt=build_receipt,
        cuda_jit_cache=cuda_jit_cache,
        generation_workload=generation_workload,
        tokenizer_contract=tokenizer_contract,
    )


def _bundle_identity(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise QualificationError(f"{label}: cannot stat bundle: {exc}") from exc
    if not resolved.is_file() or size <= 0:
        raise QualificationError(f"{label} must be a non-empty file")
    return {"path": str(resolved), "sha256": _sha256(resolved), "bytes": size}


def _all_true(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_true(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_true(item) for item in value)
    return value if isinstance(value, bool) else True


def _representative_token_stream(case: CaseEvidence) -> tuple[int, ...]:
    streams = case.generation_workload["measured_generated_token_ids"]
    return tuple(streams[0])


def _common_prefix_length(
    lhs: Sequence[int], rhs: Sequence[int]
) -> int:
    for index, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return index
    return min(len(lhs), len(rhs))


def qualify(
    *,
    static_short: Path,
    dynamic_short: Path,
    static_medium: Path,
    dynamic_medium: Path,
    static_bundle: Path,
    dynamic_bundle: Path,
) -> dict[str, Any]:
    """Validate all evidence and return a machine-readable gate report."""

    resolved_static_bundle = static_bundle.expanduser().resolve()
    resolved_dynamic_bundle = dynamic_bundle.expanduser().resolve()
    cases = {
        "static_short": _read_case(
            static_short,
            "static-short",
            "exact-head-static-split",
            resolved_static_bundle,
        ),
        "dynamic_short": _read_case(
            dynamic_short,
            "dynamic-short",
            "native-dynamic",
            resolved_dynamic_bundle,
        ),
        "static_medium": _read_case(
            static_medium,
            "static-medium",
            "exact-head-static-split",
            resolved_static_bundle,
        ),
        "dynamic_medium": _read_case(
            dynamic_medium,
            "dynamic-medium",
            "native-dynamic",
            resolved_dynamic_bundle,
        ),
    }
    bundles = {
        "static": _bundle_identity(resolved_static_bundle, "static-bundle"),
        "dynamic": _bundle_identity(resolved_dynamic_bundle, "dynamic-bundle"),
    }
    values = list(cases.values())
    static_cases = (cases["static_short"], cases["static_medium"])
    dynamic_cases = (cases["dynamic_short"], cases["dynamic_medium"])

    shared_provenance_gates = {
        field: len({case.provenance[field] for case in values}) == 1
        for field in _SHARED_PROVENANCE_FIELDS
    }
    source_stable_gates = {
        case.label: (
            case.provenance["source_state_sha256"]
            == case.provenance["source_state_pre_sha256"]
            == case.provenance["source_state_post_sha256"]
            == case.provenance["prebuild_source_state_sha256"]
            == case.provenance["postbuild_source_state_sha256"]
            and case.provenance["source_state_unchanged"] is True
        )
        for case in values
    }
    source_commit_gates = {
        case.label: all(
            boundary["git_head"] == case.provenance["git_head"]
            for boundary in case.provenance[
                "source_state_boundaries"
            ].values()
        )
        for case in values
    }
    source_boundary_identity_gates = {
        case.label: all(
            boundary["source_state_sha256"]
            == case.provenance["source_state_sha256"]
            for boundary in case.provenance[
                "source_state_boundaries"
            ].values()
        )
        for case in values
    }
    clean_source_gates = {
        case.label: all(
            boundary["git_dirty"] is False
            for boundary in case.provenance[
                "source_state_boundaries"
            ].values()
        )
        for case in values
    }
    exact_head_gates = {
        case.label: all(
            boundary["exact_head_gate_satisfied"] is True
            for boundary in case.provenance[
                "source_state_boundaries"
            ].values()
        )
        for case in values
    }
    provenance_gates = {
        "shared_fields_match": shared_provenance_gates,
        "source_stable_prebuild_to_postbuild": source_stable_gates,
        "all_source_boundaries_match_expected_commit": source_commit_gates,
        "all_source_boundaries_match_source_state": (
            source_boundary_identity_gates
        ),
        "all_source_boundaries_clean": clean_source_gates,
        "all_source_boundaries_exact_head": exact_head_gates,
        "one_model_id": len({case.model_id for case in values}) == 1,
        "static_bundle_sha_matches_file": all(
            case.provenance["bundle_sha256"] == bundles["static"]["sha256"]
            for case in static_cases
        ),
        "dynamic_bundle_sha_matches_file": all(
            case.provenance["bundle_sha256"] == bundles["dynamic"]["sha256"]
            for case in dynamic_cases
        ),
        "static_and_dynamic_bundles_are_distinct": (
            bundles["static"]["sha256"] != bundles["dynamic"]["sha256"]
        ),
        "short_request_sha_matches": (
            cases["static_short"].provenance["request_sha256"]
            == cases["dynamic_short"].provenance["request_sha256"]
        ),
        "medium_request_sha_matches": (
            cases["static_medium"].provenance["request_sha256"]
            == cases["dynamic_medium"].provenance["request_sha256"]
        ),
        "short_and_medium_requests_are_distinct": (
            cases["static_short"].provenance["request_sha256"]
            != cases["static_medium"].provenance["request_sha256"]
        ),
        "static_build_id_matches_across_prompts": (
            static_cases[0].provenance["bundle_build_id"]
            == static_cases[1].provenance["bundle_build_id"]
        ),
        "dynamic_build_id_matches_across_prompts": (
            dynamic_cases[0].provenance["bundle_build_id"]
            == dynamic_cases[1].provenance["bundle_build_id"]
        ),
        "static_and_dynamic_build_ids_are_distinct": (
            static_cases[0].provenance["bundle_build_id"]
            != dynamic_cases[0].provenance["bundle_build_id"]
        ),
        "all_bundles_declared_fresh": all(
            case.provenance["fresh_build"] is True for case in values
        ),
        "no_reused_artifacts": all(
            case.provenance["artifact_reused"] is False for case in values
        ),
        "one_dynamic_build_runtime_kv_plugin": (
            len(
                {
                    _canonical_sha(case.build_runtime_kv_plugin)
                    for case in dynamic_cases
                }
            )
            == 1
        ),
        "static_build_has_no_runtime_kv_plugin": all(
            case.build_runtime_kv_plugin is None for case in static_cases
        ),
        "one_exact_head_build_manifest": (
            len({_canonical_sha(case.build_manifest) for case in values}) == 1
        ),
        "build_receipt_paths_match_cli_bundles": all(
            Path(str(case.build_receipt["bundle"])).resolve()
            == (
                resolved_static_bundle
                if case in static_cases
                else resolved_dynamic_bundle
            )
            for case in values
        ),
        "one_tokenizer_contract": (
            len(
                {
                    _canonical_sha(case.tokenizer_contract)
                    for case in values
                }
            )
            == 1
        ),
    }

    receipt_consistency = {
        f"{role}_{field}": (
            first.receipt[field] == second.receipt[field]
        )
        for role, (first, second) in {
            "static": static_cases,
            "dynamic": dynamic_cases,
        }.items()
        for field in _RECEIPT_FIELDS
    }
    runtime_evidence_consistency = {
        "dynamic_runtime_stack_matches_across_prompts": (
            dynamic_cases[0].runtime_stack == dynamic_cases[1].runtime_stack
        ),
        "dynamic_runtime_libraries_match_across_prompts": (
            dynamic_cases[0].runtime_libraries
            == dynamic_cases[1].runtime_libraries
        ),
        "runtime_trtmc_libraries_match_all_cases": (
            len(
                {
                    _canonical_sha(case.runtime_trtmc_libraries)
                    for case in values
                }
            )
            == 1
        ),
        "mapped_dso_identities_match_across_static_prompts": (
            static_cases[0].mapped_dso_identities
            == static_cases[1].mapped_dso_identities
        ),
        "mapped_dso_identities_match_across_dynamic_prompts": (
            dynamic_cases[0].mapped_dso_identities
            == dynamic_cases[1].mapped_dso_identities
        ),
        "static_runtime_stack_absent": all(
            case.runtime_stack is None for case in static_cases
        ),
        "static_runtime_libraries_absent": all(
            case.runtime_libraries is None for case in static_cases
        ),
        "static_runtime_attention_plans_absent": all(
            not case.runtime_attention_plans for case in static_cases
        ),
        "dynamic_runtime_attention_plans_present": all(
            bool(case.runtime_attention_plans) for case in dynamic_cases
        ),
        "cuda_jit_cache_present_for_all_cases": all(
            bool(case.cuda_jit_cache) for case in values
        ),
        "one_dynamic_bundle_runtime_memory_contract": (
            all(
                case.runtime_memory_contract is not None
                for case in dynamic_cases
            )
            and len(
                {
                    _canonical_sha(case.runtime_memory_contract)
                    for case in dynamic_cases
                }
            )
            == 1
        ),
        "dynamic_module_residency_provenance_matches": (
            all(
                case.module_residency_receipt is not None
                for case in dynamic_cases
            )
            and all(
                dynamic_cases[0].module_residency_receipt[field]
                == dynamic_cases[1].module_residency_receipt[field]
                for field in (
                    "receipt_schema_version",
                    "contract_version",
                    "module_residency_plan_set_sha256",
                    "module_residency_evidence_sha256",
                    "module_residency_cuda_module_loading_mode",
                )
            )
        ),
        "static_module_residency_contract_absent": all(
            case.runtime_memory_contract is None
            and case.module_residency_receipt is None
            for case in static_cases
        ),
    }
    performance: dict[str, Any] = {}
    for prompt_kind in ("short", "medium"):
        static = cases[f"static_{prompt_kind}"]
        dynamic = cases[f"dynamic_{prompt_kind}"]
        decode_ratio = (
            dynamic.decode_tokens_per_second
            / static.decode_tokens_per_second
        )
        prefill_ratio = dynamic.mean_prefill_ms / static.mean_prefill_ms
        performance[prompt_kind] = {
            "static_decode_tokens_per_second": (
                static.decode_tokens_per_second
            ),
            "dynamic_decode_tokens_per_second": (
                dynamic.decode_tokens_per_second
            ),
            "decode_throughput_ratio": decode_ratio,
            "static_mean_prefill_ms": static.mean_prefill_ms,
            "dynamic_mean_prefill_ms": dynamic.mean_prefill_ms,
            "prefill_ratio": prefill_ratio,
            "same_iterations": static.iterations == dynamic.iterations,
            "same_warmup": static.warmup == dynamic.warmup,
            "same_output_token_counts": (
                static.output_tokens == dynamic.output_tokens
            ),
            "same_fixed_length_structural_workload": (
                static.generation_workload["structural_identity_sha256"]
                == dynamic.generation_workload[
                    "structural_identity_sha256"
                ]
            ),
            "decode_throughput_gte_95_percent_static": (
                decode_ratio >= MIN_DYNAMIC_TO_STATIC_DECODE_RATIO
            ),
            "prefill_proxy_regression_lte_10_percent": (
                prefill_ratio <= MAX_DYNAMIC_TO_STATIC_PREFILL_RATIO
            ),
        }

    token_stream_diagnostics: dict[str, Any] = {}
    for prompt_kind in ("short", "medium"):
        static_stream = _representative_token_stream(
            cases[f"static_{prompt_kind}"]
        )
        dynamic_stream = _representative_token_stream(
            cases[f"dynamic_{prompt_kind}"]
        )
        common_prefix = _common_prefix_length(
            static_stream, dynamic_stream
        )
        token_stream_diagnostics[prompt_kind] = {
            "static_token_count": len(static_stream),
            "dynamic_token_count": len(dynamic_stream),
            "exact_generated_token_ids_match": (
                static_stream == dynamic_stream
            ),
            "common_prefix_tokens": common_prefix,
            "static_generated_token_ids_sha256": _canonical_sha(
                list(static_stream)
            ),
            "dynamic_generated_token_ids_sha256": _canonical_sha(
                list(dynamic_stream)
            ),
            "gate_effect": (
                "diagnostic_only; generated token values do not participate "
                "in the performance pass/fail decision"
            ),
        }

    static_receipt = static_cases[0].receipt
    dynamic_receipt = dynamic_cases[0].receipt
    packaging = {
        "bundle_dynamic_to_static_ratio": (
            bundles["dynamic"]["bytes"] / bundles["static"]["bytes"]
        ),
        "serialized_plan_dynamic_to_static_ratio": (
            dynamic_receipt["serialized_plan_bytes"]
            / static_receipt["serialized_plan_bytes"]
        ),
        "resident_weight_dynamic_to_static_ratio": (
            dynamic_receipt["resident_weight_bytes"]
            / static_receipt["resident_weight_bytes"]
        ),
        "bundle_bytes_lte_105_percent_static": (
            bundles["dynamic"]["bytes"]
            <= bundles["static"]["bytes"] * MAX_DYNAMIC_TO_STATIC_SIZE_RATIO
        ),
        "serialized_plan_bytes_lte_105_percent_static": (
            dynamic_receipt["serialized_plan_bytes"]
            <= static_receipt["serialized_plan_bytes"]
            * MAX_DYNAMIC_TO_STATIC_SIZE_RATIO
        ),
        "resident_weight_bytes_lte_105_percent_static": (
            dynamic_receipt["resident_weight_bytes"]
            <= static_receipt["resident_weight_bytes"]
            * MAX_DYNAMIC_TO_STATIC_SIZE_RATIO
        ),
        "resident_weight_copy_count_lte_2": {
            "static": static_receipt["resident_weight_copy_count"] <= 2,
            "dynamic": dynamic_receipt["resident_weight_copy_count"] <= 2,
        },
        "weight_streaming_disabled": {
            "static": static_receipt["weight_streaming_active"] is False,
            "dynamic": dynamic_receipt["weight_streaming_active"] is False,
        },
    }

    gates = {
        "provenance": provenance_gates,
        "receipt_consistency": receipt_consistency,
        "runtime_evidence_consistency": runtime_evidence_consistency,
        "performance": performance,
        "packaging": packaging,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if _all_true(gates) else "failed",
        "thresholds": {
            "minimum_dynamic_decode_fraction_of_static": (
                MIN_DYNAMIC_TO_STATIC_DECODE_RATIO
            ),
            "maximum_dynamic_prefill_fraction_of_static": (
                MAX_DYNAMIC_TO_STATIC_PREFILL_RATIO
            ),
            "maximum_dynamic_size_fraction_of_static": (
                MAX_DYNAMIC_TO_STATIC_SIZE_RATIO
            ),
        },
        "methodology": {
            "decode_throughput": (
                "sum(output_tokens) / sum(decode_ms), measured iterations only"
            ),
            "ttft_proxy": "arithmetic mean of worker-reported prefill_ms",
            "bundle_bytes": "actual CLI-supplied bundle file size",
            "plan_and_weight_bytes": (
                "runtime_memory_receipt with required TensorRT measurement "
                "sources"
            ),
            "runtime_attention": (
                "process-scoped cuDNN LSE graph-build/cache identities, "
                "live runtime-stack tuple, and CUDA JIT-cache snapshots "
                "with canonical hashes; this is not a per-invocation trace"
            ),
            "workload_equivalence": (
                "hard gate on identical prompt/generation/measurement "
                "structure, identical tokenizer contract, repeatable greedy "
                "streams within each case, unreachable EOS, and exactly "
                "max_new_tokens decoder outputs; static/dynamic generated "
                "token-ID equality is reported separately as diagnostic "
                "because numerically acceptable logits may diverge after a "
                "near-tie"
            ),
            "artifact_discovery": "none; every path is an explicit CLI input",
        },
        "bundles": bundles,
        "cases": {name: case.summary() for name, case in cases.items()},
        "gates": gates,
        "diagnostics": {
            "token_stream_equivalence": token_stream_diagnostics,
            "runtime_attention_plan_scope": {
                "proved": (
                    "cuDNN graph identities were emitted by the dynamic "
                    "worker process and bound to the captured runtime stack"
                ),
                "not_proved": (
                    "the graph-build/cache lines are not per-invocation "
                    "records and do not prove each measured decode step's "
                    "H, A, profile_id, or cuDNN plan identity"
                ),
                "per_invocation_H_A_profile_plan_proved": False,
            },
        },
    }
    return report


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _source_state_snapshot(
    repo_root: Path,
    artifact_dir: Path,
    *,
    label: str,
) -> Mapping[str, Any]:
    boundary = _load_boundary_module()
    return boundary.source_state_provenance(
        repo_root.resolve(),
        Path(__file__).resolve(),
        artifact_dir.resolve(),
        label=label,
    )


def _apply_standalone_source_state_gate(
    report: dict[str, Any],
    *,
    source_state_pre: Mapping[str, Any],
    source_state_post: Mapping[str, Any],
) -> None:
    pre_sha = source_state_pre.get("source_state_sha256")
    unchanged = bool(
        isinstance(pre_sha, str)
        and _SHA256.fullmatch(pre_sha) is not None
        and pre_sha == source_state_post.get("source_state_sha256")
        and source_state_pre.get("git_head")
        == source_state_post.get("git_head")
    )
    clean = (
        source_state_pre.get("git_dirty") is False
        and source_state_post.get("git_dirty") is False
    )
    exact_head = (
        source_state_pre.get("exact_head_gate_satisfied") is True
        and source_state_post.get("exact_head_gate_satisfied") is True
    )
    case_provenance = [
        case.get("qualification_provenance")
        for case in report.get("cases", {}).values()
        if isinstance(case, Mapping)
    ]
    matches_cases = bool(case_provenance) and all(
        isinstance(provenance, Mapping)
        and provenance.get("git_head") == source_state_pre.get("git_head")
        and provenance.get("source_state_sha256") == pre_sha
        for provenance in case_provenance
    )
    standalone_gate = {
        "source_state_unchanged": unchanged,
        "source_state_clean": clean,
        "source_state_exact_head": exact_head,
        "source_state_matches_all_captures": matches_cases,
    }
    standalone_gate["passed"] = all(standalone_gate.values())
    report["source_state_pre"] = dict(source_state_pre)
    report["source_state_post"] = dict(source_state_post)
    report["source_state_unchanged"] = unchanged
    if isinstance(report.get("gates"), dict):
        report["gates"]["standalone_source_state"] = standalone_gate
    else:
        report["standalone_source_state_gate"] = standalone_gate
    report["status"] = (
        "passed"
        if report.get("status") == "passed"
        and standalone_gate["passed"]
        else "failed"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-short", type=Path, required=True)
    parser.add_argument("--dynamic-short", type=Path, required=True)
    parser.add_argument("--static-medium", type=Path, required=True)
    parser.add_argument("--dynamic-medium", type=Path, required=True)
    parser.add_argument("--static-bundle", type=Path, required=True)
    parser.add_argument("--dynamic-bundle", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    source_state_pre: Mapping[str, Any] | None = None
    try:
        source_state_pre = _source_state_snapshot(
            args.repo_root,
            args.output.expanduser().resolve().parent,
            label="final-perf-pre",
        )
        report = qualify(
            static_short=args.static_short,
            dynamic_short=args.dynamic_short,
            static_medium=args.static_medium,
            dynamic_medium=args.dynamic_medium,
            static_bundle=args.static_bundle,
            dynamic_bundle=args.dynamic_bundle,
        )
    except (QualificationError, OSError, ValueError) as exc:
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "failed",
            "errors": [str(exc)],
        }
    try:
        source_state_post = _source_state_snapshot(
            args.repo_root,
            args.output.expanduser().resolve().parent,
            label="final-perf-post",
        )
        if source_state_pre is None:
            raise QualificationError(
                "final performance gate has no pre-run source-state sample"
            )
        _apply_standalone_source_state_gate(
            report,
            source_state_pre=source_state_pre,
            source_state_post=source_state_post,
        )
    except (QualificationError, OSError, ValueError) as exc:
        report["status"] = "failed"
        report.setdefault("errors", []).append(str(exc))
    _write_report(args.output, report)
    print(
        json.dumps(
            {"status": report["status"], "output": str(args.output.resolve())},
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
