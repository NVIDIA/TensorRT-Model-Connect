#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the developer-only C/2 native dynamic-memory qualification bundle.

This is not a product build surface.  It accepts only one of the two exact
native dynamic-memory qualifications, derives the single canonical C/2
prefill/profile variant, invokes the existing qualified native builder, and
writes a SHA-bound receipt for ``qualify_native_dynamic_memory.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from tensorrt_model_connect.dynamic_memory_contract import (  # noqa: E402
    DEVELOPER_CHUNK_VARIANT_ENV,
    DEVELOPER_CHUNK_VARIANT_VALUE,
    DynamicMemoryContractError,
    ResolvedDynamicMemoryQualification,
    derive_developer_chunk_variant_qualification,
    require_developer_chunk_variant_opt_in,
    resolve_model_only_qualification,
    validate_runtime_memory_contract,
)


BUNDLE_MAGIC = b"TRTFB\x00\x01\x00"
SCHEMA = "trtmc.native-dynamic-memory-chunk-variant-build/v2"
BUILD_MANIFEST_SCHEMA = "trtmc.dynamic-memory-test-manifest/v2"
RUNTIME_KV_PLUGIN_ENV = "TRTMC_TRT_PLUGIN_LIBRARY"
RUNTIME_KV_PLUGIN_ABI_SYMBOL = (
    "trtmc_runtime_kv_plugin_abi_version"
)
_BINARY_IDENTITY_FIELDS = (
    "path",
    "device",
    "inode",
    "size_bytes",
    "mtime_ns",
    "ctime_ns",
    "sha256",
)


class ChunkVariantBuildError(RuntimeError):
    """The requested build is not the one legal developer C/2 variant."""


def _load_build_manifest_module() -> Any:
    path = Path(__file__).with_name(
        "capture_dynamic_memory_test_manifest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_trtmc_chunk_variant_build_manifest", path
    )
    if spec is None or spec.loader is None:
        raise ChunkVariantBuildError(
            f"cannot load build-manifest validator: {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_strict_build_manifest(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        canonical = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ChunkVariantBuildError(
            f"cannot resolve exact-head build manifest {path}: {exc}"
        ) from exc
    module = _load_build_manifest_module()
    try:
        manifest = module.load_and_validate_build_manifest(canonical)
    except Exception as exc:
        if exc.__class__.__name__ != "ManifestError":
            raise
        raise ChunkVariantBuildError(
            f"invalid exact-head build manifest: {exc}"
        ) from exc
    if manifest.get("schema_version") != BUILD_MANIFEST_SCHEMA:
        raise ChunkVariantBuildError(
            "exact-head build manifest has the wrong schema"
        )
    try:
        manifest_repo = Path(
            str(manifest["repo_root"])
        ).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise ChunkVariantBuildError(
            "exact-head build manifest repo root is invalid"
        ) from exc
    if manifest_repo != REPO_ROOT:
        raise ChunkVariantBuildError(
            "exact-head build manifest belongs to a different source tree"
        )
    source_pre = manifest.get("source_state_pre")
    source_post = manifest.get("source_state_post")
    if (
        not isinstance(source_pre, Mapping)
        or not isinstance(source_post, Mapping)
        or source_pre.get("exact_head_gate_satisfied") is not True
        or source_post.get("exact_head_gate_satisfied") is not True
        or source_pre.get("git_head") != source_post.get("git_head")
        or source_pre.get("source_state_sha256")
        != source_post.get("source_state_sha256")
    ):
        raise ChunkVariantBuildError(
            "exact-head build manifest source boundaries are invalid"
        )
    artifacts = manifest.get("build_artifacts")
    plugin = (
        artifacts.get("runtime_kv_plugin")
        if isinstance(artifacts, Mapping)
        else None
    )
    if not isinstance(plugin, Mapping):
        raise ChunkVariantBuildError(
            "exact-head build manifest has no runtime-KV plugin artifact"
        )
    try:
        plugin_identity = {
            "path": plugin["path"],
            "device": plugin["st_dev"],
            "inode": plugin["st_ino"],
            "size_bytes": plugin["size_bytes"],
            "mtime_ns": plugin["mtime_ns"],
            "ctime_ns": plugin["ctime_ns"],
            "sha256": plugin["sha256"],
        }
        binding = {
            "path": str(canonical),
            "sha256": _sha256(canonical),
            "schema_version": BUILD_MANIFEST_SCHEMA,
            "git_head": source_pre["git_head"],
            "source_state_sha256": source_pre[
                "source_state_sha256"
            ],
            "build_artifacts_sha256": manifest[
                "build_artifacts_sha256"
            ],
        }
    except KeyError as exc:
        raise ChunkVariantBuildError(
            "exact-head build manifest is missing required provenance"
        ) from exc
    return binding, plugin_identity


def _require_manifest_plugin_match(
    manifest_plugin: Mapping[str, Any],
    selected_plugin: Mapping[str, Any],
) -> None:
    if dict(manifest_plugin) != dict(selected_plugin):
        raise ChunkVariantBuildError(
            "selected runtime-KV plugin does not match the exact-head "
            "build manifest"
        )


def _load_boundary_module() -> Any:
    path = Path(__file__).with_name("qualify_native_dynamic_memory.py")
    spec = importlib.util.spec_from_file_location(
        "_trtmc_dynamic_memory_variant_boundary", path
    )
    if spec is None or spec.loader is None:
        raise ChunkVariantBuildError(
            f"cannot load source-state helper: {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_state_snapshot(
    artifact_dir: Path, *, label: str
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    try:
        relative = artifact_dir.relative_to(REPO_ROOT)
    except ValueError:
        relative = None
    if relative is not None:
        top_level = relative.parts[0] if relative.parts else ""
        if not (
            top_level == "artifacts"
            or top_level == "build"
            or top_level.startswith("build-")
        ):
            raise ChunkVariantBuildError(
                "developer C/2 output inside the repository must be under "
                "artifacts/, build/, or build-* so source snapshots exclude it"
            )
    boundary = _load_boundary_module()
    return boundary.source_state_provenance(
        REPO_ROOT,
        Path(__file__).resolve(),
        artifact_dir,
        label=label,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(fd, 8 * 1024 * 1024, offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    try:
        path = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ChunkVariantBuildError(
            f"cannot resolve receipt artifact {path}: {exc}"
        ) from exc
    if not path.is_file():
        raise ChunkVariantBuildError(
            f"receipt artifact is not a regular file: {path}"
        )
    stat = path.stat()
    if stat.st_size <= 0:
        raise ChunkVariantBuildError(f"receipt artifact is empty: {path}")
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _binary_identity_from_fd(path: Path, fd: int) -> dict[str, Any]:
    observed = os.fstat(fd)
    if not stat.S_ISREG(observed.st_mode):
        raise ChunkVariantBuildError(
            f"runtime-KV plugin is not a regular file: {path}"
        )
    if observed.st_size <= 0:
        raise ChunkVariantBuildError(
            f"runtime-KV plugin is empty: {path}"
        )
    return {
        "path": str(path),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "size_bytes": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
        "sha256": _sha256_fd(fd),
    }


@contextlib.contextmanager
def _pinned_binary(path: Path):
    try:
        canonical = path.expanduser().resolve(strict=True)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(canonical, flags)
    except OSError as exc:
        raise ChunkVariantBuildError(
            f"cannot pin runtime-KV plugin {path}: {exc}"
        ) from exc
    try:
        identity = _binary_identity_from_fd(canonical, fd)
        yield fd, identity
    finally:
        os.close(fd)


def _verify_pinned_binary(
    path: Path,
    fd: int,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    canonical = path.expanduser().resolve(strict=True)
    fd_identity = _binary_identity_from_fd(canonical, fd)
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        path_fd = os.open(canonical, flags)
    except OSError as exc:
        raise ChunkVariantBuildError(
            f"cannot reopen pinned runtime-KV plugin {canonical}: {exc}"
        ) from exc
    try:
        path_identity = _binary_identity_from_fd(canonical, path_fd)
    finally:
        os.close(path_fd)
    if (
        any(fd_identity.get(field) != expected.get(field)
            for field in _BINARY_IDENTITY_FIELDS)
        or any(path_identity.get(field) != expected.get(field)
               for field in _BINARY_IDENTITY_FIELDS)
    ):
        raise ChunkVariantBuildError(
            "runtime-KV plugin DSO changed while building the C/2 bundle"
        )
    return fd_identity


def _binary_identity(path: Path) -> dict[str, Any]:
    with _pinned_binary(path) as (fd, identity):
        return _verify_pinned_binary(path, fd, identity)


def _select_runtime_kv_plugin_path(
    explicit_plugin: Path | None,
) -> Path:
    from tensorrt_model_connect import trt_plugins

    env_was_set = RUNTIME_KV_PLUGIN_ENV in os.environ
    raw_env = os.environ.get(RUNTIME_KV_PLUGIN_ENV, "")
    if env_was_set and not raw_env:
        raise ChunkVariantBuildError(
            f"{RUNTIME_KV_PLUGIN_ENV} was explicitly set but is empty"
        )
    try:
        explicit_path = (
            explicit_plugin.expanduser().resolve(strict=True)
            if explicit_plugin is not None
            else None
        )
        environment_path = (
            Path(raw_env).expanduser().resolve(strict=True)
            if env_was_set
            else None
        )
    except OSError as exc:
        raise ChunkVariantBuildError(
            f"cannot resolve runtime-KV plugin DSO: {exc}"
        ) from exc
    if (
        explicit_path is not None
        and environment_path is not None
        and explicit_path != environment_path
    ):
        raise ChunkVariantBuildError(
            "--plugin-library conflicts with "
            f"{RUNTIME_KV_PLUGIN_ENV}; refusing to choose between DSOs"
        )
    if explicit_path is not None:
        return explicit_path
    if environment_path is not None:
        return environment_path
    try:
        selected = trt_plugins._select_runtime_kv_plugin()
    except RuntimeError as exc:
        raise ChunkVariantBuildError(
            f"cannot prove one runtime-KV plugin DSO: {exc}"
        ) from exc
    try:
        return selected.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ChunkVariantBuildError(
            f"cannot resolve selected runtime-KV plugin DSO: {exc}"
        ) from exc


@contextlib.contextmanager
def _bound_runtime_kv_plugin(path: Path):
    existed = RUNTIME_KV_PLUGIN_ENV in os.environ
    previous = os.environ.get(RUNTIME_KV_PLUGIN_ENV)
    os.environ[RUNTIME_KV_PLUGIN_ENV] = str(path)
    try:
        yield
    finally:
        if existed:
            assert previous is not None
            os.environ[RUNTIME_KV_PLUGIN_ENV] = previous
        else:
            os.environ.pop(RUNTIME_KV_PLUGIN_ENV, None)


def _loaded_runtime_kv_plugin_path() -> Path | None:
    from tensorrt_model_connect.trt_plugins import (
        loaded_runtime_kv_plugin_path,
    )

    loaded = loaded_runtime_kv_plugin_path()
    return loaded.resolve() if loaded is not None else None


def _file_contains(path: Path, needle: bytes) -> bool:
    overlap = max(len(needle) - 1, 0)
    previous = b""
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                return False
            payload = previous + chunk
            if needle in payload:
                return True
            previous = payload[-overlap:] if overlap else b""


def _exports_runtime_kv_plugin_abi(path: Path) -> bool:
    symbol = RUNTIME_KV_PLUGIN_ABI_SYMBOL
    if not _file_contains(path, symbol.encode("ascii")):
        return False
    try:
        completed = subprocess.run(
            ["nm", "-D", "--defined-only", str(path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise ChunkVariantBuildError(
            "cannot prove runtime-KV plugin DSO uniqueness because "
            f"nm failed for {path}: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise ChunkVariantBuildError(
            "cannot prove runtime-KV plugin DSO uniqueness because "
            f"nm could not inspect {path}"
        )
    for line in completed.stdout.splitlines():
        fields = line.split()
        if fields and fields[-1].split("@", 1)[0] == symbol:
            return True
    return False


def _proc_self_maps_lines() -> list[str]:
    try:
        return Path("/proc/self/maps").read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError as exc:
        raise ChunkVariantBuildError(
            f"cannot inspect current process library mappings: {exc}"
        ) from exc


def _runtime_kv_plugin_mapping_evidence(
    selected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    lines = _proc_self_maps_lines()
    selected_path = Path(str(selected_identity["path"]))
    selected_basename = selected_path.name
    deleted_shared_libraries: list[str] = []
    unique_mappings: dict[
        tuple[str, int, int], dict[str, Any]
    ] = {}
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw = fields[5]
        if raw.endswith(" (deleted)"):
            deleted_path = raw.removesuffix(" (deleted)")
            if (
                "x" in fields[1]
                or Path(deleted_path).name == selected_basename
            ):
                deleted_shared_libraries.append(deleted_path)
            continue
        try:
            mapped_path = Path(raw).resolve(strict=True)
            device_parts = fields[3].split(":", 1)
            if len(device_parts) != 2:
                raise ValueError
            device = os.makedev(
                int(device_parts[0], 16),
                int(device_parts[1], 16),
            )
            inode = int(fields[4])
        except (OSError, ValueError):
            continue
        if inode <= 0:
            continue
        key = (str(mapped_path), device, inode)
        mapping = unique_mappings.setdefault(
            key,
            {
                "path": str(mapped_path),
                "device": device,
                "inode": inode,
                "_executable": False,
            },
        )
        mapping["_executable"] = bool(
            mapping["_executable"] or "x" in fields[1]
        )
    if deleted_shared_libraries:
        raise ChunkVariantBuildError(
            "cannot prove runtime-KV plugin DSO uniqueness with deleted "
            "shared-library mappings: "
            + ", ".join(sorted(set(deleted_shared_libraries)))
        )

    candidates: list[dict[str, Any]] = []
    for mapping in unique_mappings.values():
        mapped_path = Path(mapping["path"])
        if (
            mapped_path == selected_path
            or mapped_path.name == selected_basename
            or (
                mapping["_executable"]
                and _exports_runtime_kv_plugin_abi(mapped_path)
            )
        ):
            candidates.append(
                {
                    "path": mapping["path"],
                    "device": mapping["device"],
                    "inode": mapping["inode"],
                }
            )
    candidates.sort(
        key=lambda item: (
            str(item["path"]),
            int(item["device"]),
            int(item["inode"]),
        )
    )
    expected_mapping = {
        "path": str(selected_path),
        "device": selected_identity["device"],
        "inode": selected_identity["inode"],
    }
    if candidates != [expected_mapping]:
        raise ChunkVariantBuildError(
            "qualified builder must map exactly the pinned runtime-KV "
            f"plugin DSO; observed candidates: {candidates}"
        )
    return {
        "schema_version": 1,
        "source": "/proc/self/maps",
        "pid": os.getpid(),
        "selection_rule": (
            "selected_path_or_same_basename_or_exported_abi_symbol"
        ),
        "abi_symbol": RUNTIME_KV_PLUGIN_ABI_SYMBOL,
        "candidate_count": 1,
        "deleted_candidate_count": 0,
        "selected": dict(selected_identity),
        "candidate_mappings": candidates,
    }


def _read_bundle_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as bundle:
        if bundle.read(8) != BUNDLE_MAGIC:
            raise ChunkVariantBuildError(
                f"qualified builder did not produce a TRTMC bundle: {path}"
            )
        raw_length = bundle.read(8)
        if len(raw_length) != 8:
            raise ChunkVariantBuildError(
                f"qualified bundle has a truncated header length: {path}"
            )
        header_length = struct.unpack("<Q", raw_length)[0]
        payload = bundle.read(header_length)
        if len(payload) != header_length:
            raise ChunkVariantBuildError(
                f"qualified bundle has a truncated JSON header: {path}"
            )
    try:
        header = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ChunkVariantBuildError(
            f"qualified bundle has invalid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(header, dict):
        raise ChunkVariantBuildError(
            "qualified bundle JSON header must be an object"
        )
    return header


def _bundle_plan_section_sha256(
    path: Path,
    header: Mapping[str, Any],
) -> dict[str, str]:
    """Hash both split plans from their exact on-disk bundle ranges."""

    sections = header.get("sections")
    if not isinstance(sections, Mapping):
        raise ChunkVariantBuildError(
            "qualified C/2 bundle has no section table"
        )
    required_names = ("engine_plan", "prefill_engine_plan")
    spans: dict[str, tuple[int, int]] = {}
    for name in required_names:
        section = sections.get(name)
        if not isinstance(section, Mapping):
            raise ChunkVariantBuildError(
                f"qualified C/2 bundle has no {name} section"
            )
        offset = section.get("offset")
        size = section.get("size")
        if (
            type(offset) is not int
            or offset < 0
            or type(size) is not int
            or size <= 0
        ):
            raise ChunkVariantBuildError(
                f"qualified C/2 bundle has an invalid {name} span"
            )
        spans[name] = (offset, size)

    ordered_spans = sorted(
        (offset, offset + size, name)
        for name, (offset, size) in spans.items()
    )
    if ordered_spans[0][1] > ordered_spans[1][0]:
        raise ChunkVariantBuildError(
            "qualified C/2 split plan sections overlap"
        )

    hashes: dict[str, str] = {}
    with path.open("rb") as bundle:
        if bundle.read(8) != BUNDLE_MAGIC:
            raise ChunkVariantBuildError(
                f"qualified builder did not produce a TRTMC bundle: {path}"
            )
        raw_length = bundle.read(8)
        if len(raw_length) != 8:
            raise ChunkVariantBuildError(
                f"qualified bundle has a truncated header length: {path}"
            )
        header_length = struct.unpack("<Q", raw_length)[0]
        payload_offset = 16 + header_length
        file_size = os.fstat(bundle.fileno()).st_size
        payload_size = file_size - payload_offset
        if payload_size < 0:
            raise ChunkVariantBuildError(
                f"qualified bundle has a truncated JSON header: {path}"
            )
        for name in required_names:
            offset, size = spans[name]
            if offset > payload_size or size > payload_size - offset:
                raise ChunkVariantBuildError(
                    f"qualified C/2 bundle {name} extends beyond the file"
                )
            bundle.seek(payload_offset + offset)
            remaining = size
            digest = hashlib.sha256()
            while remaining:
                chunk = bundle.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise ChunkVariantBuildError(
                        f"qualified C/2 bundle has a truncated {name}"
                    )
                digest.update(chunk)
                remaining -= len(chunk)
            hashes[name] = digest.hexdigest()
    return hashes


def _expected_contract(
    qualification: ResolvedDynamicMemoryQualification,
) -> dict[str, Any]:
    record = qualification.qualification
    return {
        # v1 exists only while the builder has not serialized both plans.
        # This producer validates the final user-consumable artifact.
        "contract_version": 2,
        "qualified_model_id": record.qualified_model_id,
        "qualified_model_revision": record.qualified_model_revision,
        "qualified_config_sha256": record.qualified_config_sha256,
        "qualified_target": record.qualified_target,
        "qualified_runtime_stack": {
            "sm": record.gpu_architecture,
            "tensorrt": record.minimum_trt_version,
            "cuda_runtime": record.cuda_runtime,
            "cudnn_backend": record.cudnn_backend,
            "cudnn_frontend_revision":
                record.cudnn_frontend_revision,
            "nvrtc": record.nvrtc,
            "driver": record.driver,
        },
        "native_kv_plugin_abi": record.native_kv_plugin_abi,
        "model_context_limit": record.model_context_limit,
        "prefill_chunk_limit": record.prefill_chunk_limit,
        "kv_layout": record.kv_layout,
        "kv_dtype": record.kv_dtype,
        "active_kv_profile_limits": list(
            record.active_kv_profile_limits
        ),
        "runtime_owned": True,
    }


def _validate_built_bundle(
    bundle: Path,
    qualification: ResolvedDynamicMemoryQualification,
) -> tuple[dict[str, Any], dict[str, Any]]:
    header = _read_bundle_header(bundle)
    raw_contract = header.get("runtime_memory")
    if not isinstance(raw_contract, Mapping):
        raise ChunkVariantBuildError(
            "qualified C/2 bundle has no runtime_memory contract"
        )
    try:
        contract = validate_runtime_memory_contract(raw_contract)
    except DynamicMemoryContractError as exc:
        raise ChunkVariantBuildError(
            f"qualified C/2 bundle contract is invalid: {exc}"
        ) from exc
    expected = _expected_contract(qualification)
    mismatches = {
        field: {
            "expected": expected_value,
            "actual": contract.get(field),
        }
        for field, expected_value in expected.items()
        if contract.get(field) != expected_value
    }
    record = qualification.qualification
    top_level_expected = {
        "model_id": record.qualified_model_id,
        "family": record.family,
        "max_cache_length": record.model_context_limit,
        "precision": record.precision,
    }
    for field, expected_value in top_level_expected.items():
        if header.get(field) != expected_value:
            mismatches[field] = {
                "expected": expected_value,
                "actual": header.get(field),
            }
    sections = header.get("sections")
    if not isinstance(sections, Mapping):
        mismatches["sections"] = {
            "expected": "engine_plan and prefill_engine_plan",
            "actual": sections,
        }
    else:
        missing_sections = sorted(
            {"engine_plan", "prefill_engine_plan"} - set(sections)
        )
        if missing_sections:
            mismatches["sections"] = {
                "expected": "engine_plan and prefill_engine_plan",
                "actual_missing": missing_sections,
            }
        else:
            actual_plan_hashes = _bundle_plan_section_sha256(
                bundle, header
            )
            calibration = contract.get(
                "module_residency_calibration"
            )
            declared_plan_hashes = (
                {
                    plan["section_name"]: plan["section_sha256"]
                    for plan in calibration["plans"]
                }
                if isinstance(calibration, Mapping)
                and isinstance(calibration.get("plans"), list)
                else {}
            )
            if declared_plan_hashes != actual_plan_hashes:
                mismatches["module_residency_calibration.plans"] = {
                    "expected_exact_bundle_plan_sha256":
                        actual_plan_hashes,
                    "actual_declared_plan_sha256":
                        declared_plan_hashes,
                }
    if mismatches:
        raise ChunkVariantBuildError(
            "qualified builder produced the wrong C/2 bundle facts: "
            f"{mismatches}"
        )
    return header, contract


def _resolve_default_qualification(
    model: str,
    revision: str | None,
) -> ResolvedDynamicMemoryQualification | None:
    from tensorrt_model_connect.engine_builder import _resolve_model

    return resolve_model_only_qualification(
        model,
        requested_revision=revision,
        resolve_model=_resolve_model,
    )


def _invoke_qualified_builder(
    qualification: ResolvedDynamicMemoryQualification,
    *,
    output: Path,
    build_timing: Path,
    verbose: bool,
) -> None:
    from tensorrt_model_connect.engine_builder import (
        _build_native_impl_qualified,
    )

    record = qualification.qualification
    _build_native_impl_qualified(
        runtime_memory_qualification=qualification,
        model_id_or_path=str(qualification.model_dir),
        output_path=str(output),
        max_cache_length=record.model_context_limit,
        decoder_engine_layout="split",
        dynamic_kv_cache=True,
        dynamic_kv_profile_rows_override=list(
            record.active_kv_profile_limits
        ),
        precision=record.precision,
        verbose=verbose,
        build_timing_path=str(build_timing),
    )


def _require_fresh_paths(paths: Mapping[str, Path]) -> None:
    rendered = [str(path) for path in paths.values()]
    if len(rendered) != len(set(rendered)):
        raise ChunkVariantBuildError(
            "bundle, receipt, and build-timing paths must be distinct"
        )
    existing = [
        f"{label}={path}"
        for label, path in paths.items()
        if path.exists()
    ]
    if existing:
        raise ChunkVariantBuildError(
            "developer C/2 build requires fresh output paths: "
            + ", ".join(existing)
        )


def build_chunk_variant(
    *,
    model: str,
    revision: str | None,
    output: Path,
    receipt: Path,
    build_timing: Path,
    verbose: bool,
    plugin_library: Path | None = None,
    build_manifest: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    receipt = receipt.expanduser().resolve()
    build_timing = build_timing.expanduser().resolve()
    paths = {
        "bundle": output,
        "receipt": receipt,
        "build_timing": build_timing,
    }
    _require_fresh_paths(paths)
    try:
        require_developer_chunk_variant_opt_in(environment)
    except DynamicMemoryContractError as exc:
        raise ChunkVariantBuildError(str(exc)) from exc

    if build_manifest is None:
        raise ChunkVariantBuildError(
            "developer C/2 qualification requires --build-manifest from "
            "the clean exact-head binary build"
        )
    build_manifest_identity, manifest_plugin = (
        _load_strict_build_manifest(build_manifest)
    )
    selected_plugin_path = _select_runtime_kv_plugin_path(plugin_library)
    with _pinned_binary(selected_plugin_path) as (
        plugin_fd,
        runtime_kv_plugin,
    ):
        _require_manifest_plugin_match(
            manifest_plugin, runtime_kv_plugin
        )
        with _bound_runtime_kv_plugin(selected_plugin_path):
            default = _resolve_default_qualification(model, revision)
            if default is None:
                raise ChunkVariantBuildError(
                    "developer C/2 build requires one of the two exact "
                    "qualified model snapshots"
                )
            try:
                variant = derive_developer_chunk_variant_qualification(
                    default,
                    environment=environment,
                )
            except DynamicMemoryContractError as exc:
                raise ChunkVariantBuildError(str(exc)) from exc

            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            source_artifact_dir = (
                receipt.parent / f"{receipt.stem}-source-state"
            )
            source_state_pre = _source_state_snapshot(
                source_artifact_dir, label="prebuild"
            )
            if (
                source_state_pre.get("git_head")
                != build_manifest_identity["git_head"]
                or source_state_pre.get("source_state_sha256")
                != build_manifest_identity["source_state_sha256"]
            ):
                raise ChunkVariantBuildError(
                    "current source does not match the exact-head build "
                    "manifest"
                )
            _invoke_qualified_builder(
                variant,
                output=output,
                build_timing=build_timing,
                verbose=verbose,
            )
            loaded_plugin_path = _loaded_runtime_kv_plugin_path()
            if loaded_plugin_path != selected_plugin_path:
                raise ChunkVariantBuildError(
                    "qualified native builder did not load the selected "
                    "runtime-KV plugin DSO"
                )
            runtime_kv_plugin_mapping = (
                _runtime_kv_plugin_mapping_evidence(runtime_kv_plugin)
            )
            _verify_pinned_binary(
                selected_plugin_path,
                plugin_fd,
                runtime_kv_plugin,
            )
            if not output.is_file():
                raise ChunkVariantBuildError(
                    "qualified native builder produced no C/2 bundle"
                )
            if not build_timing.is_file():
                raise ChunkVariantBuildError(
                    "qualified native builder produced no build-timing "
                    "artifact"
                )
            _header, contract = _validate_built_bundle(output, variant)
            source_state_post = _source_state_snapshot(
                source_artifact_dir, label="postbuild"
            )
            _verify_pinned_binary(
                selected_plugin_path,
                plugin_fd,
                runtime_kv_plugin,
            )
    source_state_unchanged = (
        source_state_pre["git_head"] == source_state_post["git_head"]
        and source_state_pre["source_state_sha256"]
        == source_state_post["source_state_sha256"]
    )
    if not source_state_unchanged:
        raise ChunkVariantBuildError(
            "source state changed while building the developer C/2 bundle"
        )

    default_record = default.qualification
    variant_record = variant.qualification
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "developer_only": True,
        "opt_in": {
            "environment": DEVELOPER_CHUNK_VARIANT_ENV,
            "value": DEVELOPER_CHUNK_VARIANT_VALUE,
        },
        "builder_entrypoint": (
            "tensorrt_model_connect.engine_builder."
            "_build_native_impl_qualified"
        ),
        "qualified_model": {
            "model_id": variant_record.qualified_model_id,
            "revision": variant_record.qualified_model_revision,
            "config_sha256":
                variant_record.qualified_config_sha256,
            "target": variant_record.qualified_target,
            "model_dir": str(variant.model_dir),
        },
        "default_policy": {
            "prefill_chunk_limit":
                default_record.prefill_chunk_limit,
            "active_kv_profile_limits": list(
                default_record.active_kv_profile_limits
            ),
        },
        "variant_policy": {
            "prefill_chunk_limit":
                variant_record.prefill_chunk_limit,
            "active_kv_profile_limits": list(
                variant_record.active_kv_profile_limits
            ),
        },
        "bundle": _file_identity(output),
        "build_timing": _file_identity(build_timing),
        "producer": _file_identity(Path(__file__).resolve()),
        "runtime_kv_plugin": runtime_kv_plugin,
        "runtime_kv_plugin_mapping": runtime_kv_plugin_mapping,
        "build_manifest": build_manifest_identity,
        "runtime_memory": contract,
        "fresh_build": True,
        "artifact_reused": False,
        "source_state_pre": source_state_pre,
        "source_state_post": source_state_post,
        "source_state_unchanged": source_state_unchanged,
    }
    receipt.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Exact qualified HF model ID or pinned local snapshot",
    )
    parser.add_argument(
        "--model-revision",
        help="Optional immutable revision; must match the qualification",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--plugin-library",
        type=Path,
        help=(
            "Qualification-only explicit runtime-KV DSO binding; this is not "
            "a product build or context-length flag"
        ),
    )
    parser.add_argument(
        "--build-manifest",
        type=Path,
        required=True,
        help=(
            "Required clean exact-head "
            "trtmc.dynamic-memory-test-manifest/v2; its source and runtime "
            "plugin identities are bound into the C/2 receipt"
        ),
    )
    parser.add_argument("--build-timing-json", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    build_timing = (
        args.build_timing_json
        if args.build_timing_json is not None
        else args.output.with_suffix(
            args.output.suffix + ".build-timing.json"
        )
    )
    try:
        report = build_chunk_variant(
            model=args.model,
            revision=args.model_revision,
            output=args.output,
            receipt=args.receipt,
            build_timing=build_timing,
            verbose=args.verbose,
            plugin_library=args.plugin_library,
            build_manifest=args.build_manifest,
        )
    except (ChunkVariantBuildError, OSError, ValueError) as exc:
        print(
            f"build_native_dynamic_memory_chunk_variant: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "bundle": report["bundle"]["path"],
                "bundle_sha256": report["bundle"]["sha256"],
                "prefill_chunk_limit": report[
                    "variant_policy"
                ]["prefill_chunk_limit"],
                "receipt": str(args.receipt.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
