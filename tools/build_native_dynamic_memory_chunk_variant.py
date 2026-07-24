#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the developer-only C/2 native dynamic-memory qualification bundle.

This is not a product build surface.  It accepts only one of the two exact
native dynamic-memory qualifications, derives the single canonical C/2
prefill/profile variant, invokes the existing qualified native builder, and
writes a SHA-bound receipt for ``qualify_native_dynamic_memory.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
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
SCHEMA = "trtmc.native-dynamic-memory-chunk-variant-build/v1"


class ChunkVariantBuildError(RuntimeError):
    """The requested build is not the one legal developer C/2 variant."""


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


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
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


def _expected_contract(
    qualification: ResolvedDynamicMemoryQualification,
) -> dict[str, Any]:
    record = qualification.qualification
    return {
        "contract_version": 1,
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

    default = _resolve_default_qualification(model, revision)
    if default is None:
        raise ChunkVariantBuildError(
            "developer C/2 build requires one of the two exact qualified "
            "model snapshots"
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
    source_artifact_dir = receipt.parent / f"{receipt.stem}-source-state"
    source_state_pre = _source_state_snapshot(
        source_artifact_dir, label="prebuild"
    )
    _invoke_qualified_builder(
        variant,
        output=output,
        build_timing=build_timing,
        verbose=verbose,
    )
    if not output.is_file():
        raise ChunkVariantBuildError(
            "qualified native builder produced no C/2 bundle"
        )
    if not build_timing.is_file():
        raise ChunkVariantBuildError(
            "qualified native builder produced no build-timing artifact"
        )
    _header, contract = _validate_built_bundle(output, variant)
    source_state_post = _source_state_snapshot(
        source_artifact_dir, label="postbuild"
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
