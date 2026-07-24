#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification-only streaming reseal of a native dynamic-memory bundle.

This tool bootstraps exact-plan qualification sweeps from an existing v1
bundle.  It hashes only the two split TensorRT plans, replaces the JSON header
with a validated v2 runtime-memory contract, and streams the existing payload
unchanged into an atomic output file.  It is not a product build surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import struct
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from tensorrt_model_connect.dynamic_memory_contract import (  # noqa: E402
    DynamicMemoryContractError,
    qualified_runtime_stack_sha256,
    validate_runtime_memory_contract,
)


BUNDLE_MAGIC = b"TRTFB\x00\x01\x00"
_MAX_HEADER_BYTES = 100 * 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_IO_CHUNK_BYTES = 8 * 1024 * 1024
_PLAN_SECTION_ORDER = ("engine_plan", "prefill_engine_plan")
_QUALIFIED_FAMILIES = frozenset({"qwen", "llama"})


class BundleResealError(RuntimeError):
    """The input cannot be safely and exactly resealed."""


def _same_file_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_mode == right.st_mode
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
    )


def _open_regular_input(path: Path) -> tuple[int, os.stat_result]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise BundleResealError(f"input bundle is unavailable: {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode):
        raise BundleResealError(
            f"input bundle must be a non-symlink regular file: {path}"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BundleResealError(
            f"cannot open input bundle without following links: {path}: {exc}"
        ) from exc
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode) or not _same_file_snapshot(
        before, opened
    ):
        os.close(descriptor)
        raise BundleResealError(f"input bundle changed while opening: {path}")
    return descriptor, opened


def _validate_output_path(
    input_path: Path,
    input_stat: os.stat_result,
    output_path: Path,
) -> int:
    if input_path.resolve(strict=True) == output_path.resolve(strict=False):
        raise BundleResealError("input and output bundle paths must differ")
    try:
        output_stat = output_path.lstat()
    except FileNotFoundError:
        output_stat = None
    except OSError as exc:
        raise BundleResealError(
            f"cannot inspect output bundle path: {output_path}: {exc}"
        ) from exc
    if output_stat is not None:
        if not stat.S_ISREG(output_stat.st_mode):
            raise BundleResealError(
                f"output bundle must be a non-symlink regular file: {output_path}"
            )
        if os.path.samestat(input_stat, output_stat):
            raise BundleResealError(
                "input and output bundle paths identify the same file"
            )
        return stat.S_IMODE(output_stat.st_mode)

    try:
        parent_stat = output_path.parent.stat()
    except OSError as exc:
        raise BundleResealError(
            f"output bundle parent is unavailable: {output_path.parent}: {exc}"
        ) from exc
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise BundleResealError(
            f"output bundle parent is not a directory: {output_path.parent}"
        )
    return stat.S_IMODE(input_stat.st_mode)


def _read_exact(
    descriptor: int,
    size: int,
    *,
    label: str,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(remaining, _IO_CHUNK_BYTES))
        if not chunk:
            raise BundleResealError(f"truncated {label}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_bundle_header(
    descriptor: int,
    file_size: int,
) -> tuple[dict[str, Any], int, bytes]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    magic = _read_exact(descriptor, len(BUNDLE_MAGIC), label="bundle magic")
    if magic != BUNDLE_MAGIC:
        raise BundleResealError("input is not a TRTFB v1 container")
    raw_length = _read_exact(descriptor, 8, label="bundle header length")
    header_length = struct.unpack("<Q", raw_length)[0]
    if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
        raise BundleResealError(
            f"bundle header length is invalid: {header_length}"
        )
    data_start = len(BUNDLE_MAGIC) + 8 + header_length
    if data_start > file_size:
        raise BundleResealError("bundle header extends beyond end of file")
    raw_header = _read_exact(
        descriptor,
        header_length,
        label="bundle JSON header",
    )
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BundleResealError(f"bundle JSON header is invalid: {exc}") from exc
    if not isinstance(header, dict):
        raise BundleResealError("bundle JSON header must be an object")
    return header, data_start, magic + raw_length + raw_header


def _is_plan_section(name: str) -> bool:
    return (
        name == "engine_plan"
        or name.endswith("_plan")
        or name.startswith("engine_plan_")
    )


def _validated_section_ranges(
    header: Mapping[str, Any],
    *,
    payload_size: int,
) -> dict[str, tuple[int, int]]:
    raw_sections = header.get("sections")
    if not isinstance(raw_sections, Mapping):
        raise BundleResealError("bundle header sections must be an object")
    plan_names = {
        name
        for name in raw_sections
        if isinstance(name, str) and _is_plan_section(name)
    }
    expected = set(_PLAN_SECTION_ORDER)
    missing = sorted(expected - plan_names)
    extra = sorted(plan_names - expected)
    if missing or extra:
        raise BundleResealError(
            "bundle must contain exactly engine_plan and "
            f"prefill_engine_plan; missing={missing}, extra={extra}"
        )

    ranges: dict[str, tuple[int, int]] = {}
    for name, raw_entry in raw_sections.items():
        if not isinstance(name, str) or not name:
            raise BundleResealError("bundle section names must be non-empty strings")
        if not isinstance(raw_entry, Mapping):
            raise BundleResealError(f"bundle section {name!r} must be an object")
        offset = raw_entry.get("offset")
        size = raw_entry.get("size")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise BundleResealError(
                f"bundle section {name!r} offset/size must be non-negative integers"
            )
        if name in expected and size == 0:
            raise BundleResealError(f"plan section {name!r} must be non-empty")
        end = offset + size
        if end > payload_size:
            raise BundleResealError(
                f"bundle section {name!r} extends beyond the payload"
            )
        ranges[name] = (offset, end)

    for plan_name in _PLAN_SECTION_ORDER:
        plan_start, plan_end = ranges[plan_name]
        for other_name, (other_start, other_end) in ranges.items():
            if other_name == plan_name:
                continue
            if max(plan_start, other_start) < min(plan_end, other_end):
                raise BundleResealError(
                    f"plan section {plan_name!r} overlaps section {other_name!r}"
                )
    return ranges


def _stream_sha256(
    descriptor: int,
    *,
    offset: int,
    size: int,
) -> str:
    digest = hashlib.sha256()
    consumed = 0
    while consumed < size:
        requested = min(_IO_CHUNK_BYTES, size - consumed)
        chunk = os.pread(descriptor, requested, offset + consumed)
        if not chunk:
            raise BundleResealError("bundle changed while hashing a plan section")
        digest.update(chunk)
        consumed += len(chunk)
    return digest.hexdigest()


def _validate_v1_bundle_contract(
    header: Mapping[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    raw_contract = header.get("runtime_memory")
    if not isinstance(raw_contract, Mapping):
        raise BundleResealError("input bundle has no runtime_memory contract")
    if raw_contract.get("contract_version") == 2:
        raise BundleResealError("input bundle is already contract_version 2")
    try:
        contract = validate_runtime_memory_contract(raw_contract)
    except DynamicMemoryContractError as exc:
        raise BundleResealError(
            f"input runtime_memory contract is invalid: {exc}"
        ) from exc
    if contract["contract_version"] != 1:
        raise BundleResealError("input bundle must use contract_version 1")
    if header.get("family") != family:
        raise BundleResealError(
            f"bundle family does not match --family {family!r}"
        )
    if header.get("model_id") != contract["qualified_model_id"]:
        raise BundleResealError(
            "bundle model_id does not match runtime_memory qualification"
        )
    if header.get("max_cache_length") != contract["model_context_limit"]:
        raise BundleResealError(
            "bundle max_cache_length does not match runtime_memory qualification"
        )
    expected_precision = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
    }[contract["kv_dtype"]]
    if header.get("precision") != expected_precision:
        raise BundleResealError(
            "bundle precision does not match runtime_memory qualification"
        )
    return contract


def _read_json_regular(
    path: Path,
    *,
    maximum_size: int,
) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BundleResealError(
            f"qualification manifest is unavailable: {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum_size:
        raise BundleResealError(
            f"qualification manifest must be a small non-symlink regular file: {path}"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not _same_file_snapshot(metadata, opened):
            raise BundleResealError(
                f"qualification manifest changed while opening: {path}"
            )
        raw = _read_exact(
            descriptor,
            opened.st_size,
            label="qualification manifest",
        )
        if not _same_file_snapshot(opened, os.fstat(descriptor)):
            raise BundleResealError(
                f"qualification manifest changed while reading: {path}"
            )
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BundleResealError(
            f"qualification manifest JSON is invalid: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise BundleResealError("qualification manifest root must be an object")
    return value


def _select_v2_contract(
    base_contract: Mapping[str, Any],
    *,
    plan_hashes: Mapping[str, str],
    manifest_path: Path,
) -> dict[str, Any]:
    manifest = _read_json_regular(
        manifest_path,
        maximum_size=_MAX_MANIFEST_BYTES,
    )
    if set(manifest) != {"schema_version", "records"}:
        raise BundleResealError(
            "qualification manifest must contain exactly schema_version and records"
        )
    if (
        isinstance(manifest["schema_version"], bool)
        or manifest["schema_version"] != 1
    ):
        raise BundleResealError("qualification manifest schema_version must be 1")
    records = manifest["records"]
    if not isinstance(records, list) or not records:
        raise BundleResealError(
            "qualification manifest records must be a non-empty array"
        )

    expected_stack = qualified_runtime_stack_sha256(
        base_contract["qualified_runtime_stack"]
    )
    matches: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        plans = record.get("plans")
        if not isinstance(plans, list):
            continue
        record_hashes: dict[str, str] = {}
        malformed = False
        for plan in plans:
            if not isinstance(plan, Mapping):
                malformed = True
                break
            name = plan.get("section_name")
            digest = plan.get("section_sha256")
            if not isinstance(name, str) or not isinstance(digest, str):
                malformed = True
                break
            record_hashes[name] = digest
        if malformed:
            continue
        if (
            record.get("qualified_runtime_stack_sha256") == expected_stack
            and record_hashes == dict(plan_hashes)
        ):
            matches.append(record)
    if len(matches) != 1:
        detail = "ambiguous" if len(matches) > 1 else "absent"
        raise BundleResealError(
            "exact stack+plan calibration is "
            f"{detail}: engine_plan={plan_hashes['engine_plan']}, "
            f"prefill_engine_plan={plan_hashes['prefill_engine_plan']}"
        )

    candidate = {
        **base_contract,
        "contract_version": 2,
        "module_residency_calibration": matches[0],
    }
    try:
        return validate_runtime_memory_contract(candidate)
    except DynamicMemoryContractError as exc:
        raise BundleResealError(
            f"selected module-residency calibration is invalid: {exc}"
        ) from exc


def _create_atomic_output(
    output_path: Path,
    *,
    mode: int,
) -> tuple[int, Path]:
    for _attempt in range(100):
        temporary = output_path.parent / (
            f".{output_path.name}.tmp.{secrets.token_hex(8)}"
        )
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(temporary, flags, 0o600)
        except FileExistsError:
            continue
        try:
            os.fchmod(descriptor, mode)
        except Exception:
            os.close(descriptor)
            temporary.unlink(missing_ok=True)
            raise
        return descriptor, temporary
    raise BundleResealError(
        f"cannot allocate atomic output beside {output_path}"
    )


def _write_all(descriptor: int, value: bytes) -> None:
    remaining = memoryview(value)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise BundleResealError("short write while creating output bundle")
        remaining = remaining[written:]


def _copy_payload(
    input_descriptor: int,
    output_descriptor: int,
    *,
    data_start: int,
    payload_size: int,
    digests: Sequence[Any],
) -> None:
    os.lseek(input_descriptor, data_start, os.SEEK_SET)
    remaining = payload_size
    while remaining:
        chunk = os.read(input_descriptor, min(_IO_CHUNK_BYTES, remaining))
        if not chunk:
            raise BundleResealError(
                "input bundle changed while copying its payload"
            )
        _write_all(output_descriptor, chunk)
        for digest in digests:
            digest.update(chunk)
        remaining -= len(chunk)
    if os.read(input_descriptor, 1):
        raise BundleResealError(
            "input bundle grew while copying its payload"
        )


def _write_resealed_bundle(
    input_descriptor: int,
    input_stat: os.stat_result,
    *,
    data_start: int,
    input_prefix: bytes,
    header: Mapping[str, Any],
    output_path: Path,
    output_mode: int,
) -> dict[str, Any]:
    header_json = json.dumps(
        header,
        indent=2,
        ensure_ascii=True,
    ).encode("utf-8")
    if len(header_json) > _MAX_HEADER_BYTES:
        raise BundleResealError(
            f"resealed bundle header exceeds {_MAX_HEADER_BYTES} bytes"
        )
    output_prefix = (
        BUNDLE_MAGIC + struct.pack("<Q", len(header_json)) + header_json
    )
    input_digest = hashlib.sha256(input_prefix)
    output_digest = hashlib.sha256(output_prefix)
    payload_digest = hashlib.sha256()
    payload_size = input_stat.st_size - data_start
    output_descriptor: int | None = None
    temporary: Path | None = None
    try:
        output_descriptor, temporary = _create_atomic_output(
            output_path,
            mode=output_mode,
        )
        _write_all(output_descriptor, output_prefix)
        _copy_payload(
            input_descriptor,
            output_descriptor,
            data_start=data_start,
            payload_size=payload_size,
            digests=(input_digest, output_digest, payload_digest),
        )
        if not _same_file_snapshot(
            input_stat, os.fstat(input_descriptor)
        ):
            raise BundleResealError(
                "input bundle changed during the reseal operation"
            )
        os.fsync(output_descriptor)
        expected_output_size = len(output_prefix) + payload_size
        if os.fstat(output_descriptor).st_size != expected_output_size:
            raise BundleResealError(
                "atomic output size changed before publication"
            )
        os.close(output_descriptor)
        output_descriptor = None
        os.replace(temporary, output_path)
        temporary = None
    finally:
        if output_descriptor is not None:
            os.close(output_descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "input_bundle_size_bytes": input_stat.st_size,
        "input_bundle_sha256": input_digest.hexdigest(),
        "output_bundle_size_bytes": len(output_prefix) + payload_size,
        "output_bundle_sha256": output_digest.hexdigest(),
        "payload_size_bytes": payload_size,
        "payload_sha256": payload_digest.hexdigest(),
    }


def reseal_bundle(
    input_bundle: str | Path,
    output_bundle: str | Path,
    *,
    family: str,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Reseal one v1 bundle and return its compact qualification receipt."""

    normalized_family = str(family or "").strip()
    if normalized_family not in _QUALIFIED_FAMILIES:
        raise BundleResealError(
            f"--family must be one of {sorted(_QUALIFIED_FAMILIES)}"
        )
    input_path = Path(input_bundle).expanduser()
    output_path = Path(output_bundle).expanduser()
    input_descriptor, input_stat = _open_regular_input(input_path)
    try:
        output_mode = _validate_output_path(
            input_path,
            input_stat,
            output_path,
        )
        header, data_start, input_prefix = _read_bundle_header(
            input_descriptor,
            input_stat.st_size,
        )
        base_contract = _validate_v1_bundle_contract(
            header,
            family=normalized_family,
        )
        ranges = _validated_section_ranges(
            header,
            payload_size=input_stat.st_size - data_start,
        )
        plan_hashes = {
            name: _stream_sha256(
                input_descriptor,
                offset=data_start + ranges[name][0],
                size=ranges[name][1] - ranges[name][0],
            )
            for name in _PLAN_SECTION_ORDER
        }
        selected_manifest = (
            Path(manifest_path).expanduser()
            if manifest_path is not None
            else (
                REPO_ROOT
                / "python"
                / "tensorrt_model_connect"
                / "families"
                / normalized_family
                / "MODULE_RESIDENCY_CALIBRATIONS.json"
            )
        )
        v2_contract = _select_v2_contract(
            base_contract,
            plan_hashes=plan_hashes,
            manifest_path=selected_manifest,
        )
        resealed_header = dict(header)
        resealed_header["runtime_memory"] = v2_contract
        provenance = _write_resealed_bundle(
            input_descriptor,
            input_stat,
            data_start=data_start,
            input_prefix=input_prefix,
            header=resealed_header,
            output_path=output_path,
            output_mode=output_mode,
        )
    finally:
        os.close(input_descriptor)

    calibration = v2_contract["module_residency_calibration"]
    return {
        "receipt_schema_version": 1,
        "input_bundle": str(input_path.resolve(strict=True)),
        "output_bundle": str(output_path.resolve(strict=True)),
        "family": normalized_family,
        **provenance,
        "plan_section_sha256": plan_hashes,
        "plan_set_sha256": calibration["plan_set_sha256"],
        "evidence_sha256": calibration["evidence_sha256"],
        "cuda_module_loading_mode": calibration[
            "cuda_module_loading_mode"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_bundle", type=Path, required=True)
    parser.add_argument("--output", dest="output_bundle", type=Path, required=True)
    parser.add_argument(
        "--family",
        required=True,
        choices=sorted(_QUALIFIED_FAMILIES),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = reseal_bundle(
            args.input_bundle,
            args.output_bundle,
            family=args.family,
        )
    except (BundleResealError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
