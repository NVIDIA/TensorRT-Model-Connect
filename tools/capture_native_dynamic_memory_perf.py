#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create reproducible performance evidence from real bundles and workers.

This is the producer for ``qualify_native_dynamic_memory_perf.py``.  It owns
two trust boundaries that the benchmark worker intentionally does not:

* ``build`` executes the supplied build argv itself, requires the output path
  to be absent, and records the complete source state both before and after
  the build.  This is how ``fresh_build`` and ``artifact_reused`` are proven.
* ``benchmark`` executes the real C++ benchmark worker, independently
  deserializes every TensorRT engine section in the bundle, and enriches the
  worker result with SHA-bound provenance and TensorRT weight accounting.

For a native dynamic-memory bundle, the worker also returns the pipeline's
runtime receipt.  The producer requires its plan/weight/copy/streaming fields
to agree with the independent bundle deserialization.  Static baselines do
not implement the dynamic-memory introspection API, so their accounting comes
only from the same independent deserialization path.

Performance requests opt into per-iteration generated token capture outside
the timed region.  The producer requires a fixed-length greedy AR contract,
binds the prompt to the exact bundle tokenizer contract, and proves that each
case is internally repeatable.  Static/dynamic token values are retained for
diagnosis; the paired qualifier intentionally does not treat a numerically
acceptable near-tie divergence as a throughput failure.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


BUILD_SCHEMA = "trtmc.native-dynamic-memory-perf-build/v1"
WORKER_SCHEMA = "trtmc.benchmark-worker-result/v1"
GENERATION_WORKLOAD_SCHEMA = (
    "trtmc.native-dynamic-memory-generation-workload/v1"
)
TOKENIZER_CONTRACT_SCHEMA = (
    "trtmc.native-dynamic-memory-tokenizer-contract/v1"
)
ROLES = ("exact-head-static-split", "native-dynamic")
BUNDLE_MAGIC = b"TRTFB\x00\x01\x00"
MEASUREMENT_SOURCES = {
    "serialized_plan_bytes": "bundle_engine_section_sizes",
    "resident_weight_bytes": (
        "tensorrt_total_weights_size_weight_streaming_disabled"
    ),
    "resident_weight_copy_count": "deduplicated_tensorrt_engine_identity",
}
RUNTIME_PLAN_PREFIX = "[trtmc.runtime_kv.plan]"
RUNTIME_PLAN_FIELDS = (
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
RUNTIME_PLAN_INTEGER_FIELDS = (
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
RUNTIME_PLAN_IDENTITY_FIELDS = (
    "device",
    "role",
    "hq",
    "hkv",
    "d",
    "C",
    "Sq",
    "T",
)
RUNTIME_STACK_PREFIX = "[trtmc.runtime_stack]"
RUNTIME_STACK_FIELDS = (
    "schema",
    "sm",
    "tensorrt",
    "cuda_runtime",
    "cudnn_backend",
    "cudnn_frontend_revision",
    "nvrtc",
    "driver",
)
RUNTIME_LIBRARY_PATTERNS = {
    "nvrtc": re.compile(r"^libnvrtc\.so\.13(?:\.[0-9]+)*$"),
    "nvrtc_builtins": re.compile(
        r"^libnvrtc-builtins\.so\.13(?:\.[0-9]+)*$"
    ),
}


class CaptureError(RuntimeError):
    """Evidence was incomplete, contradictory, or not freshly produced."""


def _load_boundary_module() -> Any:
    path = Path(__file__).with_name("qualify_native_dynamic_memory.py")
    spec = importlib.util.spec_from_file_location(
        "_trtmc_dynamic_memory_boundary", path
    )
    if spec is None or spec.loader is None:
        raise CaptureError(f"cannot load source-state helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"{label} must be a JSON object")
    return value


def _source_state(
    repo_root: Path, artifact_dir: Path, *, label: str
) -> dict[str, Any]:
    boundary = _load_boundary_module()
    return boundary.source_state_provenance(
        repo_root.resolve(),
        Path(__file__).resolve(),
        artifact_dir.resolve(),
        label=label,
    )


def _require_nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureError(f"{label} must be a non-empty string")
    return value


def _bundle_header(path: Path) -> tuple[dict[str, Any], int]:
    with path.open("rb") as stream:
        if stream.read(8) != BUNDLE_MAGIC:
            raise CaptureError(f"{path} is not a TRTMC bundle")
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise CaptureError(f"{path} has a truncated header length")
        header_length = struct.unpack("<Q", raw_length)[0]
        payload = stream.read(header_length)
        if len(payload) != header_length:
            raise CaptureError(f"{path} has a truncated JSON header")
    try:
        header = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CaptureError(f"{path} has invalid bundle JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise CaptureError("bundle JSON header must be an object")
    return header, 16 + header_length


def _bundle_section_bytes(
    path: Path,
    header: Mapping[str, Any],
    payload_offset: int,
    name: str,
) -> bytes:
    raw_sections = header.get("sections")
    if not isinstance(raw_sections, Mapping):
        raise CaptureError("bundle has no sections object")
    raw = raw_sections.get(name)
    if not isinstance(raw, Mapping):
        raise CaptureError(f"bundle has no {name} section")
    offset = raw.get("offset")
    size = raw.get("size")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise CaptureError(f"bundle has invalid {name} section metadata")
    file_size = path.stat().st_size
    absolute_offset = payload_offset + offset
    if absolute_offset > file_size or size > file_size - absolute_offset:
        raise CaptureError(f"bundle {name} section extends past end of file")
    with path.open("rb") as stream:
        stream.seek(absolute_offset)
        payload = stream.read(size)
    if len(payload) != size:
        raise CaptureError(f"bundle {name} section is truncated")
    return payload


def _tokenizer_contract(
    path: Path,
    header: Mapping[str, Any],
    payload_offset: int,
) -> dict[str, Any]:
    tokenizer = _bundle_section_bytes(
        path, header, payload_offset, "tokenizer.json"
    )
    config_payload = _bundle_section_bytes(
        path, header, payload_offset, "config.json"
    )
    try:
        config = json.loads(config_payload)
    except json.JSONDecodeError as exc:
        raise CaptureError(f"bundle config.json is invalid: {exc}") from exc
    if not isinstance(config, Mapping):
        raise CaptureError("bundle config.json must be an object")

    raw_add_special = header.get("tokenizer_add_special_tokens")
    if isinstance(raw_add_special, bool):
        add_special = raw_add_special
    elif (
        isinstance(raw_add_special, int)
        and raw_add_special in (0, 1)
    ):
        add_special = bool(raw_add_special)
    else:
        raise CaptureError(
            "bundle header must declare tokenizer_add_special_tokens as 0 or 1"
        )

    def special_ids(name: str) -> list[int]:
        raw = config.get(name, [])
        if not isinstance(raw, list) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in raw
        ):
            raise CaptureError(f"bundle config {name} must be non-negative integers")
        return list(raw)

    return {
        "schema_version": TOKENIZER_CONTRACT_SCHEMA,
        "tokenizer_json_sha256": hashlib.sha256(tokenizer).hexdigest(),
        "tokenizer_json_bytes": len(tokenizer),
        "tokenizer_add_special_tokens": add_special,
        "tokenizer_special_prefix_ids": special_ids(
            "tokenizer_special_prefix_ids"
        ),
        "tokenizer_special_suffix_ids": special_ids(
            "tokenizer_special_suffix_ids"
        ),
    }


def _generation_workload(
    result: Mapping[str, Any],
    *,
    semantic_request: Mapping[str, Any],
    measurement: Mapping[str, Any],
) -> dict[str, Any]:
    prompt = semantic_request.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise CaptureError("performance request prompt must be a non-empty string")
    max_new_tokens = semantic_request.get("max_new_tokens")
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens <= 0
    ):
        raise CaptureError(
            "performance request max_new_tokens must be a positive integer"
        )
    effective = {
        "generation_mode": semantic_request.get("generation_mode", "auto"),
        "temperature": semantic_request.get("temperature", 0.0),
        "top_k": semantic_request.get("top_k", 1),
        "top_p": semantic_request.get("top_p", 1.0),
        "min_p": semantic_request.get("min_p", 0.0),
        "num_samples": semantic_request.get("num_samples", 1),
        "eos_token_id": semantic_request.get("eos_token_id", -1),
        "use_chat_template": semantic_request.get(
            "use_chat_template", False
        ),
        "stop_on_boxed_answer": semantic_request.get(
            "stop_on_boxed_answer", False
        ),
        "capture_generated_token_ids": semantic_request.get(
            "capture_generated_token_ids", False
        ),
        "max_new_tokens": max_new_tokens,
    }
    if effective != {
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
        raise CaptureError(
            "performance request must use fixed-length greedy AR decoding "
            "(generation_mode=ar, temperature=0, top_k=1, top_p=1, "
            "min_p=0, num_samples=1, eos_token_id=INT32_MAX, "
            "use_chat_template=false, stop_on_boxed_answer=false, "
            "capture_generated_token_ids=true)"
        )

    iterations = measurement.get("iterations")
    observations = result.get("observations")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
        or not isinstance(observations, list)
        or len(observations) != iterations
    ):
        raise CaptureError(
            "worker observations do not match the requested iteration count"
        )
    streams: list[list[int]] = []
    for index, raw in enumerate(observations):
        if not isinstance(raw, Mapping):
            raise CaptureError(
                f"worker observation {index} must be an object"
            )
        stream = raw.get("generated_token_ids")
        count = raw.get("output_tokens")
        if (
            not isinstance(stream, list)
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for token in stream
            )
            or count != len(stream)
            or len(stream) != max_new_tokens
        ):
            raise CaptureError(
                "worker observation "
                f"{index} does not prove a fixed {max_new_tokens}-token decode"
            )
        streams.append(list(stream))
    if any(stream != streams[0] for stream in streams[1:]):
        raise CaptureError(
            "fixed greedy token stream changed across measured iterations"
        )
    output_summary = result.get("output_summary")
    if (
        not isinstance(output_summary, Mapping)
        or output_summary.get("token_ids") != streams[-1]
    ):
        raise CaptureError(
            "worker output_summary token IDs do not match the last observation"
        )

    structural_identity = {
        "operation": "generate",
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "generation": effective,
        "measurement": dict(measurement),
    }
    return {
        "schema_version": GENERATION_WORKLOAD_SCHEMA,
        "kind": "fixed_length_greedy_ar",
        "structural_identity": structural_identity,
        "structural_identity_sha256": _canonical_sha(structural_identity),
        "measured_generated_token_ids": streams,
        "measured_generated_token_ids_sha256": _canonical_sha(streams),
        "token_stream_repeatable_within_case": True,
    }


def _engine_sections(header: Mapping[str, Any]) -> list[tuple[str, int, int]]:
    raw_sections = header.get("sections")
    if not isinstance(raw_sections, Mapping):
        raise CaptureError("bundle has no sections object")
    selected: list[tuple[str, int, int]] = []
    for name, raw in raw_sections.items():
        if not isinstance(name, str) or not isinstance(raw, Mapping):
            raise CaptureError("bundle section metadata is malformed")
        is_engine = (
            name == "engine_plan"
            or name.endswith("_engine_plan")
            or name.startswith("engine_plan_tp_rank")
        )
        if not is_engine:
            continue
        offset = raw.get("offset")
        size = raw.get("size")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise CaptureError(f"invalid engine section metadata: {name}")
        selected.append((name, offset, size))
    if not selected:
        raise CaptureError("bundle has no TensorRT engine sections")
    return sorted(selected)


def _runtime_memory_fields(receipt: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in (
        "serialized_plan_bytes",
        "resident_weight_bytes",
        "resident_weight_copy_count",
        "weight_streaming_active",
    ):
        if name not in receipt:
            raise CaptureError(f"runtime receipt is missing {name}")
        fields[name] = receipt[name]
    sources = receipt.get("measurement_sources")
    if not isinstance(sources, Mapping):
        raise CaptureError("runtime receipt has no measurement_sources object")
    fields["measurement_sources"] = {
        name: sources.get(name) for name in MEASUREMENT_SOURCES
    }
    return fields


def _parse_runtime_attention_plans(
    stderr: str, *, artifact_role: str
) -> list[dict[str, Any]]:
    """Parse and validate the plans selected by the runtime attention plugin."""

    plan_lines = [
        line
        for line in stderr.splitlines()
        if line.startswith(RUNTIME_PLAN_PREFIX)
    ]
    if artifact_role == "exact-head-static-split":
        if plan_lines:
            raise CaptureError(
                "static baseline unexpectedly emitted runtime attention plans"
            )
        return []
    if artifact_role != "native-dynamic":
        raise CaptureError(f"unsupported benchmark artifact role: {artifact_role}")
    if "COMPILATION_FAILED" in stderr:
        raise CaptureError(
            "dynamic benchmark stderr contains COMPILATION_FAILED"
        )
    if not plan_lines:
        raise CaptureError(
            "dynamic benchmark did not emit a runtime attention plan"
        )

    expected_fields = set(RUNTIME_PLAN_FIELDS)
    by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
    for line_number, line in enumerate(plan_lines, start=1):
        raw_fields: dict[str, str] = {}
        payload = line[len(RUNTIME_PLAN_PREFIX) :].strip()
        for token in payload.split():
            name, separator, value = token.partition("=")
            if (
                not separator
                or not name
                or not value
                or name in raw_fields
            ):
                raise CaptureError(
                    "malformed runtime attention plan line "
                    f"{line_number}: invalid token {token!r}"
                )
            raw_fields[name] = value
        missing = sorted(expected_fields - raw_fields.keys())
        extra = sorted(raw_fields.keys() - expected_fields)
        if missing or extra:
            raise CaptureError(
                "malformed runtime attention plan line "
                f"{line_number}: missing={missing!r}, extra={extra!r}"
            )

        row: dict[str, Any] = {}
        for field in RUNTIME_PLAN_FIELDS:
            value = raw_fields[field]
            if field in RUNTIME_PLAN_INTEGER_FIELDS:
                if not value.isascii() or not value.isdigit():
                    raise CaptureError(
                        "malformed runtime attention plan line "
                        f"{line_number}: {field} must be an unsigned integer"
                    )
                row[field] = int(value)
            else:
                row[field] = value

        if row["schema"] != 1:
            raise CaptureError(
                "runtime attention plan schema must be 1, "
                f"got {row['schema']!r}"
            )
        if row["role"] not in ("history", "current"):
            raise CaptureError(
                "runtime attention plan role must be history or current"
            )
        if row["stats"] != "lse":
            raise CaptureError(
                "runtime attention plan must report stats=lse"
            )
        for field in ("hq", "hkv", "d", "C", "Sq", "T", "cudnn_version"):
            if row[field] <= 0:
                raise CaptureError(
                    f"runtime attention plan {field} must be positive"
                )
        if not row["heur"] or not row["plan"]:
            raise CaptureError(
                "runtime attention plan heur and plan must be non-empty"
            )

        identity = tuple(row[field] for field in RUNTIME_PLAN_IDENTITY_FIELDS)
        previous = by_identity.get(identity)
        if previous is not None and previous != row:
            raise CaptureError(
                "conflicting runtime attention plan rows for graph identity "
                f"{identity!r}"
            )
        by_identity[identity] = row

    if len({row["device"] for row in by_identity.values()}) != 1:
        raise CaptureError(
            "conflicting runtime attention plan device identities"
        )
    if len({row["cudnn_version"] for row in by_identity.values()}) != 1:
        raise CaptureError(
            "conflicting runtime attention plan cuDNN versions"
        )
    return [
        by_identity[identity]
        for identity in sorted(by_identity)
    ]


def _parse_runtime_stack(
    stderr: str, *, artifact_role: str
) -> dict[str, Any] | None:
    """Return live stack evidence emitted by the product worker process."""

    stack_lines = [
        line
        for line in stderr.splitlines()
        if line.startswith(RUNTIME_STACK_PREFIX)
    ]
    if artifact_role == "exact-head-static-split":
        if stack_lines:
            raise CaptureError(
                "static baseline unexpectedly emitted dynamic runtime-stack "
                "evidence"
            )
        return None
    if artifact_role != "native-dynamic":
        raise CaptureError(f"unsupported benchmark artifact role: {artifact_role}")
    if not stack_lines:
        raise CaptureError(
            "dynamic benchmark did not emit live runtime-stack evidence"
        )

    expected_fields = set(RUNTIME_STACK_FIELDS)
    unique: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(stack_lines, start=1):
        raw_fields: dict[str, str] = {}
        payload = line[len(RUNTIME_STACK_PREFIX) :].strip()
        for token in payload.split():
            name, separator, value = token.partition("=")
            if (
                not separator
                or not name
                or not value
                or name in raw_fields
            ):
                raise CaptureError(
                    "malformed runtime-stack line "
                    f"{line_number}: invalid token {token!r}"
                )
            raw_fields[name] = value
        missing = sorted(expected_fields - raw_fields.keys())
        extra = sorted(raw_fields.keys() - expected_fields)
        if missing or extra:
            raise CaptureError(
                "malformed runtime-stack line "
                f"{line_number}: missing={missing!r}, extra={extra!r}"
            )

        schema = raw_fields["schema"]
        if schema != "1":
            raise CaptureError(
                f"runtime-stack schema must be 1, got {schema!r}"
            )
        for field in RUNTIME_STACK_FIELDS[1:]:
            if raw_fields[field] == "unavailable":
                raise CaptureError(
                    f"runtime-stack field {field} is unavailable"
                )
        if re.fullmatch(r"sm[0-9]+", raw_fields["sm"]) is None:
            raise CaptureError("runtime-stack sm must use the smNNN form")
        if re.fullmatch(
            r"[0-9a-f]{40}", raw_fields["cudnn_frontend_revision"]
        ) is None:
            raise CaptureError(
                "runtime-stack cuDNN Frontend revision must be a full Git SHA"
            )
        for field in (
            "tensorrt",
            "cuda_runtime",
            "cudnn_backend",
            "nvrtc",
        ):
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", raw_fields[field]) is None:
                raise CaptureError(
                    f"runtime-stack {field} must be a dotted numeric version"
                )

        row: dict[str, Any] = {
            **raw_fields,
            "schema": 1,
        }
        identity = _canonical_sha(row)
        unique[identity] = row

    if len(unique) != 1:
        raise CaptureError(
            "dynamic benchmark emitted conflicting live runtime-stack evidence"
        )
    return next(iter(unique.values()))


def _encoded_cudnn_version(version: str) -> int:
    parts = version.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise CaptureError(
            "runtime-stack cudnn_backend must contain major.minor.patch"
        )
    major, minor, patch = (int(part) for part in parts)
    return major * 10000 + minor * 100 + patch


def _cuda_cache_configuration(cwd: Path) -> dict[str, Any]:
    raw_path = os.environ.get("CUDA_CACHE_PATH")
    if raw_path:
        unresolved = Path(raw_path).expanduser()
        if not unresolved.is_absolute():
            unresolved = cwd / unresolved
        path_source = "CUDA_CACHE_PATH"
    else:
        unresolved = Path.home() / ".nv" / "ComputeCache"
        path_source = "cuda_default"
    raw_disable = os.environ.get("CUDA_CACHE_DISABLE")
    return {
        "path": str(unresolved.resolve()),
        "path_source": path_source,
        "cuda_cache_path_env": raw_path,
        "cuda_cache_disable_env": raw_disable,
        "enabled": raw_disable != "1",
    }


def _cuda_cache_snapshot(path: Path) -> dict[str, Any]:
    captured_at_ns = time.time_ns()
    if not path.exists():
        return {
            "captured_at_ns": captured_at_ns,
            "exists": False,
            "is_directory": False,
            "entry_count": 0,
            "file_count": 0,
            "total_bytes": 0,
            "metadata_sha256": _canonical_sha([]),
        }
    if not path.is_dir():
        raise CaptureError(f"CUDA cache path is not a directory: {path}")

    entries: list[dict[str, Any]] = []
    file_count = 0
    total_bytes = 0
    try:
        for root, directories, files in os.walk(path):
            directories.sort()
            files.sort()
            root_path = Path(root)
            for name in directories:
                entry = root_path / name
                metadata = entry.stat(follow_symlinks=False)
                entries.append(
                    {
                        "kind": "directory",
                        "path": str(entry.relative_to(path)),
                        "size": metadata.st_size,
                        "mtime_ns": metadata.st_mtime_ns,
                    }
                )
            for name in files:
                entry = root_path / name
                metadata = entry.stat(follow_symlinks=False)
                entries.append(
                    {
                        "kind": "file",
                        "path": str(entry.relative_to(path)),
                        "size": metadata.st_size,
                        "mtime_ns": metadata.st_mtime_ns,
                    }
                )
                file_count += 1
                total_bytes += metadata.st_size
    except OSError as exc:
        raise CaptureError(
            f"cannot inspect CUDA cache path {path}: {exc}"
        ) from exc
    entries.sort(key=lambda entry: (entry["path"], entry["kind"]))
    return {
        "captured_at_ns": captured_at_ns,
        "exists": True,
        "is_directory": True,
        "entry_count": len(entries),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "metadata_sha256": _canonical_sha(entries),
    }


def _cuda_cache_initial_state(
    configuration: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> str:
    if configuration.get("enabled") is not True:
        return "disabled"
    return "warm" if int(snapshot.get("file_count", 0)) > 0 else "cold"


def _mapped_library_paths(pid: int) -> tuple[Path, ...]:
    """Return canonical file-backed mappings currently loaded by ``pid``."""

    maps = Path(f"/proc/{pid}/maps")
    try:
        lines = maps.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise CaptureError(f"cannot inspect live worker mappings: {exc}") from exc

    paths: set[Path] = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw_path = fields[5]
        if raw_path.endswith(" (deleted)"):
            raw_path = raw_path[: -len(" (deleted)")]
        path = Path(raw_path)
        try:
            canonical = path.resolve(strict=True)
        except FileNotFoundError:
            continue
        if canonical.is_file():
            paths.add(canonical)
    return tuple(sorted(paths))


def _run_worker_with_library_capture(
    command: Sequence[str],
    *,
    cwd: Path,
    stdout_output: Path,
    stderr_output: Path,
) -> tuple[subprocess.CompletedProcess[str], tuple[Path, ...]]:
    """Execute a worker while retaining every transient mapped library path."""

    stdout_output.parent.mkdir(parents=True, exist_ok=True)
    stderr_output.parent.mkdir(parents=True, exist_ok=True)
    observed_paths: set[Path] = set()
    with stdout_output.open("wb") as stdout_stream, stderr_output.open(
        "wb"
    ) as stderr_stream:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=stdout_stream,
            stderr=stderr_stream,
        )
        while True:
            observed_paths.update(_mapped_library_paths(process.pid))
            returncode = process.poll()
            if returncode is not None:
                break
            time.sleep(0.01)
        process.wait()
    completed = subprocess.CompletedProcess(
        args=list(command),
        returncode=returncode,
        stdout=stdout_output.read_text(encoding="utf-8", errors="replace"),
        stderr=stderr_output.read_text(encoding="utf-8", errors="replace"),
    )
    return completed, tuple(sorted(observed_paths))


def _runtime_library_provenance(
    paths: Sequence[Path],
    *,
    artifact_role: str,
    runtime_stack: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve the exact NVRTC pair mapped by the dynamic worker process."""

    if artifact_role == "exact-head-static-split":
        return None
    if artifact_role != "native-dynamic" or runtime_stack is None:
        raise CaptureError("dynamic runtime library provenance has no live stack")

    provenance: dict[str, Any] = {}
    for label, pattern in RUNTIME_LIBRARY_PATTERNS.items():
        matches = sorted(
            {
                path.resolve()
                for path in paths
                if pattern.fullmatch(path.name) is not None
            }
        )
        if len(matches) != 1:
            raise CaptureError(
                "dynamic worker must map exactly one "
                f"{label.replace('_', '-')} library, found "
                f"{[str(path) for path in matches]!r}"
            )
        path = matches[0]
        provenance[label] = {
            "path": str(path),
            "basename": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }

    nvrtc_version = str(runtime_stack["nvrtc"])
    version_parts = nvrtc_version.split(".")
    expected_prefix = ".".join(version_parts[:2])
    for label in ("nvrtc", "nvrtc_builtins"):
        basename = str(provenance[label]["basename"])
        if f".so.13.{expected_prefix.split('.', maxsplit=1)[1]}" not in basename:
            raise CaptureError(
                f"mapped {label.replace('_', '-')} library {basename!r} "
                f"does not match live NVRTC {nvrtc_version}"
            )
    if (
        Path(provenance["nvrtc"]["path"]).parent
        != Path(provenance["nvrtc_builtins"]["path"]).parent
    ):
        raise CaptureError(
            "NVRTC and NVRTC-builtins were not loaded from one directory"
        )
    provenance["directory"] = str(Path(provenance["nvrtc"]["path"]).parent)
    provenance["live_nvrtc_version"] = nvrtc_version
    return provenance


def _engine_accounting(
    bundle: Path, plugin_library: Path
) -> dict[str, Any]:
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - qualification environment
        raise CaptureError("TensorRT Python bindings are required") from exc

    plugin_library = plugin_library.expanduser().resolve()
    if not plugin_library.is_file():
        raise CaptureError(f"plugin library does not exist: {plugin_library}")
    ctypes.CDLL(str(plugin_library), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, "")
    runtime = trt.Runtime(logger)

    header, data_start = _bundle_header(bundle)
    sections = _engine_sections(header)
    serialized_plan_bytes = sum(size for _, _, size in sections)
    resident_weight_bytes = 0
    weight_streaming_active = False
    section_receipts: list[dict[str, Any]] = []

    with bundle.open("rb") as stream:
        for name, offset, size in sections:
            stream.seek(data_start + offset)
            plan = stream.read(size)
            if len(plan) != size:
                raise CaptureError(f"bundle engine section is truncated: {name}")
            plan_sha256 = hashlib.sha256(plan).hexdigest()
            engine = runtime.deserialize_cuda_engine(plan)
            if engine is None:
                raise CaptureError(
                    f"TensorRT could not deserialize engine section {name}"
                )
            total_weights = int(
                engine.get_engine_stat(trt.EngineStat.TOTAL_WEIGHTS_SIZE)
            )
            streamable = int(engine.streamable_weights_size)
            budget = (
                int(engine.weight_streaming_budget_v2)
                if streamable > 0
                else 0
            )
            if total_weights <= 0:
                raise CaptureError(
                    f"TensorRT reported no weights for engine section {name}"
                )
            streaming = streamable > 0 and budget < streamable
            resident_weight_bytes += total_weights
            weight_streaming_active = weight_streaming_active or streaming
            section_receipts.append(
                {
                    "name": name,
                    "plan_bytes": size,
                    "plan_sha256": plan_sha256,
                    "total_weight_bytes": total_weights,
                    "streamable_weight_bytes": streamable,
                    "weight_streaming_budget_bytes": budget,
                    "weight_streaming_active": streaming,
                }
            )
            del engine
            del plan

    return {
        "serialized_plan_bytes": serialized_plan_bytes,
        "resident_weight_bytes": resident_weight_bytes,
        "resident_weight_copy_count": len(sections),
        "weight_streaming_active": weight_streaming_active,
        "measurement_sources": dict(MEASUREMENT_SOURCES),
        "engine_sections": section_receipts,
        "tensorrt_version": str(trt.__version__),
    }


def _environment_identity() -> dict[str, Any]:
    try:
        import tensorrt as trt
        import torch
    except ImportError as exc:  # pragma: no cover - qualification environment
        raise CaptureError(
            "TensorRT and PyTorch Python bindings are required"
        ) from exc
    if not torch.cuda.is_available():
        raise CaptureError("CUDA is not available to the benchmark producer")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    gpu_rows = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,pci.bus_id,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    torch_uuid = str(properties.uuid).lower()
    selected_rows: list[tuple[str, str, str]] = []
    for raw_row in gpu_rows:
        fields = tuple(field.strip() for field in raw_row.split(","))
        if len(fields) != 3:
            raise CaptureError(
                f"nvidia-smi returned malformed GPU identity row: {raw_row!r}"
            )
        uuid = fields[0]
        normalized_uuid = uuid.removeprefix("GPU-").lower()
        if normalized_uuid == torch_uuid:
            selected_rows.append((uuid, fields[1], fields[2]))
    if len(selected_rows) != 1:
        raise CaptureError(
            "cannot map the process CUDA device to one physical GPU: "
            f"torch_uuid={torch_uuid!r}, matches={selected_rows!r}"
        )
    physical_uuid, pci_bus_id, driver_version = selected_rows[0]
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_logical_device": device,
        "cuda_device_uuid": physical_uuid,
        "cuda_pci_bus_id": pci_bus_id,
        "cuda_device_name": properties.name,
        "cuda_compute_capability": (
            f"sm{properties.major}{properties.minor}"
        ),
        "cuda_total_memory_bytes": properties.total_memory,
        "torch_build_cuda_version": str(torch.version.cuda),
        "driver_version": driver_version,
        "tensorrt_version": str(trt.__version__),
        "python": sys.version,
        "platform": sys.platform,
    }


def _cmd_build(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    source_artifact_dir = args.source_artifact_dir.expanduser().resolve()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise CaptureError("build requires an argv after '--'")
    if bundle.exists():
        raise CaptureError(
            f"fresh-build output already exists; choose an absent path: {bundle}"
        )
    bundle.parent.mkdir(parents=True, exist_ok=True)
    stdout_output = args.stdout_output.expanduser().resolve()
    stderr_output = args.stderr_output.expanduser().resolve()
    stdout_output.parent.mkdir(parents=True, exist_ok=True)
    stderr_output.parent.mkdir(parents=True, exist_ok=True)

    before = _source_state(repo_root, source_artifact_dir, label="prebuild")
    started_ns = time.time_ns()
    with stdout_output.open("wb") as stdout_stream, stderr_output.open(
        "wb"
    ) as stderr_stream:
        completed = subprocess.run(
            command,
            cwd=args.cwd,
            check=False,
            stdout=stdout_stream,
            stderr=stderr_stream,
        )
    finished_ns = time.time_ns()
    after = _source_state(repo_root, source_artifact_dir, label="postbuild")
    if completed.returncode != 0:
        raise CaptureError(
            f"build command failed with return code {completed.returncode}"
        )
    if not bundle.is_file() or bundle.stat().st_size <= 0:
        raise CaptureError(f"build command did not create bundle: {bundle}")
    if before["source_state_sha256"] != after["source_state_sha256"]:
        raise CaptureError("source state changed while building the bundle")
    if before["git_head"] != after["git_head"]:
        raise CaptureError("Git HEAD changed while building the bundle")

    header, _ = _bundle_header(bundle)
    if header.get("model_id") != args.model_id:
        raise CaptureError(
            "built bundle model_id mismatch: "
            f"expected {args.model_id!r}, got {header.get('model_id')!r}"
        )
    receipt = {
        "schema_version": BUILD_SCHEMA,
        "artifact_role": args.role,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "precision": args.precision,
        "target": args.target,
        "bundle_build_id": args.bundle_build_id,
        "fresh_build": True,
        "artifact_reused": False,
        "bundle": str(bundle),
        "bundle_sha256": _sha256(bundle),
        "bundle_bytes": bundle.stat().st_size,
        "bundle_mtime_ns": bundle.stat().st_mtime_ns,
        "build_started_ns": started_ns,
        "build_finished_ns": finished_ns,
        "command": command,
        "command_sha256": _canonical_sha(command),
        "cwd": str(Path(args.cwd).resolve()),
        "stdout": str(stdout_output),
        "stdout_sha256": _sha256(stdout_output),
        "stderr": str(stderr_output),
        "stderr_sha256": _sha256(stderr_output),
        "git_head": before["git_head"],
        "prebuild_source_state_sha256": before["source_state_sha256"],
        "postbuild_source_state_sha256": after["source_state_sha256"],
        "source_state_pre": before,
        "source_state_post": after,
    }
    _write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": "completed",
                "bundle": str(bundle),
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0


def _validate_build_receipt(
    receipt: Mapping[str, Any],
    *,
    bundle: Path,
    role: str,
    source_state: Mapping[str, Any],
) -> None:
    if receipt.get("schema_version") != BUILD_SCHEMA:
        raise CaptureError(f"build receipt must use {BUILD_SCHEMA}")
    if receipt.get("artifact_role") != role:
        raise CaptureError("build receipt role does not match benchmark role")
    if receipt.get("fresh_build") is not True:
        raise CaptureError("build receipt does not prove a fresh build")
    if receipt.get("artifact_reused") is not False:
        raise CaptureError("build receipt says the artifact was reused")
    if receipt.get("bundle_sha256") != _sha256(bundle):
        raise CaptureError("bundle SHA does not match build receipt")
    if receipt.get("bundle_bytes") != bundle.stat().st_size:
        raise CaptureError("bundle size does not match build receipt")
    if receipt.get("bundle_mtime_ns") != bundle.stat().st_mtime_ns:
        raise CaptureError("bundle mtime does not match build receipt")
    source_sha = source_state.get("source_state_sha256")
    if (
        receipt.get("prebuild_source_state_sha256") != source_sha
        or receipt.get("postbuild_source_state_sha256") != source_sha
    ):
        raise CaptureError("current source differs from the fresh-build source")
    if receipt.get("git_head") != source_state.get("git_head"):
        raise CaptureError("current Git HEAD differs from the fresh-build HEAD")


def _cmd_benchmark(args: argparse.Namespace) -> int:
    if args.comparison_sequence_limit <= 0:
        raise CaptureError("--comparison-sequence-limit must be positive")
    repo_root = args.repo_root.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    request = args.request.expanduser().resolve()
    worker = args.worker.expanduser().resolve()
    plugin_library = args.plugin_library.expanduser().resolve()
    output = args.output.expanduser().resolve()
    stderr_output = args.stderr_output.expanduser().resolve()
    build_receipt = _read_object(args.build_receipt, "build receipt")
    if not bundle.is_file() or not request.is_file() or not worker.is_file():
        raise CaptureError("bundle, request, and worker must be files")
    request_document = _read_object(request, "benchmark request")
    request_bundle = request_document.get("bundle")
    if not isinstance(request_bundle, str) or not request_bundle:
        raise CaptureError("benchmark request has no bundle path")
    request_bundle_path = Path(request_bundle).expanduser()
    if not request_bundle_path.is_absolute():
        request_bundle_path = Path(args.cwd) / request_bundle_path
    if request_bundle_path.resolve() != bundle:
        raise CaptureError(
            "benchmark request bundle does not match --bundle: "
            f"{request_bundle_path.resolve()} != {bundle}"
        )
    if request_document.get("operation") != "generate":
        raise CaptureError("performance qualification requires generate")
    measurement = request_document.get("measurement")
    semantic_request = request_document.get("request")
    runtime_options = request_document.get("runtime", {})
    if not isinstance(measurement, Mapping) or not isinstance(
        semantic_request, Mapping
    ):
        raise CaptureError(
            "benchmark request requires measurement and request objects"
        )
    if not isinstance(runtime_options, Mapping):
        raise CaptureError("benchmark request runtime must be an object")
    header, payload_offset = _bundle_header(bundle)
    tokenizer_contract = _tokenizer_contract(
        bundle, header, payload_offset
    )
    if args.role == "exact-head-static-split":
        if int(header.get("max_cache_length", 0)) != args.comparison_sequence_limit:
            raise CaptureError(
                "static bundle max_cache_length does not match "
                "--comparison-sequence-limit"
            )
        if "max_sequence_length" in runtime_options:
            raise CaptureError(
                "static baseline request must not use dynamic sequence policy"
            )
    else:
        if (
            runtime_options.get("max_sequence_length")
            != args.comparison_sequence_limit
        ):
            raise CaptureError(
                "dynamic request max_sequence_length does not match "
                "--comparison-sequence-limit"
            )
    comparison_request_sha256 = _canonical_sha(
        {
            "operation": "generate",
            "measurement": dict(measurement),
            "request": dict(semantic_request),
            "effective_sequence_limit": args.comparison_sequence_limit,
        }
    )

    source_artifact_dir = output.parent / "source-state"
    source_state_pre = _source_state(
        repo_root, source_artifact_dir, label=f"{output.stem}-pre"
    )
    _validate_build_receipt(
        build_receipt,
        bundle=bundle,
        role=args.role,
        source_state=source_state_pre,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_suffix(output.suffix + ".raw")
    stdout_output = output.with_suffix(output.suffix + ".worker.stdout.log")
    cache_configuration = _cuda_cache_configuration(Path(args.cwd).resolve())
    cache_path = Path(cache_configuration["path"])
    cache_before = _cuda_cache_snapshot(cache_path)
    worker_command = [
        str(worker),
        "--request",
        str(request),
        "--output",
        str(raw_output),
    ]
    worker_started_ns = time.time_ns()
    completed, mapped_libraries = _run_worker_with_library_capture(
        worker_command,
        cwd=args.cwd,
        stdout_output=stdout_output,
        stderr_output=stderr_output,
    )
    worker_finished_ns = time.time_ns()
    cache_after = _cuda_cache_snapshot(cache_path)
    if completed.returncode != 0:
        raise CaptureError(
            f"benchmark worker failed with return code {completed.returncode}; "
            f"see {stderr_output}"
        )
    runtime_attention_plans = _parse_runtime_attention_plans(
        completed.stderr, artifact_role=args.role
    )
    runtime_stack = _parse_runtime_stack(
        completed.stderr, artifact_role=args.role
    )
    runtime_libraries = _runtime_library_provenance(
        mapped_libraries,
        artifact_role=args.role,
        runtime_stack=runtime_stack,
    )
    if runtime_stack is not None:
        expected_cudnn = _encoded_cudnn_version(
            str(runtime_stack["cudnn_backend"])
        )
        if any(
            int(row["cudnn_version"]) != expected_cudnn
            for row in runtime_attention_plans
        ):
            raise CaptureError(
                "runtime attention plan cuDNN version disagrees with live "
                "runtime-stack evidence"
            )
    result = _read_object(raw_output, "benchmark worker result")
    raw_output.unlink(missing_ok=True)
    if result.get("schema_version") != WORKER_SCHEMA:
        raise CaptureError(f"worker result must use {WORKER_SCHEMA}")
    if result.get("status") != "completed":
        raise CaptureError("benchmark worker result is not completed")
    generation_workload = _generation_workload(
        result,
        semantic_request=semantic_request,
        measurement=measurement,
    )

    accounting = _engine_accounting(bundle, plugin_library)
    if args.role == "native-dynamic":
        raw_receipt = result.get("runtime_memory_receipt")
        if not isinstance(raw_receipt, Mapping):
            raise CaptureError(
                "dynamic benchmark worker did not return a runtime receipt"
            )
        runtime_fields = _runtime_memory_fields(raw_receipt)
        accounting_fields = {
            key: accounting[key]
            for key in (
                "serialized_plan_bytes",
                "resident_weight_bytes",
                "resident_weight_copy_count",
                "weight_streaming_active",
                "measurement_sources",
            )
        }
        if runtime_fields != accounting_fields:
            raise CaptureError(
                "dynamic runtime receipt disagrees with independent engine "
                f"accounting: runtime={runtime_fields!r}, "
                f"independent={accounting_fields!r}"
            )
    elif "runtime_memory_receipt" in result:
        raise CaptureError(
            "static baseline unexpectedly implements dynamic-memory introspection"
        )

    result["runtime_memory_receipt"] = {
        key: accounting[key]
        for key in (
            "serialized_plan_bytes",
            "resident_weight_bytes",
            "resident_weight_copy_count",
            "weight_streaming_active",
            "measurement_sources",
        )
    }
    result["runtime_attention_plans"] = runtime_attention_plans
    result["runtime_stack"] = runtime_stack
    result["runtime_libraries"] = runtime_libraries
    result["generation_workload"] = generation_workload
    result["tokenizer_contract"] = tokenizer_contract
    cuda_jit_cache = {
        **cache_configuration,
        "initial_state": _cuda_cache_initial_state(
            cache_configuration, cache_before
        ),
        "worker_started_ns": worker_started_ns,
        "worker_finished_ns": worker_finished_ns,
        "before": cache_before,
        "after": cache_after,
    }
    environment = _environment_identity()
    toolchain = {
        "worker": str(worker),
        "worker_sha256": _sha256(worker),
        "plugin_library": str(plugin_library),
        "plugin_library_sha256": _sha256(plugin_library),
        "capture_tool": str(Path(__file__).resolve()),
        "capture_tool_sha256": _sha256(Path(__file__).resolve()),
    }
    source_state_post = _source_state(
        repo_root, source_artifact_dir, label=f"{output.stem}-post"
    )
    source_state_unchanged = (
        source_state_pre["git_head"] == source_state_post["git_head"]
        and source_state_pre["source_state_sha256"]
        == source_state_post["source_state_sha256"]
    )
    if not source_state_unchanged:
        raise CaptureError("source state changed while running the benchmark")
    result["qualification_provenance"] = {
        "git_head": build_receipt["git_head"],
        "source_state_sha256": source_state_pre["source_state_sha256"],
        "source_state_pre_sha256": source_state_pre[
            "source_state_sha256"
        ],
        "source_state_post_sha256": source_state_post[
            "source_state_sha256"
        ],
        "source_state_unchanged": source_state_unchanged,
        "prebuild_source_state_sha256": build_receipt[
            "prebuild_source_state_sha256"
        ],
        "postbuild_source_state_sha256": build_receipt[
            "postbuild_source_state_sha256"
        ],
        "bundle_sha256": _sha256(bundle),
        "request_sha256": comparison_request_sha256,
        "model_revision": build_receipt["model_revision"],
        "precision": build_receipt["precision"],
        "target": build_receipt["target"],
        "toolchain_sha256": _canonical_sha(toolchain),
        "benchmark_environment_sha256": _canonical_sha(environment),
        "bundle_build_id": build_receipt["bundle_build_id"],
        "artifact_role": args.role,
        "fresh_build": build_receipt["fresh_build"],
        "artifact_reused": build_receipt["artifact_reused"],
        "runtime_attention_plans_sha256": _canonical_sha(
            runtime_attention_plans
        ),
        "runtime_stack_sha256": _canonical_sha(runtime_stack),
        "runtime_libraries_sha256": _canonical_sha(runtime_libraries),
        "cuda_jit_cache_sha256": _canonical_sha(cuda_jit_cache),
        "generation_workload_sha256": _canonical_sha(generation_workload),
        "tokenizer_contract_sha256": _canonical_sha(tokenizer_contract),
    }
    result["qualification_evidence"] = {
        "build_receipt": str(args.build_receipt.expanduser().resolve()),
        "build_receipt_sha256": _sha256(
            args.build_receipt.expanduser().resolve()
        ),
        "request_file": str(request),
        "request_file_sha256": _sha256(request),
        "comparison_request_sha256": comparison_request_sha256,
        "comparison_sequence_limit": args.comparison_sequence_limit,
        "worker_command": worker_command,
        "worker_command_sha256": _canonical_sha(worker_command),
        "engine_accounting": accounting,
        "toolchain": toolchain,
        "environment": environment,
        "live_runtime_stack": runtime_stack,
        "runtime_libraries": runtime_libraries,
        "cuda_jit_cache": cuda_jit_cache,
        "generation_workload": generation_workload,
        "tokenizer_contract": tokenizer_contract,
        "source_state": source_state_pre,
        "source_state_pre": source_state_pre,
        "source_state_post": source_state_post,
        "source_state_unchanged": source_state_unchanged,
        "worker_stderr": str(stderr_output),
        "worker_stderr_sha256": _sha256(stderr_output),
        "worker_stdout": str(stdout_output),
        "worker_stdout_sha256": _sha256(stdout_output),
    }
    _write_json(output, result)
    print(
        json.dumps(
            {"status": "completed", "output": str(output)}, sort_keys=True
        )
    )
    return 0


def _metadata_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--role", required=True, choices=ROLES)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--bundle-build-id", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="action", required=True)

    build = subparsers.add_parser(
        "build", help="run one fresh build and record its source-bound receipt"
    )
    build.add_argument("--bundle", type=Path, required=True)
    build.add_argument("--receipt", type=Path, required=True)
    build.add_argument("--source-artifact-dir", type=Path, required=True)
    build.add_argument("--stdout-output", type=Path, required=True)
    build.add_argument("--stderr-output", type=Path, required=True)
    build.add_argument("--cwd", type=Path, default=Path.cwd())
    _metadata_arguments(build)
    build.add_argument("command", nargs=argparse.REMAINDER)

    benchmark = subparsers.add_parser(
        "benchmark", help="run and enrich one real benchmark worker result"
    )
    benchmark.add_argument("--bundle", type=Path, required=True)
    benchmark.add_argument("--build-receipt", type=Path, required=True)
    benchmark.add_argument("--request", type=Path, required=True)
    benchmark.add_argument("--worker", type=Path, required=True)
    benchmark.add_argument("--plugin-library", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--stderr-output", type=Path, required=True)
    benchmark.add_argument(
        "--comparison-sequence-limit", type=int, required=True
    )
    benchmark.add_argument("--cwd", type=Path, default=Path.cwd())
    benchmark.add_argument("--role", required=True, choices=ROLES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.action == "build":
            for field in (
                "model_id",
                "model_revision",
                "precision",
                "target",
                "bundle_build_id",
            ):
                _require_nonempty(getattr(args, field), f"--{field}")
            return _cmd_build(args)
        return _cmd_benchmark(args)
    except (CaptureError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"capture_native_dynamic_memory_perf: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
