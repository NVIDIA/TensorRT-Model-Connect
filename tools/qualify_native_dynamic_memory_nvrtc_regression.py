#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reproduce and receipt the CUDA-13.0 NVRTC SDPA regression.

This producer is intentionally separate from ordinary performance parsing.
It executes the real cuDNN Frontend graphs in two fresh processes:

* the historical max/sum-exp optional-output graph must reproduce the exact
  ``eng3_k24=7`` NVRTC finalization failure; and
* the replacement standard-LSE graph must build and execute without either
  legacy optional output.

The probe executable and the CUDA-13.0 NVRTC/builtins files are held open for
the whole run.  The child is launched through the open executable descriptor,
the loader sees symlinks to the inherited library descriptors, and the
producer inspects the live process mappings before releasing the child.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import select
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_SCHEMA = "trtmc.nvrtc-optional-output-probe/v1"
RECEIPT_SCHEMA = "trtmc.nvrtc-optional-output-qualification/v1"
FRONTEND_REVISION = "7b9b711c22b6823e87150213ecd8449260db8610"
EXPECTED_SHAPE = {
    "query_heads": 16,
    "kv_heads": 8,
    "head_dim": 128,
    "query_rows": 1,
    "history_rows": 512,
    "valid_history_rows": 511,
}
EXPECTED_RUNTIME = {
    "device_name": "NVIDIA GB300",
    "sm": 103,
    "cuda_runtime_version": 13030,
    "cuda_driver_api_version": 13030,
    "cudnn_backend_version": 92000,
    "cudnn_frontend_revision": FRONTEND_REVISION,
    "nvrtc_major": 13,
    "nvrtc_minor": 0,
}
EXPECTED_DRIVER_VERSION = "580.105.08"
EXPECTED_LEGACY_PLAN = "eng3_k24=7"
EXPECTED_FAILURE_FRAGMENTS = (
    "compilationResult != NVRTC_SUCCESS",
    "CUDNN_STATUS_INTERNAL_ERROR_COMPILATION_FAILED",
)
_HEX_DEVICE = re.compile(r"^(?P<major>[0-9a-fA-F]+):(?P<minor>[0-9a-fA-F]+)$")
_DRIVER_VERSION_PATTERNS = (
    re.compile(
        r"NVRM version:.*?\s(?P<version>[0-9]+(?:\.[0-9]+)+)"
        r"\s+Release",
        re.DOTALL,
    ),
    re.compile(r"Kernel Module\s+(?P<version>[0-9]+(?:\.[0-9]+)+)"),
)
_RUNTIME_LIBRARY_NAMES = {
    "cuda_runtime": re.compile(r"^libcudart\.so\.13(?:\.[0-9]+)*$"),
    "cudnn": re.compile(r"^libcudnn\.so\.9(?:\.[0-9]+)*$"),
    # Process maps normally show the real driver filename
    # (libcuda.so.<driver-version>), not the libcuda.so.1 symlink.
    "cuda_driver": re.compile(
        r"^libcuda\.so\.(?:1|[0-9]+(?:\.[0-9]+)*)$"
    ),
    "nvrtc": re.compile(r"^libnvrtc\.so\.13(?:\.[0-9]+)*$"),
    "nvrtc_builtins": re.compile(
        r"^libnvrtc-builtins\.so\.13\.0(?:\.[0-9]+)*$"
    ),
}


class QualificationError(RuntimeError):
    """Evidence was absent, ambiguous, or contradicted the release contract."""


def _load_boundary_module():
    path = Path(__file__).with_name("qualify_native_dynamic_memory.py")
    spec = importlib.util.spec_from_file_location(
        "_trtmc_nvrtc_regression_boundary", path
    )
    if spec is None or spec.loader is None:
        raise QualificationError(
            f"cannot import source-state helpers from {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_stream(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_path(path: Path) -> str:
    with path.open("rb") as stream:
        return _sha256_stream(stream)


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity_from_fd(fd: int, *, requested_path: Path) -> dict[str, Any]:
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        raise QualificationError(f"{requested_path} is not a regular file")
    with os.fdopen(os.dup(fd), "rb") as stream:
        stream.seek(0)
        digest = _sha256_stream(stream)
    return {
        "requested_path": str(requested_path),
        "canonical_path": str(requested_path.resolve(strict=True)),
        "device": metadata.st_dev,
        "device_major": os.major(metadata.st_dev),
        "device_minor": os.minor(metadata.st_dev),
        "inode": metadata.st_ino,
        "size_bytes": metadata.st_size,
        "mtime_ns": metadata.st_mtime_ns,
        "sha256": digest,
    }


def _assert_elf(fd: int, *, label: str) -> None:
    header = os.pread(fd, 4, 0)
    if header != b"\x7fELF":
        raise QualificationError(f"{label} is not an ELF file")


@dataclass
class PinnedInputs:
    probe_fd: int
    nvrtc_fd: int
    builtins_fd: int
    probe: dict[str, Any]
    nvrtc: dict[str, Any]
    builtins: dict[str, Any]

    @classmethod
    def open(
        cls, *, probe: Path, nvrtc: Path, builtins: Path
    ) -> "PinnedInputs":
        descriptors: list[int] = []
        try:
            for path in (probe, nvrtc, builtins):
                descriptors.append(os.open(path, os.O_RDONLY))
            probe_fd, nvrtc_fd, builtins_fd = descriptors
            _assert_elf(probe_fd, label="probe")
            _assert_elf(nvrtc_fd, label="NVRTC")
            _assert_elf(builtins_fd, label="NVRTC builtins")
            probe_identity = _identity_from_fd(
                probe_fd, requested_path=probe
            )
            nvrtc_identity = _identity_from_fd(
                nvrtc_fd, requested_path=nvrtc
            )
            builtins_identity = _identity_from_fd(
                builtins_fd, requested_path=builtins
            )
            return cls(
                probe_fd,
                nvrtc_fd,
                builtins_fd,
                probe_identity,
                nvrtc_identity,
                builtins_identity,
            )
        except Exception:
            for descriptor in descriptors:
                os.close(descriptor)
            raise

    def close(self) -> None:
        for descriptor in (
            self.probe_fd,
            self.nvrtc_fd,
            self.builtins_fd,
        ):
            os.close(descriptor)

    def __enter__(self) -> "PinnedInputs":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @property
    def pass_fds(self) -> tuple[int, int, int]:
        return self.probe_fd, self.nvrtc_fd, self.builtins_fd


def _mapping_rows(pid: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    maps_path = Path(f"/proc/{pid}/maps")
    try:
        contents = maps_path.read_text(encoding="utf-8")
    except OSError as error:
        raise QualificationError(
            f"cannot inspect live child mappings at {maps_path}: {error}"
        ) from error
    for raw_line in contents.splitlines():
        fields = raw_line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        address, permissions, offset, device, inode_text, pathname = fields
        if not pathname.startswith("/"):
            continue
        match = _HEX_DEVICE.fullmatch(device)
        if match is None:
            raise QualificationError(
                f"malformed device field in {maps_path}: {device!r}"
            )
        try:
            inode = int(inode_text)
        except ValueError as error:
            raise QualificationError(
                f"malformed inode field in {maps_path}: {inode_text!r}"
            ) from error
        rows.append(
            {
                "address": address,
                "permissions": permissions,
                "offset": offset,
                "device_major": int(match.group("major"), 16),
                "device_minor": int(match.group("minor"), 16),
                "inode": inode,
                "path": pathname,
            }
        )
    return rows


def _mapping_matches_identity(
    row: Mapping[str, Any], identity: Mapping[str, Any]
) -> bool:
    return (
        row.get("device_major") == identity.get("device_major")
        and row.get("device_minor") == identity.get("device_minor")
        and row.get("inode") == identity.get("inode")
    )


def _group_mappings(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            int(row["device_major"]),
            int(row["device_minor"]),
            int(row["inode"]),
            str(row["path"]),
        )
        entry = grouped.setdefault(
            key,
            {
                "device_major": key[0],
                "device_minor": key[1],
                "inode": key[2],
                "path": key[3],
                "segments": [],
            },
        )
        entry["segments"].append(
            {
                "address": row["address"],
                "permissions": row["permissions"],
                "offset": row["offset"],
            }
        )
    return sorted(
        grouped.values(),
        key=lambda value: (
            value["path"],
            value["device_major"],
            value["device_minor"],
            value["inode"],
        ),
    )


def _one_pinned_mapping(
    rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    matches = [
        row for row in rows if _mapping_matches_identity(row, identity)
    ]
    groups = _group_mappings(matches)
    if len(groups) != 1:
        raise QualificationError(
            f"live child must map exactly one pinned {label} identity; "
            f"found {len(groups)}"
        )
    if not any(
        "x" in segment["permissions"]
        for segment in groups[0]["segments"]
    ):
        raise QualificationError(
            f"live child {label} mapping has no executable segment"
        )
    return groups[0]


def _mapped_runtime_libraries(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {
        label: [] for label in _RUNTIME_LIBRARY_NAMES
    }
    for group in _group_mappings(rows):
        basename = Path(str(group["path"])).name
        for label, pattern in _RUNTIME_LIBRARY_NAMES.items():
            if pattern.fullmatch(basename):
                result[label].append(group)
    return result


def _mapped_file_identity(
    pid: int, group: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    mapped_path = group.get("path")
    if not isinstance(mapped_path, str) or not mapped_path.startswith("/"):
        raise QualificationError(
            f"{label} mapping does not contain an absolute path"
        )
    if mapped_path.endswith(" (deleted)"):
        raise QualificationError(f"{label} mapping is deleted")
    process_path = Path(f"/proc/{pid}/root") / mapped_path.lstrip("/")
    try:
        descriptor = os.open(process_path, os.O_RDONLY)
    except OSError as error:
        raise QualificationError(
            f"cannot pin live {label} mapping {mapped_path}: {error}"
        ) from error
    try:
        metadata = os.fstat(descriptor)
        observed = {
            "device_major": os.major(metadata.st_dev),
            "device_minor": os.minor(metadata.st_dev),
            "inode": metadata.st_ino,
        }
        if not _mapping_matches_identity(observed, group):
            raise QualificationError(
                f"live {label} mapping changed before it was hashed"
            )
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            digest = _sha256_stream(stream)
        return {
            "mapped_path": mapped_path,
            "device": metadata.st_dev,
            "device_major": observed["device_major"],
            "device_minor": observed["device_minor"],
            "inode": metadata.st_ino,
            "size_bytes": metadata.st_size,
            "mtime_ns": metadata.st_mtime_ns,
            "sha256": digest,
        }
    finally:
        os.close(descriptor)


def validate_runtime_mapping_set(
    rows: Sequence[Mapping[str, Any]],
    *,
    nvrtc_identity: Mapping[str, Any],
    builtins_identity: Mapping[str, Any],
) -> dict[str, Any]:
    pinned_nvrtc = _one_pinned_mapping(
        rows, nvrtc_identity, label="NVRTC"
    )
    pinned_builtins = _one_pinned_mapping(
        rows, builtins_identity, label="NVRTC builtins"
    )
    libraries = _mapped_runtime_libraries(rows)
    for label in ("cuda_runtime", "cudnn", "cuda_driver"):
        if len(libraries[label]) != 1:
            raise QualificationError(
                f"live child must map exactly one {label} identity; "
                f"found {len(libraries[label])}"
            )
    if len(libraries["nvrtc"]) != 1 or not _mapping_matches_identity(
        libraries["nvrtc"][0], nvrtc_identity
    ):
        raise QualificationError(
            "live child has a competing or missing NVRTC mapping"
        )
    if len(
        libraries["nvrtc_builtins"]
    ) != 1 or not _mapping_matches_identity(
        libraries["nvrtc_builtins"][0], builtins_identity
    ):
        raise QualificationError(
            "live child has a competing or missing NVRTC builtins mapping"
        )
    return {
        "pinned_nvrtc": pinned_nvrtc,
        "pinned_nvrtc_builtins": pinned_builtins,
        "runtime_libraries": libraries,
    }


def pin_mapped_runtime_libraries(
    pid: int,
    mapping_evidence: Mapping[str, Any],
    *,
    nvrtc_identity: Mapping[str, Any],
    builtins_identity: Mapping[str, Any],
) -> dict[str, Any]:
    libraries = mapping_evidence.get("runtime_libraries")
    if not isinstance(libraries, Mapping):
        raise QualificationError(
            "validated mapping evidence omitted runtime_libraries"
        )
    result: dict[str, Any] = {}
    for label in _RUNTIME_LIBRARY_NAMES:
        groups = libraries.get(label)
        if not isinstance(groups, list) or len(groups) != 1:
            raise QualificationError(
                f"cannot pin ambiguous live {label} mapping"
            )
        result[label] = _mapped_file_identity(
            pid, groups[0], label=label
        )
    for label, expected in (
        ("nvrtc", nvrtc_identity),
        ("nvrtc_builtins", builtins_identity),
    ):
        actual = result[label]
        if (
            not _mapping_matches_identity(actual, expected)
            or actual["size_bytes"] != expected["size_bytes"]
            or actual["sha256"] != expected["sha256"]
        ):
            raise QualificationError(
                f"live mapped {label} file disagrees with its pinned input"
            )
    return result


def _validate_dladdr_path(
    value: object,
    *,
    pid: int,
    nvrtc_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, str) or not value.startswith("/"):
        raise QualificationError(
            "probe runtime.nvrtc_dladdr_path must be an absolute path"
        )
    # /proc/self paths emitted by the child must be resolved in that child's
    # namespace, not in the producer.
    child_path = (
        Path(f"/proc/{pid}/root") / value.lstrip("/")
        if not value.startswith("/proc/self/")
        else Path(f"/proc/{pid}") / value[len("/proc/self/") :]
    )
    try:
        metadata = child_path.stat()
    except OSError as error:
        raise QualificationError(
            f"cannot reopen probe NVRTC dladdr path {value}: {error}"
        ) from error
    actual = {
        "path": value,
        "device_major": os.major(metadata.st_dev),
        "device_minor": os.minor(metadata.st_dev),
        "inode": metadata.st_ino,
    }
    if not _mapping_matches_identity(actual, nvrtc_identity):
        raise QualificationError(
            "probe nvrtc_dladdr_path does not identify the pinned NVRTC file"
        )
    return actual


def _cache_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "entry_count": 0,
            "file_count": 0,
            "total_bytes": 0,
            "sha256": _canonical_json_sha256([]),
        }
    rows: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*")):
        relative = candidate.relative_to(path).as_posix()
        metadata = candidate.lstat()
        if candidate.is_symlink():
            raise QualificationError(
                f"CUDA cache contains a symlink: {candidate}"
            )
        if candidate.is_dir():
            rows.append({"path": relative, "type": "directory"})
        elif candidate.is_file():
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "size_bytes": metadata.st_size,
                    "sha256": _sha256_path(candidate),
                }
            )
        else:
            raise QualificationError(
                f"CUDA cache contains a non-regular entry: {candidate}"
            )
    return {
        "exists": True,
        "entry_count": len(rows),
        "file_count": sum(row["type"] == "file" for row in rows),
        "total_bytes": sum(
            int(row.get("size_bytes", 0)) for row in rows
        ),
        "sha256": _canonical_json_sha256(rows),
    }


def _write_text_absent(path: Path, value: str) -> dict[str, Any]:
    if path.exists():
        raise QualificationError(
            f"refusing to overwrite evidence artifact: {path}"
        )
    path.write_text(value, encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_path(path),
    }


def _file_identity(path: Path) -> dict[str, Any]:
    metadata = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": metadata.st_size,
        "sha256": _sha256_path(path),
    }


def validate_probe_payload(
    payload: Mapping[str, Any], *, mode: str
) -> dict[str, bool]:
    if payload.get("schema_version") != PROBE_SCHEMA:
        raise QualificationError(
            f"{mode} probe schema is not {PROBE_SCHEMA}"
        )
    if payload.get("mode") != mode or payload.get("probe_passed") is not True:
        raise QualificationError(
            f"{mode} probe did not report its exact passing mode"
        )
    if payload.get("shape") != EXPECTED_SHAPE:
        raise QualificationError(
            f"{mode} probe geometry drifted from the historical Qwen case"
        )
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise QualificationError(f"{mode} probe runtime is not an object")
    for field, expected in EXPECTED_RUNTIME.items():
        if runtime.get(field) != expected:
            raise QualificationError(
                f"{mode} runtime.{field}={runtime.get(field)!r}, "
                f"expected {expected!r}"
            )
    pid = runtime.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise QualificationError(f"{mode} runtime.pid is invalid")

    contract = payload.get("graph_contract")
    result = payload.get("result")
    if not isinstance(contract, Mapping) or not isinstance(result, Mapping):
        raise QualificationError(
            f"{mode} graph_contract/result must be objects"
        )
    common_contract = (
        contract.get("output_context") is True
        and isinstance(contract.get("serialized_contract_bytes"), int)
        and contract.get("serialized_contract_bytes", 0) > 0
    )
    if not common_contract:
        raise QualificationError(
            f"{mode} graph contract is incomplete"
        )

    if mode == "legacy":
        contract_passed = (
            contract.get("generate_stats") is False
            and contract.get("optional_logit_max") is True
            and contract.get("optional_score_sum_exp") is True
            and contract.get("output_log_sum_exp") is False
            and contract.get(
                "serialized_contains_legacy_logit_max"
            )
            is True
            and contract.get(
                "serialized_contains_legacy_score_sum_exp"
            )
            is True
            and contract.get("serialized_contains_log_sum_exp") is False
        )
        message = result.get("candidate_build_message")
        failure_passed = (
            result.get("candidate_index") == 0
            and result.get("candidate_plan") == EXPECTED_LEGACY_PLAN
            and result.get("candidate_build_succeeded") is False
            and result.get("expected_nvrtc_failure_observed") is True
            and result.get("fallback_plan_selected") is False
            and result.get("graph_executed") is False
            and isinstance(message, str)
            and all(fragment in message for fragment in EXPECTED_FAILURE_FRAGMENTS)
        )
        if not contract_passed or not failure_passed:
            raise QualificationError(
                "legacy executable did not preserve the exact optional-output "
                "graph and preferred-candidate NVRTC failure"
            )
        return {
            "graph_contract": True,
            "expected_failure": True,
            "no_fallback": True,
        }

    if mode != "lse":
        raise QualificationError(f"unknown probe mode: {mode}")
    contract_passed = (
        contract.get("generate_stats") is True
        and contract.get("optional_logit_max") is False
        and contract.get("optional_score_sum_exp") is False
        and contract.get("output_log_sum_exp") is True
        and contract.get("serialized_contains_legacy_logit_max") is False
        and contract.get("serialized_contains_legacy_score_sum_exp")
        is False
        and contract.get("serialized_contains_log_sum_exp") is True
    )
    execution_passed = (
        result.get("graph_build_succeeded") is True
        and result.get("graph_executed") is True
        and result.get("device_synchronize_succeeded") is True
        and result.get("finite_lse_observed") is True
        and result.get("legacy_optional_outputs_bound") is False
        and isinstance(result.get("selected_plan"), str)
        and bool(result.get("selected_plan"))
    )
    if not contract_passed or not execution_passed:
        raise QualificationError(
            "standard-LSE executable either retained a legacy optional "
            "output or failed to build and execute"
        )
    return {
        "graph_contract": True,
        "build_and_execute": True,
        "legacy_outputs_absent": True,
    }


def validate_graph_artifact(path: Path, *, mode: str) -> dict[str, Any]:
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(
            f"{mode} graph artifact is not readable JSON: {error}"
        ) from error
    if not isinstance(graph, Mapping):
        raise QualificationError(f"{mode} graph artifact is not an object")
    if (
        graph.get("json_version") != "1.0"
        or graph.get("cudnn_backend_version") != "9.20.0"
        or graph.get("cudnn_frontend_version") != 12100
    ):
        raise QualificationError(
            f"{mode} graph artifact runtime/frontend schema drifted"
        )
    nodes = graph.get("nodes")
    tensors = graph.get("tensors")
    if (
        not isinstance(nodes, list)
        or len(nodes) != 1
        or not isinstance(nodes[0], Mapping)
        or not isinstance(tensors, Mapping)
    ):
        raise QualificationError(
            f"{mode} graph artifact must contain exactly one SDPA node"
        )
    node = nodes[0]
    if (
        node.get("tag") != "SDPA"
        or node.get("inputs")
        != {"Q": 1, "K": 2, "V": 3, "SEQ_LEN_Q": 5, "SEQ_LEN_KV": 6}
        or node.get("padding_mask") is not True
    ):
        raise QualificationError(
            f"{mode} graph artifact is not the historical padded Qwen SDPA"
        )
    expected_names = {
        "1": ("query", [1, 16, 1, 128]),
        "2": ("key_history_token_major", [1, 8, 512, 128]),
        "3": ("value_history_token_major", [1, 8, 512, 128]),
        "4": ("context", [1, 16, 1, 128]),
        "5": ("sequence_length_q", [1, 1, 1, 1]),
        "6": ("sequence_length_history", [1, 1, 1, 1]),
    }
    for uid, (name, dimensions) in expected_names.items():
        tensor = tensors.get(uid)
        if (
            not isinstance(tensor, Mapping)
            or tensor.get("name") != name
            or tensor.get("dim") != dimensions
        ):
            raise QualificationError(
                f"{mode} graph tensor {uid} drifted from {name}"
            )

    if mode == "legacy":
        expected_outputs = {"Max": 7, "O": 4, "Sum_exp": 8}
        expected_optional = {
            "7": "legacy_logit_max",
            "8": "legacy_score_sum_exp",
        }
        if (
            node.get("name")
            != "trtmc_legacy_optional_output_history_sdpa"
            or node.get("generate_stats") is not False
            or node.get("outputs") != expected_outputs
            or set(tensors) != set(expected_names) | set(expected_optional)
        ):
            raise QualificationError(
                "legacy graph artifact does not contain exactly Max/O/Sum_exp"
            )
        for uid, name in expected_optional.items():
            tensor = tensors.get(uid)
            if (
                not isinstance(tensor, Mapping)
                or tensor.get("name") != name
                or tensor.get("dim") != [1, 16, 1, 1]
            ):
                raise QualificationError(
                    f"legacy optional-output tensor {uid} is invalid"
                )
        return {
            "node_name": node["name"],
            "generate_stats": False,
            "outputs": expected_outputs,
            "independent_contract_validation": True,
        }

    if mode != "lse":
        raise QualificationError(f"unknown graph artifact mode: {mode}")
    if (
        node.get("name") != "trtmc_standard_lse_history_sdpa"
        or node.get("generate_stats") is not True
        or node.get("outputs") != {"O": 4, "Stats": 9}
        or set(tensors) != set(expected_names) | {"9"}
    ):
        raise QualificationError(
            "standard-LSE graph artifact retained legacy optional outputs "
            "or omitted Stats"
        )
    stats = tensors.get("9")
    if (
        not isinstance(stats, Mapping)
        or stats.get("name") != "log_sum_exp"
        or stats.get("dim") != [1, 16, 1, 1]
    ):
        raise QualificationError(
            "standard-LSE Stats tensor is not log_sum_exp"
        )
    serialized = json.dumps(graph, sort_keys=True)
    if (
        "legacy_logit_max" in serialized
        or "legacy_score_sum_exp" in serialized
        or '"Max"' in serialized
        or '"Sum_exp"' in serialized
    ):
        raise QualificationError(
            "standard-LSE graph artifact still contains a legacy output"
        )
    return {
        "node_name": node["name"],
        "generate_stats": True,
        "outputs": {"O": 4, "Stats": 9},
        "legacy_outputs_absent": True,
        "independent_contract_validation": True,
    }


def _runtime_comparison(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    fields = (
        "device",
        "device_name",
        "sm",
        "cuda_runtime_version",
        "cuda_driver_api_version",
        "cudnn_backend_version",
        "cudnn_frontend_revision",
        "nvrtc_major",
        "nvrtc_minor",
    )
    return all(left.get(field) == right.get(field) for field in fields)


def _read_probe_line(
    process: subprocess.Popen[str], *, timeout_seconds: float
) -> str:
    assert process.stdout is not None
    ready, _, _ = select.select(
        [process.stdout], [], [], timeout_seconds
    )
    if not ready:
        raise QualificationError(
            f"probe did not produce its ready receipt within "
            f"{timeout_seconds:.0f}s"
        )
    line = process.stdout.readline()
    if not line:
        assert process.stderr is not None
        stderr = process.stderr.read()
        raise QualificationError(
            f"probe exited before its ready receipt: {stderr.strip()}"
        )
    return line


def _run_mode(
    *,
    mode: str,
    pinned: PinnedInputs,
    evidence_dir: Path,
    shadow_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    graph_path = evidence_dir / f"{mode}.graph.json"
    stdout_path = evidence_dir / f"{mode}.stdout.log"
    stderr_path = evidence_dir / f"{mode}.stderr.log"
    cache_path = evidence_dir / f"{mode}.cuda-cache"
    if any(
        path.exists()
        for path in (graph_path, stdout_path, stderr_path, cache_path)
    ):
        raise QualificationError(
            f"{mode} evidence paths must be absent before execution"
        )
    cache_path.mkdir()
    cache_pre = _cache_snapshot(cache_path)
    if (
        cache_pre["entry_count"] != 0
        or cache_pre["file_count"] != 0
        or cache_pre["total_bytes"] != 0
    ):
        raise QualificationError(
            f"{mode} CUDA cache was not empty before execution"
        )

    display_command = [
        pinned.probe["canonical_path"],
        "--mode",
        mode,
        "--graph-output",
        str(graph_path.resolve()),
    ]
    fd_command = [
        f"/proc/self/fd/{pinned.probe_fd}",
        "--mode",
        mode,
        "--graph-output",
        str(graph_path.resolve()),
    ]
    environment = os.environ.copy()
    previous_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        str(shadow_dir)
        + (os.pathsep + previous_library_path if previous_library_path else "")
    )
    environment["CUDA_CACHE_PATH"] = str(cache_path.resolve())
    environment["CUDA_CACHE_DISABLE"] = "0"
    environment["TRTMC_NVRTC_PROBE_WAIT_FOR_RELEASE"] = "1"

    started_ns = time.time_ns()
    process = subprocess.Popen(
        fd_command,
        cwd=REPO_ROOT,
        env=environment,
        pass_fds=pinned.pass_fds,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    ready_line = ""
    remaining_stdout = ""
    stderr = ""
    try:
        ready_line = _read_probe_line(
            process, timeout_seconds=timeout_seconds
        )
        try:
            payload = json.loads(ready_line)
        except json.JSONDecodeError as error:
            raise QualificationError(
                f"{mode} probe ready line is not JSON: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise QualificationError(
                f"{mode} probe ready payload is not an object"
            )
        gates = validate_probe_payload(payload, mode=mode)
        runtime = payload["runtime"]
        if runtime["pid"] != process.pid:
            raise QualificationError(
                f"{mode} probe-reported PID does not match the child PID"
            )
        rows = _mapping_rows(process.pid)
        mapping_evidence = validate_runtime_mapping_set(
            rows,
            nvrtc_identity=pinned.nvrtc,
            builtins_identity=pinned.builtins,
        )
        mapped_file_identities = pin_mapped_runtime_libraries(
            process.pid,
            mapping_evidence,
            nvrtc_identity=pinned.nvrtc,
            builtins_identity=pinned.builtins,
        )
        dladdr_identity = _validate_dladdr_path(
            runtime.get("nvrtc_dladdr_path"),
            pid=process.pid,
            nvrtc_identity=pinned.nvrtc,
        )
        assert process.stdin is not None
        process.stdin.write("\n")
        process.stdin.flush()
        remaining_stdout, stderr = process.communicate(
            timeout=timeout_seconds
        )
        if process.returncode != 0:
            raise QualificationError(
                f"{mode} probe exited {process.returncode}: "
                f"{stderr.strip()}"
            )
        if remaining_stdout.strip():
            raise QualificationError(
                f"{mode} probe emitted unexpected stdout after its receipt"
            )
    except Exception:
        if process.poll() is None:
            process.kill()
        try:
            tail_stdout, tail_stderr = process.communicate(timeout=5)
            remaining_stdout += tail_stdout
            stderr += tail_stderr
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finished_ns = time.time_ns()

    stdout_identity = _write_text_absent(
        stdout_path, ready_line + remaining_stdout
    )
    stderr_identity = _write_text_absent(stderr_path, stderr)
    if not graph_path.is_file():
        raise QualificationError(
            f"{mode} probe did not write its graph contract"
        )
    graph_identity = _file_identity(graph_path)
    independently_validated_graph = validate_graph_artifact(
        graph_path, mode=mode
    )
    graph_contract = payload["graph_contract"]
    if (
        graph_identity["size_bytes"]
        != graph_contract["serialized_contract_bytes"]
    ):
        raise QualificationError(
            f"{mode} graph artifact size disagrees with the probe receipt"
        )
    cache_post = _cache_snapshot(cache_path)
    return {
        "mode": mode,
        "command": display_command,
        "execution_via_open_descriptor": True,
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "returncode": process.returncode,
        "probe_receipt": payload,
        "probe_gates": gates,
        "live_mappings": mapping_evidence,
        "mapped_runtime_library_files": mapped_file_identities,
        "nvrtc_dladdr_identity": dladdr_identity,
        "cuda_cache": {
            "path": str(cache_path.resolve()),
            "private_to_process": True,
            "pre": cache_pre,
            "post": cache_post,
        },
        "artifacts": {
            "graph_contract": graph_identity,
            "independent_graph_validation": independently_validated_graph,
            "stdout": stdout_identity,
            "stderr": stderr_identity,
        },
    }


def _source_snapshot(
    artifact_dir: Path, *, label: str
) -> dict[str, Any]:
    boundary = _load_boundary_module()
    return boundary.source_state_provenance(
        REPO_ROOT,
        Path(__file__),
        artifact_dir,
        label=label,
    )


def _driver_version() -> dict[str, Any]:
    path = Path("/proc/driver/nvidia/version")
    if not path.is_file():
        raise QualificationError(
            f"NVIDIA driver version evidence is missing: {path}"
        )
    contents = path.read_text(encoding="utf-8")
    match = next(
        (
            candidate
            for pattern in _DRIVER_VERSION_PATTERNS
            if (candidate := pattern.search(contents)) is not None
        ),
        None,
    )
    if match is None:
        raise QualificationError(
            "cannot parse NVIDIA kernel driver version"
        )
    return {
        "path": str(path),
        "version": match.group("version"),
        "sha256": hashlib.sha256(contents.encode("utf-8")).hexdigest(),
    }


def _create_shadow_directory(
    directory: Path, pinned: PinnedInputs
) -> None:
    directory.mkdir()
    (directory / "libnvrtc.so.13").symlink_to(
        f"/proc/self/fd/{pinned.nvrtc_fd}"
    )
    (directory / "libnvrtc-builtins.so.13.0").symlink_to(
        f"/proc/self/fd/{pinned.builtins_fd}"
    )


def _validate_requested_names(
    nvrtc: Path, builtins: Path
) -> None:
    if nvrtc.name != "libnvrtc.so.13":
        raise QualificationError(
            "--nvrtc must name exactly libnvrtc.so.13"
        )
    if builtins.name != "libnvrtc-builtins.so.13.0":
        raise QualificationError(
            "--nvrtc-builtins must name exactly "
            "libnvrtc-builtins.so.13.0"
        )


def _write_receipt(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False
    ) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists():
        raise QualificationError(
            f"refusing to overwrite qualification receipt: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_dir = output.with_name(output.stem + ".evidence")
    if evidence_dir.exists():
        raise QualificationError(
            f"refusing to overwrite evidence directory: {evidence_dir}"
        )
    evidence_dir.mkdir()

    _validate_requested_names(args.nvrtc, args.nvrtc_builtins)
    source_pre = _source_snapshot(evidence_dir, label="nvrtc-regression-pre")
    report: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "running",
        "passed": False,
        "promotion_eligible": False,
        "source_state_pre": source_pre,
        "source_state_post": None,
        "source_state_unchanged": False,
        "qualification_gates": {},
    }
    _write_receipt(output, report)

    try:
        with PinnedInputs.open(
            probe=args.probe.resolve(strict=True),
            nvrtc=args.nvrtc.resolve(strict=True),
            builtins=args.nvrtc_builtins.resolve(strict=True),
        ) as pinned:
            shadow_dir = evidence_dir / "nvrtc-13.0-loader"
            _create_shadow_directory(shadow_dir, pinned)
            legacy = _run_mode(
                mode="legacy",
                pinned=pinned,
                evidence_dir=evidence_dir,
                shadow_dir=shadow_dir,
                timeout_seconds=args.timeout_seconds,
            )
            lse = _run_mode(
                mode="lse",
                pinned=pinned,
                evidence_dir=evidence_dir,
                shadow_dir=shadow_dir,
                timeout_seconds=args.timeout_seconds,
            )
            runtime_same = _runtime_comparison(
                legacy["probe_receipt"]["runtime"],
                lse["probe_receipt"]["runtime"],
            )
            if not runtime_same:
                raise QualificationError(
                    "legacy and LSE processes used different runtime tuples"
                )
            distinct_processes = (
                legacy["probe_receipt"]["runtime"]["pid"]
                != lse["probe_receipt"]["runtime"]["pid"]
            )
            if not distinct_processes:
                raise QualificationError(
                    "legacy and LSE evidence did not come from fresh processes"
                )

            driver = _driver_version()
            if driver.get("version") != EXPECTED_DRIVER_VERSION:
                raise QualificationError(
                    f"NVIDIA driver version {driver.get('version')!r}, "
                    f"expected {EXPECTED_DRIVER_VERSION!r}"
                )
            report.update(
                {
                    "probe": pinned.probe,
                    "forced_nvrtc": pinned.nvrtc,
                    "forced_nvrtc_builtins": pinned.builtins,
                    "driver": driver,
                    "runs": {"legacy": legacy, "lse": lse},
                }
            )
            report["qualification_gates"].update(
                {
                    "probe_executed_via_pinned_descriptor": True,
                    "nvrtc_13_0_pair_pinned": True,
                    "live_mappings_match_pinned_pair": True,
                    "mapped_cuda_stack_files_hashed": True,
                    "no_competing_nvrtc_mapping": True,
                    "legacy_optional_output_contract": True,
                    "legacy_exact_nvrtc_failure": True,
                    "legacy_fallback_not_selected": True,
                    "standard_lse_contract": True,
                    "standard_lse_build_and_execute": True,
                    "standard_lse_legacy_outputs_absent": True,
                    "fresh_processes": distinct_processes,
                    "private_initially_empty_cuda_caches": True,
                    "runtime_tuple_identical": runtime_same,
                }
            )
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        source_post = _source_snapshot(
            evidence_dir, label="nvrtc-regression-failed-post"
        )
        report["source_state_post"] = source_post
        report["source_state_unchanged"] = bool(
            source_pre.get("git_head") == source_post.get("git_head")
            and source_pre.get("source_state_sha256")
            == source_post.get("source_state_sha256")
        )
        report["qualification_gates"]["source_state_unchanged"] = report[
            "source_state_unchanged"
        ]
        _write_receipt(output, report)
        raise

    source_post = _source_snapshot(
        evidence_dir, label="nvrtc-regression-post"
    )
    source_unchanged = bool(
        source_pre.get("git_head") == source_post.get("git_head")
        and source_pre.get("source_state_sha256")
        == source_post.get("source_state_sha256")
    )
    report["source_state_post"] = source_post
    report["source_state_unchanged"] = source_unchanged
    report["qualification_gates"][
        "source_state_unchanged"
    ] = source_unchanged
    exact_head = bool(
        source_pre.get("exact_head_gate_satisfied") is True
        and source_post.get("exact_head_gate_satisfied") is True
    )
    report["qualification_gates"]["clean_exact_head"] = exact_head
    component_gates = all(
        value is True
        for key, value in report["qualification_gates"].items()
        if key != "clean_exact_head"
    )
    report["status"] = "completed" if component_gates else "failed"
    report["passed"] = component_gates
    report["promotion_eligible"] = component_gates and exact_head
    report["receipt_sha256"] = _canonical_json_sha256(
        {
            key: value
            for key, value in report.items()
            if key != "receipt_sha256"
        }
    )
    _write_receipt(output, report)
    if not component_gates:
        raise QualificationError(
            "one or more mandatory NVRTC regression gates failed"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the historical CUDA-13.0 NVRTC optional-output "
            "regression and its standard-LSE replacement"
        )
    )
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--nvrtc", type=Path, required=True)
    parser.add_argument("--nvrtc-builtins", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timeout-seconds", type=float, default=120.0
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    try:
        report = qualify(args)
    except (OSError, QualificationError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "receipt": str(args.output.resolve()),
                "passed": report["passed"],
                "promotion_eligible": report["promotion_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
