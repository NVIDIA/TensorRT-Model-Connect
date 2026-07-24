#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Produce source-bound cold/warm and cross-GPU process-isolation evidence.

This producer executes ``capture_native_dynamic_memory_perf.py benchmark``
four times against one native-dynamic bundle, request, and build receipt:

* a cold process on GPU A with a private empty CUDA JIT cache;
* a new warm process on GPU A reusing only that cache directory;
* two concurrent processes on GPUs A and B with separate empty caches.

The aggregate receipt binds those direct isolation observations to explicit
full-matrix Hugging Face correctness and static-versus-dynamic performance
qualification reports.  All three evidence producers must name the same
bundle, source snapshot, model revision, and live runtime stack.  The
performance and isolation captures must additionally name the same mapped
runtime libraries.

The aggregate does not imply that every isolation child independently
recomputed Hugging Face logits or measured the static baseline.  It records
those as exact-tuple companion receipts rather than overstating the four
child processes' work.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence, TextIO


REPORT_SCHEMA = "trtmc.native-dynamic-memory-process-isolation/v2"
CAPTURE_RESULT_SCHEMA = "trtmc.benchmark-worker-result/v1"
PERFORMANCE_REPORT_SCHEMA = (
    "trtmc.native-dynamic-memory-perf-qualification/v1"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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
_SAFE_ENVIRONMENT_NAMES = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "TZ",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "CMAKE_PREFIX_PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "CUDA_HOME",
    "CUDA_ROOT",
    "CUDA_MODULE_LOADING",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "TRTMC_BACKEND_DIR",
    "TRTMC_MODEL_PLUGIN_DIR",
    "TRTMC_OPTIMIZED_RUNTIME_DIR",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "XDG_CACHE_HOME",
)
_RUN_LABELS = (
    "gpu-a-cold",
    "gpu-a-warm",
    "gpu-a-concurrent",
    "gpu-b-concurrent",
)
_QUALIFIED_ENGINE_GRAPH_GATES = (
    "actual_split_engine_sections",
    "distinct_prefill_decode_plans",
    "no_attention_mask_input",
    "current_rows_only_present_outputs",
    "native_segmented_attention_covers_full_model",
    "no_dense_attention_mask_or_scores",
    "no_cache_concat_fallback",
)
_QUALIFIED_ENGINE_SECTIONS = (
    "prefill_engine_plan",
    "engine_plan",
)
_QUALIFIED_ENGINE_SECTION_FIELDS = (
    "engine_sha256",
    "num_optimization_profiles",
    "inputs",
    "outputs",
    "native_contiguous_attention_layer_indices",
    "dense_attention_layers",
    "cache_concat_layers",
    "inspector_path",
    "inspector_size_bytes",
    "inspector_sha256",
)


class IsolationError(RuntimeError):
    """The requested isolation evidence cannot be produced."""


def _load_boundary_module() -> Any:
    path = Path(__file__).with_name("qualify_native_dynamic_memory.py")
    spec = importlib.util.spec_from_file_location(
        "_trtmc_process_isolation_boundary", path
    )
    if spec is None or spec.loader is None:
        raise IsolationError(f"cannot load source-state helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_performance_module() -> Any:
    path = Path(__file__).with_name("qualify_native_dynamic_memory_perf.py")
    spec = importlib.util.spec_from_file_location(
        "_trtmc_process_isolation_performance", path
    )
    if spec is None or spec.loader is None:
        raise IsolationError(
            f"cannot load performance qualification helper: {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


boundary = _load_boundary_module()
performance = _load_performance_module()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(
    path: Path, *, require_nonempty: bool = True
) -> dict[str, Any]:
    try:
        canonical = path.expanduser().resolve(strict=True)
        fd = os.open(canonical, os.O_RDONLY | os.O_CLOEXEC)
    except OSError as exc:
        raise IsolationError(
            f"required file does not exist: {path}: {exc}"
        ) from exc
    try:
        before = os.fstat(fd)
        digest = hashlib.sha256()
        offset = 0
        while True:
            chunk = os.pread(fd, 8 * 1024 * 1024, offset)
            if not chunk:
                break
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(fd)
        endpoint = canonical.stat()
    except OSError as exc:
        raise IsolationError(
            f"required file cannot be identified: {canonical}: {exc}"
        ) from exc
    finally:
        os.close(fd)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(before, field) != getattr(after, field)
        or getattr(before, field) != getattr(endpoint, field)
        for field in stable
    ):
        raise IsolationError(
            f"required file changed while being identified: {canonical}"
        )
    if not stat.S_ISREG(before.st_mode):
        raise IsolationError(f"required file is not regular: {canonical}")
    if require_nonempty and before.st_size <= 0:
        raise IsolationError(f"required file is empty: {canonical}")
    return {
        "path": str(canonical),
        "device": before.st_dev,
        "inode": before.st_ino,
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "sha256": digest.hexdigest(),
    }


def _file_identity_matches(value: Mapping[str, Any]) -> bool:
    try:
        current = _file_identity(
            Path(str(value.get("path", ""))),
            require_nonempty=False,
        )
    except (IsolationError, OSError):
        return False
    return current == value


def _canonical_capture_tool_binding(
    *,
    repo_root: Path,
    requested_path: Path,
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the only allowed capture producer to one clean tracked HEAD blob."""

    repo_root = repo_root.expanduser().resolve()
    canonical = (
        repo_root / "tools" / "capture_native_dynamic_memory_perf.py"
    ).resolve()
    requested = requested_path.expanduser().resolve()
    if requested != canonical:
        raise IsolationError(
            "--capture-tool must be the canonical current-source producer: "
            f"{canonical}"
        )
    if (
        source_state.get("git_dirty") is not False
        or source_state.get("exact_head_gate_satisfied") is not True
    ):
        raise IsolationError(
            "canonical capture tool requires a clean exact source HEAD"
        )
    head = _nonempty_string(
        source_state.get("git_head"),
        "source_state.git_head",
    )
    if re.fullmatch(r"[0-9a-f]{40,64}", head) is None:
        raise IsolationError(
            "source_state.git_head must be a full lowercase Git object ID"
        )
    try:
        relative = canonical.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise IsolationError(
            "canonical capture tool resolves outside the qualification "
            "repository"
        ) from exc
    git = ["git", "-c", f"safe.directory={repo_root}"]
    tracked = subprocess.run(
        [*git, "ls-files", "--error-unmatch", "--", relative],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if tracked.returncode != 0 or tracked.stdout.strip() != relative:
        raise IsolationError(
            "canonical capture tool is not tracked by the qualification HEAD"
        )
    blob = subprocess.run(
        [*git, "show", f"{head}:{relative}"],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if blob.returncode != 0:
        raise IsolationError(
            "canonical capture tool cannot be read from the qualification HEAD"
        )
    identity = _file_identity(canonical)
    blob_sha256 = hashlib.sha256(blob.stdout).hexdigest()
    if identity["sha256"] != blob_sha256:
        raise IsolationError(
            "canonical capture tool bytes differ from the qualification HEAD"
        )
    return {
        "identity": identity,
        "repo_relative_path": relative,
        "git_head": head,
        "source_state_sha256": source_state["source_state_sha256"],
        "head_blob_sha256": blob_sha256,
    }


def _read_object(path: Path, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IsolationError(f"{where}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise IsolationError(f"{where} must be a JSON object")
    return value


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IsolationError(f"{where} must be a JSON object")
    return value


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IsolationError(f"{where} must be a non-empty string")
    return value


def _sha_field(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IsolationError(f"{where} must be a lowercase SHA-256")
    return value


def _referenced_path(
    value: Any, *, report_path: Path, where: str
) -> Path:
    raw = _nonempty_string(value, where)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = report_path.parent / path
    return path.resolve()


def _validate_source_snapshot(
    value: Any,
    *,
    expected: Mapping[str, Any],
    where: str,
) -> Mapping[str, Any]:
    snapshot = _object(value, where)
    if (
        snapshot.get("git_head") != expected.get("git_head")
        or snapshot.get("source_state_sha256")
        != expected.get("source_state_sha256")
    ):
        raise IsolationError(f"{where} does not match aggregate source state")
    if (
        snapshot.get("git_dirty") is not False
        or snapshot.get("exact_head_gate_satisfied") is not True
        or snapshot.get("status") != []
    ):
        raise IsolationError(f"{where} is not a clean exact-HEAD receipt")
    return snapshot


def _validate_qualified_engine_graph(
    value: Any,
    *,
    report_path: Path,
    spec: Any,
    bundle_path: Path,
    bundle_header: Mapping[str, Any],
    live_runtime_stack: Mapping[str, Any],
) -> dict[str, Any]:
    graph = _object(value, "correctness report.qualified_engine_graph")
    if graph.get("passed") is not True:
        raise IsolationError("correctness qualified engine graph is not passed")

    gates = _object(
        graph.get("gates"),
        "correctness report.qualified_engine_graph.gates",
    )
    if set(gates) != set(_QUALIFIED_ENGINE_GRAPH_GATES) or any(
        gates.get(name) is not True for name in _QUALIFIED_ENGINE_GRAPH_GATES
    ):
        raise IsolationError(
            "correctness qualified engine graph gates must contain exactly the "
            "required true gates"
        )

    graph_runtime_stack = _object(
        graph.get("runtime_stack"),
        "correctness report.qualified_engine_graph.runtime_stack",
    )
    normalized_graph_runtime_stack = {
        "schema": 1,
        **dict(graph_runtime_stack),
    }
    if (
        "schema" in graph_runtime_stack
        or normalized_graph_runtime_stack != dict(live_runtime_stack)
    ):
        raise IsolationError(
            "correctness qualified engine graph runtime stack does not match "
            "the correctness case runtime stack"
        )

    contract = _object(
        bundle_header.get("runtime_memory"),
        "aggregate bundle header.runtime_memory",
    )
    if dict(graph_runtime_stack) != contract.get("qualified_runtime_stack"):
        raise IsolationError(
            "correctness qualified engine graph runtime stack does not match "
            "the aggregate bundle contract"
        )
    model_contract = _object(
        graph.get("model_contract"),
        "correctness report.qualified_engine_graph.model_contract",
    )
    expected_model_contract_fields = {
        "model_context_limit",
        "prefill_chunk_limit",
        "active_kv_profile_limits",
        "num_layers",
        "vocab_size",
        "kv_dtype",
        "kv_bytes_per_token",
        "kv_width",
    }
    if set(model_contract) != expected_model_contract_fields:
        raise IsolationError(
            "correctness qualified engine graph model_contract must contain "
            "exactly the required model fields"
        )
    num_layers = bundle_header.get("num_layers")
    vocab_size = bundle_header.get("vocab_size")
    kv_dtype = contract.get("kv_dtype")
    dtype_bytes = {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
    }.get(kv_dtype)
    kv_bytes_per_token = contract.get("kv_bytes_per_token")
    if (
        isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
        or num_layers <= 0
        or isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 1
        or dtype_bytes is None
        or isinstance(kv_bytes_per_token, bool)
        or not isinstance(kv_bytes_per_token, int)
        or kv_bytes_per_token <= 0
    ):
        raise IsolationError(
            "aggregate bundle header has incomplete model/KV accounting"
        )
    kv_width_denominator = 2 * num_layers * dtype_bytes
    if kv_bytes_per_token % kv_width_denominator:
        raise IsolationError(
            "aggregate bundle kv_bytes_per_token is not divisible by "
            "2 * num_layers * dtype_bytes"
        )
    kv_width = kv_bytes_per_token // kv_width_denominator
    if kv_width <= 0:
        raise IsolationError("aggregate bundle derived KV width is not positive")
    expected_model_contract = {
        "model_context_limit": contract.get("model_context_limit"),
        "prefill_chunk_limit": contract.get("prefill_chunk_limit"),
        "active_kv_profile_limits": contract.get(
            "active_kv_profile_limits"
        ),
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "kv_dtype": kv_dtype,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_width": kv_width,
    }
    if dict(model_contract) != expected_model_contract:
        raise IsolationError(
            "correctness qualified engine graph model_contract does not "
            "match the aggregate bundle header"
        )

    sections = _object(
        graph.get("engine_sections"),
        "correctness report.qualified_engine_graph.engine_sections",
    )
    if set(sections) != set(_QUALIFIED_ENGINE_SECTIONS):
        raise IsolationError(
            "correctness qualified engine graph must contain exactly "
            "prefill_engine_plan and engine_plan sections"
        )

    try:
        boundary._validate_qualified_engine_graph_evidence(
            graph,
            spec,
            num_layers=num_layers,
            expected_runtime_stack=graph_runtime_stack,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise IsolationError(
            f"correctness qualified engine graph semantic validation failed: {exc}"
        ) from exc

    validated_sections: dict[str, Any] = {}
    inspector_artifacts: list[dict[str, Any]] = []
    inspector_paths: set[str] = set()
    engine_shas: set[str] = set()
    engine_plan_identities: list[dict[str, Any]] = []
    for section_name in _QUALIFIED_ENGINE_SECTIONS:
        section = _object(
            sections[section_name],
            (
                "correctness report.qualified_engine_graph.engine_sections."
                f"{section_name}"
            ),
        )
        missing = [
            field
            for field in _QUALIFIED_ENGINE_SECTION_FIELDS
            if field not in section
        ]
        if missing:
            raise IsolationError(
                f"correctness qualified engine graph {section_name} is "
                f"missing fields: {missing}"
            )
        engine_sha = _sha_field(
            section["engine_sha256"],
            f"correctness qualified engine graph {section_name}.engine_sha256",
        )
        try:
            engine_plan = boundary._read_bundle_section(
                bundle_path,
                bundle_header,
                section_name,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise IsolationError(
                f"cannot read aggregate bundle {section_name}: {exc}"
            ) from exc
        recomputed_engine_sha = hashlib.sha256(engine_plan).hexdigest()
        if engine_sha != recomputed_engine_sha:
            raise IsolationError(
                f"correctness qualified engine graph {section_name} engine "
                "SHA does not match the aggregate bundle section"
            )
        engine_plan_identities.append(
            {
                "section_name": section_name,
                "size_bytes": len(engine_plan),
                "sha256": recomputed_engine_sha,
            }
        )
        engine_shas.add(engine_sha)
        inspector_path = _referenced_path(
            section["inspector_path"],
            report_path=report_path,
            where=(
                "correctness qualified engine graph "
                f"{section_name}.inspector_path"
            ),
        )
        inspector_identity = _file_identity(inspector_path)
        inspector_sha = _sha_field(
            section["inspector_sha256"],
            (
                "correctness qualified engine graph "
                f"{section_name}.inspector_sha256"
            ),
        )
        if (
            section["inspector_size_bytes"] != inspector_identity["size_bytes"]
            or inspector_sha != inspector_identity["sha256"]
        ):
            raise IsolationError(
                f"correctness qualified engine graph {section_name} inspector "
                "artifact identity mismatch"
            )
        try:
            inspector_text = inspector_path.read_text(encoding="utf-8")
            inspector_json = json.loads(inspector_text)
        except (OSError, json.JSONDecodeError) as exc:
            raise IsolationError(
                f"correctness qualified engine graph {section_name} inspector "
                f"artifact is not valid JSON: {exc}"
            ) from exc
        if not isinstance(inspector_json, (dict, list)) or not inspector_json:
            raise IsolationError(
                f"correctness qualified engine graph {section_name} inspector "
                "artifact contains no layer evidence"
            )
        inspector_layer_indices = sorted(
            {
                int(index)
                for index in re.findall(
                    r"layer\.(\d+)\.attn\."
                    r"NativeContiguousAttentionV2",
                    inspector_text,
                )
            }
        )
        inspector_dense_layers = boundary._dense_attention_layers(
            inspector_json
        )
        inspector_cache_concat_layers = boundary._cache_concat_layers(
            inspector_json
        )
        if (
            inspector_layer_indices
            != section[
                "native_contiguous_attention_layer_indices"
            ]
            or inspector_dense_layers != section["dense_attention_layers"]
            or inspector_cache_concat_layers
            != section["cache_concat_layers"]
        ):
            raise IsolationError(
                f"correctness qualified engine graph {section_name} "
                "reported layer evidence does not match its inspector artifact"
            )
        if inspector_identity["path"] in inspector_paths:
            raise IsolationError(
                "correctness qualified engine graph engine sections reuse one "
                "inspector artifact path"
            )
        inspector_paths.add(inspector_identity["path"])
        inspector_artifacts.append(inspector_identity)
        validated_sections[section_name] = {
            **dict(section),
            "inspector_artifact": inspector_identity,
        }

    if len(engine_shas) != len(_QUALIFIED_ENGINE_SECTIONS):
        raise IsolationError(
            "correctness qualified engine graph prefill and decode engine SHA "
            "identities must be different"
        )

    return {
        "passed": True,
        "sha256": _canonical_sha(graph),
        "gates": dict(gates),
        "runtime_stack": dict(live_runtime_stack),
        "runtime_stack_sha256": _canonical_sha(live_runtime_stack),
        "qualified_engine_runtime_stack": dict(graph_runtime_stack),
        "qualified_engine_runtime_stack_sha256": _canonical_sha(
            graph_runtime_stack
        ),
        "model_contract": dict(model_contract),
        "num_layers": num_layers,
        "engine_sections": validated_sections,
        "engine_plan_identities": engine_plan_identities,
        "inspector_artifacts": inspector_artifacts,
    }


def _parse_runtime_stack_log(path: Path, *, where: str) -> dict[str, Any]:
    identity = _file_identity(path)
    text = Path(identity["path"]).read_text(
        encoding="utf-8", errors="replace"
    )
    prefix = "[trtmc.runtime_stack]"
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    if not lines:
        raise IsolationError(f"{where} has no live runtime-stack evidence")

    expected_fields = set(_RUNTIME_STACK_FIELDS)
    unique: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(lines, start=1):
        raw: dict[str, str] = {}
        for token in line[len(prefix) :].strip().split():
            name, separator, field_value = token.partition("=")
            if (
                not separator
                or not name
                or not field_value
                or name in raw
            ):
                raise IsolationError(
                    f"{where} runtime-stack line {line_number} has invalid "
                    f"token {token!r}"
                )
            raw[name] = field_value
        missing = sorted(expected_fields - raw.keys())
        extra = sorted(raw.keys() - expected_fields)
        if missing or extra:
            raise IsolationError(
                f"{where} runtime-stack line {line_number} has "
                f"missing={missing!r}, extra={extra!r}"
            )
        if raw["schema"] != "1":
            raise IsolationError(f"{where} runtime-stack schema is not 1")
        if re.fullmatch(r"sm[0-9]+", raw["sm"]) is None:
            raise IsolationError(f"{where} runtime-stack SM is malformed")
        if re.fullmatch(
            r"[0-9a-f]{40}", raw["cudnn_frontend_revision"]
        ) is None:
            raise IsolationError(
                f"{where} cuDNN Frontend revision is not a full Git SHA"
            )
        for field in (
            "tensorrt",
            "cuda_runtime",
            "cudnn_backend",
            "nvrtc",
        ):
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", raw[field]) is None:
                raise IsolationError(
                    f"{where} runtime-stack {field} is malformed"
                )
        if not raw["driver"] or raw["driver"] == "unavailable":
            raise IsolationError(
                f"{where} runtime-stack driver is unavailable"
            )
        parsed: dict[str, Any] = {**raw, "schema": 1}
        unique[_canonical_sha(parsed)] = parsed
    if len(unique) != 1:
        raise IsolationError(f"{where} has conflicting runtime-stack evidence")
    return next(iter(unique.values()))


def _validate_correctness_report(
    path: Path,
    *,
    bundle_identity: Mapping[str, Any],
    source_state: Mapping[str, Any],
) -> dict[str, Any]:
    report = _read_object(path, "correctness report")
    if report.get("schema_version") != 1:
        raise IsolationError("correctness report schema must be 1")
    if report.get("passed") is not True:
        raise IsolationError("correctness report is not passed")
    model_id = _nonempty_string(
        report.get("model_id"), "correctness report.model_id"
    )
    try:
        spec = boundary.SPECS[model_id]
    except KeyError as exc:
        raise IsolationError(
            f"correctness report model is not qualified: {model_id!r}"
        ) from exc
    bundle_path = _referenced_path(
        report.get("bundle"),
        report_path=path,
        where="correctness report.bundle",
    )
    if (
        bundle_path != Path(str(bundle_identity["path"]))
        or report.get("bundle_sha256") != bundle_identity["sha256"]
        or _sha256(bundle_path) != bundle_identity["sha256"]
    ):
        raise IsolationError(
            "correctness report does not identify the aggregate bundle"
        )
    try:
        bundle_header = boundary._read_bundle_header(bundle_path)
    except (OSError, ValueError) as exc:
        raise IsolationError(
            f"cannot read aggregate bundle header: {exc}"
        ) from exc
    if (
        report.get("model_context_limit") != spec.context_limit
        or report.get("prefill_chunk_limit") != spec.chunk_limit
    ):
        raise IsolationError(
            "correctness report model limits do not match the qualified model"
        )

    _validate_source_snapshot(
        report.get("source_state"),
        expected=source_state,
        where="correctness report.source_state",
    )
    _validate_source_snapshot(
        report.get("source_state_post"),
        expected=source_state,
        where="correctness report.source_state_post",
    )
    if report.get("source_state_unchanged") is not True:
        raise IsolationError("correctness source_state_unchanged is not true")
    if report.get("promotion_eligible") is not True:
        raise IsolationError(
            "correctness report is not promotion eligible"
        )
    qualification_gates = _object(
        report.get("qualification_gates"),
        "correctness report.qualification_gates",
    )
    if (
        qualification_gates.get(
            "base_artifact_binding_passed"
        )
        is not True
    ):
        raise IsolationError(
            "correctness report did not persist the base artifact binding "
            "gate"
        )
    if (
        qualification_gates.get(
            "runtime_kv_plugin_binding_passed"
        )
        is not True
    ):
        raise IsolationError(
            "correctness report did not persist the runtime-KV plugin "
            "binding gate"
        )
    runner_path = _referenced_path(
        report.get("runner"),
        report_path=path,
        where="correctness report.runner",
    )
    base_artifact_binding = _object(
        report.get("base_artifact_binding"),
        "correctness report.base_artifact_binding",
    )
    if not boundary._base_artifact_binding_passed(
        base_artifact_binding,
        bundle=bundle_path,
        runner=runner_path,
        spec=spec,
        source_state=source_state,
    ):
        raise IsolationError(
            "correctness base artifact binding did not replay"
        )
    runtime_kv_plugin_binding = _object(
        report.get("runtime_kv_plugin_binding"),
        "correctness report.runtime_kv_plugin_binding",
    )
    if not boundary._persisted_runtime_kv_plugin_binding_passed(
        runtime_kv_plugin_binding,
        base_artifact_binding=base_artifact_binding,
    ):
        raise IsolationError(
            "correctness runtime-KV plugin binding did not replay"
        )
    try:
        base_artifact_files = [
            _file_identity(
                Path(
                    str(
                        base_artifact_binding["build_manifest"][
                            "path"
                        ]
                    )
                )
            ),
            _file_identity(
                Path(
                    str(
                        base_artifact_binding[
                            "base_build_receipt"
                        ]["path"]
                    )
                )
            ),
            _file_identity(runner_path),
            _file_identity(
                Path(
                    str(
                        base_artifact_binding["benchmark_worker"][
                            "path"
                        ]
                    )
                )
            ),
            _file_identity(
                Path(
                    str(base_artifact_binding["core"]["path"])
                )
            ),
            _file_identity(
                Path(
                    str(
                        base_artifact_binding["trt_backend"][
                            "active_versioned_path"
                        ]
                    )
                )
            ),
            _file_identity(
                Path(
                    str(
                        base_artifact_binding["model_plugin"][
                            "identity"
                        ]["path"]
                    )
                )
            ),
            _file_identity(
                Path(
                    str(
                        base_artifact_binding[
                            "runtime_kv_plugin"
                        ]["path"]
                    )
                )
            ),
        ]
    except (KeyError, TypeError) as exc:
        raise IsolationError(
            "correctness base artifact binding has incomplete file "
            f"identities: {exc}"
        ) from exc

    hf_reference = _object(
        report.get("hf_reference"), "correctness report.hf_reference"
    )
    if hf_reference.get("model_id") != model_id:
        raise IsolationError("correctness HF reference model does not match")
    model_revision = _nonempty_string(
        hf_reference.get("revision"),
        "correctness report.hf_reference.revision",
    )
    _sha_field(
        hf_reference.get("config_sha256"),
        "correctness report.hf_reference.config_sha256",
    )

    expected_cases = {
        case.name: case for case in boundary._cases_for(spec)
    }
    raw_cases = report.get("cases")
    if not isinstance(raw_cases, list):
        raise IsolationError("correctness report.cases must be an array")
    by_name: dict[str, Mapping[str, Any]] = {}
    for index, raw_case in enumerate(raw_cases):
        case = _object(raw_case, f"correctness report.cases[{index}]")
        name = _nonempty_string(
            case.get("name"), f"correctness report.cases[{index}].name"
        )
        if name in by_name:
            raise IsolationError(f"correctness report repeats case {name!r}")
        by_name[name] = case
    if set(by_name) != set(expected_cases):
        raise IsolationError(
            "correctness report does not contain the complete canonical "
            "case matrix"
        )

    stacks: dict[str, dict[str, Any]] = {}
    log_identities: list[dict[str, Any]] = []
    logit_identities: list[dict[str, Any]] = []
    parity_case_count = 0
    for name, expected_case in expected_cases.items():
        case = by_name[name]
        if (
            case.get("prompt_tokens") != expected_case.prompt_tokens
            or case.get("decode_tokens") != expected_case.decode_tokens
            or case.get("passed") is not True
        ):
            raise IsolationError(
                f"correctness case {name!r} does not match the canonical case"
            )
        trace = _object(
            case.get("trace"), f"correctness case {name!r}.trace"
        )
        if expected_case.expect_admission_rejection:
            if (
                case.get("admission_rejected_before_attention") is not True
                or trace.get("status") != "rejected"
                or trace.get("stage") != "before_attention"
                or trace.get("prefill_launches") != 0
                or trace.get("decode_launches") != 0
            ):
                raise IsolationError(
                    f"correctness admission case {name!r} is incomplete"
                )
        else:
            if trace.get("status") != "ok":
                raise IsolationError(
                    f"correctness case {name!r} did not complete"
                )
            parity = _object(
                case.get("parity"), f"correctness case {name!r}.parity"
            )
            composite = _object(
                parity.get("composite_gates"),
                f"correctness case {name!r}.parity.composite_gates",
            )
            if (
                parity.get("passed") is not True
                or composite.get("numerical") is not True
                or composite.get("token_level") is not True
            ):
                raise IsolationError(
                    f"correctness case {name!r} did not pass HF parity"
                )
            parity_case_count += 1
            for artifact_name in ("trt_logits", "hf_logits"):
                artifact_path = _referenced_path(
                    case.get(f"{artifact_name}_artifact"),
                    report_path=path,
                    where=(
                        f"correctness case {name!r}."
                        f"{artifact_name}_artifact"
                    ),
                )
                identity = _file_identity(artifact_path)
                if case.get(f"{artifact_name}_sha256") != identity["sha256"]:
                    raise IsolationError(
                        f"correctness case {name!r} {artifact_name} "
                        "artifact hash mismatch"
                    )
                logit_identities.append(identity)

        stderr_path = _referenced_path(
            case.get("runner_stderr"),
            report_path=path,
            where=f"correctness case {name!r}.runner_stderr",
        )
        stack = _parse_runtime_stack_log(
            stderr_path, where=f"correctness case {name!r}"
        )
        stack_sha = _canonical_sha(stack)
        stacks[stack_sha] = stack
        log_identities.append(_file_identity(stderr_path))
    if len(stacks) != 1:
        raise IsolationError(
            "correctness matrix used more than one live runtime stack"
        )
    runtime_stack = next(iter(stacks.values()))
    qualified_engine_graph = _validate_qualified_engine_graph(
        report.get("qualified_engine_graph"),
        report_path=path,
        spec=spec,
        bundle_path=bundle_path,
        bundle_header=bundle_header,
        live_runtime_stack=runtime_stack,
    )

    memory_envelope = _object(
        report.get("context_memory_envelope"),
        "correctness report.context_memory_envelope",
    )
    if (
        memory_envelope.get("status") != "passed"
        or memory_envelope.get("passed") is not True
        or memory_envelope.get("coverage_required") is not True
        or not _all_true(memory_envelope.get("gates"))
    ):
        raise IsolationError(
            "correctness report did not pass the full context-memory envelope"
        )
    return {
        "report": _file_identity(path),
        "model_id": model_id,
        "model_revision": model_revision,
        "bundle_sha256": bundle_identity["sha256"],
        "source_state_sha256": source_state["source_state_sha256"],
        "git_head": source_state["git_head"],
        "canonical_case_count": len(expected_cases),
        "hf_parity_case_count": parity_case_count,
        "runtime_stack": runtime_stack,
        "runtime_stack_sha256": _canonical_sha(runtime_stack),
        "runner_stderr_logs": log_identities,
        "logit_artifacts": logit_identities,
        "base_artifact_binding": dict(base_artifact_binding),
        "runtime_kv_plugin_binding": dict(
            runtime_kv_plugin_binding
        ),
        "base_artifact_files": base_artifact_files,
        "qualified_engine_graph": qualified_engine_graph,
        "context_memory_envelope": {
            "status": memory_envelope["status"],
            "passed": memory_envelope["passed"],
            "coverage_required": memory_envelope["coverage_required"],
        },
    }


def _runtime_trtmc_from_correctness(
    correctness: Mapping[str, Any],
) -> dict[str, Any]:
    base = _object(
        correctness.get("base_artifact_binding"),
        "correctness base artifact binding",
    )
    model_id = _nonempty_string(
        correctness.get("model_id"),
        "correctness model_id",
    )
    family = {
        "Qwen/Qwen3-0.6B": "qwen",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "llama",
    }.get(model_id)
    if family is None:
        raise IsolationError(
            f"correctness model has no runtime DSO family: {model_id!r}"
        )
    try:
        return {
            "model_id": model_id,
            "model_family": family,
            "core": dict(base["core"]),
            "trt_backend": dict(base["trt_backend"]["identity"]),
            "runtime_kv_plugin": dict(base["runtime_kv_plugin"]),
            "model": dict(base["model_plugin"]["identity"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise IsolationError(
            "correctness base artifact binding has no complete runtime DSO "
            f"identity set: {exc}"
        ) from exc


def _validate_aggregate_base_alignment(
    *,
    inputs: Mapping[str, Any],
    correctness: Mapping[str, Any],
) -> dict[str, Any]:
    """Require aggregate argv files to be the correctness manifest inodes."""

    base = _object(
        correctness.get("base_artifact_binding"),
        "correctness base artifact binding",
    )
    expected = {
        "bundle": base.get("bundle"),
        "build_receipt": base.get("base_build_receipt"),
        "worker": base.get("benchmark_worker"),
        "plugin_library": base.get("runtime_kv_plugin"),
    }
    for name, identity in expected.items():
        if inputs.get(name) != identity:
            raise IsolationError(
                f"aggregate {name} exact identity differs from correctness "
                "base artifacts"
            )
    return _runtime_trtmc_from_correctness(correctness)


def _replay_dynamic_capture_provenance(
    *,
    result_path: Path,
    label: str,
    expected_bundle: Path,
    expected_source_state: Mapping[str, Any],
    expected_model_id: str,
    expected_build_receipt: Mapping[str, Any],
    expected_worker: Mapping[str, Any],
    expected_plugin: Mapping[str, Any],
    expected_runtime_trtmc: Mapping[str, Any],
    expected_build_manifest: Mapping[str, Any],
    expected_capture_tool: Mapping[str, Any],
    expected_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay one dynamic capture's complete source/binary provenance."""

    result = _read_object(result_path, label)
    if (
        result.get("schema_version") != CAPTURE_RESULT_SCHEMA
        or result.get("status") != "completed"
        or result.get("model_id") != expected_model_id
    ):
        raise IsolationError(f"{label}: capture result identity is invalid")
    provenance = _object(
        result.get("qualification_provenance"),
        f"{label}.qualification_provenance",
    )
    try:
        runtime_trtmc = performance._validate_runtime_trtmc_libraries(
            result.get("runtime_trtmc_libraries"),
            where=f"{label}.runtime_trtmc_libraries",
            model_id=expected_model_id,
        )
        mapped_dso_identities = (
            performance._validate_mapped_dso_identities(
                result.get("mapped_dso_identities"),
                where=f"{label}.mapped_dso_identities",
            )
        )
        build_plugin = performance._validate_build_runtime_kv_plugin(
            result.get("build_runtime_kv_plugin"),
            where=f"{label}.build_runtime_kv_plugin",
            artifact_role="native-dynamic",
        )
        validated_evidence, _ = (
            performance._validate_qualification_evidence(
                result,
                label=label,
                expected_role="native-dynamic",
                expected_bundle=expected_bundle,
                model_id=expected_model_id,
                provenance=provenance,
                runtime_trtmc_libraries=runtime_trtmc,
                mapped_dso_identities=mapped_dso_identities,
                build_runtime_kv_plugin=build_plugin,
            )
        )
    except (performance.QualificationError, OSError, ValueError) as exc:
        raise IsolationError(
            f"{label}: dynamic capture provenance replay failed: {exc}"
        ) from exc

    if (
        provenance.get("runtime_trtmc_libraries_sha256")
        != _canonical_sha(runtime_trtmc)
        or provenance.get("mapped_dso_identities_sha256")
        != _canonical_sha(mapped_dso_identities)
        or provenance.get("build_runtime_kv_plugin_sha256")
        != _canonical_sha(build_plugin)
    ):
        raise IsolationError(
            f"{label}: runtime TRTMC/mapped-DSO/build-plugin provenance hash mismatch"
        )
    _validate_source_snapshot(
        validated_evidence.get("source_state_pre"),
        expected=expected_source_state,
        where=f"{label}.qualification_evidence.source_state_pre",
    )
    _validate_source_snapshot(
        validated_evidence.get("source_state_post"),
        expected=expected_source_state,
        where=f"{label}.qualification_evidence.source_state_post",
    )
    if validated_evidence.get("source_state_unchanged") is not True:
        raise IsolationError(
            f"{label}: qualification evidence source state changed"
        )

    receipt_path = Path(
        str(validated_evidence.get("build_receipt", ""))
    )
    receipt_identity = _file_identity(receipt_path)
    if receipt_identity != expected_build_receipt:
        raise IsolationError(
            f"{label}: build receipt exact identity differs from aggregate"
        )
    toolchain = _object(
        validated_evidence.get("toolchain"),
        f"{label}.qualification_evidence.toolchain",
    )
    if toolchain.get("worker") != expected_worker:
        raise IsolationError(
            f"{label}: worker exact identity differs from aggregate"
        )
    if toolchain.get("plugin_library") != expected_plugin:
        raise IsolationError(
            f"{label}: plugin exact identity differs from aggregate"
        )
    capture_identity = _object(
        expected_capture_tool.get("identity"),
        "canonical capture tool identity",
    )
    if (
        toolchain.get("capture_tool") != capture_identity.get("path")
        or toolchain.get("capture_tool_sha256")
        != capture_identity.get("sha256")
        or _file_identity(Path(str(capture_identity["path"])))
        != capture_identity
    ):
        raise IsolationError(
            f"{label}: capture tool is not the canonical source-bound inode"
        )
    if toolchain.get("runtime_trtmc_libraries") != expected_runtime_trtmc:
        raise IsolationError(
            f"{label}: toolchain runtime TRTMC DSOs differ from correctness"
        )
    if runtime_trtmc != expected_runtime_trtmc:
        raise IsolationError(
            f"{label}: runtime TRTMC DSOs differ from correctness"
        )
    if build_plugin != expected_plugin:
        raise IsolationError(
            f"{label}: build runtime-KV plugin differs from aggregate"
        )
    if (
        validated_evidence.get("build_runtime_kv_plugin")
        != expected_plugin
        or validated_evidence.get("runtime_trtmc_libraries")
        != expected_runtime_trtmc
        or validated_evidence.get("build_manifest")
        != expected_build_manifest
        or toolchain.get("build_manifest") != expected_build_manifest
    ):
        raise IsolationError(
            f"{label}: build/runtime provenance differs from correctness"
        )

    request_identity = _file_identity(
        Path(str(validated_evidence.get("request_file", "")))
    )
    if (
        expected_request is not None
        and request_identity != expected_request
    ):
        raise IsolationError(
            f"{label}: request exact identity differs from aggregate"
        )
    evidence_files = [
        receipt_identity,
        request_identity,
        _file_identity(
            Path(str(validated_evidence.get("worker_stdout", ""))),
            require_nonempty=False,
        ),
        _file_identity(
            Path(str(validated_evidence.get("worker_stderr", ""))),
            require_nonempty=False,
        ),
        dict(capture_identity),
    ]
    return {
        "build_receipt": receipt_identity,
        "worker": dict(expected_worker),
        "plugin_library": dict(expected_plugin),
        "capture_tool": dict(expected_capture_tool),
        "build_manifest": dict(expected_build_manifest),
        "runtime_trtmc_libraries": dict(expected_runtime_trtmc),
        "build_runtime_kv_plugin": dict(expected_plugin),
        "request": request_identity,
        "evidence_files": evidence_files,
        "sha256": _canonical_sha(
            {
                "build_receipt": receipt_identity,
                "worker": expected_worker,
                "plugin_library": expected_plugin,
                "capture_tool": expected_capture_tool,
                "build_manifest": expected_build_manifest,
                "runtime_trtmc_libraries": expected_runtime_trtmc,
                "build_runtime_kv_plugin": expected_plugin,
                "request": request_identity,
            }
        ),
    }


def _validate_performance_report(
    path: Path,
    *,
    bundle_identity: Mapping[str, Any],
    source_state: Mapping[str, Any],
    correctness: Mapping[str, Any],
    aggregate_inputs: Mapping[str, Any],
    capture_tool_binding: Mapping[str, Any],
    expected_runtime_trtmc: Mapping[str, Any],
) -> dict[str, Any]:
    report = _read_object(path, "performance report")
    if report.get("schema_version") != PERFORMANCE_REPORT_SCHEMA:
        raise IsolationError("performance report schema is unexpected")
    if report.get("status") != "passed":
        raise IsolationError("performance report is not passed")
    gates = _object(report.get("gates"), "performance report.gates")
    if not _all_true(gates):
        raise IsolationError("performance report contains a failed gate")

    bundles = _object(report.get("bundles"), "performance report.bundles")
    cases = _object(report.get("cases"), "performance report.cases")
    expected_cases = {
        "static_short",
        "dynamic_short",
        "static_medium",
        "dynamic_medium",
    }
    if set(cases) != expected_cases:
        raise IsolationError(
            "performance report must contain exactly short/medium static and "
            "dynamic cases"
        )

    case_paths = {
        name: _referenced_path(
            _object(
                cases[name], f"performance report.cases.{name}"
            ).get("path"),
            report_path=path,
            where=f"performance report.cases.{name}.path",
        )
        for name in expected_cases
    }
    bundle_paths = {
        role: _referenced_path(
            _object(
                bundles.get(role), f"performance report.bundles.{role}"
            ).get("path"),
            report_path=path,
            where=f"performance report.bundles.{role}.path",
        )
        for role in ("static", "dynamic")
    }
    try:
        regenerated = performance.qualify(
            static_short=case_paths["static_short"],
            dynamic_short=case_paths["dynamic_short"],
            static_medium=case_paths["static_medium"],
            dynamic_medium=case_paths["dynamic_medium"],
            static_bundle=bundle_paths["static"],
            dynamic_bundle=bundle_paths["dynamic"],
        )
    except (performance.QualificationError, OSError, ValueError) as exc:
        raise IsolationError(
            f"performance report cannot be regenerated: {exc}"
        ) from exc
    if regenerated != report:
        raise IsolationError(
            "performance report does not exactly match regenerated gates"
        )
    if (
        bundle_paths["dynamic"] != Path(str(bundle_identity["path"]))
        or report["bundles"]["dynamic"]["sha256"]
        != bundle_identity["sha256"]
    ):
        raise IsolationError(
            "performance report dynamic bundle does not match aggregate bundle"
        )

    dynamic_libraries: dict[str, Mapping[str, Any]] = {}
    capture_identities: dict[str, dict[str, Any]] = {}
    dynamic_capture_provenance: dict[str, dict[str, Any]] = {}
    for case_name in sorted(expected_cases):
        summary = _object(cases[case_name], f"performance case {case_name}")
        if summary.get("model_id") != correctness["model_id"]:
            raise IsolationError(
                f"performance case {case_name} model does not match correctness"
            )
        capture_identities[case_name] = _file_identity(
            case_paths[case_name]
        )
        provenance = _object(
            summary.get("qualification_provenance"),
            f"performance case {case_name}.qualification_provenance",
        )
        for source_field in (
            "source_state_sha256",
            "source_state_pre_sha256",
            "source_state_post_sha256",
            "prebuild_source_state_sha256",
            "postbuild_source_state_sha256",
        ):
            if provenance.get(source_field) != source_state[
                "source_state_sha256"
            ]:
                raise IsolationError(
                    f"performance case {case_name} {source_field} does not "
                    "match aggregate source"
                )
        if (
            provenance.get("git_head") != source_state["git_head"]
            or provenance.get("model_revision")
            != correctness["model_revision"]
        ):
            raise IsolationError(
                f"performance case {case_name} provenance tuple mismatch"
            )
        if case_name.startswith("dynamic_"):
            runtime_stack = _object(
                summary.get("runtime_stack"),
                f"performance case {case_name}.runtime_stack",
            )
            runtime_libraries = _object(
                summary.get("runtime_libraries"),
                f"performance case {case_name}.runtime_libraries",
            )
            if (
                runtime_stack != correctness["runtime_stack"]
                or provenance.get("runtime_stack_sha256")
                != correctness["runtime_stack_sha256"]
            ):
                raise IsolationError(
                    f"performance case {case_name} runtime tuple mismatch"
                )
            dynamic_libraries[_canonical_sha(runtime_libraries)] = (
                runtime_libraries
            )
            dynamic_capture_provenance[case_name] = (
                _replay_dynamic_capture_provenance(
                    result_path=case_paths[case_name],
                    label=f"performance {case_name}",
                    expected_bundle=Path(str(bundle_identity["path"])),
                    expected_source_state=source_state,
                    expected_model_id=correctness["model_id"],
                    expected_build_receipt=aggregate_inputs[
                        "build_receipt"
                    ],
                    expected_worker=aggregate_inputs["worker"],
                    expected_plugin=aggregate_inputs["plugin_library"],
                    expected_runtime_trtmc=expected_runtime_trtmc,
                    expected_build_manifest=correctness[
                        "base_artifact_binding"
                    ]["build_manifest"],
                    expected_capture_tool=capture_tool_binding,
                )
            )
    if len(dynamic_libraries) != 1:
        raise IsolationError(
            "performance dynamic cases used different runtime libraries"
        )

    runtime_libraries = next(iter(dynamic_libraries.values()))
    runtime_library_files = [
        _file_identity(Path(str(runtime_libraries[name]["path"])))
        for name in ("nvrtc", "nvrtc_builtins")
    ]
    performance_gates = _object(
        gates.get("performance"), "performance report.gates.performance"
    )
    measurements = {
        prompt_kind: {
            "decode_throughput_ratio": performance_gates[prompt_kind][
                "decode_throughput_ratio"
            ],
            "prefill_ratio": performance_gates[prompt_kind]["prefill_ratio"],
        }
        for prompt_kind in ("short", "medium")
    }
    return {
        "report": _file_identity(path),
        "model_id": correctness["model_id"],
        "model_revision": correctness["model_revision"],
        "bundle_sha256": bundle_identity["sha256"],
        "source_state_sha256": source_state["source_state_sha256"],
        "git_head": source_state["git_head"],
        "runtime_stack": correctness["runtime_stack"],
        "runtime_stack_sha256": correctness["runtime_stack_sha256"],
        "runtime_libraries": runtime_libraries,
        "runtime_libraries_sha256": _canonical_sha(runtime_libraries),
        "runtime_library_files": runtime_library_files,
        "captures": capture_identities,
        "dynamic_capture_provenance": dynamic_capture_provenance,
        "split_08_09": measurements,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _artifact_output_is_excluded(repo_root: Path, output_dir: Path) -> bool:
    try:
        relative = output_dir.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return True
    top_level = relative.parts[0] if relative.parts else ""
    return (
        top_level == "artifacts"
        or top_level == "build"
        or top_level.startswith("build-")
    )


def _source_state_snapshot(
    repo_root: Path, output_dir: Path, *, label: str
) -> dict[str, Any]:
    if not _artifact_output_is_excluded(repo_root, output_dir):
        raise IsolationError(
            "process-isolation output inside the repository must be under "
            "artifacts/, build/, or build-* so source snapshots exclude it"
        )
    return boundary.source_state_provenance(
        repo_root,
        Path(__file__),
        output_dir,
        label=label,
    )


def _gpu_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,name",
        "--format=csv,noheader,nounits",
    ]
    started_ns = time.time_ns()
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    finished_ns = time.time_ns()
    if completed.returncode != 0:
        raise IsolationError(
            "cannot enumerate physical GPUs with nvidia-smi: "
            f"{completed.stderr.strip()}"
        )
    rows: list[dict[str, str]] = []
    for row in csv.reader(completed.stdout.splitlines()):
        if not row:
            continue
        if len(row) != 4:
            raise IsolationError(f"malformed nvidia-smi GPU row: {row!r}")
        index, uuid, pci_bus_id, name = (item.strip() for item in row)
        if not index.isdigit() or not uuid or not pci_bus_id or not name:
            raise IsolationError(f"incomplete nvidia-smi GPU row: {row!r}")
        rows.append(
            {
                "index": index,
                "uuid": uuid,
                "pci_bus_id": pci_bus_id,
                "name": name,
            }
        )
    if not rows:
        raise IsolationError("nvidia-smi reported no physical GPUs")
    return {
        "command": command,
        "command_sha256": _canonical_sha(command),
        "started_ns": started_ns,
        "finished_ns": finished_ns,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stdout_sha256": hashlib.sha256(
            completed.stdout.encode("utf-8")
        ).hexdigest(),
        "stderr": completed.stderr,
        "stderr_sha256": hashlib.sha256(
            completed.stderr.encode("utf-8")
        ).hexdigest(),
        "gpus": rows,
    }


def _resolve_gpu(
    selector: str, inventory: Mapping[str, Any]
) -> dict[str, str]:
    selector = selector.strip()
    if not selector or "," in selector:
        raise IsolationError(
            "each GPU selector must identify exactly one physical GPU"
        )
    rows = inventory.get("gpus")
    if not isinstance(rows, list):
        raise IsolationError("GPU inventory has no gpus array")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and (
            str(row.get("index")) == selector
            or str(row.get("uuid")) == selector
            or str(row.get("pci_bus_id")) == selector
        )
    ]
    if len(matches) != 1:
        raise IsolationError(
            f"GPU selector {selector!r} resolved to {len(matches)} devices"
        )
    row = matches[0]
    resolved = {
        field: str(row.get(field, "")).strip()
        for field in ("index", "uuid", "pci_bus_id", "name")
    }
    if any(not value for value in resolved.values()):
        raise IsolationError(f"GPU selector {selector!r} has incomplete identity")
    resolved["requested_selector"] = selector
    # Always launch by UUID. Numeric CUDA ordinals can be reordered by
    # CUDA_DEVICE_ORDER, while a singleton UUID mask is an unambiguous physical
    # device identity.
    resolved["selector"] = resolved["uuid"]
    return resolved


def _execution_environment(gpu: str, cache_path: Path) -> dict[str, str]:
    environment = {
        name: os.environ[name]
        for name in _SAFE_ENVIRONMENT_NAMES
        if name in os.environ and name != "LD_PRELOAD"
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "CUDA_CACHE_PATH": str(cache_path.resolve()),
            "CUDA_CACHE_DISABLE": "0",
        }
    )
    if "LD_PRELOAD" in environment:
        raise IsolationError("LD_PRELOAD must not be set for isolation proof")
    return dict(sorted(environment.items()))


def _directory_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file():
            entries.append(
                {
                    "path": str(candidate.resolve()),
                    "relative_path": str(candidate.relative_to(path)),
                    "size_bytes": candidate.stat().st_size,
                    "sha256": _sha256(candidate),
                }
            )
    return entries


@dataclass
class _ActiveCapture:
    label: str
    argv: list[str]
    environment: dict[str, str]
    run_dir: Path
    result_path: Path
    worker_stderr_path: Path
    stdout_path: Path
    stderr_path: Path
    started_ns: int
    spawned_ns: int
    process: subprocess.Popen[Any]
    stdout_stream: TextIO
    stderr_stream: TextIO


_PINNED_CAPTURE_TRAMPOLINE = """
import os
import sys
import types

fd = int(sys.argv[1])
canonical = sys.argv[2]
chunks = []
offset = 0
while True:
    chunk = os.pread(fd, 1024 * 1024, offset)
    if not chunk:
        break
    chunks.append(chunk)
    offset += len(chunk)
sys.argv = [canonical, *sys.argv[3:]]
sys.path[0] = os.path.dirname(canonical)
module = types.ModuleType("__main__")
module.__file__ = canonical
module.__package__ = None
module.__cached__ = None
module.__spec__ = None
module.__loader__ = None
sys.modules["__main__"] = module
exec(compile(b"".join(chunks), canonical, "exec"), module.__dict__)
""".strip()


def _capture_argv(
    *,
    python: Path,
    capture_tool: Path,
    capture_tool_fd: int,
    repo_root: Path,
    bundle: Path,
    build_receipt: Path,
    request: Path,
    worker: Path,
    plugin_library: Path,
    comparison_sequence_limit: int,
    result_path: Path,
    worker_stderr_path: Path,
) -> list[str]:
    return [
        str(python),
        "-c",
        _PINNED_CAPTURE_TRAMPOLINE,
        str(capture_tool_fd),
        str(capture_tool),
        "--repo-root",
        str(repo_root),
        "benchmark",
        "--bundle",
        str(bundle),
        "--build-receipt",
        str(build_receipt),
        "--request",
        str(request),
        "--worker",
        str(worker),
        "--plugin-library",
        str(plugin_library),
        "--output",
        str(result_path),
        "--stderr-output",
        str(worker_stderr_path),
        "--comparison-sequence-limit",
        str(comparison_sequence_limit),
        "--cwd",
        str(repo_root),
        "--role",
        "native-dynamic",
    ]


def _start_capture(
    *,
    label: str,
    gpu: str,
    cache_path: Path,
    output_dir: Path,
    python: Path,
    capture_tool: Path,
    capture_tool_fd: int,
    repo_root: Path,
    bundle: Path,
    build_receipt: Path,
    request: Path,
    worker: Path,
    plugin_library: Path,
    comparison_sequence_limit: int,
) -> _ActiveCapture:
    run_dir = output_dir / "runs" / label
    run_dir.mkdir(parents=True, exist_ok=False)
    result_path = run_dir / "capture-result.json"
    worker_stderr_path = run_dir / "worker.stderr.log"
    stdout_path = run_dir / "capture.stdout.log"
    stderr_path = run_dir / "capture.stderr.log"
    argv = _capture_argv(
        python=python,
        capture_tool=capture_tool,
        capture_tool_fd=capture_tool_fd,
        repo_root=repo_root,
        bundle=bundle,
        build_receipt=build_receipt,
        request=request,
        worker=worker,
        plugin_library=plugin_library,
        comparison_sequence_limit=comparison_sequence_limit,
        result_path=result_path,
        worker_stderr_path=worker_stderr_path,
    )
    environment = _execution_environment(gpu, cache_path)
    stdout_stream = stdout_path.open("w", encoding="utf-8")
    stderr_stream = stderr_path.open("w", encoding="utf-8")
    started_ns = time.time_ns()
    try:
        process = subprocess.Popen(
            argv,
            cwd=repo_root,
            env=environment,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
            pass_fds=(capture_tool_fd,),
        )
    except BaseException:
        stdout_stream.close()
        stderr_stream.close()
        raise
    return _ActiveCapture(
        label=label,
        argv=argv,
        environment=environment,
        run_dir=run_dir,
        result_path=result_path,
        worker_stderr_path=worker_stderr_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        started_ns=started_ns,
        spawned_ns=time.time_ns(),
        process=process,
        stdout_stream=stdout_stream,
        stderr_stream=stderr_stream,
    )


def _finish_capture(active: _ActiveCapture) -> dict[str, Any]:
    try:
        returncode = active.process.wait()
        finished_ns = time.time_ns()
    finally:
        active.stdout_stream.close()
        active.stderr_stream.close()
    result_identity = (
        _file_identity(active.result_path, require_nonempty=False)
        if active.result_path.is_file()
        else None
    )
    manifest = _directory_manifest(active.run_dir)
    return {
        "label": active.label,
        "pid": active.process.pid,
        "argv": active.argv,
        "argv_sha256": _canonical_sha(active.argv),
        "environment_mode": "explicit_allowlist",
        "environment": active.environment,
        "environment_sha256": _canonical_sha(active.environment),
        "started_ns": active.started_ns,
        "spawned_ns": active.spawned_ns,
        "finished_ns": finished_ns,
        "duration_ns": finished_ns - active.started_ns,
        "returncode": returncode,
        "attempt_count": 1,
        "retry_count": 0,
        "locking": "none",
        "ld_preload_set": "LD_PRELOAD" in active.environment,
        "stdout": _file_identity(
            active.stdout_path, require_nonempty=False
        ),
        "stderr": _file_identity(
            active.stderr_path, require_nonempty=False
        ),
        "capture_result": result_identity,
        "artifact_manifest": manifest,
        "artifact_manifest_sha256": _canonical_sha(manifest),
    }


def _run_capture_matrix(
    *,
    gpu_a: Mapping[str, str],
    gpu_b: Mapping[str, str],
    cache_paths: Mapping[str, Path],
    output_dir: Path,
    python: Path,
    capture_tool: Path,
    capture_tool_binding: Mapping[str, Any],
    repo_root: Path,
    bundle: Path,
    build_receipt: Path,
    request: Path,
    worker: Path,
    plugin_library: Path,
    comparison_sequence_limit: int,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Execute all children through one pinned canonical capture-tool inode."""

    capture_module = boundary._load_perf_provenance_module()
    expected_identity = capture_tool_binding["identity"]
    executions: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    try:
        with capture_module._PinnedFile(
            capture_tool,
            label="canonical process-isolation capture tool",
        ) as pinned:
            if pinned.identity != expected_identity:
                raise IsolationError(
                    "canonical capture tool changed before it could be pinned"
                )

            def start(label: str, gpu: Mapping[str, str]) -> _ActiveCapture:
                return _start_capture(
                    label=label,
                    gpu=gpu["selector"],
                    cache_path=cache_paths[label],
                    output_dir=output_dir,
                    python=python,
                    capture_tool=capture_tool,
                    capture_tool_fd=pinned.fd,
                    repo_root=repo_root,
                    bundle=bundle,
                    build_receipt=build_receipt,
                    request=request,
                    worker=worker,
                    plugin_library=plugin_library,
                    comparison_sequence_limit=comparison_sequence_limit,
                )

            for label in ("gpu-a-cold", "gpu-a-warm"):
                executions[label] = _finish_capture(start(label, gpu_a))

            active_a = start("gpu-a-concurrent", gpu_a)
            try:
                active_b = start("gpu-b-concurrent", gpu_b)
            except BaseException:
                executions["gpu-a-concurrent"] = _finish_capture(active_a)
                raise
            executions["gpu-a-concurrent"] = _finish_capture(active_a)
            executions["gpu-b-concurrent"] = _finish_capture(active_b)
            if pinned.verify() != expected_identity:
                raise IsolationError(
                    "canonical capture tool changed during child execution"
                )
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"cannot execute capture benchmark: {exc}")
    except Exception as exc:
        if exc.__class__.__name__ != "CaptureError":
            raise
        raise IsolationError(
            f"canonical capture tool provenance failed: {exc}"
        ) from exc
    return executions, errors


def _overlap_ns(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> int:
    return max(
        0,
        min(first_end, second_end) - max(first_start, second_start),
    )


def _validate_cache(
    *,
    label: str,
    result: Mapping[str, Any],
    expected_path: Path,
    expected_state: str,
) -> dict[str, Any]:
    cache = result.get("cuda_jit_cache")
    if not isinstance(cache, dict):
        raise IsolationError(f"{label}: capture result has no cuda_jit_cache")
    if cache.get("path") != str(expected_path.resolve()):
        raise IsolationError(f"{label}: CUDA cache path mismatch")
    if cache.get("path_source") != "CUDA_CACHE_PATH":
        raise IsolationError(f"{label}: CUDA cache path was not explicit")
    if cache.get("cuda_cache_path_env") != str(expected_path.resolve()):
        raise IsolationError(f"{label}: CUDA_CACHE_PATH receipt mismatch")
    if cache.get("cuda_cache_disable_env") != "0":
        raise IsolationError(f"{label}: CUDA_CACHE_DISABLE must be 0")
    if cache.get("enabled") is not True:
        raise IsolationError(f"{label}: CUDA JIT cache is not enabled")
    if cache.get("initial_state") != expected_state:
        raise IsolationError(
            f"{label}: expected {expected_state} cache, "
            f"got {cache.get('initial_state')!r}"
        )
    before = cache.get("before")
    after = cache.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        raise IsolationError(f"{label}: CUDA cache snapshots are incomplete")
    before_files = before.get("file_count")
    after_files = after.get("file_count")
    if (
        isinstance(before_files, bool)
        or not isinstance(before_files, int)
        or before_files < 0
        or isinstance(after_files, bool)
        or not isinstance(after_files, int)
        or after_files < 0
    ):
        raise IsolationError(f"{label}: CUDA cache file counts are invalid")
    if expected_state == "cold" and before_files != 0:
        raise IsolationError(f"{label}: cold cache was not empty")
    if expected_state == "warm" and before_files <= 0:
        raise IsolationError(f"{label}: warm cache had no compiled files")
    if after_files <= 0:
        raise IsolationError(f"{label}: benchmark produced no JIT cache files")
    started_ns = cache.get("worker_started_ns")
    finished_ns = cache.get("worker_finished_ns")
    if (
        isinstance(started_ns, bool)
        or not isinstance(started_ns, int)
        or isinstance(finished_ns, bool)
        or not isinstance(finished_ns, int)
        or started_ns <= 0
        or finished_ns <= started_ns
    ):
        raise IsolationError(f"{label}: worker interval is invalid")
    provenance = result.get("qualification_provenance")
    if not isinstance(provenance, dict):
        raise IsolationError(f"{label}: qualification provenance is missing")
    if provenance.get("cuda_jit_cache_sha256") != _canonical_sha(cache):
        raise IsolationError(f"{label}: CUDA cache provenance hash mismatch")
    return cache


def _validate_child_result(
    *,
    label: str,
    execution: Mapping[str, Any],
    expected_gpu: Mapping[str, str],
    expected_cache: Path,
    expected_cache_state: str,
    expected_bundle: Path,
    expected_bundle_sha256: str,
    expected_source_state: Mapping[str, Any],
    expected_model_id: str,
    expected_model_revision: str,
    expected_runtime_stack: Mapping[str, Any],
    expected_runtime_libraries: Mapping[str, Any],
    expected_runtime_trtmc: Mapping[str, Any],
    expected_build_receipt: Mapping[str, Any],
    expected_worker: Mapping[str, Any],
    expected_plugin: Mapping[str, Any],
    expected_build_manifest: Mapping[str, Any],
    expected_capture_tool: Mapping[str, Any],
    expected_request: Mapping[str, Any],
    request_document: Mapping[str, Any],
) -> dict[str, Any]:
    if execution.get("returncode") != 0:
        raise IsolationError(
            f"{label}: capture benchmark returned {execution.get('returncode')}"
        )
    result_identity = execution.get("capture_result")
    if not isinstance(result_identity, Mapping):
        raise IsolationError(f"{label}: capture benchmark wrote no result")
    result_path = Path(str(result_identity.get("path", "")))
    if dict(result_identity) != _file_identity(
        result_path,
        require_nonempty=False,
    ):
        raise IsolationError(f"{label}: capture result exact identity mismatch")
    result = _read_object(result_path, f"{label} capture result")
    if result.get("schema_version") != CAPTURE_RESULT_SCHEMA:
        raise IsolationError(f"{label}: unexpected capture result schema")
    if result.get("status") != "completed":
        raise IsolationError(f"{label}: capture result is not completed")
    if result.get("model_id") != expected_model_id:
        raise IsolationError(f"{label}: child model identity mismatch")
    for field in ("case_name", "case_digest", "operation"):
        if result.get(field) != request_document.get(field):
            raise IsolationError(f"{label}: worker {field} does not match request")
    measurement = request_document.get("measurement")
    if not isinstance(measurement, Mapping):
        raise IsolationError("benchmark request measurement is missing")
    if result.get("warmup") != measurement.get("warmup"):
        raise IsolationError(f"{label}: worker warmup does not match request")
    if result.get("iterations") != measurement.get("iterations"):
        raise IsolationError(f"{label}: worker iterations do not match request")
    load_started_ns = result.get("load_started_ns")
    load_finished_ns = result.get("load_finished_ns")
    if (
        isinstance(load_started_ns, bool)
        or not isinstance(load_started_ns, int)
        or isinstance(load_finished_ns, bool)
        or not isinstance(load_finished_ns, int)
        or load_started_ns <= 0
        or load_finished_ns <= load_started_ns
    ):
        raise IsolationError(f"{label}: engine-load interval is invalid")

    provenance = result.get("qualification_provenance")
    evidence = result.get("qualification_evidence")
    if not isinstance(provenance, dict) or not isinstance(evidence, dict):
        raise IsolationError(f"{label}: qualification evidence is incomplete")
    expected_source_sha = expected_source_state.get("source_state_sha256")
    expected_head = expected_source_state.get("git_head")
    source_fields_match = (
        provenance.get("git_head") == expected_head
        and provenance.get("source_state_sha256") == expected_source_sha
        and provenance.get("source_state_pre_sha256") == expected_source_sha
        and provenance.get("source_state_post_sha256") == expected_source_sha
        and provenance.get("source_state_unchanged") is True
        and evidence.get("source_state_unchanged") is True
    )
    if not source_fields_match:
        raise IsolationError(f"{label}: child source_state_unchanged failed")
    if provenance.get("artifact_role") != "native-dynamic":
        raise IsolationError(f"{label}: child artifact role is not dynamic")
    if provenance.get("bundle_sha256") != expected_bundle_sha256:
        raise IsolationError(f"{label}: child bundle identity mismatch")
    if provenance.get("model_revision") != expected_model_revision:
        raise IsolationError(f"{label}: child model revision mismatch")
    provenance_binding = _replay_dynamic_capture_provenance(
        result_path=result_path,
        label=f"{label} child",
        expected_bundle=expected_bundle,
        expected_source_state=expected_source_state,
        expected_model_id=expected_model_id,
        expected_build_receipt=expected_build_receipt,
        expected_worker=expected_worker,
        expected_plugin=expected_plugin,
        expected_runtime_trtmc=expected_runtime_trtmc,
        expected_build_manifest=expected_build_manifest,
        expected_capture_tool=expected_capture_tool,
        expected_request=expected_request,
    )

    runtime_stack = result.get("runtime_stack")
    if not isinstance(runtime_stack, dict) or not runtime_stack:
        raise IsolationError(f"{label}: child has no live runtime stack")
    if provenance.get("runtime_stack_sha256") != _canonical_sha(runtime_stack):
        raise IsolationError(f"{label}: runtime-stack hash mismatch")
    if runtime_stack != expected_runtime_stack:
        raise IsolationError(
            f"{label}: child runtime stack differs from companion receipts"
        )
    output_summary = result.get("output_summary")
    if not isinstance(output_summary, dict):
        raise IsolationError(f"{label}: output_summary is missing")
    token_ids = output_summary.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_ids
        )
    ):
        raise IsolationError(f"{label}: output token IDs are invalid")
    observations = result.get("observations")
    if (
        not isinstance(observations, list)
        or len(observations) != result.get("iterations")
    ):
        raise IsolationError(f"{label}: measured observations are incomplete")
    for index, observation in enumerate(observations):
        if (
            not isinstance(observation, Mapping)
            or observation.get("iteration") != index
            or observation.get("output_tokens") != len(token_ids)
        ):
            raise IsolationError(
                f"{label}: observation {index} does not match output tokens"
            )

    environment = evidence.get("environment")
    if not isinstance(environment, dict):
        raise IsolationError(f"{label}: capture environment is missing")
    if environment.get("cuda_visible_devices") != expected_gpu["selector"]:
        raise IsolationError(f"{label}: CUDA_VISIBLE_DEVICES mismatch")
    if environment.get("cuda_logical_device") != 0:
        raise IsolationError(f"{label}: child did not use logical CUDA device 0")
    if environment.get("cuda_device_uuid") != expected_gpu["uuid"]:
        raise IsolationError(f"{label}: physical GPU UUID mismatch")
    if environment.get("cuda_pci_bus_id") != expected_gpu["pci_bus_id"]:
        raise IsolationError(f"{label}: physical GPU PCI bus mismatch")
    if environment.get("cuda_compute_capability") != runtime_stack.get("sm"):
        raise IsolationError(f"{label}: runtime stack SM disagrees with CUDA")
    plans = result.get("runtime_attention_plans")
    if (
        not isinstance(plans, list)
        or not plans
        or any(
            not isinstance(plan, Mapping) or plan.get("device") != 0
            for plan in plans
        )
    ):
        raise IsolationError(
            f"{label}: runtime plans do not use logical CUDA device 0"
        )
    runtime_libraries = result.get("runtime_libraries")
    if not isinstance(runtime_libraries, dict) or not runtime_libraries:
        raise IsolationError(f"{label}: runtime library provenance is missing")
    if provenance.get("runtime_libraries_sha256") != _canonical_sha(
        runtime_libraries
    ):
        raise IsolationError(f"{label}: runtime-library hash mismatch")
    if runtime_libraries != expected_runtime_libraries:
        raise IsolationError(
            f"{label}: child runtime libraries differ from performance receipt"
        )
    cache = _validate_cache(
        label=label,
        result=result,
        expected_path=expected_cache,
        expected_state=expected_cache_state,
    )
    worker_started_ns = cache["worker_started_ns"]
    worker_finished_ns = cache["worker_finished_ns"]
    if (
        load_started_ns < worker_started_ns
        or load_finished_ns > worker_finished_ns
    ):
        raise IsolationError(
            f"{label}: engine-load interval is outside the worker lifetime"
        )
    return {
        "capture_result": dict(result_identity),
        "model_id": result.get("model_id"),
        "model_revision": provenance.get("model_revision"),
        "bundle_sha256": provenance.get("bundle_sha256"),
        "source_state_sha256": provenance.get("source_state_sha256"),
        "source_state_unchanged": provenance.get("source_state_unchanged"),
        "request_sha256": provenance.get("request_sha256"),
        "runtime_stack": runtime_stack,
        "runtime_stack_sha256": provenance.get("runtime_stack_sha256"),
        "runtime_libraries": runtime_libraries,
        "runtime_libraries_sha256": provenance.get(
            "runtime_libraries_sha256"
        ),
        "provenance_binding": provenance_binding,
        "output_summary": output_summary,
        "output_summary_sha256": _canonical_sha(output_summary),
        "token_ids": token_ids,
        "token_ids_sha256": _canonical_sha(token_ids),
        "cuda_environment": environment,
        "cuda_jit_cache": cache,
        "load_started_ns": load_started_ns,
        "load_finished_ns": load_finished_ns,
    }


def _all_true(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_true(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_all_true(item) for item in value)
    return value if isinstance(value, bool) else True


def run_qualification(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if os.path.lexists(output_dir):
        raise IsolationError(
            f"output directory must not already exist: {output_dir}"
        )
    if args.comparison_sequence_limit <= 0:
        raise IsolationError("--comparison-sequence-limit must be positive")

    python = args.python.expanduser().resolve()
    capture_tool = args.capture_tool.expanduser().resolve()
    bundle = args.bundle.expanduser().resolve()
    build_receipt = args.build_receipt.expanduser().resolve()
    request = args.request.expanduser().resolve()
    correctness_report = args.correctness_report.expanduser().resolve()
    performance_report = args.performance_report.expanduser().resolve()
    worker = args.worker.expanduser().resolve()
    plugin_library = args.plugin_library.expanduser().resolve()
    input_artifacts = {
        "python": _file_identity(python),
        "capture_tool": _file_identity(capture_tool),
        "bundle": _file_identity(bundle),
        "build_receipt": _file_identity(build_receipt),
        "request": _file_identity(request),
        "correctness_report": _file_identity(correctness_report),
        "performance_report": _file_identity(performance_report),
        "worker": _file_identity(worker),
        "plugin_library": _file_identity(plugin_library),
    }
    request_document = _read_object(request, "benchmark request")
    request_bundle = Path(str(request_document.get("bundle", ""))).expanduser()
    if not request_bundle.is_absolute():
        request_bundle = (request.parent / request_bundle).resolve()
    else:
        request_bundle = request_bundle.resolve()
    if request_bundle != bundle:
        raise IsolationError("benchmark request bundle does not match --bundle")
    runtime = request_document.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("max_sequence_length")
        != args.comparison_sequence_limit
    ):
        raise IsolationError(
            "benchmark request max_sequence_length does not match comparison "
            "limit"
        )
    measurement = request_document.get("measurement")
    semantic_request = request_document.get("request")
    if (
        request_document.get("operation") != "generate"
        or not isinstance(request_document.get("case_name"), str)
        or not request_document["case_name"]
        or not isinstance(request_document.get("case_digest"), str)
        or not request_document["case_digest"]
        or not isinstance(measurement, dict)
        or isinstance(measurement.get("warmup"), bool)
        or not isinstance(measurement.get("warmup"), int)
        or measurement["warmup"] < 0
        or isinstance(measurement.get("iterations"), bool)
        or not isinstance(measurement.get("iterations"), int)
        or measurement["iterations"] <= 0
        or not isinstance(semantic_request, dict)
        or semantic_request.get("temperature") not in (0, 0.0)
        or semantic_request.get("top_k") != 1
        or semantic_request.get("top_p") not in (1, 1.0)
        or semantic_request.get("num_samples", 1) != 1
        or isinstance(semantic_request.get("seed"), bool)
        or not isinstance(semantic_request.get("seed"), int)
    ):
        raise IsolationError(
            "process-isolation request must be deterministic greedy generate "
            "with fixed seed, warmup, and measured iterations"
        )

    inventory = _gpu_inventory()
    gpu_a = _resolve_gpu(args.gpu_a, inventory)
    gpu_b = _resolve_gpu(args.gpu_b, inventory)
    if gpu_a["uuid"] == gpu_b["uuid"]:
        raise IsolationError("GPU A and GPU B must resolve to different UUIDs")

    qualification_started_ns = time.time_ns()
    source_state_pre = _source_state_snapshot(
        repo_root, output_dir, label="aggregate-pre"
    )
    capture_tool_binding = _canonical_capture_tool_binding(
        repo_root=repo_root,
        requested_path=capture_tool,
        source_state=source_state_pre,
    )
    if capture_tool_binding["identity"] != input_artifacts["capture_tool"]:
        raise IsolationError(
            "canonical capture tool changed before source binding completed"
        )
    input_artifacts["capture_tool_source_binding"] = capture_tool_binding
    correctness_evidence = _validate_correctness_report(
        correctness_report,
        bundle_identity=input_artifacts["bundle"],
        source_state=source_state_pre,
    )
    expected_runtime_trtmc = _validate_aggregate_base_alignment(
        inputs=input_artifacts,
        correctness=correctness_evidence,
    )
    performance_evidence = _validate_performance_report(
        performance_report,
        bundle_identity=input_artifacts["bundle"],
        source_state=source_state_pre,
        correctness=correctness_evidence,
        aggregate_inputs=input_artifacts,
        capture_tool_binding=capture_tool_binding,
        expected_runtime_trtmc=expected_runtime_trtmc,
    )
    if (
        correctness_evidence["report"]
        != input_artifacts["correctness_report"]
        or performance_evidence["report"]
        != input_artifacts["performance_report"]
    ):
        raise IsolationError(
            "companion qualification report changed while being validated"
        )
    cache_paths = {
        "gpu-a-cold": output_dir / "cuda-cache" / "gpu-a-shared",
        "gpu-a-warm": output_dir / "cuda-cache" / "gpu-a-shared",
        "gpu-a-concurrent": output_dir / "cuda-cache" / "gpu-a-concurrent",
        "gpu-b-concurrent": output_dir / "cuda-cache" / "gpu-b-concurrent",
    }
    if any(os.path.lexists(path) for path in set(cache_paths.values())):
        raise IsolationError("private CUDA cache paths must start absent")

    executions, orchestration_errors = _run_capture_matrix(
        gpu_a=gpu_a,
        gpu_b=gpu_b,
        cache_paths=cache_paths,
        output_dir=output_dir,
        python=python,
        capture_tool=capture_tool,
        capture_tool_binding=capture_tool_binding,
        repo_root=repo_root,
        bundle=bundle,
        build_receipt=build_receipt,
        request=request,
        worker=worker,
        plugin_library=plugin_library,
        comparison_sequence_limit=args.comparison_sequence_limit,
    )

    child_results: dict[str, dict[str, Any]] = {}
    validation_errors: list[str] = []
    expected_states = {
        "gpu-a-cold": "cold",
        "gpu-a-warm": "warm",
        "gpu-a-concurrent": "cold",
        "gpu-b-concurrent": "cold",
    }
    expected_gpus = {
        "gpu-a-cold": gpu_a,
        "gpu-a-warm": gpu_a,
        "gpu-a-concurrent": gpu_a,
        "gpu-b-concurrent": gpu_b,
    }
    for label in _RUN_LABELS:
        execution = executions.get(label)
        if execution is None:
            validation_errors.append(f"{label}: execution receipt is missing")
            continue
        try:
            child_results[label] = _validate_child_result(
                label=label,
                execution=execution,
                expected_gpu=expected_gpus[label],
                expected_cache=cache_paths[label],
                expected_cache_state=expected_states[label],
                expected_bundle=bundle,
                expected_bundle_sha256=input_artifacts["bundle"]["sha256"],
                expected_source_state=source_state_pre,
                expected_model_id=correctness_evidence["model_id"],
                expected_model_revision=correctness_evidence[
                    "model_revision"
                ],
                expected_runtime_stack=correctness_evidence[
                    "runtime_stack"
                ],
                expected_runtime_libraries=performance_evidence[
                    "runtime_libraries"
                ],
                expected_runtime_trtmc=expected_runtime_trtmc,
                expected_build_receipt=input_artifacts[
                    "build_receipt"
                ],
                expected_worker=input_artifacts["worker"],
                expected_plugin=input_artifacts["plugin_library"],
                expected_build_manifest=correctness_evidence[
                    "base_artifact_binding"
                ]["build_manifest"],
                expected_capture_tool=capture_tool_binding,
                expected_request=input_artifacts["request"],
                request_document=request_document,
            )
        except (IsolationError, OSError, ValueError) as exc:
            validation_errors.append(str(exc))

    complete = len(child_results) == len(_RUN_LABELS)
    if complete:
        values = list(child_results.values())
        shared = {
            "bundle": len({item["bundle_sha256"] for item in values}) == 1,
            "source": len(
                {item["source_state_sha256"] for item in values}
            )
            == 1,
            "request": len({item["request_sha256"] for item in values}) == 1,
            "model": len({item["model_id"] for item in values}) == 1,
            "model_revision": len(
                {item["model_revision"] for item in values}
            )
            == 1,
            "runtime_stack": len(
                {item["runtime_stack_sha256"] for item in values}
            )
            == 1,
            "runtime_libraries": len(
                {item["runtime_libraries_sha256"] for item in values}
            )
            == 1,
            "full_binary_provenance": len(
                {
                    item["provenance_binding"]["sha256"]
                    for item in values
                }
            )
            == 1,
            "output_summary": len(
                {item["output_summary_sha256"] for item in values}
            )
            == 1,
            "token_ids": len(
                {item["token_ids_sha256"] for item in values}
            )
            == 1,
            "all_child_source_states_unchanged": all(
                item["source_state_unchanged"] is True for item in values
            ),
        }
        concurrent_a = child_results["gpu-a-concurrent"]
        concurrent_b = child_results["gpu-b-concurrent"]
        process_overlap = _overlap_ns(
            int(executions["gpu-a-concurrent"]["started_ns"]),
            int(executions["gpu-a-concurrent"]["finished_ns"]),
            int(executions["gpu-b-concurrent"]["started_ns"]),
            int(executions["gpu-b-concurrent"]["finished_ns"]),
        )
        cache_a = concurrent_a["cuda_jit_cache"]
        cache_b = concurrent_b["cuda_jit_cache"]
        worker_overlap = _overlap_ns(
            int(cache_a["worker_started_ns"]),
            int(cache_a["worker_finished_ns"]),
            int(cache_b["worker_started_ns"]),
            int(cache_b["worker_finished_ns"]),
        )
        engine_load_overlap = _overlap_ns(
            int(concurrent_a["load_started_ns"]),
            int(concurrent_a["load_finished_ns"]),
            int(concurrent_b["load_started_ns"]),
            int(concurrent_b["load_finished_ns"]),
        )
    else:
        shared = {
            "bundle": False,
            "source": False,
            "request": False,
            "model": False,
            "model_revision": False,
            "runtime_stack": False,
            "runtime_libraries": False,
            "full_binary_provenance": False,
            "output_summary": False,
            "token_ids": False,
            "all_child_source_states_unchanged": False,
        }
        process_overlap = 0
        worker_overlap = 0
        engine_load_overlap = 0

    cache_path_gates = {
        "cold_and_warm_share_one_private_cache": (
            cache_paths["gpu-a-cold"] == cache_paths["gpu-a-warm"]
        ),
        "concurrent_caches_are_distinct": len(
            {
                cache_paths["gpu-a-concurrent"],
                cache_paths["gpu-b-concurrent"],
            }
        )
        == 2,
        "concurrent_caches_do_not_reuse_warm_cache": (
            cache_paths["gpu-a-concurrent"]
            != cache_paths["gpu-a-cold"]
            and cache_paths["gpu-b-concurrent"]
            != cache_paths["gpu-a-cold"]
        ),
        "warm_cache_continues_from_cold_process": (
            complete
            and child_results["gpu-a-cold"]["cuda_jit_cache"]["after"][
                "metadata_sha256"
            ]
            == child_results["gpu-a-warm"]["cuda_jit_cache"]["before"][
                "metadata_sha256"
            ]
        ),
        "all_cache_paths_are_inside_output": all(
            path.resolve().is_relative_to(output_dir)
            for path in set(cache_paths.values())
        ),
    }
    execution_policy_gates = {
        label: (
            execution.get("attempt_count") == 1
            and execution.get("retry_count") == 0
            and execution.get("locking") == "none"
            and execution.get("ld_preload_set") is False
            and execution.get("environment", {}).get("CUDA_CACHE_DISABLE")
            == "0"
        )
        for label, execution in executions.items()
    }

    source_state_post = _source_state_snapshot(
        repo_root, output_dir, label="aggregate-post"
    )
    source_state_unchanged = bool(
        source_state_pre.get("git_head") == source_state_post.get("git_head")
        and source_state_pre.get("source_state_sha256")
        == source_state_post.get("source_state_sha256")
    )
    input_file_identities = [
        identity
        for name, identity in input_artifacts.items()
        if name != "capture_tool_source_binding"
    ]
    input_files_unchanged = all(
        _file_identity_matches(identity)
        for identity in input_file_identities
    )
    companion_file_identities = [
        correctness_evidence["report"],
        *correctness_evidence["base_artifact_files"],
        *correctness_evidence["runner_stderr_logs"],
        *correctness_evidence["logit_artifacts"],
        *correctness_evidence["qualified_engine_graph"][
            "inspector_artifacts"
        ],
        performance_evidence["report"],
        *performance_evidence["captures"].values(),
        *performance_evidence["runtime_library_files"],
        *[
            identity
            for replay in performance_evidence[
                "dynamic_capture_provenance"
            ].values()
            for identity in replay["evidence_files"]
        ],
        *[
            identity
            for child in child_results.values()
            for identity in child["provenance_binding"][
                "evidence_files"
            ]
        ],
    ]
    companion_files_unchanged = all(
        _file_identity_matches(identity)
        for identity in companion_file_identities
    )
    qualification_finished_ns = time.time_ns()
    gates = {
        "companion_qualification_receipts": {
            "correctness_full_canonical_matrix_passed": (
                correctness_evidence["canonical_case_count"] > 0
            ),
            "correctness_hf_logit_parity_passed": (
                correctness_evidence["hf_parity_case_count"] > 0
            ),
            "correctness_context_memory_envelope_passed": (
                correctness_evidence["context_memory_envelope"]["passed"]
                is True
            ),
            "correctness_qualified_engine_graph_passed": (
                correctness_evidence["qualified_engine_graph"]["passed"]
                is True
                and all(
                    value is True
                    for value in correctness_evidence[
                        "qualified_engine_graph"
                    ]["gates"].values()
                )
            ),
            "correctness_base_artifact_binding_replayed": (
                correctness_evidence["base_artifact_binding"].get(
                    "schema_version"
                )
                == boundary.BASE_ARTIFACT_BINDING_SCHEMA
            ),
            "correctness_runtime_kv_plugin_binding_replayed": (
                correctness_evidence[
                    "runtime_kv_plugin_binding"
                ].get("schema_version")
                == boundary.RUNTIME_KV_PLUGIN_BINDING_SCHEMA
            ),
            "performance_split_08_09_passed": all(
                values["decode_throughput_ratio"] >= 0.95
                and values["prefill_ratio"] <= 1.10
                for values in performance_evidence["split_08_09"].values()
            ),
            "correctness_and_performance_bundle_match": (
                correctness_evidence["bundle_sha256"]
                == performance_evidence["bundle_sha256"]
                == input_artifacts["bundle"]["sha256"]
            ),
            "correctness_and_performance_source_match": (
                correctness_evidence["source_state_sha256"]
                == performance_evidence["source_state_sha256"]
                == source_state_pre["source_state_sha256"]
            ),
            "correctness_and_performance_model_revision_match": (
                correctness_evidence["model_revision"]
                == performance_evidence["model_revision"]
            ),
            "correctness_and_performance_runtime_stack_match": (
                correctness_evidence["runtime_stack_sha256"]
                == performance_evidence["runtime_stack_sha256"]
            ),
            "companion_reports_artifacts_and_libraries_unchanged": (
                companion_files_unchanged
            ),
        },
        "all_four_executions_recorded": len(executions) == len(_RUN_LABELS),
        "all_four_capture_results_valid": complete,
        "shared_child_evidence": shared,
        "cache_paths": cache_path_gates,
        "execution_policy": execution_policy_gates,
        "concurrent_process_intervals_overlap": process_overlap > 0,
        "concurrent_worker_intervals_overlap": worker_overlap > 0,
        "concurrent_engine_load_intervals_overlap": engine_load_overlap > 0,
        "gpu_a_and_gpu_b_are_distinct": (
            gpu_a["uuid"] != gpu_b["uuid"]
            and gpu_a["pci_bus_id"] != gpu_b["pci_bus_id"]
        ),
        "source_state_unchanged": source_state_unchanged,
        "aggregate_input_artifacts_unchanged": input_files_unchanged,
    }
    errors = [*orchestration_errors, *validation_errors]
    status = "passed" if not errors and _all_true(gates) else "failed"
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "started_ns": qualification_started_ns,
        "finished_ns": qualification_finished_ns,
        "duration_ns": qualification_finished_ns - qualification_started_ns,
        "attempt_count": 1,
        "claim_scope": {
            "directly_proves": [
                "separate-process CUDA JIT cache cold/warm behavior",
                "different-GPU concurrent process and JIT-cache isolation",
                "different-GPU engine-load intervals overlap",
                "identical generated output summary and token IDs",
            ],
            "aggregate_proves": [
                (
                    "full canonical-matrix Hugging Face logit parity for the "
                    "same bundle/source/live-runtime-stack tuple"
                ),
                (
                    "SPLIT-08 decode and SPLIT-09 prefill gates for the same "
                    "bundle/source/live-runtime-stack/runtime-library tuple"
                ),
                (
                    "source-bound split-engine graph inspection with "
                    "complete dynamic I/O/profile evidence and no "
                    "dense-attention or cache-concatenation fallback"
                ),
            ],
            "does_not_prove": [
                (
                    "each cold/warm/concurrent child independently "
                    "recomputed Hugging Face reference logits"
                ),
                (
                    "SPLIT-08/SPLIT-09 timing was measured inside each "
                    "cold/warm/concurrent child"
                ),
            ],
        },
        "execution_contract": {
            "capture_action": "benchmark",
            "artifact_role": "native-dynamic",
            "attempts_per_child": 1,
            "retries": 0,
            "locking": "none",
            "ld_preload": "unset",
            "cuda_cache_disable": "0",
        },
        "inputs": input_artifacts,
        "comparison_sequence_limit": args.comparison_sequence_limit,
        "gpu_inventory": inventory,
        "gpu_a": gpu_a,
        "gpu_b": gpu_b,
        "cache_paths": {
            label: str(path.resolve())
            for label, path in cache_paths.items()
        },
        "source_state_pre": source_state_pre,
        "source_state_post": source_state_post,
        "source_state_unchanged": source_state_unchanged,
        "companion_qualification_evidence": {
            "correctness": correctness_evidence,
            "performance": performance_evidence,
        },
        "executions": executions,
        "child_results": child_results,
        "concurrency": {
            "process_overlap_ns": process_overlap,
            "worker_overlap_ns": worker_overlap,
            "engine_load_overlap_ns": engine_load_overlap,
        },
        "gates": gates,
        "errors": errors,
    }
    report_path = output_dir / "process-isolation-report.json"
    _write_json(report_path, report)
    report["report"] = _file_identity(report_path)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--capture-tool",
        type=Path,
        default=Path(__file__).with_name(
            "capture_native_dynamic_memory_perf.py"
        ),
    )
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable)
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument(
        "--correctness-report",
        type=Path,
        required=True,
        help=(
            "Passed full-matrix HF correctness qualification report for the "
            "same bundle/source/runtime tuple"
        ),
    )
    parser.add_argument(
        "--performance-report",
        type=Path,
        required=True,
        help=(
            "Passed SPLIT-08/09 performance qualification report for the "
            "same bundle/source/runtime tuple"
        ),
    )
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--plugin-library", type=Path, required=True)
    parser.add_argument(
        "--comparison-sequence-limit", type=int, required=True
    )
    parser.add_argument("--gpu-a", required=True)
    parser.add_argument("--gpu-b", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_qualification(args)
    except (IsolationError, OSError, ValueError) as exc:
        print(
            json.dumps({"status": "failed", "error": str(exc)}),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report["report"]["path"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
