# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private build-time calibration for a newly serialized runtime-KV plan set.

This module is deliberately not a user-facing build surface.  The native
``trtmc build`` bridge injects an exact internal helper path, and the builder
uses it only when the family manifest has no calibration for the current raw
TensorRT plan bytes.

The bootstrap contract is never returned to the user.  It exists only long
enough for two isolated helper processes to execute every decode profile.  A
failure at any point aborts the build; only the final contract contains the
measured, guarded reserves.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from . import __version__ as PACKAGE_VERSION
from .dynamic_memory_contract import (
    module_residency_plan_set_sha256,
    qualified_runtime_stack_sha256,
    seal_runtime_memory_contract,
    validate_runtime_memory_contract,
)


INTERNAL_CALIBRATOR_ENV = "_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR"
INTERNAL_CALIBRATOR_BUILD_IDENTITY_ENV = (
    "_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR_BUILD_IDENTITY"
)
CALIBRATION_EVIDENCE_ROOT = "runtime_memory_calibration"
CALIBRATION_EVIDENCE_SECTION = f"{CALIBRATION_EVIDENCE_ROOT}/evidence.json"
CALIBRATION_EVIDENCE_SCHEMA = (
    "trtmc.native-dynamic-memory-build-calibration-evidence/v2"
)
CALIBRATION_CAPTURE_SCHEMA = (
    "trtmc.native-dynamic-memory-build-calibration-capture/v1"
)
CAPTURE_PROCESS_COUNT = 2
CALIBRATION_GUARD_BYTES = 64 * 1024 * 1024
SECOND_SWEEP_PROCESS_GROWTH_LIMIT_BYTES = 64 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AutomaticDynamicMemoryCalibrationError(RuntimeError):
    """An internal per-build calibration could not be proven."""


@dataclass(frozen=True)
class CalibrationEvidenceSection:
    """One immutable bundle section carried out of the temporary capture."""

    name: str
    data: bytes


@dataclass(frozen=True)
class AutomaticCalibrationResult:
    """A final sealed contract plus its embedded evidence payload."""

    runtime_memory_contract: dict[str, Any]
    evidence_bytes: bytes
    evidence_sections: tuple[CalibrationEvidenceSection, ...]
    helper_sha256: str
    process_ids: tuple[int, int]
    gpu_uuid: str


@dataclass(frozen=True)
class _CapturedSweep:
    process_index: int
    child_pid: int
    gpu_uuid: str
    trace: dict[str, Any]
    rows: tuple[dict[str, int], ...]
    sampler_trust_anchor: dict[str, Any]
    capture_manifest_section: str
    capture_manifest_sha256: str
    artifacts: dict[str, dict[str, Any]]
    evidence_sections: tuple[CalibrationEvidenceSection, ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _file_identity(path: Path) -> dict[str, int | str]:
    """Return the exact executable identity used for a calibration phase."""

    file_stat = path.stat()
    if not stat.S_ISREG(file_stat.st_mode):
        raise AutomaticDynamicMemoryCalibrationError(
            f"Internal dynamic-memory calibrator is not a regular file: {path}"
        )
    return {
        "device": file_stat.st_dev,
        "inode": file_stat.st_ino,
        "size_bytes": file_stat.st_size,
        "mtime_ns": file_stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, separators=(",", ": "))
        + "\n"
    ).encode("utf-8")


def _section_receipt(
    section: CalibrationEvidenceSection,
) -> dict[str, int | str]:
    return {
        "section_name": section.name,
        "size_bytes": len(section.data),
        "sha256": _sha256_bytes(section.data),
    }


def resolve_internal_calibrator() -> Path:
    """Resolve the non-overridable helper path injected by native ``trtmc``."""

    raw = os.environ.get(INTERNAL_CALIBRATOR_ENV)
    if not raw or not raw.strip():
        raise AutomaticDynamicMemoryCalibrationError(
            "The native build bridge did not provide its internal dynamic-memory "
            "calibrator; run the adjacent trtmc build entrypoint"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise AutomaticDynamicMemoryCalibrationError(
            f"{INTERNAL_CALIBRATOR_ENV} must be an absolute path"
        )
    try:
        file_stat = path.stat()
    except OSError as exc:
        raise AutomaticDynamicMemoryCalibrationError(
            f"Internal dynamic-memory calibrator is unavailable: {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(file_stat.st_mode) or not os.access(path, os.X_OK):
        raise AutomaticDynamicMemoryCalibrationError(
            f"Internal dynamic-memory calibrator is not an executable file: {path}"
        )
    return path.resolve()


def _strict_single_json_line(
    stdout: str,
    *,
    operation: str,
) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise AutomaticDynamicMemoryCalibrationError(
            f"Internal calibrator {operation} must emit exactly one JSON line"
        )
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise AutomaticDynamicMemoryCalibrationError(
            f"Internal calibrator {operation} emitted invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise AutomaticDynamicMemoryCalibrationError(
            f"Internal calibrator {operation} JSON must be an object"
        )
    return value


def query_cuda_module_loading_mode(helper: Path) -> str:
    """Query the live CUDA driver before constructing the bootstrap header."""

    completed = subprocess.run(
        [str(helper), "--query-module-loading-mode"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator could not query CUDA module loading mode "
            f"(exit {completed.returncode}): {completed.stderr[-2000:]}"
        )
    result = _strict_single_json_line(
        completed.stdout,
        operation="module-loading-mode query",
    )
    expected_keys = {"schema_version", "source", "mode", "driver_value"}
    mode = result.get("mode")
    if (
        set(result) != expected_keys
        or result.get("schema_version") != 1
        or result.get("source") != "cuModuleGetLoadingMode"
        or mode not in {"lazy", "eager"}
        or result.get("driver_value") != (2 if mode == "lazy" else 1)
    ):
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator returned an invalid CUDA module-loading-mode receipt"
        )
    return str(mode)


def query_product_identity(helper: Path) -> str:
    """Prove that the helper belongs to this installed product build."""

    try:
        completed = subprocess.run(
            [str(helper), "--query-product-identity"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise AutomaticDynamicMemoryCalibrationError(
            f"Internal calibrator product-identity query could not start: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator could not report its product identity "
            f"(exit {completed.returncode}): {completed.stderr[-2000:]}"
        )
    result = _strict_single_json_line(
        completed.stdout,
        operation="product-identity query",
    )
    expected_keys = {
        "schema_version",
        "source",
        "product_version",
        "build_identity",
        "helper_protocol_version",
    }
    build_identity = result.get("build_identity")
    if (
        set(result) != expected_keys
        or result.get("schema_version") != 1
        or result.get("source") != "compiled_product_identity"
        or result.get("helper_protocol_version") != 1
        or result.get("product_version") != PACKAGE_VERSION
        or not isinstance(build_identity, str)
        or _SHA256.fullmatch(build_identity) is None
    ):
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator returned an incompatible product identity"
        )

    launcher_identity = os.environ.get(
        INTERNAL_CALIBRATOR_BUILD_IDENTITY_ENV
    )
    if launcher_identity is not None:
        if (
            _SHA256.fullmatch(launcher_identity) is None
            or build_identity != launcher_identity
        ):
            raise AutomaticDynamicMemoryCalibrationError(
                "Internal calibrator build identity does not match the native "
                "launcher"
            )
    return build_identity


def _plan_records(
    base_contract: Mapping[str, Any],
    plan_sections: Mapping[str, bytes | bytearray | memoryview],
) -> list[dict[str, Any]]:
    expected_sections = {"engine_plan", "prefill_engine_plan"}
    if set(plan_sections) != expected_sections:
        raise AutomaticDynamicMemoryCalibrationError(
            "Automatic calibration requires exactly decode and prefill plan sections"
        )
    for name, payload in plan_sections.items():
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise AutomaticDynamicMemoryCalibrationError(
                f"Automatic calibration plan section {name} is not bytes"
            )
    return [
        {
            "section_name": "engine_plan",
            "section_sha256": _sha256_bytes(bytes(plan_sections["engine_plan"])),
            "role": "decode",
            "optimization_profile_count": len(
                base_contract["active_kv_profile_limits"]
            ),
        },
        {
            "section_name": "prefill_engine_plan",
            "section_sha256": _sha256_bytes(
                bytes(plan_sections["prefill_engine_plan"])
            ),
            "role": "prefill",
            "optimization_profile_count": 1,
        },
    ]


def _bootstrap_calibration(
    base_contract: Mapping[str, Any],
    *,
    plans: list[dict[str, Any]],
    module_loading_mode: str,
) -> dict[str, Any]:
    """Return a plan-bound bootstrap only; one byte is not a product reserve."""

    plan_set_sha256 = module_residency_plan_set_sha256(plans)
    bootstrap_identity = _canonical_json_bytes(
        {
            "purpose": "ephemeral-build-calibration-bootstrap",
            "qualified_runtime_stack_sha256": qualified_runtime_stack_sha256(
                base_contract["qualified_runtime_stack"]
            ),
            "plan_set_sha256": plan_set_sha256,
            "cuda_module_loading_mode": module_loading_mode,
        }
    )
    return {
        "schema_version": 1,
        "measurement_kind": "nvml_process_cumulative_first_use",
        "cuda_module_loading_mode": module_loading_mode,
        "qualified_runtime_stack_sha256": qualified_runtime_stack_sha256(
            base_contract["qualified_runtime_stack"]
        ),
        "plan_set_sha256": plan_set_sha256,
        "plans": plans,
        "profile_reserves": [
            {
                "covering_profile_limit": limit,
                "cumulative_reserve_bytes": 1,
            }
            for limit in base_contract["active_kv_profile_limits"]
        ],
        "evidence_sha256": _sha256_bytes(bootstrap_identity),
    }


def _profile_prompt_lengths(limits: Iterable[int], model_limit: int) -> tuple[int, ...]:
    normalized = tuple(limits)
    if (
        not normalized
        or any(type(value) is not int or value <= 1 for value in normalized)
        or tuple(sorted(set(normalized))) != normalized
        or normalized[-1] != model_limit
    ):
        raise AutomaticDynamicMemoryCalibrationError(
            "Automatic calibration received invalid decode profile limits"
        )
    prompts: list[int] = []
    previous = 0
    for limit in normalized:
        prompt = limit - 1
        if prompt <= previous:
            raise AutomaticDynamicMemoryCalibrationError(
                "Automatic calibration cannot select an upper-edge shape for "
                f"profile interval ({previous}, {limit}]"
            )
        prompts.append(prompt)
        previous = limit
    return tuple(prompts)


def _token_file_bytes(length: int, vocab_size: int) -> bytes:
    if length <= 0 or vocab_size <= 1:
        raise AutomaticDynamicMemoryCalibrationError(
            "Automatic calibration token geometry is invalid"
        )
    values = (
        (
            (
                position * 48_271
                + (position >> 3) * 69_621
                + 17
            )
            % (vocab_size - 1)
        )
        + 1
        for position in range(length)
    )
    return ("".join(f"{value}\n" for value in values)).encode("ascii")


def _receipt_matches_bootstrap(
    receipt: Any,
    *,
    contract: Mapping[str, Any],
) -> bool:
    calibration = contract["module_residency_calibration"]
    terminal = calibration["profile_reserves"][-1]
    expected = {
        "receipt_schema_version": 4,
        "contract_version": 2,
        "policy": "auto",
        "request_context_limit": contract["model_context_limit"],
        "model_context_limit": contract["model_context_limit"],
        "runtime_kv_capacity_tokens": contract["model_context_limit"],
        "effective_request_limit": contract["model_context_limit"],
        "kv_bytes_per_token": contract["kv_bytes_per_token"],
        "module_residency_reserve_bytes": terminal[
            "cumulative_reserve_bytes"
        ],
        "module_residency_reserve_profile_limit": terminal[
            "covering_profile_limit"
        ],
        "module_residency_plan_set_sha256": calibration["plan_set_sha256"],
        "module_residency_evidence_sha256": calibration["evidence_sha256"],
        "module_residency_cuda_module_loading_mode": calibration[
            "cuda_module_loading_mode"
        ],
    }
    return isinstance(receipt, Mapping) and all(
        receipt.get(field) == value for field, value in expected.items()
    )


def _validated_process_rows(
    trace: Any,
    *,
    contract: Mapping[str, Any],
    child_pid: int,
    token_paths: tuple[Path, ...],
    prompt_lengths: tuple[int, ...],
    logits_path: Path,
    vocab_size: int,
) -> tuple[tuple[dict[str, int], ...], str]:
    limits = tuple(contract["active_kv_profile_limits"])
    model_id = contract["qualified_model_id"]
    loading_mode = contract["module_residency_calibration"][
        "cuda_module_loading_mode"
    ]
    expected_top_level = {
        "schema_version": 1,
        "mode": "all_profile_two_sweep",
        "status": "ok",
        "error_type": None,
        "passed": True,
        "qualification_blockers": [],
        "qualification_api_version": 1,
        "model_id": model_id,
        "pipeline_type": {
            "Qwen/Qwen3-0.6B": "QwenTextGenerationPipeline",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0": (
                "LlamaTextGenerationPipeline"
            ),
        }.get(model_id),
    }
    if not isinstance(trace, Mapping) or any(
        trace.get(field) != value for field, value in expected_top_level.items()
    ):
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator profile sweep did not pass its top-level gates"
        )
    if trace.get("cuda_module_loading") != {
        "source": "cuModuleGetLoadingMode",
        "mode": loading_mode,
        "driver_value": 2 if loading_mode == "lazy" else 1,
    }:
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator profile sweep changed CUDA module-loading mode"
        )
    if trace.get("protocol") != {
        "schema_version": 1,
        "execution_order": ["sweep_a", "sweep_b"],
        "pipeline_load_count": 1,
        "kv_allocation_count": 1,
        "max_new_tokens_per_request": 1,
        "profile_order": "ascending",
        "second_sweep_process_growth_limit_bytes": (
            SECOND_SWEEP_PROCESS_GROWTH_LIMIT_BYTES
        ),
    }:
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator did not use the required two-sweep protocol"
        )
    sampler = trace.get("memory_sampler")
    sampler_keys = {
        "source",
        "pid",
        "cuda_logical_device_index",
        "physical_device_index",
        "pci_bus_id",
        "gpu_uuid",
        "captures_all_compute_processes",
        "device_memory_source",
    }
    if (
        not isinstance(sampler, Mapping)
        or set(sampler) != sampler_keys
        or sampler.get("source")
        != "nvmlDeviceGetComputeRunningProcesses_v3"
        or sampler.get("pid") != child_pid
        or sampler.get("cuda_logical_device_index") != 0
        or type(sampler.get("physical_device_index")) is not int
        or sampler["physical_device_index"] < 0
        or not isinstance(sampler.get("pci_bus_id"), str)
        or not sampler["pci_bus_id"]
        or sampler.get("captures_all_compute_processes") is not True
        or sampler.get("device_memory_source") != "nvmlDeviceGetMemoryInfo_v2"
        or not isinstance(sampler.get("gpu_uuid"), str)
        or not sampler["gpu_uuid"]
    ):
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator NVML attribution does not bind its child process"
        )
    manifest = trace.get("input_manifest")
    if not isinstance(manifest, list) or len(manifest) != len(limits):
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator input manifest does not cover every profile"
        )
    previous_limit = 0
    for index, (row, path, prompt, limit) in enumerate(
        zip(manifest, token_paths, prompt_lengths, limits, strict=True)
    ):
        if row != {
            "row_index": index,
            "token_file": str(path.resolve()),
            "prompt_tokens": prompt,
            "expected_profile_id": index,
            "expected_profile_limit": limit,
            "expected_prompt_lower_exclusive": previous_limit,
        }:
            raise AutomaticDynamicMemoryCalibrationError(
                f"Internal calibrator input manifest row {index} is not exact"
            )
        previous_limit = limit
    expected_gates = {
        "stable_kv_allocation_id",
        "sweep_a_exact_profile_coverage",
        "sweep_b_exact_profile_coverage",
        "selected_token_ids_bitwise_equivalent",
        "complete_float32_logits_bitwise_equivalent",
        "invocation_tuples_equivalent",
        "sweep_b_process_incremental_high_water_within_limit",
    }
    gates = trace.get("gates")
    stable_allocation_id = trace.get("stable_kv_allocation_id")
    if (
        not isinstance(gates, Mapping)
        or set(gates) != expected_gates
        or any(value is not True for value in gates.values())
        or type(stable_allocation_id) is not int
        or stable_allocation_id <= 0
    ):
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator did not pass every two-sweep equivalence gate"
        )

    sweep_a = trace.get("sweep_a")
    sweep_b = trace.get("sweep_b")
    reserve_rows = trace.get("profile_reserve_rows")
    if (
        not isinstance(sweep_a, Mapping)
        or not isinstance(sweep_b, Mapping)
        or not isinstance(sweep_a.get("rows"), list)
        or not isinstance(sweep_b.get("rows"), list)
        or not isinstance(reserve_rows, list)
        or len(sweep_a["rows"]) != len(limits)
        or len(sweep_b["rows"]) != len(limits)
        or len(reserve_rows) != len(limits)
    ):
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator did not return exact all-profile rows"
        )

    normalized: list[dict[str, int]] = []
    previous_process = 0
    previous_device = 0
    for index, (row_a, row_b, reserve, prompt, limit) in enumerate(
        zip(
            sweep_a["rows"],
            sweep_b["rows"],
            reserve_rows,
            prompt_lengths,
            limits,
            strict=True,
        )
    ):
        common = {
            "row_index": index,
            "token_file": str(token_paths[index].resolve()),
            "prompt_tokens": prompt,
            "max_new_tokens": 1,
            "expected_profile_id": index,
            "expected_profile_limit": limit,
            "profile_match": True,
            "observed_decode_profile_ids": [index],
            "kv_allocation_id": stable_allocation_id,
        }
        if (
            not isinstance(row_a, Mapping)
            or not isinstance(row_b, Mapping)
            or any(row_a.get(field) != value for field, value in common.items())
            or any(row_b.get(field) != value for field, value in common.items())
            or row_a.get("expected_prompt_lower_exclusive")
            != (0 if index == 0 else limits[index - 1])
            or row_b.get("expected_prompt_lower_exclusive")
            != (0 if index == 0 else limits[index - 1])
            or not _receipt_matches_bootstrap(
                row_a.get("runtime_memory_receipt"),
                contract=contract,
            )
            or not _receipt_matches_bootstrap(
                row_b.get("runtime_memory_receipt"),
                contract=contract,
            )
            or row_b.get("equivalence_to_sweep_a")
            != {
                "selected_token_ids_bitwise_equal": True,
                "complete_float32_logits_bitwise_equal": True,
                "invocation_tuples_equal": True,
                "passed": True,
            }
        ):
            raise AutomaticDynamicMemoryCalibrationError(
                f"Internal calibrator profile row {index} is not exact"
            )
        for field in (
            "selected_token_ids",
            "step_top1_token_ids",
            "invocation_tuples",
        ):
            if row_a.get(field) != row_b.get(field):
                raise AutomaticDynamicMemoryCalibrationError(
                    f"Internal calibrator profile row {index} changed {field}"
                )
        growth = row_a.get("cumulative_first_use_growth")
        if not isinstance(growth, Mapping):
            raise AutomaticDynamicMemoryCalibrationError(
                f"Internal calibrator profile row {index} has no growth receipt"
            )
        process_bytes = growth.get("cumulative_process_high_water_bytes")
        device_bytes = growth.get("cumulative_device_wide_high_water_bytes")
        if (
            type(process_bytes) is not int
            or process_bytes < previous_process
            or type(device_bytes) is not int
            or device_bytes < previous_device
            or reserve
            != {
                "profile_id": index,
                "covering_profile_limit": limit,
                "cumulative_process_first_use_bytes": process_bytes,
                "cumulative_device_wide_first_use_bytes": device_bytes,
            }
        ):
            raise AutomaticDynamicMemoryCalibrationError(
                f"Internal calibrator profile row {index} growth is invalid"
            )
        normalized.append(
            {
                "profile_id": index,
                "covering_profile_limit": limit,
                "cumulative_process_first_use_bytes": process_bytes,
                "cumulative_device_wide_first_use_bytes": device_bytes,
            }
        )
        previous_process = process_bytes
        previous_device = device_bytes
    if (
        sweep_a.get("cumulative_process_first_use_high_water_bytes")
        != previous_process
        or sweep_a.get("cumulative_device_wide_first_use_high_water_bytes")
        != previous_device
        or sweep_b.get("process_growth_limit_bytes")
        != SECOND_SWEEP_PROCESS_GROWTH_LIMIT_BYTES
        or sweep_b.get("process_growth_within_limit") is not True
        or type(sweep_b.get("incremental_process_high_water_bytes")) is not int
        or sweep_b["incremental_process_high_water_bytes"]
        > SECOND_SWEEP_PROCESS_GROWTH_LIMIT_BYTES
    ):
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator high-water summary is inconsistent"
        )
    logits = trace.get("logits_artifact")
    if (
        not isinstance(logits, Mapping)
        or logits.get("format") != "trtmc-qualification-logits-v1"
        or logits.get("dtype") != "float32"
        or type(logits.get("rows")) is not int
        or logits["rows"] <= 0
        or type(logits.get("vocab_size")) is not int
        or logits["vocab_size"] != vocab_size
        or logits.get("path") != str(logits_path.resolve())
    ):
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal calibrator logits receipt is invalid"
        )
    return tuple(normalized), str(sampler["gpu_uuid"])


def _run_sweep(
    *,
    helper: Path,
    bootstrap_bundle: Path,
    contract: Mapping[str, Any],
    process_index: int,
    prompt_lengths: tuple[int, ...],
    vocab_size: int,
    capture_dir: Path,
) -> _CapturedSweep:
    capture_dir.mkdir(parents=False, exist_ok=False)
    token_paths: list[Path] = []
    for index, prompt_length in enumerate(prompt_lengths):
        token_path = capture_dir / f"profile-{index:02d}.tokens.txt"
        payload = _token_file_bytes(prompt_length, vocab_size)
        token_path.write_bytes(payload)
        token_paths.append(token_path)
    logits_path = capture_dir / "runner-logits.bin"
    command = [
        str(helper),
        "--bundle",
        str(bootstrap_bundle),
        "--logits",
        str(logits_path),
        "--max-new-tokens",
        "1",
        "--max-sequence-length",
        str(contract["model_context_limit"]),
    ]
    for token_path in token_paths:
        command.extend(("--profile-sweep-tokens", str(token_path)))
    child = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = child.communicate()
    try:
        stdout = stdout_bytes.decode("utf-8")
        stderr = stderr_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal all-profile calibrator output is not UTF-8"
        ) from exc
    if child.returncode != 0:
        raise AutomaticDynamicMemoryCalibrationError(
            "Internal all-profile calibrator process failed "
            f"(process={process_index}, pid={child.pid}, exit={child.returncode}): "
            f"{stderr[-4000:]}"
        )
    trace = _strict_single_json_line(
        stdout,
        operation=f"profile sweep process {process_index}",
    )
    rows, gpu_uuid = _validated_process_rows(
        trace,
        contract=contract,
        child_pid=child.pid,
        token_paths=tuple(token_paths),
        prompt_lengths=prompt_lengths,
        logits_path=logits_path,
        vocab_size=vocab_size,
    )
    if not logits_path.is_file() or logits_path.stat().st_size <= 0:
        raise AutomaticDynamicMemoryCalibrationError(
            f"Internal calibrator process {process_index} produced no logits artifact"
        )
    normalized_trace_bytes = _canonical_json_bytes(trace)
    raw_lines = [line for line in stdout.splitlines() if line.strip()]
    if len(raw_lines) != 1:
        raise AssertionError("strict JSON parser accepted a non-single-line trace")
    raw_trace_bytes = (raw_lines[0] + "\n").encode("utf-8")
    prefix = f"{CALIBRATION_EVIDENCE_ROOT}/process-{process_index:02d}"
    artifact_payloads: tuple[tuple[str, str, bytes], ...] = (
        ("command", "command.json", _canonical_json_bytes(command)),
        ("returncode", "returncode.txt", b"0\n"),
        ("runner_stdout", "runner.stdout.log", stdout_bytes),
        ("runner_stderr", "runner.stderr.log", stderr_bytes),
        ("raw_trace", "runner-output.raw.json", raw_trace_bytes),
        ("normalized_trace", "runner-trace.json", normalized_trace_bytes),
        ("logits", "runner-logits.bin", logits_path.read_bytes()),
        *tuple(
            (
                f"profile_tokens_{index:02d}",
                f"profile-{index:02d}.tokens.txt",
                token_path.read_bytes(),
            )
            for index, token_path in enumerate(token_paths)
        ),
    )
    artifact_sections = tuple(
        CalibrationEvidenceSection(f"{prefix}/{filename}", payload)
        for _logical_name, filename, payload in artifact_payloads
    )
    artifacts = {
        logical_name: _section_receipt(section)
        for (logical_name, _filename, _payload), section in zip(
            artifact_payloads,
            artifact_sections,
            strict=True,
        )
    }
    sampler = dict(trace["memory_sampler"])
    sampler_trust_anchor = {
        "pid": sampler["pid"],
        "cuda_logical_device_index": sampler["cuda_logical_device_index"],
        "physical_device_index": sampler["physical_device_index"],
        "pci_bus_id": sampler["pci_bus_id"],
        "gpu_uuid": sampler["gpu_uuid"],
    }
    capture_manifest = {
        "schema": CALIBRATION_CAPTURE_SCHEMA,
        "process_index": process_index,
        "runner_pid": child.pid,
        "gpu_uuid": gpu_uuid,
        "sampler_trust_anchor": sampler_trust_anchor,
        "artifacts": artifacts,
    }
    capture_manifest_section = f"{prefix}/capture-manifest.json"
    manifest_section = CalibrationEvidenceSection(
        capture_manifest_section,
        _canonical_json_bytes(capture_manifest),
    )
    return _CapturedSweep(
        process_index=process_index,
        child_pid=child.pid,
        gpu_uuid=gpu_uuid,
        trace=trace,
        rows=rows,
        sampler_trust_anchor=sampler_trust_anchor,
        capture_manifest_section=capture_manifest_section,
        capture_manifest_sha256=_sha256_bytes(manifest_section.data),
        artifacts=artifacts,
        evidence_sections=(*artifact_sections, manifest_section),
    )


def _aggregate_reserves(
    captures: tuple[_CapturedSweep, _CapturedSweep],
    *,
    limits: tuple[int, ...],
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    reserves: list[dict[str, int]] = []
    evidence_rows: list[dict[str, Any]] = []
    previous_reserve = 0
    for index, limit in enumerate(limits):
        process_values = [
            capture.rows[index]["cumulative_process_first_use_bytes"]
            for capture in captures
        ]
        device_values = [
            capture.rows[index]["cumulative_device_wide_first_use_bytes"]
            for capture in captures
        ]
        process_max = max(process_values)
        if process_max > (2**64 - 1) - CALIBRATION_GUARD_BYTES:
            raise AutomaticDynamicMemoryCalibrationError(
                "Automatic calibration process maximum plus guard overflows uint64"
            )
        guarded = max(
            previous_reserve,
            process_max + CALIBRATION_GUARD_BYTES,
        )
        reserves.append(
            {
                "covering_profile_limit": limit,
                "cumulative_reserve_bytes": guarded,
            }
        )
        evidence_rows.append(
            {
                "profile_id": index,
                "covering_profile_limit": limit,
                "process_first_use_bytes_by_process": process_values,
                "device_wide_first_use_bytes_by_process": device_values,
                "row_wise_max_process_first_use_bytes": process_max,
                "row_wise_max_device_wide_first_use_bytes": max(
                    device_values
                ),
                "row_wise_max_first_use_bytes": max(
                    process_max,
                    max(device_values),
                ),
                "calibration_guard_bytes": CALIBRATION_GUARD_BYTES,
                "required_guarded_process_reserve_bytes": guarded,
            }
        )
        previous_reserve = guarded
    return reserves, evidence_rows


def calibrate_unknown_plan_set(
    *,
    base_contract: Mapping[str, Any],
    plan_sections: Mapping[str, bytes | bytearray | memoryview],
    runtime_config_bytes: bytes | bytearray | memoryview,
    vocab_size: int,
    working_directory: Path,
    write_bootstrap_bundle: Callable[[Path, Mapping[str, Any]], None],
    helper: Path | None = None,
) -> AutomaticCalibrationResult:
    """Measure an unknown exact plan set and return a final sealed v2 contract.

    ``write_bootstrap_bundle`` receives a private temporary path and the
    ephemeral v2 contract.  It must write the same plan bytes supplied here.
    The temporary directory and every raw capture are removed on success or
    failure.
    """

    normalized_base = validate_runtime_memory_contract(base_contract)
    if normalized_base["contract_version"] != 1:
        raise AutomaticDynamicMemoryCalibrationError(
            "Automatic calibration requires a provisional v1 contract"
        )
    if type(vocab_size) is not int or vocab_size <= 1:
        raise AutomaticDynamicMemoryCalibrationError(
            "Automatic calibration requires a valid vocabulary size"
        )
    helper_path = resolve_internal_calibrator() if helper is None else helper.resolve()
    helper_identity_before = _file_identity(helper_path)
    query_product_identity(helper_path)
    module_loading_mode = query_cuda_module_loading_mode(helper_path)
    plans = _plan_records(normalized_base, plan_sections)
    bootstrap_calibration = _bootstrap_calibration(
        normalized_base,
        plans=plans,
        module_loading_mode=module_loading_mode,
    )
    bootstrap_contract = seal_runtime_memory_contract(
        normalized_base,
        plan_sections=plan_sections,
        module_residency_calibration=bootstrap_calibration,
        runtime_config_bytes=runtime_config_bytes,
    )
    limits = tuple(normalized_base["active_kv_profile_limits"])
    prompt_lengths = _profile_prompt_lengths(
        limits,
        normalized_base["model_context_limit"],
    )
    working_directory = working_directory.resolve()
    with tempfile.TemporaryDirectory(
        prefix=".trtmc-dynamic-memory-calibration-",
        dir=working_directory,
    ) as temporary:
        root = Path(temporary)
        bootstrap_bundle = root / "bootstrap.trtfb"
        write_bootstrap_bundle(bootstrap_bundle, bootstrap_contract)
        if not bootstrap_bundle.is_file() or bootstrap_bundle.stat().st_size <= 0:
            raise AutomaticDynamicMemoryCalibrationError(
                "Automatic calibration bootstrap writer produced no bundle"
            )
        captures = tuple(
            _run_sweep(
                helper=helper_path,
                bootstrap_bundle=bootstrap_bundle,
                contract=bootstrap_contract,
                process_index=process_index,
                prompt_lengths=prompt_lengths,
                vocab_size=vocab_size,
                capture_dir=root / f"process-{process_index:02d}",
            )
            for process_index in range(CAPTURE_PROCESS_COUNT)
        )
        if len(captures) != 2:
            raise AssertionError("automatic calibration requires exactly two captures")
        first, second = captures
        if first.child_pid == second.child_pid or first.gpu_uuid != second.gpu_uuid:
            raise AutomaticDynamicMemoryCalibrationError(
                "Automatic calibration requires two distinct processes on one GPU"
            )
        profile_reserves, evidence_rows = _aggregate_reserves(
            (first, second),
            limits=limits,
        )
        helper_identity_after = _file_identity(helper_path)
        if helper_identity_after != helper_identity_before:
            raise AutomaticDynamicMemoryCalibrationError(
                "Internal dynamic-memory calibrator changed during the build"
            )
        evidence = {
            "schema": CALIBRATION_EVIDENCE_SCHEMA,
            "measurement_kind": "nvml_process_cumulative_first_use",
            "aggregation": "row_wise_process_max_plus_64mib_guard",
            "independent_process_count": CAPTURE_PROCESS_COUNT,
            "model_id": normalized_base["qualified_model_id"],
            "model_context_limit": normalized_base["model_context_limit"],
            "active_kv_profile_limits": list(limits),
            "prompt_lengths": list(prompt_lengths),
            "terminal_active_length": prompt_lengths[-1] + 1,
            "helper_identity_before": helper_identity_before,
            "helper_identity_after": helper_identity_after,
            "contract_provenance": {
                "qualified_runtime_stack_sha256": (
                    bootstrap_calibration[
                        "qualified_runtime_stack_sha256"
                    ]
                ),
                "plan_set_sha256": bootstrap_calibration["plan_set_sha256"],
                "cuda_module_loading_mode": module_loading_mode,
                "plans": plans,
            },
            "bootstrap_contract": bootstrap_contract,
            "bootstrap_only": {
                "profile_reserve_bytes": 1,
                "evidence_sha256": bootstrap_calibration["evidence_sha256"],
                "never_published": True,
            },
            "runs": [
                {
                    "process_index": capture.process_index,
                    "runner_pid": capture.child_pid,
                    "gpu_uuid": capture.gpu_uuid,
                    "sampler_trust_anchor": capture.sampler_trust_anchor,
                    "capture_manifest": {
                        "section_name": capture.capture_manifest_section,
                        "size_bytes": next(
                            len(section.data)
                            for section in capture.evidence_sections
                            if section.name
                            == capture.capture_manifest_section
                        ),
                        "sha256": capture.capture_manifest_sha256,
                    },
                    "artifacts": capture.artifacts,
                }
                for capture in captures
            ],
            "profile_rows": evidence_rows,
            "recommended_profile_reserves": profile_reserves,
            "gates": {
                "two_distinct_processes": True,
                "single_gpu_identity": True,
                "all_profile_upper_edges_executed": True,
                "terminal_decode_reaches_model_limit": (
                    prompt_lengths[-1] + 1
                    == normalized_base["model_context_limit"]
                ),
                "second_sweep_growth_within_limit": True,
                "raw_plan_receipts_match_bootstrap": True,
                "all_capture_sections_embedded_and_hashed": True,
                "helper_identity_unchanged": True,
            },
            "passed": True,
        }
        evidence_bytes = _canonical_json_bytes(evidence)
        capture_sections = tuple(
            section
            for capture in captures
            for section in capture.evidence_sections
        )
        evidence_sections = (
            CalibrationEvidenceSection(
                CALIBRATION_EVIDENCE_SECTION,
                evidence_bytes,
            ),
            *capture_sections,
        )
        section_names = [section.name for section in evidence_sections]
        if (
            len(set(section_names)) != len(section_names)
            or section_names[0] != CALIBRATION_EVIDENCE_SECTION
            or any(
                not name.startswith(f"{CALIBRATION_EVIDENCE_ROOT}/")
                for name in section_names
            )
        ):
            raise AssertionError(
                "automatic calibration produced invalid embedded section names"
            )

    final_calibration = {
        **bootstrap_calibration,
        "evidence_provenance": "embedded_bundle_v1",
        "profile_reserves": profile_reserves,
        "evidence_sha256": _sha256_bytes(evidence_bytes),
    }
    final_contract = seal_runtime_memory_contract(
        normalized_base,
        plan_sections=plan_sections,
        module_residency_calibration=final_calibration,
        runtime_config_bytes=runtime_config_bytes,
    )
    return AutomaticCalibrationResult(
        runtime_memory_contract=final_contract,
        evidence_bytes=evidence_bytes,
        evidence_sections=evidence_sections,
        helper_sha256=str(helper_identity_after["sha256"]),
        process_ids=(first.child_pid, second.child_pid),
        gpu_uuid=first.gpu_uuid,
    )
