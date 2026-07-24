#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle and allocation-slope qualification for native dynamic KV memory."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _load_boundary_module():
    path = Path(__file__).with_name("qualify_native_dynamic_memory.py")
    spec = importlib.util.spec_from_file_location("_trtmc_dynamic_boundary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import qualification helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


boundary = _load_boundary_module()
REPO_ROOT = Path(__file__).resolve().parents[1]


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
            raise ValueError(
                "qualification output inside the repository must be under "
                "artifacts/, build/, or build-* so source snapshots exclude it"
            )
    return boundary.source_state_provenance(
        REPO_ROOT,
        Path(__file__),
        artifact_dir,
        label=label,
    )


def apply_source_state_gate(
    report: dict[str, Any],
    source_state_pre: dict[str, Any],
    source_state_post: dict[str, Any],
) -> bool:
    pre_sha = source_state_pre.get("source_state_sha256")
    post_sha = source_state_post.get("source_state_sha256")
    unchanged = bool(
        isinstance(pre_sha, str)
        and pre_sha
        and pre_sha == post_sha
        and source_state_pre.get("git_head")
        == source_state_post.get("git_head")
    )
    report["source_state_pre"] = source_state_pre
    report["source_state_post"] = source_state_post
    report["source_state_unchanged"] = unchanged
    report["passed"] = bool(report.get("passed") is True and unchanged)
    return unchanged


def _run(
    runner: Path,
    bundle: Path,
    token_file: Path,
    logits_file: Path,
    runtime_tokens: int | None,
    *,
    repeat: int = 1,
    load_cycles: int = 1,
    second_runtime_tokens: int | None = None,
    controlled_reservation_target_tokens: int | None = None,
) -> dict[str, Any]:
    command = [
        str(runner),
        "--bundle",
        str(bundle),
        "--tokens",
        str(token_file),
        "--logits",
        str(logits_file),
        "--max-new-tokens",
        "2",
        "--repeat",
        str(repeat),
        "--load-cycles",
        str(load_cycles),
    ]
    if runtime_tokens is not None:
        command.extend(
            [
                "--max-sequence-length",
                str(runtime_tokens),
            ]
        )
    if second_runtime_tokens is not None:
        command.extend(
            [
                "--second-max-sequence-length",
                str(second_runtime_tokens),
            ]
        )
    if controlled_reservation_target_tokens is not None:
        command.extend(
            [
                "--controlled-reservation-target-tokens",
                str(controlled_reservation_target_tokens),
            ]
        )
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    payload = boundary._parse_runner_json(completed.stdout)
    if completed.returncode != 0 or payload.get("status") != "ok":
        raise RuntimeError(
            f"memory qualification runner failed ({completed.returncode}): "
            f"{payload}; stderr={completed.stderr[-4000:]}"
        )
    return payload


def _used(sample: dict[str, Any]) -> int:
    if "process_used_bytes" in sample:
        return int(sample["process_used_bytes"])
    return int(sample["used_bytes"])


def validate_nvml_sampler(trace: dict[str, Any]) -> dict[str, Any]:
    metadata = trace.get("memory_sampler")
    if not isinstance(metadata, dict):
        raise RuntimeError("runner output has no memory-sampler provenance")
    if metadata.get("source") != "nvmlDeviceGetComputeRunningProcesses_v3":
        raise RuntimeError("lifecycle qualification requires per-process NVML memory")
    required = (
        "pid",
        "cuda_logical_device_index",
        "physical_device_index",
        "pci_bus_id",
        "gpu_uuid",
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise RuntimeError(f"memory-sampler provenance misses fields: {missing}")
    return metadata


def validate_receipt(trace: dict[str, Any], expected_r: int) -> dict[str, int]:
    receipt = trace.get("runtime_memory_receipt")
    if not isinstance(receipt, dict):
        raise RuntimeError("runner output has no runtime memory receipt")
    required = (
        "serialized_plan_bytes",
        "resident_weight_bytes",
        "resident_weight_copy_count",
        "engine_weight_bytes",
        "context_device_memory_bytes",
        "external_device_output_bytes",
        "host_staging_bytes",
        "graph_private_device_bytes",
        "kv_reserved_bytes",
        "kv_committed_bytes",
        "kv_metadata_bytes",
        "peak_device_bytes",
        "backend_owned_cache_input_bytes",
        "backend_owned_cache_output_bytes",
        "kv_allocation_id",
        "kv_bytes_per_token",
        "runtime_kv_capacity_tokens",
    )
    missing = [field for field in required if field not in receipt]
    if missing:
        raise RuntimeError(f"runtime memory receipt misses fields: {missing}")
    r = int(receipt["runtime_kv_capacity_tokens"])
    b = int(receipt["kv_bytes_per_token"])
    reserved = int(receipt["kv_reserved_bytes"])
    if r != expected_r:
        raise RuntimeError(f"runtime allocated R={r}, expected exactly {expected_r}")
    if b <= 0 or reserved != r * b:
        raise RuntimeError(
            f"KV reservation is not contiguous R*B: R={r}, B={b}, bytes={reserved}"
        )
    exact_zero = (
        "kv_metadata_bytes",
        "backend_owned_cache_input_bytes",
        "backend_owned_cache_output_bytes",
    )
    for field in exact_zero:
        if receipt[field] != 0:
            raise RuntimeError(f"contiguous runtime requires {field}=0")
    if int(receipt["kv_committed_bytes"]) != reserved:
        raise RuntimeError("KV committed bytes differ from reserved bytes")
    return {
        "R": r,
        "B": b,
        "kv_reserved_bytes": reserved,
        "kv_allocation_id": int(receipt["kv_allocation_id"]),
    }


def validate_sequential_requests(
    trace: dict[str, Any],
    *,
    expected_count: int,
    tolerance_bytes: int,
) -> dict[str, Any]:
    samples = trace.get("sequential_requests")
    if trace.get("sequential_request_count") != expected_count or not isinstance(
        samples, list
    ) or len(samples) != expected_count:
        raise RuntimeError("sequential request sample count mismatch")
    allocation_ids = {int(item["kv_allocation_id"]) for item in samples}
    if len(allocation_ids) != 1:
        raise RuntimeError("sequential requests did not reuse one KV allocation")
    positions = {int(item["final_kv_position"]) for item in samples}
    if len(positions) != 1:
        raise RuntimeError("sequential requests produced inconsistent final positions")
    after_used = [_used(item["after"]) for item in samples]
    window = min(10, expected_count)
    first_mean = statistics.fmean(after_used[:window])
    last_mean = statistics.fmean(after_used[-window:])
    if last_mean > first_mean + tolerance_bytes:
        raise RuntimeError(
            "sequential requests show retained device-memory growth: "
            f"first_mean={first_mean:.0f}, last_mean={last_mean:.0f}, "
            f"tolerance={tolerance_bytes}"
        )
    monotonic_growth_steps = sum(
        current > previous + tolerance_bytes
        for previous, current in zip(after_used, after_used[1:])
    )
    if monotonic_growth_steps == expected_count - 1:
        raise RuntimeError("device memory grows monotonically after every request")
    return {
        "request_count": expected_count,
        "kv_allocation_id": next(iter(allocation_ids)),
        "first_window_used_mean": first_mean,
        "last_window_used_mean": last_mean,
        "delta_used_bytes": last_mean - first_mean,
        "tolerance_bytes": tolerance_bytes,
        "monotonic_growth_steps": monotonic_growth_steps,
        "passed": True,
    }


def validate_load_cycles(
    trace: dict[str, Any],
    *,
    expected_count: int,
    expected_r: int,
    tolerance_bytes: int,
) -> dict[str, Any]:
    sampler = validate_nvml_sampler(trace)
    warmup = trace.get("load_cycle_warmup")
    if not isinstance(warmup, dict) or warmup.get("measured") is not False:
        raise RuntimeError("load/unload trace has no explicit unmeasured warm-up")
    if int(warmup.get("runtime_kv_capacity_tokens", -1)) != expected_r:
        raise RuntimeError("load/unload warm-up used the wrong runtime R")
    warmup_delta = _used(warmup["after_unload"]) - _used(warmup["before_load"])
    cycles = trace.get("load_cycles")
    if trace.get("load_cycle_count") != expected_count or not isinstance(
        cycles, list
    ) or len(cycles) != expected_count:
        raise RuntimeError("load/unload cycle sample count mismatch")
    deltas: list[int] = []
    allocation_ids: set[int] = set()
    measured_cycles: list[dict[str, Any]] = []
    external_pressure_deltas: list[int] = []
    for index, cycle in enumerate(cycles):
        if int(cycle["cycle_index"]) != index:
            raise RuntimeError("load/unload cycle indices are not contiguous")
        policy = cycle.get("policy")
        if (
            not isinstance(policy, dict)
            or policy.get("kind") != "max_sequence_length"
            or int(policy.get("requested_tokens", -1)) != expected_r
            or int(cycle.get("runtime_kv_capacity_tokens", -1)) != expected_r
        ):
            raise RuntimeError(
                f"load/unload cycle {index} used the wrong runtime policy or R"
            )
        delta = _used(cycle["after_unload"]) - _used(cycle["before_load"])
        if int(cycle.get("retained_bytes", delta)) != delta:
            raise RuntimeError(
                f"load/unload cycle {index} retained-byte receipt mismatch"
            )
        if delta > tolerance_bytes:
            raise RuntimeError(
                f"load/unload cycle {index} retained {delta} device bytes "
                f"(tolerance {tolerance_bytes})"
            )
        deltas.append(delta)
        device_wide_delta = int(
            cycle.get(
                "device_wide_retained_bytes",
                int(cycle["after_unload"]["used_bytes"])
                - int(cycle["before_load"]["used_bytes"]),
            )
        )
        external_pressure_delta = device_wide_delta - delta
        external_pressure_deltas.append(external_pressure_delta)
        allocation_ids.add(int(cycle["kv_allocation_id"]))
        measured_cycles.append(
            {
                "cycle_index": index,
                "before_load": cycle["before_load"],
                "after_requests": cycle["after_requests"],
                "after_unload": cycle["after_unload"],
                "retained_bytes": delta,
                "device_wide_retained_bytes": device_wide_delta,
                "external_pressure_delta_bytes": external_pressure_delta,
                "policy": policy,
                "runtime_kv_capacity_tokens": expected_r,
                "runtime_memory_receipt": cycle["runtime_memory_receipt"],
                "kv_allocation_id": int(cycle["kv_allocation_id"]),
            }
        )
    if len(allocation_ids) != expected_count:
        raise RuntimeError("load cycles reused an allocation identity across lifetimes")
    return {
        "memory_sampler": sampler,
        "warmup": {
            "measured": False,
            "before_load": warmup["before_load"],
            "after_requests": warmup["after_requests"],
            "after_unload": warmup["after_unload"],
            "retained_bytes": warmup_delta,
            "device_wide_retained_bytes": int(
                warmup["device_wide_retained_bytes"]
            ),
            "policy": warmup["policy"],
            "runtime_kv_capacity_tokens": expected_r,
            "runtime_memory_receipt": warmup["runtime_memory_receipt"],
            "kv_allocation_id": int(warmup["kv_allocation_id"]),
        },
        "cycle_count": expected_count,
        "measured_cycle_indices": list(range(expected_count)),
        "measured_cycles": measured_cycles,
        "max_retained_bytes": max(deltas),
        "min_retained_bytes": min(deltas),
        "max_external_pressure_delta_bytes": max(external_pressure_deltas),
        "min_external_pressure_delta_bytes": min(external_pressure_deltas),
        "tolerance_bytes": tolerance_bytes,
        "unique_allocation_ids": len(allocation_ids),
        "passed": True,
    }


def validate_two_r_slope(
    small: dict[str, int], large: dict[str, int]
) -> dict[str, Any]:
    if large["R"] <= small["R"]:
        raise RuntimeError("two-R qualification requires R2 > R1")
    if small["B"] != large["B"]:
        raise RuntimeError("KV bytes per token changed between R1 and R2")
    expected = (large["R"] - small["R"]) * small["B"]
    actual = large["kv_reserved_bytes"] - small["kv_reserved_bytes"]
    if actual != expected:
        raise RuntimeError(
            f"KV allocation slope mismatch: actual delta={actual}, expected={expected}"
        )
    if small["kv_allocation_id"] == large["kv_allocation_id"]:
        raise RuntimeError("independent R1/R2 lifetimes reused an allocation identity")
    return {
        "R1": small["R"],
        "R2": large["R"],
        "B": small["B"],
        "actual_delta_bytes": actual,
        "expected_delta_bytes": expected,
        "passed": True,
    }


def validate_same_process_two_r(
    trace: dict[str, Any],
    *,
    r1: int,
    r2: int,
    tolerance_bytes: int,
) -> dict[str, Any]:
    sampler = validate_nvml_sampler(trace)
    if trace.get("mode") != "same_process_two_r_allocation_slope":
        raise RuntimeError("runner did not use same-process two-R mode")
    warmup = trace.get("allocation_slope_warmup")
    if (
        not isinstance(warmup, dict)
        or warmup.get("measured") is not False
        or int(warmup.get("runtime_kv_capacity_tokens", -1)) != r2
    ):
        raise RuntimeError("two-R trace has no explicit unmeasured R2 warm-up")
    lifetimes = trace.get("allocation_slope_lifetimes")
    if not isinstance(lifetimes, list) or len(lifetimes) != 2:
        raise RuntimeError("two-R trace must contain exactly two measured lifetimes")

    receipts: list[dict[str, int]] = []
    expected_rs = (r1, r2)
    for index, (lifetime, expected_r) in enumerate(zip(lifetimes, expected_rs)):
        if not isinstance(lifetime, dict) or lifetime.get("measured") is not True:
            raise RuntimeError(f"two-R lifetime {index} is not marked measured")
        policy = lifetime.get("policy")
        if (
            not isinstance(policy, dict)
            or policy.get("kind") != "max_sequence_length"
            or int(policy.get("requested_tokens", -1)) != expected_r
        ):
            raise RuntimeError(f"two-R lifetime {index} used the wrong policy")
        receipts.append(validate_receipt(lifetime, expected_r))
        retained = _used(lifetime["after_unload"]) - _used(
            lifetime["before_load"]
        )
        if retained > tolerance_bytes:
            raise RuntimeError(
                f"two-R lifetime {index} retained {retained} process bytes "
                f"(tolerance {tolerance_bytes})"
            )

    plan_slope = validate_two_r_slope(receipts[0], receipts[1])
    expected_delta = int(plan_slope["expected_delta_bytes"])
    process_growth = [
        _used(item["after_requests"]) - _used(item["before_load"])
        for item in lifetimes
    ]
    nvml_delta = process_growth[1] - process_growth[0]
    if abs(nvml_delta - expected_delta) > tolerance_bytes:
        raise RuntimeError(
            "same-process NVML allocation slope mismatch: "
            f"actual delta={nvml_delta}, expected={expected_delta}, "
            f"tolerance={tolerance_bytes}"
        )
    return {
        **plan_slope,
        "memory_sampler": sampler,
        "warmup": warmup,
        "measured_lifetimes": lifetimes,
        "nvml_process_growth_r1_bytes": process_growth[0],
        "nvml_process_growth_r2_bytes": process_growth[1],
        "nvml_actual_delta_bytes": nvml_delta,
        "nvml_delta_tolerance_bytes": tolerance_bytes,
        "passed": True,
    }


def validate_controlled_reservation(
    trace: dict[str, Any],
    *,
    tolerance_bytes: int,
) -> dict[str, Any]:
    sampler = validate_nvml_sampler(trace)
    if trace.get("mode") != "same_process_controlled_external_reservation":
        raise RuntimeError("runner did not use controlled-reservation mode")
    proof = trace.get("controlled_reservation")
    if not isinstance(proof, dict):
        raise RuntimeError("runner output has no controlled-reservation receipt")
    warmup = proof.get("warmup")
    baseline = proof.get("baseline")
    constrained = proof.get("constrained")
    if (
        not isinstance(warmup, dict)
        or not isinstance(baseline, dict)
        or not isinstance(constrained, dict)
    ):
        raise RuntimeError("controlled reservation misses lifetime receipts")
    if (
        warmup.get("policy") != {"kind": "auto"}
        or warmup.get("measured") is not False
        or baseline.get("policy") != {"kind": "auto"}
        or constrained.get("policy") != {"kind": "auto"}
        or baseline.get("measured") is not True
        or constrained.get("measured") is not True
    ):
        raise RuntimeError(
            "controlled reservation did not use one unmeasured auto warmup "
            "followed by measured auto lifetimes"
        )

    baseline_r = int(proof["baseline_r"])
    constrained_r = int(proof["constrained_r"])
    if not (baseline_r > constrained_r >= int(trace["final_kv_position"])):
        raise RuntimeError(
            "controlled reservation did not reduce R while fitting the request"
        )
    baseline_receipt = validate_receipt(baseline, baseline_r)
    constrained_receipt = validate_receipt(constrained, constrained_r)
    baseline_receipt_raw = baseline["runtime_memory_receipt"]
    if baseline_receipt["B"] != constrained_receipt["B"]:
        raise RuntimeError("controlled reservation changed KV bytes per token")
    sizing = proof.get("sizing")
    if not isinstance(sizing, dict):
        raise RuntimeError("controlled reservation misses sizing components")
    expected_warmup_retained = _used(warmup["after_unload"]) - _used(
        warmup["before_load"]
    )
    if (
        int(sizing["warmup_retained_process_bytes"])
        != expected_warmup_retained
    ):
        raise RuntimeError(
            "controlled reservation warmup-retained receipt mismatch"
        )
    target_kv_bytes = int(proof["target_tokens"]) * baseline_receipt["B"]
    expected_fraction_headroom = (target_kv_bytes + 7) // 8
    expected_target_headroom = max(8 * 1024 * 1024, expected_fraction_headroom)
    expected_engine_load_device_bytes = max(
        0,
        int(sizing["baseline_pre_load_free_bytes"])
        - int(sizing["baseline_post_load_free_bytes"]),
    )
    expected_post_load_runtime_overhead = (
        int(baseline_receipt_raw["context_device_memory_bytes"])
        + int(baseline_receipt_raw["external_device_output_bytes"])
        + int(baseline_receipt_raw["graph_private_device_bytes"])
    )
    expected_required_post_load_free = (
        target_kv_bytes
        + int(sizing["safety_reserve_bytes"])
        + expected_post_load_runtime_overhead
        + expected_target_headroom
    )
    if (
        proof.get("reservation_phase") != "before runtime KV planning"
        or sizing.get("required_free_basis")
        != "post-load runtime overhead before KV planning"
        or int(sizing["target_kv_bytes"]) != target_kv_bytes
        or int(sizing["target_fraction_denominator"]) != 8
        or int(sizing["target_fraction_headroom_bytes"])
        != expected_fraction_headroom
        or int(sizing["target_headroom_bytes"]) != expected_target_headroom
        or int(sizing["baseline_engine_load_device_bytes"])
        != expected_engine_load_device_bytes
        or int(sizing["context_device_memory_bytes"])
        != int(baseline_receipt_raw["context_device_memory_bytes"])
        or int(sizing["external_device_output_bytes"])
        != int(baseline_receipt_raw["external_device_output_bytes"])
        or int(sizing["graph_private_device_bytes"])
        != int(baseline_receipt_raw["graph_private_device_bytes"])
        or int(sizing["post_load_runtime_overhead_bytes"])
        != expected_post_load_runtime_overhead
        or int(sizing["required_post_load_free_bytes"])
        != expected_required_post_load_free
    ):
        raise RuntimeError("controlled reservation sizing formula mismatch")
    expected_kv_delta = (
        baseline_r - constrained_r
    ) * baseline_receipt["B"]
    actual_kv_delta = (
        baseline_receipt["kv_reserved_bytes"]
        - constrained_receipt["kv_reserved_bytes"]
    )
    if actual_kv_delta != expected_kv_delta:
        raise RuntimeError(
            "controlled reservation KV slope mismatch: "
            f"actual={actual_kv_delta}, expected={expected_kv_delta}"
        )

    reservation_bytes = int(proof["reservation_bytes"])
    if reservation_bytes <= 0 or int(proof["reservation_address"]) == 0:
        raise RuntimeError("controlled reservation has no device allocation")
    allocations = proof.get("reservation_allocations")
    allocation_count = int(proof.get("reservation_allocation_count", 0))
    if (
        not isinstance(allocations, list)
        or len(allocations) != allocation_count
        or allocation_count <= 0
        or any(int(item.get("address", 0)) == 0 for item in allocations)
        or sum(int(item.get("bytes", 0)) for item in allocations)
        != reservation_bytes
    ):
        raise RuntimeError("controlled reservation chunk receipt is inconsistent")
    before = proof["before_reservation"]
    after = proof["after_reservation"]
    observed_process_reservation = _used(after) - _used(before)
    if abs(observed_process_reservation - reservation_bytes) > tolerance_bytes:
        raise RuntimeError(
            "NVML controlled-reservation delta mismatch: "
            f"actual={observed_process_reservation}, "
            f"expected={reservation_bytes}, tolerance={tolerance_bytes}"
        )
    observed_free_delta = int(before["free_bytes"]) - int(after["free_bytes"])

    constrained_raw_retained = _used(constrained["after_unload"]) - _used(
        constrained["before_load"]
    )
    constrained_retained = constrained_raw_retained - reservation_bytes
    if abs(constrained_retained) > tolerance_bytes:
        raise RuntimeError(
            "constrained pipeline lifetime retained process memory after "
            f"excluding held reservation: {constrained_retained} bytes"
        )
    after_constrained = proof["after_constrained_unload"]
    after_release = proof["after_release"]
    observed_release = _used(after_constrained) - _used(after_release)
    if abs(observed_release - reservation_bytes) > tolerance_bytes:
        raise RuntimeError(
            "controlled reservation release delta mismatch: "
            f"actual={observed_release}, expected={reservation_bytes}"
        )
    recovery = _used(after_release) - _used(constrained["before_load"])
    if abs(recovery) > tolerance_bytes:
        raise RuntimeError(
            "controlled reservation did not return to process baseline: "
            f"delta={recovery}, tolerance={tolerance_bytes}"
        )
    return {
        "memory_sampler": sampler,
        "warmup": warmup,
        "warmup_retained_process_bytes": expected_warmup_retained,
        "target_tokens": int(proof["target_tokens"]),
        "sizing": sizing,
        "reservation_bytes": reservation_bytes,
        "reservation_address": int(proof["reservation_address"]),
        "reservation_allocation_count": allocation_count,
        "reservation_allocations": allocations,
        "free_before_reservation_bytes": int(before["free_bytes"]),
        "free_after_reservation_bytes": int(after["free_bytes"]),
        "observed_device_free_delta_bytes": observed_free_delta,
        "observed_process_reservation_bytes": observed_process_reservation,
        "baseline_r": baseline_r,
        "constrained_r": constrained_r,
        "r_delta": baseline_r - constrained_r,
        "kv_bytes_per_token": baseline_receipt["B"],
        "actual_kv_delta_bytes": actual_kv_delta,
        "expected_kv_delta_bytes": expected_kv_delta,
        "final_kv_position": int(trace["final_kv_position"]),
        "constrained_lifetime_raw_retained_bytes": constrained_raw_retained,
        "constrained_lifetime_retained_bytes": constrained_retained,
        "observed_release_bytes": observed_release,
        "release_recovery_process_bytes": recovery,
        "tolerance_bytes": tolerance_bytes,
        "baseline": baseline,
        "constrained": constrained,
        "passed": True,
    }


def _write_raw_trace(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("build-dynkv/trtmc_dynamic_memory_qualify"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r1", type=int)
    parser.add_argument("--r2", type=int)
    parser.add_argument("--reservation-target-tokens", type=int)
    parser.add_argument("--tolerance-bytes", type=int, default=32 * 1024 * 1024)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    runner = args.runner.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source_state_pre = _source_state_snapshot(output.parent, label="pre")
    raw_dir = output.parent / "soak-raw"
    header = boundary._read_bundle_header(bundle)
    spec = boundary._resolve_spec(header)
    r1 = args.r1 or spec.chunk_limit
    r2 = args.r2 or min(spec.context_limit, 2 * r1)
    if not (4 <= r1 < r2 <= spec.context_limit):
        raise ValueError("require 4 <= R1 < R2 <= model context limit")
    if args.reservation_target_tokens is not None and not (
        66 <= args.reservation_target_tokens < spec.context_limit
    ):
        raise ValueError(
            "reservation target must fit the qualification request and be "
            "below the model context limit"
        )

    vocab_size = int(header["vocab_size"])
    with tempfile.TemporaryDirectory(prefix="trtmc-memory-soak-") as temporary:
        work = Path(temporary)
        token_file = work / "tokens.txt"
        token_count = min(64, r1 - 2)
        boundary._write_tokens(
            token_file, boundary.deterministic_token_ids(token_count, vocab_size)
        )
        sequential = _run(
            runner,
            bundle,
            token_file,
            work / "sequential.bin",
            r1,
            repeat=100,
        )
        _write_raw_trace(raw_dir / "sequential-100.json", sequential)
        load_cycles = _run(
            runner,
            bundle,
            token_file,
            work / "load-cycles.bin",
            r1,
            load_cycles=20,
        )
        _write_raw_trace(raw_dir / "load-unload-20.json", load_cycles)
        two_r = _run(
            runner,
            bundle,
            token_file,
            work / "two-r.bin",
            r1,
            second_runtime_tokens=r2,
        )
        _write_raw_trace(raw_dir / "same-process-two-r.json", two_r)
        controlled_reservation = None
        if args.reservation_target_tokens is not None:
            controlled_reservation = _run(
                runner,
                bundle,
                token_file,
                work / "controlled-reservation.bin",
                None,
                controlled_reservation_target_tokens=(
                    args.reservation_target_tokens
                ),
            )
            _write_raw_trace(
                raw_dir / "controlled-reservation.json",
                controlled_reservation,
            )

    raw_traces = {
        "sequential_100": str(raw_dir / "sequential-100.json"),
        "load_unload_20": str(raw_dir / "load-unload-20.json"),
        "same_process_two_r": str(raw_dir / "same-process-two-r.json"),
    }
    if controlled_reservation is not None:
        raw_traces["controlled_reservation"] = str(
            raw_dir / "controlled-reservation.json"
        )
    report = {
        "schema_version": 2,
        "model_id": spec.model_id,
        "bundle": str(bundle),
        "bundle_sha256": boundary._sha256(bundle),
        "runner": str(runner),
        "runner_sha256": boundary._sha256(runner),
        "raw_traces": raw_traces,
        "memory_sampler": validate_nvml_sampler(sequential),
        "sequential_requests": validate_sequential_requests(
            sequential,
            expected_count=100,
            tolerance_bytes=args.tolerance_bytes,
        ),
        "load_unload_cycles": validate_load_cycles(
            load_cycles,
            expected_count=20,
            expected_r=r1,
            tolerance_bytes=args.tolerance_bytes,
        ),
        "two_r_allocation_slope": validate_same_process_two_r(
            two_r,
            r1=r1,
            r2=r2,
            tolerance_bytes=args.tolerance_bytes,
        ),
        "passed": True,
    }
    if controlled_reservation is not None:
        report["controlled_external_reservation"] = (
            validate_controlled_reservation(
                controlled_reservation,
                tolerance_bytes=args.tolerance_bytes,
            )
        )
    source_state_post = _source_state_snapshot(output.parent, label="post")
    apply_source_state_gate(report, source_state_pre, source_state_post)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
