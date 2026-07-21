# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove the effective shared-GPU capacity exposed by generic proof runners.

Boundary: manual capacity admission evidence only; this module never runs model code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context import CiContext
from .gpu_lease import GpuLease
from .process import CiError


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_GPU_UUID = re.compile(r"GPU-[A-Za-z0-9-]+")
_LOCK_NAMESPACE = re.compile(r"[0-9a-f]{64}")
_RECEIPT_KIND = "trtmc_capacity_canary"


def _positive_integer(value: str | int, name: str, *, maximum: int) -> int:
    text = str(value)
    if not text.isdigit() or not 1 <= int(text) <= maximum:
        raise CiError(f"{name} must be an integer from 1 to {maximum}")
    return int(text)


def capacity_matrix(expected_slots: int) -> dict[str, list[int]]:
    """Return one more generic job than the capacity being proved."""
    expected = _positive_integer(expected_slots, "expected_slots", maximum=128)
    return {"leg": list(range(expected + 1))}


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _single_gpu_uuid(output: str, source: str) -> str:
    values = [line.strip() for line in output.splitlines() if line.strip()]
    if len(values) != 1 or _GPU_UUID.fullmatch(values[0]) is None:
        raise CiError(f"{source} did not report exactly one valid GPU UUID")
    return values[0]


def probe_container_gpu_uuid(
    context: CiContext,
    *,
    gpu_index: str,
    image: str,
    container_name: str,
) -> str:
    """Resolve the leased device UUID inside a short-lived, network-free container."""
    if not gpu_index.isdigit():
        raise CiError("capacity-canary GPU index must be a non-negative integer")
    if not image.strip():
        raise CiError("TRTMC_CI_IMAGE must identify an existing local image")
    if _SAFE_ID.fullmatch(container_name) is None or len(container_name) > 120:
        raise CiError("capacity-canary container name is unsafe")
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        container_name,
        "--gpus",
        f"device={gpu_index}",
        "--entrypoint=nvidia-smi",
        image,
        "--query-gpu=uuid",
        "--format=csv,noheader",
    ]
    try:
        output = context.output(command)
    finally:
        context.run(
            ["docker", "rm", "--force", container_name],
            check=False,
            capture_output=True,
        )
    return _single_gpu_uuid(output, "container nvidia-smi probe")


def _receipt_from_evidence(
    evidence: Mapping[str, object],
    *,
    leg_id: int,
    expected_slots: int,
    barrier_epoch: int,
    worker_started_at: str,
    probe_gpu_uuid: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": _RECEIPT_KIND,
        "leg_id": leg_id,
        "expected_slots": expected_slots,
        "barrier_epoch": barrier_epoch,
        "source_revision": evidence.get("source_revision"),
        "run_id": evidence.get("run_id"),
        "job_id": evidence.get("job_id"),
        "runner_name": evidence.get("runner_name"),
        "node_id": evidence.get("node_id"),
        "hostname": evidence.get("hostname"),
        "resource_class": evidence.get("resource_class"),
        "gpu_index": evidence.get("gpu_index"),
        "gpu_uuid": evidence.get("gpu_uuid"),
        "gpu_slot": evidence.get("gpu_slot"),
        "gpu_slots_per_device": evidence.get("gpu_slots_per_device"),
        "lock_namespace": evidence.get("lock_namespace"),
        "worker_started_at": worker_started_at,
        "acquired_at": evidence.get("acquired_at"),
        "released_at": evidence.get("released_at"),
        "probe_gpu_uuid": probe_gpu_uuid,
    }


def hold_capacity_slot(
    *,
    context: CiContext,
    leg_id: int,
    expected_slots: int,
    barrier_epoch: int,
    receipt_output: Path,
    lease_factory: Callable[..., GpuLease] = GpuLease,
    probe: Callable[..., str] = probe_container_gpu_uuid,
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Acquire one real shared lease and hold it until the common UTC epoch."""
    expected = _positive_integer(expected_slots, "expected_slots", maximum=128)
    leg = int(leg_id)
    if not 0 <= leg <= expected:
        raise CiError(f"leg_id must be between 0 and {expected}")
    barrier = _positive_integer(barrier_epoch, "barrier_epoch", maximum=4_102_444_800)
    source_revision = context.env.get("GITHUB_SHA", "")
    if not source_revision:
        raise CiError("GITHUB_SHA is required for capacity-canary evidence")
    run_id = context.env.get("GITHUB_RUN_ID", "local")
    run_attempt = context.env.get("GITHUB_RUN_ATTEMPT", "1")
    if _SAFE_ID.fullmatch(run_id) is None or _SAFE_ID.fullmatch(run_attempt) is None:
        raise CiError("GitHub run identity is unsafe")

    worker_started_at = datetime.now(timezone.utc).isoformat()
    lease = lease_factory(
        context,
        model=f"capacity-canary-{leg}",
        resource_class="shared",
    )
    lease.acquire()
    receipt: dict[str, object] | None = None
    try:
        evidence = lease.evidence(source_revision)
        gpu_index = str(evidence.get("gpu_index", ""))
        host_gpu_uuid = str(evidence.get("gpu_uuid", ""))
        probed_uuid = probe(
            context,
            gpu_index=gpu_index,
            image=context.env.get("TRTMC_CI_IMAGE", ""),
            container_name=f"trtmc-capacity-{run_id}-{run_attempt}-{leg}",
        )
        if probed_uuid != host_gpu_uuid:
            raise CiError("container GPU UUID does not match the device selected by the host lease")
        remaining = barrier - now()
        while remaining > 0:
            sleep(remaining)
            remaining = barrier - now()
        lease.mark_released()
        receipt = _receipt_from_evidence(
            lease.evidence(source_revision),
            leg_id=leg,
            expected_slots=expected,
            barrier_epoch=barrier,
            worker_started_at=worker_started_at,
            probe_gpu_uuid=probed_uuid,
        )
    finally:
        lease.release()

    assert receipt is not None
    _atomic_json(receipt_output, receipt)
    return receipt


def _exclusive_lease_record(
    evidence: Mapping[str, object],
    *,
    worker_started_at: str,
    probe_gpu_uuid: str,
) -> dict[str, object]:
    return {
        "source_revision": evidence.get("source_revision"),
        "run_id": evidence.get("run_id"),
        "job_id": evidence.get("job_id"),
        "runner_name": evidence.get("runner_name"),
        "node_id": evidence.get("node_id"),
        "hostname": evidence.get("hostname"),
        "resource_class": evidence.get("resource_class"),
        "gpu_index": evidence.get("gpu_index"),
        "gpu_uuid": evidence.get("gpu_uuid"),
        "gpu_slot": evidence.get("gpu_slot"),
        "gpu_slot_ids": evidence.get("gpu_slot_ids"),
        "gpu_slots_per_device": evidence.get("gpu_slots_per_device"),
        "lock_namespace": evidence.get("lock_namespace"),
        "worker_started_at": worker_started_at,
        "acquired_at": evidence.get("acquired_at"),
        "released_at": evidence.get("released_at"),
        "probe_gpu_uuid": probe_gpu_uuid,
    }


def run_exclusive_contender(
    *,
    context: CiContext,
    leg_id: int,
    attempted_output: Path,
    acquired_output: Path,
    receipt_output: Path,
    lease_factory: Callable[..., GpuLease] = GpuLease,
    probe: Callable[..., str] = probe_container_gpu_uuid,
) -> dict[str, object]:
    """Attempt one pinned exclusive lease from the canary's child process."""
    source_revision = context.env.get("GITHUB_SHA", "")
    run_id = context.env.get("GITHUB_RUN_ID", "local")
    run_attempt = context.env.get("GITHUB_RUN_ATTEMPT", "1")
    if not source_revision:
        raise CiError("GITHUB_SHA is required for exclusive contender evidence")
    if _SAFE_ID.fullmatch(run_id) is None or _SAFE_ID.fullmatch(run_attempt) is None:
        raise CiError("GitHub run identity is unsafe")
    lease = lease_factory(
        context,
        model=f"capacity-exclusive-contender-{leg_id}",
        resource_class="exclusive_gpu",
    )
    attempted_at = datetime.now(timezone.utc).isoformat()
    _atomic_json(attempted_output, {"attempted_at": attempted_at})
    lease.acquire()
    record: dict[str, object] | None = None
    try:
        acquired_evidence = lease.evidence(source_revision)
        _atomic_json(
            acquired_output,
            {
                "acquired_at": acquired_evidence.get("acquired_at"),
                "gpu_uuid": acquired_evidence.get("gpu_uuid"),
            },
        )
        probed_uuid = probe(
            context,
            gpu_index=str(acquired_evidence.get("gpu_index", "")),
            image=context.env.get("TRTMC_CI_IMAGE", ""),
            container_name=f"trtmc-exclusive-contender-{run_id}-{run_attempt}-{leg_id}",
        )
        if probed_uuid != acquired_evidence.get("gpu_uuid"):
            raise CiError("exclusive contender container UUID does not match its lease")
        lease.mark_released()
        record = _exclusive_lease_record(
            lease.evidence(source_revision),
            worker_started_at=attempted_at,
            probe_gpu_uuid=probed_uuid,
        )
    finally:
        lease.release()
    assert record is not None
    receipt = {
        "schema_version": 1,
        "kind": "trtmc_exclusive_contender",
        "attempted_at": attempted_at,
        "lease": record,
    }
    _atomic_json(receipt_output, receipt)
    return receipt


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run_exclusive_safety(
    *,
    context: CiContext,
    leg_id: int,
    receipt_output: Path,
    observation_seconds: int,
    lease_factory: Callable[..., GpuLease] = GpuLease,
    probe: Callable[..., str] = probe_container_gpu_uuid,
) -> dict[str, object]:
    """Prove same-GPU exclusive serialization with a real child contender."""
    observation = _positive_integer(observation_seconds, "observation_seconds", maximum=60)
    if leg_id < 0:
        raise CiError("leg_id must be a non-negative integer")
    source_revision = context.env.get("GITHUB_SHA", "")
    if not source_revision:
        raise CiError("GITHUB_SHA is required for exclusive-safety evidence")
    run_id = context.env.get("GITHUB_RUN_ID", "local")
    run_attempt = context.env.get("GITHUB_RUN_ATTEMPT", "1")
    if _SAFE_ID.fullmatch(run_id) is None or _SAFE_ID.fullmatch(run_attempt) is None:
        raise CiError("GitHub run identity is unsafe")
    output_dir = receipt_output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    attempted_output = output_dir / f"exclusive-attempted-{leg_id}.json"
    acquired_output = output_dir / f"exclusive-acquired-{leg_id}.json"
    contender_output = output_dir / f"exclusive-contender-{leg_id}.json"
    contender_log = output_dir / f"exclusive-contender-{leg_id}.log"
    for path in (attempted_output, acquired_output, contender_output, contender_log):
        path.unlink(missing_ok=True)

    primary_started_at = datetime.now(timezone.utc).isoformat()
    primary = lease_factory(
        context,
        model=f"capacity-exclusive-primary-{leg_id}",
        resource_class="exclusive_gpu",
    )
    primary.acquire()
    process: subprocess.Popen[str] | None = None
    primary_record: dict[str, object] | None = None
    log_handle = None
    primary_probe = ""
    try:
        primary_evidence = primary.evidence(source_revision)
        primary_uuid = str(primary_evidence.get("gpu_uuid", ""))
        primary_probe = probe(
            context,
            gpu_index=str(primary_evidence.get("gpu_index", "")),
            image=context.env.get("TRTMC_CI_IMAGE", ""),
            container_name=f"trtmc-exclusive-primary-{run_id}-{run_attempt}-{leg_id}",
        )
        if primary_probe != primary_uuid:
            raise CiError("exclusive primary container UUID does not match its lease")

        child_environment = dict(context.env)
        child_environment["TRTMC_GPU_ID"] = str(primary_evidence.get("gpu_index", ""))
        command = [
            sys.executable,
            "-m",
            "tools.ci.capacity_canary",
            "exclusive-contender",
            "--leg-id",
            str(leg_id),
            "--attempted-output",
            str(attempted_output),
            "--acquired-output",
            str(acquired_output),
            "--receipt-output",
            str(contender_output),
        ]
        log_handle = contender_log.open("w", encoding="utf-8")
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=context.repository,
            env=child_environment,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        attempt_deadline = time.monotonic() + 30
        while not attempted_output.is_file() and time.monotonic() < attempt_deadline:
            if process.poll() is not None:
                raise CiError("exclusive contender exited before attempting its lease")
            time.sleep(0.05)
        if not attempted_output.is_file():
            raise CiError("exclusive contender did not attempt its lease within 30 seconds")

        observation_deadline = time.monotonic() + observation
        while time.monotonic() < observation_deadline:
            if acquired_output.is_file():
                raise CiError("exclusive contender acquired while the primary still held the GPU")
            if process.poll() is not None:
                raise CiError("exclusive contender stopped instead of queueing for the GPU")
            time.sleep(0.05)
    except BaseException:
        if process is not None:
            _stop_process(process)
        if log_handle is not None:
            log_handle.close()
        raise
    finally:
        primary.mark_released()
        primary_record = _exclusive_lease_record(
            primary.evidence(source_revision),
            worker_started_at=primary_started_at,
            probe_gpu_uuid=primary_probe,
        )
        primary.release()

    assert process is not None
    try:
        return_code = process.wait(timeout=120)
    except subprocess.TimeoutExpired as error:
        _stop_process(process)
        raise CiError("exclusive contender did not resume within 120 seconds") from error
    finally:
        if log_handle is not None:
            log_handle.close()
    if return_code != 0:
        detail = contender_log.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise CiError(f"exclusive contender failed after primary release:\n{detail}")
    try:
        contender_receipt = json.loads(contender_output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CiError("exclusive contender did not write valid release evidence") from error
    if (
        not isinstance(contender_receipt, dict)
        or contender_receipt.get("kind") != "trtmc_exclusive_contender"
        or not isinstance(contender_receipt.get("lease"), dict)
    ):
        raise CiError("exclusive contender release evidence has an invalid shape")
    assert primary_record is not None
    receipt = {
        "schema_version": 1,
        "kind": "trtmc_exclusive_safety",
        "placement_scope": "one scheduler-selected generic runner",
        "source_revision": source_revision,
        "run_id": str(primary_record.get("run_id", "")),
        "runner_name": primary_record.get("runner_name"),
        "node_id": primary_record.get("node_id"),
        "primary": primary_record,
        "contender": {
            "attempted_at": contender_receipt.get("attempted_at"),
            **contender_receipt["lease"],
        },
    }
    _atomic_json(receipt_output, receipt)
    return receipt


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CiError(f"capacity receipt {field} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CiError(f"capacity receipt {field} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise CiError(f"capacity receipt {field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _validated_receipt(value: Mapping[str, object], *, expected_slots: int) -> dict[str, Any]:
    if value.get("schema_version") != 1 or value.get("kind") != _RECEIPT_KIND:
        raise CiError("capacity receipt schema or kind is unsupported")
    if value.get("expected_slots") != expected_slots:
        raise CiError("capacity receipt expected_slots does not match the canary")
    leg_id = value.get("leg_id")
    if isinstance(leg_id, bool) or not isinstance(leg_id, int):
        raise CiError("capacity receipt leg_id must be an integer")
    barrier_epoch = value.get("barrier_epoch")
    if isinstance(barrier_epoch, bool) or not isinstance(barrier_epoch, int) or barrier_epoch < 1:
        raise CiError("capacity receipt barrier_epoch must be a positive integer")
    if value.get("resource_class") != "shared":
        raise CiError("capacity receipt must describe a shared GPU slot")
    runner_name = value.get("runner_name")
    if not isinstance(runner_name, str) or not runner_name:
        raise CiError("capacity receipt runner_name must not be empty")
    node_id = value.get("node_id")
    if not isinstance(node_id, str) or _SAFE_ID.fullmatch(node_id) is None:
        raise CiError("capacity receipt node_id is unsafe")
    gpu_uuid = value.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or _GPU_UUID.fullmatch(gpu_uuid) is None:
        raise CiError("capacity receipt GPU UUID is invalid")
    if value.get("probe_gpu_uuid") != gpu_uuid:
        raise CiError("capacity receipt container probe does not match the leased GPU UUID")
    gpu_slot = value.get("gpu_slot")
    slots_per_device = value.get("gpu_slots_per_device")
    if isinstance(gpu_slot, bool) or not isinstance(gpu_slot, int):
        raise CiError("capacity receipt gpu_slot must be an integer")
    if (
        isinstance(slots_per_device, bool)
        or not isinstance(slots_per_device, int)
        or not 1 <= slots_per_device <= 16
    ):
        raise CiError("capacity receipt gpu_slots_per_device must be from 1 to 16")
    if not 0 <= gpu_slot < slots_per_device:
        raise CiError("capacity receipt gpu_slot is outside the configured slot range")
    lock_namespace = value.get("lock_namespace")
    if not isinstance(lock_namespace, str) or _LOCK_NAMESPACE.fullmatch(lock_namespace) is None:
        raise CiError("capacity receipt lock_namespace is invalid")
    started = _timestamp(value.get("worker_started_at"), "worker_started_at")
    acquired = _timestamp(value.get("acquired_at"), "acquired_at")
    released = _timestamp(value.get("released_at"), "released_at")
    if not started <= acquired < released:
        raise CiError("capacity receipt timestamps are not ordered")
    return {
        **value,
        "leg_id": leg_id,
        "barrier_epoch": barrier_epoch,
        "runner_name": runner_name,
        "node_id": node_id,
        "gpu_uuid": gpu_uuid,
        "gpu_slot": gpu_slot,
        "gpu_slots_per_device": slots_per_device,
        "lock_namespace": lock_namespace,
        "_started": started,
        "_acquired": acquired,
        "_released": released,
    }


def _maximum_concurrency(receipts: Sequence[Mapping[str, Any]]) -> int:
    events: list[tuple[datetime, int]] = []
    for receipt in receipts:
        events.append((receipt["_acquired"], 1))
        events.append((receipt["_released"], -1))
    active = 0
    maximum = 0
    for _, change in sorted(events, key=lambda event: (event[0], event[1])):
        active += change
        if active < 0:
            raise CiError("capacity receipt intervals are inconsistent")
        maximum = max(maximum, active)
    if active != 0:
        raise CiError("capacity receipt intervals do not all terminate")
    return maximum


def verify_capacity_receipts(
    values: Sequence[Mapping[str, object]],
    *,
    expected_slots: int,
    expected_run_id: str | None = None,
    expected_revision: str | None = None,
    expected_barrier_epoch: int | None = None,
) -> dict[str, object]:
    """Verify the full first wave, the queued extra leg, and dynamic node capacity."""
    expected = _positive_integer(expected_slots, "expected_slots", maximum=128)
    if len(values) != expected + 1:
        raise CiError(f"capacity canary requires {expected + 1} receipts, found {len(values)}")
    receipts = [_validated_receipt(value, expected_slots=expected) for value in values]
    if {receipt["leg_id"] for receipt in receipts} != set(range(expected + 1)):
        raise CiError("capacity receipts do not contain the exact matrix leg set")
    run_ids = {str(receipt.get("run_id", "")) for receipt in receipts}
    revisions = {str(receipt.get("source_revision", "")) for receipt in receipts}
    barrier_epochs = {receipt["barrier_epoch"] for receipt in receipts}
    if len(run_ids) != 1 or "" in run_ids:
        raise CiError("capacity receipts do not share one non-empty run ID")
    if len(revisions) != 1 or "" in revisions:
        raise CiError("capacity receipts do not share one non-empty source revision")
    if len(barrier_epochs) != 1:
        raise CiError("capacity receipts do not share one absolute barrier epoch")
    barrier_epoch = next(iter(barrier_epochs))
    if expected_run_id is not None and run_ids != {expected_run_id}:
        raise CiError("capacity receipt run ID does not match this workflow run")
    if expected_revision is not None and revisions != {expected_revision}:
        raise CiError("capacity receipt source revision does not match this workflow")
    if expected_barrier_epoch is not None and barrier_epoch != expected_barrier_epoch:
        raise CiError("capacity receipt barrier epoch does not match this workflow")

    ordered = sorted(
        receipts,
        key=lambda receipt: (
            receipt["_started"],
            receipt["_acquired"],
            receipt["leg_id"],
        ),
    )
    first_wave = ordered[:expected]
    extra = ordered[expected]
    barrier_time = datetime.fromtimestamp(barrier_epoch, tz=timezone.utc)
    first_release = min(receipt["_released"] for receipt in first_wave)
    latest_first_acquisition = max(receipt["_acquired"] for receipt in first_wave)
    if latest_first_acquisition >= first_release:
        raise CiError("the expected first-wave slots were never held concurrently")
    if any(receipt["_acquired"] >= barrier_time for receipt in first_wave):
        raise CiError("a first-wave lease did not acquire before the absolute barrier")
    if any(receipt["_released"] < barrier_time for receipt in first_wave):
        raise CiError("a first-wave lease released before the absolute barrier")
    if extra["_started"] < first_release:
        raise CiError("the extra matrix leg started before a first-wave lease was released")

    slot_tuples = {
        (receipt["node_id"], receipt["gpu_uuid"], receipt["gpu_slot"]) for receipt in first_wave
    }
    if len(slot_tuples) != expected:
        raise CiError("first-wave capacity receipts contain duplicate GPU slot tuples")
    runner_names = {receipt["runner_name"] for receipt in first_wave}
    if len(runner_names) != expected:
        raise CiError("first-wave capacity receipts do not use unique runner listeners")
    maximum = _maximum_concurrency(receipts)
    if maximum != expected:
        raise CiError(
            f"capacity canary observed maximum concurrency {maximum}, expected {expected}"
        )

    node_summaries: dict[str, dict[str, object]] = {}
    for node_id in sorted({receipt["node_id"] for receipt in first_wave}):
        node_receipts = [receipt for receipt in first_wave if receipt["node_id"] == node_id]
        slots_values = {receipt["gpu_slots_per_device"] for receipt in node_receipts}
        namespaces = {receipt["lock_namespace"] for receipt in node_receipts}
        if len(slots_values) != 1 or len(namespaces) != 1:
            raise CiError(f"node {node_id} reports inconsistent slot or lock policy")
        slots_per_device = next(iter(slots_values))
        gpu_uuids = sorted({receipt["gpu_uuid"] for receipt in node_receipts})
        for gpu_uuid in gpu_uuids:
            observed_slots = {
                receipt["gpu_slot"] for receipt in node_receipts if receipt["gpu_uuid"] == gpu_uuid
            }
            if observed_slots != set(range(slots_per_device)):
                raise CiError(f"node {node_id} GPU {gpu_uuid} did not expose every configured slot")
        capacity = len(gpu_uuids) * slots_per_device
        if capacity != len(node_receipts):
            raise CiError(f"node {node_id} observed capacity is not unique GPUs x slots")
        node_summaries[node_id] = {
            "capacity": capacity,
            "gpu_count": len(gpu_uuids),
            "gpu_uuids": gpu_uuids,
            "slots_per_gpu": slots_per_device,
            "runner_count": len({item["runner_name"] for item in node_receipts}),
            "lock_namespace": next(iter(namespaces)),
        }
    if sum(int(node["capacity"]) for node in node_summaries.values()) != expected:
        raise CiError("per-node observed capacities do not sum to expected_slots")

    return {
        "schema_version": 1,
        "kind": "trtmc_capacity_canary_verification",
        "outcome": "success",
        "run_id": next(iter(run_ids)),
        "source_revision": next(iter(revisions)),
        "expected_slots": expected,
        "barrier_epoch": barrier_epoch,
        "barrier_at": barrier_time.isoformat(),
        "receipt_count": len(receipts),
        "maximum_concurrency": maximum,
        "first_wave_runner_count": len(runner_names),
        "first_wave_slot_count": len(slot_tuples),
        "first_release_at": first_release.isoformat(),
        "extra_leg_id": extra["leg_id"],
        "extra_leg_started_at": extra["_started"].isoformat(),
        "nodes": node_summaries,
    }


def _validated_exclusive_lease(value: Mapping[str, object], *, label: str) -> dict[str, Any]:
    if value.get("resource_class") != "exclusive_gpu":
        raise CiError(f"{label} must use the exclusive_gpu resource class")
    node_id = value.get("node_id")
    if not isinstance(node_id, str) or _SAFE_ID.fullmatch(node_id) is None:
        raise CiError(f"{label} node_id is unsafe")
    runner_name = value.get("runner_name")
    if not isinstance(runner_name, str) or not runner_name:
        raise CiError(f"{label} runner_name must not be empty")
    gpu_uuid = value.get("gpu_uuid")
    if not isinstance(gpu_uuid, str) or _GPU_UUID.fullmatch(gpu_uuid) is None:
        raise CiError(f"{label} GPU UUID is invalid")
    if value.get("probe_gpu_uuid") != gpu_uuid:
        raise CiError(f"{label} container UUID does not match its lease")
    slots_per_device = value.get("gpu_slots_per_device")
    if (
        isinstance(slots_per_device, bool)
        or not isinstance(slots_per_device, int)
        or not 1 <= slots_per_device <= 16
    ):
        raise CiError(f"{label} slots-per-GPU policy is invalid")
    slot_ids = value.get("gpu_slot_ids")
    if (
        not isinstance(slot_ids, list)
        or any(isinstance(slot, bool) or not isinstance(slot, int) for slot in slot_ids)
        or slot_ids != list(range(slots_per_device))
        or value.get("gpu_slot") is not None
    ):
        raise CiError(f"{label} does not own every configured GPU slot")
    namespace = value.get("lock_namespace")
    if not isinstance(namespace, str) or _LOCK_NAMESPACE.fullmatch(namespace) is None:
        raise CiError(f"{label} lock namespace is invalid")
    started = _timestamp(value.get("worker_started_at"), f"{label}.worker_started_at")
    acquired = _timestamp(value.get("acquired_at"), f"{label}.acquired_at")
    released = _timestamp(value.get("released_at"), f"{label}.released_at")
    if not started <= acquired < released:
        raise CiError(f"{label} timestamps are not ordered")
    return {
        **value,
        "node_id": node_id,
        "runner_name": runner_name,
        "gpu_uuid": gpu_uuid,
        "gpu_slots_per_device": slots_per_device,
        "gpu_slot_ids": slot_ids,
        "lock_namespace": namespace,
        "_started": started,
        "_acquired": acquired,
        "_released": released,
    }


def verify_exclusive_safety_receipts(
    values: Sequence[Mapping[str, object]],
    *,
    expected_run_id: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, object]:
    """Verify same-node, same-GPU serialization for two exclusive attempts."""
    if len(values) != 1:
        raise CiError(
            "exclusive-safety mode requires exactly one scheduler-selected runner receipt"
        )
    receipt = values[0]
    if receipt.get("schema_version") != 1 or receipt.get("kind") != "trtmc_exclusive_safety":
        raise CiError("exclusive-safety receipt schema or kind is unsupported")
    primary_value = receipt.get("primary")
    contender_value = receipt.get("contender")
    if not isinstance(primary_value, dict) or not isinstance(contender_value, dict):
        raise CiError("exclusive-safety receipt must contain two lease records")
    primary = _validated_exclusive_lease(primary_value, label="primary")
    contender = _validated_exclusive_lease(contender_value, label="contender")
    attempted = _timestamp(contender.get("attempted_at"), "contender.attempted_at")

    run_id = str(receipt.get("run_id", ""))
    revision = str(receipt.get("source_revision", ""))
    if not run_id or not revision:
        raise CiError("exclusive-safety receipt lacks workflow identity")
    if expected_run_id is not None and run_id != expected_run_id:
        raise CiError("exclusive-safety run ID does not match this workflow")
    if expected_revision is not None and revision != expected_revision:
        raise CiError("exclusive-safety source revision does not match this workflow")
    if any(str(item.get("run_id", "")) != run_id for item in (primary, contender)):
        raise CiError("exclusive lease records do not share the workflow run ID")
    if any(str(item.get("source_revision", "")) != revision for item in (primary, contender)):
        raise CiError("exclusive lease records do not share the source revision")
    if (
        receipt.get("node_id") != primary["node_id"]
        or receipt.get("runner_name") != primary["runner_name"]
    ):
        raise CiError("exclusive-safety top-level runner identity is inconsistent")
    for field in ("node_id", "runner_name", "gpu_uuid", "lock_namespace"):
        if primary[field] != contender[field]:
            raise CiError(f"exclusive attempts do not share the same {field}")
    if primary["gpu_slots_per_device"] != contender["gpu_slots_per_device"]:
        raise CiError("exclusive attempts do not share one slots-per-GPU policy")
    if attempted != contender["_started"] or attempted > contender["_acquired"]:
        raise CiError("exclusive contender attempt timestamp is inconsistent")
    if not attempted < primary["_released"]:
        raise CiError("exclusive contender did not attempt while the primary held the GPU")
    if contender["_acquired"] < primary["_released"]:
        raise CiError("exclusive leases overlapped on the same GPU")

    return {
        "schema_version": 1,
        "kind": "trtmc_exclusive_safety_verification",
        "outcome": "success",
        "placement_scope": "one scheduler-selected generic runner; fleet placement is not proven",
        "run_id": run_id,
        "source_revision": revision,
        "runner_name": primary["runner_name"],
        "node_id": primary["node_id"],
        "gpu_uuid": primary["gpu_uuid"],
        "configured_slots": primary["gpu_slots_per_device"],
        "primary_owns_all_slots": True,
        "contender_owns_all_slots": True,
        "container_uuid_matches": True,
        "same_gpu_exclusive_non_overlap": True,
        "queued_work_resumed_after_release": True,
        "contender_attempted_at": attempted.isoformat(),
        "primary_released_at": primary["_released"].isoformat(),
        "contender_acquired_at": contender["_acquired"].isoformat(),
    }


def load_receipts(directory: Path) -> list[dict[str, object]]:
    paths = sorted(directory.rglob("receipt-*.json"))
    values: list[dict[str, object]] = []
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CiError(f"could not read capacity receipt {path}: {error}") from error
        if not isinstance(value, dict):
            raise CiError(f"capacity receipt must be a JSON object: {path}")
        values.append(value)
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m tools.ci.capacity_canary")
    commands = parser.add_subparsers(dest="command", required=True)
    matrix = commands.add_parser("matrix", help="Emit the expected+1 job matrix")
    matrix.add_argument("--expected-slots", required=True)
    hold = commands.add_parser("hold", help="Acquire and hold one slot to an absolute epoch")
    hold.add_argument("--leg-id", required=True, type=int)
    hold.add_argument("--expected-slots", required=True)
    hold.add_argument("--barrier-epoch", required=True)
    hold.add_argument("--receipt-output", required=True, type=Path)
    exclusive = commands.add_parser(
        "exclusive-safety", help="Prove same-GPU exclusive serialization"
    )
    exclusive.add_argument("--leg-id", required=True, type=int)
    exclusive.add_argument("--observation-seconds", default="3")
    exclusive.add_argument("--receipt-output", required=True, type=Path)
    contender = commands.add_parser("exclusive-contender", help="Internal pinned contender process")
    contender.add_argument("--leg-id", required=True, type=int)
    contender.add_argument("--attempted-output", required=True, type=Path)
    contender.add_argument("--acquired-output", required=True, type=Path)
    contender.add_argument("--receipt-output", required=True, type=Path)
    verify = commands.add_parser("verify", help="Verify downloaded worker receipts")
    verify.add_argument("--receipts-dir", required=True, type=Path)
    verify.add_argument("--expected-slots", required=True)
    verify.add_argument("--expected-run-id")
    verify.add_argument("--expected-revision")
    verify.add_argument("--expected-barrier-epoch", type=int)
    verify.add_argument("--output", required=True, type=Path)
    verify_exclusive = commands.add_parser(
        "verify-exclusive", help="Verify exclusive-safety evidence"
    )
    verify_exclusive.add_argument("--receipts-dir", required=True, type=Path)
    verify_exclusive.add_argument("--expected-run-id")
    verify_exclusive.add_argument("--expected-revision")
    verify_exclusive.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "matrix":
            print(
                json.dumps(
                    capacity_matrix(
                        _positive_integer(arguments.expected_slots, "expected_slots", maximum=128)
                    ),
                    separators=(",", ":"),
                )
            )
        elif arguments.command == "hold":
            hold_capacity_slot(
                context=CiContext(env=dict(os.environ)),
                leg_id=arguments.leg_id,
                expected_slots=_positive_integer(
                    arguments.expected_slots, "expected_slots", maximum=128
                ),
                barrier_epoch=_positive_integer(
                    arguments.barrier_epoch, "barrier_epoch", maximum=4_102_444_800
                ),
                receipt_output=arguments.receipt_output,
            )
        elif arguments.command == "exclusive-safety":
            run_exclusive_safety(
                context=CiContext(env=dict(os.environ)),
                leg_id=arguments.leg_id,
                observation_seconds=_positive_integer(
                    arguments.observation_seconds,
                    "observation_seconds",
                    maximum=60,
                ),
                receipt_output=arguments.receipt_output,
            )
        elif arguments.command == "exclusive-contender":
            run_exclusive_contender(
                context=CiContext(env=dict(os.environ)),
                leg_id=arguments.leg_id,
                attempted_output=arguments.attempted_output,
                acquired_output=arguments.acquired_output,
                receipt_output=arguments.receipt_output,
            )
        elif arguments.command == "verify":
            summary = verify_capacity_receipts(
                load_receipts(arguments.receipts_dir),
                expected_slots=_positive_integer(
                    arguments.expected_slots, "expected_slots", maximum=128
                ),
                expected_run_id=arguments.expected_run_id,
                expected_revision=arguments.expected_revision,
                expected_barrier_epoch=arguments.expected_barrier_epoch,
            )
            _atomic_json(arguments.output, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif arguments.command == "verify-exclusive":
            summary = verify_exclusive_safety_receipts(
                load_receipts(arguments.receipts_dir),
                expected_run_id=arguments.expected_run_id,
                expected_revision=arguments.expected_revision,
            )
            _atomic_json(arguments.output, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:  # pragma: no cover - argparse owns the command choices.
            parser.error("unsupported command")
    except (CiError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
