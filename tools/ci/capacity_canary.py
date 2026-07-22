# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the declared GPU pool and prove its effective shared capacity.

Boundary: topology planning and manual admission evidence only; this module never runs model code.
"""

from __future__ import annotations

import argparse
import hashlib
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
_ACQUISITION_MARKER_KIND = "trtmc_capacity_canary_acquired"
_ACQUISITION_MARKER_PREFIX = "TRTMC_CAPACITY_CANARY_ACQUIRED="
_CAPACITY_WORKFLOW_PATH = ".github/workflows/model-proof-capacity-canary.yml"
_CAPACITY_WORKER_JOB_ID = "exercise"
_TOPOLOGY_KIND = "trtmc_capacity_topology"
_NODE_LABEL_PREFIX = "trtmc-node-"


def _positive_integer(value: str | int, name: str, *, maximum: int) -> int:
    text = str(value)
    if not text.isdigit() or not 1 <= int(text) <= maximum:
        raise CiError(f"{name} must be an integer from 1 to {maximum}")
    return int(text)


def _strict_integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise CiError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def normalize_topology_contract(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and canonicalize the operator-declared expected fleet topology.

    Node labels are control-plane identities, not scheduling branches. Operators
    scale the pool by changing the declarative topology, while the same generic
    verifier binds every trusted receipt to the digest of that exact contract.
    """
    required_fields = {
        "schema_version",
        "kind",
        "slots_per_gpu",
        "rollback_baseline_node_label",
        "nodes",
    }
    if set(value) != required_fields:
        raise CiError("topology contract fields do not match schema version 1")
    if value.get("schema_version") != 1 or value.get("kind") != _TOPOLOGY_KIND:
        raise CiError("topology contract schema or kind is unsupported")

    slots_per_gpu = _strict_integer(
        value.get("slots_per_gpu"),
        "topology slots_per_gpu",
        minimum=1,
        maximum=16,
    )
    raw_nodes = value.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes or len(raw_nodes) > 32:
        raise CiError("topology nodes must be a non-empty list of at most 32 entries")

    normalized_nodes: list[dict[str, object]] = []
    node_labels: set[str] = set()
    node_ids: set[str] = set()
    for index, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, Mapping):
            raise CiError(f"topology node {index} must be an object")
        if set(raw_node) != {"node_label", "gpu_indices"}:
            raise CiError(f"topology node {index} fields do not match schema version 1")
        node_label = raw_node.get("node_label")
        if (
            not isinstance(node_label, str)
            or len(node_label) > 120
            or not node_label.startswith(_NODE_LABEL_PREFIX)
            or _SAFE_ID.fullmatch(node_label) is None
        ):
            raise CiError(f"topology node {index} has an invalid node label")
        node_id = node_label.removeprefix(_NODE_LABEL_PREFIX)
        if not node_id or _SAFE_ID.fullmatch(node_id) is None:
            raise CiError(f"topology node {index} has an invalid node identity")
        if node_label in node_labels or node_id in node_ids:
            raise CiError("topology contract contains a duplicate node identity")
        node_labels.add(node_label)
        node_ids.add(node_id)

        raw_gpu_indices = raw_node.get("gpu_indices")
        if (
            not isinstance(raw_gpu_indices, list)
            or not raw_gpu_indices
            or len(raw_gpu_indices) > 16
        ):
            raise CiError(
                f"topology node {node_label} gpu_indices must be a non-empty list "
                "of at most 16 entries"
            )
        gpu_indices = [
            _strict_integer(
                gpu_index,
                f"topology node {node_label} GPU index",
                minimum=0,
                maximum=63,
            )
            for gpu_index in raw_gpu_indices
        ]
        if gpu_indices != sorted(set(gpu_indices)):
            raise CiError(
                f"topology node {node_label} GPU indices must be sorted and unique"
            )
        normalized_nodes.append(
            {
                "node_label": node_label,
                "gpu_indices": gpu_indices,
            }
        )

    normalized_nodes.sort(key=lambda node: str(node["node_label"]))
    rollback_baseline_node_label = value.get("rollback_baseline_node_label")
    if (
        not isinstance(rollback_baseline_node_label, str)
        or len(rollback_baseline_node_label) > 120
        or not rollback_baseline_node_label.startswith(_NODE_LABEL_PREFIX)
        or _SAFE_ID.fullmatch(rollback_baseline_node_label) is None
    ):
        raise CiError("topology rollback baseline node label is invalid")
    if rollback_baseline_node_label not in node_labels:
        raise CiError("topology rollback baseline must identify a declared node")
    gpu_count = sum(len(node["gpu_indices"]) for node in normalized_nodes)
    capacity = gpu_count * slots_per_gpu
    if gpu_count > 64 or capacity > 128:
        raise CiError("derived topology exceeds 64 GPUs or 128 shared slots")
    return {
        "schema_version": 1,
        "kind": _TOPOLOGY_KIND,
        "slots_per_gpu": slots_per_gpu,
        "rollback_baseline_node_label": rollback_baseline_node_label,
        "nodes": normalized_nodes,
    }


def _topology_gpu_count(topology: Mapping[str, object]) -> int:
    return sum(len(node["gpu_indices"]) for node in topology["nodes"])  # type: ignore[index]


def _topology_capacity(topology: Mapping[str, object]) -> int:
    return _topology_gpu_count(topology) * int(topology["slots_per_gpu"])


def topology_contract_digest(value: Mapping[str, object]) -> str:
    normalized = normalize_topology_contract(value)
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def cache_warm_matrix(value: Mapping[str, object]) -> dict[str, list[dict[str, str]]]:
    """Build exactly one cache-warm job per declared physical node."""

    topology = normalize_topology_contract(value)
    return {
        "include": [
            {"node_label": str(node["node_label"])}
            for node in topology["nodes"]  # type: ignore[index]
        ]
    }


def _rollback_baseline_topology(topology: Mapping[str, object]) -> dict[str, object]:
    """Derive the exact protected rollback baseline from a normalized topology."""

    node_label = str(topology["rollback_baseline_node_label"])
    nodes = [
        dict(node)
        for node in topology["nodes"]  # type: ignore[index]
        if node["node_label"] == node_label
    ]
    if len(nodes) != 1:  # Defensive: normalize_topology_contract owns this invariant.
        raise CiError("topology rollback baseline must identify exactly one declared node")
    return {
        "schema_version": topology["schema_version"],
        "kind": topology["kind"],
        "slots_per_gpu": topology["slots_per_gpu"],
        "rollback_baseline_node_label": node_label,
        "nodes": nodes,
    }


def _validated_topology_for_mode(
    value: Mapping[str, object],
    *,
    mode: str,
    requested_slots: int,
) -> dict[str, object]:
    topology = normalize_topology_contract(value)
    requested = _positive_integer(requested_slots, "expected_slots", maximum=128)
    if mode == "rollback-capacity":
        topology = _rollback_baseline_topology(topology)
        if requested != _topology_capacity(topology):
            raise CiError(
                "rollback-capacity expected_slots must equal rollback baseline capacity"
            )
        return topology
    capacity = _topology_capacity(topology)
    if mode in {"shared-capacity", "exclusive-safety"}:
        if requested != capacity:
            raise CiError(f"{mode} expected_slots must equal topology capacity")
    elif mode == "shared-cohort":
        if requested > capacity:
            raise CiError("shared-cohort expected_slots exceeds topology capacity")
    elif mode == "cross-workflow-verify":
        if requested * 2 != capacity:
            raise CiError(
                "cross-workflow expected_slots per run must equal half the topology capacity"
            )
    else:
        raise CiError("unsupported canary mode for topology contract")
    return topology


def _validated_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _LOCK_NAMESPACE.fullmatch(value) is None:
        raise CiError(f"{label} must be a lowercase SHA-256 digest")
    return value


def capacity_matrix(expected_slots: int) -> dict[str, list[int]]:
    """Return one more generic job than the capacity being proved."""
    expected = _positive_integer(expected_slots, "expected_slots", maximum=128)
    return {"leg": list(range(expected + 1))}


def cohort_matrix(cohort_slots: int) -> dict[str, list[int]]:
    """Return exactly the requested number of generic cohort jobs."""
    expected = _positive_integer(cohort_slots, "cohort_slots", maximum=128)
    return {"leg": list(range(expected))}


def _validated_cohort_id(value: object, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value or len(value) > 120:
        raise CiError("cohort_id must be a non-empty safe identifier of at most 120 characters")
    if _SAFE_ID.fullmatch(value) is None:
        raise CiError("cohort_id must be a non-empty safe identifier of at most 120 characters")
    return value


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
    cohort_id: str | None,
    expected_topology_digest: str,
    worker_started_at: str,
    probe_gpu_uuid: str,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": _RECEIPT_KIND,
        "leg_id": leg_id,
        "expected_slots": expected_slots,
        "barrier_epoch": barrier_epoch,
        "cohort_id": cohort_id,
        "expected_topology_digest": expected_topology_digest,
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


def _emit_acquisition_marker(
    evidence: Mapping[str, object],
    *,
    leg_id: int,
    expected_slots: int,
    barrier_epoch: int,
    cohort_id: str | None,
    expected_topology_digest: str,
    worker_started_at: str,
    probe_gpu_uuid: str,
) -> dict[str, object]:
    marker = {
        "schema_version": 2,
        "kind": _ACQUISITION_MARKER_KIND,
        "leg_id": leg_id,
        "expected_slots": expected_slots,
        "barrier_epoch": barrier_epoch,
        "cohort_id": cohort_id,
        "expected_topology_digest": expected_topology_digest,
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
        "probe_gpu_uuid": probe_gpu_uuid,
    }
    print(
        _ACQUISITION_MARKER_PREFIX + json.dumps(marker, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return marker


def hold_capacity_slot(
    *,
    context: CiContext,
    leg_id: int,
    expected_slots: int,
    barrier_epoch: int,
    expected_topology_digest: str,
    receipt_output: Path,
    cohort_id: str | None = None,
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
    cohort = _validated_cohort_id(cohort_id, required=False)
    topology_digest = _validated_digest(
        expected_topology_digest, "expected_topology_digest"
    )
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
        _emit_acquisition_marker(
            evidence,
            leg_id=leg,
            expected_slots=expected,
            barrier_epoch=barrier,
            cohort_id=cohort,
            expected_topology_digest=topology_digest,
            worker_started_at=worker_started_at,
            probe_gpu_uuid=probed_uuid,
        )
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
            cohort_id=cohort,
            expected_topology_digest=topology_digest,
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
    expected_topology_digest: str,
    lease_factory: Callable[..., GpuLease] = GpuLease,
    probe: Callable[..., str] = probe_container_gpu_uuid,
) -> dict[str, object]:
    """Prove same-GPU exclusive serialization with a real child contender."""
    observation = _positive_integer(observation_seconds, "observation_seconds", maximum=60)
    topology_digest = _validated_digest(
        expected_topology_digest, "expected_topology_digest"
    )
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
        "schema_version": 2,
        "kind": "trtmc_exclusive_safety",
        "expected_topology_digest": topology_digest,
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


def _validated_receipt(
    value: Mapping[str, object],
    *,
    expected_slots: int,
    expected_topology_digest: str,
) -> dict[str, Any]:
    if value.get("schema_version") != 2 or value.get("kind") != _RECEIPT_KIND:
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
    if value.get("job_id") != _CAPACITY_WORKER_JOB_ID:
        raise CiError("capacity receipt job_id does not match the canary worker job")
    topology_digest = _validated_digest(
        value.get("expected_topology_digest"),
        "capacity receipt expected_topology_digest",
    )
    if topology_digest != expected_topology_digest:
        raise CiError("capacity receipt topology digest does not match the trusted contract")
    runner_name = value.get("runner_name")
    if not isinstance(runner_name, str) or not runner_name:
        raise CiError("capacity receipt runner_name must not be empty")
    node_id = value.get("node_id")
    if not isinstance(node_id, str) or _SAFE_ID.fullmatch(node_id) is None:
        raise CiError("capacity receipt node_id is unsafe")
    hostname = value.get("hostname")
    if not isinstance(hostname, str) or _SAFE_ID.fullmatch(hostname) is None:
        raise CiError("capacity receipt hostname is unsafe")
    gpu_index = value.get("gpu_index")
    if (
        not isinstance(gpu_index, str)
        or not gpu_index.isdigit()
        or str(int(gpu_index)) != gpu_index
        or not 0 <= int(gpu_index) <= 63
    ):
        raise CiError("capacity receipt gpu_index must be a canonical integer from 0 to 63")
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
        "hostname": hostname,
        "gpu_index": gpu_index,
        "_gpu_index_int": int(gpu_index),
        "gpu_uuid": gpu_uuid,
        "gpu_slot": gpu_slot,
        "gpu_slots_per_device": slots_per_device,
        "lock_namespace": lock_namespace,
        "expected_topology_digest": topology_digest,
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


def _slot_tuple(receipt: Mapping[str, Any]) -> tuple[str, str, int]:
    return (receipt["node_id"], receipt["gpu_uuid"], receipt["gpu_slot"])


def _validate_topology_identity(receipts: Sequence[Mapping[str, Any]]) -> None:
    """Require stable physical identity behind every logical node."""
    node_identities: dict[str, tuple[str, str]] = {}
    hostname_nodes: dict[str, str] = {}
    gpu_nodes: dict[str, str] = {}
    node_index_uuids: dict[tuple[str, int], str] = {}
    node_uuid_indices: dict[tuple[str, str], int] = {}
    for receipt in receipts:
        node_id = str(receipt["node_id"])
        hostname = str(receipt["hostname"])
        lock_namespace = str(receipt["lock_namespace"])
        gpu_uuid = str(receipt["gpu_uuid"])
        gpu_index = int(receipt["_gpu_index_int"])

        identity = (hostname, lock_namespace)
        previous_identity = node_identities.setdefault(node_id, identity)
        if previous_identity != identity:
            raise CiError(f"node {node_id} reports inconsistent hostname or lock namespace")

        previous_node = hostname_nodes.setdefault(hostname, node_id)
        if previous_node != node_id:
            raise CiError(f"hostname {hostname} maps to multiple node IDs")

        previous_node = gpu_nodes.setdefault(gpu_uuid, node_id)
        if previous_node != node_id:
            raise CiError(f"GPU UUID {gpu_uuid} maps to multiple node IDs")

        previous_uuid = node_index_uuids.setdefault((node_id, gpu_index), gpu_uuid)
        if previous_uuid != gpu_uuid:
            raise CiError(f"node {node_id} GPU index {gpu_index} maps to multiple UUIDs")
        previous_index = node_uuid_indices.setdefault((node_id, gpu_uuid), gpu_index)
        if previous_index != gpu_index:
            raise CiError(f"node {node_id} GPU UUID {gpu_uuid} maps to multiple indices")


def _cohort_node_summaries(
    receipts: Sequence[Mapping[str, Any]],
    *,
    require_full_slot_coverage: bool,
) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for node_id in sorted({str(receipt["node_id"]) for receipt in receipts}):
        node_receipts = [receipt for receipt in receipts if receipt["node_id"] == node_id]
        slots_values = {receipt["gpu_slots_per_device"] for receipt in node_receipts}
        namespaces = {receipt["lock_namespace"] for receipt in node_receipts}
        if len(slots_values) != 1 or len(namespaces) != 1:
            raise CiError(f"node {node_id} reports inconsistent slot or lock policy")
        slots_per_device = next(iter(slots_values))
        gpu_uuids = sorted({str(receipt["gpu_uuid"]) for receipt in node_receipts})
        observed_by_gpu: dict[str, list[int]] = {}
        for gpu_uuid in gpu_uuids:
            observed_slots = sorted(
                {
                    int(receipt["gpu_slot"])
                    for receipt in node_receipts
                    if receipt["gpu_uuid"] == gpu_uuid
                }
            )
            if require_full_slot_coverage and observed_slots != list(range(slots_per_device)):
                raise CiError(f"node {node_id} GPU {gpu_uuid} did not expose every configured slot")
            observed_by_gpu[gpu_uuid] = observed_slots
        if require_full_slot_coverage:
            capacity = len(gpu_uuids) * slots_per_device
            if capacity != len(node_receipts):
                raise CiError(f"node {node_id} observed capacity is not unique GPUs x slots")
        else:
            capacity = len(node_receipts)
        summaries[node_id] = {
            "capacity": capacity,
            "gpu_count": len(gpu_uuids),
            "gpu_uuids": gpu_uuids,
            "observed_slots": observed_by_gpu,
            "slots_per_gpu": slots_per_device,
            "runner_count": len({item["runner_name"] for item in node_receipts}),
            "lock_namespace": next(iter(namespaces)),
        }
    return summaries


def _topology_nodes(
    topology: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    nodes: dict[str, dict[str, object]] = {}
    for raw_node in topology["nodes"]:  # type: ignore[index]
        assert isinstance(raw_node, Mapping)
        node_label = str(raw_node["node_label"])
        node_id = node_label.removeprefix(_NODE_LABEL_PREFIX)
        nodes[node_id] = dict(raw_node)
    return nodes


def _validate_topology_member(
    receipt: Mapping[str, Any],
    *,
    expected_nodes: Mapping[str, Mapping[str, object]],
    slots_per_gpu: int,
) -> None:
    node_id = str(receipt["node_id"])
    if node_id not in expected_nodes:
        raise CiError(f"capacity receipt node {node_id} is outside the topology")
    expected_node = expected_nodes[node_id]
    allowed_indices = set(expected_node["gpu_indices"])  # type: ignore[arg-type]
    if receipt["gpu_slots_per_device"] != slots_per_gpu:
        raise CiError(f"node {node_id} receipt slots-per-GPU does not match the topology contract")
    if receipt["_gpu_index_int"] not in allowed_indices:
        raise CiError(
            f"node {node_id} GPU index {receipt['_gpu_index_int']} is not allowed "
            "by the topology contract"
        )


def _validate_topology_placement(
    receipts: Sequence[Mapping[str, Any]],
    *,
    topology: Mapping[str, object],
    require_exact: bool,
) -> dict[str, dict[str, object]]:
    """Bind observed leases to the declared node labels and GPU indices."""
    expected_nodes = _topology_nodes(topology)
    slots_per_gpu = int(topology["slots_per_gpu"])
    observed_node_ids = {str(receipt["node_id"]) for receipt in receipts}
    unexpected_nodes = sorted(observed_node_ids - set(expected_nodes))
    if unexpected_nodes:
        raise CiError(f"capacity receipts contain nodes outside the topology: {unexpected_nodes}")
    if require_exact and observed_node_ids != set(expected_nodes):
        raise CiError("capacity receipts do not cover the exact topology node set")

    for receipt in receipts:
        _validate_topology_member(
            receipt,
            expected_nodes=expected_nodes,
            slots_per_gpu=slots_per_gpu,
        )

    summaries = _cohort_node_summaries(
        receipts,
        require_full_slot_coverage=require_exact,
    )
    for node_id, summary in summaries.items():
        expected_node = expected_nodes[node_id]
        expected_indices = list(expected_node["gpu_indices"])  # type: ignore[arg-type]
        observed_indices = sorted(
            {int(receipt["_gpu_index_int"]) for receipt in receipts if receipt["node_id"] == node_id}
        )
        summary.update(
            {
                "node_label": expected_node["node_label"],
                "gpu_indices": observed_indices,
                "expected_gpu_count": len(expected_indices),
                "expected_capacity": len(expected_indices) * slots_per_gpu,
            }
        )
        if not require_exact:
            continue
        if observed_indices != expected_indices:
            raise CiError(f"node {node_id} did not expose the exact allowed GPU index set")
        if summary["gpu_count"] != len(expected_indices):
            raise CiError(f"node {node_id} did not expose the expected GPU count")
        expected_capacity = len(expected_indices) * slots_per_gpu
        if summary["capacity"] != expected_capacity:
            raise CiError(f"node {node_id} did not expose the expected capacity")
        if summary["runner_count"] != expected_capacity:
            raise CiError(f"node {node_id} did not use one runner per expected slot")
        node_receipts = [receipt for receipt in receipts if receipt["node_id"] == node_id]
        for gpu_index in expected_indices:
            index_receipts = [
                receipt
                for receipt in node_receipts
                if receipt["_gpu_index_int"] == gpu_index
            ]
            gpu_uuids = {str(receipt["gpu_uuid"]) for receipt in index_receipts}
            gpu_slots = {int(receipt["gpu_slot"]) for receipt in index_receipts}
            if len(gpu_uuids) != 1 or gpu_slots != set(range(slots_per_gpu)):
                raise CiError(
                    f"node {node_id} GPU index {gpu_index} does not map to one full GPU"
                )

    if require_exact:
        if len({str(receipt["gpu_uuid"]) for receipt in receipts}) != _topology_gpu_count(
            topology
        ):
            raise CiError("capacity receipts do not contain the exact expected GPU count")
        if len(receipts) != _topology_capacity(topology):
            raise CiError("capacity receipts do not contain the exact expected topology capacity")
    return summaries


def _validated_exact_cohort(
    values: Sequence[Mapping[str, object]],
    *,
    expected_slots: int,
    expected_topology: Mapping[str, object],
    expected_topology_digest: str,
    expected_run_id: str | None,
    expected_revision: str | None,
    expected_barrier_epoch: int | None,
    expected_cohort_id: str | None,
) -> tuple[list[dict[str, Any]], str, str, int, str]:
    expected = _positive_integer(expected_slots, "expected_slots", maximum=128)
    if len(values) != expected:
        raise CiError(f"capacity cohort requires exactly {expected} receipts, found {len(values)}")
    receipts = [
        _validated_receipt(
            value,
            expected_slots=expected,
            expected_topology_digest=expected_topology_digest,
        )
        for value in values
    ]
    if {receipt["leg_id"] for receipt in receipts} != set(range(expected)):
        raise CiError("capacity cohort receipts do not contain the exact matrix leg set")
    _validate_topology_identity(receipts)
    _validate_topology_placement(
        receipts,
        topology=expected_topology,
        require_exact=False,
    )

    run_ids = {str(receipt.get("run_id", "")) for receipt in receipts}
    revisions = {str(receipt.get("source_revision", "")) for receipt in receipts}
    barrier_epochs = {int(receipt["barrier_epoch"]) for receipt in receipts}
    cohort_ids = {
        _validated_cohort_id(receipt.get("cohort_id"), required=True) for receipt in receipts
    }
    if len(run_ids) != 1 or "" in run_ids:
        raise CiError("capacity cohort receipts do not share one non-empty run ID")
    if len(revisions) != 1 or "" in revisions:
        raise CiError("capacity cohort receipts do not share one non-empty source revision")
    if len(barrier_epochs) != 1:
        raise CiError("capacity cohort receipts do not share one absolute barrier epoch")
    if len(cohort_ids) != 1:
        raise CiError("capacity cohort receipts do not share one cohort ID")

    run_id = next(iter(run_ids))
    revision = next(iter(revisions))
    barrier_epoch = next(iter(barrier_epochs))
    cohort_id = next(iter(cohort_ids))
    assert cohort_id is not None
    if expected_run_id is not None and run_id != expected_run_id:
        raise CiError("capacity cohort run ID does not match the expected workflow run")
    if expected_revision is not None and revision != expected_revision:
        raise CiError("capacity cohort source revision does not match the expected workflow")
    if expected_barrier_epoch is not None and barrier_epoch != expected_barrier_epoch:
        raise CiError("capacity cohort barrier epoch does not match the expected workflow")
    if expected_cohort_id is not None:
        expected_cohort = _validated_cohort_id(expected_cohort_id, required=True)
        if cohort_id != expected_cohort:
            raise CiError("capacity cohort ID does not match the expected cohort")

    barrier_time = datetime.fromtimestamp(barrier_epoch, tz=timezone.utc)
    if any(receipt["_acquired"] >= barrier_time for receipt in receipts):
        raise CiError("a capacity cohort lease did not acquire before the absolute barrier")
    if any(receipt["_released"] < barrier_time for receipt in receipts):
        raise CiError("a capacity cohort lease released before the absolute barrier")
    slot_tuples = {_slot_tuple(receipt) for receipt in receipts}
    if len(slot_tuples) != expected:
        raise CiError("capacity cohort receipts contain duplicate GPU slot tuples")
    runner_names = {receipt["runner_name"] for receipt in receipts}
    if len(runner_names) != expected:
        raise CiError("capacity cohort receipts do not use unique runner listeners")
    maximum = _maximum_concurrency(receipts)
    if maximum != expected:
        raise CiError(
            f"capacity cohort observed maximum concurrency {maximum}, expected {expected}"
        )
    _cohort_node_summaries(receipts, require_full_slot_coverage=False)
    return receipts, run_id, revision, barrier_epoch, cohort_id


def verify_cohort_receipts(
    values: Sequence[Mapping[str, object]],
    *,
    expected_slots: int,
    expected_topology: Mapping[str, object],
    expected_run_id: str | None = None,
    expected_revision: str | None = None,
    expected_barrier_epoch: int | None = None,
    expected_cohort_id: str | None = None,
) -> dict[str, object]:
    """Verify one exact shared cohort held every requested lease at its barrier."""
    topology = _validated_topology_for_mode(
        expected_topology,
        mode="shared-cohort",
        requested_slots=expected_slots,
    )
    topology_digest = topology_contract_digest(topology)
    receipts, run_id, revision, barrier_epoch, cohort_id = _validated_exact_cohort(
        values,
        expected_slots=expected_slots,
        expected_topology=topology,
        expected_topology_digest=topology_digest,
        expected_run_id=expected_run_id,
        expected_revision=expected_revision,
        expected_barrier_epoch=expected_barrier_epoch,
        expected_cohort_id=expected_cohort_id,
    )
    expected = _positive_integer(expected_slots, "expected_slots", maximum=128)
    return {
        "schema_version": 1,
        "kind": "trtmc_capacity_cohort_verification",
        "outcome": "success",
        "cohort_id": cohort_id,
        "run_id": run_id,
        "source_revision": revision,
        "expected_slots": expected,
        "expected_topology_digest": topology_digest,
        "expected_topology": topology,
        "barrier_epoch": barrier_epoch,
        "barrier_at": datetime.fromtimestamp(barrier_epoch, tz=timezone.utc).isoformat(),
        "receipt_count": len(receipts),
        "maximum_concurrency": _maximum_concurrency(receipts),
        "runner_count": len({receipt["runner_name"] for receipt in receipts}),
        "slot_count": len({_slot_tuple(receipt) for receipt in receipts}),
        "nodes": _validate_topology_placement(
            receipts,
            topology=topology,
            require_exact=False,
        ),
    }


def _validated_source_run_metadata(
    payloads: Sequence[Mapping[str, object]],
    *,
    expected_run_ids: Sequence[str],
    expected_repository: str,
    expected_revision: str,
) -> list[dict[str, object]]:
    """Authenticate the two source runs whose artifacts are being combined."""
    run_ids = list(expected_run_ids)
    if len(payloads) != 2:
        raise CiError("cross-workflow verification requires exactly two source-run records")
    if not expected_repository or expected_repository.count("/") != 1:
        raise CiError("cross-workflow expected repository is invalid")
    if not expected_revision:
        raise CiError("cross-workflow expected revision is empty")

    validated: list[dict[str, object]] = []
    observed_ids: set[str] = set()
    for payload in payloads:
        run_id_value = payload.get("id")
        if isinstance(run_id_value, bool) or not isinstance(run_id_value, int):
            raise CiError("cross-workflow source run has an invalid run ID")
        run_id = str(run_id_value)
        observed_ids.add(run_id)

        repository = payload.get("repository")
        head_repository = payload.get("head_repository")
        if (
            not isinstance(repository, Mapping)
            or repository.get("full_name") != expected_repository
        ):
            raise CiError("cross-workflow source run is from the wrong repository")
        if (
            not isinstance(head_repository, Mapping)
            or head_repository.get("full_name") != expected_repository
        ):
            raise CiError("cross-workflow source run has the wrong head repository")
        run_attempt = payload.get("run_attempt")
        if isinstance(run_attempt, bool) or not isinstance(run_attempt, int) or run_attempt != 1:
            raise CiError("cross-workflow source run has the wrong run attempt")
        workflow_path = payload.get("path")
        # GitHub has emitted both the unqualified path and its @main form.
        if workflow_path not in {
            _CAPACITY_WORKFLOW_PATH,
            f"{_CAPACITY_WORKFLOW_PATH}@main",
        }:
            raise CiError("cross-workflow source run has the wrong workflow path")
        checks = (
            ("event", "workflow_dispatch", "event"),
            ("head_branch", "main", "branch"),
            ("head_sha", expected_revision, "revision"),
            ("status", "completed", "status"),
            ("conclusion", "success", "conclusion"),
        )
        for field, expected, description in checks:
            if payload.get(field) != expected:
                raise CiError(f"cross-workflow source run has the wrong {description}")
        validated.append(
            {
                "id": run_id_value,
                "repository": expected_repository,
                "head_repository": expected_repository,
                "path": workflow_path,
                "event": "workflow_dispatch",
                "head_branch": "main",
                "head_sha": expected_revision,
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
            }
        )

    if observed_ids != set(run_ids):
        raise CiError("cross-workflow source-run records do not match the requested run IDs")
    return sorted(validated, key=lambda value: int(value["id"]))


def verify_cross_workflow_receipts(
    values: Sequence[Mapping[str, object]],
    *,
    expected_slots_per_run: int,
    expected_topology: Mapping[str, object],
    expected_run_ids: Sequence[str],
    run_metadata: Sequence[Mapping[str, object]],
    expected_repository: str,
    expected_revision: str,
    expected_barrier_epoch: int | None = None,
    expected_cohort_id: str | None = None,
) -> dict[str, object]:
    """Verify two exact cohorts share one combined GPU-slot pool at one barrier."""
    expected = _positive_integer(expected_slots_per_run, "expected_slots_per_run", maximum=128)
    topology = _validated_topology_for_mode(
        expected_topology,
        mode="cross-workflow-verify",
        requested_slots=expected,
    )
    topology_digest = topology_contract_digest(topology)
    run_ids = list(expected_run_ids)
    if (
        len(run_ids) != 2
        or len(set(run_ids)) != 2
        or any(not run_id.isdigit() or int(run_id) < 1 for run_id in run_ids)
    ):
        raise CiError("cross-workflow verification requires exactly two distinct run IDs")
    source_runs = _validated_source_run_metadata(
        run_metadata,
        expected_run_ids=run_ids,
        expected_repository=expected_repository,
        expected_revision=expected_revision,
    )
    if len(values) != expected * 2:
        raise CiError(
            f"cross-workflow verification requires exactly {expected * 2} receipts, "
            f"found {len(values)}"
        )

    all_receipts: list[dict[str, Any]] = []
    identities: list[tuple[str, str, int, str]] = []
    run_summaries: dict[str, dict[str, object]] = {}
    for run_id in run_ids:
        run_values = [value for value in values if str(value.get("run_id", "")) == run_id]
        receipts, actual_run_id, revision, barrier_epoch, cohort_id = _validated_exact_cohort(
            run_values,
            expected_slots=expected,
            expected_topology=topology,
            expected_topology_digest=topology_digest,
            expected_run_id=run_id,
            expected_revision=expected_revision,
            expected_barrier_epoch=expected_barrier_epoch,
            expected_cohort_id=expected_cohort_id,
        )
        all_receipts.extend(receipts)
        identities.append((actual_run_id, revision, barrier_epoch, cohort_id))
        run_summaries[run_id] = {
            "receipt_count": len(receipts),
            "maximum_concurrency": _maximum_concurrency(receipts),
            "runner_count": len({receipt["runner_name"] for receipt in receipts}),
            "slot_count": len({_slot_tuple(receipt) for receipt in receipts}),
        }

    observed_run_ids = {str(value.get("run_id", "")) for value in values}
    if observed_run_ids != set(run_ids):
        raise CiError("cross-workflow receipts contain an unexpected run ID")
    revisions = {identity[1] for identity in identities}
    barrier_epochs = {identity[2] for identity in identities}
    cohort_ids = {identity[3] for identity in identities}
    if len(revisions) != 1:
        raise CiError("cross-workflow cohorts do not share one source revision")
    if len(barrier_epochs) != 1:
        raise CiError("cross-workflow cohorts do not share one absolute barrier epoch")
    if len(cohort_ids) != 1:
        raise CiError("cross-workflow cohorts do not share one cohort ID")
    _validate_topology_identity(all_receipts)

    combined_expected = expected * 2
    combined_tuples = {_slot_tuple(receipt) for receipt in all_receipts}
    if len(combined_tuples) != combined_expected:
        raise CiError("cross-workflow cohorts contain duplicate GPU slot tuples")
    combined_runners = {receipt["runner_name"] for receipt in all_receipts}
    if len(combined_runners) != combined_expected:
        raise CiError("cross-workflow cohorts do not use unique runner listeners")
    maximum = _maximum_concurrency(all_receipts)
    if maximum != combined_expected:
        raise CiError(
            f"cross-workflow cohorts observed maximum concurrency {maximum}, "
            f"expected {combined_expected}"
        )
    nodes = _validate_topology_placement(
        all_receipts,
        topology=topology,
        require_exact=True,
    )
    if sum(int(node["capacity"]) for node in nodes.values()) != combined_expected:
        raise CiError("cross-workflow per-node capacities do not sum to the combined cohort")

    barrier_epoch = next(iter(barrier_epochs))
    return {
        "schema_version": 1,
        "kind": "trtmc_cross_workflow_capacity_verification",
        "outcome": "success",
        "cohort_id": next(iter(cohort_ids)),
        "run_ids": run_ids,
        "source_revision": next(iter(revisions)),
        "expected_slots_per_run": expected,
        "combined_expected_slots": combined_expected,
        "expected_topology_digest": topology_digest,
        "expected_topology": topology,
        "barrier_epoch": barrier_epoch,
        "barrier_at": datetime.fromtimestamp(barrier_epoch, tz=timezone.utc).isoformat(),
        "receipt_count": len(all_receipts),
        "maximum_concurrency": maximum,
        "runner_count": len(combined_runners),
        "slot_count": len(combined_tuples),
        "runs": run_summaries,
        "source_runs": source_runs,
        "nodes": nodes,
    }


def verify_capacity_receipts(
    values: Sequence[Mapping[str, object]],
    *,
    expected_slots: int,
    expected_topology: Mapping[str, object],
    expected_run_id: str | None = None,
    expected_revision: str | None = None,
    expected_barrier_epoch: int | None = None,
) -> dict[str, object]:
    """Verify the full first wave, the queued extra leg, and dynamic node capacity."""
    expected = _positive_integer(expected_slots, "expected_slots", maximum=128)
    topology = _validated_topology_for_mode(
        expected_topology,
        mode="shared-capacity",
        requested_slots=expected,
    )
    topology_digest = topology_contract_digest(topology)
    if len(values) != expected + 1:
        raise CiError(f"capacity canary requires {expected + 1} receipts, found {len(values)}")
    receipts = [
        _validated_receipt(
            value,
            expected_slots=expected,
            expected_topology_digest=topology_digest,
        )
        for value in values
    ]
    if {receipt["leg_id"] for receipt in receipts} != set(range(expected + 1)):
        raise CiError("capacity receipts do not contain the exact matrix leg set")
    _validate_topology_identity(receipts)
    _validate_topology_placement(
        receipts,
        topology=topology,
        require_exact=False,
    )
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

    node_summaries = _validate_topology_placement(
        first_wave,
        topology=topology,
        require_exact=True,
    )
    if sum(int(node["capacity"]) for node in node_summaries.values()) != expected:
        raise CiError("per-node observed capacities do not sum to expected_slots")

    return {
        "schema_version": 1,
        "kind": "trtmc_capacity_canary_verification",
        "outcome": "success",
        "run_id": next(iter(run_ids)),
        "source_revision": next(iter(revisions)),
        "expected_slots": expected,
        "expected_topology_digest": topology_digest,
        "expected_topology": topology,
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
    if value.get("job_id") != _CAPACITY_WORKER_JOB_ID:
        raise CiError(f"{label} job_id does not match the canary worker job")
    node_id = value.get("node_id")
    if not isinstance(node_id, str) or _SAFE_ID.fullmatch(node_id) is None:
        raise CiError(f"{label} node_id is unsafe")
    hostname = value.get("hostname")
    if not isinstance(hostname, str) or _SAFE_ID.fullmatch(hostname) is None:
        raise CiError(f"{label} hostname is unsafe")
    runner_name = value.get("runner_name")
    if not isinstance(runner_name, str) or not runner_name:
        raise CiError(f"{label} runner_name must not be empty")
    gpu_index = value.get("gpu_index")
    if (
        not isinstance(gpu_index, str)
        or not gpu_index.isdigit()
        or str(int(gpu_index)) != gpu_index
        or not 0 <= int(gpu_index) <= 63
    ):
        raise CiError(f"{label} gpu_index must be a canonical integer from 0 to 63")
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
        "hostname": hostname,
        "runner_name": runner_name,
        "gpu_index": gpu_index,
        "_gpu_index_int": int(gpu_index),
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
    expected_topology: Mapping[str, object],
    expected_run_id: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, object]:
    """Verify same-node, same-GPU serialization for two exclusive attempts."""
    if len(values) != 1:
        raise CiError(
            "exclusive-safety mode requires exactly one scheduler-selected runner receipt"
        )
    topology = normalize_topology_contract(expected_topology)
    topology_digest = topology_contract_digest(topology)
    receipt = values[0]
    if receipt.get("schema_version") != 2 or receipt.get("kind") != "trtmc_exclusive_safety":
        raise CiError("exclusive-safety receipt schema or kind is unsupported")
    receipt_digest = _validated_digest(
        receipt.get("expected_topology_digest"),
        "exclusive-safety expected_topology_digest",
    )
    if receipt_digest != topology_digest:
        raise CiError("exclusive-safety topology digest does not match the trusted contract")
    primary_value = receipt.get("primary")
    contender_value = receipt.get("contender")
    if not isinstance(primary_value, dict) or not isinstance(contender_value, dict):
        raise CiError("exclusive-safety receipt must contain two lease records")
    primary = _validated_exclusive_lease(primary_value, label="primary")
    contender = _validated_exclusive_lease(contender_value, label="contender")
    _validate_topology_identity([primary, contender])
    expected_nodes = _topology_nodes(topology)
    for lease in (primary, contender):
        _validate_topology_member(
            lease,
            expected_nodes=expected_nodes,
            slots_per_gpu=int(topology["slots_per_gpu"]),
        )
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
        "expected_topology_digest": topology_digest,
        "expected_topology": topology,
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


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CiError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise CiError(f"{label} must be a JSON object: {path}")
    return value


def _append_github_outputs(path: Path, values: Mapping[str, object]) -> None:
    lines: list[str] = []
    for name, value in values.items():
        rendered = str(value)
        if "\n" in rendered or "\r" in rendered:
            raise CiError(f"GitHub output {name} must be one line")
        lines.append(f"{name}={rendered}\n")
    with path.open("a", encoding="utf-8") as output:
        output.writelines(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m tools.ci.capacity_canary")
    commands = parser.add_subparsers(dest="command", required=True)
    topology = commands.add_parser(
        "topology-contract",
        help="Validate and normalize a trusted expected-topology contract",
    )
    topology.add_argument("--input", required=True, type=Path)
    topology.add_argument(
        "--mode",
        required=True,
        choices=(
            "shared-capacity",
            "rollback-capacity",
            "shared-cohort",
            "cross-workflow-verify",
            "exclusive-safety",
        ),
    )
    topology.add_argument("--requested-slots", required=True)
    topology.add_argument("--output", type=Path)
    topology.add_argument("--github-output", type=Path)
    warm_matrix = commands.add_parser(
        "cache-warm-matrix",
        help="Emit one cache-warm matrix row per declared physical node",
    )
    warm_matrix.add_argument("--input", required=True, type=Path)
    warm_matrix.add_argument("--github-output", type=Path)
    matrix = commands.add_parser("matrix", help="Emit the expected+1 job matrix")
    matrix.add_argument("--expected-slots", required=True)
    cohort = commands.add_parser("cohort-matrix", help="Emit an exact-size cohort matrix")
    cohort.add_argument("--cohort-slots", required=True)
    hold = commands.add_parser("hold", help="Acquire and hold one slot to an absolute epoch")
    hold.add_argument("--leg-id", required=True, type=int)
    hold.add_argument("--expected-slots", required=True)
    hold.add_argument("--barrier-epoch", required=True)
    hold.add_argument("--expected-topology-digest", required=True)
    hold.add_argument("--cohort-id")
    hold.add_argument("--receipt-output", required=True, type=Path)
    exclusive = commands.add_parser(
        "exclusive-safety", help="Prove same-GPU exclusive serialization"
    )
    exclusive.add_argument("--leg-id", required=True, type=int)
    exclusive.add_argument("--observation-seconds", default="3")
    exclusive.add_argument("--expected-topology-digest", required=True)
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
    verify.add_argument("--expected-topology", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    verify_cohort = commands.add_parser("verify-cohort", help="Verify one exact shared cohort")
    verify_cohort.add_argument("--receipts-dir", required=True, type=Path)
    verify_cohort.add_argument("--expected-slots", required=True)
    verify_cohort.add_argument("--expected-run-id")
    verify_cohort.add_argument("--expected-revision")
    verify_cohort.add_argument("--expected-barrier-epoch", type=int)
    verify_cohort.add_argument("--expected-cohort-id")
    verify_cohort.add_argument("--expected-topology", required=True, type=Path)
    verify_cohort.add_argument("--output", required=True, type=Path)
    verify_cross = commands.add_parser(
        "verify-cross", help="Verify two exact shared cohorts at one barrier"
    )
    verify_cross.add_argument("--receipts-dir", required=True, type=Path)
    verify_cross.add_argument("--expected-slots-per-run", required=True)
    verify_cross.add_argument("--expected-run-id", action="append", required=True)
    verify_cross.add_argument("--run-metadata", action="append", required=True, type=Path)
    verify_cross.add_argument("--expected-repository", required=True)
    verify_cross.add_argument("--expected-revision", required=True)
    verify_cross.add_argument("--expected-barrier-epoch", type=int)
    verify_cross.add_argument("--expected-cohort-id")
    verify_cross.add_argument("--expected-topology", required=True, type=Path)
    verify_cross.add_argument("--output", required=True, type=Path)
    verify_exclusive = commands.add_parser(
        "verify-exclusive", help="Verify exclusive-safety evidence"
    )
    verify_exclusive.add_argument("--receipts-dir", required=True, type=Path)
    verify_exclusive.add_argument("--expected-run-id")
    verify_exclusive.add_argument("--expected-revision")
    verify_exclusive.add_argument("--expected-topology", required=True, type=Path)
    verify_exclusive.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "topology-contract":
            topology = _validated_topology_for_mode(
                _load_json_object(arguments.input, "topology contract"),
                mode=arguments.mode,
                requested_slots=_positive_integer(
                    arguments.requested_slots,
                    "expected_slots",
                    maximum=128,
                ),
            )
            digest = topology_contract_digest(topology)
            compact = json.dumps(topology, sort_keys=True, separators=(",", ":"))
            if arguments.output is not None:
                _atomic_json(arguments.output, topology)
            if arguments.github_output is not None:
                _append_github_outputs(
                    arguments.github_output,
                    {
                        "expected_topology_json": compact,
                        "expected_topology_digest": digest,
                    },
                )
            print(json.dumps(topology, indent=2, sort_keys=True))
        elif arguments.command == "cache-warm-matrix":
            matrix = cache_warm_matrix(
                _load_json_object(arguments.input, "pool topology contract")
            )
            compact = json.dumps(matrix, sort_keys=True, separators=(",", ":"))
            if arguments.github_output is not None:
                _append_github_outputs(arguments.github_output, {"matrix": compact})
            print(compact)
        elif arguments.command == "matrix":
            print(
                json.dumps(
                    capacity_matrix(
                        _positive_integer(arguments.expected_slots, "expected_slots", maximum=128)
                    ),
                    separators=(",", ":"),
                )
            )
        elif arguments.command == "cohort-matrix":
            print(
                json.dumps(
                    cohort_matrix(
                        _positive_integer(arguments.cohort_slots, "cohort_slots", maximum=128)
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
                expected_topology_digest=arguments.expected_topology_digest,
                cohort_id=arguments.cohort_id,
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
                expected_topology_digest=arguments.expected_topology_digest,
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
                expected_topology=_load_json_object(
                    arguments.expected_topology, "topology contract"
                ),
                expected_run_id=arguments.expected_run_id,
                expected_revision=arguments.expected_revision,
                expected_barrier_epoch=arguments.expected_barrier_epoch,
            )
            _atomic_json(arguments.output, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif arguments.command == "verify-cohort":
            summary = verify_cohort_receipts(
                load_receipts(arguments.receipts_dir),
                expected_slots=_positive_integer(
                    arguments.expected_slots, "expected_slots", maximum=128
                ),
                expected_topology=_load_json_object(
                    arguments.expected_topology, "topology contract"
                ),
                expected_run_id=arguments.expected_run_id,
                expected_revision=arguments.expected_revision,
                expected_barrier_epoch=arguments.expected_barrier_epoch,
                expected_cohort_id=arguments.expected_cohort_id,
            )
            _atomic_json(arguments.output, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif arguments.command == "verify-cross":
            summary = verify_cross_workflow_receipts(
                load_receipts(arguments.receipts_dir),
                expected_slots_per_run=_positive_integer(
                    arguments.expected_slots_per_run,
                    "expected_slots_per_run",
                    maximum=128,
                ),
                expected_topology=_load_json_object(
                    arguments.expected_topology, "topology contract"
                ),
                expected_run_ids=arguments.expected_run_id,
                run_metadata=[
                    _load_json_object(path, "source-run metadata")
                    for path in arguments.run_metadata
                ],
                expected_repository=arguments.expected_repository,
                expected_revision=arguments.expected_revision,
                expected_barrier_epoch=arguments.expected_barrier_epoch,
                expected_cohort_id=arguments.expected_cohort_id,
            )
            _atomic_json(arguments.output, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif arguments.command == "verify-exclusive":
            summary = verify_exclusive_safety_receipts(
                load_receipts(arguments.receipts_dir),
                expected_topology=_load_json_object(
                    arguments.expected_topology, "topology contract"
                ),
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
