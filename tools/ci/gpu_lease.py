# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocate shared slots or whole GPUs to concurrent isolated model proofs.

Boundary: cross-process fairness and locking only; this module never runs a model.
"""

from __future__ import annotations

from collections.abc import Callable
import fcntl
import hashlib
import os
import re
import time
from pathlib import Path

from .context import CiContext
from .process import CiError


class FileLock:
    """An open file descriptor whose flock lifetime is explicit."""

    def __init__(self, path: Path, *, mode: str = "a+"):
        self.path = path
        self.handle = path.open(mode, encoding="utf-8")

    def try_lock(self, *, shared: bool = False) -> bool:
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(self.handle.fileno(), operation | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def unlock(self) -> None:
        if not self.handle.closed:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)

    def close(self) -> None:
        if not self.handle.closed:
            self.unlock()
            self.handle.close()


class GpuLease:
    """Hold one shared slot or every slot of one GPU until explicitly released."""

    LOCK_FILE_PATTERNS = (
        "allocator.lock",
        "admission-{scope}.enqueue.lock",
        "admission-{scope}.next",
        "admission-{scope}-{ticket:020d}.lock",
        "gpu-{gpu}-reservation.lock",
        "gpu-{gpu}-slot-{slot}.lock",
        "whole-machine.lock",
    )

    def __init__(
        self,
        context: CiContext,
        model: str,
        resource_class: str,
        artifacts: Path | None = None,
        *,
        min_free_gpu_memory_mib: int = 0,
    ):
        if resource_class not in {"shared", "exclusive_gpu"}:
            raise CiError("model-proof resource class must be shared or exclusive_gpu")
        if (
            isinstance(min_free_gpu_memory_mib, bool)
            or not isinstance(min_free_gpu_memory_mib, int)
            or min_free_gpu_memory_mib < 0
        ):
            raise CiError("minimum free GPU memory must be a non-negative integer MiB value")
        if min_free_gpu_memory_mib and resource_class != "exclusive_gpu":
            raise CiError("minimum free GPU memory can only be required by exclusive_gpu proofs")
        self.context = context
        self.model = model
        self.resource_class = resource_class
        self.artifacts = artifacts
        self.min_free_gpu_memory_mib = min_free_gpu_memory_mib
        self.gpu_ids, self.slots_per_gpu, self.timeout, self.poll_interval = self._configuration()
        configured = context.env.get(
            "TRTMC_MODEL_PROOF_GPU_LOCK_DIR", "/tmp/trtmc-model-proof-gpu-locks"
        )
        if not configured:
            raise CiError("TRTMC_MODEL_PROOF_GPU_LOCK_DIR must not be empty")
        self.lock_dir = Path(configured)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir = self.lock_dir.resolve(strict=True)
        stat = self.lock_dir.stat()
        identity = f"{self.lock_dir}\0{stat.st_dev}\0{stat.st_ino}".encode()
        self.lock_namespace = hashlib.sha256(identity).hexdigest()
        self.machine: FileLock | None = None
        self.ticket: FileLock | None = None
        self.reservation: FileLock | None = None
        self.slots: list[FileLock] = []
        self.gpu_id: int | None = None
        self.slot_ids: list[int] = []
        self.gpu_memory_admission: dict[str, object] | None = None
        self.last_observed_free_mib: dict[int, int] = {}
        self.last_observed_total_mib: dict[int, int] = {}
        self.gpu_numa_nodes: dict[str, int] = {}
        self.numa_fallback_warnings: set[str] = set()

    def acquire(
        self,
        *,
        prepare_candidate: Callable[[], None] | None = None,
    ) -> "GpuLease":
        deadline = time.monotonic() + self.timeout
        capacity_rejected: set[int] = set()
        self.machine = FileLock(self.lock_dir / "whole-machine.lock")
        if not self._wait_lock(self.machine, deadline, shared=True):
            raise CiError(f"timed out after {self.timeout}s waiting for the whole-machine GPU lock")
        print(f"Acquired shared whole-machine GPU lock via {self.machine.path}")
        try:
            self.ticket = self._create_ticket("global", deadline)
            if self.artifacts:
                self.artifacts.mkdir(parents=True, exist_ok=True)
                (self.artifacts / "gpu-queue-joined.txt").write_text(
                    self.ticket.path.name + "\n", encoding="utf-8"
                )
            self._wait_for_queue_head("global", deadline)
            while time.monotonic() < deadline:
                if self.resource_class == "shared":
                    if self._try_shared(deadline):
                        self._release_ticket()
                        print(
                            f"Leased shared model-proof GPU {self.gpu_id} slot {self.slot_ids[0]} "
                            f"via {self.slots[0].path}"
                        )
                        return self
                elif len(self.gpu_ids) == 1 and not self.min_free_gpu_memory_mib:
                    if self._reserve_one_gpu(deadline):
                        self._release_ticket()
                        self._drain_reserved_gpu(deadline)
                        print(
                            f"Leased exclusive model-proof GPU {self.gpu_id} slots "
                            f"{' '.join(map(str, self.slot_ids))}"
                        )
                        return self
                elif self.min_free_gpu_memory_mib:
                    if self._reserve_capacity_candidate(deadline, exclude=capacity_rejected):
                        candidate_gpu = self.gpu_id
                        assert candidate_gpu is not None
                        self._release_ticket()
                        self._drain_reserved_gpu(deadline)
                        if prepare_candidate:
                            prepare_candidate()
                        candidates_remaining = len(self.gpu_ids) - len(capacity_rejected)
                        if self._candidate_has_capacity(
                            deadline,
                            candidates_remaining=candidates_remaining,
                        ):
                            print(
                                f"Leased exclusive model-proof GPU {self.gpu_id} slots "
                                f"{' '.join(map(str, self.slot_ids))}"
                            )
                            return self
                        capacity_rejected.add(candidate_gpu)
                        self._release_gpu()
                        if len(capacity_rejected) == len(self.gpu_ids):
                            if all(
                                gpu in self.last_observed_total_mib
                                and self.last_observed_total_mib[gpu] < self.min_free_gpu_memory_mib
                                for gpu in self.gpu_ids
                            ):
                                totals = ", ".join(
                                    f"GPU {gpu}={self.last_observed_total_mib[gpu]} MiB"
                                    for gpu in self.gpu_ids
                                )
                                raise CiError(
                                    "configured GPUs cannot meet the model-proof minimum free "
                                    f"memory requirement of {self.min_free_gpu_memory_mib} MiB "
                                    f"(total memory: {totals})"
                                )
                            capacity_rejected.clear()
                        self._requeue_after_capacity_rejection(deadline)
                        continue
                    elif capacity_rejected:
                        self._requeue_after_capacity_rejection(deadline)
                        capacity_rejected.clear()
                        continue
                elif self._try_exclusive_any(deadline):
                    if prepare_candidate:
                        prepare_candidate()
                    self._release_ticket()
                    print(
                        f"Leased exclusive model-proof GPU {self.gpu_id} slots "
                        f"{' '.join(map(str, self.slot_ids))}"
                    )
                    return self
                time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))
            raise CiError(
                f"timed out after {self.timeout}s waiting for a {self.resource_class} "
                f"model-proof GPU lease from: {' '.join(map(str, self.gpu_ids))}"
                f"{self._capacity_timeout_detail()}"
            )
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        self._release_ticket()
        self._release_gpu()
        if self.machine:
            self.machine.close()
            self.machine = None

    def _release_gpu(self) -> None:
        for lock in self.slots:
            lock.close()
        self.slots.clear()
        self.slot_ids.clear()
        if self.reservation:
            self.reservation.close()
            self.reservation = None
        self.gpu_id = None
        self.gpu_memory_admission = None

    def evidence(self, revision: str) -> dict[str, object]:
        if self.gpu_id is None or not self.slot_ids:
            raise CiError("GPU lease evidence requested before acquisition")
        evidence: dict[str, object] = {
            "schema_version": 1,
            "model": self.model,
            "source_revision": revision,
            "gpu_id": str(self.gpu_id),
            "gpu_slot": self.slot_ids[0] if self.resource_class == "shared" else None,
            "gpu_slots": self.slot_ids,
            "gpu_slot_ids": self.slot_ids,
            "slots_per_gpu": self.slots_per_gpu,
            "gpu_slots_per_device": self.slots_per_gpu,
            "resource_class": self.resource_class,
            "gpu_resource_class": self.resource_class,
            "min_free_gpu_memory_mib": self.min_free_gpu_memory_mib,
        }
        if self.min_free_gpu_memory_mib:
            if not self.gpu_memory_admission:
                raise CiError("GPU memory admission evidence requested before capacity validation")
            evidence["gpu_memory_admission"] = self.gpu_memory_admission
        return evidence

    def _configuration(self) -> tuple[list[int], int, int, float]:
        configured = self.context.env.get("TRTMC_MODEL_PROOF_GPU_IDS", "0,1,2,3")
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", configured):
            raise CiError(
                "TRTMC_MODEL_PROOF_GPU_IDS must be a comma-separated list of unique "
                "non-negative integers"
            )
        gpu_ids = [int(value) for value in configured.split(",")]
        if len(gpu_ids) != len(set(gpu_ids)):
            duplicate = next(value for value in gpu_ids if gpu_ids.count(value) > 1)
            raise CiError(f"TRTMC_MODEL_PROOF_GPU_IDS contains duplicate GPU ID: {duplicate}")
        if "TRTMC_GPU_ID" in self.context.env:
            explicit = self.context.env["TRTMC_GPU_ID"]
            if not re.fullmatch(r"0|[1-9][0-9]*", explicit):
                raise CiError("TRTMC_GPU_ID must be a non-negative integer")
            if int(explicit) not in gpu_ids:
                raise CiError("TRTMC_GPU_ID must be present in TRTMC_MODEL_PROOF_GPU_IDS")
            gpu_ids = [int(explicit)]
        slots_text = self.context.env.get("TRTMC_MODEL_PROOF_SLOTS_PER_GPU", "4")
        if not slots_text.isdigit() or not 1 <= int(slots_text) <= 16:
            raise CiError("TRTMC_MODEL_PROOF_SLOTS_PER_GPU must be an integer from 1 to 16")
        slots = int(slots_text)
        explicit_slot = self.context.env.get("TRTMC_GPU_SLOT_ID", "")
        if explicit_slot:
            if "TRTMC_GPU_ID" not in self.context.env:
                raise CiError("TRTMC_GPU_SLOT_ID requires TRTMC_GPU_ID")
            if not explicit_slot.isdigit() or not 0 <= int(explicit_slot) < slots:
                raise CiError(f"TRTMC_GPU_SLOT_ID must be an integer from 0 to {slots - 1}")
            if self.resource_class != "shared":
                raise CiError("TRTMC_GPU_SLOT_ID cannot be used with exclusive_gpu")
        timeout_text = self.context.env.get("TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS", "10800")
        if not timeout_text.isdigit() or not 1 <= int(timeout_text) <= 21600:
            raise CiError(
                "TRTMC_MODEL_PROOF_GPU_LEASE_TIMEOUT_SECONDS must be an integer from 1 to 21600"
            )
        poll_text = self.context.env.get("TRTMC_MODEL_PROOF_POLL_INTERVAL", "0.25")
        try:
            poll = float(poll_text)
        except ValueError as error:
            raise CiError(
                "TRTMC_MODEL_PROOF_POLL_INTERVAL must be a positive number no greater than "
                "21600 seconds"
            ) from error
        if not 0 < poll <= 21600 or len(poll_text.split(".", 1)[0]) > 5:
            raise CiError(
                "TRTMC_MODEL_PROOF_POLL_INTERVAL must be a positive number no greater than "
                "21600 seconds"
            )
        watchdog = self.context.env.get("TRTMC_MODEL_PROOF_FLOCK_WATCHDOG_SECONDS", "30")
        if not watchdog.isdigit() or not 1 <= int(watchdog) <= 21600 or len(watchdog) > 5:
            raise CiError(
                "TRTMC_MODEL_PROOF_FLOCK_WATCHDOG_SECONDS must be an integer from 1 to 21600"
            )
        return gpu_ids, slots, int(timeout_text), poll

    def _create_ticket(self, scope: str, deadline: float) -> FileLock:
        enqueue = FileLock(self.lock_dir / f"admission-{scope}.enqueue.lock")
        if not self._wait_lock(enqueue, deadline):
            enqueue.close()
            raise CiError("timed out waiting to create GPU admission ticket")
        try:
            counter_path = self.lock_dir / f"admission-{scope}.next"
            counter = 0
            if counter_path.exists():
                text = counter_path.read_text(encoding="utf-8").strip()
                if not text.isdigit():
                    raise CiError(f"invalid GPU admission counter in: {counter_path}")
                counter = int(text)
            existing = [
                int(match.group(1))
                for path in self.lock_dir.glob(f"admission-{scope}-*.lock")
                if (match := re.fullmatch(rf"admission-{scope}-([0-9]{{20}})\.lock", path.name))
            ]
            counter = max([counter, *existing]) + 1
            if counter >= 9_000_000_000_000_000_000:
                raise CiError(f"GPU admission ticket sequence exhausted for scope: {scope}")
            counter_tmp = counter_path.with_name(f"{counter_path.name}.tmp.{os.getpid()}")
            counter_tmp.write_text(f"{counter}\n", encoding="utf-8")
            counter_tmp.replace(counter_path)
            final = self.lock_dir / f"admission-{scope}-{counter:020d}.lock"
            temporary = final.with_name(f"{final.name}.tmp.{os.getpid()}")
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
            handle = os.fdopen(descriptor, "r+", encoding="utf-8")
            ticket = FileLock.__new__(FileLock)
            ticket.path = temporary
            ticket.handle = handle
            if not ticket.try_lock():
                raise CiError(f"could not lock new GPU admission ticket: {temporary}")
            ticket.handle.write(
                f"pid={os.getpid()} model={self.model} resource_class={self.resource_class} "
                f"queue_scope={scope}\n"
            )
            ticket.handle.flush()
            os.link(temporary, final)
            temporary.unlink()
            ticket.path = final
            return ticket
        finally:
            enqueue.close()

    def _wait_for_queue_head(self, scope: str, deadline: float) -> None:
        while time.monotonic() < deadline:
            assert self.ticket is not None
            allocator = self._allocator(deadline)
            try:
                for path in self.lock_dir.glob(f"admission-{scope}-*.lock.handoff.*"):
                    handoff = FileLock(path)
                    if handoff.try_lock():
                        path.unlink(missing_ok=True)
                        handoff.close()
                    else:
                        handoff.handle.close()
                older_live = False
                for path in sorted(self.lock_dir.glob(f"admission-{scope}-*.lock")):
                    if path == self.ticket.path:
                        break
                    candidate = FileLock(path, mode="r")
                    if candidate.try_lock():
                        candidate.close()
                        path.unlink(missing_ok=True)
                    else:
                        older_live = True
                        candidate.handle.close()
                if not older_live:
                    return
            finally:
                allocator.close()
            time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))
        raise CiError(
            f"timed out after {self.timeout}s waiting for a {self.resource_class} "
            f"model-proof GPU lease from: {' '.join(map(str, self.gpu_ids))}"
        )

    def _try_shared(self, deadline: float) -> bool:
        allocator = self._allocator(deadline)
        try:
            explicit = self.context.env.get("TRTMC_GPU_SLOT_ID")
            slots = [int(explicit)] if explicit else list(range(self.slots_per_gpu))
            for slot in slots:
                for gpu in self.gpu_ids:
                    reservation = FileLock(self.lock_dir / f"gpu-{gpu}-reservation.lock")
                    if not reservation.try_lock():
                        reservation.handle.close()
                        continue
                    candidate = FileLock(self.lock_dir / f"gpu-{gpu}-slot-{slot}.lock")
                    if candidate.try_lock():
                        reservation.close()
                        self.gpu_id, self.slot_ids, self.slots = gpu, [slot], [candidate]
                        return True
                    candidate.handle.close()
                    reservation.close()
            return False
        finally:
            allocator.close()

    def _reserve_one_gpu(self, deadline: float) -> bool:
        allocator = self._allocator(deadline)
        try:
            gpu = self.gpu_ids[0]
            reservation = FileLock(self.lock_dir / f"gpu-{gpu}-reservation.lock")
            if not reservation.try_lock():
                reservation.handle.close()
                return False
            self.gpu_id, self.reservation = gpu, reservation
            return True
        finally:
            allocator.close()

    def _reserve_capacity_candidate(
        self,
        deadline: float,
        *,
        exclude: set[int],
    ) -> bool:
        """Reserve the least-busy eligible GPU before waiting for it to drain."""
        allocator = self._allocator(deadline)
        try:
            best_gpu: int | None = None
            best_available_slots = -1
            for gpu in self.gpu_ids:
                if gpu in exclude:
                    continue
                reservation = FileLock(self.lock_dir / f"gpu-{gpu}-reservation.lock")
                if not reservation.try_lock():
                    reservation.handle.close()
                    continue
                available_slots = 0
                for slot in range(self.slots_per_gpu):
                    candidate = FileLock(self.lock_dir / f"gpu-{gpu}-slot-{slot}.lock")
                    if candidate.try_lock():
                        available_slots += 1
                        candidate.close()
                    else:
                        candidate.handle.close()
                reservation.close()
                if available_slots > best_available_slots:
                    best_gpu = gpu
                    best_available_slots = available_slots
                if available_slots == self.slots_per_gpu:
                    break
            if best_gpu is None:
                return False
            reservation = FileLock(self.lock_dir / f"gpu-{best_gpu}-reservation.lock")
            if not reservation.try_lock():
                reservation.handle.close()
                return False
            self.gpu_id, self.reservation = best_gpu, reservation
            print(
                f"Reserved GPU {best_gpu} for capacity-gated exclusive admission "
                f"with {best_available_slots}/{self.slots_per_gpu} slots already idle"
            )
            return True
        finally:
            allocator.close()

    def _drain_reserved_gpu(self, deadline: float) -> None:
        assert self.gpu_id is not None
        for slot in range(self.slots_per_gpu):
            candidate = FileLock(self.lock_dir / f"gpu-{self.gpu_id}-slot-{slot}.lock")
            if not self._wait_lock(candidate, deadline):
                candidate.close()
                raise CiError(
                    f"timed out after {self.timeout}s waiting for an exclusive_gpu "
                    f"model-proof GPU lease from: {self.gpu_id}"
                )
            self.slots.append(candidate)
            self.slot_ids.append(slot)

    def _candidate_has_capacity(
        self,
        deadline: float,
        *,
        candidates_remaining: int,
    ) -> bool:
        if not self.min_free_gpu_memory_mib:
            return True
        assert self.gpu_id is not None
        now = time.monotonic()
        settle_deadline = now + max(0.0, deadline - now) / max(1, candidates_remaining)
        settle_poll_interval = max(0.25, min(5.0, self.poll_interval))
        while time.monotonic() < settle_deadline:
            try:
                has_compute_processes = self._gpu_has_compute_processes(
                    self.gpu_id,
                    timeout_seconds=max(0.001, settle_deadline - time.monotonic()),
                )
            except CiError:
                if time.monotonic() >= settle_deadline:
                    return False
                raise
            if has_compute_processes:
                print(
                    f"GPU {self.gpu_id} has an active compute process; "
                    "memory admission requires an idle GPU"
                )
                time.sleep(
                    min(
                        settle_poll_interval,
                        max(0.0, settle_deadline - time.monotonic()),
                    )
                )
                continue
            try:
                snapshot = self._gpu_memory_snapshot(
                    self.gpu_id,
                    timeout_seconds=max(0.001, settle_deadline - time.monotonic()),
                )
            except CiError:
                if time.monotonic() >= settle_deadline:
                    return False
                raise
            try:
                has_compute_processes = self._gpu_has_compute_processes(
                    self.gpu_id,
                    timeout_seconds=max(0.001, settle_deadline - time.monotonic()),
                )
            except CiError:
                if time.monotonic() >= settle_deadline:
                    return False
                raise
            if has_compute_processes:
                print(
                    f"GPU {self.gpu_id} acquired a compute process while memory "
                    "admission was being sampled"
                )
                time.sleep(
                    min(
                        settle_poll_interval,
                        max(0.0, settle_deadline - time.monotonic()),
                    )
                )
                continue
            if time.monotonic() >= settle_deadline:
                return False
            self.last_observed_total_mib[self.gpu_id] = snapshot["total_mib"]
            self.last_observed_free_mib[self.gpu_id] = snapshot["free_mib"]
            if snapshot["free_mib"] >= self.min_free_gpu_memory_mib:
                self.gpu_memory_admission = {
                    "source": snapshot["source"],
                    "required_free_mib": self.min_free_gpu_memory_mib,
                    "observed_total_mib": snapshot["total_mib"],
                    "observed_used_mib": snapshot["used_mib"],
                    "observed_free_mib": snapshot["free_mib"],
                }
                print(
                    f"GPU {self.gpu_id} memory admission passed: "
                    f"{snapshot['free_mib']} MiB free; "
                    f"{self.min_free_gpu_memory_mib} MiB required"
                )
                return True
            print(
                f"GPU {self.gpu_id} has {snapshot['free_mib']} MiB free; "
                f"{self.model} requires {self.min_free_gpu_memory_mib} MiB"
            )
            if snapshot["total_mib"] < self.min_free_gpu_memory_mib:
                return False
            time.sleep(
                min(
                    settle_poll_interval,
                    max(0.0, settle_deadline - time.monotonic()),
                )
            )
        return False

    def _gpu_memory_snapshot(
        self,
        gpu: int,
        *,
        timeout_seconds: float = 10.0,
    ) -> dict[str, object]:
        started = time.monotonic()
        executable = self.context.executable("nvidia-smi")
        result = self.context.run(
            [
                executable,
                "--query-gpu=index,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            limit=f"{min(10.0, timeout_seconds):.6f}s",
            capture_output=True,
        )
        snapshots: dict[int, dict[str, object]] = {}
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4 or any(not field.isdigit() for field in fields):
                raise CiError(f"invalid nvidia-smi GPU memory row: {line!r}")
            index, total_mib, used_mib, free_mib = map(int, fields)
            if index in snapshots:
                raise CiError(f"nvidia-smi returned duplicate GPU index: {index}")
            if used_mib > total_mib or free_mib > total_mib:
                raise CiError(f"nvidia-smi returned invalid GPU memory values for GPU {index}")
            snapshots[index] = {
                "source": "nvidia-smi",
                "total_mib": total_mib,
                "used_mib": used_mib,
                "free_mib": free_mib,
            }
        if gpu not in snapshots:
            raise CiError(f"nvidia-smi did not report configured GPU {gpu}")
        snapshot = snapshots[gpu]
        if snapshot["free_mib"] >= self.min_free_gpu_memory_mib:
            return snapshot
        remaining = max(0.001, timeout_seconds - (time.monotonic() - started))
        coherent = self._coherent_gpu_memory_snapshot(
            gpu,
            snapshot,
            timeout_seconds=remaining,
        )
        return coherent or snapshot

    def _coherent_gpu_memory_snapshot(
        self,
        gpu: int,
        raw: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> dict[str, object] | None:
        """Count only reclaimable clean file pages on an OS-managed GPU NUMA node."""
        try:
            identity = self._gpu_numa_identity(gpu, timeout_seconds=timeout_seconds)
            if identity is None:
                return None
            pci_bus_id, reserved_mib, numa_node = identity
            meminfo = self._node_meminfo(numa_node)
            page_size = os.sysconf("SC_PAGE_SIZE")
            if not isinstance(page_size, int) or page_size <= 0 or page_size % 1024:
                raise CiError("Linux reported an invalid base page size")
            high_kib = self._node_high_watermark_kib(
                numa_node,
                page_size_kib=page_size // 1024,
            )
            total_kib = meminfo["MemTotal"]
            expected_total_kib = (int(raw["total_mib"]) - reserved_mib) * 1024
            if abs(total_kib - expected_total_kib) > page_size // 1024:
                raise CiError(
                    f"GPU {gpu} NUMA node {numa_node} total does not match "
                    f"{pci_bus_id} FB memory"
                )
            clean_file_kib = max(
                0,
                meminfo["Active(file)"]
                + meminfo["Inactive(file)"]
                - meminfo["Dirty"]
                - meminfo["Writeback"]
                - meminfo["Mapped"]
                - meminfo["Shmem"]
                - meminfo["Unevictable"],
            )
            if high_kib >= total_kib:
                raise CiError(f"GPU {gpu} NUMA memory watermark is inconsistent")
            reclaimable_kib = min(
                total_kib,
                meminfo["MemFree"] + clean_file_kib,
            )
            available_kib = max(0, reclaimable_kib - high_kib)
            if available_kib < 0 or available_kib > total_kib:
                raise CiError(f"GPU {gpu} NUMA memory accounting is inconsistent")
            total_mib = total_kib // 1024
            free_mib = available_kib // 1024
            used_mib = total_mib - free_mib
            print(
                f"GPU {gpu} raw nvidia-smi free memory is {raw['free_mib']} MiB; "
                f"NUMA node {numa_node} has {clean_file_kib // 1024} MiB of "
                f"reclaimable clean file cache and {free_mib} MiB of effective "
                "capacity"
            )
            return {
                "source": "linux-numa-meminfo",
                "total_mib": total_mib,
                "used_mib": used_mib,
                "free_mib": free_mib,
            }
        except (CiError, OSError, ValueError) as error:
            message = f"GPU {gpu} NUMA memory admission unavailable: {error}"
            if message not in self.numa_fallback_warnings:
                print(message)
                self.numa_fallback_warnings.add(message)
            return None

    def _gpu_numa_identity(
        self,
        gpu: int,
        *,
        timeout_seconds: float,
    ) -> tuple[str, int, int] | None:
        started = time.monotonic()
        executable = self.context.executable("nvidia-smi")
        result = self.context.run(
            [
                executable,
                "-i",
                str(gpu),
                "--query-gpu=pci.bus_id,memory.reserved",
                "--format=csv,noheader,nounits",
            ],
            limit=f"{min(10.0, timeout_seconds):.6f}s",
            capture_output=True,
        )
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise CiError(f"nvidia-smi returned invalid GPU identity rows for GPU {gpu}")
        fields = [field.strip() for field in rows[0].split(",")]
        if (
            len(fields) != 2
            or not re.fullmatch(r"[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]", fields[0])
            or not fields[1].isdigit()
        ):
            raise CiError(f"nvidia-smi returned an invalid GPU identity row: {rows[0]!r}")
        pci_bus_id, reserved_text = fields
        numa_node = self.gpu_numa_nodes.get(pci_bus_id)
        if numa_node is None:
            numa_node = self._gpu_numa_node(
                gpu,
                timeout_seconds=max(
                    0.001,
                    timeout_seconds - (time.monotonic() - started),
                ),
            )
            if numa_node is None:
                return None
            self.gpu_numa_nodes[pci_bus_id] = numa_node
        return pci_bus_id, int(reserved_text), numa_node

    def _gpu_numa_node(
        self,
        gpu: int,
        *,
        timeout_seconds: float,
    ) -> int | None:
        executable = self.context.executable("nvidia-smi")
        result = self.context.run(
            [
                executable,
                "topo",
                "--gpu-numa-id",
                "-i",
                str(gpu),
            ],
            limit=f"{min(10.0, timeout_seconds):.6f}s",
            capture_output=True,
        )
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise CiError(f"nvidia-smi topology has no unique NUMA ID for GPU {gpu}")
        value = rows[0]
        if value in {"", "-", "N/A"}:
            return None
        if not value.isdigit():
            raise CiError(f"nvidia-smi topology has an invalid NUMA ID for GPU {gpu}")
        return int(value)

    def _gpu_has_compute_processes(
        self,
        gpu: int,
        *,
        timeout_seconds: float,
    ) -> bool:
        executable = self.context.executable("nvidia-smi")
        result = self.context.run(
            [
                executable,
                "-i",
                str(gpu),
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            limit=f"{min(10.0, timeout_seconds):.6f}s",
            capture_output=True,
        )
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if any(not row.isdigit() for row in rows):
            raise CiError(f"nvidia-smi returned invalid compute process rows for GPU {gpu}")
        return bool(rows)

    @staticmethod
    def _node_meminfo(node: int) -> dict[str, int]:
        path = Path(f"/sys/devices/system/node/node{node}/meminfo")
        if not path.is_file():
            raise CiError(f"GPU NUMA node meminfo is unavailable: {path}")
        cpulist = path.with_name("cpulist").read_text(encoding="utf-8").strip()
        if cpulist:
            raise CiError(f"GPU NUMA node {node} unexpectedly contains CPUs")
        return GpuLease._parse_node_meminfo(node, path.read_text(encoding="utf-8"))

    @staticmethod
    def _parse_node_meminfo(node: int, text: str) -> dict[str, int]:
        required = {
            "MemTotal",
            "MemFree",
            "Active(file)",
            "Inactive(file)",
            "Unevictable",
            "Dirty",
            "Writeback",
            "Mapped",
            "Shmem",
        }
        values: dict[str, int] = {}
        prefix = f"Node {node} "
        for line in text.splitlines():
            if not line.startswith(prefix):
                continue
            body = line[len(prefix) :]
            fields = body.split()
            first_field = fields[0] if fields else ""
            name = first_field.removesuffix(":")
            if name not in required:
                continue
            if len(fields) != 3 or fields[2] != "kB" or not fields[0].endswith(":"):
                raise CiError(f"GPU NUMA node {node} has invalid meminfo: {line!r}")
            if name in values or not fields[1].isdigit():
                raise CiError(f"GPU NUMA node {node} has invalid {name}")
            values[name] = int(fields[1])
        if set(values) != required:
            missing = ", ".join(sorted(required - set(values)))
            raise CiError(f"GPU NUMA node {node} meminfo is missing: {missing}")
        if any(value < 0 or value > values["MemTotal"] for value in values.values()):
            raise CiError(f"GPU NUMA node {node} meminfo values are inconsistent")
        return values

    @staticmethod
    def _node_high_watermark_kib(node: int, *, page_size_kib: int) -> int:
        return GpuLease._parse_node_high_watermark_kib(
            node,
            Path("/proc/zoneinfo").read_text(encoding="utf-8"),
            page_size_kib=page_size_kib,
        )

    @staticmethod
    def _parse_node_high_watermark_kib(
        node: int,
        text: str,
        *,
        page_size_kib: int,
    ) -> int:
        current_node: int | None = None
        high_pages = 0
        zones = 0
        for line in text.splitlines():
            match = re.fullmatch(r"Node ([0-9]+), zone\s+\S+", line)
            if match:
                current_node = int(match.group(1))
                if current_node == node:
                    zones += 1
                continue
            if current_node == node:
                match = re.fullmatch(r"\s+high\s+([0-9]+)", line)
                if match:
                    high_pages += int(match.group(1))
        if not zones or not high_pages:
            raise CiError(f"GPU NUMA node {node} has no Linux high watermark")
        return high_pages * page_size_kib

    def _capacity_timeout_detail(self) -> str:
        if not self.min_free_gpu_memory_mib:
            return ""
        observed = ", ".join(
            f"GPU {gpu}={free_mib} MiB"
            for gpu, free_mib in sorted(self.last_observed_free_mib.items())
        )
        suffix = f"; last observed free memory: {observed}" if observed else ""
        return (
            f" requiring at least {self.min_free_gpu_memory_mib} MiB free GPU memory"
            f"{suffix}"
        )

    def _requeue_after_capacity_rejection(self, deadline: float) -> bool:
        self._release_ticket()
        print("Yielding the GPU admission queue after a memory-capacity rejection")
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        if time.monotonic() >= deadline:
            return False
        try:
            self.ticket = self._create_ticket("global", deadline)
            self._wait_for_queue_head("global", deadline)
        except CiError:
            if time.monotonic() < deadline:
                raise
            self._release_ticket()
            return False
        return True

    def _try_exclusive_any(
        self,
        deadline: float,
        *,
        exclude: set[int] | None = None,
    ) -> bool:
        allocator = self._allocator(deadline)
        try:
            for gpu in self.gpu_ids:
                if exclude and gpu in exclude:
                    continue
                slots = []
                for slot in range(self.slots_per_gpu):
                    candidate = FileLock(self.lock_dir / f"gpu-{gpu}-slot-{slot}.lock")
                    if not candidate.try_lock():
                        candidate.handle.close()
                        for held in slots:
                            held.close()
                        break
                    slots.append(candidate)
                if len(slots) != self.slots_per_gpu:
                    continue
                reservation = FileLock(self.lock_dir / f"gpu-{gpu}-reservation.lock")
                if reservation.try_lock():
                    self.gpu_id, self.slots = gpu, slots
                    self.slot_ids = list(range(self.slots_per_gpu))
                    self.reservation = reservation
                    return True
                reservation.handle.close()
                for held in slots:
                    held.close()
            return False
        finally:
            allocator.close()

    def _allocator(self, deadline: float) -> FileLock:
        lock = FileLock(self.lock_dir / "allocator.lock")
        stall_deadline = min(deadline, time.monotonic() + 10)
        if not self._wait_lock(lock, stall_deadline):
            lock.handle.close()
            if time.monotonic() >= deadline:
                raise CiError(
                    f"timed out after {self.timeout}s waiting for a {self.resource_class} "
                    f"model-proof GPU lease from: {' '.join(map(str, self.gpu_ids))}"
                )
            holder = lock.path.read_text(encoding="utf-8", errors="replace").strip() or "unknown"
            raise CiError(
                "GPU allocator mutex was held for over 10s; critical sections must never block "
                f"(last holder: {holder})"
            )
        lock.handle.seek(0)
        lock.handle.truncate()
        lock.handle.write(f"pid={os.getpid()}\n")
        lock.handle.flush()
        return lock

    @staticmethod
    def _wait_lock(lock: FileLock, deadline: float, *, shared: bool = False) -> bool:
        while time.monotonic() < deadline:
            if lock.try_lock(shared=shared):
                return True
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        return False

    def _release_ticket(self) -> None:
        if not self.ticket:
            return
        self.ticket.path.unlink(missing_ok=True)
        self.ticket.close()
        self.ticket = None
