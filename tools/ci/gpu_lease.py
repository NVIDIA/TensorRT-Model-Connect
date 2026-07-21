# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocate shared slots or whole GPUs to concurrent isolated model proofs.

Boundary: cross-process fairness and locking only; this module never runs a model.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import socket
import time
from datetime import datetime, timezone
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
    ):
        if resource_class not in {"shared", "exclusive_gpu"}:
            raise CiError("model-proof resource class must be shared or exclusive_gpu")
        self.context = context
        self.model = model
        self.resource_class = resource_class
        self.artifacts = artifacts
        self.node_id = context.env.get("TRTMC_NODE_ID", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.node_id):
            raise CiError("TRTMC_NODE_ID must be configured safely by the proof runner")
        self.gpu_ids, self.slots_per_gpu, self.timeout, self.poll_interval = self._configuration()
        configured = context.env.get("TRTMC_MODEL_PROOF_GPU_LOCK_DIR", "")
        if not configured:
            raise CiError("TRTMC_MODEL_PROOF_GPU_LOCK_DIR must not be empty")
        self.lock_dir = Path(configured)
        if (
            not self.lock_dir.is_absolute()
            or self.lock_dir == Path("/")
            or self.lock_dir.is_symlink()
        ):
            raise CiError("TRTMC_MODEL_PROOF_GPU_LOCK_DIR must be a safe absolute directory")
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir = self.lock_dir.resolve(strict=True)
        repository = context.repository.resolve()
        if (
            self.lock_dir == Path("/")
            or self.lock_dir == repository
            or self.lock_dir in repository.parents
            or repository in self.lock_dir.parents
        ):
            raise CiError("TRTMC_MODEL_PROOF_GPU_LOCK_DIR must not overlap the repository")
        stat = self.lock_dir.stat()
        identity = f"{self.lock_dir}\0{stat.st_dev}\0{stat.st_ino}".encode()
        self.lock_namespace = hashlib.sha256(identity).hexdigest()
        self.machine: FileLock | None = None
        self.ticket: FileLock | None = None
        self.reservation: FileLock | None = None
        self.slots: list[FileLock] = []
        self.gpu_id: int | None = None
        self.slot_ids: list[int] = []
        self.gpu_uuid = ""
        self.acquired_at = ""
        self.released_at: str | None = None

    def acquire(self) -> "GpuLease":
        deadline = time.monotonic() + self.timeout
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
                        self._mark_acquired()
                        print(
                            f"Leased shared model-proof GPU {self.gpu_id} slot {self.slot_ids[0]} "
                            f"via {self.slots[0].path}"
                        )
                        return self
                elif len(self.gpu_ids) == 1:
                    if self._reserve_one_gpu(deadline):
                        self._release_ticket()
                        self._drain_reserved_gpu(deadline)
                        self._mark_acquired()
                        print(
                            f"Leased exclusive model-proof GPU {self.gpu_id} slots "
                            f"{' '.join(map(str, self.slot_ids))}"
                        )
                        return self
                elif self._try_exclusive_any(deadline):
                    self._release_ticket()
                    self._mark_acquired()
                    print(
                        f"Leased exclusive model-proof GPU {self.gpu_id} slots "
                        f"{' '.join(map(str, self.slot_ids))}"
                    )
                    return self
                time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))
            raise CiError(
                f"timed out after {self.timeout}s waiting for a {self.resource_class} "
                f"model-proof GPU lease from: {' '.join(map(str, self.gpu_ids))}"
            )
        except Exception:
            self.release()
            raise

    def release(self) -> None:
        self.mark_released()
        self._release_ticket()
        for lock in self.slots:
            lock.close()
        self.slots.clear()
        self.slot_ids.clear()
        if self.reservation:
            self.reservation.close()
            self.reservation = None
        if self.machine:
            self.machine.close()
            self.machine = None

    def mark_released(self) -> None:
        if self.acquired_at and self.released_at is None:
            self.released_at = datetime.now(timezone.utc).isoformat()

    def evidence(self, revision: str) -> dict[str, object]:
        if self.gpu_id is None or not self.slot_ids:
            raise CiError("GPU lease evidence requested before acquisition")
        return {
            "schema_version": 2,
            "model": self.model,
            "source_revision": revision,
            "run_id": self.context.env.get("GITHUB_RUN_ID", "local"),
            "job_id": self.context.env.get("GITHUB_JOB", "local"),
            "runner_name": self.context.env.get("RUNNER_NAME", "local"),
            "node_id": self.node_id,
            "hostname": socket.gethostname(),
            "gpu_id": str(self.gpu_id),
            "gpu_index": str(self.gpu_id),
            "gpu_uuid": self.gpu_uuid,
            "gpu_slot": self.slot_ids[0] if self.resource_class == "shared" else None,
            "gpu_slots": self.slot_ids,
            "gpu_slot_ids": self.slot_ids,
            "slots_per_gpu": self.slots_per_gpu,
            "gpu_slots_per_device": self.slots_per_gpu,
            "resource_class": self.resource_class,
            "gpu_resource_class": self.resource_class,
            "lock_namespace": self.lock_namespace,
            "acquired_at": self.acquired_at,
            "released_at": self.released_at,
        }

    def _mark_acquired(self) -> None:
        assert self.gpu_id is not None
        value = self.context.output(
            [
                "nvidia-smi",
                "--query-gpu=uuid",
                "--format=csv,noheader",
                "-i",
                str(self.gpu_id),
            ]
        )
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        if len(lines) != 1 or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", lines[0]):
            raise CiError(f"could not resolve a unique GPU UUID for index {self.gpu_id}")
        self.gpu_uuid = lines[0]
        self.acquired_at = datetime.now(timezone.utc).isoformat()

    def _configuration(self) -> tuple[list[int], int, int, float]:
        configured = self.context.env.get("TRTMC_MODEL_PROOF_GPU_IDS", "")
        if not configured:
            raise CiError("TRTMC_MODEL_PROOF_GPU_IDS must be configured by the proof runner")
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
        slots_text = self.context.env.get("TRTMC_MODEL_PROOF_SLOTS_PER_GPU", "")
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

    def _try_exclusive_any(self, deadline: float) -> bool:
        allocator = self._allocator(deadline)
        try:
            for gpu in self.gpu_ids:
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
