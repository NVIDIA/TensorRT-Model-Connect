# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the manual shared-GPU capacity canary."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.ci.capacity_canary import (
    capacity_matrix,
    hold_capacity_slot,
    load_receipts,
    probe_container_gpu_uuid,
    verify_capacity_receipts,
    verify_exclusive_safety_receipts,
)
from tools.ci.context import CiContext
from tools.ci.process import CiError


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)


def _time(seconds: float) -> str:
    return (BASE + timedelta(seconds=seconds)).isoformat()


def _receipt(
    leg: int,
    *,
    started: float,
    acquired: float,
    released: float,
    node: str,
    gpu: str,
    slot: int,
    slots_per_gpu: int = 2,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "trtmc_capacity_canary",
        "leg_id": leg,
        "expected_slots": 4,
        "barrier_epoch": int(BASE.timestamp()) + 10,
        "source_revision": "a" * 40,
        "run_id": "42",
        "job_id": "exercise",
        "runner_name": f"runner-{leg}",
        "node_id": node,
        "hostname": f"host-{node}",
        "resource_class": "shared",
        "gpu_index": str(leg % 2),
        "gpu_uuid": gpu,
        "gpu_slot": slot,
        "gpu_slots_per_device": slots_per_gpu,
        "lock_namespace": ("1" if node == "node-a" else "2") * 64,
        "worker_started_at": _time(started),
        "acquired_at": _time(acquired),
        "released_at": _time(released),
        "probe_gpu_uuid": gpu,
    }


def _valid_receipts() -> list[dict[str, object]]:
    return [
        _receipt(
            0,
            started=0,
            acquired=1,
            released=10,
            node="node-a",
            gpu="GPU-aaaa",
            slot=0,
        ),
        _receipt(
            1,
            started=0.1,
            acquired=1.1,
            released=10.1,
            node="node-a",
            gpu="GPU-aaaa",
            slot=1,
        ),
        _receipt(
            2,
            started=0.2,
            acquired=1.2,
            released=10.2,
            node="node-b",
            gpu="GPU-bbbb",
            slot=0,
        ),
        _receipt(
            3,
            started=0.3,
            acquired=1.3,
            released=10.3,
            node="node-b",
            gpu="GPU-bbbb",
            slot=1,
        ),
        _receipt(
            4,
            started=10.05,
            acquired=10.15,
            released=12,
            node="node-a",
            gpu="GPU-aaaa",
            slot=0,
        ),
    ]


def _exclusive_lease(*, started: float, acquired: float, released: float) -> dict[str, object]:
    return {
        "source_revision": "a" * 40,
        "run_id": "42",
        "job_id": "exercise",
        "runner_name": "runner-exclusive",
        "node_id": "node-selected",
        "hostname": "host-selected",
        "resource_class": "exclusive_gpu",
        "gpu_index": "2",
        "gpu_uuid": "GPU-exclusive",
        "gpu_slot": None,
        "gpu_slot_ids": [0, 1, 2, 3],
        "gpu_slots_per_device": 4,
        "lock_namespace": "3" * 64,
        "worker_started_at": _time(started),
        "acquired_at": _time(acquired),
        "released_at": _time(released),
        "probe_gpu_uuid": "GPU-exclusive",
    }


def _valid_exclusive_receipt() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "trtmc_exclusive_safety",
        "placement_scope": "one scheduler-selected generic runner",
        "source_revision": "a" * 40,
        "run_id": "42",
        "runner_name": "runner-exclusive",
        "node_id": "node-selected",
        "primary": _exclusive_lease(started=0, acquired=1, released=5),
        "contender": {
            "attempted_at": _time(2),
            **_exclusive_lease(started=2, acquired=5.1, released=6),
        },
    }


def test_matrix_dispatches_exactly_expected_plus_one_generic_legs() -> None:
    assert capacity_matrix(4) == {"leg": [0, 1, 2, 3, 4]}
    with pytest.raises(CiError, match="expected_slots"):
        capacity_matrix(0)


def test_verifier_proves_concurrency_queueing_and_dynamic_node_capacity() -> None:
    summary = verify_capacity_receipts(
        _valid_receipts(),
        expected_slots=4,
        expected_run_id="42",
        expected_revision="a" * 40,
    )
    assert summary["outcome"] == "success"
    assert summary["maximum_concurrency"] == 4
    assert summary["barrier_epoch"] == int(BASE.timestamp()) + 10
    assert summary["first_wave_runner_count"] == 4
    assert summary["first_wave_slot_count"] == 4
    assert summary["extra_leg_id"] == 4
    assert summary["nodes"] == {
        "node-a": {
            "capacity": 2,
            "gpu_count": 1,
            "gpu_uuids": ["GPU-aaaa"],
            "slots_per_gpu": 2,
            "runner_count": 2,
            "lock_namespace": "1" * 64,
        },
        "node-b": {
            "capacity": 2,
            "gpu_count": 1,
            "gpu_uuids": ["GPU-bbbb"],
            "slots_per_gpu": 2,
            "runner_count": 2,
            "lock_namespace": "2" * 64,
        },
    }


def test_verifier_rejects_an_extra_leg_that_started_before_release() -> None:
    receipts = _valid_receipts()
    receipts[-1]["worker_started_at"] = _time(9)
    with pytest.raises(CiError, match="extra matrix leg started before"):
        verify_capacity_receipts(receipts, expected_slots=4)


def test_verifier_rejects_duplicate_first_wave_slot_or_runner() -> None:
    receipts = _valid_receipts()
    receipts[1]["gpu_slot"] = 0
    with pytest.raises(CiError, match="duplicate GPU slot tuples"):
        verify_capacity_receipts(receipts, expected_slots=4)

    receipts = _valid_receipts()
    receipts[1]["runner_name"] = receipts[0]["runner_name"]
    with pytest.raises(CiError, match="unique runner listeners"):
        verify_capacity_receipts(receipts, expected_slots=4)


def test_verifier_rejects_partial_gpu_slot_coverage_on_a_node() -> None:
    receipts = _valid_receipts()
    receipts[3]["gpu_uuid"] = "GPU-cccc"
    receipts[3]["probe_gpu_uuid"] = "GPU-cccc"
    with pytest.raises(CiError, match="did not expose every configured slot"):
        verify_capacity_receipts(receipts, expected_slots=4)


def test_verifier_rejects_mismatched_container_probe_and_stale_run() -> None:
    receipts = _valid_receipts()
    receipts[0]["probe_gpu_uuid"] = "GPU-other"
    with pytest.raises(CiError, match="container probe"):
        verify_capacity_receipts(receipts, expected_slots=4)

    with pytest.raises(CiError, match="run ID"):
        verify_capacity_receipts(_valid_receipts(), expected_slots=4, expected_run_id="43")


def test_load_receipts_reads_only_receipt_json_files(tmp_path: Path) -> None:
    nested = tmp_path / "one"
    nested.mkdir()
    (nested / "receipt-0.json").write_text(json.dumps(_valid_receipts()[0]), encoding="utf-8")
    (nested / "holder-0.log").write_text("ignored", encoding="utf-8")
    assert load_receipts(tmp_path) == [_valid_receipts()[0]]


def test_exclusive_verifier_proves_all_slots_non_overlap_and_queued_resume() -> None:
    summary = verify_exclusive_safety_receipts(
        [_valid_exclusive_receipt()],
        expected_run_id="42",
        expected_revision="a" * 40,
    )
    assert summary["outcome"] == "success"
    assert summary["configured_slots"] == 4
    assert summary["primary_owns_all_slots"] is True
    assert summary["contender_owns_all_slots"] is True
    assert summary["same_gpu_exclusive_non_overlap"] is True
    assert summary["queued_work_resumed_after_release"] is True
    assert summary["container_uuid_matches"] is True
    assert "fleet placement is not proven" in str(summary["placement_scope"])


def test_exclusive_verifier_rejects_partial_ownership_and_overlap() -> None:
    receipt = _valid_exclusive_receipt()
    receipt["primary"]["gpu_slot_ids"] = [0, 1, 2]  # type: ignore[index]
    with pytest.raises(CiError, match="does not own every configured GPU slot"):
        verify_exclusive_safety_receipts([receipt])

    receipt = _valid_exclusive_receipt()
    receipt["contender"]["acquired_at"] = _time(4)  # type: ignore[index]
    with pytest.raises(CiError, match="overlapped on the same GPU"):
        verify_exclusive_safety_receipts([receipt])


def test_exclusive_verifier_rejects_uuid_mismatch_and_missing_contention() -> None:
    receipt = _valid_exclusive_receipt()
    receipt["contender"]["probe_gpu_uuid"] = "GPU-other"  # type: ignore[index]
    with pytest.raises(CiError, match="container UUID"):
        verify_exclusive_safety_receipts([receipt])

    receipt = _valid_exclusive_receipt()
    receipt["contender"]["gpu_uuid"] = "GPU-different"  # type: ignore[index]
    receipt["contender"]["probe_gpu_uuid"] = "GPU-different"  # type: ignore[index]
    with pytest.raises(CiError, match="same gpu_uuid"):
        verify_exclusive_safety_receipts([receipt])

    receipt = _valid_exclusive_receipt()
    receipt["contender"]["attempted_at"] = _time(5.01)  # type: ignore[index]
    receipt["contender"]["worker_started_at"] = _time(5.01)  # type: ignore[index]
    with pytest.raises(CiError, match="did not attempt while"):
        verify_exclusive_safety_receipts([receipt])


def test_holder_uses_real_lease_evidence_and_writes_release_receipt(tmp_path: Path) -> None:
    receipt_output = tmp_path / "receipt-0.json"
    events: list[str] = []

    class FakeLease:
        released = False
        acquired_at = datetime.now(timezone.utc)
        released_at: datetime | None = None

        def acquire(self) -> "FakeLease":
            self.acquired_at = datetime.now(timezone.utc)
            events.append("acquire")
            return self

        def mark_released(self) -> None:
            self.released = True
            self.released_at = self.acquired_at + timedelta(microseconds=1)
            events.append("mark_released")

        def release(self) -> None:
            events.append("release")

        def evidence(self, revision: str) -> dict[str, object]:
            return {
                "source_revision": revision,
                "run_id": "42",
                "job_id": "exercise",
                "runner_name": "runner-0",
                "node_id": "node-a",
                "hostname": "host-a",
                "resource_class": "shared",
                "gpu_index": "0",
                "gpu_uuid": "GPU-aaaa",
                "gpu_slot": 0,
                "gpu_slots_per_device": 2,
                "lock_namespace": "1" * 64,
                "acquired_at": self.acquired_at.isoformat(),
                "released_at": self.released_at.isoformat() if self.released_at else None,
            }

    def lease_factory(*_: object, **__: object) -> FakeLease:
        return FakeLease()

    probe_calls: list[dict[str, object]] = []

    def probe(*_: object, **kwargs: object) -> str:
        probe_calls.append(kwargs)
        return "GPU-aaaa"

    context = CiContext(
        repository=tmp_path,
        env={
            "GITHUB_SHA": "a" * 40,
            "GITHUB_RUN_ID": "42",
            "GITHUB_RUN_ATTEMPT": "1",
            "TRTMC_CI_IMAGE": "local-image:canary",
        },
    )
    now_values = iter([100.0, 110.0])
    sleeps: list[float] = []
    receipt = hold_capacity_slot(
        context=context,
        leg_id=0,
        expected_slots=4,
        barrier_epoch=110,
        receipt_output=receipt_output,
        lease_factory=lease_factory,  # type: ignore[arg-type]
        probe=probe,
        now=lambda: next(now_values),
        sleep=sleeps.append,
    )
    assert receipt_output.is_file()
    assert receipt["barrier_epoch"] == 110
    assert receipt["released_at"] is not None
    assert receipt["probe_gpu_uuid"] == "GPU-aaaa"
    assert events == ["acquire", "mark_released", "release"]
    assert sleeps == [10.0]
    assert probe_calls == [
        {
            "gpu_index": "0",
            "image": "local-image:canary",
            "container_name": "trtmc-capacity-42-1-0",
        }
    ]


def test_container_probe_is_network_free_pinned_to_the_leased_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = CiContext(repository=tmp_path, env={})
    commands: list[list[str]] = []

    def output(command: list[str]) -> str:
        commands.append(command)
        return "GPU-aaaa\n"

    def run(command: list[str], **_: object) -> object:
        commands.append(command)
        return object()

    monkeypatch.setattr(context, "output", output)
    monkeypatch.setattr(context, "run", run)
    assert (
        probe_container_gpu_uuid(
            context,
            gpu_index="3",
            image="local-image:canary",
            container_name="trtmc-capacity-42-1-0",
        )
        == "GPU-aaaa"
    )
    assert commands[0][:8] == [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--name",
        "trtmc-capacity-42-1-0",
        "--gpus",
    ]
    assert commands[0][8] == "device=3"
    assert "--entrypoint=nvidia-smi" in commands[0]
    assert commands[1] == [
        "docker",
        "rm",
        "--force",
        "trtmc-capacity-42-1-0",
    ]


def test_capacity_workflow_is_manual_main_only_and_has_no_model_or_hf_work() -> None:
    workflow = (REPO_ROOT / ".github/workflows/model-proof-capacity-canary.yml").read_text(
        encoding="utf-8"
    )
    source = (REPO_ROOT / "tools/ci/capacity_canary.py").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "schedule:" not in workflow
    assert "refs/heads/main" in workflow
    assert "default: 28" in workflow
    assert "default: 900" in workflow
    assert "default: shared-capacity" in workflow
    assert "- exclusive-safety" in workflow
    assert "matrix='{\"leg\":[0]}'" in workflow
    assert "tools.ci.capacity_canary matrix" in workflow
    assert "max-parallel:" not in workflow
    assert "TRTMC_MODEL_RUNNER_LABELS" in workflow
    for host_policy_name in (
        "TRTMC_NODE_ID",
        "TRTMC_MODEL_PROOF_GPU_IDS",
        "TRTMC_MODEL_PROOF_SLOTS_PER_GPU",
        "TRTMC_MODEL_PROOF_GPU_LOCK_DIR",
    ):
        assert re.search(rf"^\s+{host_policy_name}:\s", workflow, re.MULTILINE) is None
    assert 'if [ -z "${TRTMC_MODEL_PROOF_GPU_IDS:-}" ]' in workflow
    assert 'if [ -z "${TRTMC_MODEL_PROOF_SLOTS_PER_GPU:-}" ]' in workflow
    assert 'if [ -z "${TRTMC_MODEL_PROOF_GPU_LOCK_DIR:-}" ]' in workflow
    assert "barrier_epoch" in workflow
    assert "--expected-barrier-epoch" in workflow
    assert "exclusive-safety" in workflow
    assert "exclusive-contender" in source
    assert "verify-exclusive" in workflow
    assert "one scheduler-selected generic runner" in source
    assert "nohup" not in workflow
    assert "release_file" not in source
    assert "GITHUB_TOKEN" not in workflow
    assert "actions: read" not in workflow
    assert "actions/download-artifact@v4" in workflow
    assert "--expected-run-id" in workflow
    assert "GpuLease" in source
    assert "--network=none" in source
    assert "--entrypoint=nvidia-smi" in source
    combined = workflow + source
    assert "secrets." not in workflow
    assert "HF_TOKEN" not in combined
    assert "huggingface" not in combined.lower()
    assert "compute01" not in combined
    assert "compute02" not in combined
