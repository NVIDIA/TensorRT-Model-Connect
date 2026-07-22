# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for declared GPU topology and the manual shared-capacity canary."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tools.ci.capacity_canary import (
    cache_warm_matrix,
    capacity_matrix,
    cohort_matrix,
    hold_capacity_slot,
    load_receipts,
    main as capacity_canary_main,
    normalize_topology_contract,
    probe_container_gpu_uuid,
    topology_contract_digest,
    verify_capacity_receipts,
    verify_cohort_receipts,
    verify_cross_workflow_receipts,
    verify_exclusive_safety_receipts,
)
from tools.ci.context import CiContext
from tools.ci.process import CiError


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASE = datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc)
REPOSITORY = "NVIDIA/TensorRT-Model-Connect"


def _topology(
    *,
    slots_per_gpu: int,
    nodes: list[tuple[str, list[int]]],
    rollback_baseline_node: str | None = None,
) -> dict[str, object]:
    node_values = [
        {
            "node_label": f"trtmc-node-{node_id}",
            "gpu_indices": gpu_indices,
        }
        for node_id, gpu_indices in nodes
    ]
    return {
        "schema_version": 1,
        "kind": "trtmc_capacity_topology",
        "slots_per_gpu": slots_per_gpu,
        "rollback_baseline_node_label": (
            f"trtmc-node-{rollback_baseline_node or nodes[-1][0]}"
        ),
        "nodes": node_values,
    }


SMALL_TOPOLOGY = _topology(
    slots_per_gpu=2,
    nodes=[("node-a", [0]), ("node-b", [1])],
)
FULL_TOPOLOGY = _topology(
    slots_per_gpu=4,
    nodes=[("node-a", [0, 1, 2, 3]), ("node-b", [1, 2, 3])],
)
ROLLBACK_TOPOLOGY = _topology(
    slots_per_gpu=4,
    nodes=[("node-b", [1, 2, 3])],
)
EXCLUSIVE_TOPOLOGY = _topology(
    slots_per_gpu=4,
    nodes=[("node-selected", [2])],
)


def _json_copy(value: object) -> object:
    return json.loads(json.dumps(value))


def _malformed_topology_cases() -> list[tuple[dict[str, object], str]]:
    cases: list[tuple[dict[str, object], str]] = []

    unknown = _json_copy(FULL_TOPOLOGY)
    assert isinstance(unknown, dict)
    unknown["unexpected"] = True
    cases.append((unknown, "fields do not match"))

    duplicate_node = _json_copy(FULL_TOPOLOGY)
    assert isinstance(duplicate_node, dict)
    duplicate_nodes = duplicate_node["nodes"]
    assert isinstance(duplicate_nodes, list)
    assert isinstance(duplicate_nodes[1], dict)
    duplicate_nodes[1]["node_label"] = "trtmc-node-node-a"
    cases.append((duplicate_node, "duplicate node identity"))

    unsorted_indices = _json_copy(FULL_TOPOLOGY)
    assert isinstance(unsorted_indices, dict)
    unsorted_nodes = unsorted_indices["nodes"]
    assert isinstance(unsorted_nodes, list)
    assert isinstance(unsorted_nodes[0], dict)
    unsorted_nodes[0]["gpu_indices"] = [1, 0, 2, 3]
    cases.append((unsorted_indices, "sorted and unique"))

    empty_indices = _json_copy(FULL_TOPOLOGY)
    assert isinstance(empty_indices, dict)
    empty_nodes = empty_indices["nodes"]
    assert isinstance(empty_nodes, list)
    assert isinstance(empty_nodes[0], dict)
    empty_nodes[0]["gpu_indices"] = []
    cases.append((empty_indices, "non-empty list"))

    boolean_index = _json_copy(FULL_TOPOLOGY)
    assert isinstance(boolean_index, dict)
    boolean_nodes = boolean_index["nodes"]
    assert isinstance(boolean_nodes, list)
    assert isinstance(boolean_nodes[0], dict)
    boolean_nodes[0]["gpu_indices"] = [False, 1, 2, 3]
    cases.append((boolean_index, "must be an integer"))

    unsafe_label = _json_copy(FULL_TOPOLOGY)
    assert isinstance(unsafe_label, dict)
    label_nodes = unsafe_label["nodes"]
    assert isinstance(label_nodes, list)
    assert isinstance(label_nodes[0], dict)
    label_nodes[0]["node_label"] = "trtmc-node-../../host"
    cases.append((unsafe_label, "invalid node label"))

    unsafe_baseline = _json_copy(FULL_TOPOLOGY)
    assert isinstance(unsafe_baseline, dict)
    unsafe_baseline["rollback_baseline_node_label"] = "trtmc-node-../../host"
    cases.append((unsafe_baseline, "rollback baseline node label is invalid"))

    unknown_baseline = _json_copy(FULL_TOPOLOGY)
    assert isinstance(unknown_baseline, dict)
    unknown_baseline["rollback_baseline_node_label"] = "trtmc-node-node-c"
    cases.append((unknown_baseline, "rollback baseline must identify a declared node"))

    missing_baseline = _json_copy(FULL_TOPOLOGY)
    assert isinstance(missing_baseline, dict)
    del missing_baseline["rollback_baseline_node_label"]
    cases.append((missing_baseline, "fields do not match"))
    return cases


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
    expected_slots: int = 4,
    barrier_epoch: int | None = None,
    run_id: str = "42",
    runner_name: str | None = None,
    cohort_id: str | None = None,
    gpu_index: int | None = None,
    topology: dict[str, object] | None = None,
) -> dict[str, object]:
    topology = topology or SMALL_TOPOLOGY
    return {
        "schema_version": 2,
        "kind": "trtmc_capacity_canary",
        "leg_id": leg,
        "expected_slots": expected_slots,
        "barrier_epoch": (
            barrier_epoch if barrier_epoch is not None else int(BASE.timestamp()) + 10
        ),
        "cohort_id": cohort_id,
        "expected_topology_digest": topology_contract_digest(topology),
        "source_revision": "a" * 40,
        "run_id": run_id,
        "job_id": "exercise",
        "runner_name": runner_name or f"runner-{leg}",
        "node_id": node,
        "hostname": f"host-{node}",
        "resource_class": "shared",
        "gpu_index": str(leg % 2 if gpu_index is None else gpu_index),
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
            gpu_index=0,
        ),
        _receipt(
            2,
            started=0.2,
            acquired=1.2,
            released=10.2,
            node="node-b",
            gpu="GPU-bbbb",
            slot=0,
            gpu_index=1,
        ),
        _receipt(
            3,
            started=0.3,
            acquired=1.3,
            released=10.3,
            node="node-b",
            gpu="GPU-bbbb",
            slot=1,
            gpu_index=1,
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


def _rollback_receipts() -> list[dict[str, object]]:
    placement = [
        (gpu_index, f"GPU-b{gpu_index}", slot)
        for gpu_index in (1, 2, 3)
        for slot in range(4)
    ]
    receipts = [
        _receipt(
            leg,
            started=0.01 * leg,
            acquired=1 + 0.01 * leg,
            released=10.2 + 0.001 * leg,
            node="node-b",
            gpu=gpu_uuid,
            slot=slot,
            slots_per_gpu=4,
            expected_slots=12,
            gpu_index=gpu_index,
            topology=ROLLBACK_TOPOLOGY,
        )
        for leg, (gpu_index, gpu_uuid, slot) in enumerate(placement)
    ]
    receipts.append(
        _receipt(
            12,
            started=10.201,
            acquired=10.202,
            released=12,
            node="node-b",
            gpu="GPU-b1",
            slot=0,
            slots_per_gpu=4,
            expected_slots=12,
            gpu_index=1,
            topology=ROLLBACK_TOPOLOGY,
        )
    )
    return receipts


def _cross_cohort_receipts() -> list[dict[str, object]]:
    placement = [
        ("node-a", gpu, f"GPU-a{gpu}", slot) for gpu in range(4) for slot in range(4)
    ] + [
        ("node-b", gpu + 1, f"GPU-b{gpu}", slot)
        for gpu in range(3)
        for slot in range(4)
    ]
    receipts: list[dict[str, object]] = []
    for index, (node, gpu_index, gpu_uuid, slot) in enumerate(placement):
        run_id = "100" if index < 14 else "200"
        leg = index if index < 14 else index - 14
        receipts.append(
            _receipt(
                leg,
                started=0.01 * index,
                acquired=1 + 0.01 * index,
                released=10.5 + 0.01 * index,
                node=node,
                gpu=gpu_uuid,
                slot=slot,
                slots_per_gpu=4,
                expected_slots=14,
                run_id=run_id,
                runner_name=f"runner-{run_id}-{leg}",
                cohort_id="cross-100-200",
                gpu_index=gpu_index,
                topology=FULL_TOPOLOGY,
            )
        )
    return receipts


def _small_cross_cohort_receipts() -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    for run_id, node, gpu_index, gpu_uuid in (
        ("100", "node-a", 0, "GPU-a0"),
        ("200", "node-b", 1, "GPU-b0"),
    ):
        for leg in range(2):
            receipts.append(
                _receipt(
                    leg,
                    started=0.1 * leg,
                    acquired=1 + 0.1 * leg,
                    released=10.5 + 0.1 * leg,
                    node=node,
                    gpu=gpu_uuid,
                    slot=leg,
                    slots_per_gpu=2,
                    expected_slots=2,
                    run_id=run_id,
                    runner_name=f"runner-{run_id}-{leg}",
                    cohort_id="cross-100-200",
                    gpu_index=gpu_index,
                    topology=SMALL_TOPOLOGY,
                )
            )
    return receipts


def _run_metadata(run_id: str) -> dict[str, object]:
    return {
        "id": int(run_id),
        "repository": {"full_name": REPOSITORY},
        "head_repository": {"full_name": REPOSITORY},
        "path": (
            ".github/workflows/model-proof-capacity-canary.yml@main"
            if run_id == "200"
            else ".github/workflows/model-proof-capacity-canary.yml"
        ),
        "event": "workflow_dispatch",
        "head_branch": "main",
        "head_sha": "a" * 40,
        "status": "completed",
        "conclusion": "success",
        "run_attempt": 1,
    }


def _verify_cross(
    receipts: list[dict[str, object]],
    *,
    expected_slots_per_run: int,
    expected_run_ids: list[str],
    **kwargs: object,
) -> dict[str, object]:
    expected_revision = str(kwargs.pop("expected_revision", "a" * 40))
    expected_topology = kwargs.pop(
        "expected_topology",
        FULL_TOPOLOGY if expected_slots_per_run == 14 else SMALL_TOPOLOGY,
    )
    return verify_cross_workflow_receipts(
        receipts,
        expected_slots_per_run=expected_slots_per_run,
        expected_topology=expected_topology,  # type: ignore[arg-type]
        expected_run_ids=expected_run_ids,
        run_metadata=[_run_metadata(run_id) for run_id in expected_run_ids],
        expected_repository=REPOSITORY,
        expected_revision=expected_revision,
        **kwargs,
    )


def _verify_capacity(
    receipts: list[dict[str, object]],
    **kwargs: object,
) -> dict[str, object]:
    return verify_capacity_receipts(
        receipts,
        expected_topology=SMALL_TOPOLOGY,
        **kwargs,
    )


def _verify_exclusive(
    receipts: list[dict[str, object]],
    **kwargs: object,
) -> dict[str, object]:
    return verify_exclusive_safety_receipts(
        receipts,
        expected_topology=EXCLUSIVE_TOPOLOGY,
        **kwargs,
    )


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
        "schema_version": 2,
        "kind": "trtmc_exclusive_safety",
        "expected_topology_digest": topology_contract_digest(EXCLUSIVE_TOPOLOGY),
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


def test_topology_contract_normalizes_exact_generic_16_plus_12_shape() -> None:
    normalized = normalize_topology_contract(FULL_TOPOLOGY)
    reversed_contract = _json_copy(FULL_TOPOLOGY)
    assert isinstance(reversed_contract, dict)
    reversed_nodes = reversed_contract["nodes"]
    assert isinstance(reversed_nodes, list)
    reversed_nodes.reverse()

    assert normalized["slots_per_gpu"] == 4
    assert normalized["rollback_baseline_node_label"] == "trtmc-node-node-b"
    assert normalized["nodes"][0]["gpu_indices"] == [0, 1, 2, 3]  # type: ignore[index]
    assert normalized["nodes"][1]["gpu_indices"] == [1, 2, 3]  # type: ignore[index]
    assert 0 not in normalized["nodes"][1]["gpu_indices"]  # type: ignore[index,operator]
    assert sum(len(node["gpu_indices"]) for node in normalized["nodes"]) == 7  # type: ignore[index]
    assert [len(node["gpu_indices"]) * 4 for node in normalized["nodes"]] == [16, 12]  # type: ignore[index]
    assert normalize_topology_contract(reversed_contract) == normalized
    assert topology_contract_digest(reversed_contract) == topology_contract_digest(
        FULL_TOPOLOGY
    )


@pytest.mark.parametrize(("contract", "message"), _malformed_topology_cases())
def test_topology_contract_rejects_malformed_or_inconsistent_data(
    contract: dict[str, object], message: str
) -> None:
    with pytest.raises(CiError, match=message):
        normalize_topology_contract(contract)


def test_topology_contract_cli_binds_mode_capacity_and_emits_canonical_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "normalized.json"
    github_output = tmp_path / "github-output"
    input_path.write_text(json.dumps(FULL_TOPOLOGY), encoding="utf-8")

    assert (
        capacity_canary_main(
            [
                "topology-contract",
                "--input",
                str(input_path),
                "--mode",
                "shared-capacity",
                "--requested-slots",
                "28",
                "--output",
                str(output_path),
                "--github-output",
                str(github_output),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == normalize_topology_contract(
        FULL_TOPOLOGY
    )
    outputs = dict(
        line.split("=", maxsplit=1)
        for line in github_output.read_text(encoding="utf-8").splitlines()
    )
    assert json.loads(outputs["expected_topology_json"]) == normalize_topology_contract(
        FULL_TOPOLOGY
    )
    assert outputs["expected_topology_digest"] == topology_contract_digest(FULL_TOPOLOGY)
    capsys.readouterr()

    assert (
        capacity_canary_main(
            [
                "topology-contract",
                "--input",
                str(input_path),
                "--mode",
                "shared-capacity",
                "--requested-slots",
                "27",
            ]
        )
        == 1
    )
    assert "expected_slots must equal topology capacity" in capsys.readouterr().err


def test_topology_contract_cli_derives_only_the_protected_rollback_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "rollback.json"
    github_output = tmp_path / "github-output"
    input_path.write_text(json.dumps(FULL_TOPOLOGY), encoding="utf-8")

    assert (
        capacity_canary_main(
            [
                "topology-contract",
                "--input",
                str(input_path),
                "--mode",
                "rollback-capacity",
                "--requested-slots",
                "12",
                "--output",
                str(output_path),
                "--github-output",
                str(github_output),
            ]
        )
        == 0
    )
    expected = normalize_topology_contract(ROLLBACK_TOPOLOGY)
    assert json.loads(output_path.read_text(encoding="utf-8")) == expected
    outputs = dict(
        line.split("=", maxsplit=1)
        for line in github_output.read_text(encoding="utf-8").splitlines()
    )
    assert json.loads(outputs["expected_topology_json"]) == expected
    assert outputs["expected_topology_digest"] == topology_contract_digest(expected)
    capsys.readouterr()

    for wrong_slots in ("11", "13", "28"):
        assert (
            capacity_canary_main(
                [
                    "topology-contract",
                    "--input",
                    str(input_path),
                    "--mode",
                    "rollback-capacity",
                    "--requested-slots",
                    wrong_slots,
                ]
            )
            == 1
        )
        assert "expected_slots must equal rollback baseline capacity" in capsys.readouterr().err


def test_cache_warm_matrix_emits_one_sorted_node_only_row_per_declared_node() -> None:
    reversed_contract = _json_copy(FULL_TOPOLOGY)
    assert isinstance(reversed_contract, dict)
    reversed_nodes = reversed_contract["nodes"]
    assert isinstance(reversed_nodes, list)
    reversed_nodes.reverse()

    normalized = normalize_topology_contract(reversed_contract)
    assert cache_warm_matrix(reversed_contract) == {
        "include": [
            {"node_label": node["node_label"]} for node in normalized["nodes"]  # type: ignore[index]
        ]
    }


def test_cache_warm_matrix_cli_reads_repository_manifest_and_emits_github_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    topology_path = REPO_ROOT / ".github/ci/gb300-pool-topology.json"
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    assert topology == normalize_topology_contract(topology)
    expected = cache_warm_matrix(topology)
    github_output = tmp_path / "github-output"

    assert (
        capacity_canary_main(
            [
                "cache-warm-matrix",
                "--input",
                str(topology_path),
                "--github-output",
                str(github_output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == expected
    outputs = dict(
        line.split("=", maxsplit=1)
        for line in github_output.read_text(encoding="utf-8").splitlines()
    )
    assert json.loads(outputs["matrix"]) == expected


def test_matrix_dispatches_exactly_expected_plus_one_generic_legs() -> None:
    assert capacity_matrix(4) == {"leg": [0, 1, 2, 3, 4]}
    with pytest.raises(CiError, match="expected_slots"):
        capacity_matrix(0)


def test_cohort_matrix_dispatches_exactly_the_requested_jobs() -> None:
    assert cohort_matrix(14) == {"leg": list(range(14))}
    with pytest.raises(CiError, match="cohort_slots"):
        cohort_matrix(0)


def test_exact_cohort_verifier_accepts_partial_per_node_placement() -> None:
    receipts = _cross_cohort_receipts()[:14]
    summary = verify_cohort_receipts(
        receipts,
        expected_slots=14,
        expected_topology=FULL_TOPOLOGY,
        expected_run_id="100",
        expected_revision="a" * 40,
        expected_barrier_epoch=int(BASE.timestamp()) + 10,
        expected_cohort_id="cross-100-200",
    )
    assert summary["outcome"] == "success"
    assert summary["receipt_count"] == 14
    assert summary["maximum_concurrency"] == 14
    assert summary["runner_count"] == 14
    assert summary["slot_count"] == 14
    assert summary["cohort_id"] == "cross-100-200"


def test_exact_cohort_verifier_rejects_missing_extra_or_unsafe_identity() -> None:
    receipts = _cross_cohort_receipts()[:14]
    with pytest.raises(CiError, match="exactly 14 receipts"):
        verify_cohort_receipts(
            receipts[:-1], expected_slots=14, expected_topology=FULL_TOPOLOGY
        )

    receipts = json.loads(json.dumps(receipts))
    receipts[0]["cohort_id"] = "../../unsafe"
    with pytest.raises(CiError, match="cohort_id"):
        verify_cohort_receipts(receipts, expected_slots=14, expected_topology=FULL_TOPOLOGY)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hostname", "host-node-a-alias"),
        ("lock_namespace", "f" * 64),
    ],
)
def test_exact_cohort_verifier_rejects_inconsistent_node_identity(field: str, value: str) -> None:
    receipts = _cross_cohort_receipts()[:14]
    receipts[-1][field] = value

    with pytest.raises(CiError, match="inconsistent hostname or lock namespace"):
        verify_cohort_receipts(receipts, expected_slots=14, expected_topology=FULL_TOPOLOGY)


def test_cross_workflow_verifier_proves_two_14_job_runs_share_all_28_slots() -> None:
    summary = _verify_cross(
        _cross_cohort_receipts(),
        expected_slots_per_run=14,
        expected_run_ids=["100", "200"],
        expected_revision="a" * 40,
        expected_barrier_epoch=int(BASE.timestamp()) + 10,
        expected_cohort_id="cross-100-200",
    )
    assert summary["outcome"] == "success"
    assert summary["combined_expected_slots"] == 28
    assert summary["receipt_count"] == 28
    assert summary["maximum_concurrency"] == 28
    assert summary["runner_count"] == 28
    assert summary["slot_count"] == 28
    assert summary["runs"] == {
        "100": {
            "receipt_count": 14,
            "maximum_concurrency": 14,
            "runner_count": 14,
            "slot_count": 14,
        },
        "200": {
            "receipt_count": 14,
            "maximum_concurrency": 14,
            "runner_count": 14,
            "slot_count": 14,
        },
    }
    assert [run["id"] for run in summary["source_runs"]] == [100, 200]
    assert all(run["run_attempt"] == 1 for run in summary["source_runs"])
    assert {run["path"] for run in summary["source_runs"]} == {
        ".github/workflows/model-proof-capacity-canary.yml",
        ".github/workflows/model-proof-capacity-canary.yml@main",
    }
    assert summary["expected_topology"] == normalize_topology_contract(FULL_TOPOLOGY)
    assert summary["expected_topology_digest"] == topology_contract_digest(FULL_TOPOLOGY)
    assert summary["nodes"]["node-a"]["capacity"] == 16
    assert summary["nodes"]["node-a"]["gpu_count"] == 4
    assert summary["nodes"]["node-a"]["gpu_indices"] == [0, 1, 2, 3]
    assert summary["nodes"]["node-b"]["capacity"] == 12
    assert summary["nodes"]["node-b"]["gpu_count"] == 3
    assert summary["nodes"]["node-b"]["gpu_indices"] == [1, 2, 3]
    assert 0 not in summary["nodes"]["node-b"]["gpu_indices"]
    assert sum(node["gpu_count"] for node in summary["nodes"].values()) == 7
    assert len(
        {
            gpu_uuid
            for node in summary["nodes"].values()
            for gpu_uuid in node["gpu_uuids"]
        }
    ) == 7


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", 999, "do not match the requested run IDs"),
        ("repository", {"full_name": "other/repo"}, "wrong repository"),
        ("head_repository", {"full_name": "fork/repo"}, "wrong head repository"),
        ("path", ".github/workflows/other.yml", "wrong workflow path"),
        (
            "path",
            ".github/workflows/model-proof-capacity-canary.yml@feature",
            "wrong workflow path",
        ),
        ("event", "pull_request", "wrong event"),
        ("head_branch", "feature", "wrong branch"),
        ("head_sha", "b" * 40, "wrong revision"),
        ("status", "in_progress", "wrong status"),
        ("conclusion", "failure", "wrong conclusion"),
        ("run_attempt", 2, "wrong run attempt"),
        ("run_attempt", True, "wrong run attempt"),
    ],
)
def test_cross_workflow_verifier_authenticates_source_run_metadata(
    field: str, value: object, message: str
) -> None:
    metadata = [_run_metadata("100"), _run_metadata("200")]
    metadata[1][field] = value

    with pytest.raises(CiError, match=message):
        verify_cross_workflow_receipts(
            _cross_cohort_receipts(),
            expected_slots_per_run=14,
            expected_topology=FULL_TOPOLOGY,
            expected_run_ids=["100", "200"],
            run_metadata=metadata,
            expected_repository=REPOSITORY,
            expected_revision="a" * 40,
        )


def test_cross_workflow_verifier_requires_exactly_two_source_run_records() -> None:
    with pytest.raises(CiError, match="exactly two source-run records"):
        verify_cross_workflow_receipts(
            _cross_cohort_receipts(),
            expected_slots_per_run=14,
            expected_topology=FULL_TOPOLOGY,
            expected_run_ids=["100", "200"],
            run_metadata=[_run_metadata("100")],
            expected_repository=REPOSITORY,
            expected_revision="a" * 40,
        )


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        ("hostname", "hostname .* maps to multiple node IDs"),
        ("gpu_uuid", "GPU UUID .* maps to multiple node IDs"),
    ],
)
def test_cross_workflow_verifier_rejects_physical_identity_reused_across_runs(
    identity: str, message: str
) -> None:
    receipts = _small_cross_cohort_receipts()
    for receipt in receipts:
        if receipt["run_id"] != "200":
            continue
        if identity == "hostname":
            receipt["hostname"] = "host-node-a"
        else:
            receipt["gpu_uuid"] = "GPU-a0"
            receipt["probe_gpu_uuid"] = "GPU-a0"

    with pytest.raises(CiError, match=message):
        _verify_cross(
            receipts,
            expected_slots_per_run=2,
            expected_run_ids=["100", "200"],
        )


def test_cross_workflow_verifier_rejects_uneven_runs_and_cross_run_duplicates() -> None:
    receipts = _cross_cohort_receipts()
    receipts[0]["run_id"] = "200"
    with pytest.raises(CiError, match="exactly 14 receipts"):
        _verify_cross(
            receipts,
            expected_slots_per_run=14,
            expected_run_ids=["100", "200"],
        )

    receipts = _cross_cohort_receipts()
    receipts[-1].update(
        {
            "node_id": receipts[0]["node_id"],
            "hostname": receipts[0]["hostname"],
            "gpu_uuid": receipts[0]["gpu_uuid"],
            "gpu_index": receipts[0]["gpu_index"],
            "gpu_slot": receipts[0]["gpu_slot"],
            "probe_gpu_uuid": receipts[0]["probe_gpu_uuid"],
            "lock_namespace": receipts[0]["lock_namespace"],
        }
    )
    with pytest.raises(CiError, match="duplicate GPU slot tuples"):
        _verify_cross(
            receipts,
            expected_slots_per_run=14,
            expected_run_ids=["100", "200"],
        )

    receipts = _cross_cohort_receipts()
    receipts[-1]["runner_name"] = receipts[0]["runner_name"]
    with pytest.raises(CiError, match="unique runner listeners"):
        _verify_cross(
            receipts,
            expected_slots_per_run=14,
            expected_run_ids=["100", "200"],
        )


def test_cross_workflow_verifier_rejects_identity_timing_and_partial_gpu_coverage() -> None:
    receipts = _cross_cohort_receipts()
    for receipt in receipts[14:]:
        receipt["cohort_id"] = "other-cohort"
    with pytest.raises(CiError, match="cohort ID"):
        _verify_cross(
            receipts,
            expected_slots_per_run=14,
            expected_run_ids=["100", "200"],
        )

    receipts = _cross_cohort_receipts()
    receipts[-1]["released_at"] = _time(9.9)
    with pytest.raises(CiError, match="released before"):
        _verify_cross(
            receipts,
            expected_slots_per_run=14,
            expected_run_ids=["100", "200"],
        )

    receipts = _cross_cohort_receipts()
    receipts[-1]["gpu_uuid"] = "GPU-b3"
    receipts[-1]["probe_gpu_uuid"] = "GPU-b3"
    with pytest.raises(CiError, match="maps to multiple UUIDs"):
        _verify_cross(
            receipts,
            expected_slots_per_run=14,
            expected_run_ids=["100", "200"],
        )


def test_cross_workflow_verifier_rejects_gpu_zero_on_excluded_node() -> None:
    receipts = _cross_cohort_receipts()
    for receipt in receipts:
        if receipt["node_id"] == "node-b" and receipt["gpu_uuid"] == "GPU-b0":
            receipt["gpu_index"] = "0"

    with pytest.raises(CiError, match="GPU index 0 is not allowed"):
        _verify_cross(
            receipts,
            expected_slots_per_run=14,
            expected_run_ids=["100", "200"],
        )


def test_cross_workflow_verifier_rejects_same_total_with_wrong_node_distribution() -> None:
    wrong_distribution = _topology(
        slots_per_gpu=4,
        nodes=[("node-a", [0, 1, 2]), ("node-b", [0, 1, 2, 3])],
    )
    receipts = _cross_cohort_receipts()
    wrong_digest = topology_contract_digest(wrong_distribution)
    for receipt in receipts:
        receipt["expected_topology_digest"] = wrong_digest

    with pytest.raises(CiError, match="GPU index 3 is not allowed"):
        _verify_cross(
            receipts,
            expected_slots_per_run=14,
            expected_run_ids=["100", "200"],
            expected_topology=wrong_distribution,
        )


def test_cross_workflow_verifier_rejects_slots_digest_and_index_uuid_spoofs() -> None:
    receipts = _cross_cohort_receipts()
    receipts[0]["gpu_slots_per_device"] = 2
    with pytest.raises(CiError, match="slots-per-GPU does not match"):
        _verify_cross(
            receipts,
            expected_slots_per_run=14,
            expected_run_ids=["100", "200"],
        )

    receipts = _cross_cohort_receipts()
    receipts[0]["expected_topology_digest"] = "f" * 64
    with pytest.raises(CiError, match="topology digest does not match"):
        _verify_cross(
            receipts,
            expected_slots_per_run=14,
            expected_run_ids=["100", "200"],
        )

    receipts = _cross_cohort_receipts()
    for receipt in receipts:
        if receipt["node_id"] == "node-a" and receipt["gpu_uuid"] == "GPU-a0":
            receipt["gpu_index"] = "1"
    with pytest.raises(CiError, match="GPU index 1 maps to multiple UUIDs"):
        _verify_cross(
            receipts,
            expected_slots_per_run=14,
            expected_run_ids=["100", "200"],
        )


def test_verifier_proves_concurrency_queueing_and_dynamic_node_capacity() -> None:
    summary = _verify_capacity(
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
    assert summary["expected_topology"] == normalize_topology_contract(SMALL_TOPOLOGY)
    assert summary["expected_topology_digest"] == topology_contract_digest(SMALL_TOPOLOGY)
    assert summary["nodes"]["node-a"]["capacity"] == 2  # type: ignore[index]
    assert summary["nodes"]["node-a"]["gpu_indices"] == [0]  # type: ignore[index]
    assert summary["nodes"]["node-b"]["capacity"] == 2  # type: ignore[index]
    assert summary["nodes"]["node-b"]["gpu_indices"] == [1]  # type: ignore[index]


def test_rollback_verifier_proves_exact_compute02_equivalent_12_plus_1_capacity() -> None:
    summary = verify_capacity_receipts(
        _rollback_receipts(),
        expected_slots=12,
        expected_topology=ROLLBACK_TOPOLOGY,
        expected_run_id="42",
        expected_revision="a" * 40,
    )

    assert summary["outcome"] == "success"
    assert summary["maximum_concurrency"] == 12
    assert summary["first_wave_runner_count"] == 12
    assert summary["first_wave_slot_count"] == 12
    assert summary["extra_leg_id"] == 12
    assert summary["expected_topology"] == normalize_topology_contract(ROLLBACK_TOPOLOGY)
    assert list(summary["nodes"]) == ["node-b"]  # type: ignore[arg-type]
    assert summary["nodes"]["node-b"]["capacity"] == 12  # type: ignore[index]
    assert summary["nodes"]["node-b"]["gpu_indices"] == [1, 2, 3]  # type: ignore[index]


def test_rollback_verifier_rejects_gpu_zero_and_non_baseline_nodes() -> None:
    receipts = _rollback_receipts()
    for receipt in receipts:
        if receipt["gpu_uuid"] == "GPU-b1":
            receipt["gpu_index"] = "0"
    with pytest.raises(CiError, match="GPU index 0 is not allowed"):
        verify_capacity_receipts(
            receipts,
            expected_slots=12,
            expected_topology=ROLLBACK_TOPOLOGY,
        )

    receipts = _rollback_receipts()
    for receipt in receipts:
        receipt["node_id"] = "node-a"
    with pytest.raises(CiError, match="outside the topology"):
        verify_capacity_receipts(
            receipts,
            expected_slots=12,
            expected_topology=ROLLBACK_TOPOLOGY,
        )


def test_verifier_rejects_an_extra_leg_that_started_before_release() -> None:
    receipts = _valid_receipts()
    receipts[-1]["worker_started_at"] = _time(9)
    with pytest.raises(CiError, match="extra matrix leg started before"):
        _verify_capacity(receipts, expected_slots=4)


def test_verifier_rejects_duplicate_first_wave_slot_or_runner() -> None:
    receipts = _valid_receipts()
    receipts[1]["gpu_slot"] = 0
    with pytest.raises(CiError, match="duplicate GPU slot tuples"):
        _verify_capacity(receipts, expected_slots=4)

    receipts = _valid_receipts()
    receipts[1]["runner_name"] = receipts[0]["runner_name"]
    with pytest.raises(CiError, match="unique runner listeners"):
        _verify_capacity(receipts, expected_slots=4)


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        ("hostname", "hostname .* maps to multiple node IDs"),
        ("gpu_uuid", "GPU UUID .* maps to multiple node IDs"),
    ],
)
def test_verifier_rejects_physical_identity_reused_by_multiple_nodes(
    identity: str, message: str
) -> None:
    receipts = _valid_receipts()
    for receipt in receipts:
        if receipt["node_id"] != "node-b":
            continue
        if identity == "hostname":
            receipt["hostname"] = "host-node-a"
        else:
            receipt["gpu_uuid"] = "GPU-aaaa"
            receipt["probe_gpu_uuid"] = "GPU-aaaa"

    with pytest.raises(CiError, match=message):
        _verify_capacity(receipts, expected_slots=4)


def test_verifier_rejects_partial_gpu_slot_coverage_on_a_node() -> None:
    receipts = _valid_receipts()
    receipts[3]["gpu_uuid"] = "GPU-cccc"
    receipts[3]["probe_gpu_uuid"] = "GPU-cccc"
    with pytest.raises(CiError, match="maps to multiple UUIDs"):
        _verify_capacity(receipts, expected_slots=4)


def test_verifier_rejects_mismatched_container_probe_and_stale_run() -> None:
    receipts = _valid_receipts()
    receipts[0]["probe_gpu_uuid"] = "GPU-other"
    with pytest.raises(CiError, match="container probe"):
        _verify_capacity(receipts, expected_slots=4)

    with pytest.raises(CiError, match="run ID"):
        _verify_capacity(_valid_receipts(), expected_slots=4, expected_run_id="43")

    receipts = _valid_receipts()
    receipts[0]["hostname"] = "../../unsafe"
    with pytest.raises(CiError, match="hostname is unsafe"):
        _verify_capacity(receipts, expected_slots=4)

    receipts = _valid_receipts()
    receipts[0]["job_id"] = "other-job"
    with pytest.raises(CiError, match="job_id does not match"):
        _verify_capacity(receipts, expected_slots=4)


def test_load_receipts_reads_only_receipt_json_files(tmp_path: Path) -> None:
    nested = tmp_path / "one"
    nested.mkdir()
    (nested / "receipt-0.json").write_text(json.dumps(_valid_receipts()[0]), encoding="utf-8")
    (nested / "holder-0.log").write_text("ignored", encoding="utf-8")
    assert load_receipts(tmp_path) == [_valid_receipts()[0]]


def test_exclusive_verifier_proves_all_slots_non_overlap_and_queued_resume() -> None:
    summary = _verify_exclusive(
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
        _verify_exclusive([receipt])

    receipt = _valid_exclusive_receipt()
    receipt["contender"]["acquired_at"] = _time(4)  # type: ignore[index]
    with pytest.raises(CiError, match="overlapped on the same GPU"):
        _verify_exclusive([receipt])

    receipt = _valid_exclusive_receipt()
    receipt["primary"]["job_id"] = "other-job"  # type: ignore[index]
    with pytest.raises(CiError, match="job_id does not match"):
        _verify_exclusive([receipt])


def test_exclusive_verifier_rejects_uuid_mismatch_and_missing_contention() -> None:
    receipt = _valid_exclusive_receipt()
    receipt["contender"]["probe_gpu_uuid"] = "GPU-other"  # type: ignore[index]
    with pytest.raises(CiError, match="container UUID"):
        _verify_exclusive([receipt])

    receipt = _valid_exclusive_receipt()
    receipt["contender"]["gpu_uuid"] = "GPU-different"  # type: ignore[index]
    receipt["contender"]["probe_gpu_uuid"] = "GPU-different"  # type: ignore[index]
    with pytest.raises(CiError, match="maps to multiple UUIDs"):
        _verify_exclusive([receipt])

    receipt = _valid_exclusive_receipt()
    receipt["contender"]["attempted_at"] = _time(5.01)  # type: ignore[index]
    receipt["contender"]["worker_started_at"] = _time(5.01)  # type: ignore[index]
    with pytest.raises(CiError, match="did not attempt while"):
        _verify_exclusive([receipt])


def test_exclusive_verifier_rejects_inconsistent_or_unsafe_hostname() -> None:
    receipt = _valid_exclusive_receipt()
    receipt["contender"]["hostname"] = "host-selected-alias"  # type: ignore[index]
    with pytest.raises(CiError, match="inconsistent hostname or lock namespace"):
        _verify_exclusive([receipt])

    receipt = _valid_exclusive_receipt()
    receipt["primary"]["hostname"] = "../../unsafe"  # type: ignore[index]
    with pytest.raises(CiError, match="primary hostname is unsafe"):
        _verify_exclusive([receipt])


def test_holder_emits_acquisition_marker_and_writes_cohort_release_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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
        expected_topology_digest=topology_contract_digest(SMALL_TOPOLOGY),
        cohort_id="cohort-42",
        receipt_output=receipt_output,
        lease_factory=lease_factory,  # type: ignore[arg-type]
        probe=probe,
        now=lambda: next(now_values),
        sleep=sleeps.append,
    )
    assert receipt_output.is_file()
    assert receipt["barrier_epoch"] == 110
    assert receipt["cohort_id"] == "cohort-42"
    assert receipt["released_at"] is not None
    assert receipt["probe_gpu_uuid"] == "GPU-aaaa"
    assert receipt["expected_topology_digest"] == topology_contract_digest(SMALL_TOPOLOGY)
    assert events == ["acquire", "mark_released", "release"]
    assert sleeps == [10.0]
    assert probe_calls == [
        {
            "gpu_index": "0",
            "image": "local-image:canary",
            "container_name": "trtmc-capacity-42-1-0",
        }
    ]
    output = capsys.readouterr().out
    marker_lines = [
        line for line in output.splitlines() if line.startswith("TRTMC_CAPACITY_CANARY_ACQUIRED=")
    ]
    assert len(marker_lines) == 1
    marker = json.loads(marker_lines[0].split("=", maxsplit=1)[1])
    assert marker["kind"] == "trtmc_capacity_canary_acquired"
    assert marker["cohort_id"] == "cohort-42"
    assert marker["run_id"] == "42"
    assert marker["gpu_uuid"] == "GPU-aaaa"
    assert marker["probe_gpu_uuid"] == "GPU-aaaa"
    assert marker["expected_topology_digest"] == topology_contract_digest(SMALL_TOPOLOGY)


def test_holder_rejects_unsafe_cohort_id_before_acquiring(tmp_path: Path) -> None:
    context = CiContext(repository=tmp_path, env={"GITHUB_SHA": "a" * 40})
    called = False

    def lease_factory(*_: object, **__: object) -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(CiError, match="cohort_id"):
        hold_capacity_slot(
            context=context,
            leg_id=0,
            expected_slots=1,
            barrier_epoch=110,
            expected_topology_digest=topology_contract_digest(SMALL_TOPOLOGY),
            cohort_id="../../unsafe",
            receipt_output=tmp_path / "receipt-0.json",
            lease_factory=lease_factory,  # type: ignore[arg-type]
        )
    assert called is False


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
    assert "\n  exercise:\n" in workflow
    assert '_CAPACITY_WORKER_JOB_ID = "exercise"' in source
    assert "pull_request:" not in workflow
    assert "schedule:" not in workflow
    assert "refs/heads/main" in workflow
    assert "default: 28" not in workflow
    assert "default: 900" in workflow
    assert "default: shared-capacity" in workflow
    assert "- rollback-capacity" in workflow
    assert "shared-capacity|rollback-capacity)" in workflow
    assert "      expected_topology:" not in workflow
    assert "inputs.expected_topology" not in workflow
    assert "      node_label:" not in workflow
    assert "inputs.node_label" not in workflow
    assert "inputs.rollback_baseline" not in workflow
    assert "--input .github/ci/gb300-pool-topology.json" in workflow
    assert "tools.ci.capacity_canary topology-contract" in workflow
    assert "expected_topology_digest" in workflow
    assert "--expected-topology-digest" in workflow
    assert workflow.count("--expected-topology \"$EXPECTED_TOPOLOGY_FILE\"") == 4
    assert "trtmc-expected-topology.json" in workflow
    assert "- shared-cohort" in workflow
    assert "- cross-workflow-verify" in workflow
    assert "- exclusive-safety" in workflow
    assert "matrix='{\"leg\":[0]}'" in workflow
    assert "tools.ci.capacity_canary matrix" in workflow
    assert "inputs.mode == 'rollback-capacity'" in workflow
    assert "cohort-matrix --cohort-slots" in workflow
    assert "verify-cohort" in workflow
    assert "verify-cross" in workflow
    assert re.search(
        r"verify-cross[\s\S]*?--expected-revision \"\$GITHUB_SHA\"",
        workflow,
    )
    assert workflow.count('if [ "$GITHUB_RUN_ATTEMPT" != "1" ]') == 3
    assert workflow.count('gh api "repos/$GITHUB_REPOSITORY/actions/runs/$RUN_') == 2
    assert workflow.count("--run-metadata") == 2
    assert '--expected-repository "$GITHUB_REPOSITORY"' in workflow
    assert "trtmc-cross-run-metadata/*.json" in workflow
    assert "max-parallel:" not in workflow
    assert "concurrency:" not in workflow
    assert "TRTMC_MODEL_RUNNER_LABELS" in workflow
    assert (
        "runs-on: ${{ fromJSON(vars.TRTMC_MODEL_RUNNER_LABELS || "
        "'[\"trtmc-gb300-proof\"]') }}" in workflow
    )
    assert "runs-on: ${{ inputs" not in workflow
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
    assert "TRTMC_CAPACITY_CANARY_ACQUIRED=" in source
    assert "verify-cancellation" not in source
    assert "verify-exclusive" in workflow
    assert "one scheduler-selected generic runner" in source
    assert "nohup" not in workflow
    assert "release_file" not in source
    assert "GITHUB_TOKEN" not in workflow
    assert "actions: read" in workflow
    assert "actions: write" not in workflow
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
