# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tools.execution_ledger import ExecutionLedger, ExecutionLedgerError


CASES = [
    {"id": "model-a::task-a", "model": "model-a", "task": "task-a"},
    {"id": "model-b::task-b", "model": "model-b", "task": "task-b"},
]


def _ledger(tmp_path) -> ExecutionLedger:
    return ExecutionLedger.open(
        tmp_path,
        campaign_id="run-1",
        task_kind="accuracy",
        fingerprint="revision-1",
        cases=CASES,
    )


def test_open_creates_an_ordered_campaign_and_pending_receipts(tmp_path) -> None:
    ledger = _ledger(tmp_path)

    assert [case["id"] for case in ledger.cases()] == [
        "model-a::task-a",
        "model-b::task-b",
    ]
    assert [entry["receipt"]["state"] for entry in ledger.snapshot()] == [
        "pending",
        "pending",
    ]
    manifest = json.loads((tmp_path / "ledger" / "campaign.json").read_text())
    assert manifest["fingerprint"] == "revision-1"
    assert not list((tmp_path / "ledger").rglob("*.tmp"))


def test_attempt_can_progress_and_publish_one_terminal_receipt(tmp_path) -> None:
    ledger = _ledger(tmp_path)

    assert ledger.begin(
        "model-a::task-a",
        stage="candidate",
        evidence={"command": "run candidate"},
    ) == 1
    ledger.update_stage(
        "model-a::task-a",
        "compare",
        evidence={"log": "candidate.log"},
    )
    ledger.finish(
        "model-a::task-a",
        result="green",
        payload={"metric": 1.0},
        evidence={"return_code": 0},
    )

    receipt = ledger.receipt("model-a::task-a")
    assert receipt["state"] == "terminal"
    assert receipt["result"] == "green"
    assert receipt["payload"] == {"metric": 1.0}
    assert receipt["attempts"] == [
        {
            "attempt": 1,
            "state": "completed",
            "stage": "compare",
            "started_at": receipt["attempts"][0]["started_at"],
            "finished_at": receipt["attempts"][0]["finished_at"],
            "evidence": {
                "command": "run candidate",
                "log": "candidate.log",
                "return_code": 0,
            },
        }
    ]
    with pytest.raises(ExecutionLedgerError, match="terminal"):
        ledger.begin("model-a::task-a", stage="candidate")


@pytest.mark.parametrize(
    "stage",
    ["preflight", "build", "reference", "candidate", "compare-or-measure"],
)
def test_resume_marks_running_attempt_interrupted_before_retry(tmp_path, stage) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("model-a::task-a", stage=stage)

    reopened = _ledger(tmp_path)
    assert reopened.recover_interrupted() == ["model-a::task-a"]
    receipt = reopened.receipt("model-a::task-a")
    assert receipt["state"] == "pending"
    assert receipt["attempts"][0]["state"] == "interrupted"
    assert receipt["attempts"][0]["stage"] == stage
    assert reopened.begin("model-a::task-a", stage="candidate") == 2


def test_retry_closes_only_the_active_attempt_before_starting_the_next(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("model-a::task-a", stage="reference")

    ledger.retry(
        "model-a::task-a",
        attempt_outcome="failed",
        evidence={"return_code": 1, "retryable": True},
    )

    pending = ledger.receipt("model-a::task-a")
    assert pending["state"] == "pending"
    assert pending["stage"] is None
    assert pending["active_attempt"] is None
    assert pending["attempts"][0]["state"] == "failed"
    assert pending["attempts"][0]["stage"] == "reference"
    assert pending["attempts"][0]["evidence"] == {
        "return_code": 1,
        "retryable": True,
    }
    assert ledger.begin("model-a::task-a", stage="reference") == 2


def test_open_rejects_campaign_identity_or_inventory_drift(tmp_path) -> None:
    _ledger(tmp_path)

    with pytest.raises(ExecutionLedgerError, match="fingerprint"):
        ExecutionLedger.open(
            tmp_path,
            campaign_id="run-1",
            task_kind="accuracy",
            fingerprint="revision-2",
            cases=CASES,
        )
    with pytest.raises(ExecutionLedgerError, match="case inventory"):
        ExecutionLedger.open(
            tmp_path,
            campaign_id="run-1",
            task_kind="accuracy",
            fingerprint="revision-1",
            cases=CASES[:1],
        )


def test_invalid_transitions_and_results_are_rejected(tmp_path) -> None:
    ledger = _ledger(tmp_path)

    with pytest.raises(ExecutionLedgerError, match="running"):
        ledger.finish("model-a::task-a", result="red", payload={})
    ledger.begin("model-a::task-a", stage="candidate")
    with pytest.raises(ExecutionLedgerError, match="traffic-light"):
        ledger.finish("model-a::task-a", result="blue", payload={})
    with pytest.raises(ExecutionLedgerError, match="unknown case"):
        ledger.receipt("missing")


def test_snapshot_is_deterministic_and_ordered_by_campaign_inventory(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("model-b::task-b", stage="candidate")
    ledger.finish("model-b::task-b", result="white", payload={"error": "failed"})

    snapshot = ledger.snapshot()

    assert [row["case"]["id"] for row in snapshot] == [
        "model-a::task-a",
        "model-b::task-b",
    ]
    assert snapshot[0]["receipt"]["state"] == "pending"
    assert snapshot[1]["receipt"]["result"] == "white"


def test_load_reopens_a_campaign_without_redeclaring_its_inventory(tmp_path) -> None:
    created = _ledger(tmp_path)
    created.begin("model-a::task-a", stage="candidate")

    loaded = ExecutionLedger.load(tmp_path, task_kind="accuracy")

    assert loaded.receipt("model-a::task-a")["state"] == "running"
    with pytest.raises(ExecutionLedgerError, match="task"):
        ExecutionLedger.load(tmp_path, task_kind="performance")


def test_existing_campaign_rejects_a_missing_case_receipt(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    receipt_path = ledger.root / ledger._manifest["cases"][0]["receipt"]
    receipt_path.unlink()

    with pytest.raises(ExecutionLedgerError, match="missing receipt"):
        _ledger(tmp_path)


def test_load_rejects_corrupt_attempt_history(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("model-a::task-a", stage="candidate")
    receipt_path = ledger.root / ledger._manifest["cases"][0]["receipt"]
    receipt = json.loads(receipt_path.read_text())
    receipt["attempts"][0]["attempt"] = 7
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(ExecutionLedgerError, match="attempt history"):
        ExecutionLedger.load(tmp_path)


def test_explicit_resume_reopens_only_retryable_white_results(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("model-a::task-a", stage="candidate")
    ledger.finish(
        "model-a::task-a",
        result="white",
        payload={"status": "failed"},
        attempt_outcome="failed",
        evidence={"retryable": True},
    )
    ledger.begin("model-b::task-b", stage="compare")
    ledger.finish(
        "model-b::task-b",
        result="white",
        payload={"status": "contract-mismatch"},
        evidence={"retryable": False},
    )

    assert ledger.reopen_retryable() == ["model-a::task-a"]
    reopened = ledger.receipt("model-a::task-a")
    assert reopened["state"] == "pending"
    assert reopened["result"] is None
    assert len(reopened["previous_results"]) == 1
    previous = reopened["previous_results"][0]
    assert previous["attempt"] == 1
    assert previous["result"] == "white"
    assert previous["payload"] == {"status": "failed"}
    assert previous["finished_at"]
    assert ledger.begin("model-a::task-a", stage="candidate") == 2
    assert ledger.receipt("model-b::task-b")["state"] == "terminal"


def test_load_rejects_conflicting_case_and_attempt_stage(tmp_path) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("model-a::task-a", stage="candidate")
    receipt_path = ledger.root / ledger._manifest["cases"][0]["receipt"]
    receipt = json.loads(receipt_path.read_text())
    receipt["stage"] = "reference"
    receipt_path.write_text(json.dumps(receipt))

    with pytest.raises(ExecutionLedgerError, match="stage differ"):
        ExecutionLedger.load(tmp_path)
