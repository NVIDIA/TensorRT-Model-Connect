# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from tools import campaign_shards, qualification_report
from tools.execution_ledger import ExecutionLedger


def test_parse_shard_uses_one_zero_based_index_count_value():
    assert campaign_shards.parse_shard("2/4") == (2, 4)
    with pytest.raises(campaign_shards.CampaignShardError, match="0 <= INDEX < COUNT"):
        campaign_shards.parse_shard("4/4")


def test_round_robin_assignment_is_deterministic_and_complete():
    cases = tuple(f"case-{index}" for index in range(7))

    assignments = [
        campaign_shards.assign_cases(cases, index=index, count=3) for index in range(3)
    ]

    assert assignments == [
        ("case-0", "case-3", "case-6"),
        ("case-1", "case-4"),
        ("case-2", "case-5"),
    ]
    assert set().union(*map(set, assignments)) == set(cases)
    assert not any(
        set(left).intersection(right)
        for left, right in zip(assignments, assignments[1:])
    )
    assert campaign_shards.assign_cases(cases, index=0, count=1) == cases


def test_only_one_consolidator_can_publish_a_campaign(tmp_path: Path):
    with campaign_shards.consolidator_lock(tmp_path):
        with pytest.raises(
            campaign_shards.CampaignShardError,
            match="another campaign consolidator",
        ):
            with campaign_shards.consolidator_lock(tmp_path):
                pass


def test_campaign_manifest_is_immutable(tmp_path: Path):
    campaign_shards.open_campaign(tmp_path, {"run_id": "one"})

    with pytest.raises(campaign_shards.CampaignShardError, match="does not match"):
        campaign_shards.open_campaign(tmp_path, {"run_id": "two"})


def _shard_report(root: Path, case_id: str, result: str) -> None:
    ledger = ExecutionLedger.open(
        root,
        campaign_id=root.name,
        task_kind="accuracy",
        fingerprint="same-input",
        cases=[{"id": case_id, "report": {"model": case_id, "workload": "suite"}}],
    )
    log = root / "artifacts" / case_id / "logs" / "run.log"
    log.parent.mkdir(parents=True)
    log.write_text(f"{case_id}\n", encoding="utf-8")
    ledger.begin(case_id, stage="compare", evidence={})
    ledger.finish(case_id, result=result, payload={"status": result})
    qualification_report.materialize_report(
        root,
        report_kind="accuracy",
        title="Shard",
        identity={"run_id": root.name, "disposition": "passed"},
        run={"hostname": root.name},
        results=[
            {
                "id": case_id,
                "model": case_id,
                "workload": "suite",
                "state": "terminal",
                "result": result,
                "precision": {"reference": "fp16", "candidate": "fp16"},
                "debug": {
                    "logs": [
                        {
                            "label": "run.log",
                            "href": log.relative_to(root).as_posix(),
                        }
                    ],
                    "command_artifacts": [],
                },
            }
        ],
    )


def test_merge_uses_receipts_keeps_missing_cases_pending_and_relocates_artifacts(
    tmp_path: Path,
):
    shard = tmp_path / "shard-0"
    output = tmp_path / "campaign" / "accuracy"
    _shard_report(shard, "case-a", "green")

    _, _, report = campaign_shards.merge_receipt_reports(
        output,
        report_kind="accuracy",
        campaign={
            "run_id": "campaign",
            "revision": "abc",
            "platform": "test",
            "shard_count": 2,
        },
        expected_cases=[
            {
                "id": "case-a",
                "shard": 0,
                "report": {"model": "case-a", "workload": "suite"},
            },
            {
                "id": "case-b",
                "shard": 1,
                "report": {"model": "case-b", "workload": "suite"},
            },
        ],
        shard_outputs=[("000-of-002", shard)],
    )

    assert [(row["id"], row["state"], row["result"]) for row in report["results"]] == [
        ("case-a", "terminal", "green"),
        ("case-b", "pending", None),
    ]
    href = report["results"][0]["debug"]["logs"][0]["href"]
    assert href == "artifacts/shards/000-of-002/artifacts/case-a/logs/run.log"
    assert (output / href).read_text(encoding="utf-8") == "case-a\n"
    receipt_href = report["receipt_sources"]["case-a"]
    assert (output / receipt_href).is_file()
    shard_environment_href = report["run"]["shards"][0]["environment_href"]
    assert (output / shard_environment_href).is_file()
    assert report["accounting"]["progress"] == {
        "pending": 1,
        "running": 0,
        "terminal": 1,
    }

    source_log = shard / "artifacts" / "case-a" / "logs" / "run.log"
    source_log.write_text("case-a\ncontinued\n", encoding="utf-8")
    campaign_shards.merge_receipt_reports(
        output,
        report_kind="accuracy",
        campaign={
            "run_id": "campaign",
            "revision": "abc",
            "platform": "test",
            "shard_count": 2,
        },
        expected_cases=[
            {
                "id": "case-a",
                "shard": 0,
                "report": {"model": "case-a", "workload": "suite"},
            },
            {
                "id": "case-b",
                "shard": 1,
                "report": {"model": "case-b", "workload": "suite"},
            },
        ],
        shard_outputs=[("000-of-002", shard)],
    )
    assert (output / href).read_text(encoding="utf-8") == "case-a\ncontinued\n"


def test_merge_rejects_a_case_from_the_wrong_shard(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _shard_report(left, "case-a", "green")
    _shard_report(right, "case-a", "green")

    with pytest.raises(campaign_shards.CampaignShardError, match="does not match"):
        campaign_shards.merge_receipt_reports(
            tmp_path / "output",
            report_kind="accuracy",
            campaign={"run_id": "campaign", "shard_count": 2},
            expected_cases=[{"id": "case-a", "shard": 0, "report": {}}],
            shard_outputs=[("000-of-002", left), ("001-of-002", right)],
        )
