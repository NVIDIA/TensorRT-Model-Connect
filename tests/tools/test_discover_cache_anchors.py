# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for label-driven Nightly cache-anchor discovery."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci.discover_cache_anchors import discover_cache_anchors
from tools.ci.process import CiError


REPO_ROOT = Path(__file__).resolve().parents[2]


def _runner(
    runner_id: int,
    name: str,
    *labels: str,
    status: str = "online",
) -> dict[str, object]:
    return {
        "id": runner_id,
        "name": name,
        "status": status,
        "busy": False,
        "labels": [{"name": label} for label in labels],
    }


def _valid_inventory() -> dict[str, object]:
    return {
        "total_count": 4,
        "runners": [
            _runner(
                1,
                "node-a-proof-00",
                "self-hosted",
                "trtmc-gb300-proof",
                "trtmc-cache-anchor",
                "trtmc-node-node-a",
            ),
            _runner(
                2,
                "node-a-proof-01",
                "self-hosted",
                "trtmc-gb300-proof",
                "trtmc-node-node-a",
            ),
            _runner(
                3,
                "node-b-proof-00",
                "self-hosted",
                "trtmc-cache-anchor",
                "trtmc-node-node-b",
            ),
            _runner(4, "generic-runner", "self-hosted"),
        ],
    }


def test_discovery_emits_one_sorted_entry_per_anchor_node() -> None:
    matrix = discover_cache_anchors(_valid_inventory())

    assert matrix == {
        "include": [
            {
                "anchor_runner": "node-a-proof-00",
                "node_label": "trtmc-node-node-a",
            },
            {
                "anchor_runner": "node-b-proof-00",
                "node_label": "trtmc-node-node-b",
            },
        ]
    }


def test_discovery_accepts_slurped_paginated_api_output() -> None:
    inventory = _valid_inventory()
    runners = inventory["runners"]
    assert isinstance(runners, list)

    matrix = discover_cache_anchors(
        [{"total_count": 4, "runners": runners[:2]}, {"total_count": 4, "runners": runners[2:]}]
    )

    assert len(matrix["include"]) == 2


@pytest.mark.parametrize(
    "inventory",
    [
        {"total_count": 5, "runners": _valid_inventory()["runners"]},
        [
            {"total_count": 4, "runners": _valid_inventory()["runners"][:2]},
            {"total_count": 5, "runners": _valid_inventory()["runners"][2:]},
        ],
    ],
)
def test_discovery_rejects_truncated_or_inconsistent_pagination(inventory: object) -> None:
    with pytest.raises(CiError, match="truncated|disagree on total_count"):
        discover_cache_anchors(inventory)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda runners: runners.append(
                _runner(
                    5,
                    "node-a-proof-02",
                    "trtmc-cache-anchor",
                    "trtmc-node-node-a",
                )
            ),
            "duplicate cache anchors",
        ),
        (
            lambda runners: runners[0]["labels"].append(  # type: ignore[index,union-attr]
                {"name": "trtmc-node-second"}
            ),
            "exactly one",
        ),
        (
            lambda runners: runners[0].update({"status": "offline"}),  # type: ignore[union-attr]
            "is offline",
        ),
        (
            lambda runners: runners[0].update(  # type: ignore[union-attr]
                {
                    "labels": [
                        {"name": "trtmc-gb300-proof"},
                        {"name": "trtmc-node-node-a"},
                    ]
                }
            ),
            "have no cache anchor",
        ),
        (
            lambda runners: [
                runner.update(  # type: ignore[union-attr]
                    {
                        "labels": [
                            label
                            for label in runner["labels"]  # type: ignore[index,union-attr]
                            if label["name"] != "trtmc-cache-anchor"  # type: ignore[index]
                        ]
                    }
                )
                for runner in runners
            ],
            "no online cache anchors",
        ),
    ],
)
def test_discovery_fails_closed_on_invalid_topology(mutate, message: str) -> None:
    inventory = _valid_inventory()
    runners = inventory["runners"]
    assert isinstance(runners, list)
    mutate(runners)
    inventory["total_count"] = len(runners)

    with pytest.raises(CiError, match=message):
        discover_cache_anchors(inventory)


def test_discovery_cli_writes_compact_github_matrix(tmp_path: Path) -> None:
    inventory = tmp_path / "runners.json"
    github_output = tmp_path / "github-output"
    inventory.write_text(json.dumps(_valid_inventory()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.ci.discover_cache_anchors",
            "--input",
            str(inventory),
            "--github-output",
            str(github_output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    line = github_output.read_text(encoding="utf-8").strip()
    assert line.startswith("matrix=")
    assert len(json.loads(line.removeprefix("matrix="))["include"]) == 2
