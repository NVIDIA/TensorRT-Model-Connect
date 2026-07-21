# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one fail-closed Nightly cache-warm matrix entry per GPU node.

Boundary: runner-inventory validation only; this module never changes labels or services.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .process import CiError


PRODUCTION_LABEL = "trtmc-gb300-proof"
ANCHOR_LABEL = "trtmc-cache-anchor"
NODE_LABEL_PATTERN = re.compile(r"trtmc-node-[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?")


def _runner_pages(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, dict):
        pages = [payload]
    elif isinstance(payload, list) and all(isinstance(page, dict) for page in payload):
        pages = payload
    else:
        raise CiError("runner inventory must be an object or a list of page objects")
    if not pages:
        raise CiError("runner inventory contains no pages")
    return pages


def _label_names(runner: dict[str, object], name: str) -> set[str]:
    raw_labels = runner.get("labels")
    if not isinstance(raw_labels, list):
        raise CiError(f"runner {name!r} has an invalid label list")
    labels: set[str] = set()
    for raw_label in raw_labels:
        if not isinstance(raw_label, dict) or not isinstance(raw_label.get("name"), str):
            raise CiError(f"runner {name!r} has an invalid label entry")
        label = str(raw_label["name"])
        if label in labels:
            raise CiError(f"runner {name!r} has duplicate label {label!r}")
        labels.add(label)
    return labels


def discover_cache_anchors(payload: object) -> dict[str, list[dict[str, str]]]:
    """Validate the runner topology and return one online anchor per node."""

    runners: list[dict[str, object]] = []
    seen_runner_ids: set[int] = set()
    expected_total: int | None = None
    for page in _runner_pages(payload):
        total_count = page.get("total_count")
        if isinstance(total_count, bool) or not isinstance(total_count, int) or total_count < 0:
            raise CiError("runner inventory page has an invalid total_count")
        if expected_total is None:
            expected_total = total_count
        elif total_count != expected_total:
            raise CiError("runner inventory pages disagree on total_count")
        raw_runners = page.get("runners")
        if not isinstance(raw_runners, list):
            raise CiError("runner inventory page has no valid runners list")
        for raw_runner in raw_runners:
            if not isinstance(raw_runner, dict):
                raise CiError("runner inventory contains a non-object runner")
            runner_id = raw_runner.get("id")
            if not isinstance(runner_id, int) or runner_id <= 0:
                raise CiError("runner inventory contains an invalid runner ID")
            if runner_id in seen_runner_ids:
                raise CiError(f"runner inventory contains duplicate runner ID {runner_id}")
            seen_runner_ids.add(runner_id)
            runners.append(raw_runner)
    if expected_total != len(runners):
        raise CiError(
            f"runner inventory is truncated: expected {expected_total}, found {len(runners)}"
        )

    production_nodes: set[str] = set()
    anchors: dict[str, str] = {}
    for runner in runners:
        name = runner.get("name")
        status = runner.get("status")
        if not isinstance(name, str) or not name:
            raise CiError("runner inventory contains an invalid runner name")
        if status not in {"online", "offline"}:
            raise CiError(f"runner {name!r} has invalid status {status!r}")
        labels = _label_names(runner, name)
        is_production = PRODUCTION_LABEL in labels
        is_anchor = ANCHOR_LABEL in labels
        if not (is_production or is_anchor):
            continue
        node_labels = sorted(label for label in labels if NODE_LABEL_PATTERN.fullmatch(label))
        if len(node_labels) != 1:
            raise CiError(
                f"runner {name!r} must have exactly one trtmc-node-* label; found {node_labels!r}"
            )
        node_label = node_labels[0]
        if is_production:
            production_nodes.add(node_label)
        if is_anchor:
            if status != "online":
                raise CiError(f"cache anchor {name!r} for {node_label!r} is offline")
            previous = anchors.get(node_label)
            if previous is not None:
                raise CiError(
                    f"node {node_label!r} has duplicate cache anchors: {previous!r}, {name!r}"
                )
            anchors[node_label] = name

    if not anchors:
        raise CiError("runner inventory contains no online cache anchors")
    missing = sorted(production_nodes - anchors.keys())
    if missing:
        raise CiError(f"production proof node(s) have no cache anchor: {missing!r}")

    return {
        "include": [
            {"node_label": node_label, "anchor_runner": anchors[node_label]}
            for node_label in sorted(anchors)
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args()

    payload = json.loads(arguments.input.read_text(encoding="utf-8"))
    matrix = discover_cache_anchors(payload)
    serialized = json.dumps(matrix, separators=(",", ":"), sort_keys=True)
    if arguments.github_output:
        with arguments.github_output.open("a", encoding="utf-8") as output:
            output.write(f"matrix={serialized}\n")
    else:
        print(json.dumps(matrix, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
