# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh benchmark timing reports from saved responses, without inference or regrading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import absolute, execution, judge
from .aiperf_runner import AiperfRun


def refresh(out: Path, report: dict) -> dict:
    if report.get("performance_source") != "quality":
        return report
    recorded_path = Path((report.get("execution") or {}).get("records", "execution.jsonl"))
    path = out / recorded_path.name
    if not path.is_file():
        if not report.get("execution") and not any(item.get("out_of_capacity") for item in report.get("accuracy", [])):
            # Older aggregate-only reports can reapply the measurement verdict,
            # but cannot reconstruct a different sample selection.
            return report
        raise ValueError(f"{out}: recorded benchmark execution is unavailable")
    batches, superseded = [], set()
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        batch = json.loads(line)
        if batch.get("event") == "supersede_failed_attempt":
            superseded.update(batch["batch_ids"])
        elif "records" in batch:
            batches.append(batch)
    batches = [batch for batch in batches
               if not batch.get("superseded") and batch["batch_id"] not in superseded]
    excluded = {item["suite"] for item in report.get("accuracy", [])
                if item.get("out_of_capacity") and item.get("status") != "error"}
    # Older execution exports omit rejection reasons. Read their original raw
    # error records only when accuracy explicitly excluded capacity rejections.
    raw = {}
    for batch in batches:
        if batch["workload"] not in excluded or batch["identity"].get("side") != "candidate":
            continue
        for row in batch["records"]:
            if row["output_valid"] or "capacity_rejection" in row:
                continue
            ref = row["output_ref"]
            directory = ref["aiperf_run"]
            if directory not in raw:
                raw[directory] = AiperfRun(Path(directory), batch["aiperf_exit"], []).raw_records()
            source = raw[directory][ref["record_index"]]
            payload = source.get("payload") or {}
            sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if row["request_sha"] != sha or str(row["sample_id"]) != source["metadata"].get("conversation_id"):
                raise ValueError("saved capacity rejection does not match its execution identity")
            row["capacity_rejection"] = absolute.capacity_rejection(source)
    same_work = lambda mine, theirs: judge.work_check(  # noqa: E731
        {"work": [mine] if mine is not None else []}, {"work": [theirs] if theirs is not None else []}) is None
    evidence = execution.Session(out, {}, lambda: None, lambda value: value, same_work, batches=batches)
    performance = [item for item in report.get("performance", []) if item.get("kind") != "natural_dataset"]
    return {**report, "performance": [*performance, *evidence.natural_performance(report.get("accuracy", []))]}
