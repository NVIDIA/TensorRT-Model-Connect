# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh benchmark timing reports from saved responses, without inference or regrading."""

from __future__ import annotations

import hashlib
import json
import math
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
    # Recover older classifications only from the matching original response.
    # Empty HTTP 200 answers remain wrong for accuracy, but have valid timings.
    raw, recovered, classified = {}, 0, 0
    for batch in batches:
        for row in batch["records"]:
            capacity = (batch["workload"] in excluded and batch["identity"].get("side") == "candidate"
                        and "capacity_rejection" not in row)
            timed = row.get("model_call_ms") is not None
            if row["output_valid"] or not (capacity or timed):
                continue
            ref = row["output_ref"]
            directory = ref["aiperf_run"]
            if directory not in raw:
                raw[directory] = AiperfRun(Path(directory), batch["aiperf_exit"], []).raw_records()
            source = raw[directory][ref["record_index"]]
            payload = source.get("payload") or {}
            sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            metadata = source.get("metadata") or {}
            sample = metadata.get("conversation_id") or metadata.get("session_num", ref["record_index"])
            if row["request_sha"] != sha or str(row["sample_id"]) != str(sample):
                raise ValueError("saved response does not match its execution identity")
            if capacity:
                row["capacity_rejection"] = absolute.capacity_rejection(source)
                classified += 1
            if (timed and (source.get("error") or {}).get("type") == "InvalidInferenceResultError"
                    and not absolute.unanswered(source) and not metadata.get("was_cancelled")):
                body = execution.response_body(source)
                ms = (body.get("trtmc_timing") or {}).get("model_call_ms")
                if ms is None or float(ms) != row["model_call_ms"] or not math.isfinite(float(ms)) or float(ms) <= 0:
                    raise ValueError("saved response does not match its execution timing")
                operation = batch["identity"].get("operation") or report.get("operation")
                row.update(valid=True, output_valid=True,
                           work=judge.work_signature(operation, execution.observation(body))
                           if operation else None)
                recovered += 1
    same_work = lambda mine, theirs: judge.work_check(  # noqa: E731
        {"work": [mine] if mine is not None else []}, {"work": [theirs] if theirs is not None else []}) is None
    evidence = execution.Session(out, {}, lambda: None, lambda value: value, same_work, batches=batches)
    performance = [item for item in report.get("performance", []) if item.get("kind") != "natural_dataset"]
    updated = {**report, "performance": [*performance, *evidence.natural_performance(report.get("accuracy", []))]}
    if recovered or classified:
        path = out / "execution.timing-recovered.jsonl"
        path.write_text("".join(json.dumps(batch) + "\n" for batch in batches))
        updated["execution"] = {**report["execution"], "records": str(path),
                                "timing_recovered_records": recovered}
    return updated
