# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover gold-scored accuracy from recorded outputs without running models."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from . import absolute, execution
from .aiperf_runner import AiperfRun

ALIGNMENT = "conversation-id/v1"


def text(payload: dict) -> str:
    if "prompt" in payload:
        return payload["prompt"]
    return "\n".join(message["content"] for message in payload.get("messages", []))


def fingerprint(texts) -> str:
    return hashlib.sha256(json.dumps(list(texts), ensure_ascii=False).encode()).hexdigest()


class SelectionArchive:
    """Find an original selection by its entire ordered exported input dataset."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._catalog = None

    def selection(self, run: AiperfRun) -> list[dict]:
        inputs = run.json("inputs.json").get("data", [])
        if not inputs or any(len(item["payloads"]) != 1 for item in inputs):
            raise ValueError(f"{run.directory}: missing or multi-turn accuracy inputs")
        key = fingerprint(text(item["payloads"][0]) for item in inputs)
        if self._catalog is None:
            catalog = {}
            for path in sorted(self.directory.glob("*.json")):
                problems = json.loads(path.read_text())
                for prompts in ([p["prompt"] for p in problems],
                                ["\n".join(m["content"] for m in p["raw_messages"]) if p.get("raw_messages")
                                 else p["prompt"] for p in problems]):
                    catalog.setdefault(fingerprint(prompts), set()).add(path)
            self._catalog = catalog
        matches = self._catalog.get(key, [])
        if not matches:
            raise ValueError(f"{run.directory}: no original cached selection matches every exported input")
        found = None
        for path in sorted(matches):
            problems = json.loads(path.read_text())
            selected = [{"gold": p["ground_truth"], "task": p.get("task")} for p in problems]
            if found is not None and selected != found:
                raise ValueError(f"{run.directory}: conflicting original gold selections")
            found = selected
        return found


async def regrade(run: AiperfRun, problems: list[dict]) -> list[dict]:
    """Keep upstream answer extraction and graders; replace only the selected gold/task."""
    from aiperf.plugin import plugins
    from aiperf.plugin.enums import PluginType

    indices = run.conversation_indices()
    graders = {}
    corrected = []
    try:
        for record in run.accuracy_records():
            identity = record.get("conversation_id")
            if identity not in indices:
                raise ValueError(f"accuracy conversation {identity!r} is absent from saved inputs")
            if "model_output" not in record:
                raise ValueError(f"{run.directory}: original model output is missing")
            problem = problems[indices[identity]]
            name = record["grader_name"]
            if name not in graders:
                graders[name] = plugins.get_class(PluginType.ACCURACY_GRADER, name)(run=None)
            result = await graders[name].grade(record["model_output"], problem["gold"])
            corrected.append({**record, "task": problem["task"], "passed": result.correct,
                "unparsed": result.unparsed, "confidence": result.confidence, "expected": result.ground_truth,
                "actual": result.extracted_answer, "explanation": result.reasoning, "alignment": ALIGNMENT})
    finally:
        for grader in graders.values():
            await grader.aclose()
    return corrected


def batches(out: Path) -> list[dict]:
    path = out / "execution.jsonl"
    values, superseded = [], set()
    for line in path.read_text().split("\n") if path.is_file() else []:
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("event") == "supersede_failed_attempt":
            superseded.update(item["batch_ids"])
        elif "records" in item:
            values.append(item)
    return [item for item in values if not item.get("superseded") and item["batch_id"] not in superseded]


def aligned_batches(recorded: list[dict]) -> list[dict]:
    """Reassociate saved timing rows with their actual inputs without changing timings or work."""
    aligned = []
    for batch in recorded:
        if not batch["records"]:
            aligned.append(batch)
            continue
        roots = {row["output_ref"]["aiperf_run"] for row in batch["records"]}
        if len(roots) != 1:
            raise ValueError("execution batch contains multiple run directories")
        run = AiperfRun(Path(next(iter(roots))), batch["aiperf_exit"], [])
        raw = run.raw_records()
        indices = run.conversation_indices()
        units = {str(row["sample_id"]): row["unit_id"] for row in batch["records"]}
        rows = []
        for row in batch["records"]:
            source = raw[row["output_ref"]["record_index"]]
            payload = source.get("payload") or {}
            request_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if row["request_sha"] != request_sha:
                raise ValueError("execution row does not match its saved raw request")
            identity = source["metadata"].get("conversation_id")
            if identity not in indices:
                raise ValueError(f"execution conversation {identity!r} is absent from saved inputs")
            unit = row["unit_id"] if str(row["sample_id"]) == identity else units.get(str(indices[identity]))
            if unit is None:
                raise ValueError("original evaluation unit is missing from the saved execution batch")
            rows.append({**row, "sample_id": identity, "unit_id": unit})
        aligned.append({**batch, "records": rows, "alignment": ALIGNMENT})
    return aligned


def recover(out: Path, model: dict, report: dict, archive: SelectionArchive) -> dict:
    """Recover each plugin benchmark from the exact active execution batches and original gates."""
    recorded = batches(out)
    updates, evidence = {}, []
    pending_exports = []
    for entry in report.get("accuracy", []):
        item = next((p for p in model.get("absolute", []) if p["suite"] == entry["suite"] and p.get("plugin")), None)
        if item is None:
            continue
        sides, plans = {}, None
        for side in ("candidate", "reference"):
            chosen = [b for b in recorded if b["workload"] == item["suite"] and b["identity"].get("side") == side]
            runs = {"records": {}, "exit": {}, "timings": {}}
            for batch in chosen:
                roots = {row["output_ref"]["aiperf_run"] for row in batch["records"]}
                if len(roots) != 1:
                    raise ValueError(f"{out}: accuracy batch has {len(roots)} run directories")
                run = AiperfRun(Path(next(iter(roots))), batch["aiperf_exit"], [])
                problems = archive.selection(run)
                if plans is not None and plans != problems:
                    raise ValueError(f"{out}: native and TRTMC original question/gold selections differ")
                plans = problems
                grades = asyncio.run(regrade(run, problems))
                name = run.directory.name.removeprefix(item["suite"] + "-")
                if name in runs["records"]:
                    raise ValueError(f"{out}: duplicate benchmark repetition {name}")
                parsed = absolute.plugin_side(run, problems, capacity=side == "candidate", grades=grades)
                for key, value in parsed.items():
                    runs.setdefault(key, {})[name] = value
                pending_exports.append((run.directory / "accuracy_export.aligned.jsonl", grades))
                evidence.append({"run": str(run.directory), "records": len(grades), "alignment": ALIGNMENT})
            sides[side] = runs
        if plans is None or not all(sides[side]["records"] for side in sides):
            # A missing run is execution failure, not a grading offset to repair.
            continue
        judged = absolute.judge_in_capacity({**item, "gate": entry["gate"]}, plans, sides["candidate"], sides["reference"])
        judged.pop("workload_perf", None)  # Unified reports derive Perf from the recorded execution stream.
        kept = {key: value for key, value in entry.items()
                if key in ("native", "candidate_replicas", "candidate_mps", "sides_concurrent", "informational")}
        updates[entry["suite"]] = {**judged, **kept, "alignment": ALIGNMENT}
    aligned = aligned_batches(recorded)
    from . import judge

    same_work = lambda mine, theirs: judge.work_check(  # noqa: E731
        {"work": [mine] if mine is not None else []}, {"work": [theirs] if theirs is not None else []}) is None
    session = execution.Session(out, {}, lambda: None, lambda value: value, same_work, batches=aligned)
    performance = [item for item in report.get("performance", []) if item.get("kind") != "natural_dataset"]
    performance.extend(session.natural_performance())
    # Validate the whole report before publishing any corrected evidence.
    for path, grades in pending_exports:
        partial = path.with_suffix(".tmp")
        partial.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in grades))
        partial.replace(path)
    result = {**report, "accuracy": [updates.get(entry["suite"], entry) for entry in report.get("accuracy", [])],
              "performance": performance}
    result["accuracy_alignment"] = {"version": ALIGNMENT, "suites": sorted(updates), "runs": evidence,
                                     "original_responses_reused": True}
    aligned_path = out / "execution.aligned.jsonl"
    aligned_path.write_text("".join(json.dumps(batch, ensure_ascii=False) + "\n" for batch in aligned))
    result["execution"] = {**report.get("execution", {}), "records": str(aligned_path), "alignment": ALIGNMENT}
    return result
