# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One evidence stream for quality, computation, and optional service workloads.

The transport records each profiling response once. Scorers keep reading that
response or its artifacts; this module derives timing evidence from the same
records. It contains no model topology, task scorer, or acceptance threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from . import aiperf_metrics
from .noninferiority import t_quantile

SCHEMA = "trtmc.qualification/v2"
TIMING_CONTRACT = "task-call-wall/v1"
_SESSION: ContextVar["Session | None"] = ContextVar("qualification_session", default=None)
_SERVICE: ContextVar[dict[str, Any]] = ContextVar("qualification_service", default={})
_WORKLOAD: ContextVar[dict[str, Any]] = ContextVar("qualification_workload", default={})


@contextmanager
def service(identity: Mapping[str, Any]) -> Iterator[None]:
    token = _SERVICE.set(dict(identity))
    try:
        yield
    finally:
        _SERVICE.reset(token)


@contextmanager
def workload(name: str, role: str, *, units: Sequence[Any] | None = None) -> Iterator[None]:
    """Roles describe consumers of one workload, rather than separate executors."""
    if role not in ("quality", "performance", "both", "service"):
        raise ValueError(f"unknown workload role {role!r}")
    token = _WORKLOAD.set({"name": name, "role": role, "units": list(units) if units is not None else None})
    try:
        yield
    finally:
        _WORKLOAD.reset(token)


def response_body(record: Mapping[str, Any]) -> dict[str, Any]:
    """The final JSON response carrying server timing, including buffered SSE."""
    found: dict[str, Any] = {}
    for response in record.get("responses") or []:
        try:
            body = json.loads(response.get("text") or "")
        except (TypeError, ValueError):
            continue
        if isinstance(body, dict):
            found = body
            if body.get("trtmc_timing"):
                return body
    return found


def observation(body: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(body.get("trtmc_observation"), Mapping):
        return dict(body["trtmc_observation"])
    result = dict(body.get("usage") or {})
    if result.get("completion_tokens") is not None:
        result["output_tokens"] = result["completion_tokens"]
    choices = body.get("choices") or []
    if choices:
        text = choices[0].get("text")
        if text is None:
            text = (choices[0].get("message") or choices[0].get("delta") or {}).get("content")
        if isinstance(text, str):
            result["text"] = text
    return result


def prepare(arguments: Sequence[str]) -> tuple[list[str], dict[str, Any] | None]:
    """Warm and observe a natural workload before sending its profiling requests."""
    args = list(arguments)
    active, descriptor = _SESSION.get(), _WORKLOAD.get()
    if active is None or not descriptor:
        return args, None
    natural = descriptor["role"] in ("quality", "both")
    warmup = 0 if active.smoke else int(active.measurement.get("warmup", 0))
    if natural and warmup > 0 and "--warmup-request-count" not in args:
        args += ["--warmup-request-count", str(warmup)]
    identity = dict(_SERVICE.get())
    concurrency = int(args[args.index("--concurrency") + 1]) if "--concurrency" in args else 1
    identity["concurrency"] = concurrency
    busy = active.gpu_probe()
    if not active.smoke and (busy is None or busy >= 20):
        from .services import ServiceError

        raise ServiceError(f"GPU idleness was not established before {descriptor['name']} "
                           f"(utilization {busy}); no benchmark requests were sent")
    metadata = {**descriptor, "identity": identity, "gpu_busy_percent": busy,
                "expected_requests": int(args[args.index("--request-count") + 1]) if "--request-count" in args else None,
                "warmup": int(args[args.index("--warmup-request-count") + 1])
                if "--warmup-request-count" in args else 0}
    return args, metadata


def record(run: Any, metadata: Mapping[str, Any] | None) -> None:
    active = _SESSION.get()
    if active is not None and metadata is not None:
        active.record(run, metadata)


def checkpoint() -> int:
    active = _SESSION.get()
    return len(active.batches) if active is not None else 0


def supersede(start: int) -> None:
    """Keep failed-attempt evidence, but never mix it with a replacement attempt."""
    active = _SESSION.get()
    if active is not None:
        batches = active.batches[start:]
        for batch in batches:
            batch["superseded"] = True
        if batches:
            with (active.out / "execution.jsonl").open("a") as handle:
                handle.write(json.dumps({"event": "supersede_failed_attempt", "batch_ids":
                                         [batch["batch_id"] for batch in batches]}) + "\n")


@dataclass
class Session:
    out: Path
    measurement: Mapping[str, Any]
    gpu_probe: Callable[[], float | None]
    work_evidence: Callable[[Mapping[str, Any]], Any]
    same_work: Callable[[Any, Any], bool]
    smoke: bool = False
    request_problems: Callable[[Mapping[str, Any]], Sequence[str]] = lambda payload: ()
    batches: list[dict[str, Any]] = field(default_factory=list)

    def record(self, run: Any, metadata: Mapping[str, Any]) -> None:
        from .absolute import capacity_rejection, unanswered

        identity = dict(metadata["identity"])
        units = metadata.get("units")
        rows = []
        indices = run.conversation_indices() if hasattr(run, "conversation_indices") else {}
        for ordinal, raw in enumerate(run.raw_records()):
            body = response_body(raw)
            payload = raw.get("payload") or {}
            request_key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            record_metadata = raw.get("metadata") or {}
            sample = record_metadata.get("conversation_id") or record_metadata.get("session_num", ordinal)
            try:
                position = indices[sample] if sample in indices else int(sample)
            except (TypeError, ValueError):
                position = ordinal
            unit = units[position] if units is not None and 0 <= position < len(units) else sample
            ms = (body.get("trtmc_timing") or {}).get("model_call_ms")
            try:
                ms = float(ms) if ms is not None else None
            except (TypeError, ValueError):
                ms = None
            valid_time = ms is not None and math.isfinite(ms) and ms > 0
            valid = not unanswered(raw) and not (
                raw.get("metadata") or {}).get("was_cancelled")
            try:
                work = self.work_evidence(observation(body)) if valid else None
            except (KeyError, TypeError, ValueError):
                work = None
            rows.append({"sample_id": sample, "unit_id": str(unit), "request_sha": request_key,
                         "model_call_ms": ms if valid_time else None, "valid": bool(valid and valid_time),
                         "output_valid": bool(valid), "work": work,
                         "capacity_rejection": capacity_rejection(raw) if identity.get("side") == "candidate" else None,
                         "request_problems": list(self.request_problems(payload)),
                         "output_ref": {"aiperf_run": str(run.directory), "record_index": ordinal,
                                        "request_id": body.get("request_id") or body.get("id")}})
        batch = {"schema_version": SCHEMA, "timing_contract": TIMING_CONTRACT, "batch_id": len(self.batches),
                 "workload": metadata["name"], "role": metadata["role"],
                 "identity": identity, "gpu_busy_percent": metadata["gpu_busy_percent"],
                 "warmup": metadata["warmup"], "expected_requests": metadata.get("expected_requests"),
                 "aiperf_exit": run.exit_code, "aiperf_metrics": aiperf_metrics.capture(run), "records": rows}
        self.batches.append(batch)
        self.out.mkdir(parents=True, exist_ok=True)
        with (self.out / "execution.jsonl").open("a") as handle:
            handle.write(json.dumps(batch, default=str) + "\n")

    def natural_performance(self, accuracy: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for batch in self.batches:
            if batch["role"] in ("quality", "both") and not batch.get("superseded"):
                grouped[batch["workload"]][batch["identity"].get("side", "unknown")].append(batch)
        excluded = {item["suite"]: int(item["out_of_capacity"]) for item in accuracy
                    if item.get("out_of_capacity") and item.get("status") != "error"}
        return [paired_dataset(name, sides.get("candidate", []), sides.get("reference", []), self.same_work,
                               excluded.get(name, 0))
                for name, sides in grouped.items()]


@contextmanager
def session(value: Session) -> Iterator[Session]:
    token = _SESSION.set(value)
    try:
        yield value
    finally:
        _SESSION.reset(token)


def paired_dataset(name: str, candidate: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]],
                   same_work: Callable[[Any, Any], bool], capacity_exclusions: int = 0) -> dict[str, Any]:
    """Describe the complete natural workload; never promote a matched subset to a gate.

    The interval describes variation across paired evaluation units, not repeated
    timing stability. Repeated seeds of one unit are averaged before the interval.
    Formal fixed-workload gates retain their existing repeated-run statistic.
    """
    sides = (candidate, reference)
    all_rows = [[row for batch in batches for row in batch["records"]] for batches in sides]
    excluded = {str(row["sample_id"]) for row in all_rows[0]
                if row.get("capacity_rejection")} if capacity_exclusions else set()
    if len(excluded) != capacity_exclusions:
        raise ValueError(f"{name}: recorded capacity rejections do not match the accuracy exclusions")
    rows = [[row for row in values if str(row["sample_id"]) not in excluded]
            for values in all_rows]
    indexed = [{(str(row["sample_id"]), row["request_sha"]): row for row in values} for values in rows]
    reasons = []
    complete = all(rows) and all(len(index) == len(values) for index, values in zip(indexed, rows))
    complete = complete and indexed[0].keys() == indexed[1].keys() and all(
        row["valid"] for values in rows for row in values)
    complete = complete and all(batch.get("aiperf_exit", 0) == 0 for batches in sides for batch in batches)
    if not complete:
        reasons.append("natural workload has missing, failed, duplicate, or unpaired responses")
    exported = all(batch.get("expected_requests") is None or len(batch["records"]) == batch["expected_requests"]
                   for batches in sides for batch in batches)
    if any(batch.get("expected_requests") is not None and len(batch["records"]) != batch["expected_requests"]
           for batches in sides for batch in batches):
        reasons.append("natural workload did not export every configured request")
    if any(row.get("request_problems") for values in rows for row in values):
        reasons.append("natural workload leaves computation parameters to backend defaults")
    identities = [[batch["identity"] for batch in batches] for batches in sides]
    precisions = [{item.get("precision") for item in values} for values in identities]
    if any(None in values or len(values) != 1 for values in precisions) or precisions[0] != precisions[1]:
        reasons.append("effective native and candidate precisions differ or are not recorded")
    allowed_scopes = {"task-call-wall", "public_task_call_wall"}
    if any(item.get("timing_scope") not in allowed_scopes for values in identities for item in values):
        reasons.append("server task-call timing boundary is not declared")
    if any(item.get("concurrency") != 1 or item.get("replicas", 1) != 1 or item.get("mps")
           for values in identities for item in values):
        reasons.append("natural workload was not executed by isolated single replicas")
    for batches in sides:
        if any(batch.get("gpu_busy_percent") is None or batch["gpu_busy_percent"] >= 20 for batch in batches):
            reasons.append("GPU idleness was not established before the natural workload")
        if any(not batch.get("warmup") for batch in batches):
            reasons.append("natural workload has no excluded warmup")
    pairs = [(indexed[0][key], indexed[1][key]) for key in indexed[0].keys() & indexed[1].keys()
             if indexed[0][key]["valid"] and indexed[1][key]["valid"]]
    def known_work(row: Mapping[str, Any]) -> bool:
        if row["work"] is None:
            return False
        signature = dict(row["work"])
        return (signature["output_tokens"] is not None if "output_tokens" in signature else
                all(value is not None for value in signature.values()))

    matches = [same_work(mine["work"], theirs["work"]) for mine, theirs in pairs]
    matched = sum(matches)
    unknown = sum(not equal and any(not known_work(row) for row in pair) for pair, equal in zip(pairs, matches))
    if pairs and matched != len(pairs):
        reasons.append(f"actual work differs or is unknown on {len(pairs) - matched} paired responses")
    result: dict[str, Any] = {"request": name, "reference_mode": "eager", "kind": "natural_dataset", "gate": False,
                              "timing_contract": TIMING_CONTRACT, "pairs": len(pairs), "matched_pairs": matched,
                              "different_work_pairs": len(pairs) - matched - unknown, "unknown_work_pairs": unknown,
                              "complete": bool(complete and exported),
                              "comparable": bool(pairs and not reasons), "light": "informational",
                              "out_of_capacity": len(excluded),
                              "reasons": list(dict.fromkeys(reasons)), "notes": [
                                  "Shared quality outputs; not an additional performance gate.",
                                  "Interval across evaluation units, not repeated-run timing stability."],
                              "candidate": {}, "reference": {}}
    # Report each side's entire valid workload, including unpaired responses.
    # Pair filtering is only for comparability, never for the displayed timings.
    for side, values, original, precision in zip(("candidate", "reference"), rows, all_rows, precisions):
        times = [row["model_call_ms"] for row in values if row["valid"]]
        result[side] = {"p50_ms": statistics.median(times) if times else None,
                        "total_ms": sum(times), "requests": len(values), "valid_requests": len(times),
                        "attempted_requests": len(original),
                        "precision": next(iter(precision)) if len(precision) == 1 else None}
        for key, unit, scale in (("output_tokens", "output tokens", 1), ("audio_10ms", "audio seconds", 0.01)):
            work = [dict(row["work"] or ()).get(key) for row in values if row["valid"]]
            work = [value * scale for value in work if isinstance(value, (int, float)) and value >= 0]
            if work:
                result[side][key] = {"p50": statistics.median(work), "total": sum(work), "requests": len(work)}
                result["notes"].append(f"{'TRTMC' if side == 'candidate' else 'Native'} {unit}: "
                                       f"p50 {statistics.median(work):g}, total {sum(work):g}, "
                                       f"recorded on {len(work)}/{len(times)} timed requests.")
    timed = all(result[side]["p50_ms"] is not None for side in ("candidate", "reference"))
    result["measurement_status"] = "unavailable" if not timed else "measured" if result["complete"] else "partial"
    if not pairs:
        return result
    mine = [row["model_call_ms"] for row, _ in pairs]
    theirs = [row["model_call_ms"] for _, row in pairs]
    result["natural_task_speedup"] = sum(theirs) / sum(mine)
    if result["comparable"]:
        # Multiple seeds belong to one evaluation unit; they are not independent samples.
        units: dict[str, list[float]] = defaultdict(list)
        for c, n in pairs:
            units[c["unit_id"]].append(math.log(n["model_call_ms"] / c["model_call_ms"]))
        values = [statistics.fmean(items) for items in units.values()]
        center = statistics.fmean(values)
        result.update(speedup=math.exp(center), total_time_speedup=sum(theirs) / sum(mine), units=len(values))
        if len(values) > 1:
            half = t_quantile(0.95, len(values) - 1) * statistics.stdev(values) / len(values) ** 0.5
            result["speedup_interval90"] = [math.exp(center - half), math.exp(center + half)]
    return result
