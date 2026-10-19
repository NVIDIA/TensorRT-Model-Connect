# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Project pinned Azure trace rows into AIPerf's existing Mooncake loader."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import urllib.request
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .config import ConfigError
from .text_profile import SHA, digest, response_text, statistics


def timestamp_ms(value: str) -> Decimal:
    clock, _, fraction = value.partition(".")
    time = datetime.strptime(clock, "%Y-%m-%d %H:%M:%S")
    delta = time - datetime(1970, 1, 1)
    return Decimal(delta.days * 86400000 + delta.seconds * 1000) + Decimal("0." + (fraction or "0")) * 1000


def select(rows, count: int, capacity: int, margin: int):
    if type(count) is not int or count < 1 or type(capacity) is not int or capacity < 1:
        raise ConfigError("trace count and context must be positive integers")
    selected, ledger = [], []
    for index, row in enumerate(rows):
        entry = {"source_index": index, "original": row}
        try:
            if not isinstance(row, dict):
                raise ValueError("trace row is not an object")
            inputs, outputs = int(row["ContextTokens"]), int(row["GeneratedTokens"])
            if not isinstance(row["TIMESTAMP"], str):
                raise ValueError("trace timestamp is not a string")
            timestamp_ms(row["TIMESTAMP"])
            if inputs < 1 or outputs < 1 or inputs + outputs + margin > capacity:
                raise ValueError("input/output target plus safety margin does not fit bundle")
        except (TypeError, ValueError, KeyError, InvalidOperation) as error:
            entry.update(status="rejected", reason=str(error))
        else:
            entry.update(status="selected", session_id=f"azure-{index}", input_length=inputs, output_length=outputs)
            selected.append(entry)
        ledger.append(entry)
        if len(selected) == count:
            break
    return selected, ledger


def prepare(config, capacity: int, out: Path):
    spec, source = config["corpus"], config["corpus"]["source"]
    revision = str(source.get("revision", ""))
    if not SHA.fullmatch(revision) or f"/{revision}/" not in source["url"]:
        raise ConfigError("trace source URL must include its immutable revision")
    scale, tolerance, margin = spec.get("time_scale", 1), spec.get("arrival_tolerance_ms", 100), spec.get("context_margin", 16)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (scale, tolerance)):
        raise ConfigError("trace time scale and arrival tolerance must be finite and positive")
    if type(margin) is not int or margin < 0 or margin >= capacity:
        raise ConfigError("trace context margin is invalid")
    with urllib.request.urlopen(source["url"], timeout=60) as response:
        data = response.read()
    source_hash = hashlib.sha256(data).hexdigest()
    if source_hash != source.get("sha256"):
        raise ConfigError("trace source checksum does not match its pinned configuration")
    (out / "source.csv").write_bytes(data)
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    selected, ledger = select(rows, spec["count"], capacity, margin)
    origin = min((timestamp_ms(row["original"]["TIMESTAMP"]) for row in selected), default=Decimal(0))
    for row in selected:
        offset = timestamp_ms(row["original"]["TIMESTAMP"]) - origin
        row.update(source_offset_ms=str(offset), timestamp=float(offset * Decimal(str(scale))))
    path = out / "inputs.jsonl"
    path.write_text("".join(json.dumps({key: row[key] for key in
        ("session_id", "timestamp", "input_length", "output_length")}) + "\n" for row in selected))
    report = {"format": "azure_trace", "client_dataset_type": "mooncake_trace", "source": source,
              "source_sha256": source_hash, "source_records": len(rows), "considered_records": len(ledger),
              "unexamined_records": len(rows) - len(ledger), "selected": selected, "ledger": ledger,
              "context_limit": capacity, "context_margin": margin, "time_scale": scale,
              "arrival_tolerance_ms": tolerance, "tokenizer": config["tokenizer"],
              "timestamp_policy": "timezone unspecified; exact decimal differences from earliest selected invocation, multiplied by time_scale, expressed in milliseconds",
              "prompt_policy": "source contains lengths, not prompt text; AIPerf generates synthetic prompts; actual token counts and native outputs are reported",
              "selection_policy": "first fitting positive input/output targets in source file order; no length truncation",
              "warmup_policy": "no AIPerf warmup; full owning validation and server startup precede the trace window; first request remains in measurement",
              "input_file": str(path), "input_file_sha256": digest(path)}
    (out / "corpus.json").write_text(json.dumps(report, indent=2) + "\n")
    if len(selected) != spec["count"]:
        raise ConfigError("not enough trace rows fit; see corpus.json; no length targets were changed")
    return report


def verify(run, joined, corpus):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(corpus["tokenizer"]["name"],
        revision=corpus["tokenizer"]["revision"], trust_remote_code=False)
    return verify_records(run, joined, corpus, lambda text: len(tokenizer.encode(text, add_special_tokens=False)))


def verify_records(run, joined, corpus, count_tokens):
    expected = {row["session_id"]: row for row in corpus["selected"]}
    native = {row["request_id"]: row["server"] for row in joined}
    raw = run.raw_records(phase=None)
    if len(raw) != len(expected) or any(not isinstance(row, dict) or not isinstance(row.get("metadata"), dict)
        or row["metadata"].get("benchmark_phase") != "profiling" for row in raw):
        raise ConfigError("trace export must account for every offered row, without hidden warmup or replay")
    seen, ids, records = set(), set(), []
    for row in raw:
        meta, payload = row["metadata"], row.get("payload")
        session, request_id = meta.get("conversation_id"), meta.get("x_request_id")
        if (not isinstance(session, str) or session not in expected or session in seen or
                not isinstance(request_id, str) or request_id in ids or not isinstance(payload, dict)):
            raise ConfigError("trace request identity is missing, unknown or duplicated")
        seen.add(session)
        ids.add(request_id)
        original = expected[session]
        prompt = payload.get("prompt")
        server = native.get(request_id)
        if not isinstance(prompt, str) or payload.get("max_tokens") != original["output_length"]:
            raise ConfigError("trace payload changed an output target or lost its generated prompt")
        tokens = count_tokens(prompt)
        if tokens + original["output_length"] > corpus["context_limit"]:
            raise ConfigError("actual generated trace prompt exceeds context; no truncation is allowed")
        transport_failure = row.get("status") is None and bool(row.get("error"))
        if server is not None and (server.get("prompt_sha256") != hashlib.sha256(prompt.encode()).hexdigest() or
                (server.get("status") != row.get("status") and not transport_failure)):
            raise ConfigError("trace wire/native input or status identity does not match")
        status = row.get("status")
        if status == 200 and server is not None:
            text = response_text(row)
            if (not isinstance(text, str) or server.get("timing_scope") != "public_task_call_wall" or
                    type(server.get("completion_tokens")) is not int or not 0 <= server["completion_tokens"] <= original["output_length"] or
                    server.get("completion_sha256") != hashlib.sha256(text.encode()).hexdigest() or
                    server.get("generation_config", {}).get("max_new_tokens") != original["output_length"]):
                raise ConfigError("trace native output, generation cap or Task timing is invalid")
            state = "completed"
        elif status == 429 and server is not None and "model_call_ms" not in server:
            state = "rejected"
        else:
            state = "failed"
        if any(type(meta.get(key)) is not int or meta[key] < 1 for key in ("credit_issued_ns", "request_start_ns", "request_end_ns")):
            raise ConfigError("trace export is missing schedule/arrival timestamps")
        records.append({"request_id": request_id, "session_id": session, "source_index": original["source_index"],
                        "state": state, "status": status, "scheduled_offset_ms": original["timestamp"],
                        "credit_issued_ns": meta["credit_issued_ns"], "request_start_ns": meta["request_start_ns"],
                        "request_end_ns": meta["request_end_ns"], "target_input_tokens": original["input_length"],
                        "actual_input_tokens_pinned_tokenizer": tokens, "target_output_tokens": original["output_length"],
                        "native_output_tokens": (server or {}).get("completion_tokens"),
                        "admission_ms": (server or {}).get("admission_ms"), "client_error": row.get("error"),
                        "server_record_present": server is not None})
    if seen != set(expected):
        raise ConfigError("trace replay omitted source rows")
    anchor = min(records, key=lambda row: row["scheduled_offset_ms"])["credit_issued_ns"]
    for row in records:
        row["credit_schedule_error_ms"] = (row["credit_issued_ns"] - anchor) / 1e6 - row["scheduled_offset_ms"]
        row["send_schedule_error_ms"] = (row["request_start_ns"] - anchor) / 1e6 - row["scheduled_offset_ms"]
    counts = {state: sum(row["state"] == state for row in records) for state in ("completed", "rejected", "failed")}
    elapsed = (max(row["request_end_ns"] for row in records) - min(row["request_start_ns"] for row in records)) / 1e9
    scheduled_span = max(row["scheduled_offset_ms"] for row in records) / 1000
    eligible = bool(counts["completed"]) and not counts["failed"] and all(
        abs(row["credit_schedule_error_ms"]) <= corpus["arrival_tolerance_ms"] for row in records)
    report = {"verified": eligible, "offered": len(records), **counts, "requests": records,
            "arrival_tolerance_ms": corpus["arrival_tolerance_ms"],
            "scheduled_window_seconds": scheduled_span, "observed_window_seconds": elapsed,
            "offered_requests_per_second_scheduled_window": len(records) / scheduled_span if scheduled_span else None,
            "completed_requests_per_second_observed_window": counts["completed"] / elapsed if elapsed > 0 else None,
            "credit_schedule_error_ms": statistics([abs(row["credit_schedule_error_ms"]) for row in records]),
            "send_schedule_error_ms": statistics([abs(row["send_schedule_error_ms"]) for row in records]),
            "native_input_token_source": "unavailable; pinned-tokenizer counts are reported separately",
            "aiperf_replay_metrics": {key: value for key, value in getattr(run, "summary", {}).items()
                                      if key.startswith("replay_")}}
    (run.directory / "trace-accounting.json").write_text(json.dumps(report, indent=2) + "\n")
    if not eligible:
        raise ConfigError("trace has unexpected failures, no native completion, or excessive credit-schedule lag; see trace-accounting.json")
    return report
