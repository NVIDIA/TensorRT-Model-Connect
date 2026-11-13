# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import csv
import hashlib
import io
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from trtmc_aiperf_qual.config import ConfigError
from trtmc_aiperf_qual.text_profile import arguments, cases
from trtmc_aiperf_qual.text_trace import select, timestamp_ms, verify_records


def test_azure_timestamp_differences_preserve_submillisecond_precision():
    first = timestamp_ms("2023-11-16 18:15:51.3910170")
    second = timestamp_ms("2023-11-16 18:15:52.5732450")
    assert second - first == Decimal("1182.2280000")


def test_selection_keeps_original_targets_and_records_context_rejections():
    row = {"TIMESTAMP": "2023-11-16 18:15:51.3910170", "ContextTokens": "91", "GeneratedTokens": "16"}
    selected, ledger = select([{**row, "ContextTokens": "300"}, {**row, "ContextTokens": "0"}, row], 1, 256, 16)
    assert [entry["status"] for entry in ledger] == ["rejected", "rejected", "selected"]
    assert selected[0]["source_index"] == 2 and selected[0]["original"] == row
    assert selected[0]["input_length"] == 91 and selected[0]["output_length"] == 16


def test_incomplete_csv_rows_are_rejected_in_the_ledger_without_aborting_selection():
    # DictReader represents missing trailing cells as None, not an empty string.
    rows = list(csv.DictReader(io.StringIO(
        "ContextTokens,GeneratedTokens,TIMESTAMP\n"
        "5,2\n"
        "5\n"
        "5,2,2023-11-16 18:15:51.3910170\n")))
    selected, ledger = select(rows, 1, 256, 16)
    assert [row["status"] for row in ledger] == ["rejected", "rejected", "selected"]
    assert selected[0]["source_index"] == 2 and selected[0]["output_length"] == 2
    assert ledger[0]["original"]["TIMESTAMP"] is None
    assert ledger[0]["reason"] == "trace timestamp is not a string"
    assert ledger[1]["original"]["GeneratedTokens"] is None and ledger[1]["reason"]


def test_trace_arguments_use_the_published_fixed_schedule_loader(tmp_path):
    app = pytest.importorskip("aiperf.cli").app
    config = {"corpus": {"format": "azure_trace", "count": 3},
              "tokenizer": {"name": "pinned", "revision": "a" * 40},
              "workload": {"endpoints": ["completions"], "requests": 3, "warmup_requests": 0}}
    case, = cases(config)
    (tmp_path / "inputs.jsonl").write_text('{"timestamp":0,"input_length":5,"output_length":2}\n')
    args = arguments("http://localhost", config, case, input_file=tmp_path / "inputs.jsonl")
    _, bound, _ = app.parse_args(["profile", "--model", "test", *args], print_error=False, exit_on_error=False)
    assert bound.arguments["cli_config"].fixed_schedule is True
    assert "--concurrency" not in args and "--output-tokens-mean" not in args
    assert args[args.index("--custom-dataset-type") + 1] == "mooncake_trace"
    config["workload"]["concurrency"] = [1]
    with pytest.raises(ConfigError, match="cannot be replaced"):
        cases(config)


@pytest.fixture
def capture(tmp_path):
    selected, raw, joined = [], [], []
    for index in range(3):
        selected.append({"session_id": f"azure-{index}", "source_index": index,
                         "timestamp": index * 500, "input_length": 5, "output_length": 2})
        status = 429 if index == 1 else 200
        credit = 1_000_000_000 + index * 500_000_000
        raw.append({"metadata": {"x_request_id": str(index), "conversation_id": f"azure-{index}",
                                  "benchmark_phase": "profiling", "credit_issued_ns": credit,
                                  "request_start_ns": credit + 1_000_000, "request_end_ns": credit + 11_000_000},
                    "payload": {"prompt": "hello", "max_tokens": 2}, "status": status,
                    "error": {"code": 429} if status == 429 else None,
                    "responses": [{"text": json.dumps({"choices": [{"text": "OK"}]})}]})
        server = {"status": status, "prompt_sha256": hashlib.sha256(b"hello").hexdigest()}
        if status == 200:
            server.update(completion_tokens=2, generation_config={"max_new_tokens": 2},
                          completion_sha256=hashlib.sha256(b"OK").hexdigest(),
                          timing_scope="public_task_call_wall", model_call_ms=1)
        joined.append({"request_id": str(index), "server": server})
    corpus = {"selected": selected, "context_limit": 256, "arrival_tolerance_ms": 5}
    return SimpleNamespace(raw_records=lambda phase: raw, directory=tmp_path), joined, corpus, raw


def test_all_offered_requests_include_visible_overload_and_schedule_measurements(capture):
    run, joined, corpus, _ = capture
    report = verify_records(run, joined, corpus, len)
    assert report["verified"] and report["offered"] == 3
    assert (report["completed"], report["rejected"], report["failed"]) == (2, 1, 0)
    assert report["requests"][1]["client_error"] == {"code": 429}
    assert report["credit_schedule_error_ms"]["p95"] == 0


def test_transport_failure_without_native_dispatch_remains_in_the_failed_census(capture):
    run, joined, corpus, raw = capture
    raw[0].update(status=None, error={"type": "connection_error"}, responses=[])
    joined[0]["server"] = None
    with pytest.raises(ConfigError, match="unexpected failures"):
        verify_records(run, joined, corpus, len)
    report = json.loads((run.directory / "trace-accounting.json").read_text())
    assert not report["verified"]
    assert (report["offered"], report["completed"], report["rejected"], report["failed"]) == (3, 1, 1, 1)
    assert report["requests"][0]["client_error"] == {"type": "connection_error"}
    assert report["requests"][0]["server_record_present"] is False


def test_transport_failure_after_native_completion_remains_in_the_failed_census(capture):
    run, joined, corpus, raw = capture
    raw[0].update(status=None, error={"type": "connection_error"}, responses=[])
    with pytest.raises(ConfigError, match="unexpected failures"):
        verify_records(run, joined, corpus, len)
    report = json.loads((run.directory / "trace-accounting.json").read_text())
    assert not report["verified"]
    assert (report["offered"], report["completed"], report["rejected"], report["failed"]) == (3, 1, 1, 1)
    assert report["requests"][0]["client_error"] == {"type": "connection_error"}
    assert report["requests"][0]["server_record_present"] is True


@pytest.mark.parametrize("status,error,changed_prompt", [
    (503, {"type": "connection_error"}, False),
    (None, None, False),
    (None, {"type": "connection_error"}, True),
])
def test_transport_exception_preserves_other_status_and_prompt_identity_checks(capture, status, error, changed_prompt):
    run, joined, corpus, raw = capture
    raw[0].update(status=status, error=error, responses=[])
    if changed_prompt:
        joined[0]["server"]["prompt_sha256"] = hashlib.sha256(b"changed").hexdigest()
    with pytest.raises(ConfigError, match="wire/native input or status identity"):
        verify_records(run, joined, corpus, len)


@pytest.mark.parametrize("change", ["missing", "cap", "failed", "schedule"])
def test_missing_changed_failed_or_mistimed_requests_cannot_qualify(capture, change):
    run, joined, corpus, raw = capture
    if change == "missing":
        raw.pop()
    elif change == "cap":
        raw[0]["payload"]["max_tokens"] = 99
    elif change == "failed":
        raw[0]["status"] = joined[0]["server"]["status"] = 503
    else:
        for key in ("credit_issued_ns", "request_start_ns", "request_end_ns"):
            raw[-1]["metadata"][key] += 500_000_000
    with pytest.raises(ConfigError):
        verify_records(run, joined, corpus, len)
    if change in {"failed", "schedule"}:
        assert json.loads((run.directory / "trace-accounting.json").read_text())["verified"] is False
