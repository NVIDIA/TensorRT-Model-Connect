# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import json

import pytest
from trtmc_aiperf_qual.aiperf_runner import AiperfRun, run_aiperf
from trtmc_aiperf_qual.config import ConfigError, Environment
from trtmc_aiperf_qual.text_profile import arguments, cases, check_aiperf, join_records, summarize


def test_installed_aiperf_package_needs_no_source_checkout(tmp_path, monkeypatch):
    import trtmc_aiperf_qual.text_profile as text_profile
    installed = {"version": "0.13.0", "module": str(tmp_path / "site-packages/aiperf/__init__.py"),
                 "distribution_root": str(tmp_path / "site-packages"), "installer": "pip",
                 "metadata_sha256": "a" * 64, "record_sha256": "b" * 64, "direct_url": None}
    calls = []
    def probe(command, **kwargs):
        calls.append(command)
        return json.dumps(installed)
    monkeypatch.setattr(text_profile.subprocess, "check_output", probe)
    environment = Environment({"aiperf_python": "/client-venv/bin/python"})
    assert check_aiperf(environment, {"version": "0.13.0"}) == installed
    assert len(calls) == 1 and calls[0][:2] == ["/client-venv/bin/python", "-c"]


@pytest.mark.parametrize("version", ["0.12.0", "0.14.0", "0.13.0.dev20261002"])
def test_installed_aiperf_rejects_untested_versions(version, monkeypatch):
    import trtmc_aiperf_qual.text_profile as text_profile
    monkeypatch.setattr(text_profile.subprocess, "check_output", lambda *args, **kwargs:
                        json.dumps({"version": version, "module": "/site-packages/aiperf/__init__.py"}))
    with pytest.raises(ConfigError, match="pip install aiperf==0.13.0"):
        check_aiperf(Environment({"aiperf_python": "python"}), {"version": "0.13.0"})
    with pytest.raises(ConfigError, match="tested AIPerf 0.13.0"):
        check_aiperf(Environment({"aiperf_python": "python"}), {"version": version})


def test_optional_aiperf_source_pin_retains_checkout_checks(tmp_path, monkeypatch):
    import trtmc_aiperf_qual.text_profile as text_profile
    source = tmp_path / "source"
    commit = "c" * 40
    environment = Environment({"aiperf_python": "python"})
    pin = {"source": str(source), "commit": commit}
    installed = {"version": "0.13.0", "module": str(source / "src/aiperf/__init__.py")}
    monkeypatch.setattr(text_profile, "git", lambda repo, *args:
                        commit if args == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(text_profile.subprocess, "check_output", lambda *args, **kwargs: json.dumps(installed))
    assert check_aiperf(environment, pin)["commit"] == commit
    installed["module"] = str(tmp_path / "another/aiperf/__init__.py")
    with pytest.raises(ConfigError, match="configured source checkout"):
        check_aiperf(environment, pin)
    monkeypatch.setattr(text_profile, "git", lambda repo, *args:
                        commit if args == ("rev-parse", "HEAD") else " M tracked.py")
    with pytest.raises(ConfigError, match="tracked modifications"):
        check_aiperf(environment, pin)
    with pytest.raises(ConfigError, match="both source and commit"):
        check_aiperf(environment, {"source": str(source)})
    with pytest.raises(ConfigError, match="exact commit"):
        check_aiperf(environment, {"source": str(source), "commit": "bad"})


def test_arguments_preserve_api_model_and_disable_unavailable_server_counts(tmp_path, monkeypatch):
    import trtmc_aiperf_qual.aiperf_runner as runner
    monkeypatch.setattr(runner, "_run", lambda command, *args: 0)
    monkeypatch.setattr(runner, "_wait_ready", lambda out: None)
    env = Environment({"aiperf": "aiperf", "hf_datasets_cache": str(tmp_path)})
    assert run_aiperf(env, tmp_path / "old", []).command[3] == "trtmc"
    run = run_aiperf(env, tmp_path / "new", [], model_name="Qwen/Qwen3-0.6B")
    assert run.command[3] == "Qwen/Qwen3-0.6B"
    config = {"workload": {}, "tokenizer": {"name": "Qwen/Qwen3-0.6B", "revision": "abc"}}
    args = arguments("http://127.0.0.1:8000", config, cases(config)[0])
    assert "--use-server-token-count" not in args and "--streaming" not in args
    assert args[args.index("--endpoint-type") + 1] == "completions"
    assert args[args.index("--warmup-concurrency") + 1] == "1"


def test_sweeps_are_explicit_and_validate_inputs():
    config = {"workload": {"endpoints": ["chat"], "input_tokens": [32, 64],
                           "concurrency": [1, 2], "streaming": [False, True]}}
    assert len(cases(config)) == 8
    for invalid in (0, True, -1, "2"):
        with pytest.raises(ConfigError):
            cases({"workload": {"concurrency": [invalid]}})


def test_join_retains_warmup_errors_and_missing_server_records(tmp_path):
    metadata = lambda key, phase: {"x_request_id": key, "benchmark_phase": phase,
                                  "request_start_ns": 1000000000, "request_end_ns": 2000000000}
    raw = [{"metadata": metadata("warm", "warmup"), "metrics": {}},
           {"metadata": metadata("ok", "profiling"), "status": 200, "metrics": {"request_latency": {"value": 30, "unit": "ms"}}},
           {"metadata": metadata("busy", "profiling"), "error": {"message": "429"}},
           {"metadata": metadata("missing", "profiling"), "error": {"message": "network"}}]
    (tmp_path / "profile_export_raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))
    run = AiperfRun(tmp_path, 0, [])
    server = [{"request_id": "warm", "status": 200}, {"request_id": "ok", "status": 200,
               "model_call_ms": 10, "completion_tokens": 3}, {"request_id": "busy", "status": 429}]
    rows = join_records(run, server)
    assert len(rows) == 4 and len(run.raw_records()) == 3
    report = summarize(rows)
    assert report["successful_requests"] == 1 and report["profiling_requests"] == 3
    assert report["status_counts"] == {"200": 1, "429": 1}
    assert report["client_errors"] == 2 and report["unmatched_client_requests"] == 1
    assert report["native_model_call_ms"]["p50"] == 10
    assert report["client_request_latency_ms"]["p50"] == 30
    assert report["client_request_lifecycle_ms"]["p50"] == 1000
    assert report["actual_completion_tokens"]["p50"] == 3
    assert report["server_input_token_source"] == "unavailable"
    with pytest.raises(ConfigError, match="duplicate"):
        join_records(run, server + [server[0]])


def test_matched_reference_ratio_requires_inputs_outputs_work_and_scope(tmp_path):
    from trtmc_aiperf_qual.text_profile import compare_runs
    candidate_dir, reference_dir = tmp_path / "candidate", tmp_path / "reference"
    candidate_dir.mkdir(); reference_dir.mkdir()
    def export(directory, key, text="answer", prompt="question"):
        record = {"metadata": {"session_num": 0, "x_request_id": key, "benchmark_phase": "profiling"},
                  "payload": {"model": key, "prompt": prompt, "max_tokens": 8}, "status": 200,
                  "responses": [{"text": json.dumps({"choices": [{"text": text}]})}]}
        (directory / "profile_export_raw.jsonl").write_text(json.dumps(record) + "\n")
    export(candidate_dir, "c"); export(reference_dir, "r")
    candidate, reference = AiperfRun(candidate_dir, 0, []), AiperfRun(reference_dir, 0, [])
    c = [{"request_id": "c", "status": 200, "completion_tokens": 3, "model_call_ms": 10, "timing_scope": "public_task_call_wall"}]
    r = [{"request_id": "r", "status": 200, "completion_tokens": 3, "model_call_ms": 20, "timing_scope": "public_task_call_wall"}]
    assert compare_runs(candidate, reference, c, r)["native_task_wall_ratio"] == 2
    export(reference_dir, "r", text="different")
    assert compare_runs(candidate, reference, c, r)["native_task_wall_ratio"] is None
    export(reference_dir, "r", prompt="different")
    assert not compare_runs(candidate, reference, c, r)["comparable"]
    export(reference_dir, "r")
    r[0]["completion_tokens"] = 8
    assert not compare_runs(candidate, reference, c, r)["comparable"]
    r[0]["completion_tokens"] = 3
    r[0]["timing_scope"] = "model-forward-only"
    assert not compare_runs(candidate, reference, c, r)["comparable"]


def test_client_metrics_join_uses_separate_aiperf_export(tmp_path):
    metadata = {"x_request_id": "client-uuid", "session_num": 0, "benchmark_phase": "profiling",
                "request_start_ns": 1_000_000_000, "request_end_ns": 2_000_000_000}
    raw = {"metadata": metadata, "status": 200, "responses": []}
    (tmp_path / "profile_export_raw.jsonl").write_text(json.dumps(raw) + "\n")
    metrics = {"metadata": metadata, "metrics": {"request_latency": {"value": 42, "unit": "ms"},
               "output_sequence_length": {"value": 3, "unit": "tokens"}}}
    (tmp_path / "profile_export.jsonl").write_text(json.dumps(metrics) + "\n")
    rows = join_records(AiperfRun(tmp_path, 0, []), [{"request_id": "client-uuid", "status": 200}])
    assert rows[0]["metrics_joined"]
    report = summarize(rows)
    assert report["client_request_latency_ms"]["p50"] == 42
    assert report["client_request_lifecycle_ms"]["p50"] == 1000
    assert report["client_output_tokens_per_second_estimate"] == 3
    assert report["missing_metric_exports"] == 0
    (tmp_path / "profile_export.jsonl").unlink()
    assert summarize(join_records(AiperfRun(tmp_path, 0, []), [{"request_id": "client-uuid", "status": 200}]))["missing_metric_exports"] == 1


def test_reference_refuses_unknown_output_counts_or_duplicate_sessions(tmp_path):
    from trtmc_aiperf_qual.text_profile import compare_runs
    raw = {"metadata": {"session_num": 0, "x_request_id": "a", "benchmark_phase": "profiling"},
           "payload": {"prompt": "same"}, "status": 200,
           "responses": [{"text": json.dumps({"choices": [{"text": "same"}]})}]}
    export = tmp_path / "profile_export_raw.jsonl"
    export.write_text(json.dumps(raw) + "\n")
    run = AiperfRun(tmp_path, 0, [])
    server = [{"request_id": "a", "status": 200, "completion_tokens": None,
               "model_call_ms": 1, "timing_scope": "public_task_call_wall"}]
    assert not compare_runs(run, run, server, server)["comparable"]
    server[0]["completion_tokens"] = 1
    server[0]["status"] = 499
    assert not compare_runs(run, run, server, server)["comparable"]
    server[0]["status"] = 200
    export.write_text((json.dumps(raw) + "\n") * 2)
    assert not compare_runs(run, run, server, server)["comparable"]
