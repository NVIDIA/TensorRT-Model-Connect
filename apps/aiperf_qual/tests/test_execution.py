# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from trtmc_aiperf_qual import aiperf_runner, compat, execution, judge, runner
from trtmc_aiperf_qual.config import Environment


def same_work(mine, theirs):
    return judge.work_check({"work": [mine] if mine is not None else []},
                            {"work": [theirs] if theirs is not None else []}) is None


def raw(index, ms, tokens=32, **extra):
    return {"metadata": {"session_num": index, "benchmark_phase": "profiling"}, "status": 200,
            "payload": {"request": {"prompt": str(index)}}, "responses": [{"text": json.dumps({
                "trtmc_timing": {"model_call_ms": ms}, "trtmc_observation": {"output_tokens": tokens}})}], **extra}


def identity(side, **extra):
    return {"side": side, "precision": "fp16", "timing_scope": "task-call-wall", "concurrency": 1, **extra}


def collect(session, side, rows, units=None, **extra):
    run = SimpleNamespace(directory=session.out / side, exit_code=0, raw_records=lambda: rows)
    session.record(run, {"name": "evaluation", "role": "both", "identity": identity(side, **extra),
                         "units": units, "gpu_busy_percent": 0, "warmup": 1})


def new_session(tmp_path):
    return execution.Session(tmp_path, {"warmup": 1}, lambda: 0,
                             lambda obs: judge.work_signature("generate", obs), same_work)


def test_same_inference_records_feed_accuracy_and_timing_without_replaying(tmp_path, monkeypatch):
    calls = []
    out = tmp_path / "aiperf"

    def send(command, log, env, timeout):
        calls.append(command)
        (out / aiperf_runner.RAW_EXPORT).write_text(json.dumps(raw(0, 10)) + "\n" +
            json.dumps({**raw(0, 999), "metadata": {"session_num": 0, "benchmark_phase": "warmup"}}) + "\n")
        (out / "accuracy_export.jsonl").write_text(json.dumps({"session_num": 0, "passed": True,
                                                               "benchmark_phase": "profiling"}) + "\n")
        return 0

    monkeypatch.setattr(aiperf_runner, "_run", send)
    monkeypatch.setattr(aiperf_runner, "_wait_ready", lambda out: None)
    evidence = new_session(tmp_path)
    with execution.session(evidence), execution.service(identity("candidate")), execution.workload("evaluation", "both"):
        result = aiperf_runner.run_aiperf(Environment({"aiperf": "aiperf", "hf_datasets_cache": str(tmp_path)}),
                                          out, ["--concurrency", "1", "--request-count", "1"])
    assert len(calls) == 1 and result.accuracy_records()[0]["passed"]
    assert "--warmup-request-count" in calls[0]
    records = evidence.batches[0]["records"]
    assert len(records) == 1 and records[0]["model_call_ms"] == 10
    assert records[0]["output_ref"]["aiperf_run"] == str(out)
    assert len((tmp_path / "execution.jsonl").read_text().splitlines()) == 1


def test_pair_work_per_sample_instead_of_demanding_every_problem_have_one_length(tmp_path):
    evidence = new_session(tmp_path)
    collect(evidence, "candidate", [raw(0, 10, 32), raw(1, 20, 128)])
    collect(evidence, "reference", [raw(0, 20, 32), raw(1, 40, 128)])
    result, = evidence.natural_performance()
    assert result["comparable"] and result["speedup"] == pytest.approx(2)
    assert result["total_time_speedup"] == pytest.approx(2)
    assert result["speedup_interval90"] == pytest.approx([2, 2])
    assert not result["gate"] and result["matched_pairs"] == result["pairs"] == 2


def test_less_generation_is_not_claimed_as_equal_work_acceleration(tmp_path):
    evidence = new_session(tmp_path)
    collect(evidence, "candidate", [raw(0, 10, 80), raw(1, 10, 200)])
    collect(evidence, "reference", [raw(0, 25, 200), raw(1, 25, 200)])
    result, = evidence.natural_performance()
    assert not result["comparable"] and "speedup" not in result
    assert result["natural_task_speedup"] == 2.5 and result["pairs"] == 2 and result["matched_pairs"] == 1
    assert "actual work differs" in " ".join(result["reasons"])


@pytest.mark.parametrize("extra", [{"precision": "bf16"}, {"concurrency": 4}, {"mps": True},
                                   {"timing_scope": None}])
def test_precision_contention_and_unknown_boundary_invalidate_acceleration(tmp_path, extra):
    evidence = new_session(tmp_path)
    collect(evidence, "candidate", [raw(0, 10)])
    collect(evidence, "reference", [raw(0, 20)], **extra)
    result, = evidence.natural_performance()
    assert not result["comparable"] and "speedup" not in result


def test_failed_or_missing_responses_cannot_disappear_into_a_fast_subset(tmp_path):
    evidence = new_session(tmp_path)
    collect(evidence, "candidate", [raw(0, 10), raw(1, 10, status=500)])
    collect(evidence, "reference", [raw(0, 20), raw(1, 20)])
    result, = evidence.natural_performance()
    assert not result["comparable"] and result["pairs"] == 1 and "speedup" not in result
    assert "unpaired" in " ".join(result["reasons"])


def test_matching_export_gaps_on_both_sides_do_not_prove_complete_work(tmp_path):
    evidence = new_session(tmp_path)
    collect(evidence, "candidate", [raw(0, 10)])
    collect(evidence, "reference", [raw(0, 20)])
    for batch in evidence.batches:
        batch["expected_requests"] = 2
    result, = evidence.natural_performance()
    assert not result["comparable"] and "speedup" not in result
    assert "every configured request" in " ".join(result["reasons"])


def test_a_failed_attempt_is_preserved_but_not_mixed_with_its_retry(tmp_path):
    evidence = new_session(tmp_path)
    with execution.session(evidence):
        start = execution.checkpoint()
        collect(evidence, "candidate", [raw(0, 10, status=500)])
        execution.supersede(start)
        collect(evidence, "candidate", [raw(0, 10)])
        collect(evidence, "reference", [raw(0, 20)])
    result, = evidence.natural_performance()
    assert result["comparable"] and result["pairs"] == 1
    assert "supersede_failed_attempt" in (tmp_path / "execution.jsonl").read_text()


def test_implicit_backend_defaults_prevent_an_equal_work_claim(tmp_path):
    evidence = new_session(tmp_path)
    evidence.request_problems = lambda payload: ["num_steps"]
    collect(evidence, "candidate", [raw(0, 10)])
    collect(evidence, "reference", [raw(0, 20)])
    result, = evidence.natural_performance()
    assert not result["comparable"] and "speedup" not in result


def test_invalid_timing_preserves_output_validity_for_the_quality_consumer(tmp_path):
    evidence = new_session(tmp_path)
    collect(evidence, "candidate", [raw(0, "bad")])
    row, = evidence.batches[0]["records"]
    assert row["output_valid"] and not row["valid"] and row["model_call_ms"] is None


def test_duplicate_requests_keep_their_distinct_sample_ids(tmp_path):
    evidence = new_session(tmp_path)
    rows = [raw(0, 10), {**raw(1, 20), "payload": raw(0, 10)["payload"]}]
    collect(evidence, "candidate", rows)
    collect(evidence, "reference", rows)
    result, = evidence.natural_performance()
    assert result["comparable"] and result["pairs"] == 2


def test_repeated_seeds_do_not_become_independent_dataset_units(tmp_path):
    evidence = new_session(tmp_path)
    collect(evidence, "candidate", [raw(0, 10), raw(1, 11)], units=["same-problem", "same-problem"])
    collect(evidence, "reference", [raw(0, 20), raw(1, 22)], units=["same-problem", "same-problem"])
    result, = evidence.natural_performance()
    assert result["units"] == 1 and "speedup_interval90" not in result


def test_buffered_openai_response_keeps_model_time_and_work_evidence():
    body = {"trtmc_timing": {"model_call_ms": 3}, "usage": {"completion_tokens": 80},
            "choices": [{"delta": {"content": "answer"}}]}
    assert execution.observation(body)["output_tokens"] == 80
    assert execution.observation(body)["text"] == "answer"


def test_informational_dataset_results_neither_satisfy_nor_change_a_formal_gate():
    base = {"accuracy_source": "none", "accuracy": [], "performance": [
        {"reference_mode": "eager", "light": "green", "request": "catalog"}]}
    expected = judge.verdict(base, expected_suites=[], expected_modes=1)
    mixed = {**base, "performance": [*base["performance"],
             {"reference_mode": "eager", "light": "white", "request": "evaluation", "gate": False}]}
    assert judge.verdict(mixed, expected_suites=[], expected_modes=1) == expected
    assert judge.verdict({**base, "performance": mixed["performance"][1:]},
                          expected_suites=[], expected_modes=1)["perf"] == "error"


def test_legacy_reports_and_configuration_keep_their_exact_gates():
    policy = {"margin_percent": 5, "guard_percent": 12, "max_ci_percent": 5}
    assert compat.configuration({"performance": {"l1": policy}})["performance"] == policy
    converted = compat.report({"performance_l1": [{"light": "green"}], "performance_l2": {"light": "red"}})
    assert converted["performance"] == [{"light": "green"}]
    assert converted["service_metrics"] == {"light": "red"} and "performance_l1" not in converted


def test_qualification_ignores_legacy_overlap_and_replica_settings(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "_qualify", lambda model, environment, out: seen.append(environment.values) or {
        "model": "test", "performance": [], "accuracy": []})
    monkeypatch.setattr(runner, "write_report", lambda out, result: None)
    runner.qualify({"operation": "generate", "performance": {}}, Environment({
        "native_replicas": 8, "candidate_replicas": 8, "acc_overlap": True, "acc_mps": True}), tmp_path)
    assert seen[0]["native_replicas"] == seen[0]["candidate_replicas"] == 1
    assert not seen[0]["acc_overlap"] and not seen[0]["acc_mps"]
