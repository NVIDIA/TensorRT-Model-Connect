# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from types import SimpleNamespace

import pytest

from trtmc_aiperf_qual import benchmark_perf, execution, judge


def row(index, ms, tokens=1, text="C", **extra):
    return {"metadata": {"session_num": index, "conversation_id": f"session_{index:06d}",
                         "benchmark_phase": "profiling"}, "status": 200,
            "payload": {"prompt": str(index)}, "responses": [{"text": json.dumps({
                "trtmc_timing": {"model_call_ms": ms},
                "trtmc_observation": {"output_tokens": tokens, "text": text}})}], **extra}


def rejected(index):
    return row(index, None, status=422, error={"message": json.dumps({"error": {
        "code": "backend_rejected_request", "message": "prompt exceeds the prefill profile"}})})


def capture(out, candidate, native, accuracy=None):
    evidence = execution.Session(out, {}, lambda: 0,
        lambda obs: judge.work_signature("generate", obs),
        lambda c, n: judge.work_check({"work": [c] if c is not None else []},
                                      {"work": [n] if n is not None else []}) is None)
    for side, records in (("candidate", candidate), ("reference", native)):
        directory = out / side
        directory.mkdir(parents=True)
        (directory / "profile_export_raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        evidence.record(SimpleNamespace(directory=directory, exit_code=0, raw_records=lambda: records), {
            "name": "mmlu-0shot", "role": "both", "warmup": 1, "gpu_busy_percent": 0,
            "expected_requests": len(records), "identity": {
                "side": side, "operation": "generate", "precision": "fp16", "concurrency": 1,
                "timing_scope": "task-call-wall"}})
    quality = accuracy or [{"suite": "mmlu-0shot", "source": "absolute", "status": "pass"}]
    return {"model": "demo", "performance_source": "quality", "accuracy": quality,
            "performance": evidence.natural_performance(quality),
            "execution": {"records": str(out / "execution.jsonl")}}


def verdict(report):
    return judge.verdict(report, expected_suites=["mmlu-0shot"], expected_modes=0)


def test_different_answer_lengths_are_descriptive_not_a_second_performance_gate(tmp_path):
    report = capture(tmp_path, [row(0, 40, 4, "C. i")], [row(0, 20, 2)])
    perf, = report["performance"]
    assert verdict(report) == {"acc": "pass", "perf": "measured", "category": "measured", "lights": {}}
    assert not perf["comparable"] and perf["matched_pairs"] == 0
    assert perf["different_work_pairs"] == 1 and perf["unknown_work_pairs"] == 0
    assert perf["candidate"]["output_tokens"] == {"p50": 4, "total": 4, "requests": 1}
    assert perf["reference"]["output_tokens"] == {"p50": 2, "total": 2, "requests": 1}
    assert perf["candidate"]["p50_ms"] == 40 and perf["reference"]["p50_ms"] == 20
    assert not perf["gate"] and perf["measurement_status"] == "measured"


def test_capacity_exclusions_use_the_accuracy_scope_on_both_sides(tmp_path):
    report = capture(tmp_path, [row(0, 40), rejected(1)], [row(0, 20), row(1, 200)], [
        {"suite": "mmlu-0shot", "source": "absolute", "status": "pass", "out_of_capacity": 1}])
    perf, = report["performance"]
    assert verdict(report)["perf"] == "measured"
    assert perf["complete"] and perf["out_of_capacity"] == 1
    assert perf["reference"]["p50_ms"] == 20
    assert all(perf[s]["requests"] == perf[s]["valid_requests"] == 1 for s in ("candidate", "reference"))
    assert all(perf[s]["attempted_requests"] == 2 for s in ("candidate", "reference"))


def test_missing_work_is_distinguished_from_different_work(tmp_path):
    report = capture(tmp_path, [row(0, 40, None, None)], [row(0, 20)])
    perf, = report["performance"]
    assert perf["unknown_work_pairs"] == 1 and perf["different_work_pairs"] == 0
    assert perf["measurement_status"] == "measured" and not perf["comparable"]


@pytest.mark.parametrize("text, matched, unknown", [("C", 1, 0), ("another answer", 0, 1)])
def test_missing_decode_count_is_unknown_unless_text_proves_matching_work(tmp_path, text, matched, unknown):
    report = capture(tmp_path, [row(0, 40, tokens=None, text=text)], [row(0, 20)])
    perf, = report["performance"]
    assert perf["matched_pairs"] == matched
    assert perf["unknown_work_pairs"] == unknown
    assert perf["different_work_pairs"] == 0
    assert perf["measurement_status"] == "measured" and perf["comparable"] == bool(matched)


@pytest.mark.parametrize("source", ["absolute", None])
def test_offline_rejudge_removes_native_floor_without_changing_scores(tmp_path, source):
    from trtmc_aiperf_qual.cli import rejudge_reports

    metrics = {"native_score": 0.0, "trtmc_score": 0.0, "test": {"outcome": "pass"}}
    quality = [{"suite": "mmlu-0shot", "source": source, "status": "not-comparable",
                "samples": 200, "expected_samples": 200, "metrics": metrics,
                "gate": {"margin": 5.0, "min_native": 30.0}}]
    report = {**capture(tmp_path, [row(0, 40)], [row(0, 20)], quality), "provenance": {}}
    (tmp_path / "report.json").write_text(json.dumps(report))
    (tmp_path / "model.json").write_text(json.dumps({"absolute": [{"suite": "mmlu-0shot"}], "supplementary": []}))
    rejudge_reports([tmp_path])
    updated = json.loads((tmp_path / "report.json").read_text())
    assert updated["accuracy"][0]["metrics"] == metrics
    assert updated["accuracy"][0]["gate"] == {"margin": 5.0}
    assert updated["verdict"]["acc"] == "pass"
    assert json.loads((tmp_path / "report.original.json").read_text())["accuracy"] == quality


def test_partial_failed_workload_reports_coverage_and_available_timings(tmp_path):
    report = capture(tmp_path, [row(0, 40), row(1, None, status=500)], [row(0, 20), row(1, 200)])
    perf, = report["performance"]
    assert verdict(report)["perf"] == "partial" and verdict(report)["category"] == "measured"
    assert not perf["complete"] and perf["candidate"]["valid_requests"] == 1
    assert perf["reference"]["p50_ms"] == 110
    assert perf["measurement_status"] == "partial"


def test_no_candidate_timing_remains_an_error(tmp_path):
    report = capture(tmp_path, [row(0, None)], [row(0, 20)])
    assert verdict(report)["perf"] == "error"
    assert report["performance"][0]["measurement_status"] == "unavailable"


def test_refresh_old_exports_preserves_accuracy_and_raw_responses(tmp_path):
    report = capture(tmp_path, [row(0, 40), rejected(1)], [row(0, 20), row(1, 200)], [
        {"suite": "mmlu-0shot", "source": "absolute", "status": "pass", "out_of_capacity": 1,
         "gate": {"margin": 1.0}, "metrics": {"trtmc_score": 80, "native_score": 80}}])
    path = tmp_path / "execution.jsonl"
    old = [json.loads(line) for line in path.read_text().splitlines()]
    for batch in old:
        for r in batch["records"]:
            r.pop("capacity_rejection")
    path.write_text("".join(json.dumps(batch) + "\n" for batch in old))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.jsonl")}
    quality = copy.deepcopy(report["accuracy"])
    updated = benchmark_perf.refresh(tmp_path, report)
    assert updated["accuracy"] == quality
    assert updated["performance"][0]["reference"]["p50_ms"] == 20
    assert verdict(updated)["perf"] == "measured"
    assert all(p.read_bytes() == data for p, data in before.items())
    assert benchmark_perf.refresh(tmp_path, updated) == updated


def test_capacity_count_mismatch_is_not_silently_filtered(tmp_path):
    with pytest.raises(ValueError, match="capacity rejections"):
        capture(tmp_path, [row(0, 40), rejected(1)], [row(0, 20), row(1, 200)], [
            {"suite": "mmlu-0shot", "status": "pass", "out_of_capacity": 2}])


def old_empty_answer_report(out):
    report = capture(out, [row(0, 40, tokens=0, text="", error={
        "type": "InvalidInferenceResultError", "message": "empty answer"})], [row(0, 20)])
    path = out / "execution.jsonl"
    batches = [json.loads(line) for line in path.read_text().splitlines()]
    batches[0]["records"][0].update(valid=False, output_valid=False, work=None)
    path.write_text("".join(json.dumps(batch) + "\n" for batch in batches))
    return report


def test_refresh_recovers_timing_of_a_wrong_empty_answer_without_changing_accuracy(tmp_path):
    report = old_empty_answer_report(tmp_path)
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.jsonl")}
    updated = benchmark_perf.refresh(tmp_path, report)
    assert verdict(updated)["perf"] == "measured"
    assert updated["performance"][0]["candidate"]["p50_ms"] == 40
    assert updated["accuracy"] == report["accuracy"]
    assert updated["execution"]["timing_recovered_records"] == 1
    assert updated["execution"]["records"] != report["execution"]["records"]
    assert all(path.read_bytes() == data for path, data in original.items())
    assert benchmark_perf.refresh(tmp_path, updated) == updated


@pytest.mark.parametrize("field", ["request_sha", "sample_id"])
def test_refresh_refuses_empty_answer_recovery_with_a_different_request_identity(tmp_path, field):
    report = old_empty_answer_report(tmp_path)
    path = tmp_path / "execution.jsonl"
    batches = [json.loads(line) for line in path.read_text().splitlines()]
    batches[0]["records"][0][field] = "different-request"
    path.write_text("".join(json.dumps(batch) + "\n" for batch in batches))
    with pytest.raises(ValueError, match="execution identity"):
        benchmark_perf.refresh(tmp_path, report)


def test_refresh_keeps_real_http_failures_partial(tmp_path):
    report = capture(tmp_path, [row(0, 40, status=500)], [row(0, 20)])
    updated = benchmark_perf.refresh(tmp_path, report)
    assert verdict(updated)["perf"] == "error"
    assert updated["execution"] == report["execution"]


def test_refresh_does_not_invent_matching_work_without_an_operation(tmp_path):
    report = old_empty_answer_report(tmp_path)
    path = tmp_path / "execution.jsonl"
    batches = [json.loads(line) for line in path.read_text().splitlines()]
    batches[0]["identity"].pop("operation")
    path.write_text("".join(json.dumps(batch) + "\n" for batch in batches))
    updated = benchmark_perf.refresh(tmp_path, report)
    perf, = updated["performance"]
    assert perf["unknown_work_pairs"] == 1
    assert perf["matched_pairs"] == 0
    assert perf["candidate"]["p50_ms"] == 40


def test_capacity_excluded_problem_leaves_all_seed_repetitions(tmp_path):
    capture(tmp_path, [row(0, 40), rejected(1)], [row(0, 20), row(1, 200)], [
        {"suite": "mmlu-0shot", "status": "pass", "out_of_capacity": 1}])
    batches = [json.loads(line) for line in (tmp_path / "execution.jsonl").read_text().splitlines()]
    more = copy.deepcopy(batches)
    for batch in more:
        for r in batch["records"]:
            r["request_sha"] += "-seed-two"
            r["capacity_rejection"] = None
            r["valid"] = True
            r["model_call_ms"] = 80
    perf = execution.paired_dataset("mmlu-0shot", [batches[0], more[0]], [batches[1], more[1]],
                                    lambda c, n: True, capacity_exclusions=1)
    assert perf["complete"] and perf["pairs"] == 2 and perf["out_of_capacity"] == 1
    assert all(perf[s]["requests"] == 2 and perf[s]["attempted_requests"] == 4
               for s in ("candidate", "reference"))
