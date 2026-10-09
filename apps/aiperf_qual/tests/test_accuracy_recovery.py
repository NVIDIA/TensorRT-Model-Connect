# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from trtmc_aiperf_qual import absolute
from trtmc_aiperf_qual.accuracy_recovery import SelectionArchive, recover
from trtmc_aiperf_qual.aiperf_runner import AiperfRun


def fixture(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    selected = [{"prompt": f"question {i}", "ground_truth": gold, "task": f"task-{i}"}
                for i, gold in enumerate("ABCD")]
    (cache / "original.json").write_text(json.dumps(selected))
    item = {"suite": "mmlu-0shot", "plugin": "trtmc_mmlu", "endpoint": "completions", "gate": {"margin": 1.0}}
    model = {"absolute": [item]}
    batches = []
    for side, warmup in (("reference", 1), ("candidate", 3)):
        run = tmp_path / side / "mmlu-0shot-greedy"
        run.mkdir(parents=True)
        (run / "inputs.json").write_text(json.dumps({"data": [
            {"session_id": f"session_{i:06d}", "payloads": [{"prompt": problem["prompt"]}]}
            for i, problem in enumerate(selected)]}))
        raw, grades, rows = [], [], []
        for session in range(4):
            i = (session + warmup) % 4
            answer = "ABCD"[i] if side == "reference" or i != 1 else "A"
            payload = {"prompt": selected[i]["prompt"]}
            metadata = {"session_num": session, "conversation_id": f"session_{i:06d}",
                        "x_request_id": f"{side}-{session}", "benchmark_phase": "profiling"}
            response = {"trtmc_timing": {"model_call_ms": 10 + i}, "usage": {"completion_tokens": 1}}
            raw.append({"metadata": metadata, "status": 200, "payload": payload,
                        "responses": [{"text": json.dumps(response)}]})
            grades.append({**metadata, "grader_name": "multiple_choice", "model_output": answer,
                           "passed": answer == "ABCD"[session], "actual": answer, "expected": "ABCD"[session]})
            rows.append({"sample_id": session, "unit_id": str(session), "model_call_ms": 10 + i,
                "valid": True, "output_valid": True, "work": [["output_tokens", 1]], "request_problems": [],
                "request_sha": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "output_ref": {"aiperf_run": str(run), "record_index": session}})
        for filename, values in (("profile_export_raw.jsonl", raw), ("accuracy_export.jsonl", grades)):
            (run / filename).write_text("".join(json.dumps(row) + "\n" for row in values))
        batches.append({"batch_id": len(batches), "workload": item["suite"], "role": "both", "records": rows,
            "aiperf_exit": 0, "expected_requests": 4, "gpu_busy_percent": 0, "warmup": warmup,
            "identity": {"side": side, "precision": "fp16", "concurrency": 1, "timing_scope": "task-call-wall"}})
    (tmp_path / "execution.jsonl").write_text("".join(json.dumps(row) + "\n" for row in batches))
    entry = {"suite": item["suite"], "source": "absolute", "benchmark": item["plugin"],
             "gate": {"margin": 40.0, "min_native": 20.0}, "native": {"precision": "fp16"}}
    return cache, model, {"accuracy": [entry], "performance": []}


def test_recovery_regrades_both_sides_and_pairs_by_question_with_original_gate(tmp_path):
    cache, model, report = fixture(tmp_path)
    result = recover(tmp_path, model, report, SelectionArchive(cache))
    entry = result["accuracy"][0]
    assert entry["metrics"]["native_score"] == 100.0 and entry["metrics"]["trtmc_score"] == 75.0
    assert entry["counts"]["native_only"] == 1 and entry["counts"]["both_correct"] == 3
    assert entry["gate"] == report["accuracy"][0]["gate"]
    perf = result["performance"][0]
    assert perf["pairs"] == 4 and perf["complete"] and perf["comparable"]
    assert perf["candidate"]["p50_ms"] == perf["reference"]["p50_ms"] == 11.5
    # Repeated recovery keeps the original raw outputs and produces identical scores.
    raw_path = tmp_path / "candidate/mmlu-0shot-greedy/profile_export_raw.jsonl"
    before = raw_path.read_bytes()
    again = recover(tmp_path, model, result, SelectionArchive(cache))
    assert again["accuracy"] == result["accuracy"] and raw_path.read_bytes() == before


def test_recovery_rejects_conflicting_original_selections_without_publishing(tmp_path):
    cache, model, report = fixture(tmp_path)
    wrong = json.loads((cache / "original.json").read_text())
    wrong[0]["ground_truth"] = "D"
    (cache / "conflict.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="conflicting"):
        recover(tmp_path, model, report, SelectionArchive(cache))
    assert not list(tmp_path.rglob("accuracy_export.aligned.jsonl"))


def test_live_reader_rejects_old_misaligned_grades(tmp_path):
    cache, _, _ = fixture(tmp_path)
    run = AiperfRun(tmp_path / "reference/mmlu-0shot-greedy", 0, [])
    problems = SelectionArchive(cache).selection(run)
    with pytest.raises(ValueError, match="mismatched gold"):
        absolute.plugin_side(run, problems)
