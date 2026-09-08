# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral controls for report evidence and managed bundle reuse."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import trtmc_benchmark.builder as builder_module
from trtmc_benchmark.builder import BundleBuilder
from trtmc_benchmark.report import generate_collection_report, write_html_report
from trtmc_benchmark.types import BenchmarkError, MeasurementSpec, ModelDescriptor, ResolvedCase


def _cache_fixture(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "snapshots" / ("a" * 40)
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type":"example"}')
    source = tmp_path / "family" / "__init__.py"
    source.parent.mkdir()
    source.write_text("BUILD_VERSION = 1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"name":"example","precision":"fp16"}')
    monkeypatch.setattr(builder_module, "_resolve_model", lambda *_: checkpoint)
    monkeypatch.setattr(
        builder_module.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(source))
    )
    monkeypatch.setattr(builder_module.tensorrt_model_connect, "__file__", str(source))
    calls = []

    def build(command, **_):
        calls.append(command)
        Path(command[command.index("-o") + 1]).write_bytes(b"built bundle")
        return subprocess.CompletedProcess(command, 0, "built", "")

    monkeypatch.setattr(builder_module.subprocess, "run", build)
    model = ModelDescriptor(
        "example",
        "example/model",
        "",
        "model.bundle",
        "example",
        "text_generation",
        "fp16",
        manifest,
        (),
        {"max_sequence_length": 64},
    )
    builder = BundleBuilder(tmp_path / "cache")
    case = ResolvedCase(
        "case",
        model,
        "case",
        builder.provisional_path(model),
        "generate",
        {"prompt": "Hello"},
        None,
        MeasurementSpec(1, 2),
        {},
    )
    return builder, case, source, checkpoint, calls


def test_managed_bundle_reuses_only_matching_recorded_build(tmp_path, monkeypatch):
    builder, case, _, _, calls = _cache_fixture(tmp_path, monkeypatch)
    _, built = builder.prepare([case], allow_build=True, rebuild=False, dry_run=False)
    _, reused = builder.prepare([case], allow_build=False, rebuild=False, dry_run=False)
    assert [built[0].status, reused[0].status] == ["built", "reused"]
    assert len(calls) == 1
    sidecar = case.bundle_path.with_suffix(".bundle.benchmark.json")
    assert str(tmp_path) not in sidecar.read_text()


@pytest.mark.parametrize(
    "change", ["manifest", "options", "source", "checkpoint", "bundle", "bundle_preserved_mtime"]
)
def test_changed_inputs_cannot_reuse_managed_bundle(tmp_path, monkeypatch, change):
    builder, case, source, checkpoint, calls = _cache_fixture(tmp_path, monkeypatch)
    builder.prepare([case], allow_build=True, rebuild=False, dry_run=False)
    if change == "manifest":
        case.model.manifest_path.write_text('{"name":"example","precision":"fp32"}')
    elif change == "options":
        case = replace(case, model=replace(case.model, build_settings={"max_sequence_length": 128}))
    elif change == "source":
        source.write_text("BUILD_VERSION = 2\n")
    elif change == "checkpoint":
        checkpoint = checkpoint.parent / ("b" * 40)
        checkpoint.mkdir()
        monkeypatch.setattr(builder_module, "_resolve_model", lambda *_: checkpoint)
    elif change == "bundle_preserved_mtime":
        original = case.bundle_path.stat()
        case.bundle_path.write_bytes(b"x" * original.st_size)
        os.utime(case.bundle_path, ns=(original.st_atime_ns, original.st_mtime_ns))
        assert case.bundle_path.stat().st_size == original.st_size
        assert case.bundle_path.stat().st_mtime_ns == original.st_mtime_ns
    else:
        case.bundle_path.write_bytes(b"replacement bundle")
    with pytest.raises(BenchmarkError, match="no matching immutable build identity"):
        builder.prepare([case], allow_build=False, rebuild=False, dry_run=False)
    _, rebuilt = builder.prepare([case], allow_build=True, rebuild=False, dry_run=False)
    assert rebuilt[0].status == "built"
    assert len(calls) == 2


def test_unreceipted_managed_bundle_does_not_silently_pass_no_build(tmp_path, monkeypatch):
    builder, case, _, _, calls = _cache_fixture(tmp_path, monkeypatch)
    case.bundle_path.parent.mkdir(parents=True)
    case.bundle_path.write_bytes(b"legacy cache bundle")
    with pytest.raises(BenchmarkError, match="no matching immutable build identity"):
        builder.prepare([case], allow_build=False, rebuild=False, dry_run=False)
    assert calls == []


def test_explicit_mutable_checkpoint_is_rebuilt_even_if_unchanged(tmp_path, monkeypatch):
    builder, case, _, checkpoint, calls = _cache_fixture(tmp_path, monkeypatch)
    builder.model_dirs[case.model.name] = checkpoint
    builder.prepare([case], allow_build=True, rebuild=False, dry_run=False)
    with pytest.raises(BenchmarkError, match="no matching immutable build identity"):
        builder.prepare([case], allow_build=False, rebuild=False, dry_run=False)
    builder.prepare([case], allow_build=True, rebuild=False, dry_run=False)
    assert len(calls) == 2
    assert not case.bundle_path.with_suffix(".bundle.benchmark.json").exists()


def _run(root, run_id, schema="v2", scope="public_task_call_wall", p50=10.0):
    root.mkdir()
    case = root / "case"
    case.mkdir()
    (case / "resolved-case.json").write_text(
        json.dumps(
            {
                "request": {"prompt": "Hello", "max_new_tokens": 8},
                "model": {"precision": "fp16", "build": {"max_sequence_length": 64}},
                "measurement": {"warmup": 3, "iterations": 10},
            }
        )
    )
    (case / "observations.jsonl").write_text('{"runtime_e2e_wall_ms": 10.0}\n')
    payload = {
        "schema_version": "trtmc.benchmark-run/" + schema,
        "run_id": run_id,
        "status": "completed",
        "started_at": "2026-01-0" + run_id + "T00:00:00Z",
        "measurement_policy": {"timing_scope": scope, "load_excluded": True},
        "environment": {
            "hostname": "synthetic-host-secret",
            "worker": "/private/worker",
            "gpus": [
                {"name": "Example GPU", "driver_version": "1.0", "uuid": "synthetic-uuid-secret"}
            ],
            "python": "3.12",
        },
        "cells": [
            {
                "model": "example <model>",
                "name": "case",
                "operation": "generate",
                "status": "completed",
                "artifact_dir": "case",
                "timing_scope": scope,
                "asset_loading_included": False,
                "metrics": {"latency_ms": {"p50": p50, "p95": p50 + 2}, "tokens_per_s": 80.0},
            }
        ],
    }
    (root / "result.json").write_text(json.dumps(payload))
    return payload


def test_report_preserves_legacy_contract_and_exposes_metrics_and_evidence(tmp_path):
    _run(tmp_path / "old", "1", "v1", "public_pipeline_call_wall")
    _run(tmp_path / "new", "2")
    report, warnings = generate_collection_report([tmp_path], tmp_path / "report")
    document = (tmp_path / "report/report.html").read_text()
    assert not warnings
    assert {run["schema_version"] for run in report["runs"]} == {
        "trtmc.benchmark-run/v1",
        "trtmc.benchmark-run/v2",
    }
    for value in (
        "10.000 ms",
        "12.000 ms",
        "80.000 tokens_per_s",
        "public_pipeline_call_wall",
        "public_task_call_wall",
        "Example GPU",
        "Resolved inputs / reproduction",
        "Task quality is not evaluated",
        'id="filter"',
    ):
        assert value in document
    assert "../old/case/observations.jsonl" in document
    assert "example &lt;model&gt;" in document
    assert "synthetic-host-secret" not in document
    assert "synthetic-uuid-secret" not in document
    assert "/private/worker" not in document
    assert "% vs" not in document


def test_history_compares_only_matching_workloads_and_timing(tmp_path):
    _run(tmp_path / "first", "1", p50=10.0)
    _run(tmp_path / "second", "2", p50=8.0)
    _, _ = generate_collection_report([tmp_path], tmp_path / "report")
    assert "-20.0% vs 1" in (tmp_path / "report/report.html").read_text()
    changed = tmp_path / "second/case/resolved-case.json"
    request = json.loads(changed.read_text())
    request["request"]["max_new_tokens"] = 16
    changed.write_text(json.dumps(request))
    generate_collection_report([tmp_path], tmp_path / "report")
    assert "% vs" not in (tmp_path / "report/report.html").read_text()


def test_legacy_result_without_id_remains_renderable(tmp_path):
    payload = _run(tmp_path / "old", "1", "v1", "public_pipeline_call_wall")
    del payload["run_id"]
    (tmp_path / "old/result.json").write_text(json.dumps(payload))
    report, _ = generate_collection_report([tmp_path / "old"], tmp_path / "report")
    assert report["runs"][0]["run_id"].startswith("legacy-")
    assert report["runs"][0]["schema_version"] == "trtmc.benchmark-run/v1"


def test_report_does_not_link_artifacts_outside_the_case_root(tmp_path):
    payload = _run(tmp_path / "run", "1")
    payload["cells"][0]["artifact_dir"] = "../outside"
    write_html_report(payload, tmp_path / "run/report.html")
    assert "../outside" not in (tmp_path / "run/report.html").read_text()


@pytest.mark.parametrize(
    "schema, row_key, reference_key", [("v1", "cases", "baseline"), ("v2", "rows", "reference")]
)
def test_performance_report_preserves_historical_measurements(
    tmp_path, schema, row_key, reference_key
):
    from tools import perf_matrix

    scope = "model_call_wall" if schema == "v1" else "public_task_call_wall"
    results = {
        "schema_version": "trtmc.perf-matrix/" + schema,
        "status": "completed",
        "selected_entry_ids": ["example.generate"],
        row_key: [
            {
                "id": "example.generate",
                "model": "example",
                "operation": "generate",
                "status": "green",
                "candidate": {"samples_ms": [8.0, 10.0, 12.0], "timing_scope": scope},
                reference_key: {
                    "samples_ms": [12.0, 14.0, 16.0],
                    "measurement_policy": {"timing_scope": "public_operation_call_wall"},
                },
                "commands": {"candidate": {"argv": ["trtmc-bench", "run", "--model", "example"]}},
            }
        ],
    }
    original = json.dumps(results)
    (tmp_path / "results.json").write_text(original)
    assert perf_matrix.main(["report", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    document = (tmp_path / "report.html").read_text()
    assert report["source_schema_version"] == results["schema_version"]
    assert report["rows"][0][reference_key] == results[row_key][0][reference_key]
    assert "p50: 10.000 ms" in document
    assert "p95: 11.800 ms" in document
    assert scope in document
    assert "public_operation_call_wall" in document
    assert "trtmc-bench" in document
    assert 'id="filter"' in document
    assert (tmp_path / "results.json").read_text() == original
    if schema == "v1":
        with pytest.raises(perf_matrix.PerfMatrixError, match="unsupported results schema"):
            perf_matrix._load_results(tmp_path)
