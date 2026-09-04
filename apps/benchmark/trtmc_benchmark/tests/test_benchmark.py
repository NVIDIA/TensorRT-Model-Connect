# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from trtmc_benchmark.builder import _build_command
from trtmc_benchmark.catalog import (
    ManifestCatalog,
    default_manifest_root,
    resolve_case,
)
from trtmc_benchmark.cli import main
from trtmc_benchmark.metrics import reduce_metrics
from trtmc_benchmark.report import generate_collection_report
from trtmc_benchmark.service import BenchmarkService
from trtmc_benchmark.types import BenchmarkError
from trtmc_benchmark.worker import find_worker


REPO = Path(__file__).resolve().parents[4]


def test_catalog_reads_family_owned_manifests_without_a_registry() -> None:
    entries = ManifestCatalog(REPO / "families").entries()
    assert {entry.family for entry in entries} == {
        path.name
        for path in (REPO / "families").iterdir()
        if path.is_dir() and not path.name.startswith("_")
    }
    distilgpt2 = next(entry for entry in entries if entry.name == "distilgpt2")
    assert distilgpt2.operation == "generate"
    assert distilgpt2.status == "ready"


def test_case_resolves_current_task_and_manifest_fields(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("distilgpt2")
    case = resolve_case(model, tmp_path / "model.bundle")
    assert case.operation == "generate"
    assert case.request["prompt"] == "Hello, I'm a language model"
    assert case.request["max_new_tokens"] == 12
    assert case.measurement.timing_scope == "public_task_call_wall"


def test_forecast_case_uses_public_forecast_request(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("chronos-bolt-tiny-official")
    case = resolve_case(model, tmp_path / "model.bundle")
    assert case.operation == "solve"
    assert case.request["past_values"][:2] == [100.1, 100.15]


def test_stereo_benchmark_uses_family_owned_images(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("fast-foundation-stereo")
    case = resolve_case(model, tmp_path / "model.bundle")
    assert Path(case.request["left_image_path"]).is_file()
    assert Path(case.request["right_image_path"]).is_file()


def test_robot_control_benchmark_uses_family_owned_observation(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("act-aloha-sim-transfer-cube")
    case = resolve_case(model, tmp_path / "model.bundle")
    assert case.operation == "control"
    assert Path(case.request["image_path"]).is_file()
    assert Path(case.request["state_path"]).is_file()


def test_build_command_is_the_current_closed_build_request(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("distilgpt2")
    case = resolve_case(model, tmp_path / "model.bundle")
    command = _build_command(model, tmp_path / "checkpoint", tmp_path / "model.bundle", (case,))
    assert command[:4] == (
        sys.executable,
        "-m",
        "tensorrt_model_connect",
        "build",
    )
    assert "--family" not in command
    assert command[command.index("--task") + 1] == "text_generation"
    joined = " ".join(command).lower()
    assert "profile" not in joined
    assert "source-revision" not in joined


def _worker(tmp_path: Path) -> Path:
    path = tmp_path / "worker"
    path.write_text(
        """#!/usr/bin/env python3
import json, sys
request = json.load(open(sys.argv[sys.argv.index('--request') + 1]))
output = sys.argv[sys.argv.index('--output') + 1]
count = request['measurement']['iterations']
result = {
  'schema_version': 'trtmc.benchmark-worker-result/v2',
  'status': 'completed',
  'case_name': request['case_name'],
  'operation': request['operation'],
  'timing_scope': 'public_task_call_wall',
  'asset_loading_included': request['measurement']['asset_loading_included'],
  'load_ms': 1.0,
  'observations': [
    {'runtime_e2e_wall_ms': 2.0, 'output_tokens': 3, 'prefill_ms': 0.5, 'decode_ms': 1.0}
    for _ in range(count)
  ],
  'output_summary': {'text': 'ok'},
}
json.dump(result, open(output, 'w'))
"""
    )
    path.chmod(0o755)
    return path


def test_service_runs_worker_and_writes_reports(tmp_path: Path) -> None:
    model = ManifestCatalog(REPO / "families").resolve("distilgpt2")
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"bundle")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    case = resolve_case(
        model,
        bundle,
        overrides={"measurement.warmup": 0, "measurement.iterations": 2, "telemetry.gpu": "off"},
    ).with_values(runtime_root=runtime)
    output = tmp_path / "results"
    result = BenchmarkService(_worker(tmp_path)).run((case,), output)
    assert result["status"] == "completed"
    assert result["cells"][0]["metrics"]["latency_ms"]["p50"] == 2.0
    assert (output / "result.json").is_file()
    assert (output / "report.html").is_file()


def test_metrics_keep_task_specific_rates() -> None:
    metrics = reduce_metrics(
        "generate_audio",
        [
            {
                "runtime_e2e_wall_ms": 100.0,
                "output_audio_seconds": 0.2,
                "output_samples": 4800,
            }
        ],
    )
    assert metrics["audio_seconds_per_s"] == pytest.approx(2.0)
    assert metrics["realtime_factor"] == pytest.approx(0.5)


def test_collection_rejects_duplicate_run_id_without_content_fingerprints(
    tmp_path: Path,
) -> None:
    for name in ("a", "b"):
        root = tmp_path / name
        root.mkdir()
        (root / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": "trtmc.benchmark-run/v2",
                    "run_id": "same",
                    "status": "completed",
                    "cells": [],
                }
            )
        )
    with pytest.raises(BenchmarkError, match="duplicate run_id"):
        generate_collection_report((tmp_path,), tmp_path / "report")


def test_cli_dry_run_uses_explicit_bundle_without_runtime(tmp_path: Path, capsys) -> None:
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"bundle")
    assert (
        main(
            [
                "run",
                "--model",
                "distilgpt2",
                "--manifest-root",
                str(REPO / "families"),
                "--bundle",
                str(bundle),
                "--dry-run",
                "--no-build",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["operation"] == "generate"


def test_native_examples_depend_only_on_public_headers() -> None:
    for name in ("benchmark_worker.cpp", "dataset_benchmark.cpp"):
        source = (REPO / "apps/benchmark/native" / name).read_text(encoding="utf-8")
        assert '#include "trtmc/task.h"' in source
        assert '#include "trtmc/runtime/family_loader.h"' in source
        assert '#include "src/' not in source


def test_worker_and_catalog_resolution_have_one_explicit_path(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    worker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    worker.chmod(0o755)
    assert find_worker(worker) == worker.resolve()

    with pytest.raises(BenchmarkError, match="use --worker"):
        find_worker()
    with pytest.raises(BenchmarkError, match="use --manifest-root"):
        default_manifest_root()

    worker_source = (REPO / "apps/benchmark/trtmc_benchmark/worker.py").read_text()
    catalog_source = (REPO / "apps/benchmark/trtmc_benchmark/catalog.py").read_text()
    assert "shutil.which" not in worker_source
    assert 'os.environ.get("TRTMC_BENCH_WORKER")' not in worker_source
    assert "TRTMC_BENCH_MANIFEST_ROOT" not in catalog_source
