# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-owned coverage for the generic OpenPI benchmark seam."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorrt_model_connect.benchmark.catalog import ManifestCatalog, resolve_case
from tensorrt_model_connect.benchmark.metrics import reduce_metrics


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
ADAPTER_SOURCE = REPOSITORY_ROOT / "src/runtime/models/openpi/benchmark_worker_adapter.h"
WORKER_SOURCE = REPOSITORY_ROOT / "examples/trtmc_benchmark_worker.cpp"


def test_openpi_manifest_resolves_to_a_deterministic_benchmark_request(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "pi05-droid.trtfb"
    bundle.write_bytes(b"bundle")
    case = resolve_case(ManifestCatalog().resolve("pi05-droid"), bundle)

    assert case.operation == "predict_actions"
    assert case.request == {
        "batch_size": 1,
        "prompt": "pick up the object",
        "seed": 42,
    }
    assert case.sources["request.prompt"] == "operation default"
    assert case.sources["request.seed"] == "model testcase"


def test_openpi_benchmark_adapter_owns_constant_synthetic_inputs() -> None:
    adapter = ADAPTER_SOURCE.read_text(encoding="utf-8")
    worker = WORKER_SOURCE.read_text(encoding="utf-8")

    assert "input.state.assign(8U, 0.0F);" in adapter
    assert "input.initial_noise.assign(15U * 32U, 0.0F);" in adapter
    assert "camera.pixels.assign(kPixelsPerCamera, camera.valid ? 0.5F : 0.0F);" in adapter
    assert "dynamic_cast<IOpenPIActionPipeline*>(&pipeline)" in adapter
    assert "ActionRequest input;" not in worker
    assert '{"predict_actions", run_predict_actions}' in worker


def test_openpi_benchmark_metrics_report_policy_rate_and_stages() -> None:
    metrics = reduce_metrics(
        "predict_actions",
        [
            {
                "runtime_e2e_wall_ms": 20.0,
                "action_chunks": 1,
                "action_steps": 15,
                "preprocess_ms": 1.0,
                "prefill_ms": 2.0,
                "denoise_ms": 15.0,
                "postprocess_ms": 1.0,
            }
        ],
    )

    assert metrics["action_chunks_per_s"] == pytest.approx(50.0)
    assert metrics["action_steps_per_s"] == pytest.approx(750.0)
    assert metrics["reported_stages_ms"]["denoise_ms"]["p50"] == 15.0
