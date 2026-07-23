# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for OpenPI's model-owned performance receipt."""

from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.e2e.models.openpi import e2e_plugins
from tests.e2e.models.openpi.e2e_plugins import performance


def _baseline_document() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "openpi_baseline_performance_measurement",
        "backend": "pytorch_eager",
        "profile_name": "pi05_droid",
        "iterations": 1000,
        "warmups": 100,
        "latency_samples_ms": list(range(1, 1001)),
        "first_request_ms": 1.0,
        "determinism": {},
        "hardware": {
            "gpu_name": "NVIDIA GB300",
            "tensorrt_version": "11.2.0.113",
        },
        "workload": copy.deepcopy(performance._EXPECTED_WORKLOAD),
        "provenance": {
            "upstream_repository": "https://github.com/Physical-Intelligence/openpi.git",
            "upstream_commit": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
            "environment": {
                "effective_compile_mode": None,
                "forbidden_compiler_environment_names": [],
                "torch_compile_guard": {
                    "enforcement": "raise_on_invocation",
                    "invocation_count": 0,
                    "scope": "policy_construction_and_measurement",
                },
            },
        },
    }


def _case() -> SimpleNamespace:
    return SimpleNamespace(
        inputs={
            "profile": "pi05_droid",
            "reference_case_id": "droid_iris_20231204154425_f000005",
            "action_horizon": 15,
            "internal_action_dim": 32,
            "flow_steps": 10,
            "fixed_external_noise": True,
        },
        metadata={
            "performance_qualification": {
                "native_backend": "tensorrt_cpp",
                "baseline_backend": "pytorch_eager",
                "baseline_artifact": "performance/pytorch-eager.json",
                "baseline_sha256": performance.TORCH_EAGER_SHA256,
                "iterations": 1000,
                "warmups": 100,
            }
        },
    )


def _stderr(*, mean: float = 40.0, p50: float = 40.0, p95: float = 50.0) -> str:
    return (
        "TensorRT runtime detail\n"
        "[trtmc.openpi.benchmark] "
        f"action_ms={mean:.6f} p50_ms={p50:.6f} p95_ms={p95:.6f} "
        "iterations=1000 warmup=100\n"
    )


def test_torch_eager_validator_recomputes_nearest_rank_summary() -> None:
    summary = performance.validate_torch_eager_document(_baseline_document())

    assert summary["backend"] == "pytorch_eager"
    assert summary["latency_ms"] == {
        "mean_ms": 500.5,
        "p50_ms": 500.0,
        "p95_ms": 950.0,
    }
    assert summary["effective_compile_mode"] is None
    assert summary["torch_compile_guard_invocation_count"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("effective_compile_mode", "max-autotune"),
        ("forbidden_compiler_environment_names", ["TORCH_COMPILE"]),
        (
            "torch_compile_guard",
            {
                "enforcement": "log_only",
                "invocation_count": 0,
                "scope": "policy_construction_and_measurement",
            },
        ),
        (
            "torch_compile_guard",
            {
                "enforcement": "raise_on_invocation",
                "invocation_count": 1,
                "scope": "policy_construction_and_measurement",
            },
        ),
    ],
)
def test_torch_eager_validator_rejects_compiler_provenance(field: str, value: object) -> None:
    document = _baseline_document()
    document["provenance"]["environment"][field] = value

    with pytest.raises(ValueError, match="Torch Eager baseline"):
        performance.validate_torch_eager_document(document)


def test_torch_eager_validator_rejects_stale_workload_and_nonfinite_samples() -> None:
    stale = _baseline_document()
    stale["workload"]["case_id"] = "other-case"
    with pytest.raises(ValueError, match="workload is stale or mismatched"):
        performance.validate_torch_eager_document(stale)

    malformed = _baseline_document()
    malformed["latency_samples_ms"][4] = float("nan")
    with pytest.raises(ValueError, match="finite and positive"):
        performance.validate_torch_eager_document(malformed)


def test_case_validator_requires_every_pinned_workload_field() -> None:
    baseline = performance.validate_torch_eager_document(_baseline_document())
    performance.validate_case(_case(), baseline)

    for field in (
        "profile",
        "reference_case_id",
        "action_horizon",
        "internal_action_dim",
        "flow_steps",
        "fixed_external_noise",
    ):
        case = _case()
        del case.inputs[field]
        with pytest.raises(ValueError, match=rf"{field} differs"):
            performance.validate_case(case, baseline)

    case = _case()
    case.metadata.clear()
    with pytest.raises(ValueError, match="qualification declaration"):
        performance.validate_case(case, baseline)


def test_baseline_loader_rejects_an_unpinned_artifact(monkeypatch, tmp_path: Path) -> None:
    artifact = tmp_path / "performance" / "pytorch-eager.json"
    artifact.parent.mkdir()
    artifact.write_text(json.dumps(_baseline_document()), encoding="utf-8")
    monkeypatch.setattr(
        performance,
        "openpi_proof_path",
        lambda *parts: tmp_path.joinpath(*parts),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        performance.load_torch_eager_baseline()


@pytest.mark.parametrize(
    "stderr",
    [
        _stderr() + _stderr(),
        _stderr() + "[trtmc.openpi.benchmark] malformed\n",
        _stderr().replace("iterations=1000", "iterations=999"),
        _stderr().replace("p95_ms=50.000000", "p95_ms=nan"),
        _stderr().replace(" warmup=100", "  warmup=100"),
        _stderr() + " " + _stderr().splitlines()[1] + "\n",
        _stderr(p50=51.0, p95=50.0),
    ],
)
def test_native_benchmark_parser_rejects_duplicate_malformed_or_stale_lines(
    stderr: str,
) -> None:
    with pytest.raises(ValueError, match="benchmark"):
        performance.parse_native_benchmark(stderr)


def test_native_benchmark_parser_records_backend_and_exact_counts() -> None:
    summary = performance.parse_native_benchmark(_stderr())

    assert summary == {
        "backend": "tensorrt_cpp",
        "runtime": "trtmc-openpi",
        "runtime_contract": "native_cpp_tensorrt",
        "iterations": 1000,
        "warmups": 100,
        "latency_ms": {
            "mean_ms": 40.0,
            "p50_ms": 40.0,
            "p95_ms": 50.0,
        },
    }


def test_persisted_receipt_is_revalidated_against_stderr_and_baseline(monkeypatch) -> None:
    baseline = performance.validate_torch_eager_document(_baseline_document())
    monkeypatch.setattr(performance, "load_torch_eager_baseline", lambda: baseline)
    receipt = performance.build_receipt(_stderr(), _case())

    native, loaded_baseline = performance.validate_receipt(receipt)
    assert native["latency_ms"]["p50_ms"] == 40.0
    assert loaded_baseline == baseline

    receipt["native"]["latency_ms"]["p50_ms"] = 41.0
    with pytest.raises(ValueError, match="differs from its stderr"):
        performance.validate_receipt(receipt)


def test_manifest_uses_one_pinned_hf_snapshot_for_all_proof_inputs() -> None:
    model_config = tomllib.loads(Path(__file__).with_name("MODEL.toml").read_text(encoding="utf-8"))
    manifest = json.loads(
        (Path(__file__).with_name("manifests") / "pi05-droid.json").read_text(encoding="utf-8")
    )

    assert "model_artifact_cache" not in model_config
    assert manifest["hf_id"] == e2e_plugins.OPENPI_SNAPSHOT_REPO_ID
    assert manifest["hf_revision"] == e2e_plugins.OPENPI_SNAPSHOT_REVISION
    declaration = manifest["testcases"][0]["metadata"]["performance_qualification"]
    assert declaration["baseline_artifact"] == "performance/pytorch-eager.json"
    assert declaration["baseline_sha256"] == performance.TORCH_EAGER_SHA256
