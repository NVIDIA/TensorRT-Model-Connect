# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fail-closed matrix model-proof report composer."""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest


REVISION = "a" * 40


def _test_gpu_uuid(owner: str) -> str:
    value = hashlib.sha256(owner.encode("utf-8")).hexdigest()[:32]
    return f"GPU-{value[:8]}-{value[8:12]}-{value[12:16]}-{value[16:20]}-{value[20:32]}"


def _import_composer():
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return importlib.import_module("generate_model_proof_report")


def _tiny_wav(path: Path) -> None:
    samples = struct.pack("<4h", 0, 100, -100, 0)
    path.write_bytes(
        b"RIFF"
        + struct.pack("<I", 36 + len(samples))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )


def _tiny_png(path: Path) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"synthetic-png")


def _result(
    case: str,
    owner: str,
    strategy: str = "text_generation_causal",
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_name": case,
        "status": "pass",
        "oracle_level": "L1_external_reference",
        "timestamp": "2026-07-08T12:00:00Z",
        "case_config": {
            "name": case,
            "family": owner,
            "task_strategy": strategy,
            "reference_backend": "hf",
            "inputs": {"prompt": "hello"},
            "metadata": {"model_name": owner},
        },
        "stages": {"compare": {"status": "passed", "metrics": {}}},
        "stage_outputs": {
            "trt_generate": {"text": "hello", "data": {}},
            "ref_generate": {"text": "hello", "data": {}},
        },
        "artifacts": artifacts or {},
        "timing": {"total_s": 1.0},
    }


def _write_part(
    parts: Path,
    owner: str,
    case: str,
    *,
    suffix: str = "",
    attempt: int = 1,
    strategy: str = "text_generation_causal",
    suite: str = "premerge",
) -> Path:
    artifact = parts / f"model-proof-{owner}-{REVISION}-{attempt}"
    root = artifact / f"artifacts{suffix}"
    case_dir = root / "e2e" / case
    case_dir.mkdir(parents=True)

    artifacts: dict[str, Any] = {}
    if strategy == "text_to_audio":
        _tiny_wav(case_dir / "trt.wav")
        _tiny_wav(case_dir / "ref.wav")
        artifacts = {"trt_wav": "trt.wav", "ref_wav": "ref.wav"}
    elif strategy == "diffusion_media_generation":
        _tiny_png(case_dir / "trt.png")
        _tiny_png(case_dir / "ref.png")
        artifacts = {"trt_frames": "trt.png", "ref_frames": "ref.png"}

    (case_dir / "result.json").write_text(
        json.dumps(_result(case, owner, strategy, artifacts)), encoding="utf-8"
    )
    (root / "e2e" / "junit.xml").write_text(
        f'<testsuite><testcase name="test_model_e2e[{case}]" /></testsuite>',
        encoding="utf-8",
    )

    gpu_fields = {
        "gpu_resource_class": "shared",
        "gpu_slot_ids": [0],
        "gpu_slots_per_device": 4,
        "gpu_lease_evidence": "gpu-lease.json",
    }
    steps = {
        name: {"status": "passed", "evidence": f"{name}.log"}
        for name in (
            "projection_validation",
            "configure",
            "scratch_build",
            "dso_isolation",
            "cpp_tests",
            "e2e_reference",
            "engine_build_budget",
            "result_verification",
            "html_report",
        )
    }
    steps["python_tests"] = {"status": "skipped", "evidence": "none"}
    status = {
        "schema_version": 1,
        "report_kind": "model_proof",
        "model": owner,
        "source_revision": REVISION,
        "suite": suite,
        "outcome": "passed",
        "exit_code": 0,
        "validation_exit_code": 0,
        "report_exit_code": 0,
        "gpu_id": "1",
        "steps": steps,
        **gpu_fields,
    }
    proof = {
        "schema_version": 1,
        "passed": True,
        "model": owner,
        "source_revision": REVISION,
        "suite": suite,
        "runtime_model": owner.replace("-", "_"),
        "runtime_library": f"libtrtmc_model_{owner.replace('-', '_')}.so",
        "runtime_library_sha256": "b" * 64,
        "staged_runtime_library_sha256": "b" * 64,
        "sibling_model_count": 0,
        "model_dso_count": 1,
        "staged_model_dso_count": 1,
        "engine_builds_per_model": 1,
        "engine_build_count": 1,
        "engine_build_verification": "engine-build-verification.json",
        "gpu_id": "1",
        "network": "disabled",
        "plugin_search": "strict",
        **gpu_fields,
    }
    selection = {
        "schema_version": 1,
        "requested_model": owner,
        "suite": suite,
        "gpu_id": "1",
        "runtime_library": proof["runtime_library"],
        "e2e_test": f"tests/e2e/models/{owner}/test_{owner}_e2e.py",
        "e2e_cases": [{"name": case, "model": owner}],
        **gpu_fields,
    }
    for filename, payload in (
        ("model-proof-status.json", status),
        ("proof.json", proof),
        ("selection.json", selection),
    ):
        (root / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (root / "model-proof-report.html").write_text(
        f"<!doctype html><html><body>rich report for {owner}</body></html>",
        encoding="utf-8",
    )
    (root / "gpu-lease.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "model": owner,
                "source_revision": REVISION,
                "run_id": "42",
                "run_attempt": str(attempt),
                "job_id": "prove",
                "runner_name": f"runner-{owner}",
                "node_id": f"node-{owner}",
                "hostname": f"host-{owner}",
                "gpu_id": "1",
                "gpu_index": "1",
                "gpu_uuid": _test_gpu_uuid(owner),
                "gpu_slot": 0,
                "gpu_slots": [0],
                "gpu_slot_ids": [0],
                "slots_per_gpu": 4,
                "gpu_slots_per_device": 4,
                "resource_class": "shared",
                "gpu_resource_class": "shared",
                "lock_namespace": hashlib.sha256(f"node-{owner}".encode("utf-8")).hexdigest(),
                "acquired_at": "2026-07-21T12:00:00+00:00",
                "released_at": "2026-07-21T12:01:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _append_result_case(root: Path, owner: str, case: str) -> None:
    case_dir = root / "e2e" / case
    case_dir.mkdir()
    (case_dir / "result.json").write_text(
        json.dumps(_result(case, owner)), encoding="utf-8"
    )

    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["e2e_cases"].append({"name": case, "model": owner})
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    junit_path = root / "e2e" / "junit.xml"
    junit = junit_path.read_text(encoding="utf-8")
    junit_path.write_text(
        junit.replace(
            "</testsuite>",
            f'<testcase name="test_model_e2e[{case}]" /></testsuite>',
        ),
        encoding="utf-8",
    )


def _write_vlm_assessment(root: Path, cases: list[str], *, revision: str = REVISION) -> None:
    (root / "diffusion_vlm_assessment.json").write_text(
        json.dumps(
            {
                "model_id": "vlm-judge",
                "source_revision": revision,
                "workflow_run_id": "42",
                "workflow_run_attempt": 2,
                "coverage_complete": True,
                "expected_case_names": sorted(cases),
                "assessed_case_names": sorted(cases),
                "results": [
                    {
                        "case_name": case,
                        "vlm_judgment": {"vlm_gate": {"failed": False}},
                    }
                    for case in cases
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _run(
    tmp_path: Path,
    expected: Any,
    *,
    upstream_results: list[str] | None = None,
    suite: str = "premerge",
    expected_result_count: int | None = None,
    expected_cases_by_model: dict[str, list[str]] | None = None,
) -> tuple[int, Path, Path]:
    composer = _import_composer()
    output = tmp_path / "combined.html"
    status = tmp_path / "combined-status.json"
    raw_expected = expected if isinstance(expected, str) else json.dumps(expected)
    args = [
        "--parts-dir",
        str(tmp_path / "parts"),
        "--expected-models",
        raw_expected,
        "--revision",
        REVISION,
        "--run-id",
        "42",
        "--expected-job-id",
        "prove",
        "--suite",
        suite,
        "--project-dir",
        str(tmp_path),
        "--output",
        str(output),
        "--status-output",
        str(status),
    ]
    if expected_result_count is not None:
        args.extend(("--expected-result-count", str(expected_result_count)))
    if expected_cases_by_model is not None:
        args.extend(
            (
                "--expected-cases-by-model",
                json.dumps(expected_cases_by_model, separators=(",", ":"), sort_keys=True),
            )
        )
    for result in upstream_results or []:
        args.extend(("--upstream-result", result))
    rc = composer.main(args)
    return rc, output, status


def test_combines_audio_and_visual_proofs_in_old_style_report(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    _write_part(parts, "audio-owner", "audio-case", strategy="text_to_audio")
    _write_part(parts, "visual-owner", "visual-case", strategy="diffusion_media_generation")

    rc, output, status_path = _run(tmp_path, ["audio-owner", "visual-owner"])

    assert rc == 0
    rendered = output.read_text(encoding="utf-8")
    assert "Isolated Model Proofs" in rendered
    assert "audio-owner" in rendered
    assert "visual-owner" in rendered
    assert "Summary" in rendered
    assert "Model Details" in rendered
    assert rendered.count("data:audio/wav;base64,") == 2
    assert rendered.count("data:image/png;base64,") == 2
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["outcome"] == "passed"
    assert status["expected_count"] == 2
    assert status["result_count"] == 2
    assert [item["status"] for item in status["models"]] == ["passed", "passed"]


def test_combined_status_records_authenticated_gpu_lease_identity(tmp_path: Path) -> None:
    root = _write_part(tmp_path / "parts", "alpha", "alpha-case")

    rc, _output, status_path = _run(tmp_path, ["alpha"])

    assert rc == 0
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["workflow_run_id"] == "42"
    assert status["workflow_job_id"] == "prove"
    assert status["gpu_lease_count"] == 1
    lease = status["gpu_leases"][0]
    source = json.loads((root / "gpu-lease.json").read_text(encoding="utf-8"))
    for field in (
        "model",
        "run_id",
        "job_id",
        "runner_name",
        "node_id",
        "hostname",
        "gpu_index",
        "gpu_uuid",
        "gpu_slot_ids",
        "slots_per_gpu",
        "resource_class",
        "lock_namespace",
        "acquired_at",
        "released_at",
    ):
        assert lease[field] == source[field]
    assert lease["run_attempt"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "GPU lease schema_version"),
        ("run_id", "41", "GPU lease run_id"),
        ("run_attempt", "2", "GPU lease run_attempt"),
        ("job_id", "other-job", "GPU lease job_id"),
        ("runner_name", "", "GPU lease runner_name"),
        ("node_id", "../node", "GPU lease node_id"),
        ("hostname", "host\nname", "GPU lease hostname"),
        ("gpu_index", "01", "GPU lease gpu_index"),
        ("gpu_uuid", "GPU-not-a-uuid", "GPU lease gpu_uuid"),
        ("slots_per_gpu", True, "slots-per-GPU evidence"),
        ("gpu_slot_ids", [True], "GPU lease gpu_slot_ids"),
        ("lock_namespace", "short", "GPU lease lock_namespace"),
        ("acquired_at", "2026-07-21T12:00:00", "acquired_at has no timezone"),
        (
            "released_at",
            "2026-07-21T11:59:00+00:00",
            "released_at precedes acquired_at",
        ),
        (
            "released_at",
            "2026-07-21T12:00:00+00:00",
            "zero-length ownership interval",
        ),
    ],
)
def test_combined_report_rejects_malformed_gpu_lease_evidence(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    root = _write_part(tmp_path / "parts", "alpha", "alpha-case")
    lease_path = root / "gpu-lease.json"
    lease = json.loads(lease_path.read_text(encoding="utf-8"))
    lease[field] = value
    lease_path.write_text(json.dumps(lease), encoding="utf-8")

    rc, _output, status_path = _run(tmp_path, ["alpha"])

    assert rc == 2
    issues = "\n".join(json.loads(status_path.read_text(encoding="utf-8"))["issues"])
    assert message in issues


def _copy_lease_topology(source: Path, destination: Path) -> None:
    source_lease = json.loads((source / "gpu-lease.json").read_text(encoding="utf-8"))
    destination_path = destination / "gpu-lease.json"
    destination_lease = json.loads(destination_path.read_text(encoding="utf-8"))
    for field in (
        "node_id",
        "hostname",
        "gpu_id",
        "gpu_index",
        "gpu_uuid",
        "gpu_slot",
        "gpu_slots",
        "gpu_slot_ids",
        "slots_per_gpu",
        "gpu_slots_per_device",
        "lock_namespace",
    ):
        destination_lease[field] = source_lease[field]
    destination_path.write_text(json.dumps(destination_lease), encoding="utf-8")


def test_combined_report_rejects_overlapping_duplicate_gpu_slot(tmp_path: Path) -> None:
    alpha = _write_part(tmp_path / "parts", "alpha", "alpha-case")
    beta = _write_part(tmp_path / "parts", "beta", "beta-case")
    _copy_lease_topology(alpha, beta)

    rc, _output, status_path = _run(tmp_path, ["alpha", "beta"])

    assert rc == 2
    issues = "\n".join(json.loads(status_path.read_text(encoding="utf-8"))["issues"])
    assert "overlapping GPU leases duplicate slot ownership" in issues


def test_combined_report_allows_non_overlapping_reuse_of_one_gpu_slot(
    tmp_path: Path,
) -> None:
    alpha = _write_part(tmp_path / "parts", "alpha", "alpha-case")
    beta = _write_part(tmp_path / "parts", "beta", "beta-case")
    _copy_lease_topology(alpha, beta)
    alpha_lease = json.loads((alpha / "gpu-lease.json").read_text(encoding="utf-8"))
    beta_path = beta / "gpu-lease.json"
    beta_lease = json.loads(beta_path.read_text(encoding="utf-8"))
    beta_lease["runner_name"] = alpha_lease["runner_name"]
    beta_lease["acquired_at"] = alpha_lease["released_at"]
    beta_lease["released_at"] = "2026-07-21T12:02:00+00:00"
    beta_path.write_text(json.dumps(beta_lease), encoding="utf-8")

    rc, _output, status_path = _run(tmp_path, ["alpha", "beta"])

    assert rc == 0
    assert json.loads(status_path.read_text(encoding="utf-8"))["outcome"] == "passed"


def test_combined_report_rejects_inconsistent_node_identity(tmp_path: Path) -> None:
    alpha = _write_part(tmp_path / "parts", "alpha", "alpha-case")
    beta = _write_part(tmp_path / "parts", "beta", "beta-case")
    alpha_lease = json.loads((alpha / "gpu-lease.json").read_text(encoding="utf-8"))
    beta_path = beta / "gpu-lease.json"
    beta_lease = json.loads(beta_path.read_text(encoding="utf-8"))
    beta_lease["node_id"] = alpha_lease["node_id"]
    beta_lease["acquired_at"] = alpha_lease["released_at"]
    beta_lease["released_at"] = "2026-07-21T12:02:00+00:00"
    beta_path.write_text(json.dumps(beta_lease), encoding="utf-8")

    rc, _output, status_path = _run(tmp_path, ["alpha", "beta"])

    assert rc == 2
    issues = "\n".join(json.loads(status_path.read_text(encoding="utf-8"))["issues"])
    assert "maps to multiple hostname/lock namespaces" in issues


def test_combined_report_rejects_inconsistent_node_slots_per_gpu(tmp_path: Path) -> None:
    alpha = _write_part(tmp_path / "parts", "alpha", "alpha-case")
    beta = _write_part(tmp_path / "parts", "beta", "beta-case")
    _copy_lease_topology(alpha, beta)
    for filename in (
        "model-proof-status.json",
        "proof.json",
        "selection.json",
        "gpu-lease.json",
    ):
        path = beta / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["gpu_slots_per_device"] = 8
        if filename == "gpu-lease.json":
            payload["slots_per_gpu"] = 8
            payload["acquired_at"] = "2026-07-21T12:01:00+00:00"
            payload["released_at"] = "2026-07-21T12:02:00+00:00"
        path.write_text(json.dumps(payload), encoding="utf-8")

    rc, _output, status_path = _run(tmp_path, ["alpha", "beta"])

    assert rc == 2
    issues = "\n".join(json.loads(status_path.read_text(encoding="utf-8"))["issues"])
    assert "maps to multiple slots-per-GPU values" in issues


def test_combined_report_rejects_one_runner_claiming_overlapping_jobs(
    tmp_path: Path,
) -> None:
    alpha = _write_part(tmp_path / "parts", "alpha", "alpha-case")
    beta = _write_part(tmp_path / "parts", "beta", "beta-case")
    alpha_lease = json.loads((alpha / "gpu-lease.json").read_text(encoding="utf-8"))
    beta_path = beta / "gpu-lease.json"
    beta_lease = json.loads(beta_path.read_text(encoding="utf-8"))
    beta_lease["runner_name"] = alpha_lease["runner_name"]
    beta_path.write_text(json.dumps(beta_lease), encoding="utf-8")

    rc, _output, status_path = _run(tmp_path, ["alpha", "beta"])

    assert rc == 2
    issues = "\n".join(json.loads(status_path.read_text(encoding="utf-8"))["issues"])
    assert "overlapping GPU leases claim the same runner" in issues


def test_combined_report_rejects_one_runner_moving_between_nodes(
    tmp_path: Path,
) -> None:
    alpha = _write_part(tmp_path / "parts", "alpha", "alpha-case")
    beta = _write_part(tmp_path / "parts", "beta", "beta-case")
    alpha_lease = json.loads((alpha / "gpu-lease.json").read_text(encoding="utf-8"))
    beta_path = beta / "gpu-lease.json"
    beta_lease = json.loads(beta_path.read_text(encoding="utf-8"))
    beta_lease["runner_name"] = alpha_lease["runner_name"]
    beta_lease["acquired_at"] = alpha_lease["released_at"]
    beta_lease["released_at"] = "2026-07-21T12:02:00+00:00"
    beta_path.write_text(json.dumps(beta_lease), encoding="utf-8")

    rc, _output, status_path = _run(tmp_path, ["alpha", "beta"])

    assert rc == 2
    issues = "\n".join(json.loads(status_path.read_text(encoding="utf-8"))["issues"])
    assert "maps to multiple node/physical identities" in issues


def test_nightly_report_uses_nightly_title(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    _write_part(parts, "alpha", "alpha-case", suite="nightly")

    rc, output, status_path = _run(
        tmp_path,
        ["alpha"],
        suite="nightly",
        expected_result_count=1,
        expected_cases_by_model={"alpha": ["alpha-case"]},
    )

    assert rc == 0
    rendered = output.read_text(encoding="utf-8")
    assert f"Nightly Isolated Model Report: {REVISION[:12]}" in rendered
    assert "Premerge Isolated Model Report" not in rendered
    assert json.loads(status_path.read_text(encoding="utf-8"))["suite"] == "nightly"


def test_expected_result_count_is_recorded_and_exact(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    _write_part(parts, "alpha", "alpha-case", suite="nightly")
    _write_part(parts, "beta", "beta-case", suite="nightly")

    rc, _, status_path = _run(
        tmp_path,
        ["alpha", "beta"],
        suite="nightly",
        expected_result_count=2,
        expected_cases_by_model={
            "alpha": ["alpha-case"],
            "beta": ["beta-case"],
        },
    )

    assert rc == 0
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["outcome"] == "passed"
    assert status["expected_cases_by_model"] == {
        "alpha": ["alpha-case"],
        "beta": ["beta-case"],
    }
    assert status["expected_result_count"] == 2
    assert status["result_count"] == 2


def test_result_count_excludes_pytest_only_summary_rows(tmp_path: Path) -> None:
    root = _write_part(tmp_path / "parts", "alpha", "alpha-case")
    (root / "e2e" / "junit.xml").write_text(
        '<testsuite><testcase name="test_model_e2e[alpha-case]" />'
        '<testcase name="test_model_e2e[diagnostic-only]" /></testsuite>',
        encoding="utf-8",
    )

    rc, _, status_path = _run(
        tmp_path,
        ["alpha"],
        expected_result_count=1,
    )

    assert rc == 2
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["outcome"] == "failed"
    assert status["expected_result_count"] == 1
    assert status["result_count"] == 1


def test_result_count_under_inventory_fails_closed(tmp_path: Path) -> None:
    _write_part(tmp_path / "parts", "alpha", "alpha-case", suite="nightly")

    rc, output, status_path = _run(
        tmp_path,
        ["alpha"],
        suite="nightly",
        expected_result_count=2,
        expected_cases_by_model={"alpha": ["alpha-case", "missing-alpha-case"]},
    )

    assert rc == 2
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["outcome"] == "failed"
    assert status["expected_result_count"] == 2
    assert status["result_count"] == 1
    assert any(
        "contains 1 certified E2E results; expected exactly 2" in issue
        for issue in status["issues"]
    )
    assert "expected exactly" in output.read_text(encoding="utf-8")


def test_result_count_over_inventory_fails_closed(tmp_path: Path) -> None:
    root = _write_part(
        tmp_path / "parts", "alpha", "alpha-case", suite="nightly"
    )
    _append_result_case(root, "alpha", "zeta-alpha-case")

    rc, output, status_path = _run(
        tmp_path,
        ["alpha"],
        suite="nightly",
        expected_result_count=1,
        expected_cases_by_model={"alpha": ["alpha-case"]},
    )

    assert rc == 2
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["outcome"] == "failed"
    assert status["expected_result_count"] == 1
    assert status["result_count"] == 2
    assert any(
        "contains 2 certified E2E results; expected exactly 1" in issue
        for issue in status["issues"]
    )
    assert "expected exactly" in output.read_text(encoding="utf-8")


def test_expected_cases_reject_equal_cardinality_substitution(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    _write_part(parts, "alpha", "alpha-case", suite="nightly")
    _write_part(parts, "beta", "beta-case", suite="nightly")

    rc, output, status_path = _run(
        tmp_path,
        ["alpha", "beta"],
        suite="nightly",
        expected_result_count=2,
        expected_cases_by_model={
            "alpha": ["replacement-alpha-case"],
            "beta": ["beta-case"],
        },
    )

    assert rc == 2
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["result_count"] == status["expected_result_count"] == 2
    assert status["expected_cases_by_model"]["alpha"] == ["replacement-alpha-case"]
    assert any("do not exactly match inventory cases" in issue for issue in status["issues"])
    assert "replacement-alpha-case" in output.read_text(encoding="utf-8")


def test_nightly_requires_both_inventory_result_contracts(tmp_path: Path) -> None:
    _write_part(tmp_path / "parts", "alpha", "alpha-case", suite="nightly")

    rc, _, status_path = _run(tmp_path, ["alpha"], suite="nightly")

    assert rc == 2
    issues = json.loads(status_path.read_text(encoding="utf-8"))["issues"]
    assert "nightly requires expected cases by model" in issues
    assert "nightly requires an expected result count" in issues


def test_nightly_diffusion_report_requires_current_complete_vlm_provenance(
    tmp_path: Path,
) -> None:
    parts = tmp_path / "parts"
    root = _write_part(
        parts,
        "visual-owner",
        "visual-case",
        strategy="diffusion_media_generation",
        suite="nightly",
    )
    _write_vlm_assessment(root, ["visual-case"])

    nightly_inventory = {
        "suite": "nightly",
        "expected_result_count": 1,
        "expected_cases_by_model": {"visual-owner": ["visual-case"]},
    }
    rc, _, status_path = _run(
        tmp_path,
        ["visual-owner"],
        **nightly_inventory,
    )

    assert rc == 0
    assert json.loads(status_path.read_text(encoding="utf-8"))["outcome"] == "passed"

    _write_vlm_assessment(root, ["visual-case"], revision="b" * 40)
    rc, _, status_path = _run(
        tmp_path,
        ["visual-owner"],
        **nightly_inventory,
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))

    assert rc != 0
    assert status["outcome"] == "failed"
    assert any(
        "diffusion assessment source_revision" in issue
        for issue in status["issues"]
    )


def test_combined_report_rebases_isolated_src_input_media(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    audio_root = _write_part(
        parts, "audio-owner", "audio-case", strategy="speech_to_text"
    )
    visual_root = _write_part(
        parts,
        "visual-owner",
        "visual-case",
        strategy="vision_language_generation",
    )

    audio_input = tmp_path / "tests/e2e/models/audio-owner/data/source.wav"
    audio_input.parent.mkdir(parents=True)
    _tiny_wav(audio_input)
    visual_input = tmp_path / "tests/e2e/models/visual-owner/data/source.png"
    visual_input.parent.mkdir(parents=True)
    _tiny_png(visual_input)

    audio_result_path = audio_root / "e2e/audio-case/result.json"
    audio_result = json.loads(audio_result_path.read_text(encoding="utf-8"))
    audio_result["case_config"]["inputs"] = {
        "audio": "/src/tests/e2e/models/audio-owner/data/source.wav"
    }
    audio_result_path.write_text(json.dumps(audio_result), encoding="utf-8")

    visual_result_path = visual_root / "e2e/visual-case/result.json"
    visual_result = json.loads(visual_result_path.read_text(encoding="utf-8"))
    visual_result["case_config"]["inputs"] = {
        "image": "/src/tests/e2e/models/visual-owner/data/source.png"
    }
    visual_result_path.write_text(json.dumps(visual_result), encoding="utf-8")

    rc, output, status_path = _run(tmp_path, ["audio-owner", "visual-owner"])

    assert rc == 0
    rendered = output.read_text(encoding="utf-8")
    assert rendered.count("data:audio/wav;base64,") == 1
    assert rendered.count("data:image/png;base64,") == 1
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["outcome"] == "passed"


def test_missing_model_still_writes_report_and_status(tmp_path: Path) -> None:
    _write_part(tmp_path / "parts", "alpha", "alpha-case")

    rc, output, status_path = _run(tmp_path, ["alpha", "beta"])

    assert rc == 2
    assert output.is_file()
    assert "beta" in output.read_text(encoding="utf-8")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["outcome"] == "failed"
    assert status["missing_models"] == ["beta"]
    assert next(item for item in status["models"] if item["model"] == "beta")["status"] == "missing"


def test_duplicate_and_unexpected_artifacts_fail_exact_set(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    _write_part(parts, "alpha", "alpha-case")
    _write_part(parts, "alpha", "alpha-case", suffix="-duplicate")
    _write_part(parts, "gamma", "gamma-case")

    rc, output, status_path = _run(tmp_path, ["alpha"])

    assert rc == 2
    assert output.is_file()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["duplicate_models"] == ["alpha"]
    assert status["unexpected_models"] == ["gamma"]
    assert status["same_attempt_duplicates"] == [
        {
            "artifact_roots": [
                f"model-proof-alpha-{REVISION}-1/artifacts",
                f"model-proof-alpha-{REVISION}-1/artifacts-duplicate",
            ],
            "attempt": 1,
            "model": "alpha",
        }
    ]


def test_latest_attempt_wins_without_duplicating_successful_prior_jobs(
    tmp_path: Path,
) -> None:
    parts = tmp_path / "parts"
    _write_part(parts, "alpha", "alpha-case", attempt=1)
    _write_part(parts, "alpha", "alpha-case", attempt=3)
    _write_part(parts, "beta", "beta-case", attempt=1)

    rc, output, status_path = _run(tmp_path, ["alpha", "beta"])

    assert rc == 0
    assert output.is_file()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["artifact_count"] == 3
    assert payload["selected_artifact_count"] == 2
    assert payload["duplicate_models"] == []
    assert {item["model"]: item["artifact_attempt"] for item in payload["models"]} == {
        "alpha": 3,
        "beta": 1,
    }
    assert payload["artifact_attempts"] == {"alpha": [1, 3], "beta": [1]}
    assert {item["model"]: item["artifact_attempts"] for item in payload["models"]} == {
        "alpha": [1, 3],
        "beta": [1],
    }


def test_malformed_older_status_does_not_poison_successful_latest_retry(
    tmp_path: Path,
) -> None:
    parts = tmp_path / "parts"
    older = _write_part(parts, "alpha", "alpha-case", attempt=1)
    (older / "model-proof-status.json").write_text("{truncated", encoding="utf-8")
    _write_part(parts, "alpha", "alpha-case", attempt=2)

    rc, output, status_path = _run(tmp_path, ["alpha"])

    assert rc == 0
    assert output.is_file()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["outcome"] == "passed"
    assert payload["issues"] == []
    assert payload["artifact_attempts"] == {"alpha": [1, 2]}
    assert payload["models"][0]["artifact_attempt"] == 2
    assert payload["models"][0]["artifact_attempts"] == [1, 2]


def test_malformed_latest_status_still_fails_closed(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    _write_part(parts, "alpha", "alpha-case", attempt=1)
    latest = _write_part(parts, "alpha", "alpha-case", attempt=2)
    (latest / "model-proof-status.json").write_text("{truncated", encoding="utf-8")

    rc, output, status_path = _run(tmp_path, ["alpha"])

    assert rc == 2
    assert output.is_file()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["outcome"] == "failed"
    assert payload["artifact_attempts"] == {"alpha": [1, 2]}
    assert payload["models"][0]["artifact_attempt"] == 2
    assert any("model-proof status is invalid" in issue for issue in payload["issues"])


def test_latest_failed_attempt_supersedes_an_older_passing_attempt(
    tmp_path: Path,
) -> None:
    parts = tmp_path / "parts"
    _write_part(parts, "alpha", "alpha-case", attempt=1)
    latest = _write_part(parts, "alpha", "alpha-case", attempt=2)
    status = json.loads((latest / "model-proof-status.json").read_text(encoding="utf-8"))
    status["outcome"] = "failed"
    (latest / "model-proof-status.json").write_text(json.dumps(status), encoding="utf-8")

    rc, output, status_path = _run(tmp_path, ["alpha"])

    assert rc == 2
    assert "status outcome" in output.read_text(encoding="utf-8")
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["models"][0]["artifact_attempt"] == 2
    assert payload["models"][0]["status"] == "failed"


def test_wrong_metadata_fallback_and_case_mismatch_fail_closed(tmp_path: Path) -> None:
    root = _write_part(tmp_path / "parts", "alpha", "alpha-case")
    status = json.loads((root / "model-proof-status.json").read_text(encoding="utf-8"))
    status["source_revision"] = "c" * 40
    (root / "model-proof-status.json").write_text(json.dumps(status), encoding="utf-8")
    (root / "model-proof-report.html").write_text(
        '<html data-report-kind="workflow-fallback"></html>', encoding="utf-8"
    )
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    selection["e2e_cases"][0]["name"] = "different-case"
    (root / "selection.json").write_text(json.dumps(selection), encoding="utf-8")

    rc, output, status_path = _run(tmp_path, ["alpha"])

    assert rc == 2
    assert "The report is incomplete" in output.read_text(encoding="utf-8")
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    errors = "\n".join(payload["issues"])
    assert "status source revision" in errors
    assert "workflow fallback" in errors
    assert "do not exactly match" in errors


def test_cross_model_case_collision_fails(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    _write_part(parts, "alpha", "shared-case")
    _write_part(parts, "beta", "shared-case")

    rc, output, status_path = _run(tmp_path, ["alpha", "beta"])

    assert rc == 2
    assert output.is_file()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert any("present in both" in issue for issue in payload["issues"])


def test_junit_failure_cannot_be_hidden_by_passing_result_json(tmp_path: Path) -> None:
    root = _write_part(tmp_path / "parts", "alpha", "alpha-case")
    (root / "e2e" / "junit.xml").write_text(
        '<testsuite><testcase name="test_model_e2e[alpha-case]">'
        '<failure message="late assertion failed" /></testcase></testsuite>',
        encoding="utf-8",
    )

    rc, output, status_path = _run(tmp_path, ["alpha"])

    assert rc == 2
    assert "late assertion failed" in output.read_text(encoding="utf-8")
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert any("JUnit reconciliation" in issue for issue in payload["issues"])


def test_exclusive_gpu_proof_requires_every_slot_and_matching_ledgers(
    tmp_path: Path,
) -> None:
    root = _write_part(tmp_path / "parts", "alpha", "alpha-case")
    for filename in (
        "model-proof-status.json",
        "proof.json",
        "selection.json",
        "gpu-lease.json",
    ):
        path = root / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["gpu_resource_class"] = "exclusive_gpu"
        payload["gpu_slot_ids"] = [0, 1, 2, 3]
        if filename == "gpu-lease.json":
            payload["resource_class"] = "exclusive_gpu"
            payload["gpu_slots"] = [0, 1, 2, 3]
            payload["gpu_slot"] = None
        path.write_text(json.dumps(payload), encoding="utf-8")

    rc, _output, status_path = _run(tmp_path, ["alpha"])
    assert rc == 0
    assert json.loads(status_path.read_text(encoding="utf-8"))["outcome"] == "passed"

    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["gpu_slot_ids"] = [0, 1, 2]
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    rc, _output, status_path = _run(tmp_path, ["alpha"])
    assert rc == 2
    errors = "\n".join(json.loads(status_path.read_text(encoding="utf-8"))["issues"])
    assert "does not match test selection" in errors


def test_invalid_expected_json_still_writes_both_outputs(tmp_path: Path) -> None:
    (tmp_path / "parts").mkdir()

    rc, output, status_path = _run(tmp_path, "not-json")

    assert rc == 2
    assert output.is_file()
    assert status_path.is_file()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["outcome"] == "failed"
    assert any("expected models JSON is invalid" in issue for issue in payload["issues"])


def test_empty_expected_set_produces_a_valid_no_model_report(tmp_path: Path) -> None:
    (tmp_path / "parts").mkdir()

    rc, output, status_path = _run(tmp_path, [])

    assert rc == 0
    assert "Summary" in output.read_text(encoding="utf-8")
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["outcome"] == "passed"
    assert payload["expected_count"] == 0
    assert payload["discovered_count"] == 0
    assert payload["result_count"] == 0
    assert payload["issues"] == []


def test_upstream_failure_is_named_in_empty_model_report(tmp_path: Path) -> None:
    (tmp_path / "parts").mkdir()

    rc, output, status_path = _run(
        tmp_path,
        [],
        upstream_results=["legal=success", "impact=failure"],
    )

    assert rc == 2
    rendered = output.read_text(encoding="utf-8")
    assert "upstream job" in rendered
    assert "impact" in rendered
    assert "failure" in rendered
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["outcome"] == "failed"
    assert payload["upstream_results"] == {
        "impact": "failure",
        "legal": "success",
    }
    assert any("impact" in issue for issue in payload["issues"])


def test_invalid_or_duplicate_upstream_declarations_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "parts").mkdir()

    rc, _output, status_path = _run(
        tmp_path,
        [],
        upstream_results=["legal=success", "legal=success", "not-a-pair"],
    )

    assert rc == 2
    issues = json.loads(status_path.read_text(encoding="utf-8"))["issues"]
    assert any("duplicate upstream" in issue for issue in issues)
    assert any("invalid upstream" in issue for issue in issues)
