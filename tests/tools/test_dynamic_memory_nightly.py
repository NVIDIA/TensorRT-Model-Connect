# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "ci" / "dynamic_memory_nightly.py"
SPEC = importlib.util.spec_from_file_location(
    "dynamic_memory_nightly",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
nightly = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = nightly
SPEC.loader.exec_module(nightly)

pytestmark = [pytest.mark.unit, pytest.mark.dynamic_memory]
TESTED_SHA = "a" * 40
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "b" * 40


def _fixture() -> dict:
    return nightly.load_fixture(nightly.DEFAULT_FIXTURE)


def test_base_environment_clears_private_calibrator_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR",
        "/stale/helper",
    )
    monkeypatch.setenv(
        "_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR_BUILD_IDENTITY",
        "a" * 64,
    )
    environment = nightly._base_environment(
        repo_root=REPO_ROOT,
        build_dir=tmp_path / "build",
        output_dir=tmp_path / "output",
        producer_gpu="0",
    )
    assert "_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR" not in environment
    assert (
        "_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR_BUILD_IDENTITY"
        not in environment
    )


def _manifest_payload() -> dict:
    commands = []
    for label in nightly._TEST_MANIFEST_COMMAND_LABELS:
        command = {
            "label": label,
            "passed": True,
            "returncode": 0,
        }
        if label == "ctest_manifest_dynamic_memory":
            command["manifest_entries"] = sorted(
                nightly._NATIVE_COMPATIBILITY_CTESTS
            )
        commands.append(command)
    return {
        "source_state_unchanged": True,
        "commands": commands,
    }


def _correctness_payload() -> dict:
    gates = {
        gate: True for gate in nightly._CORRECTNESS_PROMOTION_GATES
    }
    evidence = {
        "status": "passed",
        "passed": True,
        "base": True,
        "chunk_variant": True,
    }
    source_binding = {
        "source": "embedded_bundle_sections",
        "evidence_schema": (
            "trtmc.native-dynamic-memory-build-calibration-evidence/v2"
        ),
        "bundle": {
            "path": "/bundle.trtfb",
            "size_bytes": 1,
            "sha256": "1" * 64,
        },
        "evidence_section": {
            "section_name": "runtime_memory_calibration/evidence.json",
            "size_bytes": 1,
            "sha256": "2" * 64,
        },
        "capture_manifests": [
            {
                "section_name": (
                    f"runtime_memory_calibration/process-{index:02d}/"
                    "capture-manifest.json"
                ),
                "size_bytes": 1,
                "sha256": "3" * 64,
            }
            for index in range(2)
        ],
        "raw_captures": [
            {
                "section_name": (
                    f"runtime_memory_calibration/process-{index:02d}/"
                    "runner-output.raw.json"
                ),
                "size_bytes": 1,
                "sha256": "4" * 64,
            }
            for index in range(2)
        ],
        "logits": [
            {
                "section_name": (
                    f"runtime_memory_calibration/process-{index:02d}/"
                    "runner-logits.bin"
                ),
                "size_bytes": 1,
                "sha256": "5" * 64,
            }
            for index in range(2)
        ],
        "runner_sha256": "6" * 64,
        "contract_provenance": {
            "qualified_runtime_stack_sha256": "7" * 64,
            "plan_set_sha256": "8" * 64,
            "cuda_module_loading_mode": "lazy",
            "plans": [
                {
                    "section_name": "engine_plan",
                    "section_sha256": "9" * 64,
                },
                {
                    "section_name": "prefill_engine_plan",
                    "section_sha256": "a" * 64,
                },
            ],
        },
        "recommended_profile_reserves": [
            {
                "covering_profile_limit": 2048,
                "cumulative_reserve_bytes": 64 * 1024 * 1024,
            }
        ],
        "bootstrap_cycle_exemption": {
            "field": "module_residency_evidence_sha256",
            "reason": "test",
            "observed_bootstrap_values": ["b" * 64],
            "final_sealed_value": "c" * 64,
            "all_other_receipt_provenance_replayed": True,
        },
        "passed": True,
    }
    return {
        "model_id": MODEL_ID,
        "qualification_gates": gates,
        "source_calibration_evidence": dict(evidence),
        "all_profile_two_sweep_evidence": dict(evidence),
        "module_residency_profile_sweeps": {
            "base": {
                "source_calibration_evidence": copy.deepcopy(source_binding),
            },
            "chunk_variant": {
                "source_calibration_evidence": copy.deepcopy(source_binding),
            },
        },
    }


@pytest.mark.parametrize("model_key", nightly.EXPECTED_MODELS)
def test_plan_is_the_complete_two_bundle_promotion_graph(
    model_key: str,
) -> None:
    fixture = _fixture()
    plan = nightly.create_plan(
        repo_root=REPO_ROOT,
        build_dir=REPO_ROOT / "build-dynkv",
        python=Path(sys.executable),
        output_dir=REPO_ROOT / "artifacts" / "nightly-unit" / model_key,
        fixture=fixture,
        model_key=model_key,
        model_snapshot=Path(f"/cache/{model_key}/snapshot"),
        tested_sha=TESTED_SHA,
        producer_gpu="0",
        runner_gpu="1",
        isolation_gpu_a="2",
        isolation_gpu_b="3",
    )

    assert [command.label for command in plan.commands] == list(
        nightly._RECEIPT_CONTRACTS
    )
    assert plan.model_id == fixture["models"][model_key]["model_id"]
    assert plan.model_revision == fixture["models"][model_key]["revision"]

    commands = {command.label: command for command in plan.commands}
    product_argv = list(commands["dynamic-build"].argv)
    delimiter = product_argv.index("--")
    assert product_argv[delimiter + 1 :][-2:] == [
        "build",
        plan.model_id,
    ]
    assert len(product_argv[delimiter + 1 :]) == 3

    variant = commands["chunk-variant-build"]
    assert dict(variant.environment) == {
        "_TRTMC_INTERNAL_DYNAMIC_MEMORY_CALIBRATOR": str(
            REPO_ROOT / "build-dynkv" / "trtmc_dynamic_memory_qualify"
        ),
        "TRTMC_DEVELOPER_CHUNK_VARIANT": "C/2",
    }
    correctness = list(commands["correctness"].argv)
    assert not any("calibration-source" in argument for argument in correctness)


def test_manifest_receipt_requires_native_compatibility_tests() -> None:
    payload = _manifest_payload()
    assert nightly._test_manifest_errors(payload) == []

    tampered = copy.deepcopy(payload)
    dynamic = next(
        command
        for command in tampered["commands"]
        if command["label"] == "ctest_manifest_dynamic_memory"
    )
    dynamic["manifest_entries"].remove("test_abi_old_consumer")
    errors = nightly._test_manifest_errors(tampered)
    assert any("native compatibility coverage is missing" in error for error in errors)


def test_correctness_receipt_requires_both_source_calibration_trees() -> None:
    payload = _correctness_payload()
    assert nightly._receipt_semantic_errors(
        "correctness",
        payload,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
    ) == []

    payload["source_calibration_evidence"]["chunk_variant"] = False
    payload["qualification_gates"][
        "source_calibration_evidence_reopened"
    ] = False
    errors = nightly._receipt_semantic_errors(
        "correctness",
        payload,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
    )
    assert any(
        "source_calibration_evidence_reopened is not true" in error
        for error in errors
    )
    assert any(
        "source_calibration_evidence does not prove base and C/2" in error
        for error in errors
    )


def test_correctness_receipt_requires_embedded_bundle_source_replay() -> None:
    payload = _correctness_payload()
    assert nightly._receipt_semantic_errors(
        "correctness",
        payload,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
    ) == []

    source = payload["module_residency_profile_sweeps"]["chunk_variant"][
        "source_calibration_evidence"
    ]
    source["source"] = "external_bootstrap_files"
    source["evidence_schema"] = (
        "trtmc.native-dynamic-memory-build-calibration-evidence/v1"
    )
    source["raw_captures"].pop()
    errors = nightly._receipt_semantic_errors(
        "correctness",
        payload,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
    )
    assert any(
        ".chunk_variant.source_calibration_evidence.source is not "
        "'embedded_bundle_sections'" in error
        for error in errors
    )
    assert any(
        ".chunk_variant.source_calibration_evidence.raw_captures does not "
        "contain two captures" in error
        for error in errors
    )
    assert any(
        ".chunk_variant.source_calibration_evidence.evidence_schema is not"
        in error
        for error in errors
    )


def test_correctness_primary_receipt_is_source_and_promotion_bound(
    tmp_path: Path,
) -> None:
    payload = {
        **_correctness_payload(),
        "schema_version": 1,
        "status": "passed",
        "passed": True,
        "promotion_eligible": True,
        "source_state": {"git_head": TESTED_SHA},
    }
    path = tmp_path / "qualification-report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    expectation = nightly.ReceiptExpectation(
        label="correctness",
        path=path,
        schema=1,
        status="passed",
        passed=True,
        promotion_eligible=True,
    )
    entry, errors = nightly.inspect_receipt(
        expectation,
        tested_sha=TESTED_SHA,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
    )
    assert errors == []
    assert entry["validation"]["passed"] is True

    payload["promotion_eligible"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    _, errors = nightly.inspect_receipt(
        expectation,
        tested_sha=TESTED_SHA,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
    )
    assert any("promotion_eligible=False" in error for error in errors)


def test_performance_and_isolation_nested_gates_fail_closed() -> None:
    performance = {
        "gates": {
            "performance": {
                "short": {
                    "decode_throughput_gte_95_percent_static": True,
                },
                "medium": {
                    "prefill_proxy_regression_lte_10_percent": True,
                },
            },
            "packaging": {"bundle_bytes_lte_105_percent_static": True},
        }
    }
    assert nightly._receipt_semantic_errors(
        "performance-qualification",
        performance,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
    ) == []
    performance["gates"]["performance"]["medium"][
        "prefill_proxy_regression_lte_10_percent"
    ] = False
    assert any(
        "prefill_proxy_regression_lte_10_percent is not true" in error
        for error in nightly._receipt_semantic_errors(
            "performance-qualification",
            performance,
            expected_model_id=MODEL_ID,
            expected_model_revision=MODEL_REVISION,
        )
    )

    isolation = {
        "gates": {
            "all_four_executions_recorded": True,
            "concurrent_engine_load_intervals_overlap": False,
        },
        "source_state_unchanged": True,
    }
    assert any(
        "concurrent_engine_load_intervals_overlap is not true" in error
        for error in nightly._receipt_semantic_errors(
            "process-isolation",
            isolation,
            expected_model_id=MODEL_ID,
            expected_model_revision=MODEL_REVISION,
        )
    )


def test_dynamic_capture_requires_contract_v2_receipt_v4() -> None:
    payload = {
        "model_id": MODEL_ID,
        "artifact_role": "native-dynamic",
        "runtime_memory_receipt": {
            "receipt_schema_version": 4,
            "contract_version": 2,
        },
    }
    assert nightly._receipt_semantic_errors(
        "performance-capture-dynamic-short",
        payload,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
    ) == []
    payload["runtime_memory_receipt"]["receipt_schema_version"] = 3
    assert any(
        "receipt v4/contract v2 is absent" in error
        for error in nightly._receipt_semantic_errors(
            "performance-capture-dynamic-short",
            payload,
            expected_model_id=MODEL_ID,
            expected_model_revision=MODEL_REVISION,
        )
    )
    static_payload = {
        "model_id": MODEL_ID,
        "artifact_role": "exact-head-static-split",
        "runtime_memory_receipt": {
            "serialized_plan_bytes": 1,
            "resident_weight_bytes": 2,
            "resident_weight_copy_count": 1,
            "weight_streaming_active": False,
            "measurement_sources": {},
        },
    }
    assert nightly._receipt_semantic_errors(
        "performance-capture-static-short",
        static_payload,
        expected_model_id=MODEL_ID,
        expected_model_revision=MODEL_REVISION,
    ) == []


def test_artifact_selection_ignores_failed_older_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parts = tmp_path / "parts"
    for model_key in nightly.EXPECTED_MODELS:
        for attempt in (1, 2):
            path = (
                parts
                / f"{model_key}-{attempt}"
                / "dynamic-memory-nightly-gate.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "model_key": model_key,
                        "tested_sha": TESTED_SHA,
                        "workflow_run_id": "17",
                        "workflow_run_attempt": attempt,
                    }
                ),
                encoding="utf-8",
            )

    verified: list[tuple[str, int]] = []

    def fake_verify(path: Path, **_: object) -> dict:
        gate = json.loads(path.read_text(encoding="utf-8"))
        assert gate["workflow_run_attempt"] == 2
        verified.append((gate["model_key"], gate["workflow_run_attempt"]))
        return {
            "model_key": gate["model_key"],
            "model_id": gate["model_key"],
            "workflow_run_attempt": gate["workflow_run_attempt"],
            "aggregate": {"path": str(path)},
            "receipt_count": len(nightly._RECEIPT_CONTRACTS),
        }

    monkeypatch.setattr(nightly, "_verify_gate", fake_verify)
    output = tmp_path / "status.json"
    args = argparse.Namespace(
        parts_dir=parts,
        output=output,
        fixture=nightly.DEFAULT_FIXTURE,
        expected_tested_sha=TESTED_SHA,
        expected_run_id="17",
        max_attempt=2,
    )
    assert nightly.verify_artifacts(args) == 0
    assert verified == [("qwen", 2), ("tinyllama", 2)]
    status = json.loads(output.read_text(encoding="utf-8"))
    assert status["passed"] is True
    assert status["selected_models"] == ["qwen", "tinyllama"]


def test_duplicate_latest_artifact_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parts = tmp_path / "parts"
    for model_key in nightly.EXPECTED_MODELS:
        copies = 2 if model_key == "qwen" else 1
        for index in range(copies):
            path = (
                parts
                / f"{model_key}-{index}"
                / "dynamic-memory-nightly-gate.json"
            )
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "model_key": model_key,
                        "tested_sha": TESTED_SHA,
                        "workflow_run_id": "18",
                        "workflow_run_attempt": 3,
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(
        nightly,
        "_verify_gate",
        lambda path, **_: {
            "model_key": "tinyllama",
            "model_id": "tinyllama",
            "workflow_run_attempt": 3,
            "aggregate": {"path": str(path)},
            "receipt_count": len(nightly._RECEIPT_CONTRACTS),
        },
    )
    output = tmp_path / "status.json"
    args = argparse.Namespace(
        parts_dir=parts,
        output=output,
        fixture=nightly.DEFAULT_FIXTURE,
        expected_tested_sha=TESTED_SHA,
        expected_run_id="18",
        max_attempt=3,
    )
    assert nightly.verify_artifacts(args) == 1
    status = json.loads(output.read_text(encoding="utf-8"))
    assert status["passed"] is False
    assert any("duplicate dynamic-memory artifacts" in error for error in status["errors"])
