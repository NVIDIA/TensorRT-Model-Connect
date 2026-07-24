# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for private per-build module-residency calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tensorrt_model_connect import dynamic_memory_calibration as calibration

pytestmark = pytest.mark.dynamic_memory


def _base_contract() -> dict:
    return {
        "contract_version": 1,
        "qualified_model_id": "Qwen/Qwen3-0.6B",
        "qualified_model_revision": "1" * 40,
        "qualified_config_sha256": "2" * 64,
        "qualified_target": "gb300-trt-11.2",
        "qualified_runtime_stack": {
            "sm": "sm103",
            "tensorrt": "11.2.0.113",
            "cuda_runtime": "13.3",
            "cudnn_backend": "9.20.0",
            "cudnn_frontend_revision": "3" * 40,
            "nvrtc": "13.3",
            "driver": "580.105.08",
        },
        "native_kv_plugin_abi": 2,
        "model_context_limit": 512,
        "prefill_chunk_limit": 128,
        "kv_layout": "contiguous_runtime_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 1024,
        "active_kv_profile_limits": [128, 256, 512],
        "runtime_owned": True,
    }


def _plans() -> dict[str, bytes]:
    return {
        "engine_plan": b"fresh decode plan bytes",
        "prefill_engine_plan": b"fresh prefill plan bytes",
    }


def _capture(
    process_index: int,
    pid: int,
    process_values: tuple[int, int, int],
    *,
    gpu_uuid: str = "GPU-test",
) -> calibration._CapturedSweep:
    rows = tuple(
        {
            "profile_id": index,
            "covering_profile_limit": limit,
            "cumulative_process_first_use_bytes": process_bytes,
            "cumulative_device_wide_first_use_bytes": process_bytes + 1024,
        }
        for index, (limit, process_bytes) in enumerate(
            zip((128, 256, 512), process_values, strict=True)
        )
    )
    prefix = (
        f"{calibration.CALIBRATION_EVIDENCE_ROOT}/"
        f"process-{process_index:02d}"
    )
    raw_section = calibration.CalibrationEvidenceSection(
        f"{prefix}/runner-output.raw.json",
        b'{"passed":true}\n',
    )
    artifacts = {
        "raw_trace": {
            "section_name": raw_section.name,
            "size_bytes": len(raw_section.data),
            "sha256": hashlib.sha256(raw_section.data).hexdigest(),
        }
    }
    manifest_payload = calibration._canonical_json_bytes(
        {
            "schema": calibration.CALIBRATION_CAPTURE_SCHEMA,
            "process_index": process_index,
            "runner_pid": pid,
            "gpu_uuid": gpu_uuid,
            "sampler_trust_anchor": {
                "pid": pid,
                "cuda_logical_device_index": 0,
                "physical_device_index": 0,
                "pci_bus_id": "0000:00:00.0",
                "gpu_uuid": gpu_uuid,
            },
            "artifacts": artifacts,
        }
    )
    manifest_section = calibration.CalibrationEvidenceSection(
        f"{prefix}/capture-manifest.json",
        manifest_payload,
    )
    return calibration._CapturedSweep(
        process_index=process_index,
        child_pid=pid,
        gpu_uuid=gpu_uuid,
        trace={"passed": True},
        rows=rows,
        sampler_trust_anchor={
            "pid": pid,
            "cuda_logical_device_index": 0,
            "physical_device_index": 0,
            "pci_bus_id": "0000:00:00.0",
            "gpu_uuid": gpu_uuid,
        },
        capture_manifest_section=manifest_section.name,
        capture_manifest_sha256=hashlib.sha256(
            manifest_section.data
        ).hexdigest(),
        artifacts=artifacts,
        evidence_sections=(raw_section, manifest_section),
    )


def test_internal_calibrator_path_is_private_absolute_and_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(calibration.INTERNAL_CALIBRATOR_ENV, raising=False)
    with pytest.raises(
        calibration.AutomaticDynamicMemoryCalibrationError,
        match="native build bridge",
    ):
        calibration.resolve_internal_calibrator()

    monkeypatch.setenv(calibration.INTERNAL_CALIBRATOR_ENV, "relative-helper")
    with pytest.raises(
        calibration.AutomaticDynamicMemoryCalibrationError,
        match="absolute",
    ):
        calibration.resolve_internal_calibrator()

    helper = tmp_path / "helper"
    helper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    helper.chmod(0o644)
    monkeypatch.setenv(calibration.INTERNAL_CALIBRATOR_ENV, str(helper))
    with pytest.raises(
        calibration.AutomaticDynamicMemoryCalibrationError,
        match="executable",
    ):
        calibration.resolve_internal_calibrator()

    helper.chmod(0o755)
    assert calibration.resolve_internal_calibrator() == helper.resolve()


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ("not json\n", "invalid JSON"),
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "source": "environment",
                    "mode": "lazy",
                    "driver_value": 2,
                }
            )
            + "\n",
            "invalid CUDA",
        ),
        (
            json.dumps(
                {
                    "schema_version": 1,
                    "source": "cuModuleGetLoadingMode",
                    "mode": "lazy",
                    "driver_value": 1,
                }
            )
            + "\n",
            "invalid CUDA",
        ),
    ),
)
def test_module_loading_mode_query_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    helper = tmp_path / "helper"
    helper.write_bytes(b"helper")
    monkeypatch.setattr(
        calibration.subprocess,
        "run",
        lambda *args, **kwargs: calibration.subprocess.CompletedProcess(
            args[0],
            0,
            stdout=payload,
            stderr="",
        ),
    )
    with pytest.raises(
        calibration.AutomaticDynamicMemoryCalibrationError,
        match=message,
    ):
        calibration.query_cuda_module_loading_mode(helper)


def test_product_identity_query_accepts_exact_launcher_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper = tmp_path / "helper"
    helper.write_bytes(b"helper")
    identity = "a" * 64
    monkeypatch.setenv(
        calibration.INTERNAL_CALIBRATOR_BUILD_IDENTITY_ENV,
        identity,
    )
    monkeypatch.setattr(
        calibration.subprocess,
        "run",
        lambda *args, **kwargs: calibration.subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(
                {
                    "schema_version": 1,
                    "source": "compiled_product_identity",
                    "product_version": calibration.PACKAGE_VERSION,
                    "build_identity": identity,
                    "helper_protocol_version": 1,
                }
            )
            + "\n",
            stderr="",
        ),
    )

    assert calibration.query_product_identity(helper) == identity
    monkeypatch.delenv(
        calibration.INTERNAL_CALIBRATOR_BUILD_IDENTITY_ENV,
        raising=False,
    )
    assert calibration.query_product_identity(helper) == identity


@pytest.mark.parametrize(
    ("result_update", "launcher_identity", "message"),
    (
        ({"product_version": "9.9.9"}, None, "incompatible product"),
        ({"build_identity": "not-a-sha"}, None, "incompatible product"),
        ({"helper_protocol_version": 2}, None, "incompatible product"),
        ({}, "b" * 64, "does not match"),
        ({}, "not-a-sha", "does not match"),
    ),
)
def test_product_identity_query_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    result_update: dict[str, object],
    launcher_identity: str | None,
    message: str,
) -> None:
    helper = tmp_path / "helper"
    helper.write_bytes(b"helper")
    payload: dict[str, object] = {
        "schema_version": 1,
        "source": "compiled_product_identity",
        "product_version": calibration.PACKAGE_VERSION,
        "build_identity": "a" * 64,
        "helper_protocol_version": 1,
    }
    payload.update(result_update)
    if launcher_identity is None:
        monkeypatch.delenv(
            calibration.INTERNAL_CALIBRATOR_BUILD_IDENTITY_ENV,
            raising=False,
        )
    else:
        monkeypatch.setenv(
            calibration.INTERNAL_CALIBRATOR_BUILD_IDENTITY_ENV,
            launcher_identity,
        )
    monkeypatch.setattr(
        calibration.subprocess,
        "run",
        lambda *args, **kwargs: calibration.subprocess.CompletedProcess(
            args[0],
            0,
            stdout=json.dumps(payload) + "\n",
            stderr="",
        ),
    )

    with pytest.raises(
        calibration.AutomaticDynamicMemoryCalibrationError,
        match=message,
    ):
        calibration.query_product_identity(helper)


def test_automatic_calibration_seals_current_raw_plans_and_guarded_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper = tmp_path / "internal-calibrator"
    helper.write_bytes(b"exact helper bytes")
    helper.chmod(0o755)
    monkeypatch.setattr(
        calibration,
        "query_product_identity",
        lambda _helper: "a" * 64,
    )
    monkeypatch.setattr(
        calibration,
        "query_cuda_module_loading_mode",
        lambda _helper: "lazy",
    )
    captures = (
        _capture(0, 101, (10, 20, 70)),
        _capture(1, 202, (12, 18, 80)),
    )
    monkeypatch.setattr(
        calibration,
        "_run_sweep",
        lambda **kwargs: captures[kwargs["process_index"]],
    )
    temporary_parents: list[Path] = []

    def write_bootstrap(path: Path, contract: dict) -> None:
        assert contract["contract_version"] == 2
        assert (
            contract["module_residency_calibration"][
                "evidence_provenance"
            ]
            == "external_manifest_v1"
        )
        assert all(
            row["cumulative_reserve_bytes"] == 1
            for row in contract["module_residency_calibration"][
                "profile_reserves"
            ]
        )
        temporary_parents.append(path.parent)
        path.write_bytes(b"private bootstrap")

    result = calibration.calibrate_unknown_plan_set(
        base_contract=_base_contract(),
        plan_sections=_plans(),
        vocab_size=151_936,
        working_directory=tmp_path,
        write_bootstrap_bundle=write_bootstrap,
        helper=helper,
    )

    contract = result.runtime_memory_contract
    assert contract["contract_version"] == 2
    assert (
        contract["module_residency_calibration"]["evidence_provenance"]
        == "embedded_bundle_v1"
    )
    plans = contract["module_residency_calibration"]["plans"]
    assert plans[0]["section_sha256"] == hashlib.sha256(
        _plans()["engine_plan"]
    ).hexdigest()
    assert plans[1]["section_sha256"] == hashlib.sha256(
        _plans()["prefill_engine_plan"]
    ).hexdigest()
    assert contract["module_residency_calibration"]["profile_reserves"] == [
        {
            "covering_profile_limit": 128,
            "cumulative_reserve_bytes": (
                calibration.CALIBRATION_GUARD_BYTES + 12
            ),
        },
        {
            "covering_profile_limit": 256,
            "cumulative_reserve_bytes": (
                calibration.CALIBRATION_GUARD_BYTES + 20
            ),
        },
        {
            "covering_profile_limit": 512,
            "cumulative_reserve_bytes": (
                calibration.CALIBRATION_GUARD_BYTES + 80
            ),
        },
    ]
    assert contract["module_residency_calibration"][
        "evidence_sha256"
    ] == hashlib.sha256(result.evidence_bytes).hexdigest()
    evidence = json.loads(result.evidence_bytes)
    assert evidence["bootstrap_only"] == {
        "evidence_sha256": evidence["bootstrap_only"]["evidence_sha256"],
        "never_published": True,
        "profile_reserve_bytes": 1,
    }
    assert evidence["prompt_lengths"] == [127, 255, 511]
    assert evidence["terminal_active_length"] == 512
    assert evidence["gates"] == {
        "all_capture_sections_embedded_and_hashed": True,
        "all_profile_upper_edges_executed": True,
        "helper_identity_unchanged": True,
        "raw_plan_receipts_match_bootstrap": True,
        "second_sweep_growth_within_limit": True,
        "single_gpu_identity": True,
        "terminal_decode_reaches_model_limit": True,
        "two_distinct_processes": True,
    }
    assert result.process_ids == (101, 202)
    assert result.gpu_uuid == "GPU-test"
    assert result.evidence_sections[0] == calibration.CalibrationEvidenceSection(
        calibration.CALIBRATION_EVIDENCE_SECTION,
        result.evidence_bytes,
    )
    assert {
        section.name for section in result.evidence_sections
    } == {
        calibration.CALIBRATION_EVIDENCE_SECTION,
        *{
            f"{calibration.CALIBRATION_EVIDENCE_ROOT}/process-{index:02d}/"
            "runner-output.raw.json"
            for index in range(2)
        },
        *{
            f"{calibration.CALIBRATION_EVIDENCE_ROOT}/process-{index:02d}/"
            "capture-manifest.json"
            for index in range(2)
        },
    }
    assert temporary_parents and all(
        not path.exists() for path in temporary_parents
    )


def test_automatic_calibration_failure_never_leaves_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper = tmp_path / "internal-calibrator"
    helper.write_bytes(b"exact helper bytes")
    helper.chmod(0o755)
    monkeypatch.setattr(
        calibration,
        "query_product_identity",
        lambda _helper: "a" * 64,
    )
    monkeypatch.setattr(
        calibration,
        "query_cuda_module_loading_mode",
        lambda _helper: "lazy",
    )
    first = _capture(0, 101, (10, 20, 70))

    def run_sweep(**kwargs):
        if kwargs["process_index"] == 0:
            return first
        raise calibration.AutomaticDynamicMemoryCalibrationError(
            "second process failed"
        )

    monkeypatch.setattr(calibration, "_run_sweep", run_sweep)
    temporary_parents: list[Path] = []

    def write_bootstrap(path: Path, _contract: dict) -> None:
        temporary_parents.append(path.parent)
        path.write_bytes(b"private bootstrap")

    with pytest.raises(
        calibration.AutomaticDynamicMemoryCalibrationError,
        match="second process failed",
    ):
        calibration.calibrate_unknown_plan_set(
            base_contract=_base_contract(),
            plan_sections=_plans(),
            vocab_size=151_936,
            working_directory=tmp_path,
            write_bootstrap_bundle=write_bootstrap,
            helper=helper,
        )
    assert temporary_parents and all(
        not path.exists() for path in temporary_parents
    )
