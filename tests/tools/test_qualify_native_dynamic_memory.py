# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import hashlib
import json
import math
import os
import stat
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.tools.dynamic_memory_manifest_fixture import (
    complete_command_receipts,
    load_manifest_module,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "qualify_native_dynamic_memory.py"
SPEC = importlib.util.spec_from_file_location("qualify_native_dynamic_memory", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
qualify = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualify
SPEC.loader.exec_module(qualify)

pytestmark = pytest.mark.dynamic_memory

TEST_SPEC = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
TEST_GEOMETRY = qualify.trusted_runtime_geometry(TEST_SPEC)
TEST_SAMPLER = qualify.SamplerTrustAnchor(
    pid=123,
    cuda_logical_device_index=0,
    physical_device_index=7,
    pci_bus_id="0000:01:00.0",
    gpu_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef",
)
_MODULE_RESIDENCY_PLAN_SET_SHA256 = "a" * 64
_MODULE_RESIDENCY_EVIDENCE_SHA256 = "b" * 64


def _module_residency_receipt_fields(
    *,
    reserve_bytes: int,
    profile_limit: int,
) -> dict[str, object]:
    return {
        "module_residency_reserve_bytes": reserve_bytes,
        "module_residency_reserve_profile_limit": profile_limit,
        "module_residency_plan_set_sha256": (
            _MODULE_RESIDENCY_PLAN_SET_SHA256
        ),
        "module_residency_evidence_sha256": (
            _MODULE_RESIDENCY_EVIDENCE_SHA256
        ),
        "module_residency_cuda_module_loading_mode": "lazy",
    }


def _validate_warmup_evidence(
    trace: dict,
    **kwargs: object,
) -> dict:
    return qualify.validate_warmup_evidence(
        trace,
        trusted_geometry=TEST_GEOMETRY,
        expected_sampler=TEST_SAMPLER,
        **kwargs,
    )


def _reconcile_device_peak_with_nvml(trace: dict) -> dict:
    return qualify.reconcile_device_peak_with_nvml(
        trace,
        trusted_geometry=TEST_GEOMETRY,
        expected_sampler=TEST_SAMPLER,
    )


def _persisted_case_warmup_evidence_passed(
    evidence: object,
    *,
    trace: object,
    case: qualify.Case,
    **kwargs: object,
) -> bool:
    return qualify._persisted_case_warmup_evidence_passed(
        evidence,
        trace=trace,
        case=case,
        trusted_geometry=TEST_GEOMETRY,
        expected_sampler=TEST_SAMPLER,
        **kwargs,
    )


def _write_test_runner_capture(
    evidence_dir: Path,
    *,
    command: list[str],
    tokens: np.ndarray,
    trace: dict,
    returncode: int,
    sampler: qualify.SamplerTrustAnchor,
    include_logits: bool,
) -> None:
    (evidence_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n",
        encoding="utf-8",
    )
    qualify._write_tokens(evidence_dir / "tokens.txt", tokens)
    (evidence_dir / "runner.stdout.log").write_text(
        json.dumps(trace, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "runner.stderr.log").write_text("", encoding="utf-8")
    (evidence_dir / "returncode.txt").write_text(
        f"{returncode}\n",
        encoding="utf-8",
    )
    (evidence_dir / "runner-trace.json").write_text(
        json.dumps(trace, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    qualify._write_runner_capture_manifest(
        evidence_dir,
        child_pid=sampler.pid,
        sampler_anchor=sampler,
        include_logits=include_logits,
    )


@pytest.mark.parametrize(
    "selector",
    (
        "0",
        "17",
        "GPU-01234567-89ab-cdef-0123-456789abcdef",
    ),
)
def test_runner_cuda_visible_device_accepts_one_physical_selector(
    selector: str,
) -> None:
    assert qualify._validate_runner_cuda_visible_device(selector) == selector


@pytest.mark.parametrize(
    "selector",
    (
        "",
        " 3",
        "3 ",
        "-1",
        "cuda:3",
        "3,4",
        "GPU-01234567",
        "GPU-01234567-89ab-cdef-0123-456789abcdeg",
    ),
)
def test_runner_cuda_visible_device_rejects_invalid_or_multiple_selectors(
    selector: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="one numeric physical GPU index or one full GPU UUID",
    ):
        qualify._validate_runner_cuda_visible_device(selector)


def test_cli_requires_runner_cuda_visible_device(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--bundle",
            str(tmp_path / "missing.trtfb"),
            "--model",
            TEST_SPEC.model_id,
            "--output-dir",
            str(tmp_path / "output"),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 2
    assert "--runner-cuda-visible-device" in completed.stderr
    assert "required" in completed.stderr


def test_cli_rejects_multiple_runner_cuda_visible_devices(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--bundle",
            str(tmp_path / "missing.trtfb"),
            "--model",
            TEST_SPEC.model_id,
            "--output-dir",
            str(tmp_path / "output"),
            "--runner-cuda-visible-device",
            "2,3",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 2
    assert (
        "one numeric physical GPU index or one full GPU UUID"
        in completed.stderr
    )


def test_cli_accepts_omitted_external_source_calibration_inputs(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--bundle",
            str(tmp_path / "missing.trtfb"),
            "--model",
            TEST_SPEC.model_id,
            "--output-dir",
            str(tmp_path / "output"),
            "--runner-cuda-visible-device",
            "0",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 2
    assert "--base-calibration-source-evidence" not in completed.stderr
    assert "--base-calibration-source-raw-capture" not in completed.stderr


def test_cli_requires_exactly_two_base_source_raw_captures(
    tmp_path: Path,
) -> None:
    bundle, _ = _write_base_bundle(tmp_path / "sealed.trtfb")
    source = tmp_path / "source.json"
    raw = tmp_path / "process-00.raw.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--bundle",
            str(bundle),
            "--model",
            TEST_SPEC.model_id,
            "--output-dir",
            str(tmp_path / "output"),
            "--runner-cuda-visible-device",
            "0",
            "--base-calibration-source-evidence",
            str(source),
            "--base-calibration-source-raw-capture",
            str(raw),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
    assert (
        "--base-calibration-source-raw-capture must be provided exactly twice"
        in completed.stderr
    )


def test_embedded_base_and_chunk_variant_need_no_external_source_inputs() -> None:
    embedded_header = {
        "sections": {
            qualify.EMBEDDED_CALIBRATION_EVIDENCE_SECTION: {
                "offset": 0,
                "size": 1,
            }
        }
    }
    for label in ("base", "chunk variant"):
        qualify._validate_calibration_source_inputs(
            label=label,
            header=embedded_header,
            evidence_path=None,
            raw_paths=(),
        )


def test_cli_rejects_unpaired_chunk_variant_source_inputs(
    tmp_path: Path,
) -> None:
    bundle, _ = _write_base_bundle(tmp_path / "sealed.trtfb")
    base_source = tmp_path / "base-source.json"
    raw0 = tmp_path / "base-process-00.raw.json"
    raw1 = tmp_path / "base-process-01.raw.json"
    variant_source = tmp_path / "variant-source.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--bundle",
            str(bundle),
            "--model",
            TEST_SPEC.model_id,
            "--output-dir",
            str(tmp_path / "output"),
            "--runner-cuda-visible-device",
            "0",
            "--base-calibration-source-evidence",
            str(base_source),
            "--base-calibration-source-raw-capture",
            str(raw0),
            "--base-calibration-source-raw-capture",
            str(raw1),
            "--chunk-variant-calibration-source-evidence",
            str(variant_source),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
    assert (
        "chunk-variant calibration source inputs require "
        "--chunk-variant-bundle"
        in completed.stderr
    )


def test_runner_child_environment_overrides_only_the_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    child_environment = qualify._runner_child_environment("3")

    completed, child_pid = qualify._run_captured_command(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "print(os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))"
            ),
        ],
        environment=child_environment,
    )

    assert child_pid > 0
    assert completed.returncode == 0
    assert completed.stdout.strip() == "3"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def _profile_sweep_test_contract() -> dict:
    return {
        "contract_version": 2,
        "qualified_model_id": TEST_SPEC.model_id,
        "model_context_limit": TEST_SPEC.context_limit,
        "prefill_chunk_limit": TEST_SPEC.chunk_limit,
        "kv_bytes_per_token": TEST_SPEC.kv_bytes_per_token,
        "active_kv_profile_limits": list(TEST_SPEC.buckets),
        "module_residency_calibration": {
            "cuda_module_loading_mode": "lazy",
            "qualified_runtime_stack_sha256": "c" * 64,
            "plan_set_sha256": _MODULE_RESIDENCY_PLAN_SET_SHA256,
            "evidence_sha256": _MODULE_RESIDENCY_EVIDENCE_SHA256,
            "plans": [
                {
                    "section_name": "engine_plan",
                    "section_sha256": "d" * 64,
                    "role": "decode",
                    "optimization_profile_count": len(TEST_SPEC.buckets),
                },
                {
                    "section_name": "prefill_engine_plan",
                    "section_sha256": "e" * 64,
                    "role": "prefill",
                    "optimization_profile_count": 1,
                },
            ],
            "profile_reserves": [
                {
                    "covering_profile_limit": limit,
                    "cumulative_reserve_bytes": (
                        (index + 1) * 256 * 1024 * 1024
                    ),
                }
                for index, limit in enumerate(TEST_SPEC.buckets)
            ],
        },
    }


def _profile_sweep_test_trace(
    tmp_path: Path,
) -> tuple[dict, tuple[Path, ...], tuple[int, ...], Path]:
    contract = _profile_sweep_test_contract()
    prompt_lengths = qualify._profile_sweep_prompt_lengths(
        TEST_SPEC.buckets,
        model_context_limit=TEST_SPEC.context_limit,
    )
    token_paths: list[Path] = []
    for index, length in enumerate(prompt_lengths):
        token_path = tmp_path / f"profile-{index:02d}.tokens.txt"
        qualify._write_tokens(
            token_path,
            qualify.deterministic_token_ids(length, TEST_SPEC.vocab_size),
        )
        token_paths.append(token_path)
    logits_path = tmp_path / "runner-logits.bin"
    logits = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    logits_path.write_bytes(
        qualify.LOGITS_HEADER.pack(
            qualify.LOGITS_MAGIC,
            1,
            1,
            logits.shape[0],
            logits.shape[1],
        )
        + logits.astype("<f4").tobytes()
    )
    terminal = contract["module_residency_calibration"][
        "profile_reserves"
    ][-1]
    receipt = {
        "receipt_schema_version": 4,
        "contract_version": 2,
        "policy": "auto",
        "request_context_limit": TEST_SPEC.context_limit,
        "model_context_limit": TEST_SPEC.context_limit,
        "prefill_chunk_limit": TEST_SPEC.chunk_limit,
        "runtime_kv_capacity_tokens": TEST_SPEC.context_limit,
        "effective_request_limit": TEST_SPEC.context_limit,
        "kv_bytes_per_token": TEST_SPEC.kv_bytes_per_token,
        "kv_allocation_id": 77,
        "module_residency_reserve_bytes": terminal[
            "cumulative_reserve_bytes"
        ],
        "module_residency_reserve_profile_limit": terminal[
            "covering_profile_limit"
        ],
        "module_residency_plan_set_sha256": (
            _MODULE_RESIDENCY_PLAN_SET_SHA256
        ),
        "module_residency_evidence_sha256": (
            _MODULE_RESIDENCY_EVIDENCE_SHA256
        ),
        "module_residency_cuda_module_loading_mode": "lazy",
    }
    rows_a = []
    rows_b = []
    reserve_rows = []
    previous = 0
    for index, (token_path, prompt_tokens, limit) in enumerate(
        zip(token_paths, prompt_lengths, TEST_SPEC.buckets, strict=True)
    ):
        common = {
            "row_index": index,
            "token_file": str(token_path.resolve()),
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": 1,
            "expected_profile_id": index,
            "expected_profile_limit": limit,
            "expected_prompt_lower_exclusive": previous,
            "profile_match": True,
            "observed_decode_profile_ids": [index],
            "selected_token_ids": [index + 1],
            "step_top1_token_ids": [index + 1],
            "kv_allocation_id": 77,
            "runtime_memory_receipt": copy.deepcopy(receipt),
            "invocation_tuples": [["decode", index]],
            "invocations": [],
            "after_request": {},
        }
        process_bytes = (index + 1) * 100
        device_bytes = (index + 1) * 200
        rows_a.append(
            {
                **common,
                "cumulative_first_use_growth": {
                    "process_growth_bytes": process_bytes,
                    "device_wide_growth_bytes": device_bytes,
                    "cumulative_process_high_water_bytes": process_bytes,
                    "cumulative_device_wide_high_water_bytes": device_bytes,
                },
            }
        )
        rows_b.append(
            {
                **copy.deepcopy(common),
                "incremental_growth_from_sweep_b_baseline": {
                    "process_growth_bytes": 0,
                    "device_wide_growth_bytes": 0,
                    "cumulative_process_high_water_bytes": 0,
                    "cumulative_device_wide_high_water_bytes": 0,
                },
                "equivalence_to_sweep_a": {
                    "selected_token_ids_bitwise_equal": True,
                    "complete_float32_logits_bitwise_equal": True,
                    "invocation_tuples_equal": True,
                    "passed": True,
                },
            }
        )
        reserve_rows.append(
            {
                "profile_id": index,
                "covering_profile_limit": limit,
                "cumulative_process_first_use_bytes": process_bytes,
                "cumulative_device_wide_first_use_bytes": device_bytes,
            }
        )
        previous = limit
    trace = {
        "schema_version": 1,
        "mode": "all_profile_two_sweep",
        "status": "ok",
        "error_type": None,
        "passed": True,
        "qualification_blockers": [],
        "qualification_api_version": 1,
        "model_id": TEST_SPEC.model_id,
        "pipeline_type": "LlamaTextGenerationPipeline",
        "cuda_module_loading": {
            "source": "cuModuleGetLoadingMode",
            "mode": "lazy",
            "driver_value": 2,
        },
        "protocol": {
            "schema_version": 1,
            "execution_order": ["sweep_a", "sweep_b"],
            "pipeline_load_count": 1,
            "kv_allocation_count": 1,
            "max_new_tokens_per_request": 1,
            "profile_order": "ascending",
            "second_sweep_process_growth_limit_bytes": 64 * 1024 * 1024,
        },
        "input_manifest": [
            {
                "row_index": index,
                "token_file": str(token_path.resolve()),
                "prompt_tokens": prompt_tokens,
                "expected_profile_id": index,
                "expected_profile_limit": limit,
                "expected_prompt_lower_exclusive": (
                    0 if index == 0 else TEST_SPEC.buckets[index - 1]
                ),
            }
            for index, (token_path, prompt_tokens, limit) in enumerate(
                zip(token_paths, prompt_lengths, TEST_SPEC.buckets, strict=True)
            )
        ],
        "stable_kv_allocation_id": 77,
        "profile_reserve_rows": reserve_rows,
        "sweep_a": {
            "rows": rows_a,
            "cumulative_process_first_use_high_water_bytes": 400,
            "cumulative_device_wide_first_use_high_water_bytes": 800,
        },
        "sweep_b": {
            "rows": rows_b,
            "incremental_process_high_water_bytes": 0,
            "incremental_device_wide_high_water_bytes": 0,
            "process_growth_limit_bytes": 64 * 1024 * 1024,
            "process_growth_within_limit": True,
        },
        "gates": {
            "stable_kv_allocation_id": True,
            "sweep_a_exact_profile_coverage": True,
            "sweep_b_exact_profile_coverage": True,
            "selected_token_ids_bitwise_equivalent": True,
            "complete_float32_logits_bitwise_equivalent": True,
            "invocation_tuples_equivalent": True,
            "sweep_b_process_incremental_high_water_within_limit": True,
        },
        "memory_sampler": {
            "source": "nvmlDeviceGetComputeRunningProcesses_v3",
            "pid": TEST_SAMPLER.pid,
            "cuda_logical_device_index": 0,
            "physical_device_index": TEST_SAMPLER.physical_device_index,
            "pci_bus_id": TEST_SAMPLER.pci_bus_id,
            "gpu_uuid": TEST_SAMPLER.gpu_uuid,
            "captures_all_compute_processes": True,
            "device_memory_source": "nvmlDeviceGetMemoryInfo_v2",
        },
        "logits_artifact": {
            "format": "trtmc-qualification-logits-v1",
            "dtype": "float32",
            "rows": logits.shape[0],
            "vocab_size": logits.shape[1],
            "path": str(logits_path.resolve()),
        },
    }
    return trace, tuple(token_paths), prompt_lengths, logits_path


def _source_calibration_fixture(
    tmp_path: Path,
    *,
    receipt_plan_tamper: bool = False,
) -> tuple[Path, Path, tuple[Path, Path], Path, dict]:
    runner = tmp_path / "qualifier-runner"
    runner.write_bytes(b"manifest-bound-qualifier")
    decode_plan = b"bootstrap-decode-plan"
    prefill_plan = b"bootstrap-prefill-plan"
    runtime_stack = {
        "sm": "sm103",
        "tensorrt": "11.2.0.113",
        "cuda_runtime": "13.3",
        "cudnn_backend": "9.20.0",
        "cudnn_frontend_revision": (
            "7b9b711c22b6823e87150213ecd8449260db8610"
        ),
        "nvrtc": "13.3",
        "driver": "580.105.08",
    }
    plans = [
        {
            "section_name": "engine_plan",
            "section_sha256": hashlib.sha256(
                decode_plan
            ).hexdigest(),
            "role": "decode",
            "optimization_profile_count": len(TEST_SPEC.buckets),
        },
        {
            "section_name": "prefill_engine_plan",
            "section_sha256": hashlib.sha256(
                prefill_plan
            ).hexdigest(),
            "role": "prefill",
            "optimization_profile_count": 1,
        },
    ]
    contract = _profile_sweep_test_contract()
    contract.update(
        {
            "qualified_model_id": TEST_SPEC.model_id,
            "qualified_runtime_stack": runtime_stack,
        }
    )
    calibration = contract["module_residency_calibration"]
    calibration.update(
        {
            "qualified_runtime_stack_sha256": (
                qualify.qualified_runtime_stack_sha256(runtime_stack)
            ),
            "plans": plans,
            "plan_set_sha256": (
                qualify.module_residency_plan_set_sha256(plans)
            ),
            "evidence_sha256": "0" * 64,
        }
    )
    source_bundle = tmp_path / "bootstrap-source.trtfb"
    sections = {
        "engine_plan": {"offset": 0, "size": len(decode_plan)},
        "prefill_engine_plan": {
            "offset": len(decode_plan),
            "size": len(prefill_plan),
        },
    }
    source_header = {
        "model_id": TEST_SPEC.model_id,
        "vocab_size": TEST_SPEC.vocab_size,
        "runtime_memory": contract,
        "sections": sections,
    }
    raw_header = json.dumps(source_header).encode("utf-8")
    source_bundle.write_bytes(
        qualify.BUNDLE_MAGIC
        + struct.pack("<Q", len(raw_header))
        + raw_header
        + decode_plan
        + prefill_plan
    )

    prompt_lengths = qualify._profile_sweep_prompt_lengths(
        TEST_SPEC.buckets,
        model_context_limit=TEST_SPEC.context_limit,
    )
    per_process_rows: list[list[dict]] = []
    raw_paths: list[Path] = []
    runs: list[dict] = []
    for process_index in range(qualify.PROFILE_SWEEP_PROCESS_COUNT):
        process_dir = tmp_path / f"process-{process_index:02d}"
        process_dir.mkdir()
        trace, token_paths, _, logits_path = _profile_sweep_test_trace(
            process_dir
        )
        pid = 700 + process_index
        trace["memory_sampler"]["pid"] = pid
        for sweep_name in ("sweep_a", "sweep_b"):
            for row in trace[sweep_name]["rows"]:
                receipt = row["runtime_memory_receipt"]
                receipt["module_residency_plan_set_sha256"] = (
                    "f" * 64
                    if receipt_plan_tamper
                    else calibration["plan_set_sha256"]
                )
                receipt["module_residency_evidence_sha256"] = "0" * 64
        command = qualify._profile_sweep_command(
            runner=runner.resolve(),
            bundle=source_bundle.resolve(),
            token_paths=token_paths,
            logits_path=logits_path,
            model_context_limit=TEST_SPEC.context_limit,
        )
        (process_dir / "command.json").write_text(
            json.dumps(command) + "\n",
            encoding="utf-8",
        )
        raw_trace = json.dumps(trace, sort_keys=True) + "\n"
        (process_dir / "runner.stdout.log").write_text(
            raw_trace,
            encoding="utf-8",
        )
        (process_dir / "runner.stderr.log").write_text(
            "",
            encoding="utf-8",
        )
        (process_dir / "returncode.txt").write_text(
            "0\n",
            encoding="utf-8",
        )
        raw_path = process_dir / "runner-output.raw.json"
        raw_path.write_text(raw_trace, encoding="utf-8")
        (process_dir / "runner-trace.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        anchor = qualify.SamplerTrustAnchor(
            pid=pid,
            cuda_logical_device_index=0,
            physical_device_index=TEST_SAMPLER.physical_device_index,
            pci_bus_id=TEST_SAMPLER.pci_bus_id,
            gpu_uuid=TEST_SAMPLER.gpu_uuid,
        )
        qualify._write_profile_sweep_capture_manifest(
            process_dir,
            child_pid=pid,
            sampler_anchor=anchor,
            token_paths=token_paths,
        )
        capture_manifest = process_dir / "capture-manifest.json"
        runs.append(
            {
                "process_index": process_index,
                "runner_pid": pid,
                "gpu_uuid": TEST_SAMPLER.gpu_uuid,
                "capture_manifest_sha256": qualify._sha256(
                    capture_manifest
                ),
                "raw_trace_sha256": qualify._sha256(raw_path),
                "logits_sha256": qualify._sha256(logits_path),
            }
        )
        per_process_rows.append(trace["profile_reserve_rows"])
        raw_paths.append(raw_path)
    aggregate_rows = qualify.aggregate_profile_sweep_calibration_rows(
        per_process_rows,
        active_profile_limits=TEST_SPEC.buckets,
        configured_profile_reserves=calibration["profile_reserves"],
    )
    observed_rows = [
        {
            field: row[field]
            for field in (
                "profile_id",
                "covering_profile_limit",
                "process_first_use_bytes_by_process",
                "device_wide_first_use_bytes_by_process",
                "row_wise_max_process_first_use_bytes",
                "row_wise_max_device_wide_first_use_bytes",
                "row_wise_max_first_use_bytes",
                "calibration_guard_bytes",
                "required_guarded_process_reserve_bytes",
            )
        }
        for row in aggregate_rows
    ]
    document = {
        "schema_version": qualify.PROFILE_SWEEP_CALIBRATION_SCHEMA,
        "measurement_kind": "nvml_process_cumulative_first_use",
        "aggregation": "row_wise_max_across_independent_processes",
        "independent_process_count": qualify.PROFILE_SWEEP_PROCESS_COUNT,
        "model_id": TEST_SPEC.model_id,
        "model_context_limit": TEST_SPEC.context_limit,
        "active_kv_profile_limits": list(TEST_SPEC.buckets),
        "prompt_lengths": list(prompt_lengths),
        "terminal_active_length": TEST_SPEC.context_limit,
        "runner_sha256": qualify._sha256(runner),
        "contract_provenance": {
            "qualified_runtime_stack_sha256": calibration[
                "qualified_runtime_stack_sha256"
            ],
            "plan_set_sha256": calibration["plan_set_sha256"],
            "cuda_module_loading_mode": "lazy",
            "plans": plans,
        },
        "runs": runs,
        "profile_rows": observed_rows,
        "recommended_profile_reserves": [
            {
                "covering_profile_limit": row[
                    "covering_profile_limit"
                ],
                "cumulative_reserve_bytes": row[
                    "required_guarded_process_reserve_bytes"
                ],
            }
            for row in aggregate_rows
        ],
        "gates": {
            "two_independent_processes": True,
            "terminal_decode_reaches_model_limit": True,
            "all_fresh_process_sweeps_passed": True,
        },
        "passed": True,
    }
    evidence_path = tmp_path / "calibration-evidence.json"
    evidence_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_contract = copy.deepcopy(contract)
    final_contract["module_residency_calibration"][
        "evidence_sha256"
    ] = qualify._sha256(evidence_path)
    return (
        runner,
        evidence_path,
        (raw_paths[0], raw_paths[1]),
        source_bundle,
        final_contract,
    )


def test_profile_sweep_prompts_cover_each_interval_and_terminal_a_equals_m() -> None:
    assert qualify._profile_sweep_prompt_lengths(
        TEST_SPEC.buckets,
        model_context_limit=TEST_SPEC.context_limit,
    ) == (127, 255, 511, 2_047)
    with pytest.raises(
        ValueError,
        match="cannot choose a one-decode prompt",
    ):
        qualify._profile_sweep_prompt_lengths(
            (2, 3),
            model_context_limit=3,
        )


def test_qualification_model_resolution_requires_sealed_v2_bundle(
    tmp_path: Path,
) -> None:
    _, header = _write_base_bundle(tmp_path / "sealed.trtfb")
    assert qualify._resolve_spec(header) == TEST_SPEC

    provisional = copy.deepcopy(header)
    provisional["runtime_memory"]["contract_version"] = 1
    del provisional["runtime_memory"]["module_residency_calibration"]
    with pytest.raises(ValueError, match="contract_version"):
        qualify._resolve_spec(provisional)


def test_profile_sweep_trace_strictly_binds_v2_receipt_and_two_sweeps(
    tmp_path: Path,
) -> None:
    trace, token_paths, prompt_lengths, logits_path = (
        _profile_sweep_test_trace(tmp_path)
    )

    rows = qualify._validate_profile_sweep_trace(
        trace,
        contract=_profile_sweep_test_contract(),
        expected_model_id=TEST_SPEC.model_id,
        token_paths=token_paths,
        prompt_lengths=prompt_lengths,
        logits_path=logits_path,
        expected_sampler=TEST_SAMPLER,
    )

    assert [row["profile_id"] for row in rows] == [0, 1, 2, 3]
    assert rows[-1]["cumulative_process_first_use_bytes"] == 400


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("loading-mode", "module-loading mode"),
        ("receipt-evidence", "does not bind the sealed bundle"),
        ("profile-row", "exact profile"),
        ("a-b-equivalence", "incomplete A/B equivalence"),
        ("second-sweep-growth", "high-water summary"),
    ),
)
def test_profile_sweep_trace_fails_closed(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    trace, token_paths, prompt_lengths, logits_path = (
        _profile_sweep_test_trace(tmp_path)
    )
    if mutation == "loading-mode":
        trace["cuda_module_loading"]["mode"] = "eager"
    elif mutation == "receipt-evidence":
        trace["sweep_a"]["rows"][0]["runtime_memory_receipt"][
            "module_residency_evidence_sha256"
        ] = "f" * 64
    elif mutation == "profile-row":
        trace["sweep_b"]["rows"][1]["observed_decode_profile_ids"] = [0]
    elif mutation == "a-b-equivalence":
        trace["sweep_b"]["rows"][2]["equivalence_to_sweep_a"][
            "passed"
        ] = False
    elif mutation == "second-sweep-growth":
        trace["sweep_b"]["incremental_process_high_water_bytes"] = (
            64 * 1024 * 1024 + 1
        )
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=error):
        qualify._validate_profile_sweep_trace(
            trace,
            contract=_profile_sweep_test_contract(),
            expected_model_id=TEST_SPEC.model_id,
            token_paths=token_paths,
            prompt_lengths=prompt_lengths,
            logits_path=logits_path,
            expected_sampler=TEST_SAMPLER,
        )


def test_source_calibration_reopens_doc_raw_captures_and_bootstrap_plans(
    tmp_path: Path,
) -> None:
    runner, evidence, raws, _source_bundle, contract = (
        _source_calibration_fixture(tmp_path)
    )

    binding = qualify._validate_source_calibration_evidence(
        evidence_path=evidence,
        raw_capture_paths=raws,
        runner=runner,
        contract=contract,
        spec=TEST_SPEC,
    )

    assert binding["passed"]
    assert binding["document"]["sha256"] == contract[
        "module_residency_calibration"
    ]["evidence_sha256"]
    assert len(binding["raw_captures"]) == 2
    assert binding["bootstrap_cycle_exemption"] == {
        "field": "module_residency_evidence_sha256",
        "reason": (
            "bootstrap raw receipts predate the evidence document "
            "whose SHA is sealed into the final contract"
        ),
        "observed_bootstrap_values": ["0" * 64],
        "final_sealed_value": contract[
            "module_residency_calibration"
        ]["evidence_sha256"],
        "all_other_receipt_provenance_replayed": True,
    }


def test_source_calibration_rejects_raw_capture_hash_tamper(
    tmp_path: Path,
) -> None:
    runner, evidence, raws, _source_bundle, contract = (
        _source_calibration_fixture(tmp_path)
    )
    raws[1].write_text(
        raws[1].read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="raw/capture-manifest hash mismatch",
    ):
        qualify._validate_source_calibration_evidence(
            evidence_path=evidence,
            raw_capture_paths=raws,
            runner=runner,
            contract=contract,
            spec=TEST_SPEC,
        )


def test_source_calibration_rejects_bootstrap_plan_byte_tamper(
    tmp_path: Path,
) -> None:
    runner, evidence, raws, source_bundle, contract = (
        _source_calibration_fixture(tmp_path)
    )
    payload = bytearray(source_bundle.read_bytes())
    payload[-1] ^= 1
    source_bundle.write_bytes(payload)

    with pytest.raises(
        RuntimeError,
        match="bootstrap bundle plan bytes differ",
    ):
        qualify._validate_source_calibration_evidence(
            evidence_path=evidence,
            raw_capture_paths=raws,
            runner=runner,
            contract=contract,
            spec=TEST_SPEC,
        )


def test_source_calibration_exempts_only_evidence_sha_cycle(
    tmp_path: Path,
) -> None:
    runner, evidence, raws, _source_bundle, contract = (
        _source_calibration_fixture(
            tmp_path,
            receipt_plan_tamper=True,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="runtime-memory receipt does not bind.*plan_set",
    ):
        qualify._validate_source_calibration_evidence(
            evidence_path=evidence,
            raw_capture_paths=raws,
            runner=runner,
            contract=contract,
            spec=TEST_SPEC,
        )


def _embedded_calibration_fixture(
    tmp_path: Path,
    *,
    raw_sweep_path_tamper: bool = False,
) -> tuple[Path, Path, dict]:
    """Write one self-contained bundle with two replayable build captures."""

    runner = tmp_path / "qualifier-runner"
    runner.write_bytes(b"embedded-manifest-bound-qualifier")
    runner.chmod(0o755)
    _unused, template_header = _write_base_bundle(
        tmp_path / "template.trtfb"
    )
    final_contract = copy.deepcopy(template_header["runtime_memory"])
    decode_plan = b"embedded-decode-plan"
    prefill_plan = b"embedded-prefill-plan"
    plans = [
        {
            "section_name": "engine_plan",
            "section_sha256": hashlib.sha256(decode_plan).hexdigest(),
            "role": "decode",
            "optimization_profile_count": len(TEST_SPEC.buckets),
        },
        {
            "section_name": "prefill_engine_plan",
            "section_sha256": hashlib.sha256(prefill_plan).hexdigest(),
            "role": "prefill",
            "optimization_profile_count": 1,
        },
    ]
    calibration = final_contract["module_residency_calibration"]
    calibration["plans"] = plans
    calibration["plan_set_sha256"] = (
        qualify.module_residency_plan_set_sha256(plans)
    )
    bootstrap_contract = copy.deepcopy(final_contract)
    bootstrap_calibration = bootstrap_contract[
        "module_residency_calibration"
    ]
    bootstrap_calibration["plans"] = plans
    bootstrap_calibration["plan_set_sha256"] = calibration[
        "plan_set_sha256"
    ]
    bootstrap_calibration["evidence_sha256"] = "0" * 64
    bootstrap_calibration["profile_reserves"] = [
        {
            "covering_profile_limit": limit,
            "cumulative_reserve_bytes": 1,
        }
        for limit in TEST_SPEC.buckets
    ]
    prompt_lengths = qualify._profile_sweep_prompt_lengths(
        TEST_SPEC.buckets,
        model_context_limit=TEST_SPEC.context_limit,
    )

    def canonical(value: object) -> bytes:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                separators=(",", ": "),
            )
            + "\n"
        ).encode("utf-8")

    capture_sections: list[tuple[str, bytes]] = []
    runs: list[dict] = []
    process_rows: list[list[dict]] = []
    for process_index in range(qualify.PROFILE_SWEEP_PROCESS_COUNT):
        capture_dir = tmp_path / f"source-process-{process_index:02d}"
        capture_dir.mkdir()
        trace, token_paths, _, logits_path = _profile_sweep_test_trace(
            capture_dir
        )
        pid = 900 + process_index
        trace["memory_sampler"]["pid"] = pid
        for sweep_name in ("sweep_a", "sweep_b"):
            for row in trace[sweep_name]["rows"]:
                receipt = row["runtime_memory_receipt"]
                receipt["module_residency_reserve_bytes"] = 1
                receipt["module_residency_plan_set_sha256"] = calibration[
                    "plan_set_sha256"
                ]
                receipt["module_residency_evidence_sha256"] = "0" * 64
        prefix = (
            f"{qualify.EMBEDDED_CALIBRATION_ROOT}/"
            f"process-{process_index:02d}"
        )
        recorded_bootstrap = (
            tmp_path / "private-bootstrap.trtfb"
        ).resolve()
        command = qualify._profile_sweep_command(
            runner=runner.resolve(),
            bundle=recorded_bootstrap,
            token_paths=token_paths,
            logits_path=logits_path,
            model_context_limit=TEST_SPEC.context_limit,
        )
        if raw_sweep_path_tamper and process_index == 1:
            trace["sweep_b"]["rows"][0]["token_file"] = str(
                (capture_dir / "different-profile.tokens.txt").resolve()
            )
        raw_trace = (
            json.dumps(trace, sort_keys=True) + "\n"
        ).encode("utf-8")
        artifact_payloads = {
            "command": ("command.json", canonical(command)),
            "returncode": ("returncode.txt", b"0\n"),
            "runner_stdout": ("runner.stdout.log", raw_trace),
            "runner_stderr": ("runner.stderr.log", b""),
            "raw_trace": ("runner-output.raw.json", raw_trace),
            "normalized_trace": (
                "runner-trace.json",
                canonical(trace),
            ),
            "logits": ("runner-logits.bin", logits_path.read_bytes()),
        }
        artifact_payloads.update(
            {
                f"profile_tokens_{index:02d}": (
                    f"profile-{index:02d}.tokens.txt",
                    token_path.read_bytes(),
                )
                for index, token_path in enumerate(token_paths)
            }
        )
        artifacts: dict[str, dict] = {}
        for logical_name, (filename, payload) in artifact_payloads.items():
            section_name = f"{prefix}/{filename}"
            capture_sections.append((section_name, payload))
            artifacts[logical_name] = {
                "section_name": section_name,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        anchor = {
            "pid": pid,
            "cuda_logical_device_index": 0,
            "physical_device_index": TEST_SAMPLER.physical_device_index,
            "pci_bus_id": TEST_SAMPLER.pci_bus_id,
            "gpu_uuid": TEST_SAMPLER.gpu_uuid,
        }
        manifest = {
            "schema": qualify.EMBEDDED_CALIBRATION_CAPTURE_SCHEMA,
            "process_index": process_index,
            "runner_pid": pid,
            "gpu_uuid": TEST_SAMPLER.gpu_uuid,
            "sampler_trust_anchor": anchor,
            "artifacts": artifacts,
        }
        manifest_name = f"{prefix}/capture-manifest.json"
        manifest_bytes = canonical(manifest)
        capture_sections.append((manifest_name, manifest_bytes))
        runs.append(
            {
                "process_index": process_index,
                "runner_pid": pid,
                "gpu_uuid": TEST_SAMPLER.gpu_uuid,
                "sampler_trust_anchor": anchor,
                "capture_manifest": {
                    "section_name": manifest_name,
                    "size_bytes": len(manifest_bytes),
                    "sha256": hashlib.sha256(
                        manifest_bytes
                    ).hexdigest(),
                },
                "artifacts": artifacts,
            }
        )
        process_rows.append(trace["profile_reserve_rows"])
    aggregated = qualify.aggregate_profile_sweep_calibration_rows(
        process_rows,
        active_profile_limits=TEST_SPEC.buckets,
        configured_profile_reserves=calibration["profile_reserves"],
    )
    observed_rows = [
        {
            field: row[field]
            for field in (
                "profile_id",
                "covering_profile_limit",
                "process_first_use_bytes_by_process",
                "device_wide_first_use_bytes_by_process",
                "row_wise_max_process_first_use_bytes",
                "row_wise_max_device_wide_first_use_bytes",
                "row_wise_max_first_use_bytes",
                "calibration_guard_bytes",
                "required_guarded_process_reserve_bytes",
            )
        }
        for row in aggregated
    ]
    recommended = [
        {
            "covering_profile_limit": row["covering_profile_limit"],
            "cumulative_reserve_bytes": row[
                "required_guarded_process_reserve_bytes"
            ],
        }
        for row in aggregated
    ]
    calibration["profile_reserves"] = recommended
    runner_stat = runner.stat()
    helper_identity = {
        "device": runner_stat.st_dev,
        "inode": runner_stat.st_ino,
        "size_bytes": runner_stat.st_size,
        "mtime_ns": runner_stat.st_mtime_ns,
        "sha256": qualify._sha256(runner),
    }
    evidence = {
        "schema": qualify.EMBEDDED_CALIBRATION_EVIDENCE_SCHEMA,
        "measurement_kind": "nvml_process_cumulative_first_use",
        "aggregation": "row_wise_process_max_plus_64mib_guard",
        "independent_process_count": 2,
        "model_id": TEST_SPEC.model_id,
        "model_context_limit": TEST_SPEC.context_limit,
        "active_kv_profile_limits": list(TEST_SPEC.buckets),
        "prompt_lengths": list(prompt_lengths),
        "terminal_active_length": TEST_SPEC.context_limit,
        "helper_identity_before": helper_identity,
        "helper_identity_after": helper_identity,
        "contract_provenance": {
            "qualified_runtime_stack_sha256": calibration[
                "qualified_runtime_stack_sha256"
            ],
            "plan_set_sha256": calibration["plan_set_sha256"],
            "cuda_module_loading_mode": "lazy",
            "plans": plans,
        },
        "bootstrap_contract": bootstrap_contract,
        "bootstrap_only": {
            "profile_reserve_bytes": 1,
            "evidence_sha256": "0" * 64,
            "never_published": True,
        },
        "runs": runs,
        "profile_rows": observed_rows,
        "recommended_profile_reserves": recommended,
        "gates": {
            "two_distinct_processes": True,
            "single_gpu_identity": True,
            "all_profile_upper_edges_executed": True,
            "terminal_decode_reaches_model_limit": True,
            "second_sweep_growth_within_limit": True,
            "raw_plan_receipts_match_bootstrap": True,
            "all_capture_sections_embedded_and_hashed": True,
            "helper_identity_unchanged": True,
        },
        "passed": True,
    }
    evidence_bytes = canonical(evidence)
    calibration["evidence_sha256"] = hashlib.sha256(
        evidence_bytes
    ).hexdigest()
    all_sections = [
        ("engine_plan", decode_plan),
        ("prefill_engine_plan", prefill_plan),
        (qualify.EMBEDDED_CALIBRATION_EVIDENCE_SECTION, evidence_bytes),
        *capture_sections,
    ]
    offset = 0
    section_table: dict[str, dict[str, int]] = {}
    for name, payload in all_sections:
        section_table[name] = {"offset": offset, "size": len(payload)}
        offset += len(payload)
    header = {
        "model_id": TEST_SPEC.model_id,
        "precision": "bf16",
        "vocab_size": TEST_SPEC.vocab_size,
        "runtime_memory": final_contract,
        "sections": section_table,
    }
    header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
    bundle = tmp_path / "embedded.trtfb"
    bundle.write_bytes(
        qualify.BUNDLE_MAGIC
        + struct.pack("<Q", len(header_bytes))
        + header_bytes
        + b"".join(payload for _name, payload in all_sections)
    )
    return runner, bundle, header


def test_embedded_calibration_reopens_all_bundle_sections(
    tmp_path: Path,
) -> None:
    runner, bundle, header = _embedded_calibration_fixture(tmp_path)
    binding = qualify._validate_embedded_source_calibration_evidence(
        bundle=bundle,
        header=header,
        runner=runner,
        contract=header["runtime_memory"],
        spec=TEST_SPEC,
    )
    assert binding["source"] == "embedded_bundle_sections"
    assert (
        binding["evidence_schema"]
        == qualify.EMBEDDED_CALIBRATION_EVIDENCE_SCHEMA
    )
    assert binding["passed"] is True
    assert len(binding["capture_manifests"]) == 2
    assert len(binding["raw_captures"]) == 2
    assert binding["evidence_section"]["sha256"] == header[
        "runtime_memory"
    ]["module_residency_calibration"]["evidence_sha256"]


def test_embedded_calibration_rejects_section_byte_tamper(
    tmp_path: Path,
) -> None:
    runner, bundle, header = _embedded_calibration_fixture(tmp_path)
    raw_name = (
        f"{qualify.EMBEDDED_CALIBRATION_ROOT}/"
        "process-01/runner-output.raw.json"
    )
    raw_entry = header["sections"][raw_name]
    header_size = struct.unpack("<Q", bundle.read_bytes()[8:16])[0]
    payload = bytearray(bundle.read_bytes())
    payload[16 + header_size + raw_entry["offset"]] ^= 1
    bundle.write_bytes(payload)
    with pytest.raises(RuntimeError, match="does not match its receipt"):
        qualify._validate_embedded_source_calibration_evidence(
            bundle=bundle,
            header=header,
            runner=runner,
            contract=header["runtime_memory"],
            spec=TEST_SPEC,
        )


def test_embedded_calibration_rejects_helper_byte_tamper(
    tmp_path: Path,
) -> None:
    runner, bundle, header = _embedded_calibration_fixture(tmp_path)
    runner.write_bytes(runner.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="helper identity"):
        qualify._validate_embedded_source_calibration_evidence(
            bundle=bundle,
            header=header,
            runner=runner,
            contract=header["runtime_memory"],
            spec=TEST_SPEC,
        )


def test_embedded_calibration_rejects_raw_sweep_token_path_mismatch(
    tmp_path: Path,
) -> None:
    runner, bundle, header = _embedded_calibration_fixture(
        tmp_path,
        raw_sweep_path_tamper=True,
    )
    with pytest.raises(
        RuntimeError,
        match="raw profile token paths disagree.*sweep_b",
    ):
        qualify._validate_embedded_source_calibration_evidence(
            bundle=bundle,
            header=header,
            runner=runner,
            contract=header["runtime_memory"],
            spec=TEST_SPEC,
        )


def test_profile_sweep_producer_uses_two_processes_and_row_wise_max(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = tmp_path / "runner"
    bundle = tmp_path / "bundle.trtfb"
    runner.write_bytes(b"runner")
    bundle.write_bytes(b"bundle")
    contract = _profile_sweep_test_contract()
    header = {
        "vocab_size": TEST_SPEC.vocab_size,
        "runtime_memory": {
            "active_kv_profile_limits": list(TEST_SPEC.buckets),
        },
    }
    monkeypatch.setattr(
        qualify,
        "_sealed_profile_sweep_contract",
        lambda *args, **kwargs: contract,
    )
    observed_processes: list[int] = []

    def fake_process(**kwargs: object):
        evidence_dir = Path(str(kwargs["evidence_dir"]))
        process_index = int(evidence_dir.name.rsplit("-", 1)[-1])
        observed_processes.append(process_index)
        evidence_dir.mkdir(parents=True)
        prompt_lengths = tuple(kwargs["prompt_lengths"])
        token_paths = tuple(
            evidence_dir / f"profile-{index:02d}.tokens.txt"
            for index in range(len(prompt_lengths))
        )
        for token_path, prompt_tokens in zip(
            token_paths,
            prompt_lengths,
            strict=True,
        ):
            qualify._write_tokens(
                token_path,
                qualify.deterministic_token_ids(
                    prompt_tokens,
                    TEST_SPEC.vocab_size,
                ),
            )
        logits_path = evidence_dir / "runner-logits.bin"
        logits_path.write_bytes(f"logits-{process_index}".encode())
        command = qualify._profile_sweep_command(
            runner=runner.resolve(),
            bundle=bundle.resolve(),
            token_paths=token_paths,
            logits_path=logits_path,
            model_context_limit=TEST_SPEC.context_limit,
        )
        (evidence_dir / "command.json").write_text(
            json.dumps(command) + "\n",
            encoding="utf-8",
        )
        rows = [
            {
                "profile_id": index,
                "covering_profile_limit": limit,
                "cumulative_process_first_use_bytes": (
                    (index + 1) * 100 + process_index
                ),
                "cumulative_device_wide_first_use_bytes": (
                    (index + 1) * 200 + process_index
                ),
            }
            for index, limit in enumerate(TEST_SPEC.buckets)
        ]
        anchor = qualify.SamplerTrustAnchor(
            pid=100 + process_index,
            cuda_logical_device_index=0,
            physical_device_index=TEST_SAMPLER.physical_device_index,
            pci_bus_id=TEST_SAMPLER.pci_bus_id,
            gpu_uuid=TEST_SAMPLER.gpu_uuid,
        )
        trace = {
            "passed": True,
            "test_process_index": process_index,
            "sweep_b": {
                "incremental_process_high_water_bytes": 0,
            },
        }
        raw_trace = json.dumps(trace, sort_keys=True) + "\n"
        (evidence_dir / "runner.stdout.log").write_text(
            raw_trace,
            encoding="utf-8",
        )
        (evidence_dir / "runner.stderr.log").write_text(
            "",
            encoding="utf-8",
        )
        (evidence_dir / "returncode.txt").write_text(
            "0\n",
            encoding="utf-8",
        )
        (evidence_dir / "runner-output.raw.json").write_text(
            raw_trace,
            encoding="utf-8",
        )
        (evidence_dir / "runner-trace.json").write_text(
            json.dumps(trace, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        qualify._write_profile_sweep_capture_manifest(
            evidence_dir,
            child_pid=anchor.pid,
            sampler_anchor=anchor,
            token_paths=token_paths,
        )
        return (
            trace,
            rows,
            anchor,
        )

    monkeypatch.setattr(
        qualify,
        "_run_profile_sweep_process",
        fake_process,
    )
    source_document = tmp_path / "source-calibration-evidence.json"
    source_document.write_text("{}\n", encoding="utf-8")
    source_raw_captures = (
        tmp_path / "source-process-00.raw.json",
        tmp_path / "source-process-01.raw.json",
    )
    for path in source_raw_captures:
        path.write_text("{}\n", encoding="utf-8")
    source_binding = {
        "document": qualify._simple_file_identity(source_document),
        "raw_captures": [
            qualify._simple_file_identity(path)
            for path in source_raw_captures
        ],
        "passed": True,
    }
    monkeypatch.setattr(
        qualify,
        "_validate_source_calibration_evidence",
        lambda **_kwargs: source_binding,
    )

    evidence = qualify.produce_profile_sweep_evidence(
        runner=runner,
        bundle=bundle,
        header=header,
        spec=TEST_SPEC,
        evidence_dir=tmp_path / "evidence",
        runner_cuda_visible_device="7",
        source_calibration_evidence=source_document,
        source_calibration_raw_captures=source_raw_captures,
    )

    assert observed_processes == [0, 1]
    assert evidence["process_count"] == 2
    final = evidence["aggregation"]["profile_rows"][-1]
    assert final["process_first_use_bytes_by_process"] == [400, 401]
    assert final["row_wise_max_process_first_use_bytes"] == 401
    assert final["row_wise_max_device_wide_first_use_bytes"] == 801
    assert final["required_guarded_process_reserve_bytes"] == (
        401 + 64 * 1024 * 1024
    )
    assert final[
        "configured_reserve_covers_process_max_plus_guard"
    ]
    assert evidence["source_calibration_evidence"] == source_binding
    calibration = Path(evidence["fresh_calibration_evidence"]["path"])
    assert evidence["fresh_calibration_evidence_sha256"] == qualify._sha256(
        calibration
    )
    calibration_payload = json.loads(
        calibration.read_text(encoding="utf-8")
    )
    assert calibration_payload["schema_version"].endswith("/v2")
    assert "bundle_sha256" not in calibration_payload
    assert (
        "evidence_sha256"
        not in calibration_payload["contract_provenance"]
    )
    assert all(
        "configured_cumulative_reserve_bytes" not in row
        for row in calibration_payload["profile_rows"]
    )

    monkeypatch.setattr(
        qualify,
        "_read_bundle_header",
        lambda _bundle: header,
    )
    monkeypatch.setattr(
        qualify,
        "_validate_profile_sweep_trace",
        lambda trace, **kwargs: [
            {
                "profile_id": index,
                "covering_profile_limit": limit,
                "cumulative_process_first_use_bytes": (
                    (index + 1) * 100 + trace["test_process_index"]
                ),
                "cumulative_device_wide_first_use_bytes": (
                    (index + 1) * 200 + trace["test_process_index"]
                ),
            }
            for index, limit in enumerate(TEST_SPEC.buckets)
        ],
    )
    assert qualify._persisted_profile_sweep_evidence_passed(
        evidence,
        runner=runner,
        bundle=bundle,
        spec=TEST_SPEC,
    )

    calibration.write_text(
        calibration.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    assert not qualify._persisted_profile_sweep_evidence_passed(
        evidence,
        runner=runner,
        bundle=bundle,
        spec=TEST_SPEC,
    )


@pytest.mark.parametrize(
    "runner_selector",
    (
        "3",
        "GPU-fedcba98-7654-3210-fedc-ba9876543210",
    ),
)
def test_sampler_anchor_resolves_the_child_selector_not_parent_environment(
    runner_selector: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    def fake_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "2, 0000:02:00.0, "
                "GPU-01234567-89ab-cdef-0123-456789abcdef\n"
                "3, 0000:03:00.0, "
                "GPU-fedcba98-7654-3210-fedc-ba9876543210\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(qualify.subprocess, "run", fake_run)

    anchor = qualify._sampler_trust_anchor(
        child_pid=456,
        cuda_logical_device_index=0,
        cuda_visible_device=runner_selector,
    )

    assert anchor.pid == 456
    assert anchor.cuda_logical_device_index == 0
    assert anchor.physical_device_index == 3
    assert anchor.pci_bus_id == "0000:03:00.0"
    assert (
        anchor.gpu_uuid
        == "GPU-fedcba98-7654-3210-fedc-ba9876543210"
    )
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_sampler_anchor_rejects_tampered_child_gpu_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "3, 0000:03:00.0, "
                "GPU-fedcba98-7654-3210-fedc-ba9876543210\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(qualify.subprocess, "run", fake_run)

    with pytest.raises(
        RuntimeError,
        match="cannot resolve CUDA-visible GPU selector",
    ):
        qualify._sampler_trust_anchor(
            child_pid=456,
            cuda_logical_device_index=0,
            cuda_visible_device=(
                "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
            ),
        )


def test_deterministic_token_ids_are_prefix_stable_and_in_vocab() -> None:
    short = qualify.deterministic_token_ids(2_047, 32_000)
    long = qualify.deterministic_token_ids(2_049, 32_000)

    np.testing.assert_array_equal(short, long[: short.size])
    assert short.dtype == np.int32
    assert int(short.min()) >= 1
    assert int(long.max()) < 32_000


def test_dirty_source_provenance_captures_both_patches_and_untracked_hashes(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    staged = repo / "staged.txt"
    unstaged = repo / "unstaged.txt"
    staged.write_text("base staged\n", encoding="utf-8")
    unstaged.write_text("base unstaged\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "staged.txt", "unstaged.txt"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=TRTMC Test",
            "-c",
            "user.email=trtmc@example.invalid",
            "commit",
            "-qm",
            "base",
        ],
        cwd=repo,
        check=True,
    )

    staged.write_text("changed and staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "staged.txt"], cwd=repo, check=True)
    unstaged.write_text("changed but unstaged\n", encoding="utf-8")
    untracked = repo / "new-source.cpp"
    untracked.write_text("int value = 1;\n", encoding="utf-8")
    artifact_dir = repo / "artifacts" / "qualification"

    before = qualify.source_state_provenance(
        repo,
        MODULE_PATH,
        artifact_dir,
        label="pre",
    )

    assert before["git_dirty"]
    assert not before["exact_head_gate_satisfied"]
    assert before["artifacts"]["staged_patch"]["size_bytes"] > 0
    assert before["artifacts"]["unstaged_patch"]["size_bytes"] > 0
    assert before["untracked_files"] == [
        {
            "path": "new-source.cpp",
            "kind": "file",
            "size_bytes": untracked.stat().st_size,
            "sha256": hashlib.sha256(untracked.read_bytes()).hexdigest(),
        }
    ]
    manifest = json.loads(
        Path(before["artifacts"]["untracked_manifest"]["path"]).read_text(encoding="utf-8")
    )
    assert manifest == before["untracked_files"]

    untracked.write_text("int value = 2;\n", encoding="utf-8")
    after = qualify.source_state_provenance(
        repo,
        MODULE_PATH,
        artifact_dir,
        label="post",
    )
    assert after["source_state_sha256"] != before["source_state_sha256"]


def test_qwen_matrix_contains_exact_chunk_and_model_boundaries() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    cases = {case.name: case for case in qualify._cases_for(spec)}

    assert (
        cases["c-minus-1"].prompt_tokens,
        cases["c"].prompt_tokens,
        cases["c-plus-1"].prompt_tokens,
    ) == (1_023, 1_024, 1_025)
    assert cases["two-c-plus-17"].prompt_tokens == 2_065
    assert (cases["total-32768"].prompt_tokens, cases["total-32768"].decode_tokens) == (32_760, 8)
    assert (cases["total-model-limit"].prompt_tokens, cases["total-model-limit"].decode_tokens) == (
        40_952,
        8,
    )
    assert cases["prefill-last-position"].prompt_tokens == 40_960
    assert cases["model-limit-plus-1"].prompt_tokens == 40_961
    assert cases["model-limit-plus-1"].expect_admission_rejection
    for bucket in (128, 256, 512, 1_024, 2_048, 8_192, 32_768):
        crossing = cases[f"profile-crossing-{bucket}"]
        assert (crossing.prompt_tokens, crossing.decode_tokens) == (bucket, 2)
    for index, bucket in enumerate(spec.buckets[:-1]):
        expected = (
            ("p-minus-1", bucket - 1, index),
            ("p", bucket, index),
            ("p-plus-1", bucket + 1, index + 1),
        )
        for label, prompt_tokens, profile_id in expected:
            boundary = cases[f"decode-bucket-{bucket}-{label}"]
            assert (boundary.prompt_tokens, boundary.decode_tokens) == (
                prompt_tokens,
                1,
            )
            assert boundary.expected_decode_profile_ids == (profile_id,)
            assert boundary.expected_decode_bucket_limits == (spec.buckets[profile_id],)


def test_tiny_matrix_covers_every_bucket_neighbor_and_m_plus_one() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    cases = qualify._cases_for(spec)
    lengths = {case.prompt_tokens for case in cases}

    for bucket in spec.buckets:
        assert {bucket - 1, bucket, bucket + 1}.issubset(lengths)
    rejected = [case for case in cases if case.expect_admission_rejection]
    assert [(case.prompt_tokens, case.decode_tokens) for case in rejected] == [(2_049, 0)]
    assert any(case.prompt_tokens == 2_040 and case.decode_tokens == 8 for case in cases)
    assert any(case.prompt_tokens == 2_048 and case.decode_tokens == 0 for case in cases)
    by_name = {case.name: case for case in cases}
    for bucket in (128, 256, 512):
        assert (
            by_name[f"profile-crossing-{bucket}"].prompt_tokens,
            by_name[f"profile-crossing-{bucket}"].decode_tokens,
        ) == (bucket, 2)
    for index, bucket in enumerate(spec.buckets[:-1]):
        for label, prompt_tokens, profile_id in (
            ("p-minus-1", bucket - 1, index),
            ("p", bucket, index),
            ("p-plus-1", bucket + 1, index + 1),
        ):
            boundary = by_name[f"decode-bucket-{bucket}-{label}"]
            assert (boundary.prompt_tokens, boundary.decode_tokens) == (
                prompt_tokens,
                1,
            )
            assert boundary.expected_decode_profile_ids == (profile_id,)
            assert boundary.expected_decode_bucket_limits == (spec.buckets[profile_id],)


def test_runner_command_warms_normal_cases_but_not_admission_rejections() -> None:
    common = {
        "runner": Path("/runner"),
        "bundle": Path("/model.trtfb"),
        "token_path": Path("/tokens.txt"),
        "logits_path": Path("/logits.bin"),
        "context_limit": 2_048,
    }

    normal = qualify._runner_command(
        **common,
        case=qualify.Case("normal", 128, 1),
    )
    rejected = qualify._runner_command(
        **common,
        case=qualify.Case("rejected", 2_049, 0, expect_admission_rejection=True),
    )

    assert normal.count("--warmup-load-cycle") == 1
    assert "--warmup-load-cycle" not in rejected


def _qualified_engine_graph_evidence(spec) -> dict:
    num_layers = spec.num_layers
    width = spec.kv_bytes_per_token // (2 * num_layers * qualify._KV_DTYPE_BYTES[spec.kv_dtype])

    def section(role: str) -> dict:
        is_prefill = role == "prefill"
        profile_count = 1 if is_prefill else len(spec.buckets)
        token_profiles = (
            [
                {
                    "min": [1],
                    "opt": [spec.chunk_limit],
                    "max": [spec.chunk_limit],
                }
            ]
            if is_prefill
            else [{"min": [1], "opt": [1], "max": [1]} for _ in spec.buckets]
        )
        inputs = {
            "token_id": {
                "shape": [-1] if is_prefill else [1],
                "profiles": token_profiles,
            },
            "position_id": {
                "shape": [-1] if is_prefill else [1],
                "profiles": copy.deepcopy(token_profiles),
            },
            "history_length": {
                "shape": [1],
                "profiles": [{"min": [1], "opt": [1], "max": [1]} for _ in range(profile_count)],
            },
        }
        outputs = {
            "logits": {
                "shape": [1, spec.vocab_size],
            },
        }
        for layer in range(num_layers):
            for value_name in ("k", "v"):
                cache_profiles = (
                    [
                        {
                            "min": [1, width],
                            "opt": [spec.chunk_limit, width],
                            "max": [spec.context_limit, width],
                        }
                    ]
                    if is_prefill
                    else [
                        {
                            "min": [1, width],
                            "opt": [bucket, width],
                            "max": [bucket, width],
                        }
                        for bucket in spec.buckets
                    ]
                )
                inputs[f"cache_{value_name}_{layer}"] = {
                    "shape": [-1, width],
                    "profiles": cache_profiles,
                }
                outputs[f"present_{value_name}_{layer}"] = {
                    "shape": [-1 if is_prefill else 1, width],
                }
        return {
            "engine_sha256": ("a" * 64 if is_prefill else "b" * 64),
            "num_optimization_profiles": profile_count,
            "inputs": inputs,
            "outputs": outputs,
            "native_contiguous_attention_layer_indices": list(range(num_layers)),
            "dense_attention_layers": [],
            "cache_concat_layers": [],
            "inspector_path": f"{role}.engine-inspector.json",
            "inspector_size_bytes": 128,
            "inspector_sha256": ("c" * 64 if is_prefill else "d" * 64),
        }

    return {
        "runtime_stack": {
            "sm": "sm103",
            "tensorrt": "11.2.0.113",
            "cuda_runtime": "13.3",
            "driver": "580.105.08",
        },
        "model_contract": {
            "model_context_limit": spec.context_limit,
            "prefill_chunk_limit": spec.chunk_limit,
            "active_kv_profile_limits": list(spec.buckets),
            "num_layers": num_layers,
            "vocab_size": spec.vocab_size,
            "kv_dtype": spec.kv_dtype,
            "kv_bytes_per_token": spec.kv_bytes_per_token,
            "kv_width": width,
        },
        "engine_sections": {
            "prefill_engine_plan": section("prefill"),
            "engine_plan": section("decode"),
        },
    }


@pytest.mark.parametrize(
    ("model_id", "expected_layers", "expected_width"),
    (
        ("Qwen/Qwen3-0.6B", 28, 1_024),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 22, 256),
    ),
)
def test_full_model_engine_graph_gate_accepts_live_io_contract(
    model_id: str,
    expected_layers: int,
    expected_width: int,
) -> None:
    spec = qualify.SPECS[model_id]
    evidence = _qualified_engine_graph_evidence(spec)

    result = qualify._validate_qualified_engine_graph_evidence(
        evidence,
        spec,
        num_layers=spec.num_layers,
        expected_runtime_stack=evidence["runtime_stack"],
    )

    assert result["passed"]
    assert all(result["gates"].values())
    assert result["runtime_stack"] == {
        "sm": "sm103",
        "tensorrt": "11.2.0.113",
        "cuda_runtime": "13.3",
        "driver": "580.105.08",
    }
    assert result["model_contract"] == evidence["model_contract"]
    assert result["model_contract"]["num_layers"] == expected_layers
    assert result["model_contract"]["kv_width"] == expected_width
    assert result["engine_sections"]["prefill_engine_plan"]["outputs"]["logits"]["shape"] == [
        1,
        spec.vocab_size,
    ]


def test_graph_model_contract_is_derived_from_bundle_header() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    header = {
        "num_layers": spec.num_layers,
        "vocab_size": spec.vocab_size,
        "runtime_memory": {
            "model_context_limit": spec.context_limit,
            "prefill_chunk_limit": spec.chunk_limit,
            "active_kv_profile_limits": list(spec.buckets),
            "kv_dtype": spec.kv_dtype,
            "kv_bytes_per_token": spec.kv_bytes_per_token,
        },
    }

    assert qualify._graph_model_contract_from_bundle_header(header) == {
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "active_kv_profile_limits": list(spec.buckets),
        "num_layers": 28,
        "vocab_size": 151_936,
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 114_688,
        "kv_width": 1_024,
    }


def test_full_model_engine_graph_gate_fails_closed_on_runtime_stack_tamper() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    evidence = _qualified_engine_graph_evidence(spec)
    expected_stack = copy.deepcopy(evidence["runtime_stack"])
    evidence["runtime_stack"]["driver"] = "tampered"

    with pytest.raises(RuntimeError, match="runtime stack"):
        qualify._validate_qualified_engine_graph_evidence(
            evidence,
            spec,
            num_layers=spec.num_layers,
            expected_runtime_stack=expected_stack,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("missing-runtime-stack", "runtime stack"),
        ("attention-mask-input", "forbidden attention_mask"),
        ("missing-native-layer", "NativeContiguousAttentionV2"),
        ("dense-attention-path", "dense attention mask/score"),
        ("cache-concat", "full-history cache concat"),
        ("full-history-present", "current-row output"),
        ("same-plan", "same serialized engine identity"),
    ),
)
def test_full_model_engine_graph_gate_fails_closed(
    mutation: str,
    error: str,
) -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    evidence = _qualified_engine_graph_evidence(spec)
    expected_stack = copy.deepcopy(evidence["runtime_stack"])
    prefill = evidence["engine_sections"]["prefill_engine_plan"]
    decode = evidence["engine_sections"]["engine_plan"]
    if mutation == "missing-runtime-stack":
        evidence.pop("runtime_stack")
    elif mutation == "attention-mask-input":
        decode["inputs"]["attention_mask"] = {
            "shape": [1, -1],
            "profiles": [],
        }
    elif mutation == "missing-native-layer":
        decode["native_contiguous_attention_layer_indices"] = [0]
    elif mutation == "dense-attention-path":
        prefill["dense_attention_layers"] = ["layer.1.attn.attention_scores"]
    elif mutation == "cache-concat":
        decode["cache_concat_layers"] = ["layer.0.cache_concat"]
    elif mutation == "full-history-present":
        decode["outputs"]["present_k_0"]["shape"] = [
            -1,
            evidence["model_contract"]["kv_width"],
        ]
    elif mutation == "same-plan":
        decode["engine_sha256"] = prefill["engine_sha256"]
    else:  # pragma: no cover - keeps additions to the table explicit.
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=error):
        qualify._validate_qualified_engine_graph_evidence(
            evidence,
            spec,
            num_layers=spec.num_layers,
            expected_runtime_stack=expected_stack,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("token-shape", "token_id shape"),
        ("position-shape", "position_id shape"),
        ("token-position-profile", "profiles are not identical"),
        ("token-profile", "prefill profile does not cover"),
        ("history-shape", "history_length is not a scalar"),
        ("history-profile", "history_length profile"),
        ("logits-row-shape", "logits shape"),
        ("logits-vocab", "logits shape"),
        ("cache-width", "source-bound KV width"),
        ("present-width", "source-bound KV width"),
        ("cache-profile-width", "does not bind bucket"),
        ("derived-width", "KV width does not match"),
        ("nondivisible-b", "not exactly divisible"),
        ("qualified-b-mismatch", "model_contract mismatch"),
        ("unknown-dtype", "unsupported KV dtype"),
    ),
)
def test_full_model_engine_graph_gate_rejects_io_or_kv_geometry_mismatch(
    mutation: str,
    error: str,
) -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    evidence = _qualified_engine_graph_evidence(spec)
    expected_stack = copy.deepcopy(evidence["runtime_stack"])
    prefill = evidence["engine_sections"]["prefill_engine_plan"]
    decode = evidence["engine_sections"]["engine_plan"]
    model_contract = evidence["model_contract"]
    width = model_contract["kv_width"]
    if mutation == "token-shape":
        prefill["inputs"]["token_id"]["shape"] = [1]
    elif mutation == "position-shape":
        decode["inputs"]["position_id"]["shape"] = [-1]
    elif mutation == "token-position-profile":
        prefill["inputs"]["position_id"]["profiles"][0]["opt"] = [1]
    elif mutation == "token-profile":
        for name in ("token_id", "position_id"):
            prefill["inputs"][name]["profiles"][0]["max"] = [spec.chunk_limit - 1]
    elif mutation == "history-shape":
        decode["inputs"]["history_length"]["shape"] = [-1]
    elif mutation == "history-profile":
        decode["inputs"]["history_length"]["profiles"][0]["max"] = [2]
    elif mutation == "logits-row-shape":
        prefill["outputs"]["logits"]["shape"] = [-1, spec.vocab_size]
    elif mutation == "logits-vocab":
        decode["outputs"]["logits"]["shape"] = [1, spec.vocab_size + 1]
    elif mutation == "cache-width":
        decode["inputs"]["cache_v_1"]["shape"] = [-1, width + 1]
    elif mutation == "present-width":
        decode["outputs"]["present_k_1"]["shape"] = [1, width + 1]
    elif mutation == "cache-profile-width":
        decode["inputs"]["cache_k_0"]["profiles"][0]["opt"] = [
            spec.buckets[0],
            width + 1,
        ]
    elif mutation == "derived-width":
        model_contract["kv_width"] = width + 1
    elif mutation == "nondivisible-b":
        model_contract["kv_bytes_per_token"] += 1
    elif mutation == "qualified-b-mismatch":
        model_contract["kv_bytes_per_token"] *= 2
        model_contract["kv_width"] *= 2
    elif mutation == "unknown-dtype":
        model_contract["kv_dtype"] = "int8"
    else:  # pragma: no cover - keeps additions to the table explicit.
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=error):
        qualify._validate_qualified_engine_graph_evidence(
            evidence,
            spec,
            num_layers=spec.num_layers,
            expected_runtime_stack=expected_stack,
        )


@pytest.fixture
def qualification_outcome_inputs(tmp_path: Path) -> dict:
    runner = tmp_path / "runner"
    bundle = tmp_path / "bundle.trtfb"
    runner.write_bytes(b"runner")
    bundle.write_bytes(b"bundle")
    runner_evidence_root = tmp_path / "runner-evidence"
    runtime_capture = runner_evidence_root / "runtime-case" / "base"
    runtime_capture.mkdir(parents=True)
    runtime_trace = _attributed_peak_trace(runtime_capture)
    measured_source = Path(runtime_trace["logits_artifact"]["path"])
    measured_capture = runtime_capture / "runner-logits.bin"
    measured_source.rename(measured_capture)
    runtime_trace["logits_artifact"]["path"] = str(measured_capture)
    cold_source = Path(runtime_trace["cold_start_logits_artifact"]["path"])
    cold_capture = runtime_capture / "runner-logits.bin.cold-start.bin"
    cold_source.rename(cold_capture)
    runtime_trace["cold_start_logits_artifact"]["path"] = str(cold_capture)
    persisted_warmup_evidence = _validate_warmup_evidence(runtime_trace)
    canonical_cases = (
        qualify.Case("runtime-case", 127, 1),
        qualify.Case(
            "admission-case",
            2_049,
            0,
            expect_admission_rejection=True,
        ),
    )
    assert _persisted_case_warmup_evidence_passed(
        persisted_warmup_evidence,
        trace=runtime_trace,
        case=qualify.Case("runtime-case", 127, 1),
    )
    runtime_tokens = qualify.deterministic_token_ids(127, TEST_SPEC.vocab_size)
    runtime_case = canonical_cases[0]
    _write_test_runner_capture(
        runtime_capture,
        command=qualify._runner_command(
            runner=runner,
            bundle=bundle,
            token_path=runtime_capture / "tokens.txt",
            logits_path=runtime_capture / "runner-logits.bin",
            case=runtime_case,
            context_limit=TEST_GEOMETRY.model_context_limit,
        ),
        tokens=runtime_tokens,
        trace=runtime_trace,
        returncode=0,
        sampler=TEST_SAMPLER,
        include_logits=True,
    )
    report_trt_path = tmp_path / "runtime-case.trt-logits.bin"
    report_trt_path.write_bytes(measured_capture.read_bytes())
    report_hf_path = tmp_path / "runtime-case.hf-logits.npy"
    runtime_logits = qualify.read_logits_artifact(measured_capture)
    np.save(report_hf_path, runtime_logits)
    thresholds = {
        "logit_atol": 0.0,
        "logit_cosine_p5": 1.0,
        "logit_rel_l2_p95": 0.0,
        "stable_margin": 0.0,
        "stable_top1_match_rate": 1.0,
        "token_agreement_rate": 1.0,
        "unstable_topk_hit_rate": 1.0,
    }
    runtime_parity = qualify.compare_logits(
        runtime_logits,
        runtime_logits.copy(),
        runtime_trace["selected_token_ids"],
        thresholds,
    )
    runtime_parity["status"] = "passed"

    admission_sampler = qualify.SamplerTrustAnchor(
        pid=124,
        cuda_logical_device_index=0,
        physical_device_index=7,
        pci_bus_id="0000:01:00.0",
        gpu_uuid="GPU-01234567-89ab-cdef-0123-456789abcdef",
    )
    admission_capture = runner_evidence_root / "admission-case" / "base"
    admission_capture.mkdir(parents=True)
    admission_trace = {
        "status": "rejected",
        "error_type": "admission",
        "stage": "before_attention",
        "attention_started": False,
        "prefill_launches": 0,
        "decode_launches": 0,
        "final_kv_position": 0,
        "invocations": [],
        "selected_token_ids": [],
        "step_top1_token_ids": [],
        "attention_execution_ledger": {
            "source": (
                "runtime_memory_transfer_snapshot_v1."
                "execution_attempt_events"
            ),
            "available": True,
            "module_count": 2,
            "before": 7,
            "after": 7,
            "delta": 0,
        },
    }
    admission_tokens = qualify.deterministic_token_ids(
        2_049,
        TEST_SPEC.vocab_size,
    )
    admission_case = canonical_cases[1]
    _write_test_runner_capture(
        admission_capture,
        command=qualify._runner_command(
            runner=runner,
            bundle=bundle,
            token_path=admission_capture / "tokens.txt",
            logits_path=admission_capture / "runner-logits.bin",
            case=admission_case,
            context_limit=TEST_GEOMETRY.model_context_limit,
        ),
        tokens=admission_tokens,
        trace=admission_trace,
        returncode=3,
        sampler=admission_sampler,
        include_logits=False,
    )
    case_reports = (
        {
            "name": "runtime-case",
            "execution_passed": True,
            "passed": True,
            "trace": runtime_trace,
            "runner_evidence": {
                "base": str(runtime_capture),
            },
            "trt_logits_artifact": str(report_trt_path),
            "trt_logits_sha256": qualify._sha256(report_trt_path),
            "hf_logits_artifact": str(report_hf_path),
            "hf_logits_sha256": qualify._sha256(report_hf_path),
            "warmup_evidence": {
                "status": "passed",
                "passed": True,
                "base": persisted_warmup_evidence,
            },
            "parity": runtime_parity,
        },
        {
            "name": "admission-case",
            "execution_passed": True,
            "passed": True,
            "admission_rejected_before_attention": True,
            "trace": admission_trace,
            "runner_evidence": {
                "base": str(admission_capture),
            },
            "warmup_evidence": {
                "status": "not_applicable",
                "reason": "warmup protocol is disabled for admission rejection",
            },
            "parity": {
                "status": "not_applicable",
            },
        },
    )
    clean_source = {
        "git_head": "a" * 40,
        "git_dirty": False,
        "source_state_sha256": "b" * 64,
        "exact_head_gate_satisfied": True,
    }
    return {
        "canonical_cases": canonical_cases,
        "selected_cases": canonical_cases,
        "case_reports": case_reports,
        "skip_hf": False,
        "case_filter_used": False,
        "source_state_pre": clean_source,
        "source_state_post": copy.deepcopy(clean_source),
        "context_memory_envelope": {
            "status": "passed",
            "passed": True,
            "coverage_required": True,
            "gates": {
                "all_points_within_o_c_times_a_envelope": True,
                "all_points_below_materialized_score_bound": True,
                "coverage": {
                    "has_prefill_and_decode": True,
                    "reaches_model_context_limit": True,
                    "has_at_least_three_active_lengths": True,
                },
            },
        },
        "qualified_engine_graph": {
            "passed": True,
            "runtime_stack": {
                "sm": "sm103",
                "tensorrt": "11.2.0.113",
            },
            "gates": {
                "actual_split_engine_sections": True,
                "native_segmented_attention_covers_full_model": True,
            },
        },
        "trusted_geometry": TEST_GEOMETRY,
        "sampler_anchors": {
            "runtime-case/base": TEST_SAMPLER,
            "admission-case/base": admission_sampler,
        },
        "trusted_variant_geometry": None,
        "model_spec": TEST_SPEC,
        "runner": runner,
        "bundle": bundle,
        "runner_evidence_root": runner_evidence_root,
        "thresholds": thresholds,
        "variant_bundle": None,
        "variant_build_receipt": None,
        "qualified_variant_engine_graph": None,
    }


def _attach_base_artifact_binding(
    inputs: dict,
    tmp_path: Path,
) -> None:
    provenance_root = tmp_path / "base-provenance"
    manifest, _ = _write_exact_build_manifest(
        provenance_root,
        source_state=inputs["source_state_pre"],
    )
    _, header = _write_base_bundle(inputs["bundle"])
    receipt = _write_base_build_receipt(
        provenance_root,
        manifest_path=manifest,
        bundle=inputs["bundle"],
        header=header,
        source_state=inputs["source_state_pre"],
    )
    inputs["runner"] = (
        provenance_root
        / "build"
        / "trtmc_dynamic_memory_qualify"
    )
    for case in inputs["selected_cases"]:
        evidence_dir = (
            inputs["runner_evidence_root"] / case.name / "base"
        )
        command = qualify._runner_command(
            runner=inputs["runner"],
            bundle=inputs["bundle"],
            token_path=evidence_dir / "tokens.txt",
            logits_path=evidence_dir / "runner-logits.bin",
            case=case,
            context_limit=TEST_GEOMETRY.model_context_limit,
        )
        (evidence_dir / "command.json").write_text(
            json.dumps(command, indent=2) + "\n",
            encoding="utf-8",
        )
        qualify._write_runner_capture_manifest(
            evidence_dir,
            child_pid=inputs["sampler_anchors"][
                f"{case.name}/base"
            ].pid,
            sampler_anchor=inputs["sampler_anchors"][
                f"{case.name}/base"
            ],
            include_logits=not case.expect_admission_rejection,
        )
    inputs["base_artifact_binding"] = (
        qualify._validate_base_artifact_binding(
            build_manifest_path=manifest,
            base_build_receipt_path=receipt,
            bundle=inputs["bundle"],
            runner=inputs["runner"],
            spec=inputs["model_spec"],
            source_state=inputs["source_state_pre"],
        )
    )


def test_qualification_outcome_without_c_div_2_is_diagnostic_only(
    qualification_outcome_inputs: dict,
) -> None:
    result = qualify.evaluate_qualification_outcome(**qualification_outcome_inputs)

    assert not result["passed"]
    assert not result["promotion_eligible"]
    assert result["diagnostic_passed"]
    assert result["execution_passed"]
    assert result["status"] == "diagnostic_passed"
    assert not result["qualification_gates"][
        "c_div_2_variant_engine_graph_passed"
    ]
    assert not result["qualification_gates"][
        "c_div_2_variant_producer_receipt_passed"
    ]
    assert not result["qualification_gates"][
        "base_artifact_binding_passed"
    ]
    assert not result["qualification_gates"][
        "runtime_kv_plugin_binding_passed"
    ]


def test_qualification_outcome_accepts_replayed_base_artifact_binding(
    qualification_outcome_inputs: dict,
    tmp_path: Path,
) -> None:
    _attach_base_artifact_binding(
        qualification_outcome_inputs,
        tmp_path,
    )

    result = qualify.evaluate_qualification_outcome(
        **qualification_outcome_inputs
    )

    assert result["qualification_gates"][
        "base_artifact_binding_passed"
    ]
    assert not result["qualification_gates"][
        "runtime_kv_plugin_binding_passed"
    ]
    assert not result["passed"]
    assert not result["promotion_eligible"]
    assert result["diagnostic_passed"]


@pytest.mark.parametrize(
    "mutation",
    (
        "command",
        "returncode",
        "tokens",
        "normalized-trace",
        "report-trace",
        "captured-logits",
    ),
)
def test_qualification_outcome_replays_raw_runner_capture_fail_closed(
    qualification_outcome_inputs: dict,
    mutation: str,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    evidence_dir = (
        inputs["runner_evidence_root"] / "runtime-case" / "base"
    )
    if mutation == "command":
        command_path = evidence_dir / "command.json"
        command = json.loads(command_path.read_text(encoding="utf-8"))
        command.append("--tampered")
        command_path.write_text(
            json.dumps(command, indent=2) + "\n",
            encoding="utf-8",
        )
    elif mutation == "returncode":
        (evidence_dir / "returncode.txt").write_text(
            "1\n",
            encoding="utf-8",
        )
    elif mutation == "tokens":
        token_path = evidence_dir / "tokens.txt"
        tokens = token_path.read_text(encoding="utf-8").splitlines()
        tokens[0] = str(int(tokens[0]) + 1)
        token_path.write_text(
            "\n".join(tokens) + "\n",
            encoding="utf-8",
        )
    elif mutation == "normalized-trace":
        normalized_path = evidence_dir / "runner-trace.json"
        normalized = json.loads(
            normalized_path.read_text(encoding="utf-8")
        )
        normalized["prompt_tokens"] += 1
        normalized_path.write_text(
            json.dumps(normalized, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif mutation == "report-trace":
        inputs["case_reports"][0]["trace"]["prompt_tokens"] += 1
    elif mutation == "captured-logits":
        logits_path = evidence_dir / "runner-logits.bin"
        payload = bytearray(logits_path.read_bytes())
        payload[-1] ^= 1
        logits_path.write_bytes(payload)
    else:  # pragma: no cover - additions must remain explicit.
        raise AssertionError(mutation)

    if mutation != "report-trace":
        qualify._write_runner_capture_manifest(
            evidence_dir,
            child_pid=TEST_SAMPLER.pid,
            sampler_anchor=TEST_SAMPLER,
            include_logits=True,
        )

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["qualification_gates"][
        "raw_runner_evidence_passed"
    ]
    assert (
        result["raw_runner_evidence"]["case_states"]["runtime-case"]
        == "failed"
    )
    assert not result["passed"]


def test_qualification_outcome_recomputes_hf_parity_from_hashed_artifact(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    report = inputs["case_reports"][0]
    hf_path = Path(report["hf_logits_artifact"])
    hf_logits = np.load(hf_path, allow_pickle=False)
    hf_logits[0, 0] += 100.0
    np.save(hf_path, hf_logits)
    # Even a self-consistent new artifact hash and stale "passed" summary
    # cannot bypass the independent parity replay.
    report["hf_logits_sha256"] = qualify._sha256(hf_path)
    assert report["parity"]["status"] == "passed"

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert result["qualification_gates"]["raw_runner_evidence_passed"]
    assert not result["qualification_gates"][
        "hf_parity_executed_and_passed"
    ]
    assert result["parity_execution"]["runtime-case"] == "failed"
    assert not result["passed"]


def test_qualification_outcome_replays_chunk_variant_artifacts(
    qualification_outcome_inputs: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    _attach_base_artifact_binding(
        inputs,
        inputs["runner_evidence_root"].parent,
    )
    inputs["runtime_kv_plugin_binding"] = {}
    monkeypatch.setattr(
        qualify,
        "_runtime_kv_plugin_binding_passed",
        lambda _evidence, *, base_artifact_binding: (
            base_artifact_binding is not None
        ),
    )
    monkeypatch.setattr(
        qualify,
        "_persisted_profile_sweep_evidence_passed",
        lambda _evidence, *, runner, bundle, spec: True,
    )
    monkeypatch.setattr(
        qualify,
        "_persisted_source_calibration_evidence_passed",
        lambda _evidence, *, runner, bundle, spec: True,
    )
    inputs["profile_sweep_evidence"] = {
        "base": {"passed": True},
        "chunk_variant": {"passed": True},
    }
    case = inputs["selected_cases"][0]
    report = inputs["case_reports"][0]
    inputs["canonical_cases"] = (case,)
    inputs["selected_cases"] = (case,)
    inputs["case_reports"] = (report,)
    variant_bundle = inputs["runner_evidence_root"].parent / "variant.trtfb"
    variant_bundle.write_bytes(b"variant")
    variant_geometry = qualify.TrustedRuntimeGeometry(
        model_context_limit=TEST_GEOMETRY.model_context_limit,
        prefill_chunk_limit=TEST_GEOMETRY.prefill_chunk_limit // 2,
        kv_bytes_per_token=TEST_GEOMETRY.kv_bytes_per_token,
    )
    variant_sampler = TEST_SAMPLER
    evidence_dir = (
        inputs["runner_evidence_root"]
        / case.name
        / "chunk-variant"
    )
    evidence_dir.mkdir()
    base_trace = report["trace"]
    variant_trace = copy.deepcopy(base_trace)
    variant_trace["prefill_chunk_limit"] = (
        variant_geometry.prefill_chunk_limit
    )
    for lifetime in (
        variant_trace["load_cycle_warmup"],
        variant_trace["load_cycles"][0],
    ):
        lifetime["runtime_memory_receipt"]["prefill_chunk_limit"] = (
            variant_geometry.prefill_chunk_limit
        )
    measured_path = evidence_dir / "runner-logits.bin"
    cold_path = evidence_dir / "runner-logits.bin.cold-start.bin"
    measured_path.write_bytes(
        Path(base_trace["logits_artifact"]["path"]).read_bytes()
    )
    cold_path.write_bytes(
        Path(base_trace["cold_start_logits_artifact"]["path"]).read_bytes()
    )
    variant_trace["logits_artifact"]["path"] = str(measured_path)
    variant_trace["cold_start_logits_artifact"]["path"] = str(cold_path)
    tokens = qualify.deterministic_token_ids(
        case.prompt_tokens,
        TEST_SPEC.vocab_size,
    )
    _write_test_runner_capture(
        evidence_dir,
        command=qualify._runner_command(
            runner=inputs["runner"],
            bundle=variant_bundle,
            token_path=evidence_dir / "tokens.txt",
            logits_path=measured_path,
            case=case,
            context_limit=variant_geometry.model_context_limit,
        ),
        tokens=tokens,
        trace=variant_trace,
        returncode=0,
        sampler=variant_sampler,
        include_logits=True,
    )
    variant_logits = qualify.read_logits_artifact(measured_path)
    variant_validation = qualify._validate_trace(
        case,
        TEST_SPEC,
        variant_trace,
        variant_logits,
        expected_chunk_limit=variant_geometry.prefill_chunk_limit,
        trusted_geometry=variant_geometry,
        expected_sampler=variant_sampler,
        require_nvml_reconciliation=True,
    )
    assert variant_validation is not None
    variant_report_artifact = (
        inputs["runner_evidence_root"].parent
        / "runtime-case.variant.trt-logits.bin"
    )
    variant_report_artifact.write_bytes(measured_path.read_bytes())
    base_logits = qualify.read_logits_artifact(
        Path(base_trace["logits_artifact"]["path"])
    )
    chunk_parity = qualify.compare_logits(
        variant_logits,
        base_logits,
        variant_trace["selected_token_ids"],
        inputs["thresholds"],
    )
    report["runner_evidence"]["chunk_variant"] = str(evidence_dir)
    report["warmup_evidence"]["chunk_variant"] = variant_validation[
        "warmup_evidence"
    ]
    report["chunk_variant"] = {
        "passed": True,
        "execution_passed": True,
        "trace": variant_trace,
        "warmup_evidence": variant_validation["warmup_evidence"],
        "base_vs_variant_parity": chunk_parity,
        "trt_logits_artifact": str(variant_report_artifact),
        "trt_logits_sha256": qualify._sha256(variant_report_artifact),
    }
    inputs["variant_bundle"] = variant_bundle
    inputs["trusted_variant_geometry"] = variant_geometry
    inputs["qualified_variant_engine_graph"] = copy.deepcopy(
        inputs["qualified_engine_graph"]
    )
    inputs["variant_build_receipt"] = (
        _validated_variant_receipt_summary(
            inputs["runner_evidence_root"].parent,
            variant_bundle=variant_bundle,
            source_state=inputs["source_state_pre"],
            plugin=inputs["runner"],
        )
    )
    inputs["sampler_anchors"][
        f"{case.name}/chunk-variant"
    ] = variant_sampler

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert result["qualification_gates"]["raw_runner_evidence_passed"]
    assert result["qualification_gates"][
        "runtime_kv_plugin_binding_passed"
    ]
    assert result["qualification_gates"][
        "source_calibration_evidence_reopened"
    ]
    assert result["passed"]
    assert result["promotion_eligible"]

    monkeypatch.setattr(
        qualify,
        "_persisted_source_calibration_evidence_passed",
        lambda _evidence, *, runner, bundle, spec: False,
    )
    missing_source = qualify.evaluate_qualification_outcome(**inputs)
    assert not missing_source["qualification_gates"][
        "source_calibration_evidence_reopened"
    ]
    assert "source_calibration_evidence_reopened" in missing_source[
        "qualification_blockers"
    ]
    assert not missing_source["passed"]
    assert not missing_source["promotion_eligible"]
    monkeypatch.setattr(
        qualify,
        "_persisted_source_calibration_evidence_passed",
        lambda _evidence, *, runner, bundle, spec: True,
    )

    validated_receipt = inputs["variant_build_receipt"]
    inputs["variant_build_receipt"] = None
    missing_receipt = qualify.evaluate_qualification_outcome(**inputs)
    assert not missing_receipt["qualification_gates"][
        "c_div_2_variant_producer_receipt_passed"
    ]
    assert not missing_receipt["passed"]
    assert missing_receipt["diagnostic_passed"]
    inputs["variant_build_receipt"] = validated_receipt

    # A self-consistent report summary cannot hide a raw variant logit change.
    payload = bytearray(measured_path.read_bytes())
    payload[-1] ^= 1
    measured_path.write_bytes(payload)
    qualify._write_runner_capture_manifest(
        evidence_dir,
        child_pid=variant_sampler.pid,
        sampler_anchor=variant_sampler,
        include_logits=True,
    )
    result = qualify.evaluate_qualification_outcome(**inputs)
    assert not result["qualification_gates"]["raw_runner_evidence_passed"]
    assert not result["passed"]


def test_qualification_outcome_marks_skip_hf_as_diagnostic_only(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    inputs["skip_hf"] = True
    inputs["case_reports"][0]["parity"] = {
        "status": "not_run",
        "reason": "--skip-hf was requested",
    }
    # A stale case-level flag must never promote a skipped parity run.
    inputs["case_reports"][0]["passed"] = True

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["passed"]
    assert result["diagnostic_passed"]
    assert result["status"] == "diagnostic_passed"
    assert not result["qualification_gates"]["hf_parity_executed_and_passed"]
    assert result["parity_execution"]["runtime-case"] == "not_run"


def test_qualification_outcome_marks_any_case_filter_as_diagnostic_only(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    # Even an explicit filter that happens to name the complete matrix is not
    # a canonical unfiltered qualification invocation.
    inputs["case_filter_used"] = True

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert result["qualification_gates"]["canonical_matrix_complete"]
    assert not result["qualification_gates"]["case_filter_not_used"]
    assert not result["passed"]
    assert result["diagnostic_passed"]
    assert result["status"] == "diagnostic_passed"


def test_qualification_outcome_requires_full_context_memory_coverage(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    envelope = inputs["context_memory_envelope"]
    envelope["coverage_required"] = False
    envelope["gates"]["coverage"]["reaches_model_context_limit"] = False

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["qualification_gates"]["full_context_memory_coverage"]
    assert not result["passed"]
    assert result["diagnostic_passed"]
    assert result["status"] == "diagnostic_passed"


def test_qualification_outcome_marks_dirty_unchanged_source_diagnostic_only(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    for snapshot in (
        inputs["source_state_pre"],
        inputs["source_state_post"],
    ):
        snapshot["git_dirty"] = True
        snapshot["exact_head_gate_satisfied"] = False

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert result["qualification_gates"]["source_state_unchanged"]
    assert not result["qualification_gates"]["source_clean_exact_head"]
    assert not result["passed"]
    assert result["diagnostic_passed"]
    assert result["status"] == "diagnostic_passed"


def test_qualification_outcome_rejects_source_drift(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    inputs["source_state_post"]["source_state_sha256"] = "c" * 64

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["qualification_gates"]["source_state_unchanged"]
    assert not result["passed"]
    assert not result["diagnostic_passed"]
    assert result["status"] == "failed"


def test_qualification_outcome_rejects_false_graph_gate(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    inputs["qualified_engine_graph"]["passed"] = False
    inputs["qualified_engine_graph"]["gates"]["native_segmented_attention_covers_full_model"] = (
        False
    )

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["qualification_gates"]["qualified_engine_graph_passed"]
    assert not result["passed"]
    assert not result["diagnostic_passed"]
    assert result["status"] == "failed"


def test_qualification_outcome_requires_explicit_warmup_evidence(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    del inputs["case_reports"][0]["warmup_evidence"]

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["qualification_gates"]["warmup_evidence_passed"]
    assert result["warmup_evidence"]["case_states"]["runtime-case"] == "failed"
    assert not result["passed"]


def test_qualification_outcome_requires_raw_pre_attention_admission_evidence(
    qualification_outcome_inputs: dict,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    admission = inputs["case_reports"][1]
    admission["trace"]["lifetime_protocol"] = {
        "schema_version": 1,
        "execution_order": ["warmup", "measured"],
        "warmup_count": 1,
        "measured_count": 1,
    }

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["qualification_gates"][
        "admission_rejection_evidence_passed"
    ]
    assert (
        result["admission_rejection_evidence"]["case_states"]["admission-case"]
        == "failed"
    )
    assert not result["passed"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("invocations", [{"role": "prefill"}]),
        ("selected_token_ids", [17]),
        ("step_top1_token_ids", [17]),
        ("final_kv_position", 1),
        (
            "attention_execution_ledger",
            {
                "source": (
                    "runtime_memory_transfer_snapshot_v1."
                    "execution_attempt_events"
                ),
                "available": True,
                "module_count": 2,
                "before": 7,
                "after": 8,
                "delta": 1,
            },
        ),
    ),
)
def test_qualification_outcome_rejects_contradictory_admission_payloads(
    qualification_outcome_inputs: dict,
    field: str,
    value: object,
) -> None:
    inputs = copy.deepcopy(qualification_outcome_inputs)
    inputs["case_reports"][1]["trace"][field] = value

    result = qualify.evaluate_qualification_outcome(**inputs)

    assert not result["qualification_gates"][
        "admission_rejection_evidence_passed"
    ]
    assert (
        result["admission_rejection_evidence"]["case_states"]["admission-case"]
        == "failed"
    )
    assert not result["passed"]


def test_inspector_layer_classification_ignores_container_text() -> None:
    inspector = {
        "Metadata": "container mentions concat cache_k_0 attention_scores",
        "Layers": [
            {
                "Name": "layer.0.cache_concat",
                "LayerType": "Concatenation",
                "Inputs": [{"Name": "cache_k_0"}],
            },
            {
                "Name": "layer.1.attn.attention_scores",
                "LayerType": "MatrixMultiply",
            },
        ],
    }

    assert qualify._cache_concat_layers(inspector) == ["layer.0.cache_concat"]
    assert qualify._dense_attention_layers(inspector) == ["layer.1.attn.attention_scores"]


def test_c_div_2_variant_may_change_only_internal_chunk_policy() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    contract = {
        "contract_version": 1,
        "qualified_model_id": spec.model_id,
        "qualified_model_revision": "1" * 40,
        "qualified_config_sha256": "2" * 64,
        "qualified_target": "gb300-trt-11.2",
        "qualified_runtime_stack": {
            "sm": "sm103",
            "tensorrt": "11.2.0.113",
            "cuda_runtime": "13.3",
            "cudnn_backend": "9.20.0",
            "cudnn_frontend_revision": "7b9b711c22b6823e87150213ecd8449260db8610",
            "nvrtc": "13.3",
            "driver": "580.105.08",
        },
        "native_kv_plugin_abi": 2,
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "kv_layout": "contiguous_runtime_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 114_688,
        "active_kv_profile_limits": list(spec.buckets),
        "runtime_owned": True,
    }
    base = {"vocab_size": 151_936, "runtime_memory": contract}
    variant_contract = dict(contract)
    variant_contract["prefill_chunk_limit"] = spec.chunk_limit // 2
    variant_contract["active_kv_profile_limits"] = list(spec.buckets)
    variant = {
        "vocab_size": base["vocab_size"],
        "runtime_memory": variant_contract,
    }

    assert qualify._validate_chunk_variant(base, variant, spec) == spec.chunk_limit // 2

    variant_contract["qualified_model_revision"] = "3" * 40
    with pytest.raises(ValueError, match="changes qualified model facts"):
        qualify._validate_chunk_variant(base, variant, spec)


def _qwen_chunk_variant_headers() -> tuple[dict, dict]:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    contract = {
        "contract_version": 1,
        "qualified_model_id": spec.model_id,
        "qualified_model_revision": "1" * 40,
        "qualified_config_sha256": "2" * 64,
        "qualified_target": "gb300-trt-11.2",
        "qualified_runtime_stack": {
            "sm": "sm103",
            "tensorrt": "11.2.0.113",
            "cuda_runtime": "13.3",
            "cudnn_backend": "9.20.0",
            "cudnn_frontend_revision": "7b9b711c22b6823e87150213ecd8449260db8610",
            "nvrtc": "13.3",
            "driver": "580.105.08",
        },
        "native_kv_plugin_abi": 2,
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "kv_layout": "contiguous_runtime_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": 114_688,
        "active_kv_profile_limits": list(spec.buckets),
        "runtime_owned": True,
    }
    variant_contract = dict(contract)
    variant_contract["prefill_chunk_limit"] = spec.chunk_limit // 2
    variant_contract["active_kv_profile_limits"] = sorted({*spec.buckets, spec.chunk_limit // 2})
    return (
        {"vocab_size": 151_936, "runtime_memory": contract},
        {"vocab_size": 151_936, "runtime_memory": variant_contract},
    )


def _file_identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _binary_identity(path: Path) -> dict:
    observed = path.stat()
    return {
        "path": str(path.resolve()),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "size_bytes": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _validated_variant_receipt_summary(
    root: Path,
    *,
    variant_bundle: Path,
    source_state: dict,
    plugin: Path,
) -> dict:
    receipt = root / "chunk-variant-build-receipt.json"
    receipt.write_text('{"passed":true}\n', encoding="utf-8")
    timing = root / "chunk-variant-build-timing.json"
    timing.write_text('{"elapsed_ns":1}\n', encoding="utf-8")
    manifest = root / "chunk-variant-build-manifest.json"
    manifest.write_text('{"passed":true}\n', encoding="utf-8")
    plugin_identity = _binary_identity(plugin)
    return {
        **_file_identity(receipt),
        "schema_version": qualify.CHUNK_VARIANT_BUILD_SCHEMA,
        "bundle": _file_identity(variant_bundle),
        "producer": _file_identity(
            REPO_ROOT
            / "tools"
            / "build_native_dynamic_memory_chunk_variant.py"
        ),
        "runtime_kv_plugin": plugin_identity,
        "runtime_kv_plugin_mapping": {
            "candidate_count": 1,
            "deleted_candidate_count": 0,
            "selected": plugin_identity,
        },
        "build_manifest": {
            "path": str(manifest.resolve()),
            "sha256": qualify._sha256(manifest),
            "schema_version": qualify.BUILD_MANIFEST_SCHEMA,
            "git_head": source_state["git_head"],
            "source_state_sha256": source_state[
                "source_state_sha256"
            ],
            "build_artifacts_sha256": "1" * 64,
        },
        "build_timing": _file_identity(timing),
        "source_state_sha256": source_state["source_state_sha256"],
        "git_head": source_state["git_head"],
    }


def _manifest_artifact_identity(
    path: Path, *, key: str, relative: str
) -> dict:
    observed = path.stat()
    return {
        "artifact_key": key,
        "relative_path": relative,
        "path": str(path.resolve()),
        "st_dev": observed.st_dev,
        "st_ino": observed.st_ino,
        "size_bytes": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
        "mode": stat.S_IMODE(observed.st_mode),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _write_exact_build_manifest(
    tmp_path: Path,
    *,
    source_state: dict,
    repo_root: Path = REPO_ROOT,
) -> tuple[Path, Path]:
    repo_root = repo_root.resolve()
    build = tmp_path / "build"
    relative_paths = {
        "trtmc": "trtmc",
        "benchmark_worker": "trtmc_benchmark_worker",
        "core": "libtrtmc_core.so",
        "trt_backend": "libtrtmc_backend_trt.so",
        "runtime_kv_plugin": "libtrtmc_trt_plugins.so",
        "model_qwen": "models/qwen/libtrtmc_model_qwen.so",
        "model_llama": "models/llama/libtrtmc_model_llama.so",
        "qualify": "trtmc_dynamic_memory_qualify",
        "nvrtc_optional_output_regression": (
            "trtmc_nvrtc_optional_output_regression"
        ),
        "surfaces": "trtmc_dynamic_memory_surfaces",
    }
    for key, relative in relative_paths.items():
        path = build / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"test-{key}".encode())
        if key in {"trtmc", "qualify"}:
            path.chmod(0o755)
    cache = build / "CMakeCache.txt"
    cache.write_text(
        (
            f"CMAKE_HOME_DIRECTORY:INTERNAL={repo_root}\n"
            "TRTMC_TRT_BACKEND_ABI:STRING=11_2\n"
        ),
        encoding="utf-8",
    )
    cmake_cache = _manifest_artifact_identity(
        cache, key="cmake_cache", relative="CMakeCache.txt"
    )
    cmake_cache["configured_source"] = str(repo_root)
    active_backend = build / "libtrtmc_backend_trt_11_2.so"
    active_backend.symlink_to("libtrtmc_backend_trt.so")
    artifact_paths = dict(relative_paths)
    artifact_paths["trt_backend"] = "libtrtmc_backend_trt_11_2.so"
    artifacts = {
        key: _manifest_artifact_identity(
            build / relative, key=key, relative=relative
        )
        for key, relative in artifact_paths.items()
    }
    manifest_module = load_manifest_module(REPO_ROOT)
    commands = complete_command_receipts(
        manifest_module,
        repo_root=repo_root,
        build_dir=build,
        output_dir=build,
        python=Path(sys.executable),
    )
    manifest = {
        "schema_version": qualify.BUILD_MANIFEST_SCHEMA,
        "repo_root": str(repo_root),
        "build_dir": str(build.resolve()),
        "python": sys.executable,
        "source_state_pre": {
            **source_state,
            "git_dirty": False,
            "exact_head_gate_satisfied": True,
        },
        "commands": commands,
        "passed": True,
        "source_state_post": {
            **source_state,
            "git_dirty": False,
            "exact_head_gate_satisfied": True,
        },
        "source_state_unchanged": True,
        "cmake_cache": cmake_cache,
        "build_artifacts": artifacts,
        "build_artifacts_sha256": hashlib.sha256(
            json.dumps(
                artifacts,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "clean_build_command_sha256": hashlib.sha256(
            json.dumps(
                commands[0],
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    path = build / "build-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, build / relative_paths["runtime_kv_plugin"]


def _write_base_bundle(
    path: Path,
    *,
    spec=TEST_SPEC,
) -> tuple[Path, dict]:
    runtime_stack = {
        "sm": "sm103",
        "tensorrt": "11.2.0.113",
        "cuda_runtime": "13.3",
        "cudnn_backend": "9.20.0",
        "cudnn_frontend_revision": (
            "7b9b711c22b6823e87150213ecd8449260db8610"
        ),
        "nvrtc": "13.3",
        "driver": "580.105.08",
    }
    plans = [
        {
            "section_name": "engine_plan",
            "section_sha256": "c" * 64,
            "role": "decode",
            "optimization_profile_count": len(spec.buckets),
        },
        {
            "section_name": "prefill_engine_plan",
            "section_sha256": "d" * 64,
            "role": "prefill",
            "optimization_profile_count": 1,
        },
    ]
    contract = {
        "contract_version": 2,
        "qualified_model_id": spec.model_id,
        "qualified_model_revision": "1" * 40,
        "qualified_config_sha256": "2" * 64,
        "qualified_target": "gb300-trt-11.2",
        "qualified_runtime_stack": runtime_stack,
        "native_kv_plugin_abi": 2,
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "kv_layout": "contiguous_runtime_v1",
        "kv_dtype": spec.kv_dtype,
        "kv_bytes_per_token": spec.kv_bytes_per_token,
        "active_kv_profile_limits": list(spec.buckets),
        "runtime_owned": True,
        "module_residency_calibration": {
            "schema_version": 1,
            "measurement_kind": "nvml_process_cumulative_first_use",
            "cuda_module_loading_mode": "lazy",
            "qualified_runtime_stack_sha256": (
                qualify.qualified_runtime_stack_sha256(runtime_stack)
            ),
            "plan_set_sha256": (
                qualify.module_residency_plan_set_sha256(plans)
            ),
            "plans": plans,
            "profile_reserves": [
                {
                    "covering_profile_limit": limit,
                    "cumulative_reserve_bytes": (index + 1) * 1024,
                }
                for index, limit in enumerate(spec.buckets)
            ],
            "evidence_sha256": "e" * 64,
        },
    }
    header = {
        "model_id": spec.model_id,
        "precision": "bf16",
        "vocab_size": spec.vocab_size,
        "runtime_memory": contract,
    }
    payload = json.dumps(header, sort_keys=True).encode("utf-8")
    path.write_bytes(
        qualify.BUNDLE_MAGIC
        + struct.pack("<Q", len(payload))
        + payload
    )
    return path, header


def _write_base_build_receipt(
    tmp_path: Path,
    *,
    manifest_path: Path,
    bundle: Path,
    header: dict,
    source_state: dict,
) -> Path:
    perf = qualify._load_perf_provenance_module()
    manifest_binding, artifacts = perf._read_build_manifest(
        manifest_path
    )
    trtmc = Path(artifacts["trtmc"]["path"])
    plugin = artifacts["runtime_kv_plugin"]
    stdout = tmp_path / "base-build.stdout.log"
    stderr = tmp_path / "base-build.stderr.log"
    stdout.write_text("built\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    command = [str(trtmc), "build", header["model_id"]]
    mtime_ns = bundle.stat().st_mtime_ns
    receipt = {
        "schema_version": perf.BUILD_SCHEMA,
        "artifact_role": "native-dynamic",
        "model_id": header["model_id"],
        "model_revision": header["runtime_memory"][
            "qualified_model_revision"
        ],
        "precision": header["precision"],
        "target": header["runtime_memory"]["qualified_target"],
        "bundle_build_id": "base-build-1",
        "fresh_build": True,
        "artifact_reused": False,
        "bundle": str(bundle.resolve()),
        "bundle_sha256": qualify._sha256(bundle),
        "bundle_bytes": bundle.stat().st_size,
        "bundle_mtime_ns": mtime_ns,
        "build_started_ns": max(1, mtime_ns - 1),
        "build_finished_ns": mtime_ns + 1,
        "command": command,
        "command_sha256": perf._canonical_sha(command),
        "resolved_command": command,
        "resolved_command_sha256": perf._canonical_sha(command),
        "trtmc_executable": artifacts["trtmc"],
        "cwd": str(tmp_path.resolve()),
        "stdout": str(stdout.resolve()),
        "stdout_sha256": qualify._sha256(stdout),
        "stderr": str(stderr.resolve()),
        "stderr_sha256": qualify._sha256(stderr),
        "git_head": source_state["git_head"],
        "prebuild_source_state_sha256": source_state[
            "source_state_sha256"
        ],
        "postbuild_source_state_sha256": source_state[
            "source_state_sha256"
        ],
        "source_state_pre": dict(source_state),
        "source_state_post": dict(source_state),
        "build_manifest": manifest_binding,
        "runtime_kv_plugin": plugin,
        "runtime_kv_plugin_mapping": {
            "path": plugin["path"],
            "device": plugin["device"],
            "inode": plugin["inode"],
            "deleted": False,
            "identity_sha256": perf._canonical_sha(plugin),
        },
        "mapped_dso_identities": [plugin],
    }
    assert set(receipt) == perf.BUILD_RECEIPT_FIELDS
    path = tmp_path / "base-build-receipt.json"
    path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _base_binding_inputs(
    tmp_path: Path,
    *,
    spec=TEST_SPEC,
) -> tuple[Path, Path, Path, Path, dict]:
    source_state = {
        "git_head": "a" * 40,
        "source_state_sha256": "b" * 64,
        "git_dirty": False,
        "exact_head_gate_satisfied": True,
    }
    manifest, _ = _write_exact_build_manifest(
        tmp_path,
        source_state=source_state,
    )
    bundle, header = _write_base_bundle(
        tmp_path / "base.trtfb",
        spec=spec,
    )
    receipt = _write_base_build_receipt(
        tmp_path,
        manifest_path=manifest,
        bundle=bundle,
        header=header,
        source_state=source_state,
    )
    runner = tmp_path / "build/trtmc_dynamic_memory_qualify"
    return manifest, receipt, bundle, runner, source_state


def test_base_artifact_binding_replays_all_selected_exact_identities(
    tmp_path: Path,
) -> None:
    manifest, receipt, bundle, runner, source_state = (
        _base_binding_inputs(tmp_path)
    )

    binding = qualify._validate_base_artifact_binding(
        build_manifest_path=manifest,
        base_build_receipt_path=receipt,
        bundle=bundle,
        runner=runner,
        spec=TEST_SPEC,
        source_state=source_state,
    )

    assert set(binding) == qualify._BASE_ARTIFACT_BINDING_FIELDS
    assert binding["schema_version"] == (
        qualify.BASE_ARTIFACT_BINDING_SCHEMA
    )
    assert binding["bundle"]["sha256"] == qualify._sha256(bundle)
    assert binding["qualifier_runner"]["path"] == str(runner.resolve())
    assert binding["benchmark_worker"]["path"].endswith(
        "/trtmc_benchmark_worker"
    )
    assert binding["core"]["path"].endswith("/libtrtmc_core.so")
    assert binding["trt_backend"]["active_versioned_path"].endswith(
        "/libtrtmc_backend_trt_11_2.so"
    )
    assert binding["trt_backend"]["identity"]["path"].endswith(
        "/libtrtmc_backend_trt.so"
    )
    assert binding["model_plugin"]["artifact_key"] == "model_llama"
    assert binding["model_plugin"]["identity"]["path"].endswith(
        "/models/llama/libtrtmc_model_llama.so"
    )
    assert binding["runtime_kv_plugin"]["path"].endswith(
        "/libtrtmc_trt_plugins.so"
    )
    assert qualify._base_artifact_binding_passed(
        binding,
        bundle=bundle,
        runner=runner,
        spec=TEST_SPEC,
        source_state=source_state,
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "bundle",
        "runner-inode",
        "source",
        "model-dso-selection",
    ),
)
def test_base_artifact_binding_fails_closed_on_identity_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    manifest, receipt, bundle, runner, source_state = (
        _base_binding_inputs(tmp_path)
    )
    binding = qualify._validate_base_artifact_binding(
        build_manifest_path=manifest,
        base_build_receipt_path=receipt,
        bundle=bundle,
        runner=runner,
        spec=TEST_SPEC,
        source_state=source_state,
    )

    if mutation == "bundle":
        bundle.write_bytes(bundle.read_bytes() + b"changed")
    elif mutation == "runner-inode":
        replacement = tmp_path / "replacement-runner"
        replacement.write_bytes(runner.read_bytes())
        replacement.chmod(0o755)
        os.replace(replacement, runner)
    elif mutation == "source":
        source_state = {
            **source_state,
            "source_state_sha256": "c" * 64,
        }
    elif mutation == "model-dso-selection":
        binding["model_plugin"]["artifact_key"] = "model_qwen"
    else:  # pragma: no cover
        raise AssertionError(mutation)

    assert not qualify._base_artifact_binding_passed(
        binding,
        bundle=bundle,
        runner=runner,
        spec=TEST_SPEC,
        source_state=source_state,
    )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("dirty-prebuild", "not the current clean exact HEAD"),
        ("wrong-target", "model tuple does not match"),
    ),
)
def test_base_artifact_binding_rejects_receipt_contract_drift(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    manifest, receipt, bundle, runner, source_state = (
        _base_binding_inputs(tmp_path)
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if mutation == "dirty-prebuild":
        payload["source_state_pre"]["git_dirty"] = True
        payload["source_state_pre"][
            "exact_head_gate_satisfied"
        ] = False
    elif mutation == "wrong-target":
        payload["target"] = "different-target"
    else:  # pragma: no cover
        raise AssertionError(mutation)
    receipt.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error):
        qualify._validate_base_artifact_binding(
            build_manifest_path=manifest,
            base_build_receipt_path=receipt,
            bundle=bundle,
            runner=runner,
            spec=TEST_SPEC,
            source_state=source_state,
        )


def test_base_artifact_binding_selects_manifest_runtime_plugin_before_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, receipt, bundle, runner, source_state = (
        _base_binding_inputs(tmp_path)
    )
    binding = qualify._validate_base_artifact_binding(
        build_manifest_path=manifest,
        base_build_receipt_path=receipt,
        bundle=bundle,
        runner=runner,
        spec=TEST_SPEC,
        source_state=source_state,
    )
    monkeypatch.delitem(
        sys.modules,
        "tensorrt_model_connect.trt_plugins",
        raising=False,
    )
    previous_plugin_environment = os.environ.pop(
        qualify.RUNTIME_KV_PLUGIN_ENV,
        None,
    )
    try:
        selected = (
            qualify._bind_runtime_kv_plugin_from_base_artifacts(
                binding
            )
        )

        assert not selected["environment_was_set"]
        assert selected["selected"] == binding["runtime_kv_plugin"]
        assert os.environ[qualify.RUNTIME_KV_PLUGIN_ENV] == binding[
            "runtime_kv_plugin"
        ]["path"]
        perf = qualify._load_perf_provenance_module()
        manifest_plugin = binding["runtime_kv_plugin"]
        monkeypatch.setattr(
            perf,
            "_mapped_library_records",
            lambda _pid: (
                {
                    "path": manifest_plugin["path"],
                    "device": manifest_plugin["device"],
                    "inode": manifest_plugin["inode"],
                    "deleted": False,
                },
            ),
        )
        finalized = qualify._finalize_runtime_kv_plugin_binding(
            selected
        )
        assert finalized["loaded_mapping"]["inode"] == (
            manifest_plugin["inode"]
        )
        assert qualify._runtime_kv_plugin_binding_passed(
            finalized,
            base_artifact_binding=binding,
        )
    finally:
        if previous_plugin_environment is None:
            os.environ.pop(qualify.RUNTIME_KV_PLUGIN_ENV, None)
        else:
            os.environ[
                qualify.RUNTIME_KV_PLUGIN_ENV
            ] = previous_plugin_environment


def test_base_artifact_binding_rejects_wrong_explicit_runtime_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, receipt, bundle, runner, source_state = (
        _base_binding_inputs(tmp_path)
    )
    binding = qualify._validate_base_artifact_binding(
        build_manifest_path=manifest,
        base_build_receipt_path=receipt,
        bundle=bundle,
        runner=runner,
        spec=TEST_SPEC,
        source_state=source_state,
    )
    wrong_plugin = tmp_path / "wrong-runtime-kv-plugin.so"
    wrong_plugin.write_bytes(b"different-plugin")
    monkeypatch.delitem(
        sys.modules,
        "tensorrt_model_connect.trt_plugins",
        raising=False,
    )
    monkeypatch.setenv(
        qualify.RUNTIME_KV_PLUGIN_ENV,
        str(wrong_plugin),
    )

    with pytest.raises(
        ValueError,
        match="selects a different runtime-KV plugin",
    ):
        qualify._bind_runtime_kv_plugin_from_base_artifacts(
            binding
        )


def test_base_artifact_binding_rejects_wrong_preloaded_runtime_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, receipt, bundle, runner, source_state = (
        _base_binding_inputs(tmp_path)
    )
    binding = qualify._validate_base_artifact_binding(
        build_manifest_path=manifest,
        base_build_receipt_path=receipt,
        bundle=bundle,
        runner=runner,
        spec=TEST_SPEC,
        source_state=source_state,
    )
    wrong_plugin = tmp_path / "renamed-wrong-plugin.so"
    wrong_plugin.write_bytes(
        b"test\0trtmc_runtime_kv_plugin_abi_version\0"
    )
    wrong_stat = wrong_plugin.stat()
    perf = qualify._load_perf_provenance_module()
    monkeypatch.setattr(
        perf,
        "_mapped_library_records",
        lambda _pid: (
            {
                "path": str(wrong_plugin.resolve()),
                "device": wrong_stat.st_dev,
                "inode": wrong_stat.st_ino,
                "deleted": False,
            },
        ),
    )
    monkeypatch.delitem(
        sys.modules,
        "tensorrt_model_connect.trt_plugins",
        raising=False,
    )
    monkeypatch.delenv(
        qualify.RUNTIME_KV_PLUGIN_ENV,
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="already maps a different runtime-KV plugin",
    ):
        qualify._bind_runtime_kv_plugin_from_base_artifacts(
            binding
        )


def test_runtime_plugin_binding_rejects_wrong_post_load_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, receipt, bundle, runner, source_state = (
        _base_binding_inputs(tmp_path)
    )
    binding = qualify._validate_base_artifact_binding(
        build_manifest_path=manifest,
        base_build_receipt_path=receipt,
        bundle=bundle,
        runner=runner,
        spec=TEST_SPEC,
        source_state=source_state,
    )
    monkeypatch.delitem(
        sys.modules,
        "tensorrt_model_connect.trt_plugins",
        raising=False,
    )
    monkeypatch.delenv(
        qualify.RUNTIME_KV_PLUGIN_ENV,
        raising=False,
    )
    selected = (
        qualify._bind_runtime_kv_plugin_from_base_artifacts(
            binding
        )
    )
    wrong_plugin = tmp_path / "post-load-wrong-plugin.so"
    wrong_plugin.write_bytes(
        b"test\0trtmc_runtime_kv_plugin_abi_version\0"
    )
    wrong_stat = wrong_plugin.stat()
    perf = qualify._load_perf_provenance_module()
    monkeypatch.setattr(
        perf,
        "_mapped_library_records",
        lambda _pid: (
            {
                "path": str(wrong_plugin.resolve()),
                "device": wrong_stat.st_dev,
                "inode": wrong_stat.st_ino,
                "deleted": False,
            },
        ),
    )

    with pytest.raises(
        ValueError,
        match="loaded mapping is not exact",
    ):
        qualify._finalize_runtime_kv_plugin_binding(
            selected
        )


def _plugin_mapping_evidence(identity: dict) -> dict:
    return {
        "schema_version": 1,
        "source": "/proc/self/maps",
        "pid": os.getpid(),
        "selection_rule": (
            "selected_path_or_same_basename_or_exported_abi_symbol"
        ),
        "abi_symbol": qualify.RUNTIME_KV_PLUGIN_ABI_SYMBOL,
        "candidate_count": 1,
        "deleted_candidate_count": 0,
        "selected": dict(identity),
        "candidate_mappings": [
            {
                "path": identity["path"],
                "device": identity["device"],
                "inode": identity["inode"],
            }
        ],
    }


def _write_chunk_variant_receipt(
    tmp_path: Path,
    *,
    source_sha: str = "b" * 64,
) -> tuple[Path, Path, dict, dict, dict]:
    base, variant = _qwen_chunk_variant_headers()
    bundle = tmp_path / "variant.trtfb"
    bundle.write_bytes(b"variant-bundle")
    timing = tmp_path / "timing.json"
    timing.write_text('{"schema_version": 1}\n', encoding="utf-8")
    producer_path = REPO_ROOT / "tools" / "build_native_dynamic_memory_chunk_variant.py"
    source_state = {
        "git_head": "a" * 40,
        "source_state_sha256": source_sha,
    }
    build_manifest, plugin = _write_exact_build_manifest(
        tmp_path, source_state=source_state
    )
    manifest_payload = json.loads(
        build_manifest.read_text(encoding="utf-8")
    )
    build_manifest_binding = {
        "path": str(build_manifest.resolve()),
        "sha256": qualify._sha256(build_manifest),
        "schema_version": qualify.BUILD_MANIFEST_SCHEMA,
        "git_head": source_state["git_head"],
        "source_state_sha256": source_state["source_state_sha256"],
        "build_artifacts_sha256": manifest_payload[
            "build_artifacts_sha256"
        ],
    }
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    receipt = {
        "schema_version": qualify.CHUNK_VARIANT_BUILD_SCHEMA,
        "developer_only": True,
        "fresh_build": True,
        "artifact_reused": False,
        "source_state_unchanged": True,
        "opt_in": {
            "environment": qualify.DEVELOPER_CHUNK_VARIANT_ENV,
            "value": qualify.DEVELOPER_CHUNK_VARIANT_VALUE,
        },
        "builder_entrypoint": (
            "tensorrt_model_connect.engine_builder._build_native_impl_qualified"
        ),
        "qualified_model": {
            "model_id": spec.model_id,
            "revision": variant["runtime_memory"]["qualified_model_revision"],
            "config_sha256": variant["runtime_memory"]["qualified_config_sha256"],
            "target": variant["runtime_memory"]["qualified_target"],
            "model_dir": str(tmp_path / "model"),
        },
        "default_policy": {
            "prefill_chunk_limit": spec.chunk_limit,
            "active_kv_profile_limits": list(spec.buckets),
        },
        "variant_policy": {
            "prefill_chunk_limit": spec.chunk_limit // 2,
            "active_kv_profile_limits": sorted({*spec.buckets, spec.chunk_limit // 2}),
        },
        "bundle": _file_identity(bundle),
        "build_timing": _file_identity(timing),
        "producer": _file_identity(producer_path),
        "runtime_kv_plugin": _binary_identity(plugin),
        "runtime_kv_plugin_mapping": _plugin_mapping_evidence(
            _binary_identity(plugin)
        ),
        "build_manifest": build_manifest_binding,
        "runtime_memory": variant["runtime_memory"],
        "source_state_pre": source_state,
        "source_state_post": dict(source_state),
    }
    receipt_path = tmp_path / "variant.receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt_path, bundle, base, variant, source_state


def test_chunk_variant_qualification_consumes_source_bound_build_receipt(
    tmp_path: Path,
) -> None:
    receipt, bundle, base, variant, source_state = _write_chunk_variant_receipt(tmp_path)

    validated = qualify._validate_chunk_variant_build_receipt(
        receipt_path=receipt,
        variant_bundle=bundle,
        base_header=base,
        variant_header=variant,
        spec=qualify.SPECS["Qwen/Qwen3-0.6B"],
        source_state=source_state,
    )

    assert validated["sha256"] == qualify._sha256(receipt)
    assert validated["bundle"]["sha256"] == qualify._sha256(bundle)
    assert validated["runtime_kv_plugin"]["path"] == str(
        (tmp_path / "build/libtrtmc_trt_plugins.so").resolve()
    )
    assert validated["runtime_kv_plugin"]["inode"] == (
        tmp_path / "build/libtrtmc_trt_plugins.so"
    ).stat().st_ino
    assert validated["runtime_kv_plugin_mapping"][
        "candidate_count"
    ] == 1
    assert validated["build_manifest"]["sha256"] == qualify._sha256(
        tmp_path / "build/build-manifest.json"
    )
    assert validated["source_state_sha256"] == source_state["source_state_sha256"]


def test_chunk_variant_receipt_fails_closed_on_source_or_bundle_drift(
    tmp_path: Path,
) -> None:
    receipt, bundle, base, variant, source_state = _write_chunk_variant_receipt(tmp_path)
    changed_source = dict(source_state)
    changed_source["source_state_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="does not match qualification source"):
        qualify._validate_chunk_variant_build_receipt(
            receipt_path=receipt,
            variant_bundle=bundle,
            base_header=base,
            variant_header=variant,
            spec=qualify.SPECS["Qwen/Qwen3-0.6B"],
            source_state=changed_source,
        )

    bundle.write_bytes(b"changed")
    with pytest.raises(ValueError, match="size identity mismatch"):
        qualify._validate_chunk_variant_build_receipt(
            receipt_path=receipt,
            variant_bundle=bundle,
            base_header=base,
            variant_header=variant,
            spec=qualify.SPECS["Qwen/Qwen3-0.6B"],
            source_state=source_state,
        )


def test_chunk_variant_receipt_reopens_and_rejects_plugin_drift(
    tmp_path: Path,
) -> None:
    receipt, bundle, base, variant, source_state = (
        _write_chunk_variant_receipt(tmp_path)
    )
    plugin = tmp_path / "build/libtrtmc_trt_plugins.so"
    plugin.write_bytes(b"changed-runtime-kv-plugin")

    with pytest.raises(
        ValueError,
        match=(
            "runtime-KV plugin .*identity mismatch|"
            "build manifest replay failed"
        ),
    ):
        qualify._validate_chunk_variant_build_receipt(
            receipt_path=receipt,
            variant_bundle=bundle,
            base_header=base,
            variant_header=variant,
            spec=qualify.SPECS["Qwen/Qwen3-0.6B"],
            source_state=source_state,
        )


def test_chunk_variant_receipt_rejects_same_bytes_plugin_inode_swap(
    tmp_path: Path,
) -> None:
    receipt, bundle, base, variant, source_state = (
        _write_chunk_variant_receipt(tmp_path)
    )
    plugin = tmp_path / "build/libtrtmc_trt_plugins.so"
    replacement = tmp_path / "replacement.so"
    replacement.write_bytes(plugin.read_bytes())
    os.replace(replacement, plugin)

    with pytest.raises(
        ValueError,
        match="runtime-KV plugin inode identity mismatch",
    ):
        qualify._validate_chunk_variant_build_receipt(
            receipt_path=receipt,
            variant_bundle=bundle,
            base_header=base,
            variant_header=variant,
            spec=qualify.SPECS["Qwen/Qwen3-0.6B"],
            source_state=source_state,
        )


def test_chunk_variant_receipt_rejects_mapping_evidence_drift(
    tmp_path: Path,
) -> None:
    receipt, bundle, base, variant, source_state = (
        _write_chunk_variant_receipt(tmp_path)
    )
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["runtime_kv_plugin_mapping"]["candidate_count"] = 2
    receipt.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="mapping does not prove one exact DSO",
    ):
        qualify._validate_chunk_variant_build_receipt(
            receipt_path=receipt,
            variant_bundle=bundle,
            base_header=base,
            variant_header=variant,
            spec=qualify.SPECS["Qwen/Qwen3-0.6B"],
            source_state=source_state,
        )


def test_chunk_variant_receipt_reopens_and_rejects_manifest_drift(
    tmp_path: Path,
) -> None:
    receipt, bundle, base, variant, source_state = (
        _write_chunk_variant_receipt(tmp_path)
    )
    (tmp_path / "build/build-manifest.json").write_text(
        '{"git_head":"changed"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="build manifest (replay failed|binding changed)",
    ):
        qualify._validate_chunk_variant_build_receipt(
            receipt_path=receipt,
            variant_bundle=bundle,
            base_header=base,
            variant_header=variant,
            spec=qualify.SPECS["Qwen/Qwen3-0.6B"],
            source_state=source_state,
        )


def _hf_contract(*, revision: str, config_sha256: str) -> dict:
    return {
        "qualified_model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "qualified_model_revision": revision,
        "qualified_config_sha256": config_sha256,
    }


def test_hf_reference_rejects_wrong_snapshot_revision(tmp_path: Path) -> None:
    config = b'{"model_type":"llama"}\n'
    expected_revision = "a" * 40
    snapshot = tmp_path / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0" / "snapshots" / ("b" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(config)
    contract = _hf_contract(
        revision=expected_revision,
        config_sha256=hashlib.sha256(config).hexdigest(),
    )

    with pytest.raises(ValueError, match="exact qualified cache snapshot"):
        qualify.verify_hf_reference(str(snapshot), contract, remote_revision=None)


def test_hf_reference_rejects_wrong_config_fingerprint(tmp_path: Path) -> None:
    expected_revision = "a" * 40
    snapshot = (
        tmp_path / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0" / "snapshots" / expected_revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"tampered"}\n', encoding="utf-8")
    contract = _hf_contract(
        revision=expected_revision,
        config_sha256=hashlib.sha256(b'{"model_type":"llama"}\n').hexdigest(),
    )

    with pytest.raises(ValueError, match="config fingerprint mismatch"):
        qualify.verify_hf_reference(str(snapshot), contract, remote_revision=None)


def test_remote_hf_reference_requires_exact_immutable_revision() -> None:
    contract = _hf_contract(
        revision="a" * 40,
        config_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="explicit immutable"):
        qualify.verify_hf_reference(
            contract["qualified_model_id"],
            contract,
            remote_revision=None,
        )


def test_logits_artifact_reader_checks_version_shape_and_payload(tmp_path: Path) -> None:
    path = tmp_path / "logits.bin"
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    path.write_bytes(
        qualify.LOGITS_HEADER.pack(qualify.LOGITS_MAGIC, 1, 1, values.shape[0], values.shape[1])
        + values.astype("<f4").tobytes()
    )

    np.testing.assert_array_equal(qualify.read_logits_artifact(path), values)

    path.write_bytes(
        struct.pack("<8sIIQQ", qualify.LOGITS_MAGIC, 1, 1, 3, 4)
        + values.astype("<f4").tobytes()[:-1]
    )
    with pytest.raises(ValueError, match="payload size mismatch"):
        qualify.read_logits_artifact(path)


def test_exact_logits_pass_all_existing_family_gates() -> None:
    logits = np.asarray(
        [[-1.0, 0.2, 2.0, 0.0], [0.1, 4.0, -2.0, 0.5]],
        dtype=np.float32,
    )
    thresholds = {
        "logit_atol": 0.001,
        "logit_cosine_p5": 0.99,
        "logit_rel_l2_p95": 0.05,
        "stable_margin": 0.1,
        "stable_top1_match_rate": 0.9,
        "token_agreement_rate": 0.8,
        "unstable_topk_hit_rate": 0.8,
    }

    result = qualify.compare_logits(logits, logits.copy(), [2], thresholds)

    assert result["passed"]
    assert all(result["gates"].values())


def test_stable_top1_divergence_fails_without_weakening_gate() -> None:
    hf = np.asarray([[0.0, 4.0, -2.0]], dtype=np.float32)
    trt = np.asarray([[5.0, 0.0, -2.0]], dtype=np.float32)
    thresholds = {
        "logit_atol": 10.0,
        "logit_cosine_p5": -1.0,
        "logit_rel_l2_p95": 10.0,
        "stable_margin": 0.1,
        "stable_top1_match_rate": 0.9,
        "token_agreement_rate": 0.8,
        "unstable_topk_hit_rate": 0.8,
    }

    result = qualify.compare_logits(trt, hf, [], thresholds)

    assert not result["passed"]
    assert not result["gates"]["stable_top1_match_rate"]
    assert not result["gates"]["token_agreement_rate"]


def test_family_composite_does_not_promote_atol_to_hard_gate() -> None:
    hf = np.asarray([[0.0, 4.0, -2.0]], dtype=np.float32)
    trt = np.asarray([[0.02, 4.02, -1.98]], dtype=np.float32)
    thresholds = {
        "logit_atol": 0.001,
        "logit_cosine_p5": 0.99,
        "logit_rel_l2_p95": 0.05,
        "stable_margin": 0.1,
        "stable_top1_match_rate": 0.9,
        "token_agreement_rate": 0.8,
        "unstable_topk_hit_rate": 0.8,
    }

    result = qualify.compare_logits(trt, hf, [1], thresholds)

    assert not result["gates"]["logit_atol"]
    assert result["composite_gates"] == {
        "numerical": True,
        "token_level": True,
    }
    assert result["passed"]


def test_family_composite_fails_when_cosine_and_relative_l2_both_fail() -> None:
    hf = np.asarray([[0.0, 4.0, -2.0]], dtype=np.float32)
    trt = np.asarray([[0.0, 0.1, 4.0]], dtype=np.float32)
    thresholds = {
        "logit_atol": 10.0,
        "logit_cosine_p5": 0.99,
        "logit_rel_l2_p95": 0.05,
        "stable_margin": 10.0,
        "stable_top1_match_rate": 0.0,
        "token_agreement_rate": 0.0,
        "unstable_topk_hit_rate": 0.0,
    }

    result = qualify.compare_logits(trt, hf, [], thresholds)

    assert not result["gates"]["logit_cosine_p5"]
    assert not result["gates"]["logit_rel_l2_p95"]
    assert not result["composite_gates"]["numerical"]
    assert not result["passed"]


def test_unstable_tie_uses_family_top5_fallback() -> None:
    # HF has a zero-margin tie between IDs 0 and 1; TRT selecting the other
    # tied ID must fail exact agreement but pass the family top-k fallback.
    hf = np.asarray([[4.0, 4.0, 3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    trt = np.asarray([[4.01, 4.02, 3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    thresholds = {
        "logit_atol": 0.001,
        "logit_cosine_p5": 0.99,
        "logit_rel_l2_p95": 0.05,
        "stable_margin": 0.1,
        "stable_top1_match_rate": 0.9,
        "token_agreement_rate": 0.8,
        "unstable_topk_hit_rate": 0.8,
    }

    result = qualify.compare_logits(trt, hf, [1], thresholds)

    assert result["metrics"]["hf_top1_margins"] == [0.0]
    assert not result["gates"]["token_agreement_rate"]
    assert result["gates"]["unstable_topk_hit_rate"]
    assert result["composite_gates"]["token_level"]
    assert result["passed"]


def _sampled_peak_receipt(
    *,
    model_context_limit: int,
    prefill_chunk_limit: int,
    capacity_tokens: int,
    bytes_per_token: int,
    context_bytes: int = 4096,
) -> dict:
    module_residency_reserve_bytes = 1
    final_non_kv_overhead_bytes = context_bytes + 1_024 + 2_048 + 4_096
    capacity_decision_free_bytes = math.ceil(
        capacity_tokens * bytes_per_token / 0.9
    ) + module_residency_reserve_bytes
    total_bytes = capacity_decision_free_bytes + 1_000_000_000
    settled_free_bytes = max(
        1,
        capacity_decision_free_bytes - capacity_tokens * bytes_per_token,
    )
    return {
        "receipt_schema_version": 4,
        "contract_version": 2,
        "policy": "auto",
        "policy_fraction": 0.9,
        "requested_kv_bytes": 0,
        "model_context_limit": model_context_limit,
        "prefill_chunk_limit": prefill_chunk_limit,
        "request_context_limit": model_context_limit,
        "runtime_kv_capacity_tokens": capacity_tokens,
        "effective_request_limit": capacity_tokens,
        "kv_bytes_per_token": bytes_per_token,
        "safety_reserve_bytes": 0,
        **_module_residency_receipt_fields(
            reserve_bytes=module_residency_reserve_bytes,
            profile_limit=model_context_limit,
        ),
        "capacity_decision_free_bytes": capacity_decision_free_bytes,
        "capacity_decision_total_bytes": total_bytes,
        "capacity_decision_device_used_bytes": (
            total_bytes - capacity_decision_free_bytes
        ),
        "capacity_decision_resident_overhead_bytes": (
            final_non_kv_overhead_bytes
        ),
        "final_non_kv_overhead_delta_bytes": 0,
        "settled_free_bytes": settled_free_bytes,
        "settled_total_bytes": total_bytes,
        "settled_device_used_bytes": total_bytes - settled_free_bytes,
        "settled_snapshot_unavailable_reason": None,
        "final_free_bytes": capacity_decision_free_bytes,
        "final_total_bytes": total_bytes,
        "final_device_used_bytes": (
            total_bytes - capacity_decision_free_bytes
        ),
        "kv_budget_bytes": qualify._binary64_fraction_floor(
            0.9,
            capacity_decision_free_bytes
            - module_residency_reserve_bytes,
        ),
        "kv_reserved_bytes": capacity_tokens * bytes_per_token,
        "kv_committed_bytes": capacity_tokens * bytes_per_token,
        "capped_by_model": capacity_tokens == model_context_limit,
        "capped_by_request_limit": capacity_tokens == model_context_limit,
        "context_device_memory_bytes": context_bytes,
        "ordinary_device_input_bytes": 1_024,
        "ordinary_device_output_bytes": 2_048,
        "external_device_output_bytes": 4_096,
        "graph_private_device_bytes": 0,
        "peak_device_bytes": 8192,
        "peak_device_bytes_scope": "device_wide",
        "peak_device_sample_count": 2,
        "peak_device_sample_boundaries": [
            "after_runtime_kv_allocation",
            "after_successful_request_completion",
        ],
    }


def _bind_receipt_to_phase_samples(receipt: dict, samples: list[dict]) -> None:
    receipt.update(
        {
            "pre_load_free_bytes": samples[0]["free_bytes"],
            "pre_load_total_bytes": samples[0]["total_bytes"],
            "post_load_free_bytes": samples[1]["free_bytes"],
            "post_load_total_bytes": samples[1]["total_bytes"],
            "post_load_device_used_bytes": (
                samples[1]["total_bytes"] - samples[1]["free_bytes"]
            ),
            "capacity_decision_free_bytes": samples[2]["free_bytes"],
            "capacity_decision_total_bytes": samples[2]["total_bytes"],
            "capacity_decision_device_used_bytes": (
                samples[2]["total_bytes"] - samples[2]["free_bytes"]
            ),
            "final_free_bytes": samples[2]["free_bytes"],
            "final_total_bytes": samples[2]["total_bytes"],
            "final_device_used_bytes": (
                samples[2]["total_bytes"] - samples[2]["free_bytes"]
            ),
            "settled_free_bytes": samples[3]["free_bytes"],
            "settled_total_bytes": samples[3]["total_bytes"],
            "settled_device_used_bytes": (
                samples[3]["total_bytes"] - samples[3]["free_bytes"]
            ),
        }
    )
    receipt["kv_budget_bytes"] = qualify._binary64_fraction_floor(
        receipt["policy_fraction"],
        max(
            0,
            receipt["capacity_decision_free_bytes"]
            - receipt["safety_reserve_bytes"]
            - receipt["module_residency_reserve_bytes"]
            - receipt["final_non_kv_overhead_delta_bytes"],
        ),
    )


def _plan_id(role: str) -> str:
    if role == "prefill":
        return "prefill_engine_plan@engine=0x1000"
    if role == "decode":
        return "engine_plan@engine=0x2000"
    raise ValueError(role)


def test_trace_validation_requires_exact_launch_formula_and_allocation_id() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    case = qualify.Case("c-plus-1", 1_025, 0)
    trace = {
        "prompt_tokens": 1_025,
        "prefill_chunk_limit": 1_024,
        "prefill_launches": 2,
        "decode_launches": 0,
        "final_kv_position": 1_025,
        "effective_request_limit": 40_960,
        "runtime_memory_receipt": {
            **_sampled_peak_receipt(
                model_context_limit=40_960,
                prefill_chunk_limit=1_024,
                capacity_tokens=40_960,
                bytes_per_token=114_688,
            ),
            "kv_allocation_id": 7,
        },
        "runtime_kv_capacity_tokens": 40_960,
        "kv_allocation_id": 7,
        "invocations": [
            {
                "invocation_index": 0,
                "role": "prefill",
                "plan_id": _plan_id("prefill"),
                "profile_id": 6,
                "chunk_range": [0, 1_024],
                "launch_count": 1,
                "kv_allocation_id": 7,
                "kv_base_address": 4096,
                "H": 0,
                "A": 1_024,
                "T": 1,
                "R": 40_960,
                "context_device_memory_bytes": 2048,
                "cuda_graph_status": "uncaptured",
                "kv_device_to_host_bytes": 0,
                "kv_append_bytes": 1_024 * 114_688,
                "full_history_device_to_device_bytes": 0,
            },
            {
                "invocation_index": 1,
                "role": "prefill",
                "plan_id": _plan_id("prefill"),
                "profile_id": 6,
                "chunk_range": [1_024, 1_025],
                "launch_count": 1,
                "kv_allocation_id": 7,
                "kv_base_address": 4096,
                "H": 1_024,
                "A": 1_025,
                "T": 1_024,
                "R": 40_960,
                "context_device_memory_bytes": 4096,
                "cuda_graph_status": "uncaptured",
                "kv_device_to_host_bytes": 0,
                "kv_append_bytes": 114_688,
                "full_history_device_to_device_bytes": 0,
            },
        ],
    }

    qualify._validate_trace(case, spec, trace, np.zeros((1, 8), dtype=np.float32))
    trace["prefill_launches"] = 1
    with pytest.raises(RuntimeError, match="prefill_launches"):
        qualify._validate_trace(case, spec, trace, np.zeros((1, 8), dtype=np.float32))


def test_trace_validation_rejects_full_history_copy_traffic() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    case = qualify.Case("one", 1, 0)
    trace = {
        "prompt_tokens": 1,
        "prefill_chunk_limit": 512,
        "prefill_launches": 1,
        "decode_launches": 0,
        "final_kv_position": 1,
        "effective_request_limit": 2_048,
        "runtime_memory_receipt": {
            **_sampled_peak_receipt(
                model_context_limit=2_048,
                prefill_chunk_limit=512,
                capacity_tokens=2_048,
                bytes_per_token=22_528,
            ),
            "kv_allocation_id": 3,
        },
        "runtime_kv_capacity_tokens": 2_048,
        "kv_allocation_id": 3,
        "invocations": [
            {
                "invocation_index": 0,
                "role": "prefill",
                "plan_id": _plan_id("prefill"),
                "profile_id": 3,
                "chunk_range": [0, 1],
                "launch_count": 1,
                "kv_allocation_id": 3,
                "kv_base_address": 8192,
                "H": 0,
                "A": 1,
                "T": 1,
                "R": 2_048,
                "context_device_memory_bytes": 1024,
                "cuda_graph_status": "uncaptured",
                "kv_device_to_host_bytes": 0,
                "kv_append_bytes": 22_528,
                "full_history_device_to_device_bytes": 22_528,
            }
        ],
    }

    with pytest.raises(RuntimeError, match="full_history_device_to_device_bytes"):
        qualify._validate_trace(case, spec, trace, np.zeros((1, 8), dtype=np.float32))


def test_profile_crossing_trace_switches_decode_profile_without_reprefill() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    case = qualify.Case(
        "profile-crossing-128",
        128,
        2,
        expected_decode_profile_ids=(0, 1),
        expected_decode_bucket_limits=(128, 256),
    )
    b = 22_528
    common = {
        "launch_count": 1,
        "kv_allocation_id": 5,
        "kv_base_address": 12_288,
        "R": 2_048,
        "context_device_memory_bytes": 2048,
        "cuda_graph_status": "uncaptured",
        "kv_device_to_host_bytes": 0,
        "full_history_device_to_device_bytes": 0,
    }
    invocations = [
        {
            **common,
            "invocation_index": 0,
            "role": "prefill",
            "plan_id": _plan_id("prefill"),
            "profile_id": 3,
            "chunk_range": [0, 128],
            "H": 0,
            "A": 128,
            "T": 1,
            "kv_append_bytes": 128 * b,
        },
        {
            **common,
            "invocation_index": 1,
            "role": "decode",
            "plan_id": _plan_id("decode"),
            "profile_id": 0,
            "chunk_range": [128, 129],
            "H": 128,
            "A": 129,
            "T": 128,
            "kv_append_bytes": b,
        },
        {
            **common,
            "invocation_index": 2,
            "role": "decode",
            "plan_id": _plan_id("decode"),
            "profile_id": 1,
            "chunk_range": [129, 130],
            "H": 129,
            "A": 130,
            "T": 256,
            "kv_append_bytes": b,
        },
    ]
    trace = {
        "prompt_tokens": 128,
        "prefill_chunk_limit": 512,
        "prefill_launches": 1,
        "decode_launches": 2,
        "final_kv_position": 130,
        "effective_request_limit": 2_048,
        "runtime_memory_receipt": {
            **_sampled_peak_receipt(
                model_context_limit=2_048,
                prefill_chunk_limit=512,
                capacity_tokens=2_048,
                bytes_per_token=b,
            ),
            "kv_allocation_id": 5,
        },
        "runtime_kv_capacity_tokens": 2_048,
        "kv_allocation_id": 5,
        "invocations": invocations,
    }
    qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))
    assert qualify.context_shape_sweep(trace) == [
        {
            "role": "prefill",
            "Sq": 128,
            "H": 0,
            "A": 128,
            "T": 1,
            "R": 2_048,
            "context_device_memory_bytes": 2048,
        },
        {
            "role": "decode",
            "Sq": 1,
            "H": 128,
            "A": 129,
            "T": 128,
            "R": 2_048,
            "context_device_memory_bytes": 2048,
        },
        {
            "role": "decode",
            "Sq": 1,
            "H": 129,
            "A": 130,
            "T": 256,
            "R": 2_048,
            "context_device_memory_bytes": 2048,
        },
    ]

    invocations[-1]["profile_id"] = 0
    with pytest.raises(RuntimeError, match="decode profiles"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))
    invocations[-1]["profile_id"] = 1

    invocations[0]["plan_id"] = "engine_plan:prefill"
    with pytest.raises(RuntimeError, match="invalid prefill plan identity"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))
    invocations[0]["plan_id"] = _plan_id("prefill")

    invocations[0]["plan_id"] = "prefill_engine_plan@engine=0x0"
    with pytest.raises(RuntimeError, match="invalid prefill plan identity"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))
    invocations[0]["plan_id"] = _plan_id("prefill")

    invocations[-1]["plan_id"] = "engine_plan@engine=0x3000"
    with pytest.raises(RuntimeError, match="share one engine identity"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))
    invocations[-1]["plan_id"] = _plan_id("decode")

    for invocation in invocations[1:]:
        invocation["plan_id"] = "engine_plan@engine=0x1000"
    with pytest.raises(RuntimeError, match="same engine identity"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))


def _context_memory_case(
    name: str,
    *,
    active_tokens: tuple[int, ...],
    context_bytes: tuple[int, ...],
    capacity_tokens: int = 40_960,
) -> dict[str, object]:
    assert len(active_tokens) == len(context_bytes)
    sweep: list[dict[str, object]] = []
    for role in ("prefill", "decode"):
        for active, measured_bytes in zip(
            active_tokens,
            context_bytes,
            strict=True,
        ):
            sweep.append(
                {
                    "role": role,
                    "Sq": 128 if role == "prefill" else 1,
                    "H": max(0, active - 1),
                    "A": active,
                    "T": active,
                    "R": capacity_tokens,
                    "context_device_memory_bytes": measured_bytes,
                }
            )
    return {
        "name": name,
        "actual_shape_context_sweep": sweep,
    }


def test_context_memory_envelope_accepts_linear_full_context_sweep() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    active_tokens = (128, 8_192, spec.context_limit)
    base_bytes = 2 * 1024 * 1024
    context_bytes = tuple(
        base_bytes + (spec.chunk_limit * (active - active_tokens[0]) * spec.num_query_heads)
        for active in active_tokens
    )

    result = qualify.validate_context_memory_envelope(
        spec,
        [
            _context_memory_case(
                "full-context",
                active_tokens=active_tokens,
                context_bytes=context_bytes,
            )
        ],
        require_full_coverage=True,
    )

    assert result["passed"]
    assert result["status"] == "passed"
    assert result["gates"]["all_points_within_o_c_times_a_envelope"]
    assert result["gates"]["all_points_below_materialized_score_bound"]
    assert result["gates"]["coverage"] == {
        "has_prefill_and_decode": True,
        "reaches_model_context_limit": True,
        "has_at_least_three_active_lengths": True,
    }


def test_context_memory_envelope_accepts_tinyllama_workspace_scaling() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    result = qualify.validate_context_memory_envelope(
        spec,
        [
            _context_memory_case(
                "tinyllama-full-context",
                active_tokens=(128, 512, spec.context_limit),
                context_bytes=(11_223_552, 23_920_640, 23_920_640),
                capacity_tokens=spec.context_limit,
            )
        ],
        require_full_coverage=True,
    )

    assert result["passed"]
    assert result["max_score_equivalent_surfaces"] == 2


def test_context_memory_envelope_rejects_quadratic_growth() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    active_tokens = (128, 8_192, spec.context_limit)
    quadratic_bytes = tuple(spec.num_query_heads * active * active * 2 for active in active_tokens)

    result = qualify.validate_context_memory_envelope(
        spec,
        [
            _context_memory_case(
                "quadratic-full-score",
                active_tokens=active_tokens,
                context_bytes=quadratic_bytes,
            )
        ],
        require_full_coverage=True,
    )

    assert not result["passed"]
    assert result["status"] == "failed"
    assert not result["gates"]["all_points_within_o_c_times_a_envelope"]
    assert not result["gates"]["all_points_below_materialized_score_bound"]


def test_context_memory_envelope_requires_full_model_coverage() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    active_tokens = (128, 512, 8_192)
    context_bytes = (2_000_000, 3_000_000, 8_000_000)

    result = qualify.validate_context_memory_envelope(
        spec,
        [
            _context_memory_case(
                "short-only",
                active_tokens=active_tokens,
                context_bytes=context_bytes,
            )
        ],
        require_full_coverage=True,
    )

    assert not result["passed"]
    assert not result["gates"]["coverage"]["reaches_model_context_limit"]


def test_context_memory_envelope_allows_partial_case_qualification() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    result = qualify.validate_context_memory_envelope(
        spec,
        [
            _context_memory_case(
                "one-selected-case",
                active_tokens=(128,),
                context_bytes=(2_000_000,),
            )
        ],
        require_full_coverage=False,
    )

    assert result["passed"]
    assert not result["gates"]["coverage"]["reaches_model_context_limit"]


def test_trace_validation_rejects_kv_base_change_across_bucket() -> None:
    spec = qualify.SPECS["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
    case = qualify.Case(
        "profile-crossing-128",
        128,
        2,
        expected_decode_profile_ids=(0, 1),
        expected_decode_bucket_limits=(128, 256),
    )
    b = 22_528
    receipt = {
        **_sampled_peak_receipt(
            model_context_limit=2_048,
            prefill_chunk_limit=512,
            capacity_tokens=2_048,
            bytes_per_token=b,
        ),
        "kv_allocation_id": 5,
    }
    invocations = []
    for index, (role, begin, end, bound) in enumerate(
        (
            ("prefill", 0, 128, 1),
            ("decode", 128, 129, 128),
            ("decode", 129, 130, 256),
        )
    ):
        invocations.append(
            {
                "invocation_index": index,
                "role": role,
                "plan_id": _plan_id(role),
                "profile_id": index,
                "chunk_range": [begin, end],
                "launch_count": 1,
                "kv_allocation_id": 5,
                "kv_base_address": 4096 if index < 2 else 8192,
                "H": begin,
                "A": end,
                "T": bound,
                "R": 2_048,
                "context_device_memory_bytes": 2048,
                "cuda_graph_status": "uncaptured",
                "kv_device_to_host_bytes": 0,
                "kv_append_bytes": (end - begin) * b,
                "full_history_device_to_device_bytes": 0,
            }
        )
    trace = {
        "prompt_tokens": 128,
        "prefill_chunk_limit": 512,
        "prefill_launches": 1,
        "decode_launches": 2,
        "final_kv_position": 130,
        "effective_request_limit": 2_048,
        "runtime_kv_capacity_tokens": 2_048,
        "kv_allocation_id": 5,
        "runtime_memory_receipt": receipt,
        "invocations": invocations,
    }

    with pytest.raises(RuntimeError, match="replaced the KV base address"):
        qualify._validate_trace(case, spec, trace, np.zeros((3, 8), dtype=np.float32))


def _attributed_phase_sample(
    *,
    phase: str,
    cuda_free: int,
    current_process: int,
    other_process: int,
    nvml_used: int,
    post_nvml_free: int | None = None,
) -> dict:
    nvml_reserved = 100_000_000
    nvml_total = 1_100_000_000
    return {
        "phase": phase,
        "device": 0,
        "free_bytes": cuda_free,
        "total_bytes": 1_000_000_000,
        "used_bytes": 1_000_000_000 - cuda_free,
        "process_used_bytes": current_process,
        "all_compute_process_used_bytes": current_process + other_process,
        "other_compute_process_used_bytes": other_process,
        "nvml_device_total_bytes": nvml_total,
        "nvml_device_reserved_bytes": nvml_reserved,
        "nvml_device_free_bytes": nvml_total - nvml_reserved - nvml_used,
        "nvml_device_used_bytes": nvml_used,
        "post_nvml_free_bytes": (cuda_free if post_nvml_free is None else post_nvml_free),
        "post_nvml_total_bytes": 1_000_000_000,
        "compute_processes": [
            {"pid": 123, "used_bytes": current_process},
            *(
                [{"pid": 456, "used_bytes": other_process}]
                if other_process > 0
                else []
            ),
        ],
    }


def _write_test_logits_artifact(path: Path, values: np.ndarray) -> dict:
    payload = np.asarray(values, dtype="<f4")
    path.write_bytes(
        qualify.LOGITS_HEADER.pack(
            qualify.LOGITS_MAGIC,
            1,
            1,
            payload.shape[0],
            payload.shape[1],
        )
        + payload.tobytes(order="C")
    )
    return {
        "format": "trtmc-qualification-logits-v1",
        "dtype": "float32",
        "rows": payload.shape[0],
        "vocab_size": payload.shape[1],
        "path": str(path.resolve()),
    }


def _load_lifetime(
    *,
    role: str,
    execution_ordinal: int,
    measured: bool,
    receipt: dict,
    phase_samples: list[dict],
    before_load: dict,
    after_unload: dict,
) -> dict:
    lifetime = {
        "execution_ordinal": execution_ordinal,
        "role": role,
        "measured": measured,
        "label": (
            "unmeasured-load-cycle-warmup"
            if role == "warmup"
            else "measured-load-cycle"
        ),
        "policy": {
            "kind": "max_sequence_length",
            "requested_tokens": 2_048,
        },
        "runtime_kv_capacity_tokens": 2_048,
        "prompt_tokens": 127,
        "prefill_launches": 1,
        "decode_launches": 1,
        "final_kv_position": 128,
        "selected_token_ids": [11],
        "step_top1_token_ids": [11, 7],
        "kv_allocation_id": receipt["kv_allocation_id"],
        "runtime_memory_receipt": receipt,
        "before_load": before_load,
        "after_requests": phase_samples[-1],
        "after_unload": after_unload,
        "process_growth_bytes": (
            phase_samples[-1]["process_used_bytes"] - before_load["process_used_bytes"]
        ),
        "device_wide_growth_bytes": (
            phase_samples[-1]["used_bytes"] - before_load["used_bytes"]
        ),
        "retained_bytes": (
            after_unload["process_used_bytes"] - before_load["process_used_bytes"]
        ),
        "device_wide_retained_bytes": (
            after_unload["used_bytes"] - before_load["used_bytes"]
        ),
        "runtime_phase_memory_samples": phase_samples,
    }
    if measured:
        lifetime["cycle_index"] = 0
    return lifetime


def _attributed_peak_trace(tmp_path: Path) -> dict:
    logits = np.full((2, 16), -4.0, dtype=np.float32)
    logits[0, 11] = 3.0
    logits[1, 7] = 2.0
    cold_logits_artifact = _write_test_logits_artifact(
        tmp_path / "cold-start-logits.bin",
        logits,
    )
    measured_logits_artifact = _write_test_logits_artifact(
        tmp_path / "measured-logits.bin",
        logits.copy(),
    )

    cold_samples = [
        _attributed_phase_sample(
            phase="before runtime-memory Qwen engine deserialization",
            cuda_free=800_000_000,
            current_process=100_000_000,
            other_process=0,
            nvml_used=150_000_000,
        ),
        _attributed_phase_sample(
            phase="before runtime KV planning",
            cuda_free=740_000_000,
            current_process=160_000_000,
            other_process=0,
            nvml_used=210_000_000,
        ),
        _attributed_phase_sample(
            phase="after shared context and output allocation",
            cuda_free=720_000_000,
            current_process=178_000_000,
            other_process=0,
            nvml_used=228_000_000,
        ),
        _attributed_phase_sample(
            phase="after runtime KV allocation",
            cuda_free=700_000_000,
            current_process=198_000_000,
            other_process=0,
            nvml_used=248_000_000,
        ),
        _attributed_phase_sample(
            phase="after successful runtime-memory request completion",
            cuda_free=705_000_000,
            current_process=190_000_000,
            other_process=0,
            nvml_used=240_000_000,
        ),
    ]
    measured_samples = [
        _attributed_phase_sample(
            phase="before runtime-memory Qwen engine deserialization",
            cuda_free=800_000_000,
            current_process=100_000_000,
            other_process=50_000_000,
            nvml_used=200_000_000,
        ),
        _attributed_phase_sample(
            phase="before runtime KV planning",
            cuda_free=740_000_000,
            current_process=160_000_000,
            other_process=50_000_000,
            nvml_used=260_000_000,
        ),
        _attributed_phase_sample(
            phase="after shared context and output allocation",
            cuda_free=720_000_000,
            current_process=178_000_000,
            other_process=50_000_000,
            nvml_used=278_000_000,
        ),
        _attributed_phase_sample(
            phase="after runtime KV allocation",
            cuda_free=700_000_000,
            current_process=198_000_000,
            other_process=50_000_000,
            nvml_used=298_000_000,
        ),
        _attributed_phase_sample(
            phase="after successful runtime-memory request completion",
            cuda_free=705_000_000,
            current_process=190_000_000,
            other_process=55_000_000,
            nvml_used=295_000_000,
        ),
    ]

    def receipt(samples: list[dict], allocation_id: int) -> dict:
        safety_reserve_bytes = 67_108_864
        module_residency_reserve_bytes = 512 * 1024 * 1024
        capacity_decision_free_bytes = samples[2]["free_bytes"]
        capacity_decision_total_bytes = samples[2]["total_bytes"]
        settled_free_bytes = samples[3]["free_bytes"]
        settled_total_bytes = samples[3]["total_bytes"]
        final_non_kv_overhead_bytes = 44_000_000
        return {
            "receipt_schema_version": 4,
            "contract_version": 2,
            "policy": "auto",
            "policy_fraction": 0.9,
            "requested_kv_bytes": 0,
            "safety_reserve_bytes": safety_reserve_bytes,
            **_module_residency_receipt_fields(
                reserve_bytes=module_residency_reserve_bytes,
                profile_limit=2_048,
            ),
            "model_context_limit": 2_048,
            "prefill_chunk_limit": 512,
            "request_context_limit": 2_048,
            "runtime_kv_capacity_tokens": 2_048,
            "effective_request_limit": 2_048,
            "kv_bytes_per_token": 22_528,
            "kv_budget_bytes": qualify._binary64_fraction_floor(
                0.9,
                capacity_decision_free_bytes
                - safety_reserve_bytes
                - module_residency_reserve_bytes,
            ),
            "serialized_plan_bytes": 100_000_000,
            "resident_weight_bytes": 400_000_000,
            "engine_weight_bytes": 350_000_000,
            "context_device_memory_bytes": 20_000_000,
            "ordinary_device_input_bytes": 1_000_000,
            "ordinary_device_output_bytes": 2_000_000,
            "external_device_output_bytes": 1_000_000,
            "graph_private_device_bytes": 20_000_000,
            "capacity_decision_resident_overhead_bytes": (
                final_non_kv_overhead_bytes
            ),
            "final_non_kv_overhead_delta_bytes": 0,
            "kv_reserved_bytes": 46_137_344,
            "kv_committed_bytes": 46_137_344,
            "kv_allocation_id": allocation_id,
            "peak_device_bytes": 100_000_000,
            "pre_load_free_bytes": samples[0]["free_bytes"],
            "pre_load_total_bytes": samples[0]["total_bytes"],
            "post_load_free_bytes": samples[1]["free_bytes"],
            "post_load_total_bytes": samples[1]["total_bytes"],
            "post_load_device_used_bytes": (
                samples[1]["total_bytes"] - samples[1]["free_bytes"]
            ),
            "capacity_decision_free_bytes": capacity_decision_free_bytes,
            "capacity_decision_total_bytes": capacity_decision_total_bytes,
            "capacity_decision_device_used_bytes": (
                capacity_decision_total_bytes
                - capacity_decision_free_bytes
            ),
            "settled_free_bytes": settled_free_bytes,
            "settled_total_bytes": settled_total_bytes,
            "settled_device_used_bytes": (
                settled_total_bytes - settled_free_bytes
            ),
            "settled_snapshot_unavailable_reason": None,
            "final_free_bytes": capacity_decision_free_bytes,
            "final_total_bytes": capacity_decision_total_bytes,
            "final_device_used_bytes": (
                capacity_decision_total_bytes
                - capacity_decision_free_bytes
            ),
            "capped_by_model": True,
            # The default fixture models --max-sequence-length=M, so the same
            # semantic edge is truthfully capped by both M and U.
            "capped_by_request_limit": True,
            "peak_device_bytes_scope": "device_wide",
            "peak_device_sample_boundaries": [
                "after_runtime_kv_allocation",
                "after_successful_request_completion",
            ],
            "peak_device_sample_count": 2,
        }

    measured_receipt = receipt(measured_samples, 42)
    warmup_receipt = receipt(cold_samples, 41)
    measured_before_load = copy.deepcopy(measured_samples[0])
    measured_after_unload = copy.deepcopy(measured_before_load)
    warmup_before_load = copy.deepcopy(cold_samples[0])
    warmup_after_unload = copy.deepcopy(cold_samples[0])
    warmup = _load_lifetime(
        role="warmup",
        execution_ordinal=0,
        measured=False,
        receipt=warmup_receipt,
        phase_samples=cold_samples,
        before_load=warmup_before_load,
        after_unload=warmup_after_unload,
    )
    measured = _load_lifetime(
        role="measured",
        execution_ordinal=1,
        measured=True,
        receipt=measured_receipt,
        phase_samples=measured_samples,
        before_load=measured_before_load,
        after_unload=measured_after_unload,
    )
    return {
        "lifetime_protocol": {
            "schema_version": 1,
            "execution_order": ["warmup", "measured"],
            "warmup_count": 1,
            "measured_count": 1,
        },
        "effective_request_limit": 2_048,
        "runtime_kv_capacity_tokens": 2_048,
        "kv_allocation_id": 42,
        "prompt_tokens": 127,
        "prefill_chunk_limit": 512,
        "prefill_launches": 1,
        "decode_launches": 1,
        "final_kv_position": 128,
        "selected_token_ids": [11],
        "step_top1_token_ids": [11, 7],
        "invocations": [
            {
                "invocation_index": 0,
                "role": "prefill",
                "plan_id": "prefill_engine_plan@engine=0x1000",
                "profile_id": 0,
                "chunk_range": [0, 127],
                "launch_count": 1,
                "kv_allocation_id": 42,
                "kv_base_address": 0x100000,
                "context_device_memory_bytes": 1_000_000,
                "H": 0,
                "A": 127,
                "T": 1,
                "R": 2_048,
                "cuda_graph_status": "uncaptured",
                "kv_device_to_host_bytes": 0,
                "kv_append_bytes": 127 * 22_528,
                "full_history_device_to_device_bytes": 0,
            },
            {
                "invocation_index": 1,
                "role": "decode",
                "plan_id": "engine_plan@engine=0x2000",
                "profile_id": 0,
                "chunk_range": [127, 128],
                "launch_count": 1,
                "kv_allocation_id": 42,
                "kv_base_address": 0x100000,
                "context_device_memory_bytes": 1_000_000,
                "H": 127,
                "A": 128,
                "T": 128,
                "R": 2_048,
                "cuda_graph_status": "uncaptured",
                "kv_device_to_host_bytes": 0,
                "kv_append_bytes": 22_528,
                "full_history_device_to_device_bytes": 0,
            },
        ],
        "cold_warm_output_equivalence": {
            "schema_version": 1,
            "warmup_execution_ordinal": 0,
            "measured_execution_ordinal": 1,
            "prompt_tokens_equal": True,
            "prefill_launches_equal": True,
            "decode_launches_equal": True,
            "final_kv_position_equal": True,
            "selected_token_ids_equal": True,
            "step_top1_token_ids_equal": True,
            "full_float32_logits_bitwise_equal": True,
            "passed": True,
        },
        "memory_sampler": {
            "source": "nvmlDeviceGetComputeRunningProcesses_v3",
            "pid": 123,
            "captures_all_compute_processes": True,
            "device_memory_source": "nvmlDeviceGetMemoryInfo_v2",
            "cuda_logical_device_index": 0,
            "physical_device_index": 7,
            "pci_bus_id": "0000:01:00.0",
            "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
        },
        "cold_start_logits_artifact": cold_logits_artifact,
        "logits_artifact": measured_logits_artifact,
        "runtime_memory_receipt": measured_receipt,
        "load_cycle_warmup": warmup,
        "load_cycle_count": 1,
        "load_cycles": [measured],
    }


def _set_attributed_trace_capacity(trace: dict, capacity_tokens: int) -> None:
    trace["runtime_kv_capacity_tokens"] = capacity_tokens
    trace["effective_request_limit"] = capacity_tokens
    receipts = (
        trace["load_cycle_warmup"]["runtime_memory_receipt"],
        trace["load_cycles"][0]["runtime_memory_receipt"],
    )
    for receipt in receipts:
        target_reserved_bytes = (
            capacity_tokens * receipt["kv_bytes_per_token"]
        )
        numerator, denominator = (0.9).as_integer_ratio()
        safely_available_bytes = (
            target_reserved_bytes * denominator + numerator - 1
        ) // numerator
        receipt["safety_reserve_bytes"] = (
            receipt["capacity_decision_free_bytes"]
            - receipt["module_residency_reserve_bytes"]
            - safely_available_bytes
        )
        receipt["kv_budget_bytes"] = qualify._binary64_fraction_floor(
            0.9,
            safely_available_bytes,
        )
        receipt["runtime_kv_capacity_tokens"] = capacity_tokens
        receipt["effective_request_limit"] = capacity_tokens
        receipt["kv_reserved_bytes"] = (
            capacity_tokens * receipt["kv_bytes_per_token"]
        )
        receipt["kv_committed_bytes"] = receipt["kv_reserved_bytes"]
        receipt["capped_by_model"] = capacity_tokens == TEST_SPEC.context_limit
        receipt["capped_by_request_limit"] = (
            receipt["request_context_limit"] != 0
            and capacity_tokens == receipt["request_context_limit"]
        )
    trace["load_cycle_warmup"][
        "runtime_kv_capacity_tokens"
    ] = capacity_tokens
    trace["load_cycles"][0]["runtime_kv_capacity_tokens"] = capacity_tokens


def test_warmup_evidence_accepts_real_five_phase_lifetimes(tmp_path: Path) -> None:
    result = _validate_warmup_evidence(_attributed_peak_trace(tmp_path))

    assert result["passed"]
    assert result["warmup_excluded_from_measured_peak"]
    assert result["warmup_independently_hard_gated"]
    assert result["reconciliation_basis"] == "cold_start_and_measured_lifetimes"
    assert len(result["warmup_phase_order"]) == 5
    assert result["warmup_phase_order"][2] == (
        "after shared context and output allocation"
    )
    assert result["continuity_reconciliation"]["passed"]
    assert result["cold_start_peak_reconciliation"]["passed"]
    assert result["measured_peak_reconciliation"]["passed"]
    assert result["cold_start_output_equivalence"]["passed"]
    assert result["cold_start_retention_gate"]["passed"]
    assert result["measured_retention_gate"]["passed"]


def test_policy_budget_uses_exact_rational_value_of_binary64_fraction() -> None:
    safely_available_bytes = 1_234_567_890_123_456_789
    numerator, denominator = (0.9).as_integer_ratio()
    exact_budget = (
        numerator * safely_available_bytes // denominator
    )
    rounded_multiply_budget = math.floor(0.9 * safely_available_bytes)
    assert rounded_multiply_budget == exact_budget + 31

    receipt = _sampled_peak_receipt(
        model_context_limit=TEST_SPEC.context_limit,
        prefill_chunk_limit=TEST_SPEC.chunk_limit,
        capacity_tokens=TEST_SPEC.context_limit,
        bytes_per_token=TEST_SPEC.kv_bytes_per_token,
    )
    capacity_decision_free_bytes = (
        safely_available_bytes
        + receipt["module_residency_reserve_bytes"]
    )
    total_bytes = capacity_decision_free_bytes + 1_000_000_000
    settled_free_bytes = (
        capacity_decision_free_bytes
        - receipt["kv_reserved_bytes"]
    )
    receipt.update(
        {
            "capacity_decision_free_bytes": capacity_decision_free_bytes,
            "capacity_decision_total_bytes": total_bytes,
            "capacity_decision_device_used_bytes": (
                total_bytes - capacity_decision_free_bytes
            ),
            "final_free_bytes": capacity_decision_free_bytes,
            "final_total_bytes": total_bytes,
            "final_device_used_bytes": (
                total_bytes - capacity_decision_free_bytes
            ),
            "settled_free_bytes": settled_free_bytes,
            "settled_total_bytes": total_bytes,
            "settled_device_used_bytes": (
                total_bytes - settled_free_bytes
            ),
            "kv_budget_bytes": exact_budget,
        }
    )
    policy = {
        "kind": "max_sequence_length",
        "requested_tokens": TEST_SPEC.context_limit,
    }

    qualify._validate_receipt_policy_binding(
        policy,
        receipt,
        trusted_geometry=TEST_GEOMETRY,
        expected_capacity_tokens=TEST_SPEC.context_limit,
        expected_effective_request_limit=TEST_SPEC.context_limit,
    )

    receipt["kv_budget_bytes"] = rounded_multiply_budget
    with pytest.raises(RuntimeError, match="exactly resolve"):
        qualify._validate_receipt_policy_binding(
            policy,
            receipt,
            trusted_geometry=TEST_GEOMETRY,
            expected_capacity_tokens=TEST_SPEC.context_limit,
            expected_effective_request_limit=TEST_SPEC.context_limit,
        )


def test_policy_budget_replays_positive_final_overhead_delta() -> None:
    receipt = _sampled_peak_receipt(
        model_context_limit=TEST_SPEC.context_limit,
        prefill_chunk_limit=TEST_SPEC.chunk_limit,
        capacity_tokens=TEST_SPEC.context_limit,
        bytes_per_token=TEST_SPEC.kv_bytes_per_token,
    )
    delta_bytes = 1_024
    receipt["capacity_decision_resident_overhead_bytes"] -= delta_bytes
    receipt["final_non_kv_overhead_delta_bytes"] = delta_bytes
    receipt["capacity_decision_free_bytes"] += delta_bytes
    receipt["capacity_decision_device_used_bytes"] -= delta_bytes
    receipt["final_free_bytes"] += delta_bytes
    receipt["final_device_used_bytes"] -= delta_bytes
    policy = {
        "kind": "max_sequence_length",
        "requested_tokens": TEST_SPEC.context_limit,
    }

    qualify._validate_receipt_policy_binding(
        policy,
        receipt,
        trusted_geometry=TEST_GEOMETRY,
        expected_capacity_tokens=TEST_SPEC.context_limit,
        expected_effective_request_limit=TEST_SPEC.context_limit,
    )

    receipt["final_non_kv_overhead_delta_bytes"] = 0
    with pytest.raises(RuntimeError, match=r"replay O\(final\)-O\(resident\)"):
        qualify._validate_receipt_policy_binding(
            policy,
            receipt,
            trusted_geometry=TEST_GEOMETRY,
            expected_capacity_tokens=TEST_SPEC.context_limit,
            expected_effective_request_limit=TEST_SPEC.context_limit,
        )


def test_policy_binding_accepts_conservative_monotonic_underfill() -> None:
    model_limit = 4_096
    initial_capacity = 2_048
    final_capacity = 1_024
    receipt = _sampled_peak_receipt(
        model_context_limit=model_limit,
        prefill_chunk_limit=TEST_SPEC.chunk_limit,
        capacity_tokens=initial_capacity,
        bytes_per_token=TEST_SPEC.kv_bytes_per_token,
    )
    receipt["runtime_kv_capacity_tokens"] = final_capacity
    receipt["effective_request_limit"] = final_capacity
    receipt["kv_reserved_bytes"] = (
        final_capacity * TEST_SPEC.kv_bytes_per_token
    )
    receipt["kv_committed_bytes"] = receipt["kv_reserved_bytes"]
    receipt["capped_by_model"] = False
    receipt["capped_by_request_limit"] = False
    receipt["settled_free_bytes"] = (
        receipt["capacity_decision_free_bytes"]
        - receipt["kv_reserved_bytes"]
    )
    receipt["settled_device_used_bytes"] = (
        receipt["settled_total_bytes"] - receipt["settled_free_bytes"]
    )

    qualify._validate_receipt_policy_binding(
        {
            "kind": "max_sequence_length",
            "requested_tokens": model_limit,
        },
        receipt,
        trusted_geometry=replace(
            TEST_GEOMETRY,
            model_context_limit=model_limit,
        ),
        expected_capacity_tokens=final_capacity,
        expected_effective_request_limit=final_capacity,
    )

    assert (
        receipt["kv_budget_bytes"] // receipt["kv_bytes_per_token"]
        == initial_capacity
    )


def test_policy_binding_rejects_capacity_above_monotonic_ceiling() -> None:
    model_limit = 4_096
    initial_capacity = 2_048
    invalid_capacity = initial_capacity + 1
    receipt = _sampled_peak_receipt(
        model_context_limit=model_limit,
        prefill_chunk_limit=TEST_SPEC.chunk_limit,
        capacity_tokens=initial_capacity,
        bytes_per_token=TEST_SPEC.kv_bytes_per_token,
    )
    receipt["runtime_kv_capacity_tokens"] = invalid_capacity
    receipt["effective_request_limit"] = invalid_capacity
    receipt["kv_reserved_bytes"] = (
        invalid_capacity * TEST_SPEC.kv_bytes_per_token
    )
    receipt["kv_committed_bytes"] = receipt["kv_reserved_bytes"]
    receipt["capped_by_model"] = False
    receipt["capped_by_request_limit"] = False
    receipt["settled_free_bytes"] = (
        receipt["capacity_decision_free_bytes"]
        - receipt["kv_reserved_bytes"]
    )
    receipt["settled_device_used_bytes"] = (
        receipt["settled_total_bytes"] - receipt["settled_free_bytes"]
    )

    with pytest.raises(RuntimeError, match="conservative monotonic.*ceiling"):
        qualify._validate_receipt_policy_binding(
            {
                "kind": "max_sequence_length",
                "requested_tokens": model_limit,
            },
            receipt,
            trusted_geometry=replace(
                TEST_GEOMETRY,
                model_context_limit=model_limit,
            ),
            expected_capacity_tokens=invalid_capacity,
            expected_effective_request_limit=invalid_capacity,
        )


def test_explicit_byte_policy_cannot_claim_silent_capacity_reduction() -> None:
    reserved_bytes = (
        TEST_SPEC.context_limit * TEST_SPEC.kv_bytes_per_token
    )
    capacity_decision_free_bytes = reserved_bytes - 1
    total_bytes = capacity_decision_free_bytes + 1_000_000_000
    receipt = _sampled_peak_receipt(
        model_context_limit=TEST_SPEC.context_limit,
        prefill_chunk_limit=TEST_SPEC.chunk_limit,
        capacity_tokens=TEST_SPEC.context_limit,
        bytes_per_token=TEST_SPEC.kv_bytes_per_token,
    )
    receipt.update(
        {
            "policy": "bytes",
            "policy_fraction": 0.0,
            "requested_kv_bytes": reserved_bytes,
            "request_context_limit": 0,
            "kv_budget_bytes": reserved_bytes,
            "capped_by_request_limit": False,
            "capacity_decision_free_bytes": (
                capacity_decision_free_bytes
            ),
            "capacity_decision_total_bytes": total_bytes,
            "capacity_decision_device_used_bytes": (
                total_bytes - capacity_decision_free_bytes
            ),
            "final_free_bytes": capacity_decision_free_bytes,
            "final_total_bytes": total_bytes,
            "final_device_used_bytes": (
                total_bytes - capacity_decision_free_bytes
            ),
            "settled_free_bytes": capacity_decision_free_bytes,
            "settled_total_bytes": total_bytes,
            "settled_device_used_bytes": (
                total_bytes - capacity_decision_free_bytes
            ),
        }
    )

    with pytest.raises(RuntimeError, match="did not fit"):
        qualify._validate_receipt_policy_binding(
            {
                "kind": "bytes",
                "requested_bytes": reserved_bytes,
            },
            receipt,
            trusted_geometry=TEST_GEOMETRY,
            expected_capacity_tokens=TEST_SPEC.context_limit,
            expected_effective_request_limit=TEST_SPEC.context_limit,
        )


@pytest.mark.parametrize(
    ("policy", "receipt_policy"),
    (
        ({"kind": "auto"}, "auto"),
        ({"kind": "fraction", "requested_fraction": 0.8}, "fraction"),
        ({"kind": "bytes", "requested_bytes": 46_137_344}, "bytes"),
        (
            {"kind": "max_sequence_length", "requested_tokens": 2_048},
            "auto",
        ),
    ),
)
def test_warmup_evidence_preserves_typed_policy_without_equating_u_and_r(
    policy: dict,
    receipt_policy: str,
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    trace["load_cycle_warmup"]["policy"] = copy.deepcopy(policy)
    trace["load_cycles"][0]["policy"] = copy.deepcopy(policy)
    receipts = (
        trace["load_cycle_warmup"]["runtime_memory_receipt"],
        trace["load_cycles"][0]["runtime_memory_receipt"],
    )
    for receipt in receipts:
        receipt["policy"] = receipt_policy
        receipt["policy_fraction"] = (
            policy["requested_fraction"]
            if policy["kind"] == "fraction"
            else 0
            if policy["kind"] == "bytes"
            else 0.9
        )
        receipt["requested_kv_bytes"] = (
            policy["requested_bytes"] if policy["kind"] == "bytes" else 0
        )
        receipt["request_context_limit"] = (
            policy["requested_tokens"]
            if policy["kind"] == "max_sequence_length"
            else 0
        )
        receipt["capped_by_model"] = (
            receipt["runtime_kv_capacity_tokens"]
            == TEST_SPEC.context_limit
        )
        receipt["capped_by_request_limit"] = (
            receipt["request_context_limit"] != 0
            and receipt["runtime_kv_capacity_tokens"]
            == receipt["request_context_limit"]
        )
        safely_available_bytes = (
            receipt["capacity_decision_free_bytes"]
            - receipt["safety_reserve_bytes"]
            - receipt["module_residency_reserve_bytes"]
        )
        receipt["kv_budget_bytes"] = (
            policy["requested_bytes"]
            if policy["kind"] == "bytes"
            else qualify._binary64_fraction_floor(
                (
                    policy["requested_fraction"]
                    if policy["kind"] == "fraction"
                    else 0.9
                ),
                safely_available_bytes,
            )
        )

    result = _validate_warmup_evidence(
        trace,
        expected_lifetime_policy=policy,
    )

    assert result["passed"]
    assert result["typed_policy"] == policy
    assert result["runtime_kv_capacity_tokens"] == 2_048
    if policy["kind"] == "max_sequence_length":
        assert (
            trace["runtime_memory_receipt"]["request_context_limit"]
            == policy["requested_tokens"]
        )


def test_warmup_evidence_keeps_user_max_sequence_distinct_from_runtime_r(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    policy = {"kind": "max_sequence_length", "requested_tokens": 2_048}
    _set_attributed_trace_capacity(trace, 1_024)

    result = _validate_warmup_evidence(
        trace,
        expected_lifetime_policy=policy,
    )

    assert result["typed_policy"] == policy
    assert result["runtime_kv_capacity_tokens"] == 1_024
    assert trace["runtime_memory_receipt"]["request_context_limit"] == 2_048
    assert trace["runtime_memory_receipt"]["effective_request_limit"] == 1_024


@pytest.mark.parametrize(
    ("policy", "receipt_policy", "receipt_updates"),
    (
        (
            {"kind": "fraction", "requested_fraction": 0.8},
            "fraction",
            {
                "policy_fraction": 0.9,
                "requested_kv_bytes": 0,
                "request_context_limit": 0,
            },
        ),
        (
            {"kind": "bytes", "requested_bytes": 46_137_345},
            "bytes",
            {
                "policy_fraction": 0,
                "requested_kv_bytes": 46_137_344,
                "request_context_limit": 0,
                "kv_budget_bytes": 46_137_344,
            },
        ),
        (
            {"kind": "max_sequence_length", "requested_tokens": 1_024},
            "auto",
            {
                "policy_fraction": 0.9,
                "requested_kv_bytes": 0,
                "request_context_limit": 2_048,
            },
        ),
    ),
    ids=("ignored-fraction", "ignored-bytes", "ignored-max-sequence"),
)
def test_warmup_evidence_rejects_runner_ignoring_typed_policy_value(
    policy: dict,
    receipt_policy: str,
    receipt_updates: dict,
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    trace["load_cycle_warmup"]["policy"] = copy.deepcopy(policy)
    trace["load_cycles"][0]["policy"] = copy.deepcopy(policy)
    receipts = (
        trace["load_cycle_warmup"]["runtime_memory_receipt"],
        trace["load_cycles"][0]["runtime_memory_receipt"],
    )
    for receipt in receipts:
        receipt["policy"] = receipt_policy
        receipt.update(receipt_updates)

    with pytest.raises(
        RuntimeError,
        match="does not bind the typed request policy",
    ):
        _validate_warmup_evidence(
            trace,
            expected_lifetime_policy=policy,
        )


def test_warmup_evidence_rejects_consistently_false_kv_byte_ledger(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    receipts = (
        trace["load_cycle_warmup"]["runtime_memory_receipt"],
        trace["load_cycles"][0]["runtime_memory_receipt"],
    )
    for receipt in receipts:
        receipt["kv_reserved_bytes"] = 1
        receipt["kv_committed_bytes"] = 1

    with pytest.raises(
        RuntimeError,
        match="exact contiguous KV ledger",
    ):
        _validate_warmup_evidence(trace)


@pytest.mark.parametrize(
    "kv_budget_bytes",
    (46_137_343, 46_137_345),
    ids=("budget-does-not-resolve-r", "budget-exceeds-request"),
)
def test_warmup_evidence_rejects_invalid_resolved_byte_budget(
    kv_budget_bytes: int,
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    policy = {"kind": "bytes", "requested_bytes": 46_137_344}
    trace["load_cycle_warmup"]["policy"] = copy.deepcopy(policy)
    trace["load_cycles"][0]["policy"] = copy.deepcopy(policy)
    receipts = (
        trace["load_cycle_warmup"]["runtime_memory_receipt"],
        trace["load_cycles"][0]["runtime_memory_receipt"],
    )
    for receipt in receipts:
        receipt["policy"] = "bytes"
        receipt["policy_fraction"] = 0
        receipt["requested_kv_bytes"] = policy["requested_bytes"]
        receipt["request_context_limit"] = 0
        receipt["kv_budget_bytes"] = kv_budget_bytes

    with pytest.raises(RuntimeError, match="KV byte budget|KV budget"):
        _validate_warmup_evidence(
            trace,
            expected_lifetime_policy=policy,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ("missing", "missing the warmup"),
        ("wrong", "warmup lifetime role"),
        ("duplicate", "exactly one measured"),
        ("order", "execution_ordinal"),
        ("request-mismatch", "request policy"),
        ("phase-missing", "exactly five"),
    ),
)
def test_warmup_evidence_fails_closed_on_protocol_or_lifetime_drift(
    mutation: str,
    error: str,
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    if mutation == "missing":
        del trace["load_cycle_warmup"]
    elif mutation == "wrong":
        trace["load_cycle_warmup"]["role"] = "measured"
    elif mutation == "duplicate":
        trace["load_cycles"].append(copy.deepcopy(trace["load_cycles"][0]))
    elif mutation == "order":
        trace["load_cycle_warmup"]["execution_ordinal"] = 1
    elif mutation == "request-mismatch":
        trace["load_cycle_warmup"]["policy"]["requested_tokens"] = 1_024
    elif mutation == "phase-missing":
        trace["load_cycle_warmup"]["runtime_phase_memory_samples"].pop(2)
    else:  # pragma: no cover - keeps additions explicit.
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=error):
        _validate_warmup_evidence(trace)


def test_warmup_evidence_rejects_unattributed_inter_lifetime_drift(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    warmup = trace["load_cycle_warmup"]
    measured = trace["load_cycles"][0]
    for lifetime in (warmup, measured):
        lifetime["runtime_memory_receipt"][
            "module_residency_reserve_bytes"
        ] = 256 * 1024 * 1024
    _bind_receipt_to_phase_samples(
        warmup["runtime_memory_receipt"],
        warmup["runtime_phase_memory_samples"],
    )
    unattributed_drift_bytes = 200_000_000
    for sample in measured["runtime_phase_memory_samples"]:
        sample["free_bytes"] -= unattributed_drift_bytes
        sample["used_bytes"] += unattributed_drift_bytes
        sample["nvml_device_free_bytes"] -= unattributed_drift_bytes
        sample["nvml_device_used_bytes"] += unattributed_drift_bytes
        sample["post_nvml_free_bytes"] -= unattributed_drift_bytes
    samples = measured["runtime_phase_memory_samples"]
    measured["before_load"] = copy.deepcopy(samples[0])
    measured["after_requests"] = copy.deepcopy(samples[-1])
    measured["after_unload"] = copy.deepcopy(samples[0])
    measured["process_growth_bytes"] = (
        measured["after_requests"]["process_used_bytes"]
        - measured["before_load"]["process_used_bytes"]
    )
    measured["device_wide_growth_bytes"] = (
        measured["after_requests"]["used_bytes"]
        - measured["before_load"]["used_bytes"]
    )
    receipt = trace["runtime_memory_receipt"]
    _bind_receipt_to_phase_samples(receipt, samples)

    with pytest.raises(RuntimeError, match="continuity does not reconcile"):
        _validate_warmup_evidence(trace)


def test_warmup_evidence_rejects_unattributed_cold_transient(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    warmup = trace["load_cycle_warmup"]
    warmup["runtime_phase_memory_samples"][3] = _attributed_phase_sample(
        phase="after runtime KV allocation",
        cuda_free=400_000_000,
        current_process=198_000_000,
        other_process=0,
        nvml_used=248_000_000,
    )
    warmup["runtime_memory_receipt"]["peak_device_bytes"] = 400_000_000
    _bind_receipt_to_phase_samples(
        warmup["runtime_memory_receipt"],
        warmup["runtime_phase_memory_samples"],
    )

    with pytest.raises(RuntimeError, match="cold_start.*external attribution"):
        _validate_warmup_evidence(trace)


def test_warmup_evidence_allows_bounded_attributed_cold_retention(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    warmup = trace["load_cycle_warmup"]
    warmup["before_load"] = _attributed_phase_sample(
        phase="unused-point-label",
        cuda_free=810_000_000,
        current_process=90_000_000,
        other_process=0,
        nvml_used=140_000_000,
    )
    warmup["process_growth_bytes"] = 100_000_000
    warmup["device_wide_growth_bytes"] = 105_000_000
    warmup["retained_bytes"] = 10_000_000
    warmup["device_wide_retained_bytes"] = 10_000_000

    result = _validate_warmup_evidence(trace)

    assert result["passed"]
    assert result["cold_start_retention_gate"]["process_retained_bytes"] == 10_000_000
    assert (
        result["cold_start_retention_gate"]["device_wide_retained_bytes"]
        == 10_000_000
    )
    assert result["cold_start_retention_gate"]["limit_bytes"] == (
        warmup["runtime_memory_receipt"][
            "module_residency_reserve_bytes"
        ]
    )
    assert result["cold_start_retention_gate"]["limit_rule"] == (
        "plan_bound_profile_calibration"
    )


def test_warmup_evidence_rejects_cold_warm_output_drift(tmp_path: Path) -> None:
    trace = _attributed_peak_trace(tmp_path)
    trace["cold_warm_output_equivalence"]["full_float32_logits_bitwise_equal"] = False
    trace["cold_warm_output_equivalence"]["passed"] = False

    with pytest.raises(RuntimeError, match="not exactly equivalent"):
        _validate_warmup_evidence(trace)


def test_warmup_evidence_rechecks_artifacts_when_runner_boolean_stays_true(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    assert trace["cold_warm_output_equivalence"]["passed"] is True
    assert (
        trace["cold_warm_output_equivalence"]["full_float32_logits_bitwise_equal"]
        is True
    )
    cold_path = Path(trace["cold_start_logits_artifact"]["path"])
    cold_bytes = bytearray(cold_path.read_bytes())
    cold_bytes[qualify.LOGITS_HEADER.size] ^= 1
    cold_path.write_bytes(cold_bytes)

    with pytest.raises(
        RuntimeError,
        match="artifacts or independently derived token IDs differ",
    ):
        _validate_warmup_evidence(trace)


def test_warmup_evidence_rejects_wrong_logical_device(tmp_path: Path) -> None:
    trace = _attributed_peak_trace(tmp_path)
    trace["load_cycles"][0]["runtime_phase_memory_samples"][2]["device"] = 1

    with pytest.raises(RuntimeError, match="runtime memory sample is invalid"):
        _validate_warmup_evidence(trace)


def test_warmup_evidence_requires_sampler_pid_in_every_process_ledger(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    sample = trace["load_cycles"][0]["runtime_phase_memory_samples"][2]
    sample["compute_processes"][0]["pid"] = 789

    with pytest.raises(RuntimeError, match="process ledger disagrees"):
        _validate_warmup_evidence(trace)


@pytest.mark.parametrize("endpoint", ("before_load", "after_requests"))
def test_warmup_evidence_binds_lifetime_endpoints_to_phase_samples(
    endpoint: str,
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    measured = trace["load_cycles"][0]
    if endpoint == "before_load":
        measured[endpoint] = _attributed_phase_sample(
            phase="detached-before-load",
            cuda_free=650_000_000,
            current_process=100_000_000,
            other_process=50_000_000,
            nvml_used=200_000_000,
        )
    else:
        measured[endpoint] = _attributed_phase_sample(
            phase="detached-after-requests",
            cuda_free=550_000_000,
            current_process=190_000_000,
            other_process=55_000_000,
            nvml_used=295_000_000,
        )
    measured["process_growth_bytes"] = (
        measured["after_requests"]["process_used_bytes"]
        - measured["before_load"]["process_used_bytes"]
    )
    measured["device_wide_growth_bytes"] = (
        measured["after_requests"]["used_bytes"]
        - measured["before_load"]["used_bytes"]
    )

    with pytest.raises(
        RuntimeError,
        match="endpoints do not bind the synchronized phase samples",
    ):
        _validate_warmup_evidence(trace)


@pytest.mark.parametrize(
    "field",
    ("kv_bytes_per_token", "kv_reserved_bytes", "kv_committed_bytes"),
)
def test_warmup_evidence_rejects_cold_receipt_stable_field_drift(
    field: str,
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    trace["load_cycle_warmup"]["runtime_memory_receipt"][field] += 1

    with pytest.raises(
        RuntimeError,
        match="does not bind the typed request policy",
    ):
        _validate_warmup_evidence(trace)


def test_warmup_evidence_requires_stable_plan_bound_residency_hashes(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    trace["load_cycle_warmup"]["runtime_memory_receipt"][
        "module_residency_evidence_sha256"
    ] = "c" * 64

    with pytest.raises(
        RuntimeError,
        match="cold/measured receipts disagree.*module_residency_evidence_sha256",
    ):
        _validate_warmup_evidence(trace)


def test_warmup_evidence_rejects_provisional_contract_receipts(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    for lifetime in (
        trace["load_cycle_warmup"],
        trace["load_cycles"][0],
    ):
        lifetime["runtime_memory_receipt"]["contract_version"] = 1

    with pytest.raises(
        RuntimeError,
        match="does not bind the typed request policy",
    ):
        _validate_warmup_evidence(trace)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("module_residency_reserve_bytes", 0),
        ("module_residency_reserve_profile_limit", 2_047),
        ("module_residency_plan_set_sha256", "A" * 64),
        ("module_residency_evidence_sha256", "bad"),
        ("module_residency_cuda_module_loading_mode", "unknown"),
    ),
)
def test_warmup_evidence_rejects_invalid_plan_bound_residency_receipt(
    field: str,
    value: object,
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    for lifetime in (
        trace["load_cycle_warmup"],
        trace["load_cycles"][0],
    ):
        lifetime["runtime_memory_receipt"][field] = value

    with pytest.raises(RuntimeError, match="plan-bound module residency"):
        _validate_warmup_evidence(trace)


def test_persisted_warmup_gate_rejects_summary_only_booleans(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)

    assert not _persisted_case_warmup_evidence_passed(
        {"status": "passed", "passed": True},
        trace=trace,
        case=qualify.Case("runtime-case", 127, 1),
    )


@pytest.mark.parametrize(
    "mutation",
    ("numeric-sample", "logits-hash", "artifact-path"),
)
def test_persisted_warmup_gate_rejects_tampered_derived_evidence(
    mutation: str,
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    evidence = _validate_warmup_evidence(trace)
    tampered = copy.deepcopy(evidence)
    if mutation == "numeric-sample":
        baseline = tampered["cold_start_peak_reconciliation"][
            "baseline_sample"
        ]
        baseline["process_used_bytes"] += 1
    elif mutation == "logits-hash":
        tampered["cold_start_output_equivalence"][
            "cold_start_logits_sha256"
        ] = "f" * 64
    elif mutation == "artifact-path":
        tampered["cold_start_output_equivalence"][
            "cold_start_logits_artifact"
        ] = str(tmp_path / "forged-cold-start-logits.bin")
    else:  # pragma: no cover - keeps additions to the table explicit.
        raise AssertionError(mutation)

    assert not _persisted_case_warmup_evidence_passed(
        tampered,
        trace=trace,
        case=qualify.Case("runtime-case", 127, 1),
    )


def test_persisted_warmup_gate_reopens_logits_artifacts(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    evidence = _validate_warmup_evidence(trace)
    cold_path = Path(trace["cold_start_logits_artifact"]["path"])
    cold_payload = bytearray(cold_path.read_bytes())
    cold_payload[-1] ^= 1
    cold_path.write_bytes(cold_payload)

    assert not _persisted_case_warmup_evidence_passed(
        evidence,
        trace=trace,
        case=qualify.Case("runtime-case", 127, 1),
    )


def test_persisted_warmup_gate_rejects_tampered_trace_artifact_path(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    evidence = _validate_warmup_evidence(trace)
    trace["cold_start_logits_artifact"]["path"] = str(
        tmp_path / "missing-cold-start-logits.bin"
    )

    assert not _persisted_case_warmup_evidence_passed(
        evidence,
        trace=trace,
        case=qualify.Case("runtime-case", 127, 1),
    )


def test_warmup_replay_rejects_workload_longer_than_fabricated_capacity(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    _set_attributed_trace_capacity(trace, 64)

    with pytest.raises(RuntimeError, match="workload exceeds"):
        _validate_warmup_evidence(trace)


def test_warmup_evidence_rejects_integer_fraction_policy(tmp_path: Path) -> None:
    trace = _attributed_peak_trace(tmp_path)
    malformed = {"kind": "fraction", "requested_fraction": 1}
    trace["load_cycle_warmup"]["policy"] = copy.deepcopy(malformed)
    trace["load_cycles"][0]["policy"] = copy.deepcopy(malformed)
    trace["load_cycle_warmup"]["runtime_memory_receipt"]["policy"] = "fraction"
    trace["runtime_memory_receipt"]["policy"] = "fraction"

    with pytest.raises(RuntimeError, match="lifetime policy is invalid"):
        _validate_warmup_evidence(trace)


def test_peak_reconciliation_rejects_measured_transient_after_warmup(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    measured_allocation = trace["load_cycles"][0]["runtime_phase_memory_samples"][3]
    measured_allocation.update(
        _attributed_phase_sample(
            phase="after runtime KV allocation",
            cuda_free=500_000_000,
            current_process=198_000_000,
            other_process=50_000_000,
            nvml_used=298_000_000,
        )
    )
    trace["runtime_memory_receipt"]["peak_device_bytes"] = 300_000_000
    _bind_receipt_to_phase_samples(
        trace["runtime_memory_receipt"],
        trace["load_cycles"][0]["runtime_phase_memory_samples"],
    )

    with pytest.raises(RuntimeError, match="external attribution"):
        _reconcile_device_peak_with_nvml(trace)


def test_peak_reconciliation_uses_independent_nvml_process_samples(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)

    result = _reconcile_device_peak_with_nvml(trace)

    assert result["passed"]
    assert result["nvml_process_peak_bytes"] == 98_000_000
    assert result["absolute_difference_bytes"] == 2_000_000
    assert result["tolerance_bytes"] == 64 * 1024 * 1024
    assert result["synchronized_cuda_peak_bytes"] == 100_000_000

    trace["runtime_memory_receipt"]["peak_device_bytes"] = 200_000_000
    with pytest.raises(RuntimeError, match="does not match synchronized"):
        _reconcile_device_peak_with_nvml(trace)


def test_peak_reconciliation_accepts_signed_visible_external_growth(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    allocation = trace["load_cycles"][0]["runtime_phase_memory_samples"][3]
    allocation.update(
        _attributed_phase_sample(
            phase="after runtime KV allocation",
            cuda_free=500_000_000,
            current_process=198_000_000,
            other_process=250_000_000,
            nvml_used=498_000_000,
        )
    )
    trace["runtime_memory_receipt"]["peak_device_bytes"] = 300_000_000
    _bind_receipt_to_phase_samples(
        trace["runtime_memory_receipt"],
        trace["load_cycles"][0]["runtime_phase_memory_samples"],
    )

    result = _reconcile_device_peak_with_nvml(trace)

    assert result["passed"]
    allocation_row = result["boundary_reconciliation"][0]
    assert allocation_row["nvml_visible_other_process_growth_bytes"] == 200_000_000
    assert allocation_row["nvml_non_current_device_growth_bytes"] == 200_000_000
    assert allocation_row["unexplained_growth_bytes"] == 2_000_000

    completion = trace["load_cycles"][0]["runtime_phase_memory_samples"][4]
    completion.update(
        _attributed_phase_sample(
            phase="after successful runtime-memory request completion",
            cuda_free=760_000_000,
            current_process=190_000_000,
            other_process=0,
            nvml_used=240_000_000,
        )
    )
    measured = trace["load_cycles"][0]
    measured["after_requests"] = copy.deepcopy(completion)
    measured["process_growth_bytes"] = (
        measured["after_requests"]["process_used_bytes"]
        - measured["before_load"]["process_used_bytes"]
    )
    measured["device_wide_growth_bytes"] = (
        measured["after_requests"]["used_bytes"]
        - measured["before_load"]["used_bytes"]
    )
    result = _reconcile_device_peak_with_nvml(trace)
    assert result["passed"]
    assert (
        result["boundary_reconciliation"][1]["nvml_visible_other_process_growth_bytes"]
        == -50_000_000
    )


def test_peak_reconciliation_rejects_unexplained_or_unlisted_growth(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    allocation = trace["load_cycles"][0]["runtime_phase_memory_samples"][3]
    allocation["free_bytes"] = 600_000_000
    allocation["used_bytes"] = 400_000_000
    trace["runtime_memory_receipt"]["peak_device_bytes"] = 200_000_000
    _bind_receipt_to_phase_samples(
        trace["runtime_memory_receipt"],
        trace["load_cycles"][0]["runtime_phase_memory_samples"],
    )
    with pytest.raises(RuntimeError, match="external attribution"):
        _reconcile_device_peak_with_nvml(trace)

    trace = _attributed_peak_trace(tmp_path)
    allocation = trace["load_cycles"][0]["runtime_phase_memory_samples"][3]
    allocation["nvml_device_used_bytes"] += 100_000_000
    allocation["nvml_device_free_bytes"] -= 100_000_000
    with pytest.raises(RuntimeError, match="external attribution"):
        _reconcile_device_peak_with_nvml(trace)

    trace = _attributed_peak_trace(tmp_path)
    trace["load_cycles"][0]["runtime_phase_memory_samples"][3]["post_nvml_free_bytes"] -= (
        100_000_000
    )
    with pytest.raises(RuntimeError, match="external attribution"):
        _reconcile_device_peak_with_nvml(trace)

    trace = _attributed_peak_trace(tmp_path)
    del trace["load_cycles"][0]["runtime_phase_memory_samples"][3]["all_compute_process_used_bytes"]
    with pytest.raises(RuntimeError, match="sample is invalid"):
        _reconcile_device_peak_with_nvml(trace)


def test_peak_reconciliation_rejects_duplicate_required_boundary(
    tmp_path: Path,
) -> None:
    trace = _attributed_peak_trace(tmp_path)
    samples = trace["load_cycles"][0]["runtime_phase_memory_samples"]
    samples.append(copy.deepcopy(samples[3]))

    with pytest.raises(RuntimeError, match="exactly five"):
        _reconcile_device_peak_with_nvml(trace)


def test_peak_reconciliation_rejects_unsynchronized_lifetime_samples() -> None:
    trace = {
        "memory_sampler": {
            "source": "nvmlDeviceGetComputeRunningProcesses_v3",
            "pid": 123,
            "captures_all_compute_processes": True,
            "device_memory_source": "nvmlDeviceGetMemoryInfo_v2",
        },
        "runtime_memory_receipt": {
            "peak_device_bytes": 100_000_000,
            "pre_load_total_bytes": 1_000_000_000,
        },
        "load_cycles": [
            {
                "before_load": {"process_used_bytes": 100_000_000},
                "after_requests": {"process_used_bytes": 198_000_000},
            }
        ],
    }

    with pytest.raises(RuntimeError, match="lifetime_protocol"):
        _reconcile_device_peak_with_nvml(trace)


def test_failure_checkpoint_persists_first_case_and_source_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_state = {
        "git_head": "a" * 40,
        "source_state_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        qualify,
        "source_state_provenance",
        lambda *_args, **_kwargs: dict(source_state),
    )
    report_path = tmp_path / "qualification-report.json"
    report = {
        "source_state_pre": dict(source_state),
        "status": "running",
        "passed": False,
        "cases": [],
    }

    with qualify.qualification_failure_checkpoint(
        report=report,
        report_path=report_path,
        repo_root=tmp_path,
        output_dir=tmp_path,
    ):
        report["cases"].append(
            {
                "name": "first-case",
                "status": "running",
                "stage": "chunk_variant_validation",
                "runner_evidence": {
                    "base": str(tmp_path / "runner-evidence" / "first-case" / "base")
                },
            }
        )
        raise RuntimeError("injected attribution failure")

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["source_state_unchanged"] is True
    assert persisted["failure"]["type"] == "RuntimeError"
    assert persisted["failure"]["stage"] == "chunk_variant_validation"
    assert persisted["cases"][0]["status"] == "failed"
    assert persisted["cases"][0]["execution_passed"] is False
    assert persisted["cases"][0]["failure"]["message"] == ("injected attribution failure")


def test_failure_checkpoint_persists_post_case_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_state = {
        "git_head": "a" * 40,
        "source_state_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        qualify,
        "source_state_provenance",
        lambda *_args, **_kwargs: dict(source_state),
    )
    report_path = tmp_path / "qualification-report.json"
    report = {
        "source_state_pre": dict(source_state),
        "status": "running",
        "stage": "context_memory_envelope",
        "passed": False,
        "cases": [{"name": "last-case", "status": "passed"}],
    }

    with qualify.qualification_failure_checkpoint(
        report=report,
        report_path=report_path,
        repo_root=tmp_path,
        output_dir=tmp_path,
    ):
        raise RuntimeError("injected finalization failure")

    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["failure"]["stage"] == "context_memory_envelope"
    assert persisted["cases"][0]["status"] == "passed"
