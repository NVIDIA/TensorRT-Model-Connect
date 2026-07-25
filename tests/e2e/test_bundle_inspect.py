# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E: Inspect bundles and verify header fields."""

from __future__ import annotations

import copy
import json
import os
import struct
import subprocess

import pytest

from tensorrt_model_connect.dynamic_memory_contract import (
    module_residency_plan_set_sha256,
    qualified_runtime_stack_sha256,
)


def _write_sealed_v2_bundle(path):
    stack = {
        "sm": "sm103",
        "tensorrt": "11.2.0.113",
        "cuda_runtime": "13.3",
        "cudnn_backend": "9.20.0",
        "cudnn_frontend_revision": "c" * 40,
        "nvrtc": "13.3",
        "driver": "580.105.08",
    }
    profiles = [128, 256, 512, 1024, 2048, 8192, 32768, 40960]
    plans = [
        {
            "section_name": "engine_plan",
            "section_sha256": "d" * 64,
            "role": "decode",
            "optimization_profile_count": len(profiles),
        },
        {
            "section_name": "prefill_engine_plan",
            "section_sha256": "e" * 64,
            "role": "prefill",
            "optimization_profile_count": 1,
        },
    ]
    header = {
        "model_id": "Qwen/Qwen3-0.6B",
        "model_type": "qwen3",
        "family": "qwen",
        "precision": "bf16",
        "max_cache_length": 40960,
        "runtime_memory": {
            "contract_version": 2,
            "qualified_model_id": "Qwen/Qwen3-0.6B",
            "qualified_model_revision": "a" * 40,
            "qualified_config_sha256": "b" * 64,
            "qualified_target": "gb300-trt-11.2",
            "qualified_runtime_stack": stack,
            "native_kv_plugin_abi": 2,
            "model_context_limit": 40960,
            "prefill_chunk_limit": 1024,
            "kv_layout": "contiguous_runtime_v1",
            "kv_dtype": "bfloat16",
            "kv_bytes_per_token": 114688,
            "active_kv_profile_limits": profiles,
            "runtime_owned": True,
            "runtime_config_sha256": "9" * 64,
            "module_residency_calibration": {
                "schema_version": 1,
                "measurement_kind": "nvml_process_cumulative_first_use",
                "cuda_module_loading_mode": "lazy",
                "evidence_provenance": "embedded_bundle_v1",
                "qualified_runtime_stack_sha256":
                    qualified_runtime_stack_sha256(stack),
                "plan_set_sha256": module_residency_plan_set_sha256(plans),
                "evidence_sha256": "f" * 64,
                "plans": plans,
                "profile_reserves": [
                    {
                        "covering_profile_limit": limit,
                        "cumulative_reserve_bytes": 268435456 * (index + 1),
                    }
                    for index, limit in enumerate(profiles)
                ],
            },
        },
        "sections": {},
    }
    payload = json.dumps(header).encode()
    path.write_bytes(
        b"TRTFB\x00\x01\x00" + struct.pack("<Q", len(payload)) + payload
    )
    return header


def _static_inspect_env(ld_library_path):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ld_library_path
    # A static inspector must not require a visible GPU or initialize CUDA.
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


@pytest.mark.e2e
def test_inspect_produces_output(model_entry, trtmc_binary, ld_library_path):
    """trtmc inspect <bundle> should produce valid output."""
    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        [str(trtmc_binary), "inspect", model_entry["bundle_path"]],
        capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0, f"inspect failed: {result.stderr}"
    assert len(result.stdout.strip()) > 0, "inspect produced no output"


@pytest.mark.e2e
def test_inspect_shows_runtime_strategy(model_entry, trtmc_binary, ld_library_path):
    """Inspect output should mention the runtime strategy."""
    env = {"LD_LIBRARY_PATH": ld_library_path}
    result = subprocess.run(
        [str(trtmc_binary), "inspect", model_entry["bundle_path"]],
        capture_output=True, text=True, timeout=30, env=env)
    assert result.returncode == 0
    # Verify runtime_strategy is printed and matches the expected value.
    expected = str(model_entry.get("runtime_strategy") or "")
    assert expected, "model entry must declare runtime_strategy"
    assert "Runtime strategy:" in result.stdout, (
        "Expected 'Runtime strategy:' field in inspect output")
    actual_line = [l for l in result.stdout.splitlines()
                   if "Runtime strategy:" in l]
    assert actual_line, "Expected Runtime strategy line in inspect output"
    actual = actual_line[0].split(":")[-1].strip()
    assert expected == actual, (
        f"Expected runtime_strategy '{expected}', got '{actual}'")


@pytest.mark.e2e
@pytest.mark.dynamic_memory
def test_inspect_sealed_v2_reports_complete_static_contract_without_cuda(
    tmp_path,
    trtmc_binary,
    ld_library_path,
):
    bundle = tmp_path / "sealed-v2.trtfb"
    header = _write_sealed_v2_bundle(bundle)

    result = subprocess.run(
        [str(trtmc_binary), "inspect", str(bundle)],
        capture_output=True,
        text=True,
        timeout=30,
        env=_static_inspect_env(ld_library_path),
    )

    assert result.returncode == 0, result.stderr
    required = {
        "runtime_kv_contract_version": "2",
        "model_context_limit": "40960",
        "prefill_chunk_limit": "1024",
        "kv_layout": "contiguous_runtime_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": "114688",
        "active_kv_profile_limits":
            "128, 256, 512, 1024, 2048, 8192, 32768, 40960",
        "qualified_model_revision": "a" * 40,
        "qualified_config_fingerprint": "b" * 64,
        "runtime_config_sha256": "9" * 64,
        "module_residency_plan_set_sha256":
            header["runtime_memory"]["module_residency_calibration"][
                "plan_set_sha256"
            ],
        "module_residency_cuda_module_loading_mode": "lazy",
        "module_residency_evidence_provenance": "embedded_bundle_v1",
        "module_residency_evidence_sha256": "f" * 64,
    }
    for field, value in required.items():
        assert f"{field}:" in result.stdout
        assert value in result.stdout
    assert "qualified_runtime_stack:" in result.stdout
    assert (
        "module_residency_profile_reserves: "
        "128=>268435456, 256=>536870912"
    ) in result.stdout
    assert "runtime_kv_capacity_tokens" not in result.stdout
    assert "post_load_free_bytes" not in result.stdout
    assert "[trtmc.memory]" not in result.stderr
    assert "[trtmc.runtime_stack]" not in result.stderr


@pytest.mark.e2e
@pytest.mark.dynamic_memory
def test_inspect_malformed_v2_fails_before_runtime_initialization(
    tmp_path,
    trtmc_binary,
    ld_library_path,
):
    bundle = tmp_path / "malformed-v2.trtfb"
    header = copy.deepcopy(_write_sealed_v2_bundle(bundle))
    header["runtime_memory"]["module_residency_calibration"][
        "plan_set_sha256"
    ] = "0" * 64
    payload = json.dumps(header).encode()
    bundle.write_bytes(
        b"TRTFB\x00\x01\x00" + struct.pack("<Q", len(payload)) + payload
    )

    result = subprocess.run(
        [str(trtmc_binary), "inspect", str(bundle)],
        capture_output=True,
        text=True,
        timeout=30,
        env=_static_inspect_env(ld_library_path),
    )

    assert result.returncode != 0
    assert "runtime_memory" in result.stderr
    assert "[trtmc.memory]" not in result.stderr
    assert "[trtmc.runtime_stack]" not in result.stderr
