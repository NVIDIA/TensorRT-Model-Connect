# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only exact qualification tests for the first two native models."""

from __future__ import annotations

import copy
import json
import re
import struct
from pathlib import Path

import pytest

from tensorrt_model_connect.dynamic_memory_contract import (
    BuildTarget,
    DynamicMemoryContractError,
    load_dynamic_memory_qualifications,
    module_residency_plan_set_sha256,
    qualification_for_model_ref,
    qualified_runtime_stack_sha256,
    resolve_model_only_qualification,
)

pytestmark = pytest.mark.dynamic_memory


QWEN_ID = "Qwen/Qwen3-0.6B"
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
QWEN_CONFIG_SHA256 = (
    "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"
)
TINY_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINY_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
TINY_CONFIG_SHA256 = (
    "486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9"
)
QUALIFIED_STACK = {
    "cuda_runtime": "13.3",
    "cudnn_backend": "9.20.0",
    "cudnn_frontend_revision":
        "7b9b711c22b6823e87150213ecd8449260db8610",
    "nvrtc": "13.3",
    "driver": "580.105.08",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def _sealed_qwen_runtime_memory_header() -> dict:
    runtime_stack = {
        "sm": "sm103",
        "tensorrt": "11.2.0.113",
        "cuda_runtime": "13.3",
        "cudnn_backend": "9.20.0",
        "cudnn_frontend_revision": "c" * 40,
        "nvrtc": "13.3",
        "driver": "580.105.08",
    }
    profile_limits = [128, 256, 512, 1024, 2048, 8192, 32768, 40960]
    plans = [
        {
            "section_name": "engine_plan",
            "section_sha256": "d" * 64,
            "role": "decode",
            "optimization_profile_count": len(profile_limits),
        },
        {
            "section_name": "prefill_engine_plan",
            "section_sha256": "e" * 64,
            "role": "prefill",
            "optimization_profile_count": 1,
        },
    ]
    reserves = [
        {
            "covering_profile_limit": limit,
            "cumulative_reserve_bytes": 268435456 * (index + 1),
        }
        for index, limit in enumerate(profile_limits)
    ]
    return {
        "model_id": QWEN_ID,
        "model_type": "qwen3",
        "family": "qwen",
        "precision": "bf16",
        "max_cache_length": 40960,
        "runtime_memory": {
            "contract_version": 2,
            "qualified_model_id": QWEN_ID,
            "qualified_model_revision": "a" * 40,
            "qualified_config_sha256": "b" * 64,
            "qualified_target": "gb300-trt-11.2",
            "qualified_runtime_stack": runtime_stack,
            "native_kv_plugin_abi": 2,
            "model_context_limit": 40960,
            "prefill_chunk_limit": 1024,
            "kv_layout": "contiguous_runtime_v1",
            "kv_dtype": "bfloat16",
            "kv_bytes_per_token": 114688,
            "active_kv_profile_limits": profile_limits,
            "runtime_owned": True,
            "module_residency_calibration": {
                "schema_version": 1,
                "measurement_kind": "nvml_process_cumulative_first_use",
                "cuda_module_loading_mode": "lazy",
                "qualified_runtime_stack_sha256":
                    qualified_runtime_stack_sha256(runtime_stack),
                "plan_set_sha256": module_residency_plan_set_sha256(plans),
                "evidence_sha256": "f" * 64,
                "plans": plans,
                "profile_reserves": reserves,
            },
        },
        "sections": {},
    }


def _write_header_only_bundle(path: Path, header: dict) -> None:
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(
        b"TRTFB\x00\x01\x00" + struct.pack("<Q", len(payload)) + payload
    )


def test_inspect_qwen_runtime_memory_bundle_reports_only_static_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import argparse
    import ctypes
    import subprocess
    import sys

    import tensorrt_model_connect.trt_compat as trt_compat
    from tensorrt_model_connect.build_cli import _cmd_inspect

    bundle_path = tmp_path / "dynamic.trtfb"
    header = _sealed_qwen_runtime_memory_header()
    _write_header_only_bundle(bundle_path, header)

    def unexpected_runtime_touch(*_args, **_kwargs):
        pytest.fail(
            "static bundle inspection must not initialize CUDA or load a backend"
        )

    monkeypatch.setattr(trt_compat, "load_module", unexpected_runtime_touch)
    monkeypatch.setattr(ctypes, "CDLL", unexpected_runtime_touch)
    monkeypatch.setattr(subprocess, "run", unexpected_runtime_touch)
    monkeypatch.setattr(subprocess, "Popen", unexpected_runtime_touch)
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    monkeypatch.setitem(sys.modules, "tensorrt_rtx", None)
    monkeypatch.setitem(sys.modules, "cuda", None)

    assert _cmd_inspect(argparse.Namespace(bundle_path=str(bundle_path))) == 0
    output = capsys.readouterr().out
    expected_static_fields = {
        "runtime_kv_contract_version": "2",
        "qualified_model_id": QWEN_ID,
        "qualified_model_revision": "a" * 40,
        "qualified_config_fingerprint": "b" * 64,
        "model_context_limit": "40960",
        "prefill_chunk_limit": "1024",
        "kv_layout": "contiguous_runtime_v1",
        "kv_dtype": "bfloat16",
        "kv_bytes_per_token": "114688",
        "module_residency_plan_set_sha256":
            header["runtime_memory"]["module_residency_calibration"][
                "plan_set_sha256"
            ],
        "module_residency_cuda_module_loading_mode": "lazy",
        "module_residency_evidence_sha256": "f" * 64,
    }
    for label, value in expected_static_fields.items():
        assert f"{label + ':':<48} {value}" in output
    assert (
        f"{'active_kv_profile_limits:':<48} "
        "128, 256, 512, 1024, 2048, 8192, 32768, 40960"
    ) in output
    assert (
        f"{'module_residency_profile_reserves:':<48} "
        "128=>268435456, 256=>536870912, 512=>805306368, "
        "1024=>1073741824, 2048=>1342177280, 8192=>1610612736, "
        "32768=>1879048192, 40960=>2147483648"
    ) in output
    assert (
        f"{'qualified_runtime_stack:':<48} "
        "SM=sm103, TensorRT=11.2.0.113, CUDA=13.3, cuDNN=9.20.0, "
        f"Frontend={'c' * 40}, NVRTC=13.3, driver=580.105.08"
    ) in output
    assert "Max cache length:" not in output
    assert "runtime_kv_capacity_tokens" not in output
    assert "post_load_free_bytes" not in output


@pytest.mark.parametrize(
    ("mutate", "error_fragment"),
    (
        (
            lambda header: header.update(runtime_memory=[]),
            "must be a JSON object",
        ),
        (
            lambda header: header["runtime_memory"][
                "module_residency_calibration"
            ].pop("evidence_sha256"),
            "missing required field",
        ),
        (
            lambda header: header["runtime_memory"][
                "module_residency_calibration"
            ].update(plan_set_sha256="0" * 64),
            "plan_set_sha256 does not bind",
        ),
        (
            lambda header: header["runtime_memory"][
                "module_residency_calibration"
            ]["profile_reserves"][0].update(covering_profile_limit=127),
            "must align exactly",
        ),
    ),
)
def test_inspect_malformed_v2_runtime_memory_fails_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutate,
    error_fragment: str,
) -> None:
    import argparse

    from tensorrt_model_connect.build_cli import _cmd_inspect

    header = copy.deepcopy(_sealed_qwen_runtime_memory_header())
    mutate(header)
    bundle_path = tmp_path / "malformed-v2.trtfb"
    _write_header_only_bundle(bundle_path, header)

    assert _cmd_inspect(argparse.Namespace(bundle_path=str(bundle_path))) == 1
    error = capsys.readouterr().err
    assert "Invalid runtime_memory contract" in error
    assert error_fragment in error


def test_inspect_duplicate_v2_key_fails_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import argparse

    from tensorrt_model_connect.build_cli import _cmd_inspect

    payload = json.dumps(_sealed_qwen_runtime_memory_header())
    payload = payload.replace(
        '"contract_version": 2',
        '"contract_version": 2, "contract_version": 2',
        1,
    ).encode()
    bundle_path = tmp_path / "duplicate-v2-key.trtfb"
    bundle_path.write_bytes(
        b"TRTFB\x00\x01\x00" + struct.pack("<Q", len(payload)) + payload
    )

    assert _cmd_inspect(argparse.Namespace(bundle_path=str(bundle_path))) == 1
    assert "Duplicate JSON key" in capsys.readouterr().err


def _target(
    trt_version: str = "11.2.0.113",
    gpu_architecture: str = "sm103",
    cuda_runtime: str = QUALIFIED_STACK["cuda_runtime"],
    cudnn_backend: str = QUALIFIED_STACK["cudnn_backend"],
    cudnn_frontend_revision: str = QUALIFIED_STACK[
        "cudnn_frontend_revision"
    ],
    nvrtc: str = QUALIFIED_STACK["nvrtc"],
    driver: str = QUALIFIED_STACK["driver"],
) -> BuildTarget:
    return BuildTarget(
        trt_version=trt_version,
        gpu_architecture=gpu_architecture,
        cuda_runtime=cuda_runtime,
        cudnn_backend=cudnn_backend,
        cudnn_frontend_revision=cudnn_frontend_revision,
        nvrtc=nvrtc,
        driver=driver,
    )


def test_only_two_exact_qualification_records_are_declared() -> None:
    records = load_dynamic_memory_qualifications()
    assert [
        (
            record.qualified_model_id,
            record.qualified_model_revision,
            record.qualified_config_sha256,
            record.family,
            record.model_context_limit,
            record.prefill_chunk_limit,
            record.active_kv_profile_limits,
        )
        for record in records
    ] == [
        (
            QWEN_ID,
            QWEN_REVISION,
            QWEN_CONFIG_SHA256,
            "qwen",
            40960,
            1024,
            (128, 256, 512, 1024, 2048, 8192, 32768, 40960),
        ),
        (
            TINY_ID,
            TINY_REVISION,
            TINY_CONFIG_SHA256,
            "llama",
            2048,
            512,
            (128, 256, 512, 2048),
        ),
    ]
    assert all(record.precision == "bf16" for record in records)
    assert all(record.minimum_trt_version == "11.2.0.113" for record in records)
    assert all(record.qualified_target == "gb300-trt-11.2" for record in records)
    assert all(record.gpu_architecture == "sm103" for record in records)
    assert all(
        {
            "cuda_runtime": record.cuda_runtime,
            "cudnn_backend": record.cudnn_backend,
            "cudnn_frontend_revision":
                record.cudnn_frontend_revision,
            "nvrtc": record.nvrtc,
            "driver": record.driver,
        }
        == QUALIFIED_STACK
        for record in records
    )


@pytest.mark.parametrize(
    ("family", "expected_chunk", "expected_buckets"),
    (
        (
            "qwen",
            1024,
            (128, 256, 512, 1024, 2048, 8192, 32768, 40960),
        ),
        ("llama", 512, (128, 256, 512, 2048)),
    ),
)
def test_native_plugin_allowlist_matches_builder_qualification(
    family: str,
    expected_chunk: int,
    expected_buckets: tuple[int, ...],
) -> None:
    """Keep the independently compiled native trust boundary in lockstep."""

    record = next(
        item
        for item in load_dynamic_memory_qualifications()
        if item.family == family
    )
    assert record.prefill_chunk_limit == expected_chunk
    assert record.active_kv_profile_limits == expected_buckets

    source = (
        REPO_ROOT / "src" / "runtime" / "models" / family / "plugin.cpp"
    ).read_text(encoding="utf-8")
    chunk = re.search(r"value\.prefill_chunk_limit\s*=\s*(\d+);", source)
    buckets = re.search(
        r"value\.active_kv_profile_limits\s*=\s*\{([^}]+)\};",
        source,
        flags=re.DOTALL,
    )
    assert chunk is not None
    assert buckets is not None
    native_buckets = tuple(
        int(value)
        for value in re.findall(r"\d+", buckets.group(1))
    )
    assert int(chunk.group(1)) == record.prefill_chunk_limit
    assert native_buckets == record.active_kv_profile_limits


@pytest.mark.parametrize(
    ("target", "matches"),
    (
        (_target(), True),
        (_target("11.2.1", "SM103"), False),
        (_target("11.3.0"), False),
        (_target("12.0.0", "SM103"), False),
        (_target("11.1.9"), False),
        (_target(gpu_architecture="sm100"), False),
        (_target(gpu_architecture="sm90"), False),
        (_target("unknown"), False),
    ),
)
def test_target_gate_is_exact_trt_build_and_gb300_sm103(
    target: BuildTarget,
    matches: bool,
) -> None:
    record = load_dynamic_memory_qualifications()[0]
    assert record.matches_target(target) is matches


def test_build_target_probe_preserves_live_gb300_sm103_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.dynamic_memory_contract as contract_module
    import tensorrt_model_connect.trt_plugins as trt_plugins

    monkeypatch.setattr(
        trt_plugins,
        "query_runtime_kv_plugin_stack",
        lambda: {
            "sm": "sm103",
            "tensorrt": "11.2.0.113",
            **QUALIFIED_STACK,
        },
    )
    assert contract_module.probe_build_target() == _target()


def test_canonical_model_id_resolves_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.dynamic_memory_contract as contract_module

    model_dir = tmp_path / "resolved"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    calls: list[tuple[str, str | None]] = []

    def resolve(model_id: str, *, revision: str | None = None) -> str:
        calls.append((model_id, revision))
        return str(model_dir)

    monkeypatch.setattr(
        contract_module,
        "probe_build_target",
        _target,
    )
    monkeypatch.setattr(
        contract_module,
        "_sha256_file",
        lambda _path: QWEN_CONFIG_SHA256,
    )

    resolved = resolve_model_only_qualification(
        QWEN_ID,
        requested_revision=None,
        resolve_model=resolve,
    )
    assert resolved is not None
    assert resolved.qualified_model_id == QWEN_ID
    assert resolved.qualified_model_revision == QWEN_REVISION
    assert resolved.model_dir == model_dir
    assert calls == [(QWEN_ID, QWEN_REVISION)]


def test_hf_snapshot_path_canonicalizes_to_the_same_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.dynamic_memory_contract as contract_module

    snapshot = (
        tmp_path
        / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
        / "snapshots"
        / TINY_REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    calls: list[str] = []

    def resolve(path: str, **_kwargs) -> str:
        calls.append(path)
        return path

    monkeypatch.setattr(
        contract_module,
        "probe_build_target",
        _target,
    )
    monkeypatch.setattr(
        contract_module,
        "_sha256_file",
        lambda _path: TINY_CONFIG_SHA256,
    )

    resolved = resolve_model_only_qualification(
        str(snapshot),
        requested_revision=None,
        resolve_model=resolve,
    )
    assert resolved is not None
    assert resolved.qualified_model_id == TINY_ID
    assert resolved.qualified_model_revision == TINY_REVISION
    assert calls == [str(snapshot)]


@pytest.mark.parametrize("use_symlink", (False, True))
def test_recognized_local_snapshot_resolver_revision_mismatch_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    use_symlink: bool,
) -> None:
    import tensorrt_model_connect.dynamic_memory_contract as contract_module

    expected = (
        tmp_path
        / "models--Qwen--Qwen3-0.6B"
        / "snapshots"
        / QWEN_REVISION
    )
    expected.mkdir(parents=True)
    model_ref = expected
    if use_symlink:
        model_ref = tmp_path / "qualified-model"
        model_ref.symlink_to(expected, target_is_directory=True)

    wrong = (
        tmp_path
        / "models--Qwen--Qwen3-0.6B"
        / "snapshots"
        / ("0" * 40)
    )
    wrong.mkdir(parents=True)
    monkeypatch.setattr(
        contract_module,
        "probe_build_target",
        _target,
    )

    with pytest.raises(
        DynamicMemoryContractError,
        match="local model resolved to a different HF snapshot",
    ):
        resolve_model_only_qualification(
            str(model_ref),
            requested_revision=None,
            resolve_model=lambda *_args, **_kwargs: str(wrong),
        )


def test_unknown_local_identity_fails_closed_without_resolution(
    tmp_path: Path,
) -> None:
    local = tmp_path / "qwen3-0.6b"
    local.mkdir()
    (local / "config.json").write_text("{}")

    assert qualification_for_model_ref(str(local)) is None
    resolved_calls: list[str] = []
    assert (
        resolve_model_only_qualification(
            str(local),
            requested_revision=None,
            resolve_model=lambda path, **_kwargs: (
                resolved_calls.append(path) or path
            ),
        )
        is None
    )
    assert resolved_calls == []


@pytest.mark.parametrize(
    ("model_ref", "revision"),
    (
        ("Qwen/Qwen3-1.7B", None),
        ("Qwen/Qwen3-1.7B", "0" * 40),
        ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", None),
    ),
)
def test_family_or_name_similarity_never_inherits_qualification(
    model_ref: str,
    revision: str | None,
) -> None:
    assert (
        qualification_for_model_ref(
            model_ref,
            requested_revision=revision,
        )
        is None
    )


def test_canonical_model_id_revision_mismatch_is_explicit() -> None:
    wrong_revision = "0" * 40
    with pytest.raises(
        DynamicMemoryContractError,
        match=(
            "Recognized runtime-memory-qualified model revision mismatch.*"
            f"{wrong_revision}"
        ),
    ):
        qualification_for_model_ref(
            QWEN_ID,
            requested_revision=wrong_revision,
        )


def test_recognized_hf_snapshot_revision_mismatch_is_explicit(
    tmp_path: Path,
) -> None:
    wrong_revision = "0" * 40
    snapshot = (
        tmp_path
        / "models--Qwen--Qwen3-0.6B"
        / "snapshots"
        / wrong_revision
    )
    snapshot.mkdir(parents=True)

    with pytest.raises(
        DynamicMemoryContractError,
        match=(
            "Recognized runtime-memory-qualified model revision mismatch.*"
            f"{wrong_revision}"
        ),
    ):
        qualification_for_model_ref(str(snapshot))


def test_recognized_hf_snapshot_conflicting_requested_revision_is_explicit(
    tmp_path: Path,
) -> None:
    requested_revision = "0" * 40
    snapshot = (
        tmp_path
        / "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0"
        / "snapshots"
        / TINY_REVISION
    )
    snapshot.mkdir(parents=True)

    with pytest.raises(
        DynamicMemoryContractError,
        match="snapshot revision conflicts.*explicitly requested revision",
    ):
        qualification_for_model_ref(
            str(snapshot),
            requested_revision=requested_revision,
        )


def test_unknown_model_never_probes_the_runtime_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.dynamic_memory_contract as contract_module

    probe_calls = 0

    def fail_if_probed() -> BuildTarget:
        nonlocal probe_calls
        probe_calls += 1
        raise AssertionError("unknown models must not probe the plugin")

    monkeypatch.setattr(contract_module, "probe_build_target", fail_if_probed)
    resolved_calls: list[str] = []
    assert (
        resolve_model_only_qualification(
            "Qwen/Qwen3-1.7B",
            requested_revision="0" * 40,
            resolve_model=lambda model, **_kwargs: (
                resolved_calls.append(model) or model
            ),
        )
        is None
    )
    assert probe_calls == 0
    assert resolved_calls == []


def test_recognized_identity_with_config_drift_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.dynamic_memory_contract as contract_module

    model_dir = tmp_path / "resolved"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"changed": true}')
    monkeypatch.setattr(
        contract_module,
        "probe_build_target",
        _target,
    )

    with pytest.raises(
        DynamicMemoryContractError,
        match="config fingerprint mismatch",
    ):
        resolve_model_only_qualification(
            QWEN_ID,
            requested_revision=None,
            resolve_model=lambda *_args, **_kwargs: str(model_dir),
        )


def test_recognized_model_target_miss_fails_explicitly_without_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.dynamic_memory_contract as contract_module

    monkeypatch.setattr(
        contract_module,
        "probe_build_target",
        lambda: _target(gpu_architecture="sm90"),
    )
    calls: list[str] = []
    with pytest.raises(
        DynamicMemoryContractError,
        match="build target is not qualified",
    ):
        resolve_model_only_qualification(
            QWEN_ID,
            requested_revision=None,
            resolve_model=lambda model, **_kwargs: (
                calls.append(model) or model
            ),
        )
    assert calls == []


@pytest.mark.parametrize(
    ("target_field", "stack_field", "wrong_value"),
    (
        ("trt_version", "tensorrt", "11.2.0.114"),
        ("gpu_architecture", "sm", "sm100"),
        ("cuda_runtime", "cuda_runtime", "13.2"),
        ("cudnn_backend", "cudnn_backend", "9.19.0"),
        (
            "cudnn_frontend_revision",
            "cudnn_frontend_revision",
            "0" * 40,
        ),
        ("nvrtc", "nvrtc", "13.2"),
        ("driver", "driver", "580.105.07"),
    ),
)
def test_recognized_model_rejects_every_live_stack_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    target_field: str,
    stack_field: str,
    wrong_value: str,
) -> None:
    import tensorrt_model_connect.dynamic_memory_contract as contract_module

    values = {
        "trt_version": "11.2.0.113",
        "gpu_architecture": "sm103",
        **QUALIFIED_STACK,
    }
    values[target_field] = wrong_value
    monkeypatch.setattr(
        contract_module,
        "probe_build_target",
        lambda: _target(**values),
    )
    with pytest.raises(
        DynamicMemoryContractError,
        match=rf"build target is not qualified.*{stack_field}",
    ):
        resolve_model_only_qualification(
            QWEN_ID,
            requested_revision=None,
            resolve_model=lambda model, **_kwargs: model,
        )


def test_recognized_model_target_probe_failure_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.dynamic_memory_contract as contract_module

    def fail_probe() -> BuildTarget:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(contract_module, "probe_build_target", fail_probe)
    with pytest.raises(
        DynamicMemoryContractError,
        match="could not be verified",
    ):
        resolve_model_only_qualification(
            QWEN_ID,
            requested_revision=None,
            resolve_model=lambda model, **_kwargs: model,
        )
