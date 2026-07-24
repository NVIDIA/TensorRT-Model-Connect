# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only exact qualification tests for the first two native models."""

from __future__ import annotations

from pathlib import Path

import pytest

from tensorrt_model_connect.dynamic_memory_contract import (
    BuildTarget,
    DynamicMemoryContractError,
    load_dynamic_memory_qualifications,
    qualification_for_model_ref,
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


def _target(
    trt_version: str = "11.2.0.113",
    gpu_architecture: str = "sm103",
) -> BuildTarget:
    return BuildTarget(
        trt_version=trt_version,
        gpu_architecture=gpu_architecture,
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
            2048,
            (128, 512, 2048, 8192, 32768, 40960),
        ),
        (
            TINY_ID,
            TINY_REVISION,
            TINY_CONFIG_SHA256,
            "llama",
            2048,
            512,
            (128, 512, 2048),
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
    import tensorrt_model_connect.runtime_provider.target as target_module
    import tensorrt_model_connect.trt_compat as trt_compat

    monkeypatch.setattr(
        target_module,
        "_probe_current_target_with_device",
        lambda: (
            {
                "target_id": "current-discrete-sm103",
                "gpu_architecture": "sm103",
                "gpu_name": "NVIDIA GB300",
            },
            0,
        ),
    )
    monkeypatch.setattr(
        trt_compat,
        "tensorrt_version",
        lambda: "11.2.0.113",
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
        (QWEN_ID, "0" * 40),
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
