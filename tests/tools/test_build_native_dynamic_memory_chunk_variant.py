# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

import pytest

from tensorrt_model_connect.dynamic_memory_contract import (
    BuildTarget,
    DEVELOPER_CHUNK_VARIANT_ENV,
    DEVELOPER_CHUNK_VARIANT_VALUE,
    DynamicMemoryContractError,
    ResolvedDynamicMemoryQualification,
    derive_developer_chunk_variant_qualification,
    load_dynamic_memory_qualifications,
    validate_qualified_native_build,
)
import tensorrt_model_connect.dynamic_memory_contract as contract_module


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT
    / "tools"
    / "build_native_dynamic_memory_chunk_variant.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_native_dynamic_memory_chunk_variant",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
producer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = producer
SPEC.loader.exec_module(producer)

QUALIFY_PATH = REPO_ROOT / "tools" / "qualify_native_dynamic_memory.py"
QUALIFY_SPEC = importlib.util.spec_from_file_location(
    "qualify_native_dynamic_memory_chunk_variant_test",
    QUALIFY_PATH,
)
assert QUALIFY_SPEC is not None and QUALIFY_SPEC.loader is not None
qualify = importlib.util.module_from_spec(QUALIFY_SPEC)
sys.modules[QUALIFY_SPEC.name] = qualify
QUALIFY_SPEC.loader.exec_module(qualify)

pytestmark = [pytest.mark.unit, pytest.mark.dynamic_memory]


@pytest.fixture(autouse=True)
def _isolate_runtime_kv_plugin_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep explicit-plugin tests independent of the manifest runner."""

    monkeypatch.delenv(producer.RUNTIME_KV_PLUGIN_ENV, raising=False)


def _resolved(
    model_id: str,
    *,
    model_dir: Path,
) -> ResolvedDynamicMemoryQualification:
    record = next(
        item
        for item in load_dynamic_memory_qualifications()
        if item.qualified_model_id == model_id
    )
    return ResolvedDynamicMemoryQualification(
        qualification=record,
        model_dir=model_dir,
        target=BuildTarget(
            trt_version=record.minimum_trt_version,
            gpu_architecture=record.gpu_architecture,
            cuda_runtime=record.cuda_runtime,
            cudnn_backend=record.cudnn_backend,
            cudnn_frontend_revision=record.cudnn_frontend_revision,
            nvrtc=record.nvrtc,
            driver=record.driver,
        ),
    )


@pytest.mark.parametrize(
    ("model_id", "chunk_limit", "buckets"),
    (
        (
            "Qwen/Qwen3-0.6B",
            512,
            (128, 256, 512, 1_024, 2_048, 8_192, 32_768, 40_960),
        ),
        (
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            256,
            (128, 256, 512, 2_048),
        ),
    ),
)
def test_derives_the_one_canonical_c_div_2_variant(
    tmp_path: Path,
    model_id: str,
    chunk_limit: int,
    buckets: tuple[int, ...],
) -> None:
    base = _resolved(model_id, model_dir=tmp_path)
    variant = derive_developer_chunk_variant_qualification(
        base,
        environment={
            DEVELOPER_CHUNK_VARIANT_ENV:
                DEVELOPER_CHUNK_VARIANT_VALUE
        },
    )

    assert variant.qualification.prefill_chunk_limit == chunk_limit
    assert variant.qualification.active_kv_profile_limits == buckets
    assert replace(
        variant.qualification,
        prefill_chunk_limit=base.qualification.prefill_chunk_limit,
        active_kv_profile_limits=(
            base.qualification.active_kv_profile_limits
        ),
    ) == base.qualification


@pytest.mark.parametrize(
    "environment",
    (
        {},
        {DEVELOPER_CHUNK_VARIANT_ENV: "true"},
        {DEVELOPER_CHUNK_VARIANT_ENV: "C/4"},
    ),
)
def test_variant_derivation_requires_exact_developer_opt_in(
    tmp_path: Path,
    environment: dict[str, str],
) -> None:
    base = _resolved("Qwen/Qwen3-0.6B", model_dir=tmp_path)
    with pytest.raises(
        DynamicMemoryContractError,
        match="explicit opt-in",
    ):
        derive_developer_chunk_variant_qualification(
            base,
            environment=environment,
        )


def test_qualified_builder_guard_accepts_only_default_or_exact_c_div_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _resolved("Qwen/Qwen3-0.6B", model_dir=tmp_path)
    revision = "a" * 40
    config = b'{"model_type":"qwen3"}\n'
    record = replace(
        original.qualification,
        qualified_model_revision=revision,
        qualified_config_sha256=hashlib.sha256(config).hexdigest(),
    )
    snapshot = (
        tmp_path
        / "models--Qwen--Qwen3-0.6B"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(config)
    resolved = ResolvedDynamicMemoryQualification(
        qualification=record,
        model_dir=snapshot,
        target=BuildTarget(
            trt_version=record.minimum_trt_version,
            gpu_architecture=record.gpu_architecture,
            cuda_runtime=record.cuda_runtime,
            cudnn_backend=record.cudnn_backend,
            cudnn_frontend_revision=record.cudnn_frontend_revision,
            nvrtc=record.nvrtc,
            driver=record.driver,
        ),
    )
    monkeypatch.setattr(
        contract_module,
        "load_dynamic_memory_qualifications",
        lambda: (record,),
    )

    assert validate_qualified_native_build(
        resolved,
        environment={},
    ) == "default"
    variant = derive_developer_chunk_variant_qualification(
        resolved,
        environment={
            DEVELOPER_CHUNK_VARIANT_ENV:
                DEVELOPER_CHUNK_VARIANT_VALUE
        },
    )
    with pytest.raises(
        DynamicMemoryContractError,
        match="explicit opt-in",
    ):
        validate_qualified_native_build(variant, environment={})
    assert validate_qualified_native_build(
        variant,
        environment={
            DEVELOPER_CHUNK_VARIANT_ENV:
                DEVELOPER_CHUNK_VARIANT_VALUE
        },
    ) == "developer_c_div_2"

    arbitrary = replace(
        variant,
        qualification=replace(
            variant.qualification,
            prefill_chunk_limit=768,
            active_kv_profile_limits=tuple(
                sorted(
                    {
                        *variant.qualification.active_kv_profile_limits,
                        768,
                    }
                )
            ),
        ),
    )
    with pytest.raises(
        DynamicMemoryContractError,
        match="only the exact developer C/2",
    ):
        validate_qualified_native_build(
            arbitrary,
            environment={
                DEVELOPER_CHUNK_VARIANT_ENV:
                    DEVELOPER_CHUNK_VARIANT_VALUE
            },
        )


def test_existing_qualified_builder_calls_the_fail_closed_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder

    monkeypatch.setattr(
        contract_module,
        "validate_qualified_native_build",
        lambda _qualification: (_ for _ in ()).throw(
            DynamicMemoryContractError("guard reached")
        ),
    )
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **_kwargs: pytest.fail(
            "native builder must not run after guard rejection"
        ),
    )

    with pytest.raises(
        DynamicMemoryContractError,
        match="guard reached",
    ):
        engine_builder._build_native_impl_qualified(
            runtime_memory_qualification=object(),
            model_id_or_path="unused",
            output_path="unused.trtfb",
        )


def _bundle_bytes(
    qualification: ResolvedDynamicMemoryQualification,
) -> bytes:
    record = qualification.qualification
    contract = producer._expected_contract(qualification)
    contract["kv_bytes_per_token"] = 114_688
    header = {
        "model_id": record.qualified_model_id,
        "family": record.family,
        "max_cache_length": record.model_context_limit,
        "precision": record.precision,
        "vocab_size": 151_936,
        "runtime_memory": contract,
        "sections": {
            "engine_plan": {"offset": 0, "size": 1},
            "prefill_engine_plan": {"offset": 1, "size": 1},
        },
    }
    payload = json.dumps(header).encode("utf-8")
    return (
        producer.BUNDLE_MAGIC
        + struct.pack("<Q", len(payload))
        + payload
        + b"12"
    )


def _fake_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    plugin = tmp_path / "test-runtime-kv-plugin.so"
    plugin.write_bytes(b"unit-test-runtime-kv-plugin")
    canonical = plugin.resolve()
    monkeypatch.setattr(
        producer,
        "_loaded_runtime_kv_plugin_path",
        lambda: canonical,
    )
    monkeypatch.setattr(
        producer,
        "_runtime_kv_plugin_mapping_evidence",
        _fake_mapping_evidence,
    )
    return plugin


def _bind_fake_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plugin: Path,
    *,
    source_state: dict,
) -> Path:
    manifest = tmp_path / "build-manifest.json"
    manifest.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    binding = {
        "path": str(manifest.resolve()),
        "sha256": producer._sha256(manifest),
        "schema_version": producer.BUILD_MANIFEST_SCHEMA,
        "git_head": source_state["git_head"],
        "source_state_sha256": source_state["source_state_sha256"],
        "build_artifacts_sha256": "a" * 64,
    }
    monkeypatch.setattr(
        producer,
        "_load_strict_build_manifest",
        lambda _path: (binding, producer._binary_identity(plugin)),
    )
    return manifest


def _fake_mapping_evidence(identity: dict) -> dict:
    mapping = {
        "path": identity["path"],
        "device": identity["device"],
        "inode": identity["inode"],
    }
    return {
        "schema_version": 1,
        "source": "/proc/self/maps",
        "pid": os.getpid(),
        "selection_rule": (
            "selected_path_or_same_basename_or_exported_abi_symbol"
        ),
        "abi_symbol": producer.RUNTIME_KV_PLUGIN_ABI_SYMBOL,
        "candidate_count": 1,
        "deleted_candidate_count": 0,
        "selected": dict(identity),
        "candidate_mappings": [mapping],
    }


def _proc_map_line(path: Path, *, deleted: bool = False) -> str:
    observed = path.stat()
    suffix = " (deleted)" if deleted else ""
    return (
        "7f0000000000-7f0000001000 r-xp 00000000 "
        f"{os.major(observed.st_dev):02x}:"
        f"{os.minor(observed.st_dev):02x} "
        f"{observed.st_ino} {path.resolve()}{suffix}"
    )


def test_mapping_evidence_binds_canonical_path_device_and_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = tmp_path / "libtrtmc_trt_plugins.so"
    plugin.write_bytes(b"runtime-kv-plugin")
    identity = producer._binary_identity(plugin)
    monkeypatch.setattr(
        producer,
        "_proc_self_maps_lines",
        lambda: [_proc_map_line(plugin)],
    )
    monkeypatch.setattr(
        producer,
        "_exports_runtime_kv_plugin_abi",
        lambda _path: False,
    )

    assert producer._runtime_kv_plugin_mapping_evidence(identity) == (
        _fake_mapping_evidence(identity)
    )


def test_mapping_evidence_rejects_extra_same_basename_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "selected" / "libtrtmc_trt_plugins.so"
    extra = tmp_path / "extra" / "libtrtmc_trt_plugins.so"
    selected.parent.mkdir()
    extra.parent.mkdir()
    selected.write_bytes(b"selected")
    extra.write_bytes(b"extra")
    identity = producer._binary_identity(selected)
    monkeypatch.setattr(
        producer,
        "_proc_self_maps_lines",
        lambda: [_proc_map_line(selected), _proc_map_line(extra)],
    )
    monkeypatch.setattr(
        producer,
        "_exports_runtime_kv_plugin_abi",
        lambda _path: False,
    )

    with pytest.raises(
        producer.ChunkVariantBuildError,
        match="exactly the pinned runtime-KV plugin",
    ):
        producer._runtime_kv_plugin_mapping_evidence(identity)


def test_mapping_evidence_rejects_renamed_plugin_exporting_abi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "libtrtmc_trt_plugins.so"
    renamed = tmp_path / "renamed-plugin.bin"
    selected.write_bytes(b"selected")
    renamed.write_bytes(b"renamed")
    identity = producer._binary_identity(selected)
    monkeypatch.setattr(
        producer,
        "_proc_self_maps_lines",
        lambda: [_proc_map_line(selected), _proc_map_line(renamed)],
    )
    monkeypatch.setattr(
        producer,
        "_exports_runtime_kv_plugin_abi",
        lambda path: path == renamed.resolve(),
    )

    with pytest.raises(
        producer.ChunkVariantBuildError,
        match="exactly the pinned runtime-KV plugin",
    ):
        producer._runtime_kv_plugin_mapping_evidence(identity)


def test_mapping_evidence_rejects_deleted_shared_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = tmp_path / "libtrtmc_trt_plugins.so"
    deleted = tmp_path / "deleted.so"
    selected.write_bytes(b"selected")
    deleted.write_bytes(b"deleted")
    identity = producer._binary_identity(selected)
    monkeypatch.setattr(
        producer,
        "_proc_self_maps_lines",
        lambda: [
            _proc_map_line(selected),
            _proc_map_line(deleted, deleted=True),
        ],
    )

    with pytest.raises(
        producer.ChunkVariantBuildError,
        match="deleted shared-library mappings",
    ):
        producer._runtime_kv_plugin_mapping_evidence(identity)


def test_pinned_plugin_rejects_in_place_modify_and_restore(
    tmp_path: Path,
) -> None:
    plugin = tmp_path / "libtrtmc_trt_plugins.so"
    original = b"runtime-kv-plugin"
    plugin.write_bytes(original)

    with producer._pinned_binary(plugin) as (fd, identity):
        plugin.write_bytes(b"tampered")
        plugin.write_bytes(original)
        with pytest.raises(
            producer.ChunkVariantBuildError,
            match="changed while building",
        ):
            producer._verify_pinned_binary(plugin, fd, identity)


def test_producer_calls_existing_qualified_builder_and_writes_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = _resolved(
        "Qwen/Qwen3-0.6B",
        model_dir=tmp_path / "snapshot",
    )
    monkeypatch.setattr(
        producer,
        "_resolve_default_qualification",
        lambda model, revision: default,
    )
    captured: dict[str, object] = {}

    def fake_builder(
        qualification: ResolvedDynamicMemoryQualification,
        *,
        output: Path,
        build_timing: Path,
        verbose: bool,
    ) -> None:
        captured.update(
            {
                "qualification": qualification,
                "output": output,
                "build_timing": build_timing,
                "verbose": verbose,
            }
        )
        output.write_bytes(_bundle_bytes(qualification))
        build_timing.write_text(
            '{"schema_version":1}\n',
            encoding="utf-8",
        )

    monkeypatch.setattr(
        producer,
        "_invoke_qualified_builder",
        fake_builder,
    )
    output = tmp_path / "qwen-c-div-2.trtfb"
    receipt = tmp_path / "qwen-c-div-2.receipt.json"
    timing = tmp_path / "qwen-c-div-2.timing.json"
    plugin = _fake_plugin(tmp_path, monkeypatch)
    source_state = {
        "git_head": "a" * 40,
        "source_state_sha256": "1" * 64,
    }
    monkeypatch.setattr(
        producer,
        "_source_state_snapshot",
        lambda *_args, **_kwargs: dict(source_state),
    )
    build_manifest = _bind_fake_manifest(
        tmp_path,
        monkeypatch,
        plugin,
        source_state=source_state,
    )
    report = producer.build_chunk_variant(
        model=default.qualified_model_id,
        revision=default.qualified_model_revision,
        output=output,
        receipt=receipt,
        build_timing=timing,
        verbose=True,
        plugin_library=plugin,
        build_manifest=build_manifest,
        environment={
            DEVELOPER_CHUNK_VARIANT_ENV:
                DEVELOPER_CHUNK_VARIANT_VALUE
        },
    )

    built = captured["qualification"]
    assert isinstance(built, ResolvedDynamicMemoryQualification)
    assert built.qualification.prefill_chunk_limit == 512
    assert captured["output"] == output.resolve()
    assert captured["build_timing"] == timing.resolve()
    assert captured["verbose"] is True
    assert report["developer_only"] is True
    assert report["fresh_build"] is True
    assert report["artifact_reused"] is False
    assert report["variant_policy"] == {
        "prefill_chunk_limit": 512,
        "active_kv_profile_limits": [
            128,
            256,
            512,
            1_024,
            2_048,
            8_192,
            32_768,
            40_960,
        ],
    }
    persisted = json.loads(receipt.read_text(encoding="utf-8"))
    assert persisted == report
    assert report["bundle"]["sha256"] == producer._sha256(output)
    assert report["runtime_kv_plugin"] == producer._binary_identity(plugin)
    assert report["runtime_kv_plugin_mapping"] == (
        _fake_mapping_evidence(report["runtime_kv_plugin"])
    )
    assert set(report["runtime_kv_plugin"]) == {
        "path",
        "device",
        "inode",
        "size_bytes",
        "mtime_ns",
        "ctime_ns",
        "sha256",
    }
    assert report["build_manifest"]["path"] == str(build_manifest.resolve())
    assert report["build_manifest"]["schema_version"] == (
        producer.BUILD_MANIFEST_SCHEMA
    )
    assert report["source_state_unchanged"] is True
    assert (
        report["source_state_pre"]["source_state_sha256"]
        == report["source_state_post"]["source_state_sha256"]
    )


def test_producer_fails_closed_when_source_changes_during_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = _resolved(
        "Qwen/Qwen3-0.6B",
        model_dir=tmp_path / "snapshot",
    )
    monkeypatch.setattr(
        producer,
        "_resolve_default_qualification",
        lambda _model, _revision: default,
    )

    def fake_builder(
        qualification: ResolvedDynamicMemoryQualification,
        *,
        output: Path,
        build_timing: Path,
        verbose: bool,
    ) -> None:
        del verbose
        output.write_bytes(_bundle_bytes(qualification))
        build_timing.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        producer,
        "_invoke_qualified_builder",
        fake_builder,
    )
    snapshots = iter(
        (
            {"git_head": "a" * 40, "source_state_sha256": "1" * 64},
            {"git_head": "a" * 40, "source_state_sha256": "2" * 64},
        )
    )
    monkeypatch.setattr(
        producer,
        "_source_state_snapshot",
        lambda *_args, **_kwargs: next(snapshots),
    )
    receipt = tmp_path / "receipt.json"
    plugin = _fake_plugin(tmp_path, monkeypatch)
    build_manifest = _bind_fake_manifest(
        tmp_path,
        monkeypatch,
        plugin,
        source_state={
            "git_head": "a" * 40,
            "source_state_sha256": "1" * 64,
        },
    )

    with pytest.raises(
        producer.ChunkVariantBuildError,
        match="source state changed",
    ):
        producer.build_chunk_variant(
            model=default.qualified_model_id,
            revision=default.qualified_model_revision,
            output=tmp_path / "variant.trtfb",
            receipt=receipt,
            build_timing=tmp_path / "timing.json",
            verbose=False,
            plugin_library=plugin,
            build_manifest=build_manifest,
            environment={
                DEVELOPER_CHUNK_VARIANT_ENV:
                    DEVELOPER_CHUNK_VARIANT_VALUE
            },
        )
    assert not receipt.exists()


def test_producer_rejects_builder_that_did_not_load_selected_plugin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default = _resolved(
        "Qwen/Qwen3-0.6B",
        model_dir=tmp_path / "snapshot",
    )
    monkeypatch.setattr(
        producer,
        "_resolve_default_qualification",
        lambda _model, _revision: default,
    )

    def fake_builder(
        qualification: ResolvedDynamicMemoryQualification,
        *,
        output: Path,
        build_timing: Path,
        verbose: bool,
    ) -> None:
        del verbose
        output.write_bytes(_bundle_bytes(qualification))
        build_timing.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(producer, "_invoke_qualified_builder", fake_builder)
    monkeypatch.setattr(
        producer,
        "_loaded_runtime_kv_plugin_path",
        lambda: tmp_path / "different-plugin.so",
    )
    plugin = tmp_path / "selected-plugin.so"
    plugin.write_bytes(b"selected")
    source_state = {
        "git_head": "a" * 40,
        "source_state_sha256": "1" * 64,
    }
    monkeypatch.setattr(
        producer,
        "_source_state_snapshot",
        lambda *_args, **_kwargs: dict(source_state),
    )
    build_manifest = _bind_fake_manifest(
        tmp_path,
        monkeypatch,
        plugin,
        source_state=source_state,
    )

    with pytest.raises(
        producer.ChunkVariantBuildError,
        match="did not load the selected",
    ):
        producer.build_chunk_variant(
            model=default.qualified_model_id,
            revision=default.qualified_model_revision,
            output=tmp_path / "variant.trtfb",
            receipt=tmp_path / "receipt.json",
            build_timing=tmp_path / "timing.json",
            verbose=False,
            plugin_library=plugin,
            build_manifest=build_manifest,
            environment={
                DEVELOPER_CHUNK_VARIANT_ENV:
                    DEVELOPER_CHUNK_VARIANT_VALUE
            },
        )


def test_producer_rejects_preexisting_artifacts_before_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing.trtfb"
    output.write_bytes(b"existing")
    monkeypatch.setattr(
        producer,
        "_resolve_default_qualification",
        lambda *_args: pytest.fail(
            "fresh-artifact guard must run before model resolution"
        ),
    )

    with pytest.raises(
        producer.ChunkVariantBuildError,
        match="fresh output paths",
    ):
        producer.build_chunk_variant(
            model="Qwen/Qwen3-0.6B",
            revision=None,
            output=output,
            receipt=tmp_path / "receipt.json",
            build_timing=tmp_path / "timing.json",
            verbose=False,
            environment={
                DEVELOPER_CHUNK_VARIANT_ENV:
                    DEVELOPER_CHUNK_VARIANT_VALUE
            },
        )


def test_producer_requires_opt_in_before_model_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        producer,
        "_resolve_default_qualification",
        lambda *_args: pytest.fail(
            "environment gate must run before model resolution"
        ),
    )

    with pytest.raises(
        producer.ChunkVariantBuildError,
        match="explicit opt-in",
    ):
        producer.build_chunk_variant(
            model="Qwen/Qwen3-0.6B",
            revision=None,
            output=tmp_path / "bundle.trtfb",
            receipt=tmp_path / "receipt.json",
            build_timing=tmp_path / "timing.json",
            verbose=False,
            environment={},
        )


def test_producer_requires_exact_head_build_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        producer.ChunkVariantBuildError,
        match="requires --build-manifest",
    ):
        producer.build_chunk_variant(
            model="Qwen/Qwen3-0.6B",
            revision=None,
            output=tmp_path / "bundle.trtfb",
            receipt=tmp_path / "receipt.json",
            build_timing=tmp_path / "timing.json",
            verbose=False,
            environment={
                DEVELOPER_CHUNK_VARIANT_ENV:
                    DEVELOPER_CHUNK_VARIANT_VALUE
            },
        )


def test_qualification_runner_requires_env_before_accepting_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    base_contract = {
        "contract_version": 1,
        "qualified_model_id": spec.model_id,
        "model_context_limit": spec.context_limit,
        "prefill_chunk_limit": spec.chunk_limit,
        "kv_bytes_per_token": spec.kv_bytes_per_token,
        "active_kv_profile_limits": list(spec.buckets),
        "runtime_owned": True,
    }
    monkeypatch.setattr(
        qualify,
        "_read_bundle_header",
        lambda _path: {
            "runtime_memory": base_contract,
            "vocab_size": 151_936,
        },
    )
    monkeypatch.setattr(
        qualify,
        "require_developer_chunk_variant_opt_in",
        lambda: (_ for _ in ()).throw(RuntimeError("env gate reached")),
    )
    monkeypatch.delenv(DEVELOPER_CHUNK_VARIANT_ENV, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(QUALIFY_PATH),
            "--bundle",
            str(tmp_path / "base.trtfb"),
            "--model",
            "Qwen/Qwen3-0.6B",
            "--chunk-variant-bundle",
            str(tmp_path / "variant.trtfb"),
            "--chunk-variant-build-receipt",
            str(tmp_path / "variant.receipt.json"),
            "--runner-cuda-visible-device",
            "3",
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )

    with pytest.raises(RuntimeError, match="env gate reached"):
        qualify.main()


def test_runner_rejects_noncanonical_variant_buckets() -> None:
    spec = qualify.SPECS["Qwen/Qwen3-0.6B"]
    base_contract = {
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
            "cudnn_frontend_revision":
                "7b9b711c22b6823e87150213ecd8449260db8610",
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
    base = {
        "vocab_size": 151_936,
        "runtime_memory": base_contract,
    }
    variant_contract = dict(base_contract)
    variant_contract["prefill_chunk_limit"] = spec.chunk_limit // 2
    variant_contract["active_kv_profile_limits"] = [
        128,
        512,
        1_024,
        2_048,
        8_192,
        32_768,
        40_960,
    ]
    variant = {
        "vocab_size": 151_936,
        "runtime_memory": variant_contract,
    }

    with pytest.raises(ValueError, match="canonical C/2 buckets"):
        qualify._validate_chunk_variant(base, variant, spec)
