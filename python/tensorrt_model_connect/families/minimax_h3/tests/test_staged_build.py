# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tensorrt_model_connect import trt_compat
from tensorrt_model_connect.families.minimax_h3 import checkpoint
from tensorrt_model_connect.families.minimax_h3 import staged_build
from tensorrt_model_connect.families.minimax_h3.plugin import plugin
from tests.builder.conftest import read_bundle_file


SOURCE_REVISION = "a" * 40


@pytest.fixture(autouse=True)
def _source_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRTMC_MINIMAX_H3_SOURCE_REVISION", SOURCE_REVISION)


@pytest.fixture(autouse=True)
def _synthetic_checkpoint_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    def record(model: Path) -> dict:
        files = {}
        for path in sorted(model.rglob("*")):
            if not path.is_file() or "transformer_ref" in path.relative_to(model).parts:
                continue
            relative = path.relative_to(model).as_posix()
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            files[relative] = {
                "blob_id": digest,
                "bytes": len(payload),
                "sha256": digest,
            }
        payload = {
            "repository": "MiniMaxAI/MiniMax-H3",
            "revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
            "files": files,
        }
        return {
            **payload,
            "file_count": len(files),
            "inventory_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }

    monkeypatch.setattr(staged_build, "checkpoint_snapshot_record", record)
    monkeypatch.setattr(staged_build, "validate_checkpoint_snapshot_record", lambda value: value)


class _StreamWriterBase:
    def __init__(self) -> None:
        pass


def _write_audio_vae_config(model: Path) -> None:
    path = model / "audio_vae" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "encoder_rates": [2, 4, 4, 5, 5],
                "latent_channels": 32,
                "latent_dim": 2048,
                "decoder_dim": 1024,
                "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
                "decoder_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "sampling_rate": 32000,
                "latents_mean": [0.0] * 32,
                "latents_std": [1.0] * 32,
            }
        ),
        encoding="utf-8",
    )


def _write_plan_record(path: Path, payload: bytes) -> dict[str, int | str]:
    path.write_bytes(payload)
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_singular_fbc_builders_keep_124_to_345_dynamic_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = (
        "adaln_precompute",
        "denoiser_head",
        "denoiser_tail",
        "denoiser_finish",
    )
    assert tuple(item[0] for item in staged_build._DENSE_FBC_COMPONENTS) == components

    calls = {}

    def builder(name):
        def build(_weights, profile, **options):
            payload = name.encode()
            Path(options["output_path"]).write_bytes(payload)
            calls[name] = (profile, options)
            return {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

        return build

    family = "tensorrt_model_connect.families.minimax_h3"
    adaln = ModuleType(f"{family}.adaln_builder")
    adaln.build_adaln_precompute_engine = builder("adaln_precompute")
    adaln.checkpoint_keys = lambda _profile: ("adaln",)
    monkeypatch.setitem(sys.modules, adaln.__name__, adaln)

    dit = ModuleType(f"{family}.dit_builder")
    dit.build_dit_finish_engine = builder("denoiser_finish")
    dit.build_dit_head_engine = builder("denoiser_head")
    dit.build_dit_tail_engine = builder("denoiser_tail")
    dit.finish_checkpoint_keys = lambda: ("finish",)
    dit.head_checkpoint_keys = lambda _profile: ("head",)
    dit.tail_checkpoint_keys = lambda _profile: ("tail",)
    monkeypatch.setitem(sys.modules, dit.__name__, dit)

    monkeypatch.setattr(staged_build.trt_compat, "configure_backend", lambda **_kwargs: None)
    monkeypatch.setattr(
        checkpoint,
        "load_selected_component_state_dict",
        lambda _root, keys: {key: object() for key in keys},
    )
    monkeypatch.setattr(checkpoint, "numpy_state", dict)

    records = []
    for component in components:
        records.append(
            staged_build._build_component(
                component,
                tmp_path,
                tmp_path / f"{component}.plan",
                verbose=False,
            )
        )

    assert tuple(calls) == components
    assert all(staged_build._valid_plan_record(record) for record in records)
    assert (staged_build.VIDEO_NUM_FRAMES_MIN, staged_build.VIDEO_NUM_FRAMES_MAX) == (124, 345)
    for profile, options in calls.values():
        assert profile.first_block_cache is True
        assert profile.packed_row_profile == (19285, 37838, 112367)
        assert options["workspace_bytes"] is None
        assert options["weight_streaming"] is True


def test_plan_writer_streams_memoryview_and_cleans_failed_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        trt_compat,
        "load_module",
        lambda: SimpleNamespace(IStreamWriter=_StreamWriterBase),
    )
    payload = bytearray(b"streamed-plan" * 1024)

    class Builder:
        def build_serialized_network_to_stream(self, _network, _config, writer) -> bool:
            assert writer.write(memoryview(payload), len(payload)) == len(payload)
            return True

    output = tmp_path / "engine.plan"
    record = trt_compat.build_serialized_network_to_file(Builder(), object(), object(), output)
    assert output.read_bytes() == payload
    assert record == {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    capsule_payload = b"capsule-plan"
    capsule_storage = ctypes.create_string_buffer(capsule_payload)
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule_new.restype = ctypes.py_object
    capsule = capsule_new(ctypes.addressof(capsule_storage), None, None)

    class CapsuleBuilder:
        def build_serialized_network_to_stream(self, _network, _config, writer) -> bool:
            assert writer.write(capsule, len(capsule_payload)) == len(capsule_payload)
            return True

    capsule_output = tmp_path / "capsule.plan"
    capsule_record = trt_compat.build_serialized_network_to_file(
        CapsuleBuilder(), object(), object(), capsule_output
    )
    assert capsule_output.read_bytes() == capsule_payload
    assert capsule_record["sha256"] == hashlib.sha256(capsule_payload).hexdigest()

    output.write_bytes(b"previous")

    class FailingBuilder:
        def build_serialized_network_to_stream(self, _network, _config, writer) -> bool:
            assert writer.write(b"partial") == len(b"partial")
            return False

    with pytest.raises(RuntimeError, match="failed to serialize"):
        trt_compat.build_serialized_network_to_file(FailingBuilder(), object(), object(), output)
    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".engine.plan.tmp.*"))


def test_staged_build_uses_fresh_segment_children_resumes_and_sanitizes_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text('{"model": {}}', encoding="utf-8")
    _write_audio_vae_config(model)
    shard = model / "transformer" / "weights.safetensors"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"aaaa")
    output = tmp_path / "h3.bundle"
    calls: list[list[str]] = []

    def run(command, *, check):
        assert check is True
        calls.append(list(command))
        assert command[:3] == [sys.executable, "-m", staged_build._MODULE]
        assert "--child" in command
        component = command[command.index("--component") + 1]
        plan = Path(command[command.index("--output") + 1])
        record_output = Path(command[command.index("--record-output") + 1])
        record = _write_plan_record(plan, f"plan:{component}".encode())
        record_output.write_text(json.dumps(record), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(staged_build.subprocess, "run", run)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1.120")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    assert staged_build.build_staged_bundle(model, output) == output
    assert [call[call.index("--component") + 1] for call in calls] == [
        item[0] for item in staged_build._COMPONENTS
    ]
    assert not list(tmp_path.rglob(".*.plan.record.json"))

    header, sections = read_bundle_file(str(output))
    config = json.loads(sections["config.json"])
    assert header["gpu_name"] == ""
    assert config["engine_backend"] == "trt_rtx"
    assert config["cuda_major"] == 12
    assert config["runtime_memory"] == {
        "mode": "staged",
        "weight_streaming_budget_bytes": 32 << 30,
    }
    assert config["denoiser_profile_count"] == 2
    assert config["denoiser_profile_layout"] == "five_second_reference_then_public_dynamic"
    assert config["checkpoint_revision"] == "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc"
    assert config["source_revision"] == SOURCE_REVISION
    assert len(config["builder_source_sha256"]) == 64
    assert len(config["checkpoint_inventory_sha256"]) == 64
    assert config["workspace_limit_bytes"] == staged_build._workspace_limits_for_components(
        staged_build._COMPONENTS, ref2va=False
    )
    assert (
        config["num_frames_min"],
        config["num_frames_opt"],
        config["num_frames_max"],
    ) == (124, 124, 345)
    assert (
        config["video_rows_min"],
        config["video_rows_opt"],
        config["video_rows_max"],
    ) == (18870, 37296, 108576)
    assert (
        config["audio_rows_min"],
        config["audio_rows_opt"],
        config["audio_rows_max"],
    ) == (414, 414, 1150)
    assert (
        config["packed_sequence_length_min"],
        config["packed_sequence_length_opt"],
        config["packed_sequence_length_max"],
    ) == (19285, 37838, 112367)
    assert "adaln_precompute_mode" not in config
    assert "first_block_cache_abi" not in config
    assert "dense_tail_segment_sections" not in config
    assert config["bundle_loading"]["lazy_sections"][2:6] == [
        "adaln_precompute_plan",
        "denoiser_head_plan",
        "denoiser_tail_plan",
        "denoiser_finish_plan",
    ]
    assert set(config["plan_sha256"]) == {item[1] for item in staged_build._COMPONENTS}
    assert all(
        len(value) == 64 and value == value.lower() for value in config["plan_sha256"].values()
    )
    serialized = json.dumps(config).lower()
    assert str(tmp_path).lower() not in serialized
    for forbidden in (
        "hostname",
        "gpu_preflight",
        "tensorrt_build_environment",
        "uuid",
        "pci_bus_id",
        "installation_root",
    ):
        assert forbidden not in serialized

    calls.clear()
    staged_build.build_staged_bundle(model, output)
    assert calls == []

    plans = output.with_name(f"{output.name}.plans")
    target_component = "denoiser_tail"
    target_filename = f"{target_component}.plan"
    (plans / target_filename).write_bytes(b"corrupt")
    staged_build.build_staged_bundle(model, output)
    assert calls == []
    assert (plans / target_filename).read_bytes() == b"corrupt"

    # With no final or partial assembly, exact surviving sources are reused and
    # only the changed plan is rebuilt.
    output.unlink()
    for component, filename, _section in staged_build._COMPONENTS:
        if filename != target_filename:
            (plans / filename).write_bytes(f"plan:{component}".encode())
    calls.clear()
    staged_build.build_staged_bundle(model, output)
    assert [call[call.index("--component") + 1] for call in calls] == [target_component]

    shard.write_bytes(b"bbbb")
    with pytest.raises(ValueError, match="different checkpoint, builder"):
        staged_build.build_staged_bundle(model, output)
    shard.write_bytes(b"aaaa")
    tokenizer.write_text('{"model": {"changed": true}}', encoding="utf-8")
    with pytest.raises(ValueError, match="different checkpoint, builder"):
        staged_build.build_staged_bundle(model, output)


def test_staged_receipt_contains_only_settings_and_plan_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}", encoding="utf-8")
    _write_audio_vae_config(model)
    output = tmp_path / "h3.bundle"

    def build(component: str, _model: Path, plan: Path, *, verbose: bool):
        assert verbose is False
        return _write_plan_record(plan, component.encode())

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1.120")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")
    staged_build.build_staged_bundle(model, output)

    receipt_path = output.with_name(f"{output.name}.plans") / staged_build._RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert set(receipt) == {
        "schema_version",
        "build_identity",
        "plans",
    }
    assert set(receipt["build_identity"]) == {
        "checkpoint_revision",
        "checkpoint_inventory_sha256",
        "source_revision",
        "builder_source_sha256",
        "backend",
        "trt_version",
        "trt_abi",
        "cuda_major",
        "workspace_limit_bytes",
        "weight_streaming_budget_bytes",
    }
    assert str(tmp_path) not in json.dumps(receipt)
    assert all(set(record) == {"bytes", "sha256"} for record in receipt["plans"].values())


def test_documented_staged_route_consumes_one_shared_base_snapshot_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}", encoding="utf-8")
    _write_audio_vae_config(model)
    base_shard = model / "transformer" / "base.safetensors"
    base_shard.parent.mkdir()
    base_shard.write_bytes(b"base")
    ref_shard = model / "transformer_ref" / "ref.safetensors"
    ref_shard.parent.mkdir()
    ref_shard.write_bytes(b"independent-ref")
    output = tmp_path / "h3.bundle"
    original_record = staged_build.checkpoint_snapshot_record
    observed: list[dict] = []

    def record(path: Path) -> dict:
        assert path == model.resolve()
        value = original_record(path)
        observed.append(value)
        return value

    def build(component: str, _model: Path, plan: Path, *, verbose: bool):
        assert verbose is False
        return _write_plan_record(plan, component.encode())

    monkeypatch.setattr(staged_build, "checkpoint_snapshot_record", record)
    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1.120")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    staged_build.build_staged_bundle(model, output)

    assert len(observed) == 2
    assert observed[1] == observed[0]
    assert all("transformer_ref" not in relative for relative in observed[0]["files"])
    receipt_path = output.with_name(f"{output.name}.plans") / staged_build._RECEIPT_NAME
    identity = json.loads(receipt_path.read_text(encoding="utf-8"))["build_identity"]
    assert identity["checkpoint_inventory_sha256"] == observed[0]["inventory_sha256"]
    assert "transformer_ref" not in json.dumps(identity)


def test_staged_build_rejects_checkpoint_changed_while_plans_were_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}", encoding="utf-8")
    _write_audio_vae_config(model)
    output = tmp_path / "h3.bundle"
    mutated = False
    calls: list[str] = []

    def build(component: str, _model: Path, plan: Path, *, verbose: bool):
        nonlocal mutated
        assert verbose is False
        calls.append(component)
        record = _write_plan_record(plan, component.encode())
        if not mutated:
            tokenizer.write_text('{"changed":true}', encoding="utf-8")
            mutated = True
        return record

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1.120")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    with pytest.raises(ValueError, match="base checkpoint changed while staged plans were built"):
        staged_build.build_staged_bundle(model, output)
    assert not output.exists()
    plans = output.with_name(f"{output.name}.plans")
    invalid_marker = plans / staged_build._INVALID_RECEIPT_NAME
    assert invalid_marker.is_file()
    assert calls == [component for component, _filename, _section in staged_build._COMPONENTS]

    # Restoring the source does not make plans produced during the failed consistency
    # window reusable. A normal crash can still resume after full entry validation, but
    # an observed final-revalidation failure forces every plan to be rebuilt.
    tokenizer.write_text("{}", encoding="utf-8")
    calls.clear()
    staged_build.build_staged_bundle(model, output)

    assert calls == [component for component, _filename, _section in staged_build._COMPONENTS]
    assert output.is_file()
    assert not invalid_marker.exists()


def test_staged_build_rejects_builder_source_changed_while_plans_were_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}", encoding="utf-8")
    _write_audio_vae_config(model)
    output = tmp_path / "h3.bundle"
    source_digests = iter(("1" * 64, "2" * 64))

    def build(component: str, _model: Path, plan: Path, *, verbose: bool):
        assert verbose is False
        return _write_plan_record(plan, component.encode())

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build, "builder_source_sha256", lambda: next(source_digests))
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1.120")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    with pytest.raises(ValueError, match="builder source changed while staged plans were built"):
        staged_build.build_staged_bundle(model, output)

    invalid_marker = output.with_name(f"{output.name}.plans") / staged_build._INVALID_RECEIPT_NAME
    assert invalid_marker.is_file()
    assert not output.exists()


def test_staged_finalizer_consumes_only_pinned_checkpoint_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}", encoding="utf-8")
    _write_audio_vae_config(model)
    output = tmp_path / "h3.bundle"
    validate_sources = staged_build._validate_staged_sources_unchanged

    def build(component: str, _model: Path, plan: Path, *, verbose: bool):
        assert verbose is False
        return _write_plan_record(plan, component.encode())

    def validate_then_mutate(*args, **kwargs) -> None:
        validate_sources(*args, **kwargs)
        tokenizer.write_text('{"changed":true}', encoding="utf-8")

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build, "_validate_staged_sources_unchanged", validate_then_mutate)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1.120")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    with pytest.raises(ValueError, match="pinned checkpoint file changed before finalization"):
        staged_build.build_staged_bundle(model, output)
    assert not output.exists()


def test_staged_resume_uses_complete_receipt_before_missing_plan_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text("{}", encoding="utf-8")
    _write_audio_vae_config(model)
    output = tmp_path / "h3.bundle"
    calls: list[str] = []

    def build(component: str, _model: Path, plan: Path, *, verbose: bool):
        assert verbose is False
        calls.append(component)
        return _write_plan_record(plan, f"plan:{component}".encode())

    actual_writer = staged_build.write_consuming_bundle
    interrupt = True

    def interrupted_writer(destination, info, sections):
        def fail(event: str) -> None:
            if interrupt and event == "after_source_unlink:text_encoder_plan":
                raise RuntimeError("injected assembly interruption")

        return actual_writer(destination, info, sections, failure_injector=fail)

    monkeypatch.setattr(staged_build, "_run_component", build)
    monkeypatch.setattr(staged_build, "write_consuming_bundle", interrupted_writer)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1.120")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    with pytest.raises(RuntimeError, match="injected assembly interruption"):
        staged_build.build_staged_bundle(model, output)
    plans = output.with_name(f"{output.name}.plans")
    assert not (plans / "text_encoder.plan").exists()
    assert len(calls) == len(staged_build._COMPONENTS)

    calls.clear()
    interrupt = False
    staged_build.build_staged_bundle(model, output)
    assert calls == []
    _header, payloads = read_bundle_file(output)
    assert payloads["text_encoder_plan"] == b"plan:text_encoder"


def test_plugin_routes_only_fixed_bf16_single_gpu_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    monkeypatch.setattr(
        staged_build,
        "build_staged_bundle",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Path(args[1]),
    )
    config = SimpleNamespace(raw={})
    model = tmp_path / "model"
    output = tmp_path / "model.bundle"

    assert (
        plugin.build_staged_bundle(
            str(model),
            str(output),
            config,
            {"_model_dir": str(model)},
            precision="bf16",
            parallel_config=SimpleNamespace(mode="single"),
        )
        == output
    )
    assert calls == [((model, str(output)), {"verbose": False})]

    with pytest.raises(ValueError, match="require BF16"):
        plugin.build_staged_bundle(str(model), str(output), config, {}, precision="fp16")
    with pytest.raises(ValueError, match="require max_batch_size=1"):
        plugin.build_staged_bundle(
            str(model), str(output), config, {}, precision="bf16", max_batch_size=2
        )
    with pytest.raises(ValueError, match="require one GPU"):
        plugin.build_staged_bundle(
            str(model),
            str(output),
            config,
            {},
            precision="bf16",
            parallel_config=SimpleNamespace(mode="tensor_parallel"),
        )
