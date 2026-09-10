# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Edge selection, API mapping and fault-isolation contracts."""

from __future__ import annotations

from dataclasses import replace
import importlib
import json
import logging
from pathlib import Path
import shutil
import struct
import subprocess
from unittest.mock import Mock

import pytest

from tensorrt_model_connect.build import BuildRequest
from tensorrt_model_connect.bundle_writer import BundleWriter
from families.qwen3_5 import dispatch, edge_llm

FAMILY = "qwen3_5"
NEWER_VARIANT = False
TARGET = {
    "os": "linux", "os_version": "24.04", "arch": "x86_64", "sm": 80,
    "cuda_version": "13.3", "tensorrt_version": "11.1.0.106",
}
ENGINE_FILES = ("llm.engine", "config.json", "tokenizer.json", "tokenizer_config.json",
                "processed_chat_template.json")


@pytest.fixture
def inputs(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    text = {"linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "max_position_embeddings": 32768}
    if NEWER_VARIANT:
        text["output_gate_type"] = "sigmoid"
    raw = {"model_type": "qwen3_5", "text_config": text}
    (checkpoint / "config.json").write_text(json.dumps(raw))
    (checkpoint / "model.safetensors").write_bytes(b"original checkpoint contents")
    (checkpoint / "tokenizer.json").write_text("{}")
    (checkpoint / "tokenizer_config.json").write_text("{}")
    (checkpoint / "unrelated.log").write_text("not part of deployment")
    request = BuildRequest(checkpoint, tmp_path / "model.bundle", FAMILY,
                           "text_generation", "fp16", max_sequence_length=1024)
    return request, raw


def route(monkeypatch, prepare):
    monkeypatch.setattr(edge_llm, "local_target", lambda: dict(TARGET))
    monkeypatch.setattr(dispatch, "EDGE_DISPATCH", {("linux", "x86_64", 80, "fp16"): prepare})


def test_supported_family_config_matches(inputs):
    request, raw = inputs
    assert dispatch.candidate(request, raw)
    assert dispatch.EDGE_DISPATCH[("linux", "x86_64", 80, "fp16")] is edge_llm.prepare


@pytest.mark.parametrize("updates", [
    {"precision": "fp32"}, {"backend": "trt_rtx"}, {"task": "image_generation"},
    {"quantization": "fp8"}, {"max_batch_size": 2}, {"tensor_parallel_size": 2},
    {"context_parallel_size": 2}, {"dynamic_kv_cache": True}, {"fp32_layers": (0,)},
    {"graph_transform": lambda *args: None}, {"image_height": 32},
    {"image_width": 32}, {"video_num_frames": 1},
])
def test_unmapped_build_options_stay_native(inputs, updates):
    request, raw = inputs
    assert not dispatch.candidate(replace(request, **updates), raw)


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3", "qwen3_5_moe", "llama"])
def test_old_or_other_model_types_are_not_edge_candidates(inputs, model_type):
    request, raw = inputs
    raw["model_type"] = model_type
    assert not dispatch.candidate(request, raw)


@pytest.mark.parametrize("updates", [
    {"linear_key_head_dim": 64}, {"linear_value_head_dim": 64}, {"num_experts": 8},
    {"quantization_config": {"quant_method": "awq"}},
])
def test_unsupported_checkpoint_profiles_do_not_dispatch(inputs, updates):
    request, raw = inputs
    raw["text_config"].update(updates)
    assert not dispatch.candidate(request, raw)


def test_family_variant_marker_is_not_cross_owned(inputs):
    request, raw = inputs
    if NEWER_VARIANT:
        raw["text_config"].pop("output_gate_type")
    else:
        raw["text_config"]["output_gate_type"] = "sigmoid"
    assert not dispatch.candidate(request, raw)


def test_nonmatch_calls_native_once_even_when_native_fails(inputs, monkeypatch):
    request, _ = inputs
    request = replace(request, precision="fp32")
    failure = RuntimeError("native failed independently")
    native = Mock(side_effect=failure)
    discover = Mock(side_effect=AssertionError("must not discover Edge"))
    monkeypatch.setattr(edge_llm, "local_target", discover)
    writer = object()
    with pytest.raises(RuntimeError) as caught:
        dispatch.build(request, writer, native)
    assert caught.value is failure
    assert caught.value.__cause__ is None
    native.assert_called_once_with(request, writer)
    discover.assert_not_called()


def test_gpu_nonmatch_is_silent_native_once(inputs, monkeypatch, caplog):
    request, _ = inputs
    monkeypatch.setattr(edge_llm, "local_target", lambda: {**TARGET, "sm": 75})
    native, writer = Mock(), object()
    dispatch.build(request, writer, native)
    native.assert_called_once_with(request, writer)
    assert not caplog.records
    assert not list(request.output_path.parent.glob(".*.edge-*.log"))


@pytest.mark.parametrize("invalid", [[], {"text_config": []}])
def test_invalid_common_config_does_not_attempt_either_builder(inputs, invalid, monkeypatch):
    request, _ = inputs
    (request.model_dir / "config.json").write_text(json.dumps(invalid))
    native, discovery = Mock(), Mock()
    monkeypatch.setattr(edge_llm, "local_target", discovery)
    with pytest.raises(ValueError):
        dispatch.build(request, object(), native)
    native.assert_not_called()
    discovery.assert_not_called()


@pytest.mark.parametrize("failure", [FileNotFoundError("Edge missing"), RuntimeError("Edge failed")])
def test_edge_failure_warns_before_one_unchanged_native_retry(inputs, monkeypatch, caplog, failure):
    request, _ = inputs
    prepare = Mock(side_effect=failure)
    route(monkeypatch, prepare)
    writer = object()
    calls = []

    def native(actual_request, actual_writer):
        assert actual_request is request and actual_writer is writer
        assert any(record.levelno == logging.WARNING and "Retrying native once" in record.message
                   for record in caplog.records)
        calls.append(actual_request)

    dispatch.build(request, writer, native)
    assert calls == [request]
    prepare.assert_called_once()
    logs = list(request.output_path.parent.glob(".*.edge-*.log"))
    assert len(logs) == 1 and str(failure) in logs[0].read_text()


def test_native_failure_preserves_original_edge_cause(inputs, monkeypatch):
    request, _ = inputs
    edge_failure = RuntimeError("upstream builder")
    native_failure = ValueError("native builder")
    route(monkeypatch, Mock(side_effect=edge_failure))
    native = Mock(side_effect=native_failure)
    with pytest.raises(ValueError) as caught:
        dispatch.build(request, object(), native)
    assert caught.value is native_failure
    assert caught.value.__cause__ is edge_failure
    native.assert_called_once()


@pytest.mark.parametrize("cancel", [KeyboardInterrupt(), SystemExit(130)])
def test_cancellation_never_falls_back(inputs, monkeypatch, cancel):
    request, _ = inputs
    route(monkeypatch, Mock(side_effect=cancel))
    native = Mock()
    with pytest.raises(type(cancel)):
        dispatch.build(request, object(), native)
    native.assert_not_called()


def test_success_publishes_once_without_native(inputs, monkeypatch):
    request, _ = inputs
    files, marker = {}, {"version": 1}
    route(monkeypatch, Mock(return_value=(files, marker)))
    publish, native, writer = Mock(), Mock(), object()
    monkeypatch.setattr(edge_llm, "publish", publish)
    dispatch.build(request, writer, native)
    publish.assert_called_once_with(request, writer, files, marker)
    native.assert_not_called()


def test_publication_failure_does_not_retry_native(inputs, monkeypatch):
    request, _ = inputs
    route(monkeypatch, Mock(return_value=({}, {})))
    failure = OSError("disk full during publication")
    monkeypatch.setattr(edge_llm, "publish", Mock(side_effect=failure))
    native = Mock()
    with pytest.raises(OSError) as caught:
        dispatch.build(request, object(), native)
    assert caught.value is failure
    native.assert_not_called()


@pytest.fixture
def package(tmp_path, monkeypatch):
    prefix = tmp_path / "install"
    (prefix / "share/trtmc").mkdir(parents=True)
    (prefix / "bin").mkdir()
    (prefix / "lib").mkdir()
    (prefix / "bin/python").write_bytes(b"python")
    (prefix / "lib/plugin.so").write_bytes(b"plugin")
    metadata = {"schema_version": 1, "version": "0.10.1", "revision": edge_llm.EDGE_REVISION,
                "arch": "x86_64", "architectures": [80], "cuda_version": "13.3",
                "tensorrt_version": TARGET["tensorrt_version"],
                "python": "bin/python", "plugin": "lib/plugin.so"}
    manifest = prefix / "share/trtmc/edge-llm.json"
    manifest.write_text(json.dumps(metadata))
    monkeypatch.setattr(edge_llm, "cmake_prefixes", lambda: [prefix])
    return prefix, manifest, metadata


def test_package_resolution_uses_pinned_contained_artifacts(package):
    prefix, _, _ = package
    found = edge_llm.installed_package(TARGET)
    assert found["python"] == str(prefix / "bin/python")
    assert found["plugin"] == str(prefix / "lib/plugin.so")


@pytest.mark.parametrize("updates", [
    {"schema_version": 2}, {"version": "0.10.0"}, {"revision": "main"},
    {"arch": "aarch64"}, {"architectures": [86]}, {"cuda_version": "13.2"},
    {"tensorrt_version": "11.1.0.105"}, {"python": "/usr/bin/python3"},
    {"plugin": "../outside.so"},
])
def test_package_pin_platform_and_path_mismatches_are_rejected(package, updates):
    _, manifest, metadata = package
    manifest.write_text(json.dumps({**metadata, **updates}))
    with pytest.raises(ValueError):
        edge_llm.installed_package(TARGET)


def test_package_symlink_escape_is_rejected(package, tmp_path):
    prefix, _, _ = package
    target = tmp_path / "outside"
    target.write_bytes(b"outside")
    plugin = prefix / "lib/plugin.so"
    plugin.unlink()
    plugin.symlink_to(target)
    with pytest.raises(ValueError, match="contained"):
        edge_llm.installed_package(TARGET)


def test_missing_package_and_artifact_fail_without_installing(package):
    prefix, manifest, _ = package
    (prefix / "lib/plugin.so").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        edge_llm.installed_package(TARGET)
    manifest.unlink()
    with pytest.raises(FileNotFoundError, match="not installed"):
        edge_llm.installed_package(TARGET)


def fake_upstream(command, *, check, stdout, stderr, cwd):
    assert check is True and stderr is subprocess.STDOUT
    assert command[1:4] == ["-I", "-c", "from experimental.builder.cli import main; main()"]
    options = dict(zip(command[4::2], command[5::2]))
    assert options["--components"] == "llm"
    assert options["--max-input-len"] == options["--max-kv-cache-capacity"] == "1024"
    assert options["--max-batch-size"] == "1"
    assert options["--dense"] == "fp16"
    assert options["--externalize-weights"] == "all"
    checkpoint = Path(options["--model-dir"])
    assert checkpoint.is_relative_to(cwd)
    assert (checkpoint / "model.safetensors").read_bytes() == b"original checkpoint contents"
    engine = Path(options["--engine-dir"])
    engine.mkdir(parents=True)
    for name in ENGINE_FILES:
        (engine / name).write_bytes(b"upstream artifact")
    stdout.write("upstream API called\n")
    return subprocess.CompletedProcess(command, 0)


def test_prepare_maps_api_and_preserves_private_checkpoint(inputs, package, tmp_path, monkeypatch):
    request, raw = inputs
    upstream = Mock(side_effect=fake_upstream)
    monkeypatch.setattr(edge_llm.subprocess, "run", upstream)
    staging = tmp_path / "stage"
    files, marker = edge_llm.prepare(request, raw, TARGET, staging, tmp_path / "edge.log")
    upstream.assert_called_once()
    assert upstream.call_args.args[0][0] == str(package[0] / "bin/python")
    assert marker["target"] == TARGET and marker["edge_revision"] == edge_llm.EDGE_REVISION
    assert marker["artifacts"] == list(files)
    assert marker["precision"] == "fp16" and marker["max_sequence_length"] == 1024
    assert all(name.startswith("edge_llm/") for name in files)
    assert "edge_llm/checkpoint/unrelated.log" not in files
    shutil.rmtree(request.model_dir)
    assert files["edge_llm/checkpoint/model.safetensors"].read_bytes() == b"original checkpoint contents"


@pytest.mark.parametrize("missing", ENGINE_FILES)
def test_incomplete_upstream_output_is_rejected(inputs, package, tmp_path, monkeypatch, missing):
    request, raw = inputs

    def incomplete(command, **kwargs):
        result = fake_upstream(command, **kwargs)
        Path(command[command.index("--engine-dir") + 1], missing).unlink()
        return result

    monkeypatch.setattr(edge_llm.subprocess, "run", incomplete)
    with pytest.raises(ValueError, match="required artifact"):
        edge_llm.prepare(request, raw, TARGET, tmp_path / "stage", tmp_path / "edge.log")


def test_upstream_process_failure_propagates(inputs, package, tmp_path, monkeypatch):
    request, raw = inputs
    failure = subprocess.CalledProcessError(1, ["edge"], output="compile failed")
    monkeypatch.setattr(edge_llm.subprocess, "run", Mock(side_effect=failure))
    with pytest.raises(subprocess.CalledProcessError) as caught:
        edge_llm.prepare(request, raw, TARGET, tmp_path / "stage", tmp_path / "edge.log")
    assert caught.value is failure


def test_publish_streams_self_contained_bundle(inputs, package, tmp_path, monkeypatch):
    request, raw = inputs
    monkeypatch.setattr(edge_llm.subprocess, "run", fake_upstream)
    files, marker = edge_llm.prepare(request, raw, TARGET, tmp_path / "stage", tmp_path / "edge.log")
    writer = BundleWriter(request.output_path)
    real_copy = shutil.copyfileobj
    copies = []

    def stream(source, destination, length):
        assert length == 1024 * 1024
        copies.append(source.name)
        real_copy(source, destination, length)

    monkeypatch.setattr(edge_llm.shutil, "copyfileobj", stream)
    edge_llm.publish(request, writer, files, marker)
    assert len(copies) == len(files)
    monkeypatch.setattr(edge_llm.shutil, "copyfileobj", real_copy)
    writer.finish()
    shutil.rmtree(request.model_dir)
    shutil.rmtree(tmp_path / "stage")
    with request.output_path.open("rb") as bundle:
        assert bundle.read(8) == b"BUNDLE\x01\x00"
        header = json.loads(bundle.read(struct.unpack("<Q", bundle.read(8))[0]))
        payload = bundle.read()
    assert header["family"] == FAMILY and "engine.plan" not in header["sections"]
    for name, expected in (("edge_llm/checkpoint/model.safetensors", b"original checkpoint contents"),
                           ("edge_llm/engine/llm.engine", b"upstream artifact")):
        section = header["sections"][name]
        assert payload[section["offset"]:section["offset"] + section["length"]] == expected
    section = header["sections"]["edge_llm.json"]
    assert json.loads(payload[section["offset"]:section["offset"] + section["length"]]) == marker


def test_public_family_build_invokes_family_dispatch(inputs, monkeypatch):
    pytest.importorskip("tensorrt")
    model = importlib.import_module(f"families.{FAMILY}.model")
    request, _ = inputs
    adapter = Mock()
    monkeypatch.setattr(dispatch, "build", adapter)
    writer = object()
    model.build(request, writer)
    adapter.assert_called_once_with(request, writer, model._build_native)


@pytest.mark.parametrize("name", ["hf_quant_config.json", "quantize_config.json", "quant_config.json"])
def test_quantized_checkpoint_sidecars_stay_native(inputs, name):
    request, raw = inputs
    (request.model_dir / name).write_text(json.dumps({"quantization": {"quant_algo": "FP8"}}))
    assert not dispatch.candidate(request, raw)


@pytest.mark.parametrize("model_type", ["qwen3.5", "qwen3_8", "qwen3.8"])
def test_unrecognized_upstream_aliases_do_not_dispatch(inputs, model_type):
    request, raw = inputs
    raw["model_type"] = model_type
    assert not dispatch.candidate(request, raw)


def test_platform_discovery_failure_warns_and_retries_once(inputs, monkeypatch, caplog):
    request, _ = inputs
    failure = RuntimeError("CUDA discovery unavailable")
    monkeypatch.setattr(edge_llm, "local_target", Mock(side_effect=failure))
    native, writer = Mock(), object()
    dispatch.build(request, writer, native)
    native.assert_called_once_with(request, writer)
    assert any("CUDA discovery unavailable" in record.message for record in caplog.records)


def test_map_contains_only_explicit_supported_native_platforms():
    expected = {
        ("linux", arch, sm, "fp16")
        for arch, sms in (("x86_64", (80, 86, 100, 120)), ("aarch64", (87, 110, 121)))
        for sm in sms
    }
    assert set(dispatch.EDGE_DISPATCH) == expected


def test_missing_checkpoint_weights_never_invokes_upstream(inputs, package, tmp_path, monkeypatch):
    request, raw = inputs
    (request.model_dir / "model.safetensors").unlink()
    upstream = Mock()
    monkeypatch.setattr(edge_llm.subprocess, "run", upstream)
    with pytest.raises(ValueError, match="safetensors"):
        edge_llm.prepare(request, raw, TARGET, tmp_path / "stage", tmp_path / "edge.log")
    upstream.assert_not_called()


def test_upstream_symlink_output_is_rejected(inputs, package, tmp_path, monkeypatch):
    request, raw = inputs

    def linked(command, **kwargs):
        result = fake_upstream(command, **kwargs)
        engine = Path(command[command.index("--engine-dir") + 1])
        (engine / "unexpected.bin").symlink_to(request.model_dir / "model.safetensors")
        return result

    monkeypatch.setattr(edge_llm.subprocess, "run", linked)
    with pytest.raises(ValueError, match="symlinks"):
        edge_llm.prepare(request, raw, TARGET, tmp_path / "stage", tmp_path / "edge.log")


@pytest.mark.parametrize("capacity", [None, True, 1024.5, "1024", 0, -1, 8])
def test_invalid_candidate_context_fails_before_either_builder(inputs, monkeypatch, caplog, capacity):
    request, raw = inputs
    raw["text_config"]["max_position_embeddings"] = capacity
    (request.model_dir / "config.json").write_text(json.dumps(raw))
    native, discovery = Mock(), Mock()
    monkeypatch.setattr(edge_llm, "local_target", discovery)
    with pytest.raises(ValueError):
        dispatch.build(request, object(), native)
    native.assert_not_called()
    discovery.assert_not_called()
    assert not caplog.records
    assert not list(request.output_path.parent.glob(".*.edge-*.log"))


def test_noncandidate_context_validation_remains_native_owned(inputs, monkeypatch):
    request, raw = inputs
    request = replace(request, precision="fp32")
    raw["text_config"]["max_position_embeddings"] = "native-specific metadata"
    (request.model_dir / "config.json").write_text(json.dumps(raw))
    writer, native = object(), Mock()
    discovery = Mock(side_effect=AssertionError("not an Edge candidate"))
    monkeypatch.setattr(edge_llm, "local_target", discovery)
    dispatch.build(request, writer, native)
    native.assert_called_once_with(request, writer)
    discovery.assert_not_called()
