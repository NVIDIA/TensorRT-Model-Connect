# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect import trt_compat
from tensorrt_model_connect.families.minimax_h3 import staged_build
from tensorrt_model_connect.families.minimax_h3.plugin import plugin
from tests.builder.conftest import read_bundle_file


class _StreamWriterBase:
    def __init__(self) -> None:
        pass


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
        trt_compat.build_serialized_network_to_file(
            FailingBuilder(), object(), object(), output
        )
    assert output.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".engine.plan.tmp.*"))


def test_staged_build_uses_six_fresh_children_resumes_and_sanitizes_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    tokenizer = model / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text('{"model": {}}', encoding="utf-8")
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
        plan.write_bytes(f"plan:{component}".encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(staged_build.subprocess, "run", run)
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_version", lambda: "1.6.1.120")
    monkeypatch.setattr(staged_build.trt_compat, "tensorrt_abi", lambda _version: "1.6")

    assert staged_build.build_staged_bundle(model, output) == output
    assert [call[call.index("--component") + 1] for call in calls] == [
        item[0] for item in staged_build._COMPONENTS
    ]

    header, sections = read_bundle_file(str(output))
    config = json.loads(sections["config.json"])
    assert header["gpu_name"] == ""
    assert config["engine_backend"] == "trt_rtx"
    assert config["cuda_major"] == 12
    assert config["runtime_memory"] == {
        "mode": "staged",
        "weight_streaming_budget_bytes": 32 << 30,
    }
    assert set(config["plan_sha256"]) == {
        item[1] for item in staged_build._COMPONENTS
    }
    assert all(
        len(value) == 64 and value == value.lower()
        for value in config["plan_sha256"].values()
    )
    serialized = json.dumps(config).lower()
    assert str(tmp_path).lower() not in serialized
    for forbidden in (
        "hostname",
        "source_revision",
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
    (plans / "denoiser_tail.plan").write_bytes(b"corrupt")
    staged_build.build_staged_bundle(model, output)
    assert [call[call.index("--component") + 1] for call in calls] == ["denoiser_tail"]

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
    output = tmp_path / "h3.bundle"

    def build(component: str, _model: Path, plan: Path, *, verbose: bool) -> None:
        assert verbose is False
        plan.write_bytes(component.encode())

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
        "model_metadata_sha256",
        "checkpoint_shards",
        "builder_sha256",
        "backend",
        "trt_version",
        "trt_abi",
        "cuda_major",
        "workspace_limit_bytes",
        "weight_streaming_budget_bytes",
    }
    assert str(tmp_path) not in json.dumps(receipt)
    assert all(set(record) == {"bytes", "sha256"} for record in receipt["plans"].values())


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

    assert plugin.build_staged_bundle(
        str(model),
        str(output),
        config,
        {"_model_dir": str(model)},
        precision="bf16",
        parallel_config=SimpleNamespace(mode="single"),
    ) == output
    assert calls == [((model, str(output)), {"verbose": False})]

    with pytest.raises(ValueError, match="require BF16"):
        plugin.build_staged_bundle(
            str(model), str(output), config, {}, precision="fp16"
        )
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
