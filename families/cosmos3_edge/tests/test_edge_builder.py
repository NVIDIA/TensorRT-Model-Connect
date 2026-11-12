# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owning adapter mechanics; mocked builds are not model qualification."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorrt_model_connect.build import BuildRequest
from families.cosmos3_edge.edge_llm import builder, dispatch
from families.cosmos3_edge.tests.test_build import Writer


@pytest.fixture
def request_(tmp_path, monkeypatch):
    request = BuildRequest(model_dir=tmp_path, output_path=tmp_path / "model.bundle",
                           family="cosmos3_edge", task="image_video_text_to_text",
                           precision="fp16", max_sequence_length=128)
    monkeypatch.setattr(dispatch.sys, "platform", "linux")
    monkeypatch.setattr(builder, "package_present", lambda: True)
    monkeypatch.setattr(builder, "local_target", lambda: {"sm": 89})
    return request


@pytest.mark.parametrize("reason", ["no_sdk", "fp32", "non_linux"])
def test_nonmatches_use_unchanged_native_request(request_, monkeypatch, reason):
    if reason == "no_sdk":
        monkeypatch.setattr(builder, "package_present", lambda: False)
    elif reason == "fp32":
        request_ = replace(request_, precision="fp32")
    else:
        monkeypatch.setattr(dispatch.sys, "platform", "win32")
    monkeypatch.setattr(builder, "prepare", lambda *a: pytest.fail("unexpected Edge preparation"))
    calls = []
    writer = Writer()
    dispatch.build(request_, writer, lambda r, w: calls.append((r, w)), 128)
    assert calls == [(request_, writer)]
    assert not writer.sections


def test_preparation_failure_warns_and_retries_original_request_once(request_, monkeypatch, caplog):
    def fail(*args):
        raise RuntimeError("upstream builder error")
    monkeypatch.setattr(builder, "prepare", fail)
    calls = []
    writer = Writer()
    dispatch.build(request_, writer, lambda r, w: calls.append((r, w)), 128)
    assert calls == [(request_, writer)]
    assert writer.header is None
    assert "unchanged request" in caplog.text
    logs = list(request_.output_path.parent.glob(".*.log"))
    assert len(logs) == 1
    assert "upstream builder error" in logs[0].read_text()


def test_success_publishes_without_native_or_sm_whitelist(request_, monkeypatch):
    path = request_.model_dir / "engine"
    path.write_bytes(b"engine")
    monkeypatch.setattr(builder, "prepare", lambda *a: ({"edge_llm/engine/llm.engine": path}, {"version": 1}))
    writer = Writer()
    dispatch.build(request_, writer, lambda *a: pytest.fail("native retry"), 128)
    assert writer.sections["edge_llm/engine/llm.engine"] == b"engine"
    assert writer.sections["edge_llm.json"] == {"version": 1}
    assert not list(request_.model_dir.glob(".*.log"))


def test_publication_error_is_not_retried(request_, monkeypatch):
    monkeypatch.setattr(builder, "prepare", lambda *a: ({}, {}))
    def fail(*args):
        raise OSError("bundle publication failed")
    monkeypatch.setattr(builder, "publish", fail)
    with pytest.raises(OSError, match="publication failed"):
        dispatch.build(request_, Writer(), lambda *a: pytest.fail("native retry"), 128)


def test_cancelled_build_is_not_retried(request_, monkeypatch):
    def fail(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(builder, "prepare", fail)
    with pytest.raises(KeyboardInterrupt):
        dispatch.build(request_, Writer(), lambda *a: pytest.fail("native retry"), 128)


def test_nested_checkpoint_packaged_without_copying_shards(request_, monkeypatch, tmp_path):
    root = request_.model_dir
    shard = root / "transformer/part.safetensors"
    shard.parent.mkdir()
    shard.write_bytes(b"weights")
    (root / "config.json").write_text('{"model_type": "cosmos3_edge"}')
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"embed_tokens.weight": "transformer/part.safetensors"}}))
    ignored = root / "vae/policy.safetensors"
    ignored.parent.mkdir()
    ignored.write_bytes(b"not part of reasoning")
    monkeypatch.setattr(builder, "installed_package", lambda t: {"python": "/sdk/python", "plugin": "/sdk/plugin"})
    monkeypatch.setattr(builder, "builder_python", lambda p, t: p["python"])
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        engine = Path(command[command.index("--engine-dir") + 1])
        for name in ("llm.engine", "config.json", "tokenizer.json", "tokenizer_config.json",
                     "chat_template.jinja", "visual/visual.engine", "visual/config.json"):
            p = engine / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"artifact")
    monkeypatch.setattr(builder.subprocess, "run", run)
    staging = tmp_path / "staging"
    staging.mkdir()
    files, marker = builder.prepare(request_, {"sm": 89}, staging, tmp_path / "build.log", 128)
    command = commands[0]
    assert command[command.index("--model-dir") + 1] == str(root.resolve())
    assert command[command.index("--components") + 1] == "llm,visual"
    assert command[command.index("--externalize-weights") + 1] == "all"
    assert files["edge_llm/checkpoint/transformer/part.safetensors"] == shard
    assert not (staging / "edge_llm/checkpoint").exists()
    assert not any("vae/" in name for name in files)
    assert marker["max_sequence_length"] == marker["max_input_length"] == 128
    assert marker["artifacts"] == list(files)


@pytest.mark.parametrize("compatible", [True, False])
def test_wheel_or_source_python_uses_same_sdk_contract(monkeypatch, compatible):
    monkeypatch.setattr(builder.subprocess, "run", lambda *a, **k: SimpleNamespace(
        stdout=json.dumps(["0.11.0" if compatible else "0.10.0", "11.1.0.106"])))
    selected = builder.builder_python({"python": "/sdk/python"}, {"tensorrt_version": "11.1.0.106"})
    assert selected == (builder.sys.executable if compatible else "/sdk/python")
