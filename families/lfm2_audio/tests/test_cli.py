# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of family command defaults and bundle completion/abort."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from families.lfm2_audio import cli
from tensorrt_model_connect.family_cli import load_family_cli


def test_build_command_owns_its_supported_options():
    declaration = load_family_cli("lfm2_audio")
    command = next(item for item in declaration["commands"] if item["name"] == "build")
    options = {item["name"]: item for item in command["arguments"]}
    assert command["handler"] == "cli:build"
    assert options["task"]["default"] == "offline_speech_dialogue"
    assert options["precision"]["choices"] == ["bf16"]
    assert options["max_sequence_length"]["default"] == 32768
    assert not {"quantization", "tensor_parallel_size", "dynamic_kv_cache"} & options.keys()


@pytest.mark.parametrize("failure", [False, True])
def test_build_command_finishes_or_aborts_bundle(monkeypatch, tmp_path, failure):
    from tensorrt_model_connect import model_support

    checkpoint = tmp_path / "checkpoint"
    calls = []

    def resolve(model, revision):
        assert (model, revision) == ("model-id", "pinned-revision")
        return checkpoint

    def build(request, writer):
        calls.append(request)
        writer.set_header(family="lfm2_audio", task=request.task, backend="trt")
        writer.add_json("probe.json", {"max_context": request.max_sequence_length})
        if failure:
            raise ValueError("engine construction failed")

    model = ModuleType("families.lfm2_audio.model")
    model.BuildRequest = SimpleNamespace
    model.build = build
    monkeypatch.setitem(sys.modules, model.__name__, model)
    monkeypatch.setattr(model_support, "resolve_model", resolve)
    output = tmp_path / "audio.bundle"
    options = {
        "model": "model-id",
        "revision": "pinned-revision",
        "output": output,
        "task": "text_to_speech",
        "max_sequence_length": 1024,
    }
    if failure:
        with pytest.raises(ValueError, match="engine construction failed"):
            cli.build(**options)
        assert not output.exists()
    else:
        assert cli.build(**options) == 0
        assert output.read_bytes().startswith(b"BUNDLE\x01\x00")
    assert len(calls) == 1
    assert calls[0].model_dir == checkpoint
    assert calls[0].task == "text_to_speech"
    assert calls[0].precision == "bf16"
    assert calls[0].max_sequence_length == 1024
