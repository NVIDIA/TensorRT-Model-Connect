# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from types import SimpleNamespace

import numpy as np

from families.internvl import model
from families.internvl.checkpoint_mapper import WeightDict
from families.internvl.parallel import ParallelConfig, shard_standard_decoder_weights
from families.internvl.tests.test_e2e import CASES, _official_prompt


def test_swiglu_weights_are_sharded_for_each_rank() -> None:
    config = SimpleNamespace(
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=8,
    )
    weights = WeightDict(
        {
            "layer.0.w_q": np.arange(32, dtype=np.float32).reshape(4, 8),
            "layer.0.w_o": np.arange(32, dtype=np.float32).reshape(8, 4),
            "layer.0.w_gate": np.arange(32, dtype=np.float32).reshape(4, 8),
            "layer.0.w_up": np.arange(32, dtype=np.float32).reshape(4, 8),
            "layer.0.w_down": np.arange(32, dtype=np.float32).reshape(8, 4),
            "_attention_size": 8,
            "_kv_attention_size": 8,
            "_mlp_size": 8,
        }
    )
    sharded = shard_standard_decoder_weights(config, weights, ParallelConfig(2, 1))
    assert sharded["layer.0.w_q"].shape == (4, 4)
    assert sharded["layer.0.w_o"].shape == (4, 4)
    assert sharded["layer.0.w_gate"].shape == (4, 4)
    assert sharded["layer.0.w_up"].shape == (4, 4)
    assert sharded["layer.0.w_down"].shape == (4, 4)
    assert sharded["_attention_size"] == 4
    assert sharded["_kv_attention_size"] == 4
    assert sharded["_mlp_size"] == 4


def test_build_emits_one_dual_profile_plan_per_rank(monkeypatch, tmp_path) -> None:
    config = SimpleNamespace(
        model_type="internvl",
        max_position_embeddings=4096,
        num_hidden_layers=2,
        vocab_size=32,
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=8,
        raw={},
    )
    ranks = []

    class FakeModel:
        @staticmethod
        def load_weights(_model_dir, _config):
            return WeightDict()

        @staticmethod
        def build_engine(_config, _weights, _length, **kwargs):
            parallel = kwargs["parallel_config"]
            ranks.append(parallel.rank)
            return f"rank-{parallel.rank}".encode()

        @staticmethod
        def build_vision_engine(*_args, **_kwargs):
            return b"vision"

        @staticmethod
        def get_vl_config(_config):
            return {"image_token_id": 7, "vision_output_dim": 8, "prefill_max_length": 16}

    class Writer:
        def __init__(self):
            self.sections = {}

        @staticmethod
        def set_header(**_kwargs):
            return None

        def add_bytes(self, name, value):
            self.sections[name] = value

        def add_json(self, name, value):
            self.sections[name] = value

    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _path: config)
    monkeypatch.setattr(model, "_InternVLModel", FakeModel)
    monkeypatch.setattr(model, "_tokenizer_runtime_contract", lambda _path: {})
    writer = Writer()
    request = SimpleNamespace(
        family="internvl", output_path=tmp_path / "model.bundle", graph_transform=None,
        model_dir=tmp_path,
        backend="trt",
        dynamic_kv_cache=False,
        image_height=None,
        image_width=None,
        video_num_frames=None,
        max_batch_size=1,
        context_parallel_size=1,
        task="vision_language_generation",
        tensor_parallel_size=2,
        quantization=None,
        fp32_layers=(),
        precision="fp16",
        max_sequence_length=16,
        verbose=False,
    )

    model.build(request, writer)

    assert ranks == [0, 1]
    assert writer.sections["engine.rank0.plan"] == b"rank-0"
    assert writer.sections["engine.rank1.plan"] == b"rank-1"
    assert "engine.plan" not in writer.sections
    assert "prefill.plan" not in writer.sections
    assert writer.sections["runtime.json"]["tensor_parallel_size"] == 2
    assert writer.sections["vision.plan"] == b"vision"

    ranks.clear()
    request.tensor_parallel_size = 1
    writer = Writer()
    model.build(request, writer)
    assert ranks == [-1, -1]
    assert writer.sections["engine.plan"] == b"rank--1"
    assert writer.sections["prefill.plan"] == b"rank--1"
    assert not any(name.startswith("engine.rank") for name in writer.sections)
    assert writer.sections["runtime.json"]["tensor_parallel_size"] == 1


def test_tp_manifests_remain_active() -> None:
    assert CASES["internvl3-2b-tp2"][1]["tensor_parallel_size"] == 2
    assert CASES["internvl3-8b-tp4"][1]["tensor_parallel_size"] == 4


def test_official_prompt_adds_image_placeholder_without_changing_user_text() -> None:
    user_prompt = "What color is the vehicle?"

    class Processor:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            assert messages[0]["content"][1] == {"type": "text", "text": user_prompt}
            return f"<IMG_CONTEXT>\n{user_prompt}\nassistant"

    prompt = _official_prompt(Processor(), user_prompt)
    assert prompt.count(user_prompt) == 1
    assert "<IMG_CONTEXT>" in prompt


def test_ordinary_cli_keeps_edge_selection_in_the_family(tmp_path, monkeypatch):
    import json
    from tensorrt_model_connect import family_cli as build_cli
    from families.internvl.edge_llm import dispatch

    source = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "internvl"}))
    output = tmp_path / "model.bundle"
    seen = []

    def select(request, writer, native):
        assert callable(native)
        assert request.family == "internvl"
        assert request.task == "vision_language_generation"
        seen.append(request)
        writer.set_header(family=request.family, task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"family": request.family})

    monkeypatch.setattr(dispatch, "build", select)
    assert build_cli.main(["internvl", "build", str(source), "-o", str(output)]) == 0
    assert len(seen) == 1
    assert output.is_file()


def test_internvl_does_not_register_unowned_companion_options():
    from tensorrt_model_connect.model_support import ModelMetadata
    from families.internvl.support import describe

    support = describe(ModelMetadata({"model_type": "internvl"}, {}))
    assert support is not None
    from tensorrt_model_connect.family_cli import load_family_cli
    declaration = load_family_cli("internvl")
    flags = {flag for argument in declaration["commands"][0]["arguments"]
             for flag in argument.get("flags", [])}
    assert "--execution-variant" not in flags
    assert "--companion" not in flags


@pytest.mark.parametrize("mode", ["absent", "success", "corrupt", "failure", "cancel", "device_failure"])
def test_edge_optional_package_and_output_local_staging(tmp_path, monkeypatch, caplog, mode):
    import json
    from tensorrt_model_connect.build import BuildRequest
    from families.internvl.edge_llm import builder, dispatch

    source = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "max_position_embeddings": 4096, "hidden_size": 896,
    }))
    prefix = tmp_path / "package"
    manifest = prefix / "share/trtmc/edge-llm.json"
    if mode != "absent":
        manifest.parent.mkdir(parents=True)
        manifest.write_text("{" if mode == "corrupt" else "{}")
    monkeypatch.setattr(builder, "cmake_prefixes", lambda: [prefix])
    monkeypatch.setattr(dispatch, "candidate", lambda *_: True)
    for name in ("mapped_request", "request_matches", "platform_matches"):
        if hasattr(dispatch, name):
            monkeypatch.setattr(dispatch, name, lambda *_: True)
    if hasattr(dispatch, "source_quantization"):
        monkeypatch.setattr(dispatch, "source_quantization", lambda *_: "fp16")
    if hasattr(builder, "request_weight_format"):
        monkeypatch.setattr(builder, "request_weight_format", lambda *_: "fp16")
    request = BuildRequest(source, tmp_path / "out", "internvl", "vision_language_generation", "fp16")
    writer = object()
    target = {"os": "linux", "arch": "x86_64", "sm": 80}
    target_calls = []

    def local_target():
        target_calls.append(True)
        if mode == "device_failure":
            raise RuntimeError("CUDA discovery failed")
        return target

    monkeypatch.setattr(builder, "local_target", local_target)
    stages, native_calls, publications = [], [], []

    def prepare(original, raw, platform, staging, log):
        assert original is request and platform is target
        assert staging.parent == request.output_path.parent
        assert staging.name.startswith(f".{request.output_path.name}.edge-")
        stages.append(staging)
        (staging / "large-checkpoint").write_bytes(b"fixture")
        if mode == "corrupt":
            builder.installed_package(target)
        if mode == "failure":
            raise FileNotFoundError("installed SDK artifact missing")
        if mode == "cancel":
            raise KeyboardInterrupt()
        return {}, {}

    monkeypatch.setattr(dispatch, "EDGE_DISPATCH", {("linux", "x86_64", 80, "fp16"): prepare})
    monkeypatch.setattr(builder, "publish", lambda *args: publications.append(args))
    def native(*args):
        native_calls.append(args)
    if mode == "cancel":
        with pytest.raises(KeyboardInterrupt):
            dispatch.build(request, writer, native)
    else:
        dispatch.build(request, writer, native)
    assert all(not path.exists() for path in stages)
    assert native_calls == ([(request, writer)] if mode in {
        "absent", "corrupt", "failure", "device_failure",
    } else [])
    assert len(publications) == (1 if mode == "success" else 0)
    assert len(target_calls) == (0 if mode == "absent" else 1)
    logs = list(tmp_path.glob(".out.edge-*.log"))
    if mode in {"corrupt", "failure", "device_failure"}:
        assert len(logs) == 1 and "Traceback" in logs[0].read_text()
        assert "Retrying native once" in caplog.text
    else:
        assert not logs and "Edge build failed" not in caplog.text


@pytest.mark.parametrize("options", [[], ["--precision", "fp16", "--max-sequence-length", "64"]])
def test_declared_build_matches_legacy_request(tmp_path, monkeypatch, options):
    """Owner command preserves ordinary request defaults and explicit controls."""
    import json
    from families.internvl import cli as owner
    from tensorrt_model_connect import build_cli, family_cli

    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "internvl"}))
    output = tmp_path / "model.bundle"
    captured = []
    monkeypatch.setattr(owner, "build_bundle", lambda request, output: captured.append(request))
    monkeypatch.setattr(build_cli, "build", captured.append)
    args = [str(source), "-o", str(output), *options]
    assert family_cli.main(["internvl", "build", *args]) == 0
    assert build_cli.main(["build", *args, "--family", "internvl"]) == 0
    assert len(captured) == 2
    from dataclasses import fields
    assert isinstance(captured[0], owner.BuildRequest)
    for field in fields(captured[1]):
        assert getattr(captured[0], field.name) == getattr(captured[1], field.name)
    from dataclasses import replace
    from families.internvl.build_request import coerce_request
    assert coerce_request(captured[1]) == captured[0]
    assert coerce_request(replace(captured[1], fp32_layers=[])) == captured[0]
    with pytest.raises(NotImplementedError, match="fp32_layers"):
        coerce_request(replace(captured[1], fp32_layers=[0]))
    with pytest.raises(NotImplementedError, match="image_height"):
        coerce_request(replace(captured[1], image_height=32))
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="unknown"):
        coerce_request(SimpleNamespace(**vars(captured[1]), unexpected_option=True))
    assert captured[0].family == "internvl"
    assert captured[0].task == "vision_language_generation"
    assert captured[0].precision == ("fp16" if options else "fp32")
    assert not output.exists()


def test_declared_help_is_offline_and_dependency_free():
    """Actual child-process help needs neither a checkpoint nor GPU imports."""
    import subprocess
    import sys

    code = """
import sys
from tensorrt_model_connect.family_cli import main
try:
    main(["internvl", "build", "--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("help did not exit")
assert "families.internvl.cli" not in sys.modules
assert "tensorrt" not in sys.modules
assert "huggingface_hub" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "trtmc internvl build" in result.stdout
