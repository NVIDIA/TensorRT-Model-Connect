# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision contract for Qwen3.5 recurrent state inputs."""

from __future__ import annotations

import pytest


trt = pytest.importorskip("tensorrt")

from families.qwen3_5 import model  # noqa: E402


def test_fp16_runtime_inputs_preserve_fp32_recurrent_state() -> None:
    """FP16 storage must not quantize the persistent DeltaNet state."""

    class _Tensor:
        def __init__(self, name: str, dtype) -> None:
            self.name = name
            self.dtype = dtype

    class _Layer:
        def __init__(self, output: _Tensor) -> None:
            self.output = output

        def get_output(self, index: int) -> _Tensor:
            assert index == 0
            return self.output

    class _Network:
        def __init__(self) -> None:
            self.cast_inputs: list[_Tensor] = []

        def add_cast(self, tensor: _Tensor, dtype):
            self.cast_inputs.append(tensor)
            return _Layer(_Tensor(f"{tensor.name}_cast", dtype))

    network = _Network()
    attention_mask = _Tensor("attention_mask", trt.float32)
    conv_state = _Tensor("conv_state", trt.float32)
    ssm_state = _Tensor("ssm_state", trt.float32)
    cache_k = _Tensor("cache_k", trt.float16)
    cache_v = _Tensor("cache_v", trt.float16)

    (
        prepared_mask,
        prepared_conv,
        prepared_ssm,
        prepared_cache_k,
        prepared_cache_v,
    ) = model._prepare_runtime_inputs(
        network,
        trt.float16,
        attention_mask,
        [conv_state],
        [ssm_state],
        [cache_k],
        [cache_v],
    )

    assert prepared_mask.dtype == trt.float16
    assert prepared_conv[0].dtype == trt.float16
    assert prepared_cache_k[0].dtype == trt.float16
    assert prepared_cache_v[0].dtype == trt.float16
    assert prepared_ssm == [ssm_state]
    assert prepared_ssm[0].dtype == trt.float32
    assert ssm_state not in network.cast_inputs


def _edge_cli_source(tmp_path):
    import json

    source = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    draft = tmp_path / "draft"
    draft.mkdir()
    return source, draft


def test_edge_cli_uses_ordinary_family_build(tmp_path, monkeypatch):
    from tensorrt_model_connect import family_cli as build_cli
    from families.qwen3_5.edge_llm import dispatch
    from families.qwen3_5.edge_llm.config import Qwen35BuildRequest

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    seen = []

    def paired(request, writer, execution):
        assert isinstance(request, Qwen35BuildRequest)
        assert request.execution is execution
        assert execution.variant == "dflash"
        assert [(item.role, item.model_dir) for item in execution.checkpoints] == [("draft", draft)]
        seen.append(request)
        writer.set_header(family="qwen3_5", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})

    monkeypatch.setattr(dispatch, "build_paired", paired)
    assert build_cli.main(["qwen3_5",
        "build", str(source), "--precision", "fp16", "-o", str(output),
        "--execution-variant", "dflash", "--companion", f"draft={draft}",
    ]) == 0
    assert len(seen) == 1
    assert output.is_file()


@pytest.mark.parametrize("options", [
    ["--companion", "draft=/missing"],
    ["--execution-variant", "dflash", "--companion", "missing_separator"],
    ["--execution-variant", "dflash", "--companion", "=path"],
    ["--execution-variant", "dflash", "--companion", "draft="],
    ["--execution-variant", "dflash", "--companion", "draft=https://example.com/model"],
])
def test_bad_edge_cli_inputs_fail_before_backend(tmp_path, monkeypatch, options):
    from tensorrt_model_connect import family_cli as build_cli

    from families.qwen3_5 import cli as owner
    source, _ = _edge_cli_source(tmp_path)
    monkeypatch.setattr(owner, "select_backend", lambda *_: pytest.fail("backend touched"))
    monkeypatch.setattr(owner, "BundleWriter", lambda *_: pytest.fail("writer created"))
    with pytest.raises(ValueError):
        build_cli.main(["qwen3_5", "build", str(source), "-o", str(tmp_path / "out"), *options])


def test_edge_cli_help_is_family_owned(tmp_path, capsys):
    from tensorrt_model_connect import family_cli as build_cli

    source, _ = _edge_cli_source(tmp_path)
    with pytest.raises(SystemExit) as caught:
        build_cli.main(["qwen3_5", "build", str(source), "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--execution-variant {dflash}" in help_text
    assert "--companion" in help_text


def test_edge_request_preserves_fields_and_family_owner(tmp_path):
    from families.qwen3_5.edge_llm import cli
    from dataclasses import fields, replace
    from tensorrt_model_connect.build import BuildRequest
    from families.qwen3_5.edge_llm.config import (
        BuildExecutionInputs, NamedCheckpoint, with_execution,
    )

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "qwen3_5", "text_generation", "fp16",
                           graph_transform=lambda layer: layer)
    execution = BuildExecutionInputs("dflash", (NamedCheckpoint("draft", draft),))
    assert cli.execution_inputs(None) is None
    extended = with_execution(request, execution)
    for field in fields(BuildRequest):
        assert getattr(extended, field.name) is getattr(request, field.name)
    with pytest.raises(ValueError, match="requires the qwen3_5 family"):
        with_execution(replace(request, family="other"), execution)
    with pytest.raises(ValueError, match="unique"):
        BuildExecutionInputs("dflash", execution.checkpoints * 2)


@pytest.mark.parametrize("failure", [RuntimeError("paired build failed"), KeyboardInterrupt()])
def test_edge_cli_failure_preserves_existing_bundle(tmp_path, monkeypatch, failure):
    from tensorrt_model_connect import family_cli as build_cli
    from families.qwen3_5.edge_llm import dispatch

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    output.write_bytes(b"previous publication")

    def fail(request, writer, execution):
        writer.set_header(family="qwen3_5", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})
        raise failure

    monkeypatch.setattr(dispatch, "build_paired", fail)
    with pytest.raises(type(failure)) as caught:
        build_cli.main(["qwen3_5",
            "build", str(source), "-o", str(output), "--execution-variant", "dflash",
            "--companion", f"draft={draft}",
        ])
    assert caught.value is failure
    assert output.read_bytes() == b"previous publication"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["draft", "pair.bundle", "target"]


def test_edge_pair_requires_draft_and_rechecks_local_inputs(tmp_path):
    from tensorrt_model_connect.build import BuildRequest
    from families.qwen3_5.edge_llm.config import BuildExecutionInputs, NamedCheckpoint
    from families.qwen3_5.edge_llm.dispatch import build_paired

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "qwen3_5", "text_generation", "fp16")
    with pytest.raises(ValueError, match="paired execution requires"):
        build_paired(request, None, BuildExecutionInputs("dflash"))
    execution = BuildExecutionInputs("dflash", (NamedCheckpoint("draft", draft),))
    draft.rmdir()
    with pytest.raises(ValueError, match="existing local directory"):
        build_paired(request, None, execution)


@pytest.mark.parametrize("mode", ["absent", "success", "corrupt", "failure", "cancel", "device_failure"])
def test_edge_optional_package_and_output_local_staging(tmp_path, monkeypatch, caplog, mode):
    import json
    from tensorrt_model_connect.build import BuildRequest
    from families.qwen3_5.edge_llm import builder, dispatch

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
    request = BuildRequest(source, tmp_path / "out", "qwen3_5", "text_generation", "fp16")
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
    from families.qwen3_5 import cli as owner
    from tensorrt_model_connect import build_cli, family_cli

    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    output = tmp_path / "model.bundle"
    captured = []
    monkeypatch.setattr(owner, "build_bundle", lambda request, output: captured.append(request))
    monkeypatch.setattr(build_cli, "build", captured.append)
    args = [str(source), "-o", str(output), *options]
    assert family_cli.main(["qwen3_5", "build", *args]) == 0
    assert build_cli.main(["build", *args, "--family", "qwen3_5"]) == 0
    assert len(captured) == 2
    from dataclasses import fields
    assert isinstance(captured[0], owner.BuildRequest)
    for field in fields(captured[1]):
        assert getattr(captured[0], field.name) == getattr(captured[1], field.name)
    from dataclasses import replace
    from families.qwen3_5.build_request import coerce_request
    assert coerce_request(captured[1]) == captured[0]
    with pytest.raises(NotImplementedError, match="image_height"):
        coerce_request(replace(captured[1], image_height=32))
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="unknown"):
        coerce_request(SimpleNamespace(**vars(captured[1]), unexpected_option=True))
    assert captured[0].family == "qwen3_5"
    assert captured[0].task == "text_generation"
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
    main(["qwen3_5", "build", "--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("help did not exit")
assert "families.qwen3_5.cli" not in sys.modules
assert "tensorrt" not in sys.modules
assert "huggingface_hub" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "trtmc qwen3_5 build" in result.stdout
