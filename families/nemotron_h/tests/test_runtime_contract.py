# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from families.nemotron_h.model import _runtime_config, _stop_token_ids


def test_runtime_keeps_a_scalar_eos_and_every_declared_stop_token(tmp_path) -> None:
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [2, 11, 12]}), encoding="utf-8"
    )
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        bos_token_id=1,
        eos_token_id=12,
        pad_token_id=0,
    )
    family_model = SimpleNamespace(get_bundle_config_overrides=lambda _config: {})

    runtime = _runtime_config(tmp_path, config, family_model)

    assert runtime["eos_token_id"] == 2
    assert runtime["stop_token_ids"] == [2, 11, 12]


@pytest.mark.parametrize("value", [[], [2, 2], [True], [32], "2"])
def test_stop_token_contract_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="eos_token_id"):
        _stop_token_ids(value, 32)


def test_runtime_keeps_checkpoint_prompt_and_builder_policies(tmp_path) -> None:
    family = Path(__file__).resolve().parents[1]
    plugin = (family / "runtime/plugin.cpp").read_text(encoding="utf-8")
    model = (family / "model.py").read_text(encoding="utf-8")
    assert 'require_text_section(bundle, "tokenizer_config.json")' in plugin
    assert 'config.find("chat_template")' in plugin
    assert "chat_template.jinja" not in plugin
    assert '"chat_template.jinja"' not in model

    # Preserve the runtime's single embedded-template contract while admitting
    # newer source checkpoints whose standalone template overrides JSON.
    from families.nemotron_h.tokenizer_assets import native_tokenizer_assets

    source_config = json.dumps({"chat_template": "old template", "extra": 7})
    (tmp_path / "tokenizer_config.json").write_text(source_config)
    names = ("tokenizer_config.json",)
    assert json.loads(native_tokenizer_assets(tmp_path, names)[names[0]])["chat_template"] == "old template"
    (tmp_path / "chat_template.jinja").write_text("standalone template")
    normalized = json.loads(native_tokenizer_assets(tmp_path, names)[names[0]])
    assert normalized == {"chat_template": "standalone template", "extra": 7}
    assert (tmp_path / "tokenizer_config.json").read_text() == source_config
    (tmp_path / "tokenizer_config.json").write_text('{"extra": 7}')
    assert json.loads(native_tokenizer_assets(tmp_path, names)[names[0]]) == normalized
    (tmp_path / "chat_template.jinja").write_text("")
    with pytest.raises(ValueError, match="chat template"):
        native_tokenizer_assets(tmp_path, names)

    for path in (family / "model.py", family / "tp_builder.py"):
        source = path.read_text(encoding="utf-8")
        assert "builder_optimization_level = 3" in source
        assert "builder_optimization_level = 1" not in source

    # Edge has native scalar prefill even when the optimized head80 SSD is absent.
    from families.nemotron_h.edge_llm.dispatch import EDGE_DISPATCH, platform_matches

    for sm in (80, 120):
        assert platform_matches({"mamba_head_dim": 80}, {"sm": sm})
        assert ("linux", "x86_64", sm, "fp16") in EDGE_DISPATCH
    assert ("linux", "x86_64", 80, "nvfp4") not in EDGE_DISPATCH

    from families.nemotron_h.model import _NemotronHModel

    # Paired provider indices are strict and retain their declared order.
    model_config = SimpleNamespace(num_hidden_layers=3)
    for indices in ((-1,), (3,), (0, 0), (True,), ("1",)):
        with pytest.raises(ValueError, match="target feature layers"):
            _NemotronHModel().build_engine(
                model_config, {}, 8, target_feature_layers=indices
            )


def _edge_cli_source(tmp_path):
    import json

    source = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    draft = tmp_path / "draft"
    draft.mkdir()
    return source, draft


@pytest.mark.parametrize("precision", [None, "fp16", "fp32"])
@pytest.mark.parametrize("variant", ["dflash", "dspark", "dspark_tree"])
def test_edge_cli_uses_ordinary_family_build(tmp_path, monkeypatch, precision, variant):
    from tensorrt_model_connect import family_cli as build_cli
    from families.nemotron_h.edge_llm import dispatch
    from families.nemotron_h.edge_llm.config import NemotronHBuildRequest

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    seen = []

    def paired(request, writer, execution):
        assert isinstance(request, NemotronHBuildRequest)
        assert request.execution is execution
        assert execution.variant == variant
        assert [(item.role, item.model_dir) for item in execution.checkpoints] == [("draft", draft)]
        seen.append(request)
        writer.set_header(family="nemotron_h", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})

    monkeypatch.setattr(dispatch, "build_paired", paired)
    options = ["--precision", precision] if precision else []
    assert build_cli.main(["nemotron_h",
        "build", str(source), *options, "-o", str(output),
        "--execution-variant", variant, "--companion", f"draft={draft}",
    ]) == 0
    assert len(seen) == 1
    assert seen[0].precision == (precision or "fp16")
    assert output.is_file()

    from families.nemotron_h.tests.test_e2e import _build_bundle
    from families.nemotron_h.edge_llm.config import BuildExecutionInputs, NamedCheckpoint

    _build_bundle(
        {"precision": "fp16", "max_sequence_length": 64, "tensor_parallel_size": 1},
        source, output,
        execution=BuildExecutionInputs(variant, (NamedCheckpoint("draft", draft),)),
    )
    assert len(seen) == 2
    assert seen[-1].precision == "fp16"
    assert seen[-1].max_sequence_length == 64
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

    from families.nemotron_h import cli as owner
    source, _ = _edge_cli_source(tmp_path)
    monkeypatch.setattr(owner, "select_backend", lambda *_: pytest.fail("backend touched"))
    monkeypatch.setattr(owner, "BundleWriter", lambda *_: pytest.fail("writer created"))
    with pytest.raises(ValueError):
        build_cli.main(["nemotron_h", "build", str(source), "-o", str(tmp_path / "out"), *options])


def test_edge_cli_help_is_family_owned(tmp_path, capsys):
    from tensorrt_model_connect import family_cli as build_cli

    source, _ = _edge_cli_source(tmp_path)
    with pytest.raises(SystemExit) as caught:
        build_cli.main(["nemotron_h", "build", str(source), "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--execution-variant {dflash,dspark,dspark_tree}" in help_text
    assert "--companion" in help_text


def test_edge_request_preserves_fields_and_family_owner(tmp_path):
    from families.nemotron_h.edge_llm import cli
    from dataclasses import fields, replace
    from families.nemotron_h.build_request import BuildRequest, coerce_request
    from families.nemotron_h.edge_llm.config import (
        BuildExecutionInputs, NamedCheckpoint, with_execution,
    )

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "nemotron_h", "text_generation", "fp16",
                           graph_transform=lambda layer: layer)
    assert coerce_request(request) is request
    # Native fallback must reject an unsupported graph callback instead of
    # silently publishing a graph that never applied the caller's transform.
    # The deliberately incomplete config proves rejection precedes weight IO.
    from families.nemotron_h.model import build

    with pytest.raises(NotImplementedError, match="graph_transform"):
        build(request, None)
    execution = BuildExecutionInputs("dflash", (NamedCheckpoint("draft", draft),))
    assert cli.execution_inputs(None) is None
    extended = with_execution(request, execution)
    for field in fields(BuildRequest):
        assert getattr(extended, field.name) is getattr(request, field.name)
    with pytest.raises(ValueError, match="requires the nemotron_h family"):
        with_execution(replace(request, family="other"), execution)
    with pytest.raises(ValueError, match="unique"):
        BuildExecutionInputs("dflash", execution.checkpoints * 2)


@pytest.mark.parametrize("failure", [RuntimeError("paired build failed"), KeyboardInterrupt()])
def test_edge_cli_failure_preserves_existing_bundle(tmp_path, monkeypatch, failure):
    from tensorrt_model_connect import family_cli as build_cli
    from families.nemotron_h.edge_llm import dispatch

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    output.write_bytes(b"previous publication")

    def fail(request, writer, execution):
        writer.set_header(family="nemotron_h", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})
        raise failure

    monkeypatch.setattr(dispatch, "build_paired", fail)
    with pytest.raises(type(failure)) as caught:
        build_cli.main(["nemotron_h",
            "build", str(source), "-o", str(output), "--execution-variant", "dflash",
            "--companion", f"draft={draft}",
        ])
    assert caught.value is failure
    assert output.read_bytes() == b"previous publication"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["draft", "pair.bundle", "target"]


def test_edge_pair_requires_draft_and_rechecks_local_inputs(tmp_path, monkeypatch, caplog):
    from tensorrt_model_connect.build import BuildRequest
    from families.nemotron_h.edge_llm.config import BuildExecutionInputs, NamedCheckpoint
    from families.nemotron_h.edge_llm.dispatch import build_paired

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "nemotron_h", "text_generation", "fp16")
    with pytest.raises(ValueError, match="paired execution requires"):
        build_paired(request, None, BuildExecutionInputs("dflash"))
    execution = BuildExecutionInputs("dflash", (NamedCheckpoint("draft", draft),))
    draft.rmdir()
    with pytest.raises(ValueError, match="existing local directory"):
        build_paired(request, None, execution)

    # Native retries retain the exact complete pair; cancellation/publication
    # failures must not retry or leave a partial public bundle.
    from families.nemotron_h.edge_llm import dispatch
    from families.nemotron_h import paired
    draft.mkdir()
    (source / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}))
    monkeypatch.setattr(dispatch, "candidate", lambda *_: True)
    monkeypatch.setattr(dispatch, "source_quantization", lambda *_: "nvfp4")
    monkeypatch.setattr(dispatch.edge_llm, "local_target", lambda: {
        "os": "linux", "arch": "x86_64", "sm": 120,
    })
    writer = object()
    native_calls = []

    def native(original, destination, companion, variant):
        assert original is request and destination is writer and companion is draft
        native_calls.append(variant)

    monkeypatch.setattr(paired, "build", native)
    for variant in ("dflash", "dspark", "dspark_tree"):
        monkeypatch.setattr(dispatch.edge_llm, "package_present", lambda: False)
        dispatch.build_speculative(request, writer, draft, variant)
        assert native_calls.pop() == variant
    monkeypatch.setattr(dispatch.edge_llm, "package_present", lambda: True)

    def failed(*_):
        raise RuntimeError("paired preparation failed")

    monkeypatch.setattr(dispatch.edge_llm, "prepare_paired", failed)
    dispatch.build_speculative(request, writer, draft, "dspark")
    assert native_calls.pop() == "dspark"
    assert "unchanged native pair" in caplog.text
    diagnostics = list(tmp_path.glob(".out.edge-onnx-*.log"))
    assert len(diagnostics) == 1 and "paired preparation failed" in diagnostics[0].read_text()
    diagnostics[0].unlink()

    def cancel(*_):
        raise KeyboardInterrupt()

    monkeypatch.setattr(dispatch.edge_llm, "prepare_paired", cancel)
    with pytest.raises(KeyboardInterrupt):
        dispatch.build_speculative(request, writer, draft, "dflash")
    assert not native_calls
    assert not list(tmp_path.glob(".out.edge-onnx-*.log"))
    monkeypatch.setattr(dispatch.edge_llm, "prepare_paired", lambda *_: ([], {}))
    monkeypatch.setattr(dispatch.edge_llm, "publish", failed)
    with pytest.raises(RuntimeError, match="paired preparation failed"):
        dispatch.build_speculative(request, writer, draft, "dflash")
    assert not native_calls


@pytest.mark.parametrize("mode", ["absent", "success", "corrupt", "failure", "cancel", "device_failure"])
def test_edge_optional_package_and_output_local_staging(tmp_path, monkeypatch, caplog, mode):
    import json
    from tensorrt_model_connect.build import BuildRequest
    from families.nemotron_h.edge_llm import builder, dispatch

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
    request = BuildRequest(source, tmp_path / "out", "nemotron_h", "text_generation", "fp16")
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


@pytest.mark.parametrize("options", [[], ["--precision", "fp16", "--max-sequence-length", "64"],
    ["--precision", "fp16", "--quantization", "fp8"],
    ["--precision", "fp16", "--quantization", "nvfp4"]])
def test_declared_build_matches_legacy_request(tmp_path, monkeypatch, options):
    """Owner command preserves ordinary request defaults and explicit controls."""
    import json
    from families.nemotron_h import cli as owner
    from tensorrt_model_connect import build_cli, family_cli

    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    output = tmp_path / "model.bundle"
    captured = []
    monkeypatch.setattr(owner, "build_bundle", lambda request, output: captured.append(request))
    monkeypatch.setattr(build_cli, "build", captured.append)
    args = [str(source), "-o", str(output), *options]
    assert family_cli.main(["nemotron_h", "build", *args]) == 0
    assert build_cli.main(["build", *args, "--family", "nemotron_h"]) == 0
    assert len(captured) == 2
    from dataclasses import fields
    assert isinstance(captured[0], owner.BuildRequest)
    for field in fields(captured[1]):
        assert getattr(captured[0], field.name) == getattr(captured[1], field.name)
    from dataclasses import replace
    from families.nemotron_h.build_request import coerce_request
    assert coerce_request(captured[1]) == captured[0]
    assert coerce_request(replace(captured[1], fp32_layers=[])) == captured[0]
    with pytest.raises(NotImplementedError, match="fp32_layers"):
        coerce_request(replace(captured[1], fp32_layers=[0]))
    with pytest.raises(NotImplementedError, match="image_height"):
        coerce_request(replace(captured[1], image_height=32))
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="unknown"):
        coerce_request(SimpleNamespace(**vars(captured[1]), unexpected_option=True))
    assert captured[0].family == "nemotron_h"
    assert captured[0].task == "text_generation"
    assert captured[0].precision == ("fp16" if options else "fp32")
    assert not output.exists()

    from families.nemotron_h.tests import test_e2e as e2e

    def capture_bundle(request):
        captured.append(request)
        request.output_path.write_bytes(b"test bundle")

    monkeypatch.setattr(e2e, "build", capture_bundle)
    e2e._build_bundle(
        {"precision": captured[1].precision,
         "max_sequence_length": captured[1].max_sequence_length,
         "tensor_parallel_size": captured[1].tensor_parallel_size,
         "quantization": captured[1].quantization},
        source, output,
    )
    assert captured[-1] == captured[1]


def test_declared_help_is_offline_and_dependency_free():
    """Actual child-process help needs neither a checkpoint nor GPU imports."""
    import subprocess
    import sys

    code = """
import sys
from tensorrt_model_connect.family_cli import main
try:
    main(["nemotron_h", "build", "--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("help did not exit")
assert "families.nemotron_h.cli" not in sys.modules
assert "tensorrt" not in sys.modules
assert "huggingface_hub" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "trtmc nemotron_h build" in result.stdout
    assert "bf16" not in result.stdout
    from families.nemotron_h import cli as owner
    with pytest.raises(ValueError, match="precision"):
        owner.build(model="/missing", output=Path("/unused"), precision="bf16")
