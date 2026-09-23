# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current single-GPU Llama standard-decoder builder variants."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from ..checkpoint_mapper import WeightDict  # noqa: E402
from ..config import ModelConfig  # noqa: E402
from ..standard_decoder_builder import build_standard_decoder_engine  # noqa: E402


pytestmark = [pytest.mark.gpu, pytest.mark.trt]


def _make_weights(
    hidden: int,
    vocab: int,
    num_layers: int,
    attention_size: int,
    mlp_size: int,
    *,
    mlp_type: str = "swiglu",
    position_type: str = "rope",
) -> WeightDict:
    rng = np.random.RandomState(42)
    weights = WeightDict()
    weights["embedding"] = rng.randn(vocab, hidden).astype(np.float32)
    for layer in range(num_layers):
        prefix = f"layer.{layer}"
        weights[f"{prefix}.input_norm"] = rng.randn(hidden).astype(np.float32)
        weights[f"{prefix}.post_attn_norm"] = rng.randn(hidden).astype(np.float32)
        weights[f"{prefix}.w_q"] = rng.randn(hidden, attention_size).astype(np.float32)
        weights[f"{prefix}.w_k"] = rng.randn(hidden, attention_size).astype(np.float32)
        weights[f"{prefix}.w_v"] = rng.randn(hidden, attention_size).astype(np.float32)
        weights[f"{prefix}.w_o"] = rng.randn(attention_size, hidden).astype(np.float32)
        if mlp_type == "swiglu":
            weights[f"{prefix}.w_gate"] = rng.randn(hidden, mlp_size).astype(np.float32)
            weights[f"{prefix}.w_up"] = rng.randn(hidden, mlp_size).astype(np.float32)
            weights[f"{prefix}.w_down"] = rng.randn(mlp_size, hidden).astype(np.float32)
        else:
            weights[f"{prefix}.w_fc1"] = rng.randn(hidden, mlp_size).astype(np.float32)
            weights[f"{prefix}.w_fc2"] = rng.randn(mlp_size, hidden).astype(np.float32)
    weights["final_norm"] = rng.randn(hidden).astype(np.float32)
    weights["w_out"] = rng.randn(hidden, vocab).astype(np.float32)
    weights["_attention_size"] = attention_size
    weights["_mlp_size"] = mlp_size
    if position_type == "learned":
        weights["position_embedding"] = rng.randn(64, hidden).astype(np.float32)
    return weights


def _build_engine(**kwargs) -> bytes:
    hidden, vocab, num_layers = 16, 32, 2
    config = ModelConfig(
        model_type="llama",
        hidden_size=hidden,
        vocab_size=vocab,
        intermediate_size=32,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=64,
    )
    mlp_type = kwargs.get("mlp_type", "swiglu")
    position_type = kwargs.get("position_type", "rope")
    weights = _make_weights(
        hidden,
        vocab,
        num_layers,
        hidden,
        32,
        mlp_type=mlp_type,
        position_type=position_type,
    )
    return build_standard_decoder_engine(config, weights, 4, **kwargs)


def _deserialize(plan: bytes):
    return trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(plan)


def _io_names(plan: bytes) -> tuple[set[str], set[str]]:
    engine = _deserialize(plan)
    assert engine is not None
    inputs: set[str] = set()
    outputs: set[str] = set()
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        target = inputs if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs
        target.add(name)
    return inputs, outputs


def test_default_rope_swiglu() -> None:
    plan = _build_engine()
    inputs, outputs = _io_names(plan)

    assert {
        "token_id",
        "position_id",
        "attention_mask",
        "cache_k_0",
        "cache_k_1",
        "cache_v_0",
        "cache_v_1",
    } <= inputs
    assert {"logits", "present_k_0", "present_k_1", "present_v_0", "present_v_1"} <= outputs
    engine = _deserialize(plan)
    assert engine is not None
    assert engine.get_tensor_profile_shape("attention_mask", 0) == [(1, 5), (4, 8), (4, 8)]
    assert engine.get_tensor_profile_shape("attention_mask", 1) == [(1, 5), (1, 5), (1, 5)]


def test_layernorm_gelu_fc() -> None:
    inputs, outputs = _io_names(
        _build_engine(norm_type="layernorm", mlp_type="gelu_fc", activation="gelu_new")
    )
    assert "token_id" in inputs
    assert {"logits", "present_k_0"} <= outputs


def test_learned_positions() -> None:
    inputs, outputs = _io_names(_build_engine(position_type="learned"))
    assert {"token_id", "position_id"} <= inputs
    assert "logits" in outputs


def test_alibi_positions() -> None:
    inputs, outputs = _io_names(_build_engine(position_type="alibi"))
    assert {"token_id", "position_id"} <= inputs
    assert "logits" in outputs


def test_embed_input() -> None:
    inputs, outputs = _io_names(_build_engine(embed_input=True))
    assert {"input_embed", "use_input_embed", "token_id"} <= inputs
    assert "logits" in outputs


def _edge_cli_source(tmp_path):
    import json

    source = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    draft = tmp_path / "draft"
    draft.mkdir()
    return source, draft


def test_edge_cli_uses_ordinary_family_build(tmp_path, monkeypatch):
    from tensorrt_model_connect import family_cli as build_cli
    from families.llama.edge_llm import dispatch
    from families.llama.edge_llm.config import LlamaBuildRequest

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    seen = []

    def paired(request, writer, execution):
        assert isinstance(request, LlamaBuildRequest)
        assert request.execution is execution
        assert execution.variant == "eagle3"
        assert [(item.role, item.model_dir) for item in execution.checkpoints] == [("draft", draft)]
        seen.append(request)
        writer.set_header(family="llama", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})

    monkeypatch.setattr(dispatch, "build_paired", paired)
    assert build_cli.main(["llama",
        "build", str(source), "--precision", "fp16", "-o", str(output),
        "--execution-variant", "eagle3", "--companion", f"draft={draft}",
    ]) == 0
    assert len(seen) == 1
    assert output.is_file()


@pytest.mark.parametrize("options", [
    ["--companion", "draft=/missing"],
    ["--execution-variant", "eagle3", "--companion", "missing_separator"],
    ["--execution-variant", "eagle3", "--companion", "=path"],
    ["--execution-variant", "eagle3", "--companion", "draft="],
    ["--execution-variant", "eagle3", "--companion", "draft=https://example.com/model"],
])
def test_bad_edge_cli_inputs_fail_before_backend(tmp_path, monkeypatch, options):
    from tensorrt_model_connect import family_cli as build_cli

    from families.llama import cli as owner
    source, _ = _edge_cli_source(tmp_path)
    monkeypatch.setattr(owner, "select_backend", lambda *_: pytest.fail("backend touched"))
    monkeypatch.setattr(owner, "BundleWriter", lambda *_: pytest.fail("writer created"))
    with pytest.raises(ValueError):
        build_cli.main(["llama", "build", str(source), "-o", str(tmp_path / "out"), *options])


def test_edge_cli_help_is_family_owned(tmp_path, capsys):
    from tensorrt_model_connect import family_cli as build_cli

    source, _ = _edge_cli_source(tmp_path)
    with pytest.raises(SystemExit) as caught:
        build_cli.main(["llama", "build", str(source), "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--execution-variant {eagle3}" in help_text
    assert "--companion" in help_text


def test_edge_request_preserves_fields_and_family_owner(tmp_path):
    from families.llama.edge_llm import cli
    from dataclasses import fields, replace
    from tensorrt_model_connect.build import BuildRequest
    from families.llama.edge_llm.config import (
        BuildExecutionInputs, NamedCheckpoint, with_execution,
    )

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "llama", "text_generation", "fp16",
                           graph_transform=lambda layer: layer)
    execution = BuildExecutionInputs("eagle3", (NamedCheckpoint("draft", draft),))
    assert cli.execution_inputs(None) is None
    extended = with_execution(request, execution)
    for field in fields(BuildRequest):
        assert getattr(extended, field.name) is getattr(request, field.name)
    with pytest.raises(ValueError, match="requires the llama family"):
        with_execution(replace(request, family="other"), execution)
    with pytest.raises(ValueError, match="unique"):
        BuildExecutionInputs("eagle3", execution.checkpoints * 2)


@pytest.mark.parametrize("failure", [RuntimeError("paired build failed"), KeyboardInterrupt()])
def test_edge_cli_failure_preserves_existing_bundle(tmp_path, monkeypatch, failure):
    from tensorrt_model_connect import family_cli as build_cli
    from families.llama.edge_llm import dispatch

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    output.write_bytes(b"previous publication")

    def fail(request, writer, execution):
        writer.set_header(family="llama", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})
        raise failure

    monkeypatch.setattr(dispatch, "build_paired", fail)
    with pytest.raises(type(failure)) as caught:
        build_cli.main(["llama",
            "build", str(source), "-o", str(output), "--execution-variant", "eagle3",
            "--companion", f"draft={draft}",
        ])
    assert caught.value is failure
    assert output.read_bytes() == b"previous publication"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["draft", "pair.bundle", "target"]


def test_edge_pair_requires_draft_and_rechecks_local_inputs(tmp_path):
    from tensorrt_model_connect.build import BuildRequest
    from families.llama.edge_llm.config import BuildExecutionInputs, NamedCheckpoint
    from families.llama.edge_llm.dispatch import build_paired

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "llama", "text_generation", "fp16")
    with pytest.raises(ValueError, match="paired execution requires"):
        build_paired(request, None, BuildExecutionInputs("eagle3"))
    execution = BuildExecutionInputs("eagle3", (NamedCheckpoint("draft", draft),))
    draft.rmdir()
    with pytest.raises(ValueError, match="existing local directory"):
        build_paired(request, None, execution)


@pytest.mark.parametrize("mode", ["absent", "success", "corrupt", "failure", "cancel", "device_failure"])
def test_edge_optional_package_and_output_local_staging(tmp_path, monkeypatch, caplog, mode):
    import json
    from tensorrt_model_connect.build import BuildRequest
    from families.llama.edge_llm import builder, dispatch

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
    request = BuildRequest(source, tmp_path / "out", "llama", "text_generation", "fp16")
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


def test_edge_windows_nonmatch_does_not_probe_compiler(tmp_path, monkeypatch, caplog):
    import json
    from tensorrt_model_connect.build import BuildRequest
    from families.llama.edge_llm import builder, dispatch

    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}))
    request = BuildRequest(tmp_path, tmp_path / "out", "llama", "text_generation", "fp16")
    monkeypatch.setattr(dispatch, "candidate", lambda *_: True)
    monkeypatch.setattr(dispatch.sys, "platform", "win32")
    monkeypatch.setenv("CUDACXX", r"C:\Program Files\CUDA\bin\nvcc.exe")
    monkeypatch.setattr(builder, "package_present", lambda: True)
    monkeypatch.setattr(builder, "local_target", lambda: pytest.fail("unqualified platform probed"))
    seen = []
    dispatch.build(request, None, lambda *args: seen.append(args))
    assert seen == [(request, None)]
    assert not caplog.text and not list(tmp_path.glob(".out.edge-*"))


@pytest.mark.parametrize("options", [[], ["--precision", "fp16", "--max-sequence-length", "64"]])
def test_declared_build_matches_legacy_request(tmp_path, monkeypatch, options):
    """Owner command preserves ordinary request defaults and explicit controls."""
    import json
    from families.llama import cli as owner
    from tensorrt_model_connect import build_cli, family_cli

    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "llama"}))
    output = tmp_path / "model.bundle"
    captured = []
    monkeypatch.setattr(owner, "build_bundle", lambda request, output: captured.append(request))
    monkeypatch.setattr(build_cli, "build", captured.append)
    args = [str(source), "-o", str(output), *options]
    assert family_cli.main(["llama", "build", *args]) == 0
    assert build_cli.main(["build", *args, "--family", "llama"]) == 0
    assert len(captured) == 2
    from dataclasses import fields
    assert isinstance(captured[0], owner.BuildRequest)
    for field in fields(captured[1]):
        assert getattr(captured[0], field.name) == getattr(captured[1], field.name)
    from dataclasses import replace
    from families.llama.build_request import coerce_request
    assert coerce_request(captured[1]) == captured[0]
    with pytest.raises(NotImplementedError, match="image_height"):
        coerce_request(replace(captured[1], image_height=32))
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="unknown"):
        coerce_request(SimpleNamespace(**vars(captured[1]), unexpected_option=True))
    assert captured[0].family == "llama"
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
    main(["llama", "build", "--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("help did not exit")
assert "families.llama.cli" not in sys.modules
assert "tensorrt" not in sys.modules
assert "huggingface_hub" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "trtmc llama build" in result.stdout
