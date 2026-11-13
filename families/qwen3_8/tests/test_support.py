# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_model_connect.model_support import ModelMetadata, resolve_family


@pytest.mark.parametrize("vision", [None, {"model_type": "qwen3_5"}])
def test_qwen38_marker_default_task_is_checkpoint_owned(vision) -> None:
    family, support = resolve_family(
        ModelMetadata(
            {
                "model_type": "qwen3_5",
                "text_config": {"output_gate_type": "sigmoid"},
                **({"vision_config": vision} if vision is not None else {}),
            },
            {},
        )
    )
    assert family == "qwen3_8"
    assert support.default_task == "text_generation"
    assert ("vision_language_generation" in support.tasks) == (vision is not None)
    from tensorrt_model_connect.family_cli import load_family_cli
    assert load_family_cli("qwen3_8")["commands"][0]["handler"] == "cli:build"


def _edge_cli_source(tmp_path):
    import json

    source = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5", "text_config": {"output_gate_type": "sigmoid"},
    }))
    draft = tmp_path / "draft"
    draft.mkdir()
    return source, draft


@pytest.mark.parametrize("variant, quantization", [
    ("dspark", None), ("dflash", None), ("mtp", None),
    ("dflash", "int4_awq"), ("mtp", "int4_awq"),
])
@pytest.mark.parametrize("precision", [None, "fp16", "fp32", "bf16"])
def test_edge_cli_uses_ordinary_family_build(tmp_path, monkeypatch, precision, variant, quantization):
    from tensorrt_model_connect import family_cli as build_cli
    from families.qwen3_8.edge_llm import dispatch
    from families.qwen3_8.edge_llm.config import Qwen38BuildRequest

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    seen = []

    def paired(request, writer, execution):
        assert isinstance(request, Qwen38BuildRequest)
        assert request.execution is execution
        assert execution.variant == variant
        assert [(item.role, item.model_dir) for item in execution.checkpoints] == (
            [] if variant == "mtp" else [("draft", draft)]
        )
        seen.append(request)
        writer.set_header(family="qwen3_8", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})

    monkeypatch.setattr(dispatch, "build_paired", paired)
    options = ["--precision", precision] if precision else []
    if quantization is not None:
        options.extend(["--quantization", quantization])
    companions = [] if variant == "mtp" else ["--companion", f"draft={draft}"]
    assert build_cli.main(["qwen3_8",
        "build", str(source), *options, "-o", str(output),
        "--execution-variant", variant, *companions,
    ]) == 0
    assert len(seen) == 1
    assert seen[0].precision == (precision or "fp16")
    assert seen[0].quantization == quantization
    assert output.is_file()

    from families.qwen3_8.tests.test_e2e import _build_bundle
    from families.qwen3_8.edge_llm.config import BuildExecutionInputs, NamedCheckpoint

    _build_bundle(
        {"precision": "fp16", "max_sequence_length": 64, "tensor_parallel_size": 1},
        source, output,
        execution=BuildExecutionInputs(variant, () if variant == "mtp" else (NamedCheckpoint("draft", draft),)),
    )
    assert len(seen) == 2
    assert seen[-1].precision == "fp16"
    assert seen[-1].max_sequence_length == 64
    assert output.is_file()


@pytest.mark.parametrize("options", [
    ["--companion", "draft=/missing"],
    ["--execution-variant", "dspark", "--companion", "missing_separator"],
    ["--execution-variant", "dspark", "--companion", "=path"],
    ["--execution-variant", "dspark", "--companion", "draft="],
    ["--execution-variant", "dspark", "--companion", "draft=https://example.com/model"],
])
def test_bad_edge_cli_inputs_fail_before_backend(tmp_path, monkeypatch, options):
    from tensorrt_model_connect import family_cli as build_cli
    from families.qwen3_8 import cli as owner
    source, _ = _edge_cli_source(tmp_path)
    monkeypatch.setattr(owner, "select_backend", lambda *_: pytest.fail("backend touched"))
    monkeypatch.setattr(owner, "BundleWriter", lambda *_: pytest.fail("writer created"))
    with pytest.raises(ValueError):
        build_cli.main(["qwen3_8", "build", str(source), "-o", str(tmp_path / "out"), *options])


def test_edge_cli_help_is_family_owned(tmp_path, capsys):
    from tensorrt_model_connect import family_cli as build_cli

    source, _ = _edge_cli_source(tmp_path)
    with pytest.raises(SystemExit) as caught:
        build_cli.main(["qwen3_8", "build", str(source), "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert "--execution-variant {dspark,dflash,mtp}" in help_text
    assert "--companion" in help_text
    assert "--quantization {none,nvfp4,fp8,int4_awq}" in help_text


@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("quantization", ["nvfp4", "int4_awq"])
def test_edge_request_preserves_fields_and_family_owner(tmp_path, monkeypatch, vision, quantization):
    from families.qwen3_8.edge_llm import cli
    from dataclasses import fields, replace
    from families.qwen3_8.build_request import BuildRequest, coerce_request
    from families.qwen3_8.edge_llm.config import (
        BuildExecutionInputs, NamedCheckpoint, with_execution,
    )

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "qwen3_8", "text_generation", "fp16",
                           graph_transform=lambda layer: layer)
    assert coerce_request(request) is request
    from families.qwen3_8.model import _build_native
    with pytest.raises(NotImplementedError, match="graph_transform"):
        _build_native(request, None)
    execution = BuildExecutionInputs("dspark", (NamedCheckpoint("draft", draft),))
    assert cli.execution_inputs(None) is None
    extended = with_execution(request, execution)
    for field in fields(BuildRequest):
        assert getattr(extended, field.name) is getattr(request, field.name)
    with pytest.raises(ValueError, match="requires the qwen3_8 family"):
        with_execution(replace(request, family="other"), execution)
    with pytest.raises(ValueError, match="unique"):
        BuildExecutionInputs("dspark", execution.checkpoints * 2)

    import json
    from families.qwen3_8.edge_llm import builder, dispatch

    algorithm, group = ("NVFP4", 16) if quantization == "nvfp4" else ("W4A16_AWQ", 128)
    raw = {"model_type": "qwen3_5", "text_config": {
        "num_hidden_layers": 64, "hidden_size": 5120, "intermediate_size": 17408,
        "num_attention_heads": 24, "num_key_value_heads": 4, "vocab_size": 248320,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "output_gate_type": "swish", "max_position_embeddings": 4096,
        "mtp_num_hidden_layers": 1,
    }, "quantization_config": {"quant_method": "modelopt", "quant_algo": algorithm}}
    if vision:
        raw["vision_config"] = {}
    (source / "config.json").write_text(json.dumps(raw))
    metadata = {"quantization": {"quant_algo": algorithm, "group_size": group,
                                  "kv_cache_quant_algo": None}}
    (source / "hf_quant_config.json").write_text(json.dumps(metadata))
    weights = source / "original.safetensors"
    weights.write_bytes(b"original-runtime-weights")
    mtp = replace(request, graph_transform=None, max_sequence_length=512, quantization=quantization,
                  task="vision_language_generation" if vision else "text_generation")
    assert builder.prepared_checkpoint_quantization(source, raw) == quantization
    assert builder.checkpoint_quantization(source, raw) is None
    assert dispatch.ordinary_candidate(mtp, raw, embedded_mtp=True)
    assert not dispatch.ordinary_candidate(mtp, raw)
    text_request = replace(mtp, task="text_generation")
    assert builder.paired_checkpoint_quantization(source, raw, "dspark") is None
    assert dispatch.candidate(text_request, raw, "dflash") is (quantization == "int4_awq")
    assert not dispatch.candidate(text_request, raw, "dspark")
    if quantization == "int4_awq":
        assert not dispatch.candidate(replace(text_request, quantization="nvfp4"), raw, "dflash")
    with pytest.raises(ValueError, match="embedded draft layers"):
        dispatch.build_paired(mtp, None, BuildExecutionInputs("mtp", execution.checkpoints))
    for change in ({"tensor_parallel_size": 2}, {"precision": "bf16"}, {"quantization": "fp8"}):
        with pytest.raises(ValueError, match="uniform NVFP4"):
            dispatch.build_paired(replace(mtp, **change), None, BuildExecutionInputs("mtp"))
    with pytest.raises(ValueError, match="four verification"):
        dispatch.build_paired(replace(mtp, max_sequence_length=4), None, BuildExecutionInputs("mtp"))
    target = {"os": "linux", "arch": "x86_64", "sm": 120}
    monkeypatch.setattr(builder, "installed_package", lambda target: {"plugin": "/plugin.so"})
    monkeypatch.setattr(builder, "builder_python", lambda package, target: "python")
    commands = []

    def build_mtp(command, **kwargs):
        commands.append(command)
        from pathlib import Path
        engine = Path(command[command.index("--engine-dir") + 1])
        engine.mkdir(parents=True, exist_ok=True)
        identity = {"version": 1, "sources": {"component": {
            "build_source": str(source), "files": {weights.name: {"bytes": weights.stat().st_size}},
        }}}
        if command[command.index("--components") + 1] == "visual":
            (engine / "visual").mkdir()
            for name in ("visual.engine", "preprocessor_config.json"):
                (engine / "visual" / name).write_bytes(b"output")
            (engine / "visual/config.json").write_text(json.dumps({"checkpoint_identity": identity}))
        else:
            for name in ("spec_base.engine", "spec_draft.engine", "tokenizer.json",
                         "tokenizer_config.json", "chat_template.jinja"):
                (engine / name).write_bytes(b"output")
            for name in ("base_config.json", "draft_config.json"):
                (engine / name).write_text(json.dumps({"spec_decode_type": "mtp",
                                                      "checkpoint_identity": identity}))

    monkeypatch.setattr(builder.subprocess, "run", build_mtp)
    stage = tmp_path / "mtp-stage"
    stage.mkdir()
    files, marker = builder.prepare_mtp(mtp, raw, target, stage, tmp_path / "mtp.log")
    assert len(commands) == (2 if vision else 1)
    assert commands[0][commands[0].index("--spec-type") + 1] == "mtp"
    assert commands[0][commands[0].index("--max-verify-tree-size") + 1] == "4"
    assert commands[0][commands[0].index("--max-draft-tree-size") + 1] == "4"
    assert commands[0][commands[0].index("--dense") + 1] == "auto"
    assert "--draft-model-dir" not in commands[0]
    assert marker["execution_variant"] == "mtp" and marker["mtp_draft_steps"] == 3
    assert marker["mtp_verify_size"] == 4 and marker["tensor_parallel_size"] == 1
    assert marker["checkpoint_quantization"] == quantization
    assert files["edge_llm/checkpoint/original.safetensors"] == weights
    if vision:
        assert "--spec-type" not in commands[1]
        assert "edge_llm/engine/visual/visual.engine" in files
    identity = {"version": 1, "sources": {
        role: {"build_source": str(source),
               "files": {weights.name: {"bytes": weights.stat().st_size}}}
        for role in ("component", "target")
    }}
    runtime_config = {"checkpoint_identity": identity}
    dependencies = builder.runtime_checkpoint_files(source, runtime_config, embedded_mtp=True)
    assert dependencies == {"edge_llm/checkpoint/original.safetensors": weights}
    with pytest.raises(ValueError, match="Unexpected Edge checkpoint identity"):
        builder.runtime_checkpoint_files(source, runtime_config)
    identity["sources"]["target"]["build_source"] = str(tmp_path / "other-checkpoint")
    with pytest.raises(ValueError, match="original base checkpoint"):
        builder.runtime_checkpoint_files(source, runtime_config, embedded_mtp=True)
    identity["sources"]["target"]["build_source"] = str(source)
    identity["sources"]["target"]["files"][weights.name]["bytes"] += 1
    with pytest.raises(ValueError, match="Conflicting embedded MTP"):
        builder.runtime_checkpoint_files(source, runtime_config, embedded_mtp=True)
    monkeypatch.setattr(builder, "local_target", lambda: {**target, "sm": 80})
    from families.qwen3_8 import model
    native_requests = []
    monkeypatch.setattr(model, "_build_native",
                        lambda request, writer, **kwargs: native_requests.append((request, writer, kwargs)))
    dispatch.build_paired(mtp, None, BuildExecutionInputs("mtp"))
    native_request, native_writer, native_options = native_requests.pop()
    assert native_request.quantization == quantization
    assert native_request.mtp_seq_len == 4
    assert native_request.task == mtp.task and native_request.model_dir == mtp.model_dir
    assert native_writer is None and native_options == {"require_mtp": True}
    with pytest.raises(ValueError, match="four verification"):
        dispatch.build_paired(replace(mtp, mtp_seq_len=2), None, BuildExecutionInputs("mtp"))
    assert len(commands) == (2 if vision else 1)
    metadata["quantization"]["group_size"] = 32
    (source / "hf_quant_config.json").write_text(json.dumps(metadata))
    assert builder.prepared_checkpoint_quantization(source, raw) is None


@pytest.mark.parametrize("failure", [RuntimeError("paired build failed"), KeyboardInterrupt()])
def test_edge_cli_failure_preserves_existing_bundle(tmp_path, monkeypatch, failure):
    from tensorrt_model_connect import family_cli as build_cli
    from families.qwen3_8.edge_llm import dispatch

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    output.write_bytes(b"previous publication")

    def fail(request, writer, execution):
        writer.set_header(family="qwen3_8", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})
        raise failure

    monkeypatch.setattr(dispatch, "build_paired", fail)
    with pytest.raises(type(failure)) as caught:
        build_cli.main(["qwen3_8",
            "build", str(source), "-o", str(output), "--execution-variant", "dspark",
            "--companion", f"draft={draft}",
        ])
    assert caught.value is failure
    assert output.read_bytes() == b"previous publication"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["draft", "pair.bundle", "target"]


def test_edge_pair_requires_draft_and_rechecks_local_inputs(tmp_path):
    from tensorrt_model_connect.build import BuildRequest
    from families.qwen3_8.edge_llm.config import BuildExecutionInputs, NamedCheckpoint
    from families.qwen3_8.edge_llm.dispatch import build_paired

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "qwen3_8", "text_generation", "fp16")
    with pytest.raises(ValueError, match="paired execution requires"):
        build_paired(request, None, BuildExecutionInputs("dspark"))
    execution = BuildExecutionInputs("dspark", (NamedCheckpoint("draft", draft),))
    draft.rmdir()
    with pytest.raises(ValueError, match="existing local directory"):
        build_paired(request, None, execution)


@pytest.mark.parametrize("model_type", ["qwen38", "qwen3.8", "qwen3_8"])
def test_qwen38_aliases_register_the_same_family_cli(model_type):
    from families.qwen3_8.support import describe

    support = describe(ModelMetadata({"model_type": model_type}, {}))
    assert support is not None
    from tensorrt_model_connect.family_cli import load_family_cli
    assert load_family_cli("qwen3_8")["commands"][0]["handler"] == "cli:build"


def _dspark_pair_request(tmp_path, variant="dspark"):
    import json
    from tensorrt_model_connect.build import BuildRequest
    from families.qwen3_8.edge_llm.config import BuildExecutionInputs, NamedCheckpoint

    source, draft = _edge_cli_source(tmp_path)
    layers = {"layer0": {"quant_algo": "FP8"}, "layer1": {"quant_algo": "NVFP4"}}
    base = {
        "model_type": "qwen3_5", "output_gate_type": "sigmoid",
        "hidden_size": 16, "vocab_size": 32, "num_hidden_layers": 8,
        "max_position_embeddings": 4096, "linear_key_head_dim": 128,
        "linear_value_head_dim": 128, "quantization_config": {
            "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
            "quantized_layers": layers, "kv_cache_scheme": {"type": "float"},
        },
    }
    (source / "config.json").write_text(json.dumps(base))
    (source / "hf_quant_config.json").write_text(json.dumps({"quantization": {
        "quant_algo": "MIXED_PRECISION", "quantized_layers": layers,
        "kv_cache_quant_algo": "FP8",
    }}))
    (draft / "config.json").write_text(json.dumps({
        "architectures": ["DSparkDraftModel" if variant == "dspark" else "DFlash2DraftModel"],
        "hidden_size": 16, "vocab_size": 32,
        "num_target_layers": 8, "max_position_embeddings": 4096,
        f"{variant}_config": {"block_size": 7 if variant == "dspark" else 8,
                              "target_layer_ids": [1], "mask_token_id": 3},
    }))
    return (
        BuildRequest(source, tmp_path / "out", "qwen3_8", "text_generation", "fp16"),
        BuildExecutionInputs(variant, (NamedCheckpoint("draft", draft),)),
    )


@pytest.mark.parametrize("overrides, message", [
    ({"precision": "bf16"}, "--precision fp16"),
    ({"backend": "trt_rtx"}, "requires backend=trt"),
    ({"max_batch_size": 2}, "batch/TP/CP=1"),
    ({"tensor_parallel_size": 2}, "batch/TP/CP=1"),
    ({"dynamic_kv_cache": True}, "no dynamic KV"),
    ({"quantization": "fp8"}, "quantization unset or nvfp4"),
    ({"max_sequence_length": 8}, "above 8 and at most 1024"),
    ({"max_sequence_length": 1025}, "above 8 and at most 1024"),
    ({"max_sequence_length": 2048}, "above 8 and at most 1024"),
])
def test_dspark_request_errors_precede_adapter_work(tmp_path, monkeypatch, overrides, message):
    from dataclasses import replace
    from families.qwen3_8.edge_llm import dispatch

    request, execution = _dspark_pair_request(tmp_path)
    monkeypatch.setattr(dispatch, "build", lambda *_args, **_kw: pytest.fail("adapter work started"))
    with pytest.raises(ValueError, match=message) as caught:
        dispatch.build_paired(replace(request, **overrides), None, execution)
    assert caught.value.__cause__ is None
    assert not list(tmp_path.glob(".out.edge-*"))


@pytest.mark.parametrize("variant, limit, quantization", [
    ("dspark", 9, "nvfp4"), ("dspark", 1024, "nvfp4"), ("dflash", 9, "nvfp4"),
    ("dflash", 4096, "nvfp4"), ("dflash", 9, "int4_awq"), ("dflash", 4096, "int4_awq")])
def test_dspark_capacity_boundaries_keep_the_requested_pair(tmp_path, monkeypatch, variant, limit, quantization):
    from dataclasses import replace
    from families.qwen3_8.edge_llm import dispatch

    request, execution = _dspark_pair_request(tmp_path, variant)
    request = replace(request, max_sequence_length=limit, quantization=quantization)
    if quantization == "int4_awq":
        import json
        path = request.model_dir / "config.json"
        raw = json.loads(path.read_text())
        raw["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "W4A16_AWQ"}
        path.write_text(json.dumps(raw))
        (request.model_dir / "hf_quant_config.json").write_text(json.dumps({
            "quantization": {"quant_algo": "W4A16_AWQ", "group_size": 128,
                             "kv_cache_quant_algo": None},
        }))
    seen = []
    native_seen = []
    from families.qwen3_8 import model
    monkeypatch.setattr(model, "_build_native",
                        lambda original, writer, **kwargs: native_seen.append((original, kwargs)))

    def build(original, writer, native, *, draft_dir, execution_variant="dspark"):
        assert execution_variant == variant
        assert original is request and draft_dir == execution.checkpoints[0].model_dir
        native(original, writer)
        assert native_seen[-1][0].max_sequence_length == limit
        assert native_seen[-1][0].quantization == quantization
        expected = {"draft_dir": execution.checkpoints[0].model_dir}
        if variant == "dflash":
            expected["draft_variant"] = variant
        assert native_seen[-1][1] == expected
        seen.append(original)

    monkeypatch.setattr(dispatch, "build", build)
    dispatch.build_paired(request, None, execution)
    assert seen == [request]


def test_dspark_checkpoint_error_is_distinct_from_request_error(tmp_path, monkeypatch):
    from families.qwen3_8.edge_llm import dispatch

    request, execution = _dspark_pair_request(tmp_path)
    (request.model_dir / "hf_quant_config.json").unlink()
    monkeypatch.setattr(dispatch, "build", lambda *_args, **_kw: pytest.fail("adapter work started"))
    with pytest.raises(ValueError, match="matching mixed-NVFP4 base"):
        dispatch.build_paired(request, None, execution)


@pytest.mark.parametrize("arch, sm", [("x86_64", 90), ("aarch64", 120)])
def test_dspark_unqualified_platform_uses_native_pair(tmp_path, monkeypatch, arch, sm):
    from families.qwen3_8.edge_llm import builder, dispatch
    from families.qwen3_8 import model

    request, execution = _dspark_pair_request(tmp_path)
    captured = []
    monkeypatch.setattr(builder, "local_target", lambda: {"os": "linux", "arch": arch, "sm": sm})
    monkeypatch.setattr(model, "_build_native",
                        lambda request, writer, **kwargs: captured.append((request, writer, kwargs)))
    dispatch.build_paired(request, None, execution)
    assert len(captured) == 1
    native_request, writer, kwargs = captured[0]
    assert native_request.quantization == "nvfp4"
    assert native_request.precision == "fp16"
    assert native_request.model_dir == request.model_dir
    assert native_request.max_sequence_length == request.max_sequence_length
    assert kwargs == {"draft_dir": execution.checkpoints[0].model_dir}
    assert writer is None
    assert not list(tmp_path.glob(".out.edge-*"))


@pytest.mark.parametrize("options", [
    [], ["--precision", "fp16", "--max-sequence-length", "64"],
    ["--precision", "fp16", "--quantization", "nvfp4", "--max-sequence-length", "256"],
    ["--precision", "fp16", "--max-sequence-length", "4096", "--tensor-parallel-size", "2"],
    ["--task", "vision_language_generation", "--precision", "fp16", "--max-sequence-length", "4096", "--tensor-parallel-size", "2"],
])
def test_declared_build_matches_legacy_request(tmp_path, monkeypatch, options):
    """Owner command preserves ordinary request defaults and explicit controls."""
    import json
    from pathlib import Path
    from families.qwen3_8 import cli as owner
    from tensorrt_model_connect import build_cli, family_cli

    source = tmp_path / "checkpoint"
    source.mkdir()
    selected_task = options[options.index("--task") + 1] if "--task" in options else "text_generation"
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3_8",
        **({"vision_config": {"model_type": "qwen3_5"}} if selected_task == "vision_language_generation" else {}),
    }))
    output = tmp_path / "model.bundle"
    captured = []
    monkeypatch.setattr(owner, "build_bundle", lambda request, output: captured.append(request))
    monkeypatch.setattr(build_cli, "build", captured.append)
    args = [str(source), "-o", str(output), *options]
    assert family_cli.main(["qwen3_8", "build", *args]) == 0
    assert build_cli.main(["build", *args, "--family", "qwen3_8"]) == 0
    assert len(captured) == 2
    from dataclasses import fields
    assert isinstance(captured[0], owner.BuildRequest)
    for field in fields(captured[1]):
        assert getattr(captured[0], field.name) == getattr(captured[1], field.name)
    from dataclasses import replace
    from families.qwen3_8.build_request import coerce_request
    assert coerce_request(captured[1]) == captured[0]
    with pytest.raises(NotImplementedError, match="image_height"):
        coerce_request(replace(captured[1], image_height=32))
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="unknown"):
        coerce_request(SimpleNamespace(**vars(captured[1]), unexpected_option=True))
    assert captured[0].family == "qwen3_8"
    assert captured[0].task == selected_task
    assert captured[0].precision == ("fp16" if options else "bf16")
    assert not output.exists()
    if captured[0].quantization == "nvfp4":
        from families.qwen3_8.edge_llm import builder, dispatch
        raw = {"model_type": "qwen3_5", "text_config": {
            "num_hidden_layers": 64, "hidden_size": 5120, "intermediate_size": 17408,
            "num_attention_heads": 24, "num_key_value_heads": 4, "vocab_size": 248320,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "output_gate_type": "swish", "max_position_embeddings": 4096,
        }}
        layers = {"layer0": {"quant_algo": "FP8"}, "layer1": {"quant_algo": "NVFP4"}}
        raw["quantization_config"] = {
            "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
            "quantized_layers": layers, "kv_cache_scheme": {"type": "float"},
        }
        metadata = {"quantization": {"quant_algo": "MIXED_PRECISION",
                    "quantized_layers": layers, "kv_cache_quant_algo": "FP8"}}
        (source / "hf_quant_config.json").write_text(json.dumps(metadata))
        (source / "config.json").write_text(json.dumps(raw))
        weights = source / "original.safetensors"
        weights.write_bytes(b"original-runtime-weights")
        original = captured[0]
        assert dispatch.ordinary_candidate(original, raw)
        assert dispatch.ordinary_candidate(replace(original, quantization=None), raw)
        for overrides in ({"quantization": "none"}, {"quantization": "fp8"},
                          {"tensor_parallel_size": 2}, {"precision": "bf16"}):
            assert not dispatch.ordinary_candidate(replace(original, **overrides), raw)
        target = {"os": "linux", "arch": "x86_64", "sm": 120}
        assert dispatch.ORDINARY_EDGE_DISPATCH[("linux", "x86_64", 120, "fp16", 1, "nvfp4")] is builder.prepare_nvfp4
        monkeypatch.setattr(builder, "installed_package", lambda target: {"plugin": "/plugin.so"})
        monkeypatch.setattr(builder, "builder_python", lambda package, target: "python")
        commands = []

        def original_builder(command, **kwargs):
            commands.append(command)
            engine = Path(command[command.index("--engine-dir") + 1])
            engine.mkdir(parents=True, exist_ok=True)
            for name in ("llm.engine", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
                (engine / name).write_bytes(b"builder-output")
            (engine / "config.json").write_text(json.dumps({"checkpoint_identity": {
                "version": 1, "sources": {"component": {"files": {
                    weights.name: {"bytes": weights.stat().st_size},
                }}},
            }}))

        monkeypatch.setattr(builder.subprocess, "run", original_builder)
        stage = tmp_path / "ordinary-stage"
        stage.mkdir()
        files, marker = builder.prepare_nvfp4(original, raw, target, stage, tmp_path / "build.log")
        assert len(commands) == 1
        command = commands[0]
        assert command[command.index("--dense") + 1] == "auto"
        assert command[command.index("--tp-size") + 1] == "1"
        assert command[command.index("--tp-rank") + 1] == "0"
        assert "--spec-decode" not in command
        assert marker["execution_variant"] == "autoregressive"
        assert marker["tensor_parallel_size"] == 1
        assert marker["checkpoint_quantization"] == "nvfp4"
        assert files["edge_llm/checkpoint/original.safetensors"] == weights
        assert json.loads((source / "hf_quant_config.json").read_text()) == metadata
        with pytest.raises(ValueError, match="TP1 without a draft"):
            builder.prepare_nvfp4(replace(original, tensor_parallel_size=2), raw, target, stage, tmp_path / "bad.log")
        # A platform mismatch must select native without invoking the wrong Edge profile.
        monkeypatch.setattr(builder, "local_target", lambda: {**target, "sm": 80})
        native_requests = []
        dispatch.build(original, None, lambda request, writer: native_requests.append(request))
        assert native_requests == [original] and len(commands) == 1
        metadata["quantization"]["quantized_layers"] = {}
        (source / "hf_quant_config.json").write_text(json.dumps(metadata))
        assert not dispatch.ordinary_candidate(original, raw)
    if captured[0].tensor_parallel_size == 2:
        from families.qwen3_8.edge_llm.dispatch import ordinary_candidate
        raw = {"model_type": "qwen3_5", "text_config": {
            "num_hidden_layers": 64, "hidden_size": 5120, "intermediate_size": 17408,
            "num_attention_heads": 24, "num_key_value_heads": 4, "vocab_size": 248320,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "output_gate_type": "swish",
        }}
        text_request = replace(captured[0], task="text_generation")
        assert ordinary_candidate(text_request, raw)
        assert not ordinary_candidate(replace(captured[0], tensor_parallel_size=1), raw)
        assert not ordinary_candidate(replace(captured[0], precision="bf16"), raw)
        image_request = replace(captured[0], task="vision_language_generation")
        assert not ordinary_candidate(image_request, raw)
        raw["vision_config"] = {"model_type": "qwen3_5"}
        assert ordinary_candidate(image_request, raw)
        assert not ordinary_candidate(replace(image_request, tensor_parallel_size=1), raw)
        assert not ordinary_candidate(replace(image_request, task="video_language_generation"), raw)
        raw["text_config"]["output_gate_type"] = "sigmoid"
        assert not ordinary_candidate(captured[0], raw)
        from families.qwen3_8.edge_llm.builder import runtime_checkpoint_files
        used, baked = source / "runtime.safetensors", source / "baked.safetensors"
        used.write_bytes(b"original-runtime-tensors")
        baked.write_bytes(b"already-in-engine")
        config = {"checkpoint_identity": {"version": 1, "sources": {"component": {
            "files": {used.name: {"bytes": used.stat().st_size}},
        }}}}
        assert runtime_checkpoint_files(source, config) == {
            "edge_llm/checkpoint/runtime.safetensors": used,
        }
        used.write_bytes(b"changed")
        with pytest.raises(ValueError, match="Missing or changed"):
            runtime_checkpoint_files(source, config)
        config["checkpoint_identity"]["sources"]["component"]["files"] = {
            "../outside": {"bytes": 1},
        }
        with pytest.raises(ValueError, match="Unsafe"):
            runtime_checkpoint_files(source, config)

    # Publishing an ordinary image bundle must select the explicit Task SDK
    # path; implementing IModel alone does not migrate a legacy header.
    from families.qwen3_8.edge_llm.builder import publish
    from tensorrt_model_connect.bundle_writer import BundleWriter
    import struct
    for declared_task, expected_task in (("text_generation", "text_generation"),
                                         ("vision_language_generation", "images_text_to_text")):
        published = tmp_path / (declared_task + ".bundle")
        writer = BundleWriter(published)
        publish(replace(captured[0], task=declared_task), writer, {},
                {"execution_variant": "autoregressive", "task": declared_task})
        writer.finish()
        with published.open("rb") as stream:
            stream.seek(8)
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        assert header["task"] == expected_task

    from families.qwen3_8.tests.test_e2e import _build_bundle

    def capture_bundle(request, destination):
        captured.append(request)
        destination.write_bytes(b"test bundle")

    monkeypatch.setattr(owner, "build_bundle", capture_bundle)
    _build_bundle(
        {"precision": captured[0].precision, "task": captured[0].task,
         "max_sequence_length": captured[0].max_sequence_length,
         "tensor_parallel_size": captured[0].tensor_parallel_size,
         "quantization": captured[0].quantization},
        source, output,
    )
    assert captured[-1] == captured[0]
    if selected_task == "vision_language_generation":
        from families.qwen3_8.tests.test_e2e import _native_arguments
        case = {"max_new_tokens": 128, "images": ["red.jpeg", "giant.jpeg"]}
        arguments = _native_arguments(output, tmp_path, "Compare these images.", case)
        assert arguments[-2] == "--images"
        assert json.loads(arguments[-1]) == ["red.jpeg", "giant.jpeg"]
        with pytest.raises(ValueError, match="not both"):
            _native_arguments(output, tmp_path, "Compare.", {**case, "image": "other.jpeg"})
        with pytest.raises(ValueError, match="empty"):
            _native_arguments(output, tmp_path, "Compare.", {**case, "images": []})
        with pytest.raises(ValueError, match="video arguments are not mapped"):
            _native_arguments(output, tmp_path, "Compare.", {
                "max_new_tokens": 128, "video_frames": ["first.jpeg", "second.jpeg"],
            })


def test_declared_help_is_offline_and_dependency_free():
    """Actual child-process help needs neither a checkpoint nor GPU imports."""
    import subprocess
    import sys

    code = """
import sys
from tensorrt_model_connect.family_cli import main
try:
    main(["qwen3_8", "build", "--help"])
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("help did not exit")
assert "families.qwen3_8.cli" not in sys.modules
assert "tensorrt" not in sys.modules
assert "huggingface_hub" not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "trtmc qwen3_8 build" in result.stdout


@pytest.mark.parametrize("variant", ["dspark", "dflash"])
@pytest.mark.parametrize("field, value, error", [
    ("architectures", ["DFlashDraftModel"], "Expected a"),
    ("hidden_size", 32, "disagree on hidden_size"),
    ("num_target_layers", 9, "layer count"),
    ("block_size", 6, "documented block"),
])
def test_paired_draft_contract_rejected_before_export(tmp_path, monkeypatch, variant, field, value, error):
    import json
    from families.qwen3_8.edge_llm import dispatch

    request, execution = _dspark_pair_request(tmp_path, variant)
    path = execution.checkpoints[0].model_dir / "config.json"
    draft = json.loads(path.read_text())
    if field == "block_size":
        draft[f"{variant}_config"][field] = value
    else:
        draft[field] = value
    path.write_text(json.dumps(draft))
    monkeypatch.setattr(dispatch, "build", lambda *_args, **_kw: pytest.fail("export started"))
    with pytest.raises(ValueError, match=error):
        dispatch.build_paired(request, None, execution)


@pytest.mark.parametrize("onnx", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_builder_interpreter_checks_requested_api(monkeypatch, onnx, available):
    import json
    import subprocess
    from types import SimpleNamespace
    from families.qwen3_8.edge_llm import builder

    def probe(command, **options):
        assert command[1:3] == ["-I", "-c"]
        assert ("tensorrt_edgellm.scripts.export" in command[3]) is onnx
        assert ("experimental.builder.cli" in command[3]) is not onnx
        assert options["timeout"] == 30
        if not available:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=json.dumps(["0.11.0", "test-trt"]))

    monkeypatch.setattr(builder.subprocess, "run", probe)
    result = builder.builder_python({"python": "/sdk/python"}, {"tensorrt_version": "test-trt"}, onnx=onnx)
    assert result == (builder.sys.executable if available else "/sdk/python")


@pytest.mark.parametrize("edge", [False, True])
def test_e2e_tp2_uses_one_family_owned_process(monkeypatch, tmp_path, edge):
    import json
    import subprocess
    from pathlib import Path
    from families.qwen3_8.tests import test_e2e as owner

    calls = []
    sections = ({"edge_llm.json": {}, "edge_llm/engine/llm_world2_rank0.engine": {},
                 "edge_llm/engine/llm_world2_rank1.engine": {},
                 "edge_llm/engine/visual/visual.engine": {}} if edge else
                {"engine.rank0.plan": {}, "engine.rank1.plan": {}, "native_visual.json": {},
                 "native_visual.rank0.plan": {}, "native_visual.rank1.plan": {}})
    payload = {"family": "qwen3_8", "task": "images_text_to_text", "sections": sections}
    def run(command, **kwargs):
        calls.append(command)
        output = payload if command[1] == "inspect" else {"text": "Paris", "token_ids": [1]}
        return subprocess.CompletedProcess(command, 0, json.dumps(output), "")
    monkeypatch.setattr(owner.subprocess, "run", run)
    assert owner._assert_rank_sections(Path("trtmc"), tmp_path / "model.bundle", 2,
                                       "vision_language_generation") is edge
    result = owner._run_native(Path("trtmc"), tmp_path, tmp_path / "model.bundle", "Capital?",
                               {"max_new_tokens": 4}, 2, tmp_path, edge=edge)
    assert result == {"text": "Paris", "token_ids": [1]}
    assert calls[-1] == ["trtmc", "run", str(tmp_path / "model.bundle"), "--runtime-root",
                        str(tmp_path), "--prompt", "Capital?", "--max-new-tokens", "4"]
    if not edge:
        del sections["native_visual.rank1.plan"]
        with pytest.raises(AssertionError):
            owner._assert_rank_sections(Path("trtmc"), tmp_path / "model.bundle", 2,
                                         "vision_language_generation")


@pytest.mark.parametrize("image_field", ["image", "images"])
def test_reference_inputs_preserves_both_image_and_video(monkeypatch, tmp_path, image_field):
    from types import SimpleNamespace
    from PIL import Image
    import torch
    from transformers import AutoProcessor
    from families.qwen3_8.tests import test_e2e as owner

    path = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), "red").save(path)
    class Processor:
        image_processor = SimpleNamespace(patch_size=2, merge_size=2)
        video_processor = SimpleNamespace(patch_size=2, merge_size=2, temporal_patch_size=2)
        def apply_chat_template(self, messages, **kwargs):
            assert [x["type"] for x in messages[0]["content"]] == ["image", "video", "text"]
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return "both modalities"
        def __call__(self, **kwargs):
            assert kwargs["text"] == ["both modalities"] and len(kwargs["images"]) == 1
            assert kwargs["videos"][0].shape == (2, 4, 4, 3)
            assert kwargs["video_metadata"][0].frames_indices == [0, 1]
            assert kwargs["video_metadata"][0].fps == 1
            assert kwargs["do_sample_frames"] is False and kwargs["return_tensors"] == "pt"
            assert self.image_processor.size == {"shortest_edge": 2048, "longest_edge": 20480}
            assert self.video_processor.size == {"shortest_edge": 4096, "longest_edge": 40960}
            return {"input_ids": torch.tensor([[1, 2]]),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                    "video_grid_thw": torch.tensor([[1, 2, 2]])}
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *_args, **_kwargs: Processor())
    case = {image_field: path if image_field == "image" else [path],
            "video_frames": [path, path], "video_fps": 1.0}
    result = owner._reference_inputs(tmp_path, {}, None, "Compare.", case)
    assert result["input_ids"].tolist() == [[1, 2]]
