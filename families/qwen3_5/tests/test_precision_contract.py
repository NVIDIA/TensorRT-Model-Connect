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


@pytest.mark.parametrize("precision", [None, "fp16", "fp32", "bf16"])
@pytest.mark.parametrize("variant", ["dflash", "dflash_ddtree", "mtp"])
def test_edge_cli_uses_ordinary_family_build(tmp_path, monkeypatch, precision, variant):
    from tensorrt_model_connect import family_cli as build_cli
    from families.qwen3_5.edge_llm import dispatch
    from families.qwen3_5.edge_llm.config import Qwen35BuildRequest

    source, draft = _edge_cli_source(tmp_path)
    output = tmp_path / "pair.bundle"
    seen = []

    def paired(request, writer, execution):
        assert isinstance(request, Qwen35BuildRequest)
        assert request.execution is execution
        assert request.precision == ("fp16" if precision is None else precision)
        assert execution.variant == variant
        assert [(item.role, item.model_dir) for item in execution.checkpoints] == (
            [("draft", draft)] if variant != "mtp" else []
        )
        seen.append(request)
        writer.set_header(family="qwen3_5", task=request.task, backend=request.backend)
        writer.add_json("edge-test.json", {"variant": execution.variant})

    monkeypatch.setattr(dispatch, "build_paired", paired)
    precision_args = [] if precision is None else ["--precision", precision]
    arguments = ["qwen3_5",
        "build", str(source), *precision_args, "-o", str(output),
        "--execution-variant", variant,
    ]
    if variant != "mtp":
        arguments += ["--companion", f"draft={draft}"]
    if precision == "bf16":
        with pytest.raises(SystemExit) as error:
            build_cli.main(arguments)
        assert error.value.code == 2
        assert not seen and not output.exists()
        return
    assert build_cli.main(arguments) == 0
    assert len(seen) == 1
    assert output.is_file()


@pytest.mark.parametrize("options", [
    ["--companion", "draft=/missing"],
    ["--execution-variant", "dflash", "--companion", "missing_separator"],
    ["--execution-variant", "dflash", "--companion", "=path"],
    ["--execution-variant", "dflash", "--companion", "draft="],
    ["--execution-variant", "dflash", "--companion", "draft=https://example.com/model"],
    ["--execution-variant", "mtp", "--companion", "draft=/missing"],
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
    assert "--execution-variant {dflash,dflash_ddtree,mtp}" in help_text
    assert "--companion" in help_text
    assert "vision_language_generation" in help_text


def test_edge_request_preserves_fields_and_family_owner(tmp_path):
    from families.qwen3_5.edge_llm import cli
    from dataclasses import fields, replace
    from families.qwen3_5.build_request import BuildRequest, coerce_request
    from families.qwen3_5.edge_llm.config import (
        BuildExecutionInputs, NamedCheckpoint, with_execution,
    )

    source, draft = _edge_cli_source(tmp_path)
    request = BuildRequest(source, tmp_path / "out", "qwen3_5", "text_generation", "fp16",
                           graph_transform=lambda layer: layer)
    assert coerce_request(request) is request
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


@pytest.mark.parametrize("mtp_shape,quantized,visual", [
    ((1024, 24, 3584, 8, 2), False, False), ((2048, 24, 6144, 8, 2), False, False),
    ((2560, 32, 9216, 16, 4), False, False), ((4096, 32, 12288, 16, 4), False, False),
    ((2560, 32, 9216, 16, 4), True, False),
    ((2560, 32, 9216, 16, 4), False, True), ((2560, 32, 9216, 16, 4), True, True),
    ((4096, 32, 12288, 16, 4), False, True),
])
def test_edge_pair_requires_draft_and_rechecks_local_inputs(
    tmp_path, monkeypatch, mtp_shape, quantized, visual,
):
    import json
    from dataclasses import replace
    from tensorrt_model_connect.build import BuildRequest
    from families.qwen3_5.edge_llm import dispatch
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
    hidden, layers, intermediate, heads, kv_heads = mtp_shape
    raw = {"model_type": "qwen3_5", "text_config": {
        "num_hidden_layers": layers, "hidden_size": hidden, "intermediate_size": intermediate,
        "num_attention_heads": heads, "num_key_value_heads": kv_heads, "vocab_size": 248320,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "max_position_embeddings": 262144, "mtp_num_hidden_layers": 1,
    }}
    if quantized:
        raw["quantization_config"] = {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        (source / "hf_quant_config.json").write_text(json.dumps({
            "quantization": {"quant_algo": "NVFP4", "group_size": 16},
        }))
    if visual:
        raw["vision_config"] = {}
        request = replace(request, task="vision_language_generation")
    (source / "config.json").write_text(json.dumps(raw))
    calls = []
    def delegate(original, writer, native, **options):
        assert original is request and options == {"spec_type": "mtp"}
        with pytest.raises(NotImplementedError, match="requested MTP"):
            native(original, writer)
        calls.append(True)
    monkeypatch.setattr(dispatch, "build", delegate)
    mtp = BuildExecutionInputs("mtp")
    build_paired(request, None, mtp)
    assert calls == [True]
    with pytest.raises(ValueError, match="single-device"):
        build_paired(replace(request, tensor_parallel_size=2), None, mtp)
    with pytest.raises(ValueError, match="paired execution requires"):
        build_paired(replace(request, task="vision_language_generation"), None,
                     BuildExecutionInputs("dflash"))
    with pytest.raises(ValueError, match="four verification"):
        build_paired(replace(request, max_sequence_length=4), None, mtp)
    for change in ({"mtp_num_hidden_layers": 0}, {"hidden_size": 4097}):
        (source / "config.json").write_text(json.dumps({
            **raw, "text_config": {**raw["text_config"], **change},
        }))
        with pytest.raises(ValueError, match="single-draft"):
            build_paired(request, None, mtp)
    assert calls == [True]
    if hidden in {2560, 4096}:
        # Both linear and tree requests preserve the exact paired checkpoints.
        (source / "config.json").write_text(json.dumps(raw))
        draft.mkdir()
        (draft / "config.json").write_text(json.dumps({
            "architectures": ["DFlashDraftModel"], "hidden_size": hidden,
            "vocab_size": 248320, "num_target_layers": layers,
            "max_position_embeddings": 262144,
            "dflash_config": {"block_size": 16, "target_layer_ids": [1], "mask_token_id": 2},
        }))
        if quantized:
            (draft / "hf_quant_config.json").write_text(json.dumps({
                "producer": {"name": "modelopt"},
                "quantization": {"quant_algo": "NVFP4", "group_size": 16},
            }))
        for variant in ("dflash", "dflash_ddtree"):
            def paired(original, writer, native, **options):
                assert original is request
                expected = {"draft_dir": draft}
                if variant == "dflash_ddtree":
                    expected["spec_type"] = variant
                assert options == expected
                with pytest.raises(NotImplementedError, match="requested DFlash"):
                    native(original, writer)
            monkeypatch.setattr(dispatch, "build", paired)
            build_paired(request, None, BuildExecutionInputs(
                variant, (NamedCheckpoint("draft", draft),),
            ))


@pytest.mark.parametrize("sm", [80, 120])
@pytest.mark.parametrize("mode", ["absent", "success", "corrupt", "failure", "cancel", "device_failure"])
def test_edge_optional_package_and_output_local_staging(tmp_path, monkeypatch, caplog, mode, sm):
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
    from dataclasses import replace
    candidate_request = BuildRequest(source, tmp_path / "out", "qwen3_5", "text_generation", "fp16")
    raw_27b = {"model_type": "qwen3_5", "text_config": {
        "num_hidden_layers": 64, "hidden_size": 5120, "intermediate_size": 17408,
        "num_attention_heads": 24, "num_key_value_heads": 4, "vocab_size": 248320,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    }}
    assert not dispatch.candidate(candidate_request, raw_27b)
    tp2 = replace(candidate_request, tensor_parallel_size=2)
    assert dispatch.candidate(tp2, raw_27b)
    assert not dispatch.candidate(replace(tp2, tensor_parallel_size=4), raw_27b)
    vision_tp2 = replace(tp2, task="vision_language_generation")
    assert not dispatch.candidate(vision_tp2, raw_27b)
    assert dispatch.candidate(vision_tp2, {**raw_27b, "vision_config": {}})
    packed = tmp_path / "packed"
    packed.mkdir()
    raw_packed = {"model_type": "qwen3_5", "text_config": {
        "num_hidden_layers": 32, "hidden_size": 2560, "intermediate_size": 9216,
        "num_attention_heads": 16, "num_key_value_heads": 4, "vocab_size": 248320,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    }, "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}
    sidecar = packed / "hf_quant_config.json"
    sidecar.write_text(json.dumps({"quantization": {"quant_algo": "NVFP4", "group_size": 16}}))
    packed_request = replace(candidate_request, model_dir=packed)
    assert dispatch.candidate(packed_request, raw_packed)
    nine_b_config = {**raw_packed["text_config"],
                     "hidden_size": 4096, "intermediate_size": 12288}
    nine_b = {**raw_packed, "text_config": nine_b_config}
    assert dispatch.candidate(packed_request, nine_b)
    assert not dispatch.candidate(replace(packed_request, tensor_parallel_size=2), nine_b)
    for hidden, intermediate in ((1024, 3584), (2048, 6144)):
        small_nvfp4 = {**raw_packed, "text_config": {**raw_packed["text_config"],
            "num_hidden_layers": 24, "hidden_size": hidden,
            "intermediate_size": intermediate, "num_attention_heads": 8,
            "num_key_value_heads": 2,
        }}
        assert dispatch.candidate(packed_request, small_nvfp4)
        assert not dispatch.candidate(
            replace(packed_request, tensor_parallel_size=2), small_nvfp4,
        )
    image_request = replace(packed_request, task="vision_language_generation")
    assert not dispatch.candidate(image_request, raw_packed)
    assert dispatch.candidate(image_request, {**raw_packed, "vision_config": {}})
    assert not dispatch.candidate(replace(packed_request, tensor_parallel_size=2), raw_packed)
    for algorithm, label, group_size in (
        ("FP8", "fp8", 0), ("MXFP8", "mxfp8", 32),
        ("W4A16_AWQ", "int4_awq", 128),
        ("W8A8_SQ_PER_CHANNEL", "int8_sq", 0),
    ):
        exported = {**raw_packed, "quantization_config": {
            "quant_method": "modelopt", "quant_algo": algorithm,
        }}
        quant = {"quant_algo": algorithm, "group_size": group_size}
        sidecar.write_text(json.dumps({"quantization": quant}))
        assert builder.checkpoint_quantization(packed, exported) == label
        assert dispatch.candidate(packed_request, exported)
        nine_b = {**exported, "text_config": nine_b_config}
        assert dispatch.candidate(packed_request, nine_b)
        assert not dispatch.candidate(replace(packed_request, tensor_parallel_size=2), nine_b)
        for hidden, intermediate in ((1024, 3584), (2048, 6144)):
            small = {**exported, "text_config": {**exported["text_config"],
                "num_hidden_layers": 24, "hidden_size": hidden,
                "intermediate_size": intermediate, "num_attention_heads": 8,
                "num_key_value_heads": 2,
            }}
            assert dispatch.candidate(packed_request, small) == (
                label in {"fp8", "mxfp8", "int4_awq", "int8_sq"}
            )
            assert not dispatch.candidate(
                replace(packed_request, tensor_parallel_size=2), small,
            )
        sidecar.write_text(json.dumps({"quantization": {**quant, "group_size": 1}}))
        assert builder.checkpoint_quantization(packed, exported) is None
        sidecar.write_text(json.dumps({"quantization": quant}))
        assert not dispatch.candidate(replace(packed_request, tensor_parallel_size=2), exported)
        assert not dispatch.candidate(replace(packed_request, quantization="nvfp4"), exported)
        assert builder.checkpoint_quantization(
            packed, {**exported, "text_config": {**exported["text_config"],
                "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}},
        ) is None
        sidecar.write_text(json.dumps({"quantization": {**quant, "kv_cache_quant_algo": "FP8"}}))
        assert builder.checkpoint_quantization(packed, exported) is None
        if algorithm == "FP8":
            fp8_kv = {**exported, "quantization_config": {
                **exported["quantization_config"],
                "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
            }, "text_config": {
                **exported["text_config"], "num_hidden_layers": 24,
                "hidden_size": 2048, "intermediate_size": 6144,
                "num_attention_heads": 8, "num_key_value_heads": 2,
            }}
            assert builder.checkpoint_quantization(packed, fp8_kv) == "fp8"
            assert dispatch.candidate(packed_request, fp8_kv)
            assert not dispatch.candidate(image_request, {**fp8_kv, "vision_config": {}})
            assert not dispatch.candidate(
                packed_request, {**fp8_kv, "text_config": exported["text_config"]},
            )
            for scheme in ({"dynamic": True, "num_bits": 8, "type": "float"},
                           {"dynamic": False, "num_bits": 4, "type": "float"}):
                assert builder.checkpoint_quantization(packed, {
                    **fp8_kv, "quantization_config": {
                        **fp8_kv["quantization_config"], "kv_cache_scheme": scheme,
                    },
                }) is None
            sidecar.write_text(json.dumps({"quantization": quant}))
            assert builder.checkpoint_quantization(packed, fp8_kv) is None
        sidecar.write_text(json.dumps({"producer": {"name": "modelopt"}, "quantization": quant}))
        assert builder.checkpoint_quantization(packed, {"architectures": ["DFlashDraftModel"]}) is None
    sidecar.write_text(json.dumps({"quantization": {"quant_algo": "FP8", "group_size": 16}}))
    assert not dispatch.candidate(packed_request, raw_packed)
    # Edge's official quantize draft command keeps config.json unmodified.
    draft_raw = {"architectures": ["DFlashDraftModel"]}
    draft_quant = {"producer": {"name": "modelopt"},
                   "quantization": {"quant_algo": "NVFP4", "group_size": 16}}
    sidecar.write_text(json.dumps(draft_quant))
    assert builder.checkpoint_quantization(packed, draft_raw) == "nvfp4"
    assert builder.checkpoint_quantization(packed, {"architectures": ["UnknownDraft"]}) is None
    assert builder.checkpoint_quantization(packed, {"model_type": "qwen3_5"}) is None
    draft_quant["producer"]["name"] = "unknown"
    sidecar.write_text(json.dumps(draft_quant))
    assert builder.checkpoint_quantization(packed, draft_raw) is None
    draft_quant["producer"]["name"] = "modelopt"
    draft_quant["quantization"]["group_size"] = 32
    sidecar.write_text(json.dumps(draft_quant))
    assert builder.checkpoint_quantization(packed, draft_raw) is None
    monkeypatch.setattr(dispatch, "candidate", lambda *_: True)
    for name in ("mapped_request", "request_matches", "platform_matches"):
        if hasattr(dispatch, name):
            monkeypatch.setattr(dispatch, name, lambda *_: True)
    if hasattr(dispatch, "source_quantization"):
        monkeypatch.setattr(dispatch, "source_quantization", lambda *_: "fp16")
    if hasattr(builder, "request_weight_format"):
        monkeypatch.setattr(builder, "request_weight_format", lambda *_: "fp16")
    request = BuildRequest(source, tmp_path / "out", "qwen3_5", "text_generation", "fp16")
    if mode == "success":
        # Exercise real staging with immutable cache-style metadata; no GPU builder is needed.
        tokenizer = source / "tokenizer.json"
        tokenizer.write_text("{}")
        tokenizer.chmod(0o444)
        (source / "model.safetensors").write_bytes(b"fixture")
        staged = tmp_path / "permission-probe"
        staged.mkdir()

        class StagingChecked(Exception):
            pass

        expected_dense = "fp16"
        expected_mtp = False
        expected_visual = False
        expected_dflash = False
        expected_tree = False

        def check_staging(argv, **kwargs):
            assert argv[argv.index("--dense") + 1] == expected_dense
            assert argv[argv.index("--components") + 1] == ("all" if expected_visual else "llm")
            if expected_visual:
                assert argv[argv.index("--max-input-len") + 1] == "4096"
                assert argv[argv.index("--max-image-tokens-per-image") + 1] == "1280"
            if expected_mtp:
                assert argv[argv.index("--spec-type") + 1] == "mtp"
                assert argv[argv.index("--max-verify-tree-size") + 1] == "4"
                assert argv[argv.index("--max-draft-tree-size") + 1] == "4"
                assert "--draft-model-dir" not in argv
            assert ("--tree-base" in argv) == expected_tree
            if expected_dflash:
                assert argv[argv.index("--spec-type") + 1] == "dflash"
                assert argv[argv.index("--max-verify-tree-size") + 1] == "16"
                assert argv[argv.index("--max-draft-tree-size") + 1] == "16"
                assert "--draft-model-dir" in argv
            copied = staged / "edge_llm/checkpoint/tokenizer.json"
            assert copied.read_bytes() == tokenizer.read_bytes()
            assert copied.stat().st_mode & 0o200
            assert not tokenizer.stat().st_mode & 0o200
            raise StagingChecked

        with monkeypatch.context() as patch:
            patch.setattr(builder, "installed_package", lambda _: {"python": "python", "plugin": "plugin"})
            patch.setattr(builder, "builder_python", lambda *_: "python")
            patch.setattr(builder.subprocess, "run", check_staging)
            with pytest.raises(StagingChecked):
                builder.prepare(request, {"max_position_embeddings": 4096}, {"sm": 80},
                                staged, tmp_path / "permission-probe.log")
            expected_dense = "auto"
            for quantization in ("nvfp4", "fp8", "mxfp8", "int4_awq", "int8_sq"):
                staged = tmp_path / (quantization + "-policy-probe")
                staged.mkdir()
                patch.setattr(builder, "checkpoint_quantization", lambda *_: quantization)
                with pytest.raises(StagingChecked):
                    builder.prepare(request, {"max_position_embeddings": 4096}, {"sm": 120},
                                    staged, tmp_path / (quantization + "-policy-probe.log"))
            expected_mtp = True
            expected_dense = "fp16"
            staged = tmp_path / "mtp-policy-probe"
            staged.mkdir()
            patch.setattr(builder, "checkpoint_quantization", lambda *_: "none")
            with pytest.raises(StagingChecked):
                builder.prepare(request, {"max_position_embeddings": 4096}, {"sm": sm},
                                staged, tmp_path / "mtp-policy-probe.log", spec_type="mtp")
            expected_visual = True
            for quantization, expected_dense in (("none", "fp16"), ("nvfp4", "auto")):
                staged = tmp_path / f"mtp-visual-{quantization}"
                staged.mkdir()
                patch.setattr(builder, "checkpoint_quantization", lambda *_: quantization)
                with pytest.raises(StagingChecked):
                    builder.prepare(
                        replace(request, task="vision_language_generation"),
                        {"max_position_embeddings": 8192, "vision_config": {}}, {"sm": sm},
                        staged, tmp_path / f"mtp-visual-{quantization}.log", spec_type="mtp",
                    )
        with monkeypatch.context() as patch:
            patch.setattr(builder, "installed_package", lambda _: {"python": "python", "plugin": "plugin"})
            patch.setattr(builder, "builder_python", lambda *_: "python")
            patch.setattr(builder.subprocess, "run", check_staging)
            expected_mtp = expected_visual = False
            expected_dflash = True
            expected_dense = "fp16"
            patch.setattr(builder, "checkpoint_quantization", lambda *_: "none")
            draft_probe = tmp_path / "draft-policy-source"
            draft_probe.mkdir()
            (draft_probe / "config.json").write_text("{}")
            (draft_probe / "model.safetensors").write_bytes(b"fixture")
            patch.setattr(builder, "copy_draft_safetensors",
                          lambda source, target: target.write_bytes(source.read_bytes()))
            for variant, expected_visual, quantization in (
                ("dflash", False, "none"), ("dflash_ddtree", False, "none"),
                ("dflash", True, "none"), ("dflash_ddtree", True, "none"),
                ("dflash", True, "nvfp4"), ("dflash_ddtree", True, "nvfp4"),
            ):
                expected_tree = variant == "dflash_ddtree"
                expected_dense = "auto" if quantization == "nvfp4" else "fp16"
                patch.setattr(builder, "checkpoint_quantization", lambda *_: quantization)
                label = f"{variant}-{expected_visual}-{quantization}"
                staged = tmp_path / (label + "-policy-probe")
                staged.mkdir()
                probe = (replace(request, task="vision_language_generation")
                         if expected_visual else request)
                raw = {"max_position_embeddings": 4096}
                if expected_visual:
                    raw["vision_config"] = {}
                with pytest.raises(StagingChecked):
                    builder.prepare(
                        probe, raw, {"sm": sm}, staged, tmp_path / (label + "-policy.log"),
                        draft_dir=draft_probe, spec_type=variant,
                    )
            staged = tmp_path / "tp2-visual-probe"
            staged.mkdir()
            commands = []
            (source / "baked-unused.safetensors").write_bytes(b"already embedded in the engine")

            def build_component(argv, **kwargs):
                commands.append(argv)
                from pathlib import Path
                engine = Path(argv[argv.index("--engine-dir") + 1])
                if argv[argv.index("--components") + 1] == "llm":
                    assert argv[argv.index("--tp-size") + 1] == "2"
                    rank = argv[argv.index("--tp-rank") + 1]
                    names = [f"llm_world2_rank{rank}.engine", "config_world2.json",
                             "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
                else:
                    assert argv[argv.index("--components") + 1] == "visual"
                    assert argv[argv.index("--tp-size") + 1] == "1"
                    assert "--tp-rank" not in argv
                    names = ["visual/visual.engine", "visual/config.json",
                             "visual/preprocessor_config.json"]
                for name in names:
                    output = engine / name
                    output.parent.mkdir(parents=True, exist_ok=True)
                    payload = ({"checkpoint_identity": {"version": 1, "sources": {
                        "component": {"files": {"model.safetensors": {"bytes": 7}}}
                    }}} if name in {"config_world2.json", "visual/config.json"} else {})
                    output.write_text(json.dumps(payload))

            patch.setattr(builder.subprocess, "run", build_component)
            files, marker = builder.prepare(
                replace(request, task="vision_language_generation", tensor_parallel_size=2),
                {"max_position_embeddings": 8192, "vision_config": {}}, {"sm": sm},
                staged, tmp_path / "tp2-visual-probe.log",
            )
            assert len(commands) == 3
            assert marker["tensor_parallel_size"] == 2
            assert marker["task"] == "vision_language_generation"
            assert "edge_llm/engine/visual/visual.engine" in files
            assert not (staged / "edge_llm/checkpoint").exists()
            assert files["edge_llm/checkpoint/model.safetensors"] == source / "model.safetensors"
            assert "edge_llm/checkpoint/baked-unused.safetensors" not in files
            assert (source / "baked-unused.safetensors").is_file()
            with pytest.raises(ValueError, match="checkpoint identity"):
                builder.runtime_checkpoint_files(source, {})
            with pytest.raises(ValueError, match="Missing or changed"):
                builder.runtime_checkpoint_files(source, {"checkpoint_identity": {
                    "version": 1, "sources": {"component": {"files": {
                        "model.safetensors": {"bytes": 8}
                    }}}
                }})

    class Published:
        def set_header(self, **header):
            self.header = header

        def add_json(self, name, value):
            assert name == "edge_llm.json"

    published = Published()
    for marker in ({}, {"execution_variant": "mtp"}):
        builder.publish(replace(request, task="vision_language_generation"), published, {}, marker)
        assert published.header["task"] == "images_text_to_text"
    builder.publish(request, published, {}, {})
    assert published.header["task"] == "text_generation"
    writer = object()
    target = {"os": "linux", "arch": "x86_64", "sm": sm}
    assert dispatch.EDGE_DISPATCH[("linux", "x86_64", sm, "fp16")] is builder.prepare
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

    monkeypatch.setattr(dispatch, "EDGE_DISPATCH", {("linux", "x86_64", sm, "fp16"): prepare})
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

    if mode == "success":
        # An installed wheel supplies Python tooling, never the native C++ SDK.
        from types import SimpleNamespace

        package = {"python": "/sdk/private/python"}
        platform = {"tensorrt_version": "11.1.0.106"}
        responses = iter([
            '["0.11.0", "11.1.0.106"]',
            '["0.10.1", "11.1.0.106"]',
            '["0.11.0", "11.0.0.114"]',
            "not JSON",
        ])
        def probe(command, **kwargs):
            assert command[0] == builder.sys.executable
            assert command[1:3] == ["-I", "-c"]
            assert kwargs == dict(check=True, capture_output=True, text=True, timeout=30)
            return SimpleNamespace(stdout=next(responses))

        monkeypatch.setattr(builder.subprocess, "run", probe)
        assert builder.builder_python(package, platform) == builder.sys.executable
        for _ in range(3):
            assert builder.builder_python(package, platform) == package["python"]

        def unavailable(*args, **kwargs):
            raise builder.subprocess.TimeoutExpired("python", 30)

        monkeypatch.setattr(builder.subprocess, "run", unavailable)
        assert builder.builder_python(package, platform) == package["python"]



@pytest.mark.parametrize("options", [
    [], ["--precision", "fp16", "--max-sequence-length", "64"],
    ["--precision", "fp16", "--tensor-parallel-size", "2"],
])
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


    if not options:
        from pathlib import Path
        from types import SimpleNamespace
        from families.qwen3_5.tests import test_e2e as e2e
        with monkeypatch.context() as patch:
            cases_dir = tmp_path / "cases"
            (cases_dir / "manifests").mkdir(parents=True)
            manifest = {"family": "qwen3_5", "task": "vision_language_generation",
                        "precision": "fp16", "max_sequence_length": 4096,
                        "tensor_parallel_size": 2, "testcases": [{"name": "image-tp2"}]}
            (cases_dir / "manifests/image.json").write_text(json.dumps(manifest))
            patch.setattr(e2e, "_TEST_DIR", cases_dir)
            assert e2e._load_cases()["image-tp2"][0]["task"] == "vision_language_generation"
            requests = []
            def build_case(request):
                requests.append(request)
                request.output_path.write_bytes(b"bundle fixture")
            patch.setattr(e2e, "build", build_case)
            e2e._build_bundle(manifest, source, output)
            assert requests[0].task == "vision_language_generation"
            assert requests[0].tensor_parallel_size == 2
            patch.setattr(e2e.shutil, "which", lambda _name: "/usr/bin/mpirun")
            for edge in (False, True):
                for task in ("text_generation", "vision_language_generation"):
                    sections = ["engine.rank0.plan", "engine.rank1.plan"]
                    if edge:
                        sections = ["edge_llm.json", "edge_llm/engine/llm_world2_rank0.engine",
                                    "edge_llm/engine/llm_world2_rank1.engine"]
                        if task == "vision_language_generation":
                            sections.append("edge_llm/engine/visual/visual.engine")
                    inspected = {"family": "qwen3_5", "task": (
                        "images_text_to_text" if edge and task == "vision_language_generation" else task
                    ), "sections": sections}
                    calls = []
                    result = {"text": "fixture", "token_ids": [1]}
                    def run(command, **kwargs):
                        calls.append((command, kwargs))
                        if command[1] == "inspect":
                            stdout = json.dumps(inspected)
                        else:
                            stdout = ("" if edge else "[1,0]<stdout>:") + json.dumps(result)
                        return SimpleNamespace(stdout=stdout, stderr="", args=command,
                                               check_returncode=lambda: None)
                    patch.setattr(e2e.subprocess, "run", run)
                    assert e2e._assert_rank_sections(Path("trtmc"), output, 2, task) is edge
                    if edge:
                        assert e2e._assert_rank_sections(
                            Path("trtmc"), output, 2, task, require_edge=True,
                        )
                    else:
                        with pytest.raises(AssertionError, match="requires Edge-LLM"):
                            e2e._assert_rank_sections(
                                Path("trtmc"), output, 2, task, require_edge=True,
                            )
                    assert e2e._run_native(Path("trtmc"), tmp_path, output, "Describe.",
                                          {"max_new_tokens": 8}, 2, tmp_path, edge=edge) == result
                    assert (calls[-1][0][0] == "trtmc") is edge
                    if edge:
                        assert "TRTMC_NCCL_RENDEZVOUS" not in calls[-1][1]["env"]
            def timed_out(command, **kwargs):
                assert kwargs["timeout"] == 600
                raise e2e.subprocess.TimeoutExpired(
                    command, 600, output=b"partial stdout", stderr=b"partial stderr",
                )
            patch.setattr(e2e.subprocess, "run", timed_out)
            with pytest.raises(e2e.subprocess.TimeoutExpired):
                e2e._run_native(
                    Path("trtmc"), tmp_path, output, "Describe.",
                    {"max_new_tokens": 8}, 2, tmp_path, edge=True,
                )
            assert (tmp_path / "native.stdout.log").read_text() == "partial stdout"
            assert (tmp_path / "native.stderr.log").read_text() == "partial stderr"
            # Compare identical prompt-template defaults, including the omitted option.
            from PIL import Image
            from transformers import AutoProcessor
            image = tmp_path / "image.png"
            Image.new("RGB", (2, 2)).save(image)
            templates = []
            class Processor:
                image_processor = SimpleNamespace(patch_size=14, merge_size=2)
                chat_template = "template"
                def apply_chat_template(self, messages, **kwargs):
                    templates.append(kwargs)
                    return "rendered"
                def __call__(self, **kwargs):
                    return {"image_grid_thw": SimpleNamespace(tolist=lambda: [[1, 1, 1]])}
            patch.setattr(AutoProcessor, "from_pretrained", lambda *_a, **_k: Processor())
            for option in ({}, {"enable_thinking": False}, {"enable_thinking": True}):
                e2e._reference_inputs(source, manifest, SimpleNamespace(chat_template="fallback"),
                                      "Describe.", {"image": image, **option})
                if option:
                    assert templates[-1]["enable_thinking"] is option["enable_thinking"]
                else:
                    assert "enable_thinking" not in templates[-1]


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
