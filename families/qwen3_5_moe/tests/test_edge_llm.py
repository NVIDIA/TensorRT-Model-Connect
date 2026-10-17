# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-local CPU route, metadata and source precision contracts."""

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from tensorrt_model_connect.build import BuildRequest
from tensorrt_model_connect.model_support import ModelMetadata
from families.qwen3_5_moe.edge_llm import dispatch, builder as edge_llm, precision
from families.qwen3_5_moe.build_request import BuildRequest as FamilyBuildRequest
from families.qwen3_5_moe import support

TARGET = {
    "os": "linux",
    "os_version": "24.04",
    "arch": "x86_64",
    "sm": 80,
    "cuda_version": "13.3",
    "tensorrt_version": "11.1.0.106",
}


@pytest.fixture(params=["gptq", "nvfp4"])
def source(request, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    fixtures = Path(__file__).parent / "data"
    raw = json.loads((fixtures / f"{request.param}-config.json").read_text())
    (checkpoint / "config.json").write_text(json.dumps(raw))
    side = fixtures / f"{request.param}-hf_quant_config.json"
    if side.exists():
        (checkpoint / "hf_quant_config.json").write_bytes(side.read_bytes())
    build = BuildRequest(
        checkpoint, tmp_path / "model.bundle", "qwen3_5_moe", "vision_language_generation", "fp16"
    )
    return build, raw, request.param


def test_documented_config_and_source_policy(source):
    request, raw, kind = source
    precision.validate_config(raw)
    policy = precision.source_policy(request, raw)
    assert policy["visual"] == "fp16"
    assert policy["weights"] == ("int4" if kind == "gptq" else "nvfp4")
    assert policy["kv"] == ("fp16" if kind == "gptq" else "fp8")
    if kind == "nvfp4":
        assert len(policy["layers"]) == 291


@pytest.mark.parametrize("quantization", [None, "none", "int4", "nvfp4", "fp8", "awq"])
def test_unspecified_preserves_named_exact_none_rejects(source, quantization):
    request, raw, kind = source
    request = replace(request, quantization=quantization)
    if quantization in (None, "int4" if kind == "gptq" else "nvfp4"):
        assert precision.source_policy(request, raw)["weights"]
    else:
        with pytest.raises(ValueError):
            precision.source_policy(request, raw)


@pytest.mark.parametrize(
    "updates",
    [
        {"family": "qwen3_5"},
        {"task": "transcription"},
        {"backend": "trt_rtx"},
        {"precision": "bf16"},
        {"precision": "fp32"},
        {"max_batch_size": 2},
        {"tensor_parallel_size": 2},
        {"context_parallel_size": 2},
        {"dynamic_kv_cache": True},
        {"mtp_seq_len": 2},
        {"fp32_layers": (0,)},
        {"graph_transform": lambda *a: None},
        {"image_height": 448},
        {"image_width": 448},
        {"video_num_frames": 1},
        {"max_sequence_length": 0},
        {"max_sequence_length": 1},
        {"max_sequence_length": -1},
    ],
)
def test_unmapped_public_build_controls_fail(source, updates):
    request, _, _ = source
    with pytest.raises(ValueError):
        dispatch.validate_request(replace(request, **updates))


def test_family_build_request_rejects_another_family(tmp_path):
    with pytest.raises(ValueError, match="family must be 'qwen3_5_moe'"):
        FamilyBuildRequest(
            tmp_path,
            tmp_path / "model.bundle",
            "other_family",
            "vision_language_generation",
            "fp16",
        )


@pytest.mark.parametrize(
    "key",
    [
        "num_hidden_layers",
        "hidden_size",
        "num_experts",
        "num_experts_per_tok",
        "shared_expert_intermediate_size",
        "linear_num_value_heads",
        "head_dim",
    ],
)
def test_conflicting_topology_rejects(source, key):
    _, raw, _ = source
    raw["text_config"][key] += 1
    with pytest.raises(ValueError):
        precision.validate_config(raw)


@pytest.mark.parametrize(
    "key",
    [
        "draft_config",
        "dflash_config",
        "dspark_config",
        "jetspec_config",
        "eagle_config",
        "speculative_config",
        "audio_config",
    ],
)
def test_explicit_other_execution_mode_rejects(source, key):
    _, raw, _ = source
    raw[key] = {"enabled": True}
    with pytest.raises(ValueError):
        precision.validate_config(raw)


def test_config_layer_order_and_architecture_reject(source):
    _, raw, _ = source
    raw["text_config"]["layer_types"][0] = "full_attention"
    with pytest.raises(ValueError):
        precision.validate_config(raw)
    raw["architectures"] = ["Qwen3MoeForCausalLM"]
    with pytest.raises(ValueError):
        precision.validate_config(raw)


def test_component_quantization_cannot_override_root(source):
    request, raw, _ = source
    raw["vision_config"]["quantization_config"] = {"quant_algo": "FP8"}
    with pytest.raises(ValueError):
        precision.source_policy(request, raw)


def test_mixed_layer_coverage_group_and_raw_dominance():
    valid = precision.expected_mixed_layers()
    assert precision._mixed_layers(valid) == valid
    for mutation in ("missing", "extra", "group", "alias", "unknown"):
        value = json.loads(json.dumps(valid))
        key = next(k for k, v in value.items() if v["quant_algo"] == "W4A16_NVFP4")
        if mutation == "missing":
            value.pop(key)
        if mutation == "extra":
            value["unknown"] = {"quant_algo": "FP8"}
        if mutation == "group":
            value[key]["group_size"] = 32
        if mutation == "alias":
            value[key]["quant_algo"] = "NVFP4"
        if mutation == "unknown":
            value[key]["quant_method"] = "awq"
        with pytest.raises(ValueError):
            precision._mixed_layers(value)


def test_source_sidecar_conflict_rejects(source):
    request, raw, kind = source
    if kind == "gptq":
        (request.model_dir / "hf_quant_config.json").write_text("{}")
    else:
        raw["quantization_config"]["ignore"] = ["model.visual.*"]
    with pytest.raises(ValueError):
        precision.source_policy(request, raw)


def test_success_publishes_without_fake_native(source, monkeypatch):
    request, _, kind = source
    target = {**TARGET, "sm": 100 if kind == "nvfp4" else 80}
    monkeypatch.setattr(edge_llm, "local_target", lambda: target)
    prepare = Mock(return_value=({}, {}))
    publish = Mock()
    monkeypatch.setattr(
        dispatch,
        "EDGE_DISPATCH",
        {("linux", "x86_64", target["sm"], "nvfp4" if kind == "nvfp4" else "int4"): prepare},
    )
    monkeypatch.setattr(edge_llm, "publish", publish)
    writer = object()
    dispatch.build(request, writer)
    prepare.assert_called_once()
    publish.assert_called_once_with(request, writer, {}, {})
    staged = prepare.call_args.args[3]
    assert staged.parent == request.output_path.parent
    assert staged.name.startswith(f".{request.output_path.name}.edge-stage-")
    assert not staged.exists()
    assert not prepare.call_args.args[4].exists()


def test_preparation_failure_warns_and_preserves_cause_once(source, monkeypatch, caplog):
    request, _, kind = source
    target = {**TARGET, "sm": 100 if kind == "nvfp4" else 80}
    monkeypatch.setattr(edge_llm, "local_target", lambda: target)
    error = RuntimeError("genuine upstream error")
    prepare = Mock(side_effect=error)
    monkeypatch.setattr(
        dispatch,
        "EDGE_DISPATCH",
        {("linux", "x86_64", target["sm"], "nvfp4" if kind == "nvfp4" else "int4"): prepare},
    )
    with pytest.raises(RuntimeError, match="native implementation unavailable") as result:
        dispatch.build(request, object())
    assert result.value.__cause__ is error
    prepare.assert_called_once()
    assert len(caplog.records) == 1 and "Diagnostics:" in caplog.text
    logs = list(request.output_path.parent.glob(".*.edge-*.log"))
    assert len(logs) == 1 and "genuine upstream error" in logs[0].read_text()


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(130)])
def test_cancellation_never_wrapped(source, monkeypatch, error):
    request, _, kind = source
    target = {**TARGET, "sm": 100 if kind == "nvfp4" else 80}
    monkeypatch.setattr(edge_llm, "local_target", lambda: target)
    prepare = Mock(side_effect=error)
    monkeypatch.setattr(
        dispatch,
        "EDGE_DISPATCH",
        {("linux", "x86_64", target["sm"], "nvfp4" if kind == "nvfp4" else "int4"): prepare},
    )
    with pytest.raises(type(error)):
        dispatch.build(request, object())
    prepare.assert_called_once()


def test_raw_tokenizer_and_complete_eos_union():
    tokenizer = {
        "post_processor": {"type": "ByteLevel"},
        "added_tokens": [{"content": "<|im_end|>", "id": 248046}],
    }
    config = {"eos_token": "<|im_end|>", "add_bos_token": False}
    assert edge_llm.validate_tokenizer(tokenizer, config, {"eos_token_id": [248044]}) == [
        248044,
        248046,
    ]
    for invalid in ([248046], [248044, 99], [], None):
        with pytest.raises(ValueError):
            edge_llm.validate_tokenizer(tokenizer, config, {"eos_token_id": invalid})
    with pytest.raises(ValueError):
        edge_llm.validate_tokenizer(tokenizer, {**config, "add_bos_token": True})


def test_nvfp4_does_not_route_to_ampere():
    assert ("linux", "x86_64", 80, "nvfp4") not in dispatch.EDGE_DISPATCH
    assert ("linux", "x86_64", 80, "int4") in dispatch.EDGE_DISPATCH


def test_full_checkpoint_defaults_and_no_standalone_text_discovery(source):
    _, raw, _ = source
    found = support.describe(ModelMetadata(raw, {}))
    assert found.default_task == "vision_language_generation"
    assert found.default_precision == "fp16"
    assert set(found.tasks) == {"text_generation", "vision_language_generation"}
    assert support.describe(ModelMetadata({"model_type": "qwen3_5_moe_text"}, {})) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("kv_cache_quant_algo", "FP16"),
        ("kv_cache_quant_algo", None),
        ("kv_cache_quant_algo", "future"),
        ("kv_cache_scheme", None),
        ("kv_cache_scheme", {"num_bits": 4, "type": "float"}),
        ("kv_cache_scheme", {"num_bits": 8, "type": "int"}),
        ("kv_cache_scheme", {"num_bits": 8, "type": "float", "dynamic": True}),
    ],
)
def test_explicit_embedded_kv_conflict(source, field, value):
    request, raw, kind = source
    if kind != "nvfp4":
        return
    raw["quantization_config"][field] = value
    with pytest.raises(ValueError, match="KV"):
        precision.source_policy(request, raw)


@pytest.mark.parametrize(
    "field,value", [("symmetric", False), ("strategy", "channel"), ("zero_point", True)]
)
def test_group_precision_modifiers_conflict(source, field, value):
    request, raw, kind = source
    if kind != "nvfp4":
        return
    raw["quantization_config"]["config_groups"]["group_0"]["weights"][field] = value
    with pytest.raises(ValueError):
        precision.source_policy(request, raw)


def test_consistent_explicit_kv_and_group_modifiers(source):
    request, raw, kind = source
    if kind != "nvfp4":
        return
    quant = raw["quantization_config"]
    quant["kv_cache_quant_algo"] = "FP8"
    quant["kv_cache_scheme"] = {
        "num_bits": 8,
        "type": "float",
        "strategy": "tensor",
        "symmetric": True,
    }
    for name, strategy in (("group_0", "tensor"), ("group_1", "group")):
        for field in ("weights", "input_activations"):
            quant["config_groups"][name][field].update(
                strategy=strategy, symmetric=True, exporter_note="harmless"
            )
    assert precision.source_policy(request, raw)["kv"] == "fp8"


@pytest.mark.parametrize("task", ["text_generation", "vision_language_generation"])
@pytest.mark.parametrize("capacity", [None, 1024, 2])
@pytest.mark.parametrize("variant", [None, "dflash", "dflash_ddtree", "mtp"])
def test_direct_builder_defaults_complete_components_and_packed_arguments(
    source, monkeypatch, tmp_path, task, capacity, variant
):
    from families.qwen3_5_moe.edge_llm import checkpoint

    request, raw, kind = source
    request = replace(request, task=task, max_sequence_length=capacity)
    if variant == "mtp":
        from families.qwen3_5_moe.edge_llm.cli import execution_inputs
        from families.qwen3_5_moe.edge_llm.config import with_execution

        request = with_execution(request, execution_inputs("mtp"))
        if kind == "gptq" or capacity == 2:
            with pytest.raises(ValueError, match="NVFP4|context"):
                dispatch.validate_execution(request, raw)
            return
        assert dispatch.validate_execution(request, raw) is None
        invalid = json.loads(json.dumps(raw))
        invalid["text_config"]["mtp_num_hidden_layers"] = 0
        with pytest.raises(ValueError, match="prediction layer"):
            dispatch.validate_execution(request, invalid)
        with pytest.raises(ValueError, match="embedded"):
            dispatch.validate_execution(with_execution(request, execution_inputs(
                "mtp", [f"draft={request.model_dir}"])), raw)
    elif variant is not None:
        from families.qwen3_5_moe.edge_llm.cli import execution_inputs
        from families.qwen3_5_moe.edge_llm.config import with_execution

        draft_dir = tmp_path / "draft"
        draft_dir.mkdir()
        draft = {
            "architectures": ["DFlashDraftModel"], "hidden_size": 2048,
            "vocab_size": 248320, "num_target_layers": 40,
            "max_position_embeddings": 262144,
            "dflash_config": {"block_size": 16, "mask_token_id": 248077,
                              "target_layer_ids": [1, 6, 11, 16, 22, 27, 32, 37]},
        }
        (draft_dir / "config.json").write_text(json.dumps(draft))
        (draft_dir / "model.safetensors").write_bytes(b"draft fixture")
        execution = execution_inputs(variant, [f"draft={draft_dir}"])
        request = with_execution(request, execution)
        if capacity == 2:
            with pytest.raises(ValueError, match="context"):
                edge_llm.prepare(request, raw, TARGET, tmp_path / "stage", tmp_path / "build.log")
            return
        for field in ("hidden_size", "vocab_size", "num_target_layers"):
            (draft_dir / "config.json").write_text(json.dumps({**draft, field: -1}))
            with pytest.raises(ValueError):
                dispatch.validate_execution(request, raw)
        (draft_dir / "config.json").write_text(json.dumps(draft))
        assert dispatch.validate_execution(request, raw) == draft_dir
        with pytest.raises(ValueError, match="exactly one"):
            dispatch.validate_execution(with_execution(request, execution_inputs(variant)), raw)
    root = Path(request.model_dir)
    (root / "model.safetensors").write_bytes(b"fixture")
    tokenizer = {
        "post_processor": {"type": "ByteLevel"},
        "added_tokens": [{"id": 248046, "content": "<|im_end|>"}],
    }
    tokconfig = {"eos_token": "<|im_end|>"}
    (root / "tokenizer.json").write_text(json.dumps(tokenizer))
    (root / "tokenizer_config.json").write_text(json.dumps(tokconfig))
    monkeypatch.setattr(
        edge_llm,
        "installed_package",
        lambda target: {"python": "edge-python", "plugin": "edge-plugin"},
    )
    checked = []
    monkeypatch.setattr(
        checkpoint,
        "validate_checkpoint",
        lambda path, fmt: (
            checked.append((path, fmt)) or {"tensor_count": 1, "shard_count": 1, "layout": fmt}
        ),
    )
    commands = []

    def builder(command, **kwargs):
        commands.append(command)
        engine = Path(command[command.index("--engine-dir") + 1])
        engine.mkdir(parents=True)
        for name, value in (
            ("base_config.json" if variant else "config.json", {"eos_token_id": [248044]}),
            ("tokenizer.json", tokenizer),
            ("tokenizer_config.json", tokconfig),
            ("chat_template.jinja", {}),
        ):
            (engine / name).write_text(json.dumps(value))
        if variant:
            (engine / "spec_base.engine").write_bytes(b"base engine")
            (engine / "spec_draft.engine").write_bytes(b"draft engine")
            (engine / "draft_config.json").write_text("{}")
        else:
            (engine / "llm.engine").write_bytes(b"engine")
        if task == "vision_language_generation":
            (engine / "visual").mkdir()
            (engine / "visual/visual.engine").write_bytes(b"visual")
            (engine / "visual/config.json").write_text("{}")

    monkeypatch.setattr(edge_llm, "builder_python", lambda package, target: package["python"])
    monkeypatch.setattr(edge_llm.subprocess, "run", builder)
    files, marker = edge_llm.prepare(
        request, raw, TARGET, tmp_path / "stage", tmp_path / "build.log"
    )
    command = commands[0]
    assert len(commands) == 1 and checked == [(tmp_path / "stage/edge_llm/checkpoint", "int4" if kind == "gptq" else "nvfp4")]
    if variant == "mtp":
        assert marker["execution_variant"] == "mtp"
        assert marker["mtp_draft_steps"] == 3 and marker["mtp_verify_size"] == 4
        assert "dflash_block_size" not in marker and "--draft-model-dir" not in command
        assert command[command.index("--spec-type") + 1] == "mtp"
        assert command[command.index("--max-verify-tree-size") + 1] == "4"
        assert command[command.index("--max-draft-tree-size") + 1] == "4"
        assert "edge_llm/engine/spec_draft.engine" in files
        assert not (tmp_path / "stage/edge_llm/checkpoint/draft").exists()
    elif variant:
        assert marker["execution_variant"] == variant and marker["dflash_block_size"] == 16
        assert command[command.index("--spec-type") + 1] == "dflash"
        assert command[command.index("--max-verify-tree-size") + 1] == "16"
        staged_draft = tmp_path / "stage/edge_llm/checkpoint/draft"
        assert command[command.index("--draft-model-dir") + 1] == str(staged_draft)
        assert (staged_draft / "model.safetensors").read_bytes() == b"draft fixture"
        assert "edge_llm/engine/spec_draft.engine" in files
    else:
        assert "--spec-type" not in command and "execution_variant" not in marker
    assert command[command.index("--dense") + 1] == "auto"
    assert command[command.index("--components") + 1] == (
        "llm,visual" if task == "vision_language_generation" else "llm"
    )
    assert marker["max_sequence_length"] == (4096 if capacity is None else capacity)
    assert marker["task"] == task and marker["eos_token_ids"] == [248044, 248046]
    assert marker["visual_profile"] == (
        {"min": 4, "max": 1024, "per_image": 512} if task == "vision_language_generation" else None
    )
    assert [command[i + 1] for i, arg in enumerate(command) if arg == "--externalize-weights"] == [
        "all",
    ]
    assert ("edge_llm/engine/visual/visual.engine" in files) == (
        task == "vision_language_generation"
    )


def test_missing_direct_builder_artifacts_are_preparation_failures(source, monkeypatch, tmp_path):
    from families.qwen3_5_moe.edge_llm import checkpoint

    request, raw, _ = source
    root = Path(request.model_dir)
    (root / "model.safetensors").write_bytes(b"fixture")
    (root / "tokenizer.json").write_text(
        json.dumps(
            {
                "post_processor": {"type": "ByteLevel"},
                "added_tokens": [{"id": 248046, "content": "<|im_end|>"}],
            }
        )
    )
    (root / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<|im_end|>"}))
    monkeypatch.setattr(checkpoint, "validate_checkpoint", lambda *args: {})
    monkeypatch.setattr(
        edge_llm,
        "installed_package",
        lambda target: {"python": "edge-python", "plugin": "edge-plugin"},
    )
    monkeypatch.setattr(edge_llm, "builder_python", lambda package, target: package["python"])
    monkeypatch.setattr(edge_llm.subprocess, "run", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="required artifact"):
        edge_llm.prepare(request, raw, TARGET, tmp_path / "stage", tmp_path / "build.log")
