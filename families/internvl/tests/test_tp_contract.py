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


@pytest.mark.parametrize("format_name", ["safetensors", "torch_sharded"])
def test_source_awq_weights_preserve_packed_values(tmp_path, format_name) -> None:
    import json
    from families.internvl.checkpoint_mapper import _load_projection

    # Independent forward packing includes high-bit INT32 values and two groups.
    # Changing either the column permutation or AWQ zero offset breaks equality.
    values = (np.arange(256 * 16).reshape(256, 16) % 16).astype(np.uint32)
    zeros = np.array([[8] * 16, [3] * 16], dtype=np.uint32)
    scales = np.array([[0.125] * 16, [0.0625] * 16], dtype=np.float16)

    def pack(array):
        reordered = array.reshape(array.shape[0], -1, 8)[:, :, [0, 2, 4, 6, 1, 3, 5, 7]]
        return np.bitwise_or.reduce(reordered << np.arange(0, 32, 4, dtype=np.uint32), axis=2).view(np.int32)

    tensors = {"proj.qweight": pack(values), "proj.qzeros": pack(zeros), "proj.scales": scales}
    if format_name == "safetensors":
        from safetensors.numpy import save_file
        save_file(tensors, tmp_path / "model.safetensors")
    else:
        import torch
        shard = "pytorch_model-00001-of-00001.bin"
        torch.save({name: torch.from_numpy(value) for name, value in tensors.items()}, tmp_path / shard)
        (tmp_path / "pytorch_model.bin.index.json").write_text(
            json.dumps({"weight_map": {name: shard for name in tensors}}))
    readers = model._open_checkpoint(tmp_path)
    config = {"quant_method": "awq", "bits": 4, "group_size": 128, "version": "GEMM", "zero_point": True}
    actual = _load_projection(readers, "proj", config)
    expected = ((values.astype(np.float32) - np.repeat(zeros, 128, axis=0))
                * np.repeat(scales, 128, axis=0)).astype(np.float16).T
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert actual.dtype == np.float32
    for bad in ({**config, "version": "gemv"}, {**config, "group_size": 64},
                {**config, "zero_point": False}, None):
        with pytest.raises(ValueError, match="group128 GEMM AWQ"):
            _load_projection(readers, "proj", bad)


def test_original_vision_layout_maps_norm_before_projector() -> None:
    mapping = {
        "vision_model.embeddings.class_embedding": "vision_tower.embeddings.cls_token",
        "vision_model.embeddings.patch_embedding.weight": "vision_tower.embeddings.patch_embeddings.projection.weight",
        "vision_model.encoder.layers.0.norm1.weight": "vision_tower.encoder.layer.0.layernorm_before.weight",
        "vision_model.encoder.layers.0.ls1": "vision_tower.encoder.layer.0.lambda_1",
        "vision_model.encoder.layers.0.attn.qkv.weight": "vision_tower.encoder.layer.0.attn.qkv.weight",
        "mlp1.0.weight": "multi_modal_projector.layer_norm.weight",
        "mlp1.1.weight": "multi_modal_projector.linear_1.weight",
        "mlp1.3.bias": "multi_modal_projector.linear_2.bias",
        "multi_modal_projector.linear_1.weight": "multi_modal_projector.linear_1.weight",
    }
    assert {name: model._native_vision_key(name) for name in mapping} == mapping


def test_native_pixels_use_published_internvl_normalization() -> None:
    config = SimpleNamespace(raw={"vision_config": {"patch_size": 14}}, hidden_size=1536)
    runtime = model._InternVLModel().get_vl_config(config)
    assert runtime["image_mean"] == [0.485, 0.456, 0.406]
    assert runtime["image_std"] == [0.229, 0.224, 0.225]


def test_original_nested_decoder_config_survives_root_hidden_size() -> None:
    import json

    config = model.ModelConfig.from_json(json.dumps({
        "model_type": "internvl_chat", "hidden_size": 16,
        "vision_config": {"hidden_size": 32},
        "llm_config": {"model_type": "qwen2", "hidden_size": 16, "vocab_size": 128,
                       "rope_theta": 1000000, "num_hidden_layers": 2, "num_attention_heads": 4},
    }))
    assert config.model_type == "internvl_chat"
    assert config.hidden_size == 16
    assert config.vocab_size == 128
    assert config.rope_theta == 1000000
    assert config.num_hidden_layers == 2
    assert config.raw["vision_config"]["hidden_size"] == 32


def test_text_loader_preserves_qwen3_per_head_norms(tmp_path) -> None:
    from safetensors.numpy import save_file

    config = model.ModelConfig.create_tiny("internvl", num_hidden_layers=1)
    tensors = {
        "model.embed_tokens.weight": np.ones((32, 16), dtype=np.float32),
        "model.layers.0.input_layernorm.weight": np.ones(16, dtype=np.float32),
        "model.layers.0.post_attention_layernorm.weight": np.ones(16, dtype=np.float32),
        "model.layers.0.self_attn.q_norm.weight": np.arange(4, dtype=np.float32),
        "model.layers.0.self_attn.k_norm.weight": np.arange(4, dtype=np.float32) + 1,
    }
    for name in ("q", "k", "v", "o"):
        tensors[f"model.layers.0.self_attn.{name}_proj.weight"] = np.ones((16, 16), dtype=np.float32)
    for name in ("gate", "up", "down"):
        shape = (16, 32) if name == "down" else (32, 16)
        tensors[f"model.layers.0.mlp.{name}_proj.weight"] = np.ones(shape, dtype=np.float32)
    save_file(tensors, tmp_path / "model.safetensors")
    weights = model._load_internvl_text_weights(str(tmp_path), config)
    np.testing.assert_array_equal(weights["layer.0.q_norm"], np.arange(4))
    np.testing.assert_array_equal(weights["layer.0.k_norm"], np.arange(4) + 1)


def test_tokenizer_contract_uses_instruction_end_of_turn(monkeypatch, tmp_path) -> None:
    import transformers

    tokenizer = SimpleNamespace(eos_token_id=151645, encode=lambda *_args, **_kwargs: [42])
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: tokenizer)
    contract = model._tokenizer_runtime_contract(tmp_path)
    assert contract["id_eos"] == 151645
    assert contract["tokenizer_prefix_ids"] == []
    assert contract["tokenizer_suffix_ids"] == []


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
    ranks, roles = [], []

    class FakeModel:
        @staticmethod
        def load_weights(_model_dir, _config, **_kwargs):
            return WeightDict()

        @staticmethod
        def build_engine(_config, _weights, _length, **kwargs):
            parallel = kwargs["parallel_config"]
            ranks.append(parallel.rank)
            roles.append(_config.raw.get("_decoder_engine_role"))
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
        mtp_seq_len=None,
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
    assert writer.sections["runtime.json"]["preprocessor_type"] == "simple_chw"
    assert writer.sections["runtime.json"]["max_image_tiles"] == 1
    assert writer.sections["runtime.json"]["decoder_engine_layout"] == "dual_profile"
    assert writer.sections["vision.plan"] == b"vision"

    ranks.clear()
    roles.clear()
    request.tensor_parallel_size = 1
    writer = Writer()
    model.build(request, writer)
    assert ranks == [-1]
    assert roles == ["dual_profile"]
    assert writer.sections["engine.plan"] == b"rank--1"
    assert "prefill.plan" not in writer.sections
    assert not any(name.startswith("engine.rank") for name in writer.sections)
    assert writer.sections["runtime.json"]["tensor_parallel_size"] == 1
    assert writer.sections["runtime.json"]["decoder_engine_layout"] == "dual_profile"
    assert "_decoder_engine_role" not in config.raw


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


@pytest.mark.parametrize("mode", ["absent", "success", "corrupt", "failure", "cancel", "device_failure", "capacity_failure", "larger_device", "quality_failure", "quality_failure_sm120", "awq_unmapped_qwen2", "awq_unmapped_qwen3", "native_capacity_failure", "native_larger_device"])
def test_edge_optional_package_and_output_local_staging(tmp_path, monkeypatch, caplog, mode):
    import json
    from tensorrt_model_connect.build import BuildRequest
    from families.internvl.edge_llm import builder, dispatch

    source = tmp_path / "target"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "max_position_embeddings": 4096, "hidden_size": 896,
    }))
    if mode in {"capacity_failure", "larger_device", "native_capacity_failure", "native_larger_device"}:
        (source / "config.json").write_text(json.dumps({
            "max_position_embeddings": 4096, "model_type": "qwen3",
            "num_hidden_layers": 40, "hidden_size": 5120,
        }))
    if mode == "quality_failure_sm120":
        (source / "config.json").write_text(json.dumps({
            "max_position_embeddings": 32768, "model_type": "qwen2",
            "num_hidden_layers": 28, "hidden_size": 1536,
        }))
    if mode.startswith("awq_unmapped"):
        (source / "config.json").write_text(json.dumps({
            "max_position_embeddings": 32768,
            "model_type": "qwen3" if mode.endswith("qwen3") else "qwen2",
            "num_hidden_layers": 32, "hidden_size": 4096,
        }))
    monkeypatch.setattr(dispatch, "device_memory_bytes", lambda: (
        40 if mode == "capacity_failure" else 64 if mode == "native_larger_device" else 48
    ) * 1024 ** 3)
    prefix = tmp_path / "package"
    manifest = prefix / "share/trtmc/edge-llm.json"
    if mode not in {"absent", "native_capacity_failure", "native_larger_device"}:
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
        monkeypatch.setattr(builder, "request_weight_format", lambda *_: "int4_awq" if mode.startswith(("quality_failure", "awq_unmapped")) else "fp16")
    request = BuildRequest(source, tmp_path / "out", "internvl", "vision_language_generation", "fp16",
                           max_sequence_length=384)
    writer = object()
    target = {"os": "linux", "arch": "x86_64", "sm": 80, "tensorrt_version": "11.1.0.106"}
    if mode == "quality_failure_sm120" or mode.startswith("awq_unmapped"):
        target["sm"] = 120
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

    monkeypatch.setattr(dispatch, "EDGE_DISPATCH", {("linux", "x86_64", target["sm"], fmt): prepare for fmt in ("fp16", "int4_awq")})
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
        "absent", "corrupt", "failure", "device_failure", "native_larger_device",
        "capacity_failure", "native_capacity_failure", "awq_unmapped_qwen2", "awq_unmapped_qwen3",
    } else [])
    assert len(publications) == (1 if mode in {"success", "larger_device", "quality_failure", "quality_failure_sm120"} else 0)
    assert len(target_calls) == (0 if mode in {
        "absent", "native_capacity_failure", "native_larger_device",
    } else 1)
    logs = list(tmp_path.glob(".out.edge-*.log"))
    if mode in {"corrupt", "failure", "device_failure", "capacity_failure",
                "awq_unmapped_qwen2", "awq_unmapped_qwen3"}:
        assert len(logs) == 1 and "Traceback" in logs[0].read_text()
        assert "Retrying native once" in caplog.text
    else:
        assert not logs and "Edge build failed" not in caplog.text
    if mode in {"capacity_failure", "native_capacity_failure"} or mode.startswith("awq_unmapped"):
        assert not stages  # Do not execute a known unavailable Edge configuration.


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
    from families.internvl.edge_llm.builder import int4_plugin_version, UnavailableEdgeConfiguration
    args = [str(source), "-o", str(output), "--int4-gemm-plugin-version", "1"]
    assert family_cli.main(["internvl", "build", *args]) == 0
    selected = captured[-1]
    assert selected.int4_gemm_plugin_version == 1
    with pytest.raises(ValueError, match="int4_gemm_plugin_version"):
        replace(selected, int4_gemm_plugin_version=True)
    config = {"model_type": "qwen2", "hidden_size": 1536, "num_hidden_layers": 28}
    assert int4_plugin_version(selected, config, {"sm": 80}, "int4_awq") == 1
    with pytest.raises(UnavailableEdgeConfiguration, match="SM80"):
        int4_plugin_version(selected, config, {"sm": 120}, "int4_awq")
    with pytest.raises(UnavailableEdgeConfiguration, match="AWQ"):
        int4_plugin_version(selected, config, {"sm": 80}, "fp16")
    assert int4_plugin_version(replace(selected, int4_gemm_plugin_version=None),
                               config, {"sm": 120}, "int4_awq") == 2


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
    assert "--int4-gemm-plugin-version" in result.stdout


@pytest.mark.parametrize("kind", ["safetensors", "pytorch"])
def test_checkpoint_layout_does_not_retry_a_failed_reader(tmp_path, monkeypatch, kind):
    filenames = ("model.safetensors", "pytorch_model.bin") if kind == "safetensors" else ("pytorch_model.bin",)
    for name in filenames:
        (tmp_path / name).touch()
    selected = "_open_safetensors" if kind == "safetensors" else "_open_torch_checkpoint"
    other = "_open_torch_checkpoint" if kind == "safetensors" else "_open_safetensors"
    def fail(_directory):
        raise ValueError("malformed selected checkpoint")
    monkeypatch.setattr(model, selected, fail)
    monkeypatch.setattr(model, other, lambda _directory: pytest.fail("must not retry another format"))
    with pytest.raises(ValueError, match="malformed selected checkpoint"):
        model._open_checkpoint(tmp_path)
