# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import sys
import types
from contextlib import nullcontext

import pytest

from tensorrt_model_connect import build_timing, bundle_writer, trt_compat


FAMILIES = (
    "bloom",
    "codegen",
    "deepseek_v2",
    "falcon",
    "gemma",
    "glm",
    "gpt2",
    "gpt_neo",
    "gpt_neox",
    "gpt_oss",
    "granite",
    "internlm",
    "llama",
    "mistral",
    "mixtral",
    "nemotron",
    "olmo",
    "olmo2",
    "opt",
    "phi",
    "phi_moe",
    "qwen",
    "qwen_moe",
    "stablelm",
    "starcoder2",
    "xglm",
)


def _fake_trt() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        __version__="11.1.0",
        ElementWiseOperation=types.SimpleNamespace(
            SUM="sum", SUB="sub", PROD="prod", DIV="div", POW="pow"
        ),
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        AttentionNormalizationOp=types.SimpleNamespace(SOFTMAX="softmax"),
        ActivationType=types.SimpleNamespace(
            SIGMOID="sigmoid", TANH="tanh", RELU="relu"
        ),
        ReduceOperation=types.SimpleNamespace(AVG="avg", SUM="sum", MAX="max"),
        UnaryOperation=types.SimpleNamespace(
            SQRT="sqrt", RECIP="recip", EXP="exp", LOG="log"
        ),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(
            EXPLICIT_BATCH=0, STRONGLY_TYPED=1
        ),
        BuilderFlag=types.SimpleNamespace(
            TF32="tf32", DISABLE_TIMING_CACHE="disable"
        ),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
        TopKOperation=types.SimpleNamespace(MAX="max"),
        SliceMode=types.SimpleNamespace(WRAP="wrap"),
        Permutation=lambda value: tuple(value),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        int32="int32",
        int64="int64",
        bool="bool",
    )


@pytest.fixture(autouse=True)
def fake_trt_module(monkeypatch: pytest.MonkeyPatch):
    fake = _fake_trt()
    monkeypatch.setitem(sys.modules, "tensorrt", fake)
    monkeypatch.setattr(trt_compat, "_module", fake)


def _config(family: str) -> types.SimpleNamespace:
    model_type = {"qwen": "qwen2", "llama": "llama2"}.get(family, family)
    return types.SimpleNamespace(
        model_type=model_type,
        raw={"model_type": model_type, "architectures": []},
        architectures=[],
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


def _patch_build_leaves(monkeypatch: pytest.MonkeyPatch, model, config):
    written = []
    roles = []
    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _path: config)
    monkeypatch.setattr(model, "load_weights", lambda *_args, **_kwargs: {})

    def build_engine(config_arg, *_args, **_kwargs):
        roles.append(config_arg.raw.get("_decoder_engine_role"))
        return str(roles[-1]).encode("utf-8")

    monkeypatch.setattr(model, "build_engine", build_engine)
    if model.name == "qwen":
        monkeypatch.setattr(model, "_try_optimized_runtime", lambda *_args: False)
    if model.name == "starcoder2":
        monkeypatch.setattr(
            model,
            "tokenizer_json_bundle_override",
            lambda _path: b"{}",
        )
    monkeypatch.setattr(trt_compat, "scoped_timing_cache", lambda _scope: nullcontext())
    monkeypatch.setattr(build_timing, "new_build_timing", lambda _path: {})
    monkeypatch.setattr(build_timing, "add_build_timing", lambda *_args: None)
    monkeypatch.setattr(build_timing, "write_build_timing", lambda *_args: None)
    monkeypatch.setattr(bundle_writer, "tensorrt_version", lambda: "11.1.0")
    monkeypatch.setattr(bundle_writer, "tensorrt_abi", lambda _version: "11.1")
    monkeypatch.setattr(bundle_writer, "gpu_name", lambda: "test-gpu")
    monkeypatch.setattr(
        bundle_writer,
        "write_bundle",
        lambda path, info, sections: written.append((path, info, list(sections))),
    )
    tokenizer = importlib.import_module("tensorrt_model_connect.tokenizer_conversion")
    monkeypatch.setattr(
        tokenizer,
        "prepare_tokenizer_special_frame",
        lambda *_args, **_kwargs: ([], []),
    )
    return written, roles


@pytest.mark.parametrize("family", FAMILIES)
def test_decoder_family_owns_complete_bundle_build(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model = importlib.import_module(
        f"tensorrt_model_connect.models.{family}.model"
    )
    config = _config(family)
    written, roles = _patch_build_leaves(monkeypatch, model, config)
    (tmp_path / "config.json").write_text(json.dumps(config.raw))

    model.build(str(tmp_path), str(tmp_path / f"{family}.bundle"))

    assert len(written) == 1
    _, info, sections = written[0]
    section_map = {section.name: section.data for section in sections}
    runtime = json.loads(section_map["config.json"])
    assert info.family == family
    assert runtime["runtime_strategy"] == model.runtime_strategy
    assert roles
    if model.supports_split_decoder_roles(config):
        assert roles == ["prefill", "decode"]
        assert "prefill_engine_plan" in section_map
        assert runtime["decoder_engine_layout"] == "split"
    else:
        assert roles == ["decode"]
        assert "prefill_engine_plan" not in section_map
        assert runtime["decoder_engine_layout"] == "single"


def test_qwen_owns_optimized_runtime_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model = importlib.import_module("tensorrt_model_connect.models.qwen.model")
    config = _config("qwen")
    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _path: config)
    calls = []
    monkeypatch.setattr(
        model,
        "_try_optimized_runtime",
        lambda model_dir, output_path, options: calls.append(
            (model_dir, output_path, options)
        )
        or True,
    )
    monkeypatch.setattr(
        model,
        "load_weights",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("native Qwen build ran after optimized selection")
        ),
    )

    model.build(str(tmp_path), str(tmp_path / "qwen.bundle"), precision="fp16")

    assert len(calls) == 1
    assert calls[0][2]["precision"] == "fp16"


def test_gpt2_dynamic_kv_uses_single_engine_and_serializes_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model = importlib.import_module("tensorrt_model_connect.models.gpt2.model")
    config = _config("gpt2")
    written, roles = _patch_build_leaves(monkeypatch, model, config)
    (tmp_path / "config.json").write_text(json.dumps(config.raw))

    model.build(
        str(tmp_path),
        str(tmp_path / "gpt2.bundle"),
        max_cache_length=64,
        dynamic_kv_cache=True,
        dynamic_kv_profile_rows_override=[16, 64],
    )

    runtime = json.loads(
        next(
            section.data
            for section in written[0][2]
            if section.name == "config.json"
        )
    )
    assert roles == ["decode"]
    assert runtime["decoder_engine_layout"] == "single"
    assert runtime["dynamic_kv_profile_rows"] == [16, 64]


@pytest.mark.parametrize(
    ("model_eos", "generation_eos"),
    [(5, [5, 7]), ([5, 7], 3)],
)
def test_gpt2_generation_config_owns_effective_eos(
    model_eos,
    generation_eos,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model = importlib.import_module("tensorrt_model_connect.models.gpt2.model")
    config = _config("gpt2")
    config.raw["eos_token_id"] = model_eos
    written, _roles = _patch_build_leaves(monkeypatch, model, config)
    (tmp_path / "config.json").write_text(json.dumps(config.raw))
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": generation_eos})
    )

    model.build(str(tmp_path), str(tmp_path / "gpt2.bundle"))

    runtime = json.loads(
        next(
            section.data
            for section in written[0][2]
            if section.name == "config.json"
        )
    )
    assert runtime["eos_token_id"] == generation_eos


def test_decoder_packages_do_not_eagerly_import_models() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "python/tensorrt_model_connect/models"
    for family in ("bert", *FAMILIES):
        source = (root / family / "__init__.py").read_text(encoding="utf-8")
        assert "import model" not in source
        assert "plugin" not in source
