# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

from tensorrt_model_connect import build_timing, bundle_writer, trt_compat


def _fake_trt() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        __version__="11.1.0",
        ElementWiseOperation=types.SimpleNamespace(SUM="sum", SUB="sub", PROD="prod"),
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        AttentionNormalizationOp=types.SimpleNamespace(SOFTMAX="softmax"),
        ActivationType=types.SimpleNamespace(SIGMOID="sigmoid", TANH="tanh", RELU="relu"),
        ReduceOperation=types.SimpleNamespace(AVG="avg"),
        UnaryOperation=types.SimpleNamespace(SQRT="sqrt", RECIP="recip"),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
        BuilderFlag=types.SimpleNamespace(TF32="tf32"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
        Permutation=lambda value: tuple(value),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        int32="int32",
    )


@pytest.fixture
def bert_model(monkeypatch: pytest.MonkeyPatch):
    fake_trt = _fake_trt()
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setattr(trt_compat, "_module", fake_trt)
    for module_name in tuple(sys.modules):
        if module_name == "tensorrt_model_connect.families.bert" or module_name.startswith(
            "tensorrt_model_connect.families.bert."
        ):
            sys.modules.pop(module_name, None)
    return importlib.import_module("tensorrt_model_connect.families.bert.model")


def _config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_type="bert",
        raw={"model_type": "bert"},
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )


def test_build_owns_checkpoint_to_bundle_pipeline(
    bert_model,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    written = []
    config = _config()
    monkeypatch.setattr(bert_model.ModelConfig, "from_dir", lambda _path: config)
    monkeypatch.setattr(bert_model, "load_weights", lambda *_args: {"owned": True})
    monkeypatch.setattr(
        bert_model,
        "build_engine",
        lambda *_args, **_kwargs: b"bert-plan",
    )
    monkeypatch.setattr(bert_model, "_detect_tokenizer_frame", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(bert_model, "_ensure_tokenizer_json", lambda _path: None)
    monkeypatch.setattr(build_timing, "new_build_timing", lambda _path: {})
    monkeypatch.setattr(build_timing, "add_build_timing", lambda *_args: None)
    monkeypatch.setattr(build_timing, "write_build_timing", lambda _timing: None)
    monkeypatch.setattr(
        bundle_writer,
        "write_bundle",
        lambda path, info, sections: written.append((path, info, list(sections))),
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))

    output_path = tmp_path / "bert.bundle"
    bert_model.build(str(tmp_path), str(output_path), precision="fp16")

    path, info, sections = written[0]
    by_name = {section.name: section.data for section in sections}
    assert path == str(output_path)
    assert info.family == "bert"
    assert info.runtime_strategy == "bert_encoder_only"
    assert info.precision == "fp16"
    assert by_name["engine_plan"] == b"bert-plan"
    runtime_config = json.loads(by_name["config.json"])
    assert runtime_config["runtime_strategy"] == "bert_encoder_only"
    assert runtime_config["precision"] == "fp16"
    assert not any(key.startswith("_") for key in runtime_config)


def test_build_rejects_explicit_zero_cache_length(
    bert_model,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(bert_model.ModelConfig, "from_dir", lambda _path: _config())

    with pytest.raises(ValueError, match="max_cache_length must be >= 1"):
        bert_model.build(str(tmp_path), str(tmp_path / "bert.bundle"), max_cache_length=0)


def test_bert_has_no_plugin_compatibility_surface(bert_model) -> None:
    family_dir = __import__("pathlib").Path(bert_model.__file__).parent

    assert not hasattr(bert_model, "plugin")
    assert not (family_dir / "plugin.py").exists()
    assert not (family_dir / "model").exists()
