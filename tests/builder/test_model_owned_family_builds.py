# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

from tensorrt_model_connect import build_timing, bundle_writer, config, trt_compat


def _fake_trt() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        __version__="11.1.0",
        ElementWiseOperation=types.SimpleNamespace(SUM="sum", SUB="sub", PROD="prod"),
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        AttentionNormalizationOp=types.SimpleNamespace(SOFTMAX="softmax"),
        ActivationType=types.SimpleNamespace(SIGMOID="sigmoid", TANH="tanh"),
        ReduceOperation=types.SimpleNamespace(AVG="avg"),
        UnaryOperation=types.SimpleNamespace(SQRT="sqrt", RECIP="recip"),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
        BuilderFlag=types.SimpleNamespace(TF32="tf32", DISABLE_TIMING_CACHE="disable"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
        Permutation=lambda value: tuple(value),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        int32="int32",
    )


def _import_family(monkeypatch: pytest.MonkeyPatch, module_name: str):
    fake = _fake_trt()
    monkeypatch.setitem(sys.modules, "tensorrt", fake)
    monkeypatch.setattr(trt_compat, "_module", fake)
    for name in tuple(sys.modules):
        if name == module_name or name.startswith(module_name.rpartition(".")[0] + "."):
            sys.modules.pop(name, None)
    return importlib.import_module(module_name)


def _config(model_type: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_type=model_type,
        raw={"model_type": model_type},
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )


def _patch_common(
    monkeypatch: pytest.MonkeyPatch,
    model_config: object,
) -> list[tuple[object, list[object]]]:
    written: list[tuple[object, list[object]]] = []
    monkeypatch.setattr(config.ModelConfig, "from_dir", lambda _path: model_config)
    monkeypatch.setattr(build_timing, "new_build_timing", lambda _path: {})
    monkeypatch.setattr(build_timing, "add_build_timing", lambda *_args: None)
    monkeypatch.setattr(build_timing, "write_build_timing", lambda _timing: None)
    monkeypatch.setattr(
        bundle_writer,
        "write_bundle",
        lambda _path, info, sections: written.append((info, list(sections))),
    )
    return written


def test_bert_model_owned_build_writes_complete_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model = _import_family(
        monkeypatch,
        "tensorrt_model_connect.families.bert.model.model",
    )
    plugin_module = importlib.import_module("tensorrt_model_connect.families.bert.plugin")
    plugin = plugin_module.plugin
    cfg = _config("bert")
    written = _patch_common(monkeypatch, cfg)
    monkeypatch.setattr(model.ModelConfig, "from_dir", lambda _path: cfg)
    monkeypatch.setattr(plugin, "load_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(plugin, "build_engine", lambda *_args, **_kwargs: b"bert-plan")
    monkeypatch.setattr(model, "_detect_build_tokenizer_frame", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(model, "_ensure_build_tokenizer", lambda _path: None)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))

    model.build(str(tmp_path), str(tmp_path / "bert.bundle"), options={"precision": "fp16"})

    info, sections = written[0]
    by_name = {section.name: section.data for section in sections}
    assert info.family == "bert"
    assert info.precision == "fp16"
    assert by_name["engine_plan"] == b"bert-plan"
    runtime = json.loads(by_name["config.json"])
    assert runtime["runtime_strategy"] == "bert_encoder_only"
    assert runtime["precision"] == "fp16"
    assert not any(key.startswith("_") for key in runtime)


def test_timm_vit_model_owned_build_writes_preprocess_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model = _import_family(
        monkeypatch,
        "tensorrt_model_connect.families.timm_vit.model.model",
    )
    plugin_module = importlib.import_module("tensorrt_model_connect.families.timm_vit.plugin")
    plugin = plugin_module.plugin
    cfg = _config("vit_base_patch16_224")
    cfg.raw.update(
        {
            "input_size": [3, 224, 224],
            "patch_size": 16,
            "num_features": 8,
            "depth": 1,
            "num_heads": 2,
            "num_classes": 5,
        }
    )
    written = _patch_common(monkeypatch, cfg)
    timm_config = importlib.import_module(
        "tensorrt_model_connect.families.timm_vit.config"
    )
    monkeypatch.setattr(timm_config.ModelConfig, "from_dir", lambda _path: cfg)
    monkeypatch.setattr(plugin, "load_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(plugin, "build_engine", lambda *_args, **_kwargs: b"vit-plan")
    (tmp_path / "config.json").write_text(json.dumps(cfg.raw))

    model.build(str(tmp_path), str(tmp_path / "vit.bundle"), options={"precision": "fp16"})

    info, sections = written[0]
    by_name = {section.name: section.data for section in sections}
    assert info.family == "timm_vit"
    assert by_name["engine_plan"] == b"vit-plan"
    runtime = json.loads(by_name["config.json"])
    assert runtime["runtime_strategy"] == "timm_vit_image_classification"
    assert runtime["input_image_h"] == 224
    assert runtime["num_classes"] == 5
    assert not any(key.startswith("_") for key in runtime)


def test_gpt2_model_owned_build_preserves_split_decoder_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model = _import_family(
        monkeypatch,
        "tensorrt_model_connect.families.gpt2.model",
    )
    plugin_module = importlib.import_module("tensorrt_model_connect.families.gpt2.plugin")
    plugin = plugin_module.plugin
    cfg = _config("gpt2")
    written = _patch_common(monkeypatch, cfg)
    gpt2_config = importlib.import_module("tensorrt_model_connect.families.gpt2.config")
    monkeypatch.setattr(gpt2_config.ModelConfig, "from_dir", lambda _path: cfg)
    monkeypatch.setattr(plugin, "load_weights", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        plugin,
        "build_engine",
        lambda config_value, *_args, **_kwargs: str(
            config_value.raw["_decoder_engine_role"]
        ).encode("utf-8"),
    )
    monkeypatch.setattr(model, "_detect_build_tokenizer_frame", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(model, "_ensure_build_tokenizer", lambda _path: None)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2"}))

    model.build(
        str(tmp_path),
        str(tmp_path / "gpt2.bundle"),
        options={"precision": "fp16", "decoder_engine_layout": "split"},
    )

    info, sections = written[0]
    by_name = {section.name: section.data for section in sections}
    assert info.family == "gpt2"
    assert by_name["engine_plan"] == b"decode"
    assert by_name["prefill_engine_plan"] == b"prefill"
    runtime = json.loads(by_name["config.json"])
    assert runtime["runtime_strategy"] == "gpt2_decoder_kv_cache"
    assert runtime["decoder_engine_layout"] == "split"
    assert not any(key.startswith("_") for key in runtime)


@pytest.mark.parametrize(
    ("maximum", "budget"),
    ((16, 1), (256, 1), (4096, 512), (8192, 4096)),
)
def test_gpt2_model_owned_dynamic_kv_profiles_match_legacy(
    monkeypatch: pytest.MonkeyPatch,
    maximum: int,
    budget: int,
) -> None:
    model = _import_family(
        monkeypatch,
        "tensorrt_model_connect.families.gpt2.model",
    )
    from tensorrt_model_connect.engine_builder import _compute_dynamic_kv_profile_rows

    assert model._dynamic_kv_profile_rows(maximum, budget) == (
        _compute_dynamic_kv_profile_rows(maximum, budget)
    )
