# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import inspect
import json
import sys
import types
from contextlib import contextmanager

import pytest

from tensorrt_model_connect import engine_builder, families, trt_compat


def _fake_trt() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        ElementWiseOperation=types.SimpleNamespace(
            SUM="sum", SUB="sub", PROD="prod", DIV="div"
        ),
        MatrixOperation=types.SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        AttentionNormalizationOp=types.SimpleNamespace(SOFTMAX="softmax"),
        ActivationType=types.SimpleNamespace(SIGMOID="sigmoid", TANH="tanh", RELU="relu"),
        ReduceOperation=types.SimpleNamespace(AVG="avg", SUM="sum"),
        UnaryOperation=types.SimpleNamespace(SQRT="sqrt", RECIP="recip", EXP="exp"),
        NetworkDefinitionCreationFlag=types.SimpleNamespace(EXPLICIT_BATCH=0, STRONGLY_TYPED=1),
        BuilderFlag=types.SimpleNamespace(TF32="tf32"),
        MemoryPoolType=types.SimpleNamespace(WORKSPACE="workspace"),
        Permutation=lambda value: tuple(value),
        float32="float32",
        float16="float16",
        bfloat16="bfloat16",
        int32="int32",
    )


def test_build_sets_up_backend_before_family_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    events = []
    family_model = types.SimpleNamespace(
        build=lambda model_dir, output_path, **options: events.append(
            ("build", model_dir, output_path, options)
        )
    )

    monkeypatch.setattr(
        engine_builder,
        "_setup_trt_import",
        lambda rtx: events.append(("setup", rtx)),
    )
    monkeypatch.setattr(
        engine_builder,
        "_resolve_model",
        lambda model, *, revision=None: model,
    )

    def resolve(model_dir):
        assert events == [("setup", True)]
        return "bert", family_model

    monkeypatch.setattr(
        engine_builder,
        "_resolve_family_model_from_model_dir",
        resolve,
    )

    engine_builder.build(
        str(tmp_path),
        str(tmp_path / "out.bundle"),
        rtx=True,
        fp8_scales={"layer": {"scale": 1.0}},
        save_fp8_scales="scales.json",
    )

    assert [event[0] for event in events] == ["setup", "build"]
    options = events[1][3]
    assert options["fp8_scales"] == {"layer": {"scale": 1.0}}
    assert options["save_fp8_scales"] == "scales.json"


def test_public_build_resolves_once_and_dispatches_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        engine_builder,
        "_resolve_model",
        lambda model, *, revision=None: calls.append(("resolve", model, revision))
        or "/models/bert",
    )
    monkeypatch.setattr(
        engine_builder,
        "_dispatch_model_build",
        lambda model_dir, output_path, options: calls.append(
            ("build", model_dir, output_path, options)
        ),
    )

    engine_builder.build(
        "org/bert",
        "bert.bundle",
        model_revision="revision-1",
        fp8_scales="auto",
    )

    assert [call[0] for call in calls] == ["resolve", "build"]
    assert calls[1][3]["max_cache_length"] is None
    assert calls[1][3]["fp8_scales"] == "auto"
    assert calls[1][3]["tokenizer_source_model_id_or_path"] == "org/bert"
    assert calls[1][3]["tokenizer_source_revision"] == "revision-1"


def test_real_bert_candidate_is_loaded_and_called(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_trt = _fake_trt()
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setattr(trt_compat, "_module", fake_trt)
    for module_name in tuple(sys.modules):
        if module_name == "tensorrt_model_connect.families.bert" or module_name.startswith(
            "tensorrt_model_connect.families.bert."
        ):
            sys.modules.pop(module_name, None)
    bert_model = importlib.import_module("tensorrt_model_connect.families.bert.model")
    calls = []
    monkeypatch.setattr(
        bert_model,
        "build",
        lambda model_dir, output_path, **options: calls.append(
            (model_dir, output_path, options)
        ),
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "bert", "architectures": ["BertModel"]})
    )

    engine_builder.build(
        str(tmp_path),
        str(tmp_path / "bert.bundle"),
        precision="fp16",
    )

    assert len(calls) == 1
    assert calls[0][2]["precision"] == "fp16"


def test_candidate_model_must_define_required_matcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        families.importlib,
        "import_module",
        lambda _module: types.SimpleNamespace(build=lambda **_kwargs: None),
    )

    with pytest.raises(TypeError, match=r"must define matches\(\)"):
        families.load_model_by_id("bert")


def test_config_candidate_precedes_generic_model_type_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "t5",
                "architectures": ["ChronosBoltModelForForecasting"],
                "chronos_config": {"context_length": 16},
            }
        )
    )
    monkeypatch.setattr(
        families,
        "_candidate_module_names_from_config",
        lambda _config: ["chronos_bolt"],
    )
    monkeypatch.setattr(
        families,
        "_candidate_module_names",
        lambda _model_type: ["t5"],
    )
    models = {
        "chronos_bolt": types.SimpleNamespace(
            build=lambda *_args, **_kwargs: None,
            matches=lambda config: "chronos_config" in config.raw,
        ),
        "t5": types.SimpleNamespace(
            build=lambda *_args, **_kwargs: None,
            matches=lambda config: config.model_type == "t5",
        ),
    }
    monkeypatch.setattr(
        families,
        "load_model_by_id",
        lambda family: models[family],
    )

    config = engine_builder.ModelConfig.from_dir(tmp_path)
    model = families.find_model(config)

    assert model is models["chronos_bolt"]


def test_chronos_config_resolves_before_generic_t5(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_trt = _fake_trt()
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setattr(trt_compat, "_module", fake_trt)
    for module_name in tuple(sys.modules):
        if module_name == "tensorrt_model_connect.families.chronos_bolt" or module_name.startswith(
            "tensorrt_model_connect.families.chronos_bolt."
        ):
            sys.modules.pop(module_name, None)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "t5",
                "architectures": ["ChronosBoltModelForForecasting"],
                "chronos_config": {"context_length": 16},
                "vocab_size": 32,
                "d_model": 8,
                "num_layers": 1,
                "num_heads": 2,
            }
        )
    )

    family, model = engine_builder._resolve_family_model_from_model_dir(tmp_path)

    assert family == "chronos_bolt"
    assert model.__name__.endswith(".chronos_bolt.model")


def test_engine_builder_contains_no_legacy_route() -> None:
    source = inspect.getsource(engine_builder)

    for forbidden in (
        "_build_native_impl",
        "_try_build_optimized_runtime",
        "find_plugin",
        "plugin.build",
        "getattr_static",
    ):
        assert forbidden not in source


def test_cli_keeps_graph_context_around_its_single_public_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from tensorrt_model_connect import build_cli
    from tensorrt_model_connect.tvm_ffi import graph_build

    events = []

    @contextmanager
    def graph_context(*_args, **_kwargs):
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(graph_build, "inspect_graph", graph_context)
    monkeypatch.setattr(
        engine_builder,
        "build",
        lambda **_options: events.append("build"),
    )
    args = types.SimpleNamespace(
        model="org/bert",
        output=str(tmp_path / "bert.bundle"),
        max_cache_length=32,
        precision="fp32",
        quantize=None,
        quant_scales=None,
        quant_calibration_samples=512,
        verbose=False,
        graph_snapshot=str(tmp_path / "graph.json"),
        graph_patch=None,
        graph_role="decode",
        recipe=None,
        tensor_parallel_size=1,
        context_parallel_size=1,
        _skip_profile_resolution=True,
    )

    assert build_cli._cmd_build(args) == 0
    assert events == ["enter", "build", "exit"]
