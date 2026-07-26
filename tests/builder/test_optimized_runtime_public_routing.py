# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression gates for transparent optimized-runtime routing."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def test_public_python_build_signature_is_unchanged() -> None:
    import tensorrt_model_connect as trtmc
    from tensorrt_model_connect.engine_builder import _build_native_impl

    assert inspect.signature(trtmc.build) == inspect.signature(_build_native_impl)
    assert trtmc.build.__defaults__ == _build_native_impl.__defaults__
    assert trtmc.build.__kwdefaults__ == _build_native_impl.__kwdefaults__
    assert trtmc.__all__ == [
        "__version__",
        "build",
        "build_bundle",
        "write_bundle",
        "ModelConfig",
        "Pipeline",
    ]


def test_internal_dispatch_resolves_exactly_one_model_family_before_discovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "model.trtfb"
    calls: list[tuple[str, str, dict]] = []
    selected = object()

    monkeypatch.setattr(engine_builder, "_resolve_model", lambda _model: str(model))
    monkeypatch.setattr(
        engine_builder.ModelConfig,
        "from_dir",
        lambda _model_dir: SimpleNamespace(model_type="example_family"),
    )
    monkeypatch.setattr(engine_builder, "resolve_family_id", lambda _config: "example_family")
    monkeypatch.setattr(
        engine_builder,
        "find_plugin",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("optimized dispatch imported the native family plugin")
        ),
    )

    def delegated(model_ref: str, output_path: str, **kwargs):
        calls.append((model_ref, output_path, kwargs))
        return selected

    monkeypatch.setattr(orchestrator, "try_build_optimized_runtime", delegated)

    result = engine_builder._try_build_optimized_runtime(
        "Example/Model",
        output,
        {"precision": "fp16"},
    )

    assert result is selected
    assert len(calls) == 1
    assert calls[0][0] == str(model)
    assert calls[0][1] == output
    assert calls[0][2]["family_name"] == "example_family"
    assert calls[0][2]["parameters"] == {
        "public_options": {"precision": "fp16"}
    }


def test_python_build_delegates_before_native_with_explicit_options(
    monkeypatch,
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder

    calls: list[tuple[str, str, dict]] = []

    def delegated(
        model: str,
        output: str,
        options: dict,
        *,
        explicit_public_options: frozenset[str],
    ):
        calls.append((model, output, options))
        assert explicit_public_options == {"precision", "max_cache_length"}
        return object()

    monkeypatch.setattr(engine_builder, "_try_build_optimized_runtime", delegated)
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("native build was selected")),
    )

    engine_builder.build(
        "example/model",
        "model.trtfb",
        precision="fp16",
        max_cache_length=4096,
    )

    assert len(calls) == 1
    model, output, options = calls[0]
    assert (model, output) == ("example/model", "model.trtfb")
    assert options["max_cache_length"] == 4096
    assert options["precision"] == "fp16"
    assert options["max_batch_size"] == 1
    assert options["decoder_engine_layout"] == "split"


def test_python_build_treats_omitted_and_explicit_defaults_identically(
    monkeypatch,
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder

    calls: list[dict] = []

    def delegated(
        _model: str,
        _output: str,
        options: dict,
        *,
        explicit_public_options: frozenset[str],
    ):
        calls.append(options)
        calls.append({"_explicit": explicit_public_options})
        return object()

    monkeypatch.setattr(engine_builder, "_try_build_optimized_runtime", delegated)
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("native build was selected")),
    )

    engine_builder.build("example/model", "omitted.trtfb")
    engine_builder.build(
        "example/model",
        "explicit.trtfb",
        256,
        precision="fp32",
        max_batch_size=1,
    )

    assert calls[0] == calls[2]
    assert calls[0]["max_batch_size"] == 1
    assert calls[0]["max_cache_length"] == 256
    assert calls[0]["precision"] == "fp32"
    assert calls[1]["_explicit"] == frozenset()
    assert calls[3]["_explicit"] == {"precision", "max_cache_length"}


def test_python_build_preserves_native_call_when_no_capsule_matches(monkeypatch) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder

    native_calls: list[dict] = []
    monkeypatch.setattr(
        engine_builder,
        "_try_build_optimized_runtime",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **kwargs: native_calls.append(kwargs),
    )

    engine_builder.build("native/model", "native.trtfb")

    assert len(native_calls) == 1
    assert native_calls[0]["model_id_or_path"] == "native/model"
    assert native_calls[0]["output_path"] == "native.trtfb"
    assert native_calls[0]["precision"] is None
    assert native_calls[0]["max_cache_length"] is None


def test_cli_delegation_uses_existing_options_without_native_discovery(monkeypatch) -> None:
    import tensorrt_model_connect.build_cli as build_cli
    import tensorrt_model_connect.engine_builder as engine_builder

    calls: list[tuple[str, str, dict]] = []

    def delegated(
        model: str,
        output: str,
        options: dict,
        *,
        explicit_public_options: frozenset[str],
    ):
        calls.append((model, output, options))
        assert explicit_public_options == {"precision", "max_cache_length"}
        return object()

    monkeypatch.setattr(engine_builder, "_try_build_optimized_runtime", delegated)
    monkeypatch.setattr(
        build_cli,
        "_auto_select_build_backend",
        lambda _model: (_ for _ in ()).throw(AssertionError("native discovery ran")),
    )
    args = argparse.Namespace(
        command="build",
        model="example/model",
        output="model.trtfb",
        precision="fp16",
        max_cache_length=256,
    )

    assert build_cli._cmd_build(args) == 0
    assert calls == [
        (
            "example/model",
            "model.trtfb",
            {"max_cache_length": 256, "precision": "fp16"},
        )
    ]


def test_cli_delegation_preserves_explicit_default_options(monkeypatch) -> None:
    import tensorrt_model_connect.build_cli as build_cli
    import tensorrt_model_connect.engine_builder as engine_builder

    calls: list[dict] = []

    def delegated(
        _model: str,
        _output: str,
        options: dict,
        *,
        explicit_public_options: frozenset[str],
    ):
        calls.append(options)
        assert explicit_public_options == {"precision", "max_cache_length"}
        return object()

    monkeypatch.setattr(engine_builder, "_try_build_optimized_runtime", delegated)
    args = argparse.Namespace(
        command="build",
        model="example/model",
        output="model.trtfb",
        precision="fp32",
        max_cache_length=256,
        max_batch_size=1,
    )

    assert build_cli._cmd_build(args) == 0
    assert calls == [
        {
            "max_batch_size": 1,
            "max_cache_length": 256,
            "precision": "fp32",
        }
    ]


def test_cli_delegation_preserves_optimized_precision_default() -> None:
    import tensorrt_model_connect.build_cli as build_cli

    args = argparse.Namespace(
        command="build",
        model="example/model",
        output="model.trtfb",
        precision=None,
        max_cache_length=256,
        max_batch_size=1,
    )

    options = build_cli._optimized_cli_public_options(args)

    assert options["precision"] == "fp32"
    assert args.precision is None


def test_cli_treats_model_revision_as_identity_not_plugin_option(monkeypatch) -> None:
    import tensorrt_model_connect.build_cli as build_cli
    import tensorrt_model_connect.engine_builder as engine_builder

    captured: dict[str, object] = {}

    def delegated(
        model: str,
        output: str,
        options: dict,
        *,
        model_revision: str,
        explicit_public_options: frozenset[str],
    ):
        captured.update(
            model=model,
            output=output,
            options=options,
            model_revision=model_revision,
            explicit_public_options=explicit_public_options,
        )
        return object()

    monkeypatch.setattr(engine_builder, "_try_build_optimized_runtime", delegated)
    args = argparse.Namespace(
        command="build",
        model="example/model",
        model_revision="0123456789abcdef0123456789abcdef01234567",
        output="model.trtfb",
        precision="fp32",
        max_cache_length=256,
    )

    assert build_cli._cmd_build(args) == 0
    assert captured["model_revision"] == args.model_revision
    assert captured["explicit_public_options"] == {
        "precision",
        "max_cache_length",
    }
    assert "model_revision" not in captured["options"]


def test_cli_native_fallback_does_not_probe_capsules_twice(monkeypatch) -> None:
    import tensorrt_model_connect.build_cli as build_cli
    import tensorrt_model_connect.engine_builder as engine_builder

    probe_calls: list[str] = []
    native_calls: list[dict] = []

    def no_delegation(
        model: str,
        _output: str,
        _options: dict,
        *,
        explicit_public_options: frozenset[str],
    ):
        probe_calls.append(model)
        assert explicit_public_options == {"precision", "max_cache_length"}
        return None

    monkeypatch.setattr(engine_builder, "_try_build_optimized_runtime", no_delegation)
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **kwargs: native_calls.append(kwargs),
    )
    monkeypatch.setattr(
        engine_builder,
        "build",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("public build wrapper would repeat capsule discovery")
        ),
    )
    args = argparse.Namespace(
        command="build",
        model="native/model",
        output="native.trtfb",
        max_cache_length=256,
        precision="fp32",
        method="trt",
        quantize=None,
        quant_scales=None,
        quant_calibration_samples=512,
        verbose=False,
        _skip_profile_resolution=True,
    )

    assert build_cli._cmd_build(args) == 0
    assert probe_calls == ["native/model"]
    assert len(native_calls) == 1
    assert native_calls[0]["model_id_or_path"] == "native/model"


def _write_qwen3_06b_config(path: Path, *, hidden_size: int = 1024) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": hidden_size,
                "intermediate_size": 3072,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "max_position_embeddings": 40960,
            }
        ),
        encoding="utf-8",
    )


def test_qwen3_06b_model_only_request_prefers_native_without_plugin_import(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    model = tmp_path / "qwen3-06b"
    _write_qwen3_06b_config(model)
    delegated: list[object] = []
    monkeypatch.setattr(engine_builder, "_resolve_model", lambda _model: str(model))
    monkeypatch.setattr(
        engine_builder,
        "find_plugin",
        lambda _config: (_ for _ in ()).throw(
            AssertionError("default routing imported the native Qwen plugin")
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "try_build_optimized_runtime",
        lambda *_args, **_kwargs: delegated.append(object()),
    )

    result = engine_builder._try_build_optimized_runtime(
        "Qwen/Qwen3-0.6B",
        tmp_path / "qwen.trtfb",
        {"precision": "fp32", "max_cache_length": 256},
        explicit_public_options=frozenset(),
    )

    assert result is None
    assert delegated == []


def test_qwen3_06b_explicit_deployment_option_preserves_optimized_probe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    model = tmp_path / "qwen3-06b"
    _write_qwen3_06b_config(model)
    selected = object()
    calls: list[dict] = []
    monkeypatch.setattr(engine_builder, "_resolve_model", lambda _model: str(model))

    def delegated(*_args, **kwargs):
        calls.append(kwargs)
        return selected

    monkeypatch.setattr(orchestrator, "try_build_optimized_runtime", delegated)
    result = engine_builder._try_build_optimized_runtime(
        "Qwen/Qwen3-0.6B",
        tmp_path / "qwen.trtfb",
        {"precision": "fp16", "max_cache_length": 4096},
        explicit_public_options=frozenset({"precision", "max_cache_length"}),
    )

    assert result is selected
    assert len(calls) == 1


def test_other_qwen3_model_only_request_keeps_existing_optimized_probe(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    model = tmp_path / "other-qwen3"
    _write_qwen3_06b_config(model, hidden_size=2048)
    selected = object()
    monkeypatch.setattr(engine_builder, "_resolve_model", lambda _model: str(model))
    monkeypatch.setattr(
        orchestrator,
        "try_build_optimized_runtime",
        lambda *_args, **_kwargs: selected,
    )

    result = engine_builder._try_build_optimized_runtime(
        "Qwen/Other-Qwen3",
        tmp_path / "qwen.trtfb",
        {"precision": "fp32", "max_cache_length": 256},
        explicit_public_options=frozenset(),
    )

    assert result is selected


def test_build_cli_does_not_expose_runtime_selection_or_target_flags() -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository / "python")
    result = subprocess.run(
        [sys.executable, "-m", "tensorrt_model_connect", "build", "--help"],
        cwd=repository,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--target" not in result.stdout
    assert "--runtime" not in result.stdout
    assert "optimized-runtime" not in result.stdout.lower()
    assert "--max-cache-length" not in result.stdout


def test_optimized_factory_header_is_private() -> None:
    repository = Path(__file__).resolve().parents[2]
    assert not (repository / "include" / "trtmc" / "runtime" / "runtime_provider_abi.h").exists()
    assert (repository / "src" / "runtime" / "providers" / "optimized_runtime_factory.h").is_file()


def test_optimized_runtime_python_package_has_no_public_exports() -> None:
    import tensorrt_model_connect.runtime_provider as implementation_package

    assert not hasattr(implementation_package, "__all__")
    assert not hasattr(implementation_package, "try_build_optimized_runtime")
