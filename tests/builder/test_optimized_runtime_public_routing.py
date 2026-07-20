# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression gates for transparent optimized-runtime routing."""

from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


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

    def delegated(model: str, output: str, options: dict):
        calls.append((model, output, options))
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

    def delegated(_model: str, _output: str, options: dict):
        calls.append(options)
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

    assert calls[0] == calls[1]
    assert calls[0]["max_batch_size"] == 1
    assert calls[0]["max_cache_length"] == 256
    assert calls[0]["precision"] == "fp32"


def test_python_build_preflights_family_dependencies_before_resolution(
    monkeypatch,
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    monkeypatch.setattr(
        engine_builder,
        "_public_build_family_hint",
        lambda _model: "wan2_2_ti2v",
    )

    def missing_dependency(family: str) -> None:
        assert family == "wan2_2_ti2v"
        raise RuntimeError("missing family build dependency")

    monkeypatch.setattr(
        engine_builder,
        "preflight_family_build_dependencies",
        missing_dependency,
    )
    monkeypatch.setattr(
        orchestrator,
        "discover_family_implementations_for_model",
        lambda _family, _model: (),
    )
    monkeypatch.setattr(
        engine_builder,
        "_try_build_optimized_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("checkpoint resolution must not start")
        ),
    )

    with pytest.raises(RuntimeError, match="missing family build dependency"):
        engine_builder.build(
            "Wan-AI/Wan2.2-TI2V-5B",
            "wan.trtfb",
        )


def test_python_build_capsule_does_not_require_native_family_dependencies(
    monkeypatch,
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    monkeypatch.setattr(
        engine_builder,
        "_public_build_family_hint",
        lambda _model: "wan2_2_ti2v",
    )
    monkeypatch.setattr(
        orchestrator,
        "discover_family_implementations_for_model",
        lambda _family, _model: (object(),),
    )
    monkeypatch.setattr(
        engine_builder,
        "preflight_family_build_dependencies",
        lambda _family: (_ for _ in ()).throw(
            AssertionError("optimized capsule must not require native dependencies")
        ),
    )
    monkeypatch.setattr(
        engine_builder,
        "_try_build_optimized_runtime",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("native build was selected")
        ),
    )

    engine_builder.build(
        "Wan-AI/Wan2.2-TI2V-5B",
        "wan.trtfb",
    )


def test_python_build_preflights_before_native_after_capsule_declines(
    monkeypatch,
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    monkeypatch.setattr(
        engine_builder,
        "_public_build_family_hint",
        lambda _model: "wan2_2_ti2v",
    )
    monkeypatch.setattr(
        orchestrator,
        "discover_family_implementations_for_model",
        lambda _family, _model: (object(),),
    )
    monkeypatch.setattr(
        engine_builder,
        "_try_build_optimized_runtime",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        engine_builder,
        "preflight_family_build_dependencies",
        lambda _family: (_ for _ in ()).throw(
            RuntimeError("missing native family dependency")
        ),
    )
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("native build must not start without its dependency")
        ),
    )

    with pytest.raises(RuntimeError, match="missing native family dependency"):
        engine_builder.build(
            "Wan-AI/Wan2.2-TI2V-5B",
            "wan.trtfb",
        )


def test_python_build_preserves_local_snapshot_capsule_routing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.engine_builder as engine_builder
    import tensorrt_model_connect.runtime_provider.orchestrator as orchestrator

    snapshot = tmp_path / "hub" / "models--Example--Model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        '{"model_type":"example_family"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        engine_builder,
        "_public_build_family_hint",
        lambda _model: "example_family",
    )
    monkeypatch.setattr(
        orchestrator,
        "discover_family_implementations_for_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("local snapshot identity must be resolved by optimized routing")
        ),
    )
    monkeypatch.setattr(
        engine_builder,
        "preflight_family_build_dependencies",
        lambda _family: (_ for _ in ()).throw(
            AssertionError("successful capsule must not require native dependencies")
        ),
    )
    monkeypatch.setattr(
        engine_builder,
        "_try_build_optimized_runtime",
        lambda model, _output, _options: object()
        if model == str(snapshot)
        else (_ for _ in ()).throw(AssertionError("unexpected model reference")),
    )
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("native build was selected")
        ),
    )

    engine_builder.build(str(snapshot), "model.trtfb")


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
    assert native_calls[0]["precision"] == "fp32"
    assert native_calls[0]["max_cache_length"] == 256


def test_cli_delegation_uses_existing_options_without_native_discovery(monkeypatch) -> None:
    import tensorrt_model_connect.build_cli as build_cli
    import tensorrt_model_connect.engine_builder as engine_builder

    calls: list[tuple[str, str, dict]] = []

    def delegated(model: str, output: str, options: dict):
        calls.append((model, output, options))
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

    def delegated(_model: str, _output: str, options: dict):
        calls.append(options)
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


def test_cli_delegation_treats_hidden_method_aliases_as_no_ops(monkeypatch) -> None:
    import tensorrt_model_connect.build_cli as build_cli
    import tensorrt_model_connect.engine_builder as engine_builder

    calls: list[dict] = []

    def delegated(_model: str, _output: str, options: dict):
        calls.append(options)
        return object()

    monkeypatch.setattr(engine_builder, "_try_build_optimized_runtime", delegated)

    for method in ("trt", "auto"):
        args = argparse.Namespace(
            command="build",
            model="example/model",
            output=f"{method}.trtfb",
            precision="fp32",
            max_cache_length=256,
            method=method,
        )
        assert build_cli._cmd_build(args) == 0

    assert calls == [
        {"max_cache_length": 256, "precision": "fp32"},
        {"max_cache_length": 256, "precision": "fp32"},
    ]


def test_cli_native_fallback_does_not_probe_capsules_twice(monkeypatch) -> None:
    import tensorrt_model_connect.build_cli as build_cli
    import tensorrt_model_connect.engine_builder as engine_builder

    probe_calls: list[str] = []
    native_calls: list[dict] = []

    def no_delegation(model: str, _output: str, _options: dict):
        probe_calls.append(model)
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
    assert "--max-cache-length" in result.stdout


def test_optimized_factory_header_is_private() -> None:
    repository = Path(__file__).resolve().parents[2]
    assert not (repository / "include" / "trtmc" / "runtime" / "runtime_provider_abi.h").exists()
    assert (repository / "src" / "runtime" / "providers" / "optimized_runtime_factory.h").is_file()


def test_optimized_runtime_python_package_has_no_public_exports() -> None:
    import tensorrt_model_connect.runtime_provider as implementation_package

    assert not hasattr(implementation_package, "__all__")
    assert not hasattr(implementation_package, "try_build_optimized_runtime")
