# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for the registered SAM2 native build adapter."""

from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import tensorrt_model_connect.engine_builder as engine_builder
from tensorrt_model_connect.config import ModelConfig
from tensorrt_model_connect.families.base import CompleteBundleBuildRequest
from tensorrt_model_connect.families.sam2 import model_config
from tensorrt_model_connect.families.sam2 import plugin as registered_plugin
from tensorrt_model_connect.families.sam2 import archive_contract
from tensorrt_model_connect.families.sam2.native_builder import Sam2NativeBuilderError
from tensorrt_model_connect.families.sam2.plugin import Sam2Plugin
from tensorrt_model_connect.families.sam2.plugin import _validate_request
from tensorrt_model_connect.parallel_config import ParallelConfig


# Import the module separately from the package-level singleton.
plugin_module = importlib.import_module("tensorrt_model_connect.families.sam2.plugin")


def _request(tmp_path: Path, **updates) -> CompleteBundleBuildRequest:
    request = CompleteBundleBuildRequest(
        model_dir=tmp_path,
        output_path=tmp_path / "sam2.bundle",
        config=ModelConfig.create_tiny("sam2"),
        max_cache_length=None,
        decoder_engine_layout="split",
        dynamic_kv_cache=False,
        dynamic_kv_profile_rows_override=None,
        precision=None,
        fp32_layers=(),
        quantize=None,
        quant_scales=None,
        quant_calibration_samples=512,
        verbose=False,
        kernel_artifacts=(),
        rtx=False,
        fp8_scales=None,
        save_fp8_scales=None,
        triattention_stats_path=None,
        triattention_kv_budget=None,
        triattention_divide_length=128,
        triattention_recent_window=128,
        triattention_score_aggregation="mean",
        triattention_count_prompt_tokens=True,
        triattention_protect_prefill=True,
        triattention_disable_mlr=False,
        triattention_disable_trig=False,
        family_build_options={},
        parallel_config=ParallelConfig(),
        diffusion_overrides={},
        build_timing_path=None,
        max_batch_size=1,
        source_model_id_or_path=None,
        source_revision=None,
    )
    return replace(request, **updates)


def _write_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _materialize_archive_paths(root: Path) -> None:
    checkpoint = root / archive_contract.CHECKPOINT_RELATIVE_PATH
    config = root / archive_contract.CONFIG_RELATIVE_PATH
    checkpoint.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config.write_bytes(b"config")


def _authenticated_description(root: Path) -> SimpleNamespace:
    return SimpleNamespace(root=root)


def test_package_exports_one_registered_plugin_singleton() -> None:
    assert registered_plugin is plugin_module.plugin
    assert isinstance(registered_plugin, Sam2Plugin)


@pytest.mark.parametrize("precision", [None, "bf16", "BF16", "mixed_bf16_fp32"])
def test_only_native_mixed_precision_semantics_are_accepted(
    tmp_path: Path,
    precision: str | None,
) -> None:
    assert _validate_request(_request(tmp_path, precision=precision)) == {}


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("precision", "fp32", "precision"),
        ("max_cache_length", 1, "max_cache_length"),
        ("decoder_engine_layout", "dual_profile", "decoder_engine_layout"),
        ("dynamic_kv_cache", True, "dynamic_kv_cache"),
        ("dynamic_kv_profile_rows_override", (32,), "dynamic_kv_profile"),
        ("fp32_layers", (1,), "fp32_layers"),
        ("quantize", "int8", "quantize"),
        ("quant_scales", "scales.json", "quant_scales"),
        ("quant_calibration_samples", 1, "quant_calibration_samples"),
        ("verbose", True, "verbose"),
        ("kernel_artifacts", (("kernel", "artifact"),), "kernel_artifacts"),
        ("rtx", True, "rtx"),
        ("fp8_scales", {"layer": 1.0}, "fp8_scales"),
        ("save_fp8_scales", "scales.json", "save_fp8_scales"),
        ("triattention_stats_path", "stats.json", "triattention_stats_path"),
        ("triattention_kv_budget", 64, "triattention_kv_budget"),
        ("triattention_divide_length", 64, "triattention_divide_length"),
        ("triattention_recent_window", 64, "triattention_recent_window"),
        ("triattention_score_aggregation", "max", "triattention_score"),
        ("triattention_count_prompt_tokens", False, "triattention_count"),
        ("triattention_protect_prefill", False, "triattention_protect"),
        ("triattention_disable_mlr", True, "triattention_disable_mlr"),
        ("triattention_disable_trig", True, "triattention_disable_trig"),
        ("diffusion_overrides", {"image_height": 1024}, "diffusion_overrides"),
        ("build_timing_path", "timing.json", "build_timing_path"),
        ("max_batch_size", 2, "max_batch_size"),
        ("source_revision", "main", "source_revision"),
        ("source_model_id_or_path", "org/tokenizer", "tokenizer source"),
    ],
)
def test_nondefault_generic_build_options_are_rejected(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _validate_request(_request(tmp_path, **{field: value}))


def test_parallel_and_foreign_family_options_are_rejected(tmp_path: Path) -> None:
    parallel = ParallelConfig(mode="tensor_parallel", tp_size=2)
    with pytest.raises(ValueError, match="parallel"):
        _validate_request(_request(tmp_path, parallel_config=parallel))
    with pytest.raises(ValueError, match="unrelated family_build_options"):
        _validate_request(
            _request(tmp_path, family_build_options={"other": {"workspace_bytes": 1}})
        )
    with pytest.raises(ValueError, match="Unsupported SAM2"):
        _validate_request(_request(tmp_path, family_build_options={"sam2": {"unexpected": 1}}))


def test_original_local_model_path_is_an_accepted_tokenizer_noop(tmp_path: Path) -> None:
    assert _validate_request(_request(tmp_path, source_model_id_or_path=str(tmp_path))) == {}


def test_exact_builder_argv_and_sanitized_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _materialize_archive_paths(tmp_path)
    builder = _write_executable(tmp_path / "sam2_native_builder")
    request = _request(
        tmp_path,
        precision="bf16",
        family_build_options={
            "sam2": {
                "workspace_bytes": 8 << 30,
                "gpu_device": 2,
                "created_at": "2026-08-16T17:30:00Z",
            }
        },
    )
    observed: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="built\n", stderr="")

    monkeypatch.setattr(
        plugin_module,
        "require_reference_archive",
        lambda _path: _authenticated_description(tmp_path),
    )
    monkeypatch.setattr(plugin_module, "locate_native_builder", lambda: builder)
    monkeypatch.setattr(plugin_module.subprocess, "run", fake_run)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/tensorrt/lib:/usr/local/cuda/lib64")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/forbidden.so")
    monkeypatch.setenv("PYTHONPATH", "/tmp/forbidden-python")

    Sam2Plugin().build_complete_bundle(request)

    assert observed["argv"] == [
        str(builder.resolve()),
        "--checkpoint",
        str((tmp_path / archive_contract.CHECKPOINT_RELATIVE_PATH).resolve()),
        "--config",
        str((tmp_path / archive_contract.CONFIG_RELATIVE_PATH).resolve()),
        "--output",
        str(request.output_path.absolute()),
        "--workspace-bytes",
        str(8 << 30),
        "--gpu-device",
        "2",
        "--created-at",
        "2026-08-16T17:30:00Z",
    ]
    assert observed["kwargs"] == {
        "capture_output": True,
        "check": False,
        "env": {
            "CUDA_VISIBLE_DEVICES": "3",
            "LANG": "C",
            "LC_ALL": "C",
            "LD_LIBRARY_PATH": "/opt/tensorrt/lib:/usr/local/cuda/lib64",
            "PATH": "/usr/bin:/bin",
        },
        "text": True,
    }


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"workspace_bytes": 0}, "workspace_bytes"),
        ({"workspace_bytes": True}, "workspace_bytes"),
        ({"gpu_device": -1}, "gpu_device"),
        ({"gpu_device": 1 << 31}, "gpu_device"),
        ({"created_at": "2026-99-16T17:30:00Z"}, "created_at"),
    ],
)
def test_family_builder_option_values_fail_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    options: dict[str, object],
    error: str,
) -> None:
    _materialize_archive_paths(tmp_path)
    builder = _write_executable(tmp_path / "sam2_native_builder")
    monkeypatch.setattr(
        plugin_module,
        "require_reference_archive",
        lambda _path: _authenticated_description(tmp_path),
    )
    monkeypatch.setattr(plugin_module, "locate_native_builder", lambda: builder)
    monkeypatch.setattr(
        plugin_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run"),
    )

    with pytest.raises(ValueError, match=error):
        Sam2Plugin().build_complete_bundle(
            _request(tmp_path, family_build_options={"sam2": options})
        )


def test_subprocess_failure_surfaces_exact_stderr_and_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _materialize_archive_paths(tmp_path)
    builder = _write_executable(tmp_path / "sam2_native_builder")
    monkeypatch.setattr(
        plugin_module,
        "require_reference_archive",
        lambda _path: _authenticated_description(tmp_path),
    )
    monkeypatch.setattr(plugin_module, "locate_native_builder", lambda: builder)
    monkeypatch.setattr(
        plugin_module.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(
            argv,
            23,
            stdout="builder stdout\n",
            stderr="exact native failure: attention build rejected\n",
        ),
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/trt/lib")

    with pytest.raises(Sam2NativeBuilderError) as caught:
        Sam2Plugin().build_complete_bundle(_request(tmp_path))

    message = str(caught.value)
    assert "status 23" in message
    assert "builder stdout\n" in message
    assert "exact native failure: attention build rejected\n" in message
    assert "'LD_LIBRARY_PATH': '/opt/trt/lib'" in message


def _patch_build_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    builder: Path,
) -> None:
    synthetic_config = {
        "model_type": "sam2",
        "architectures": ["Sam2BBoxVideoTracking"],
        "hidden_size": 256,
        "intermediate_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
    }
    monkeypatch.setattr(model_config, "config_from_dir", lambda _path: synthetic_config)
    monkeypatch.setattr(
        plugin_module,
        "require_reference_archive",
        lambda _path: _authenticated_description(root),
    )
    monkeypatch.setattr(plugin_module, "locate_native_builder", lambda: builder)
    monkeypatch.setattr(engine_builder.build_bundle, "_fp8_scales", None, raising=False)
    monkeypatch.setattr(engine_builder.build_bundle, "_save_fp8_scales", None, raising=False)
    monkeypatch.setattr(
        engine_builder,
        "_setup_trt_import",
        lambda _rtx: pytest.fail("complete bundle path must not import TensorRT Python"),
    )


def test_existing_output_is_never_replaced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _materialize_archive_paths(tmp_path)
    builder = _write_executable(tmp_path / "sam2_native_builder")
    _patch_build_entrypoint(monkeypatch, tmp_path, builder)
    monkeypatch.setattr(
        plugin_module.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("builder must not run for an existing output"),
    )
    output = tmp_path / "existing.bundle"
    output.write_bytes(b"preserve-me")

    with pytest.raises(FileExistsError, match="already exists"):
        engine_builder.build_bundle(str(tmp_path), str(output), precision="bf16")

    assert output.read_bytes() == b"preserve-me"


def test_invalid_native_output_is_preserved_and_rejected_by_shared_seam(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _materialize_archive_paths(tmp_path)
    builder = _write_executable(tmp_path / "sam2_native_builder")
    _patch_build_entrypoint(monkeypatch, tmp_path, builder)
    output = tmp_path / "invalid.bundle"

    def write_invalid(argv, **_kwargs):
        output_arg = Path(argv[argv.index("--output") + 1])
        output_arg.write_bytes(b"not-a-bundle")
        return subprocess.CompletedProcess(argv, 0, stdout="built\n", stderr="")

    monkeypatch.setattr(plugin_module.subprocess, "run", write_invalid)

    with pytest.raises(ValueError, match="invalid bundle magic/header"):
        engine_builder.build_bundle(str(tmp_path), str(output), precision="bf16")

    assert output.read_bytes() == b"not-a-bundle"
