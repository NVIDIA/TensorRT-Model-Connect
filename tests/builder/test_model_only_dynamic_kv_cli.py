# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for exact Qwen/TinyLlama model-only build routing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.dynamic_memory


def _model_config(path: Path, *, model_type: str, max_positions: int) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "max_position_embeddings": max_positions,
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            }
        )
    )


def _build_args(model: str) -> argparse.Namespace:
    return argparse.Namespace(
        command="build",
        model=model,
        output=None,
        max_cache_length=None,
        decoder_engine_layout="split",
        dynamic_kv_cache=None,
        dynamic_kv_profile_rows=None,
        precision="fp32",
        method="trt",
        quantize=None,
        quant_scales=None,
        quant_calibration_samples=512,
        verbose=False,
        _skip_profile_resolution=False,
    )


@pytest.mark.parametrize(
    (
        "model_id",
        "family",
        "model_type",
        "semantic_limit",
        "engine_limit",
        "profile_limits",
        "output",
    ),
    (
        (
            "Qwen/Qwen3-0.6B",
            "qwen",
            "qwen3",
            40960,
            40960,
            (128, 256, 512, 1024, 2048, 8192, 32768, 40960),
            "qwen3-0.6b.trtfb",
        ),
        (
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "llama",
            "llama",
            2048,
            2048,
            (128, 256, 512, 2048),
            "tinyllama-1.1b-chat-v1.0.trtfb",
        ),
    ),
)
def test_model_only_qwen_and_llama_builds_select_dynamic_split_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    model_id: str,
    family: str,
    model_type: str,
    semantic_limit: int,
    engine_limit: int,
    profile_limits: tuple[int, ...],
    output: str,
) -> None:
    import tensorrt_model_connect.build_cli as cli
    import tensorrt_model_connect.engine_builder as engine_builder

    model_dir = tmp_path / output.removesuffix(".trtfb")
    _model_config(
        model_dir,
        model_type=model_type,
        max_positions=semantic_limit,
    )
    captured: dict[str, object] = {}
    optimized_calls: list[str] = []

    def unexpected_optimized_probe(model: str, *_args, **_kwargs):
        optimized_calls.append(model)
        raise AssertionError(
            "model-only dynamic-capable family must prefer native builder")

    qualification = SimpleNamespace(
        family=family,
        model_dir=model_dir,
        precision="bf16",
        model_context_limit=semantic_limit,
        active_kv_profile_limits=profile_limits,
        qualified_model_id=model_id,
        qualified_model_revision="1" * 40,
    )
    monkeypatch.setattr(
        cli,
        "_model_only_native_dynamic_qualification",
        lambda _args: qualification,
    )
    monkeypatch.setattr(
        engine_builder,
        "_try_build_optimized_runtime",
        unexpected_optimized_probe,
    )
    monkeypatch.setattr(
        cli,
        "_resolve_build_model_metadata",
        lambda *_args, **_kwargs: (str(model_dir), family),
    )
    monkeypatch.setattr(
        cli,
        "_maybe_reexec_build_in_profile",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl_qualified",
        lambda **kwargs: captured.update(kwargs),
    )

    args = _build_args(model_id)
    args.precision = None
    assert cli._cmd_build(args) == 0

    assert args.output == output
    assert captured["output_path"] == output
    assert captured["model_id_or_path"] == str(model_dir)
    assert captured["max_cache_length"] == engine_limit
    assert captured["dynamic_kv_cache"] is True
    assert captured["decoder_engine_layout"] == "split"
    assert captured["dynamic_kv_profile_rows_override"] == list(profile_limits)
    assert captured["runtime_memory_qualification"] is qualification
    assert captured["precision"] == "bf16"
    assert optimized_calls == []

    stderr = capsys.readouterr().err
    assert f"Output: {output} (derived from model)" in stderr
    assert (
        f"Model-only dynamic KV capability selected native builder: family={family}"
        in stderr
    )
    assert f"model_context_limit={semantic_limit}" in stderr
    assert f"engine_context_limit={engine_limit}" in stderr
    assert "engine_context_limit=4096," not in stderr
    assert "dynamic_kv_cache=true" in stderr
    assert "decoder_layout=split" in stderr


def test_legacy_qwen_build_overrides_remain_accepted(
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.build_cli as cli

    model_dir = tmp_path / "qwen"
    _model_config(model_dir, model_type="qwen3", max_positions=40960)

    policy = cli._resolve_native_llm_build_policy(
        str(model_dir),
        "qwen",
        max_cache_length=512,
        dynamic_kv_cache=False,
        decoder_engine_layout="dual_profile",
    )

    assert policy == (512, False, "dual_profile")

    layout_only_policy = cli._resolve_native_llm_build_policy(
        str(model_dir),
        "qwen",
        max_cache_length=None,
        dynamic_kv_cache=None,
        decoder_engine_layout="dual_profile",
    )
    assert layout_only_policy == (256, False, "dual_profile")

    tensor_parallel_policy = cli._resolve_native_llm_build_policy(
        str(model_dir),
        "qwen",
        max_cache_length=None,
        dynamic_kv_cache=None,
        decoder_engine_layout="split",
        tensor_parallel_size=2,
    )
    assert tensor_parallel_policy == (256, False, "split")


def test_real_cli_explicit_cache_length_preserves_legacy_static_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_model_connect.build_cli as cli

    captured: dict[str, object] = {}

    def capture(args: argparse.Namespace) -> int:
        captured.update(vars(args))
        return 0

    monkeypatch.setattr(cli, "_cmd_build", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        ["trtmc", "build", "Qwen/Qwen3-0.6B", "--max-cache-length", "512"],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    policy = cli._resolve_native_llm_build_policy(
        str(captured["model"]),
        "qwen",
        max_cache_length=int(captured["max_cache_length"]),
        dynamic_kv_cache=captured["dynamic_kv_cache"],
        decoder_engine_layout=str(captured["decoder_engine_layout"]),
    )
    assert policy == (512, False, "split")


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    (
        (["--decoder-engine-layout", "dual_profile"], (256, False, "dual_profile")),
        (["--tensor-parallel-size", "2"], (256, False, "split")),
    ),
)
def test_real_cli_advanced_build_profiles_do_not_enable_automatic_dynamic_kv(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected: tuple[int, bool, str],
) -> None:
    import tensorrt_model_connect.build_cli as cli

    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "_cmd_build",
        lambda args: captured.update(vars(args)) or 0,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["trtmc", "build", "Qwen/Qwen3-0.6B", *extra_args],
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    policy = cli._resolve_native_llm_build_policy(
        str(captured["model"]),
        "qwen",
        max_cache_length=captured["max_cache_length"],
        dynamic_kv_cache=captured["dynamic_kv_cache"],
        decoder_engine_layout=captured["decoder_engine_layout"],
        tensor_parallel_size=int(captured["tensor_parallel_size"]),
    )
    assert policy == expected


def test_explicit_legacy_qwen_cache_profile_can_still_select_optimized_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.build_cli as cli
    import tensorrt_model_connect.engine_builder as engine_builder

    model_dir = tmp_path / "qwen"
    _model_config(model_dir, model_type="qwen3", max_positions=40960)
    optimized_calls: list[tuple[str, str, dict[str, object]]] = []

    monkeypatch.setattr(
        engine_builder,
        "_resolve_model",
        lambda *_args, **_kwargs: str(model_dir),
    )

    def select_optimized(model: str, output: str, options: dict[str, object]):
        optimized_calls.append((model, output, options))
        return object()

    monkeypatch.setattr(
        engine_builder,
        "_try_build_optimized_runtime",
        select_optimized,
    )
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **_kwargs: pytest.fail(
            "explicit legacy profile should retain optimized routing"),
    )

    args = _build_args("Qwen/Qwen3-0.6B")
    args.output = "legacy-qwen.trtfb"
    args.max_cache_length = 256

    assert cli._cmd_build(args) == 0
    assert optimized_calls == [
        (
            "Qwen/Qwen3-0.6B",
            "legacy-qwen.trtfb",
            {
                "decoder_engine_layout": "split",
                "dynamic_kv_cache": False,
                "dynamic_kv_profile_rows": None,
                "max_cache_length": 256,
                "method": "trt",
                "precision": "fp32",
                "quant_calibration_samples": 512,
                "quant_scales": None,
                "quantize": None,
                "verbose": False,
            },
        )
    ]


@pytest.mark.parametrize(
    ("model_id", "revision", "model_type", "max_positions"),
    (
        ("Qwen/Qwen3-1.7B", "d" * 40, "qwen3", 40960),
        (
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            "e" * 40,
            "llama",
            2048,
        ),
    ),
)
def test_unqualified_qwen_and_llama_snapshots_preserve_previous_optimized_routing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_id: str,
    revision: str,
    model_type: str,
    max_positions: int,
) -> None:
    import tensorrt_model_connect.build_cli as cli
    import tensorrt_model_connect.engine_builder as engine_builder

    model_dir = tmp_path / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
    model_dir.parent.mkdir(parents=True)
    _model_config(
        model_dir,
        model_type=model_type,
        max_positions=max_positions,
    )
    optimized_calls: list[tuple[str, str, dict[str, object]]] = []

    def select_optimized(
        model: str,
        output: str,
        options: dict[str, object],
        **_kwargs: object,
    ) -> object:
        optimized_calls.append((model, output, options))
        return object()

    monkeypatch.setattr(
        engine_builder,
        "_try_build_optimized_runtime",
        select_optimized,
    )
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl",
        lambda **_kwargs: pytest.fail(
            "an unqualified model must retain the pre-existing optimized route"
        ),
    )
    monkeypatch.setattr(
        engine_builder,
        "_build_native_impl_qualified",
        lambda **_kwargs: pytest.fail(
            "an unqualified model must never enter the qualified native builder"
        ),
    )

    args = _build_args(str(model_dir))
    args.precision = None

    assert cli._model_only_native_dynamic_qualification(args) is None
    assert cli._cmd_build(args) == 0
    assert len(optimized_calls) == 1
    routed_model, output, options = optimized_calls[0]
    assert routed_model == str(model_dir)
    assert output == cli._default_bundle_output(str(model_dir))
    assert options["max_cache_length"] == 256
    assert options["dynamic_kv_cache"] is False
    assert "runtime_memory_qualification" not in options
    assert "_runtime_memory_contract" not in options


def test_model_only_native_preference_is_exact_qualification_gated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import tensorrt_model_connect.build_cli as cli
    import tensorrt_model_connect.engine_builder as engine_builder

    model_dir = tmp_path / "gpt2"
    _model_config(model_dir, model_type="gpt2", max_positions=1024)
    monkeypatch.setattr(
        engine_builder,
        "_resolve_model",
        lambda *_args, **_kwargs: str(model_dir),
    )

    assert cli._model_only_native_dynamic_family(
        _build_args("openai-community/gpt2")
    ) is None


def test_model_only_build_parser_hides_build_time_kv_controls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tensorrt_model_connect.build_cli as cli

    monkeypatch.setattr(sys, "argv", ["trtmc", "build", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
    assert "-o OUTPUT, --output OUTPUT" in help_text
    assert "default: derived from model name" in normalized_help
    assert "--max-cache-length" not in help_text
    assert "--dynamic-kv-cache" not in help_text
    assert "--dynamic-kv-profile-rows" not in help_text
    assert "--decoder-engine-layout" not in help_text
    assert "--triattention-" not in help_text
