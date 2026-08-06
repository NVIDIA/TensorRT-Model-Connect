# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tools/auto_perf_tune.py metadata loading."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def test_load_default_validation_models_reads_model_owned_sidecars(tmp_path: Path) -> None:
    mod = importlib.import_module("auto_perf_tune")

    owner_a = tmp_path / "model_a"
    owner_a.mkdir()
    (owner_a / "perf_validation.json").write_text(
        """
        {
          "models": [
            {"model": "org/a", "pipeline_type": "qwen_decoder_kv_cache", "label": "a"}
          ]
        }
        """,
        encoding="utf-8",
    )
    owner_b = tmp_path / "model_b"
    owner_b.mkdir()
    (owner_b / "perf_validation.json").write_text(
        """
        [
          {"model": "org/b", "pipeline_type": "embedding"}
        ]
        """,
        encoding="utf-8",
    )

    models = mod.load_default_validation_models(tmp_path)

    assert models == [
        {"model": "org/a", "pipeline_type": "qwen_decoder_kv_cache", "label": "a"},
        {"model": "org/b", "pipeline_type": "embedding", "label": "model_b-1"},
    ]


def test_build_benchmark_command_expands_model_owned_template() -> None:
    mod = importlib.import_module("auto_perf_tune")

    cmd, metric, label = mod._build_bench_cmd(
        "/tmp/model.bundle",
        prompt="unused for this command",
        max_tokens=32,
        gpu_argmax=True,
        benchmark={
            "label": "CPU path",
            "gpu_argmax_label": "GPU path",
            "metric": "pipeline_ms",
            "command": [
                "{binary}",
                "custom-benchmark",
                "{bundle}",
                "--input",
                "{repo_root}/tests/fixtures/generic_input.bin",
                "--max-new-tokens",
                "{max_tokens}",
                "{hf_python_args}",
                "{config_args}",
            ],
        },
    )

    assert metric == "pipeline_ms"
    assert label == "GPU path"
    assert cmd == [
        "/tmp/build/trtmc",
        "custom-benchmark",
        "/tmp/model.bundle",
        "--input",
        f"{mod.PROJECT_ROOT}/tests/fixtures/generic_input.bin",
        "--max-new-tokens",
        "32",
        "--hf-python",
        "/opt/venv/bin/python",
        "--set",
        "platform.trt_log_stderr=true",
        "--set",
        "runtime.prefer_gpu_greedy=true",
    ]


def test_build_benchmark_command_rejects_unknown_placeholder() -> None:
    mod = importlib.import_module("auto_perf_tune")

    with pytest.raises(ValueError, match="Unknown benchmark command placeholder"):
        mod._build_bench_cmd(
            "/tmp/model.bundle",
            prompt="hello",
            max_tokens=8,
            gpu_argmax=False,
            benchmark={
                "command": ["{binary}", "run", "{bundle}", "{family_owned_token}"],
            },
        )


def test_default_perf_sidecars_own_benchmark_templates() -> None:
    mod = importlib.import_module("auto_perf_tune")

    models = mod.load_default_validation_models()

    missing = [entry["label"] for entry in models if "benchmark" not in entry]
    assert not missing
    for entry in models:
        benchmark = entry["benchmark"]
        assert benchmark["metric"] in {"tok/s", "pipeline_ms", "rtf"}
        assert isinstance(benchmark["command"], list)
        assert all(isinstance(token, str) for token in benchmark["command"])


def test_run_cmd_uses_argv_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = importlib.import_module("auto_perf_tune")
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "stdout", "stderr")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    rc, output = mod.run_cmd(["command", "value; touch /tmp/not-created"])

    assert rc == 0
    assert output == "stdoutstderr"
    assert calls == [
        (
            ["command", "value; touch /tmp/not-created"],
            {
                "shell": False,
                "capture_output": True,
                "text": True,
                "timeout": 600,
            },
        )
    ]


def test_run_cmd_dry_run_renders_quoted_argv(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mod = importlib.import_module("auto_perf_tune")

    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail("dry-run must not start a subprocess")

    monkeypatch.setattr(mod.subprocess, "run", unexpected_run)

    assert mod.run_cmd(
        ["command", "value; touch /tmp/not-created"], dry_run=True
    ) == (0, "")
    assert capsys.readouterr().out == (
        "  [dry-run] command 'value; touch /tmp/not-created'\n"
    )


def test_step_build_passes_untrusted_values_as_single_argv_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = importlib.import_module("auto_perf_tune")
    calls: list[list[str]] = []

    def fake_run_cmd(command: list[str], **_kwargs: object) -> tuple[int, str]:
        calls.append(command)
        return 0, ""

    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)

    assert mod.step_build(
        "org/model; touch /tmp/not-created",
        "/tmp/output bundle.bundle",
        precision="fp16",
        max_cache=512,
    )
    assert calls == [[
        "./build/trtmc",
        "build",
        "org/model; touch /tmp/not-created",
        "-o",
        "/tmp/output bundle.bundle",
        "--max-cache-length",
        "512",
        "--precision",
        "fp16",
    ]]


def test_benchmark_passes_prompt_as_single_argv_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = importlib.import_module("auto_perf_tune")
    calls: list[list[str]] = []

    def fake_run_cmd(command: list[str], **_kwargs: object) -> tuple[int, str]:
        calls.append(command)
        return 0, "[trtmc] Decode: 10 tokens, 10 ms, 1000 tok/s\n"

    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)

    value = mod.step_benchmark(
        "/tmp/model.bundle",
        "hello; touch /tmp/not-created",
        max_tokens=10,
    )

    assert value == 1000
    prompt_index = calls[0].index("--prompt")
    assert calls[0][prompt_index + 1] == "hello; touch /tmp/not-created"


def test_nsys_wraps_benchmark_argv_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = importlib.import_module("auto_perf_tune")
    calls: list[list[str]] = []

    def fake_run_cmd(command: list[str], **_kwargs: object) -> tuple[int, str]:
        calls.append(command)
        return 1, "profile failed"

    monkeypatch.setattr(mod.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)

    result = mod.step_nsys_profile(
        "/tmp/model.bundle",
        "hello; touch /tmp/not-created",
        "/tmp/profile output",
    )

    assert result is None
    assert calls[0][:9] == [
        "/tmp/nsys_install/opt/nvidia/nsight-systems-cli/2026.2.1/target-linux-x64/nsys",
        "profile",
        "-t",
        "cuda,nvtx",
        "--cuda-graph-trace=node",
        "-o",
        "/tmp/profile output",
        "--force-overwrite",
        "true",
    ]
    prompt_index = calls[0].index("--prompt")
    assert calls[0][prompt_index + 1] == "hello; touch /tmp/not-created"
