"""Unit tests for tools/auto_perf_tune.py metadata loading."""

from __future__ import annotations

import importlib
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
        "/tmp/model.trtfb",
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
    assert "/tmp/build/trtmc custom-benchmark /tmp/model.trtfb" in cmd
    assert f"{mod.PROJECT_ROOT}/tests/fixtures/generic_input.bin" in cmd
    assert "--hf-python /opt/venv/bin/python" in cmd
    assert "--set runtime.prefer_gpu_greedy=true" in cmd


def test_build_benchmark_command_rejects_unknown_placeholder() -> None:
    mod = importlib.import_module("auto_perf_tune")

    with pytest.raises(ValueError, match="Unknown benchmark command placeholder"):
        mod._build_bench_cmd(
            "/tmp/model.trtfb",
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
