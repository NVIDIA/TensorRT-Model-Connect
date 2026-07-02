# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for model-owned E2E runner bundle grouping."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.e2e_harness import bundle_group_runner as model_case_runner


def _load_runner(family: str, module_name: str):
    runner_path = (
        Path(__file__).resolve().parents[1]
        / "e2e"
        / "models"
        / family
        / "runner.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


canary_runner = _load_runner("canary", "canary_e2e_runner")
nemotron_speech_runner = _load_runner(
    "nemotron_speech_streaming",
    "nemotron_speech_streaming_e2e_runner",
)


class _Config:
    def __init__(self, **options):
        self._options = options

    def getoption(self, name: str, default=None):
        return self._options.get(name, default)


class _Request:
    def __init__(self, config: _Config):
        self.config = config


def test_canary_runner_collects_shared_bundle_as_one_case() -> None:
    case_names = canary_runner.model_case_names(
        _Config(
            **{
                "--e2e-group-by-bundle": True,
                "--e2e-exclude-ci-tier": [],
                "--e2e-model": [],
            }
        )
    )

    grouped = [
        name for name in case_names
        if name.startswith("bundle:canary-1b-v2+")
    ]

    assert grouped == [
        "bundle:canary-1b-v2"
        "+canary-1b-v2-asr-probe01"
        "+canary-1b-v2-asr-probe02"
        "+canary-1b-v2-asr-probe03"
        "+canary-1b-v2-asr-probe04"
        "+canary-1b-v2-asr-probe05"
        "+canary-1b-v2-asr-probe06"
        "+canary-1b-v2-asr-probe08"
    ]
    assert "canary-1b-v2" not in case_names
    assert "canary-1b-v2-asr-probe05" not in case_names
    assert "canary-1b-v2-tp4" not in case_names


def test_canary_runner_filters_exact_names_before_bundle_grouping(tmp_path) -> None:
    models_file = tmp_path / "models.txt"
    models_file.write_text("canary-1b-v2\n", encoding="utf-8")

    case_names = canary_runner.model_case_names(
        _Config(
            **{
                "--e2e-group-by-bundle": True,
                "--e2e-models-file": str(models_file),
                "--e2e-model": ["canary"],
            }
        )
    )

    assert case_names == ["canary-1b-v2"]


def test_canary_runner_treats_case_name_as_exact_before_bundle_grouping() -> None:
    case_names = canary_runner.model_case_names(
        _Config(
            **{
                "--e2e-group-by-bundle": True,
                "--e2e-model": ["canary-1b-v2"],
            }
        )
    )

    assert case_names == ["canary-1b-v2"]


def test_nemotron_speech_runner_does_not_collect_shared_bundle_as_group() -> None:
    case_names = nemotron_speech_runner.model_case_names(
        _Config(
            **{
                "--e2e-group-by-bundle": True,
                "--e2e-exclude-ci-tier": [],
                "--e2e-model": [],
            }
        )
    )

    assert not any(
        name.startswith("bundle:nemotron-speech-streaming-en-0.6b")
        for name in case_names
    )
    assert "nemotron-speech-streaming-en-0.6b" in case_names
    assert "nemotron-speech-streaming-en-0.6b-asr-probe01" in case_names


def test_grouped_runner_rebuilds_bundle_once(monkeypatch) -> None:
    calls = []

    def fake_run_case(
        case_name,
        request,
        model_dir,
        load_waives,
        resolve_hf_python,
        resolve_artifacts_dir,
        resolve_binary,
        resolve_ld_library_path,
        resolve_engine_dir,
        resolve_model_plugin_dir,
        model_plugin_dir_env,
        *,
        rebuild_override,
        mark_xfail,
    ):
        calls.append((case_name, rebuild_override, mark_xfail))
        return {
            "name": case_name,
            "status": "pass",
            "message": "",
            "bundle_exists": True,
        }

    monkeypatch.setattr(model_case_runner, "_run_case", fake_run_case)

    canary_runner.run_model_e2e(
        "bundle:canary-a+canary-b+canary-c",
        _Request(_Config(**{"--rebuild-engines": True})),
    )

    assert calls == [
        ("canary-a", True, False),
        ("canary-b", False, False),
        ("canary-c", False, False),
    ]
