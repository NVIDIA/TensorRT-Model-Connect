# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the model-owned CPU runner."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parent / "ci_run.py"


def _load_runner():
    name = f"trtmc_qwen_edgellm_ci_run_{id(object())}"
    specification = importlib.util.spec_from_file_location(name, RUNNER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_leaf_scope_is_exact_and_keeps_coexistence() -> None:
    runner = _load_runner()
    profile = "qwen3_0_6b_fp16_a100_pcie80_sm80"

    selected = runner.selected_tests("leaf", profile)

    assert selected == (
        f"tests/e2e/models/qwen/edge_llm_adapter/{profile}",
        "tests/e2e/models/qwen/edge_llm_adapter/coexistence/test_coexistence_contract.py",
    )


def test_family_scope_is_only_the_owned_adapter_tree() -> None:
    runner = _load_runner()

    assert runner.selected_tests("family", "") == ("tests/e2e/models/qwen/edge_llm_adapter",)


def test_provider_scope_runs_only_model_owned_contracts() -> None:
    runner = _load_runner()

    assert runner.selected_tests("provider", "") == (
        "tests/e2e/models/qwen/edge_llm_adapter",
    )


def test_symlinked_leaf_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load_runner()
    repository = tmp_path / "repo"
    root = repository / "tests/e2e/models/qwen/edge_llm_adapter"
    target = tmp_path / "outside"
    root.mkdir(parents=True)
    target.mkdir()
    (root / "profile_a").symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(runner, "REPOSITORY", repository)
    monkeypatch.setattr(runner, "TEST_ROOT", root)

    with pytest.raises(ValueError, match="unknown EdgeLLM profile"):
        runner.selected_tests("leaf", "profile_a")


def test_local_runner_does_not_provision_or_replace_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    monkeypatch.delenv("TRTMC_OPTIMIZED_CI_PROVISION", raising=False)

    python, environment = runner._provision_runtime()

    assert python == sys.executable
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_ci_runner_provisions_only_model_owned_pinned_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    include = tmp_path / "include"
    calls: list[tuple[str, ...]] = []
    monkeypatch.setenv("TRTMC_OPTIMIZED_CI_PROVISION", "1")
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path / "runner"))
    for name in tuple(os.environ):
        if name.startswith("_TRTMC_INTERNAL_QWEN3_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(runner, "NLOHMANN_INCLUDE", include)

    def run(command, **_kwargs):
        normalized = tuple(str(argument) for argument in command)
        calls.append(normalized)
        if normalized[1:3] == ("-m", "venv"):
            (Path(normalized[3]) / "bin").mkdir(parents=True)
            (Path(normalized[3]) / "bin/python").touch()
        if normalized[:3] == ("sudo", "apt-get", "install"):
            header = include / "nlohmann/json.hpp"
            header.parent.mkdir(parents=True)
            header.touch()
        return object()

    monkeypatch.setattr(runner.subprocess, "run", run)

    python, environment = runner._provision_runtime()

    expected_environment = tmp_path / "runner/trtmc-qwen-edgellm-ci"
    assert python == str(expected_environment / "bin/python")
    assert calls[0] == (sys.executable, "-m", "venv", str(expected_environment))
    assert calls[1] == (
        python,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "pytest==9.1.1",
        "numpy==2.4.6",
    )
    assert calls[2] == ("sudo", "apt-get", "update")
    assert calls[3] == (
        "sudo",
        "apt-get",
        "install",
        "--yes",
        "--no-install-recommends",
        "nlohmann-json3-dev=3.11.3-1",
    )
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not any(name.startswith("_TRTMC_INTERNAL_QWEN3_") for name in environment)


@pytest.mark.parametrize(
    ("scope", "profile"),
    (("leaf", "../escape"), ("leaf", "missing"), ("family", "unexpected")),
)
def test_invalid_selection_fails_closed(scope: str, profile: str) -> None:
    runner = _load_runner()

    with pytest.raises(ValueError):
        runner.selected_tests(scope, profile)
