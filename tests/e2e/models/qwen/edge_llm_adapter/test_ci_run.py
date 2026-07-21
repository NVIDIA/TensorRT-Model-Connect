# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for the model-owned CPU runner."""

from __future__ import annotations

import importlib.util
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


def test_provider_scope_adds_generic_provider_contracts() -> None:
    runner = _load_runner()

    selected = runner.selected_tests("provider", "")

    assert selected[0] == "tests/e2e/models/qwen/edge_llm_adapter"
    assert selected[1:] == runner.PROVIDER_TESTS


@pytest.mark.parametrize(
    ("scope", "profile"),
    (("leaf", "../escape"), ("leaf", "missing"), ("family", "unexpected")),
)
def test_invalid_selection_fails_closed(scope: str, profile: str) -> None:
    runner = _load_runner()

    with pytest.raises(ValueError):
        runner.selected_tests(scope, profile)
