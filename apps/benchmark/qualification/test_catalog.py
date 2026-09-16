# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from .catalog import discover, load_benchmark, select


REPOSITORY = Path(__file__).resolve().parents[3]


def test_family_configs_auto_discover_both_kinds_without_l0() -> None:
    cases = discover(REPOSITORY)

    assert {(case.model, case.kind) for case in cases} == {
        ("gpt2-125m", "accuracy"),
        ("gpt2-125m", "performance"),
        ("chronos-bolt-tiny-official", "accuracy"),
        ("chronos-bolt-tiny-official", "performance"),
    }
    assert not any("l0" in case.model.lower() for case in cases)


def test_exact_model_selection_keeps_all_cases_in_one_file() -> None:
    cases = select(discover(REPOSITORY), ["gpt2-125m"])

    assert {case.kind for case in cases} == {"accuracy", "performance"}
    assert {case.name for case in cases} == {"mmlu-continuation", "generate-64"}


def test_shared_benchmarks_do_not_register_models() -> None:
    for case in discover(REPOSITORY):
        definition = load_benchmark(REPOSITORY, case)
        assert "models" not in definition
        assert definition["kind"] == case.kind
