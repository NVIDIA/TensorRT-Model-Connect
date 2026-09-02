# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The family's E2E plugins satisfy the harness protocols.

A plugin that misses a protocol member is not rejected -- the registry logs it
and moves on -- so the manifest check fails much later with a missing runner
and no cause attached. These assertions put the failure next to the plugin.
"""

from __future__ import annotations

import importlib

from tests.e2e_harness.contracts import (
    Comparator,
    ReferenceBackendRunner,
    TaskStrategyRunner,
)

_PACKAGE = "tests.e2e.models.minimax_music3.e2e_plugins"


def _plugin(module: str, attribute: str):
    return getattr(importlib.import_module(f"{_PACKAGE}.{module}"), attribute)


def test_runner_satisfies_the_task_strategy_protocol() -> None:
    runner = _plugin("runner", "runner")

    assert isinstance(runner, TaskStrategyRunner)
    assert runner.strategy_name == "text_to_audio"


def test_runner_also_names_its_runtime_strategy() -> None:
    assert _plugin("runner", "runner").runtime_strategy == (
        "minimax_music3_text_to_music"
    )


def test_reference_satisfies_the_backend_protocol() -> None:
    reference = _plugin("reference", "reference")

    assert isinstance(reference, ReferenceBackendRunner)
    assert reference.backend_name == "minimax_music3_modular"


def test_comparator_satisfies_the_comparator_protocol() -> None:
    comparator = _plugin("comparator", "comparator")

    assert isinstance(comparator, Comparator)
    assert comparator.task_strategy == "text_to_audio"


def test_the_manifest_names_the_backend_the_reference_declares() -> None:
    import json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).with_name("manifests") / "minimax-music3-l0.json")
        .read_text(encoding="utf-8")
    )
    declared = {case["reference_backend"] for case in manifest["testcases"]}

    assert declared == {_plugin("reference", "reference").backend_name}


def test_the_manifest_names_the_strategies_the_plugins_declare() -> None:
    import json
    from pathlib import Path

    manifest = json.loads(
        (Path(__file__).with_name("manifests") / "minimax-music3-l0.json")
        .read_text(encoding="utf-8")
    )

    assert manifest["task_strategy"] == _plugin("runner", "runner").strategy_name
    assert manifest["runtime_strategy"] == _plugin("runner", "runner").runtime_strategy
