# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral timing checks use declarations and actual reference sessions."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.perf_matrix as perf
from apps.benchmark.performance.baselines import task_reference
from apps.benchmark.performance.baselines.timing_contracts import timing_contract
from trtmc_benchmark.types import ModelDescriptor


def _declaration(scope="task-pipeline-call-wall", preparation=True, assets=False):
    return {"timing_scope": scope, "input_preparation_included": preparation,
            "asset_loading_included": assets}


def test_every_release_entry_preserves_its_existing_explicit_timing():
    _, entries, _ = perf.load_suite(perf.REPOSITORY / "apps/benchmark/performance/release.yaml")
    assert entries
    for entry in entries:
        baseline = entry["baseline"]
        contract = timing_contract(runner=baseline["runner"], declared=baseline)
        assert {name: contract[name] for name in _declaration()} == {
            name: baseline[name] for name in _declaration()
        }
        assert contract["candidate_timing_scope"] == "public_task_call_wall"


@pytest.mark.parametrize("change", [
    {"timing_scope": "guessed-clock"},
    {"input_preparation_included": "false"},
    {"asset_loading_included": 0},
    {"timing_scope": "task-model-call-wall"},
    {"input_preparation_included": False},
])
def test_invalid_timing_declarations_are_not_coerced(change):
    with pytest.raises(ValueError):
        timing_contract(runner="task-reference", declared={**_declaration(), **change})


@pytest.mark.parametrize("missing", tuple(_declaration()))
def test_all_reference_timing_fields_are_required(missing):
    declared = _declaration()
    del declared[missing]
    with pytest.raises(ValueError, match=missing):
        timing_contract(runner="task-reference", declared=declared)


def test_hf_reference_boundary_and_candidate_boundary_remain_distinct():
    declared = _declaration("public_operation_call_wall")
    contract = timing_contract(runner="hf-transformers", declared=declared)
    assert contract["timing_scope"] == "public_operation_call_wall"
    assert contract["candidate_timing_scope"] == "public_task_call_wall"
    with pytest.raises(ValueError, match="inconsistent"):
        timing_contract(runner="hf-transformers", declared={**declared, "asset_loading_included": True})


def test_candidate_and_reference_use_the_same_effective_declaration(tmp_path, monkeypatch):
    manifest = tmp_path / "model.json"
    manifest.write_text("{}", encoding="utf-8")
    model = ModelDescriptor("fixture", "", "", "fixture.bundle", "new_owner", "text_continuation", "fp32",
                            manifest, ({"name": "case", "prompt": "hello"},), {})
    monkeypatch.setattr(perf, "ManifestCatalog", lambda _: SimpleNamespace(resolve=lambda _selector: model))
    declaration = _declaration(assets=True)
    spec = {
        "id": "fixture.generate", "family": model.family, "model": model.name,
        "operation": "generate", "workload": {"testcase": "case"},
        "measurement": {"warmup": 0, "iterations": 1},
        "baseline": {"runner": "task-reference", "adapter": "test", **declaration},
    }
    environment = SimpleNamespace(bundle_cache=tmp_path / "cache", runtime_root=tmp_path / "runtime")
    entry = perf.resolve_entries([spec], environment)[0]
    assert entry.baseline_timing == declaration
    assert entry.case.measurement.asset_loading_included is True
    assert entry.case.measurement.timing_scope == "public_task_call_wall"


def _arguments(tmp_path: Path, declaration, family="new_owner"):
    return SimpleNamespace(
        adapter="hf-transformers-vision", family=family, operation="classify", model="fixture",
        mode="hf-eager", precision="fp32", padding="longest", warmup=1, iterations=2,
        request_json="{}", adapter_options_json="{}", timing_contract_json=json.dumps(declaration),
        output=tmp_path / "result.json", case_name="fixture",
    )


def _loader(monkeypatch, actual):
    calls = []

    def load(*_args):
        calls.append("load")

        def invoke():
            calls.append("invoke")
            return {"value": 7}

        return task_reference.Session(invoke, "fixture", **actual)

    def measure(session, warmup, iterations):
        calls.append("measure")
        for _ in range(warmup + iterations):
            summary = session.invoke()
        return [2.0] * iterations, summary

    monkeypatch.setitem(task_reference.LOADERS, "hf-transformers-vision", load)
    monkeypatch.setattr(task_reference, "_measure", measure)
    monkeypatch.setattr(task_reference, "_environment", lambda: {})
    return calls


@pytest.mark.parametrize("family", ["bert", "eagle_vlm", "not_in_any_family_list"])
def test_real_session_metadata_not_family_name_controls_validation(tmp_path, monkeypatch, family):
    actual = _declaration("task-model-call-wall", False)
    calls = _loader(monkeypatch, actual)
    arguments = _arguments(tmp_path, actual, family)
    assert task_reference.run(arguments) == 0
    result = json.loads(arguments.output.read_text(encoding="utf-8"))
    assert {name: result["measurement_policy"][name] for name in actual} == actual
    assert calls == ["load", "measure", "invoke", "invoke", "invoke"]


@pytest.mark.parametrize("actual", [
    _declaration("task-model-call-wall", False),
    _declaration(assets=True),
])
def test_session_timing_drift_is_rejected_before_warmup_or_measurement(tmp_path, monkeypatch, actual):
    calls = _loader(monkeypatch, actual)
    arguments = _arguments(tmp_path, _declaration())
    with pytest.raises(RuntimeError, match="timing drifted"):
        task_reference.run(arguments)
    assert calls == ["load"]
    assert not arguments.output.exists()


def test_unconfigured_fixed_loader_reports_its_actual_session_boundary(tmp_path, monkeypatch):
    actual = _declaration("task-model-call-wall", False)
    _loader(monkeypatch, actual)
    arguments = _arguments(tmp_path, {})
    assert task_reference.run(arguments) == 0
    result = json.loads(arguments.output.read_text(encoding="utf-8"))
    assert {name: result[name] for name in actual} == actual


def test_ambiguous_embedding_reference_requires_existing_explicit_timing_option():
    with pytest.raises(ValueError, match="requires explicit --timing-contract-json"):
        task_reference._load_embedding(SimpleNamespace(family="bert"), {"prompt": "hello"}, {})
    with pytest.raises(ValueError, match="preloads its assets"):
        task_reference._load_embedding(
            SimpleNamespace(timing_contract_json=json.dumps(_declaration(assets=True))), {}, {},
        )


def test_delegated_runner_cannot_echo_the_expected_asset_flag(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path, _declaration(assets=True))
    arguments.adapter, arguments.mode = "upstream-lance", "pytorch-eager"
    monkeypatch.setattr(task_reference, "_run_lance", lambda *_: (
        [2.0, 3.0], {}, "fixture", "task-pipeline-call-wall", True, False,
    ))
    with pytest.raises(RuntimeError, match="timing drifted"):
        task_reference.run(arguments)
    assert not arguments.output.exists()
