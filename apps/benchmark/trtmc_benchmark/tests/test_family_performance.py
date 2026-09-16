# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary family fixtures for the shared performance process boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

import tools.perf_matrix as perf


REFERENCE_SCRIPT = '''"""Synthetic reference fixture; no model or GPU qualification."""
import argparse
import json
from pathlib import Path
import statistics
import time

parser = argparse.ArgumentParser()
for name in ("model", "family", "manifest", "operation", "selected-task", "request-json",
             "adapter-options-json", "timing-contract-json", "precision", "mode",
             "padding", "case-name", "output"):
    parser.add_argument("--" + name, required=True)
for name in ("warmup", "iterations"):
    parser.add_argument("--" + name, required=True, type=int)
parser.add_argument("--revision")
parser.add_argument("--local-files-only", action="store_true")
parser.add_argument("--trust-remote-code", action="store_true")
args = parser.parse_args()
request = json.loads(args.request_json)
timing = json.loads(args.timing_contract_json)
loaded_models = 0
invocations = 0
class FixtureModel:
    def __init__(self, task):
        global loaded_models
        loaded_models += 1
        # Selection changes the fixture operation, not only its output label.
        self.offset = {"text_continuation": 0, "conditional_text_generation": 1}[task]
    def run(self, source):
        return {"token_ids": [len(source) + self.offset], "text": "fixture"}
model = FixtureModel(args.selected_task)
def invoke():
    global invocations
    invocations += 1
    source = request["token_ids"] if "token_ids" in request else request["prompt"]
    return model.run(source)
for _ in range(args.warmup):
    invoke()
samples = []
for _ in range(args.iterations):
    started = time.perf_counter_ns()
    summary = invoke()
    samples.append((time.perf_counter_ns() - started) / 1_000_000)
summary.update(loaded_models=loaded_models, invocations=invocations)
result = {
    "schema_version": "trtmc.perf-baseline/v1", "status": "completed",
    "model": args.model, "family": args.family, "operation": args.operation,
    "case_name": args.case_name, "selected_task": args.selected_task,
    "precision": args.precision, "mode": args.mode, "framework": "synthetic-fixture",
    "measurement": {"warmup": args.warmup, "iterations": args.iterations},
    "measurement_policy": timing, **timing, "samples_ms": samples,
    "metrics": {"latency_ms": {"p50": statistics.median(samples)}},
    "output_summary": summary,
}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result))
'''


def test_custom_script_default_mode_does_not_claim_torch_compilation():
    assert perf._baseline_mode({"script": "tests/reference.py"}) == "reference"
    assert perf._baseline_mode({"script": "tests/reference.py", "mode": "pytorch-eager"}) == "pytorch-eager"
    assert perf._baseline_mode({"runner": "hf-transformers"}) == "torch-compile"


def _entry(family: str, model: str, *, script: bool = False, entry_id: str | None = None):
    baseline = {
        "runner": "task-reference" if script else "hf-transformers",
        "mode": "pytorch-eager" if script else "hf-eager",
        "timing_scope": "task-model-call-wall" if script else "public_operation_call_wall",
        "input_preparation_included": not script,
        "asset_loading_included": False,
    }
    if script:
        baseline["script"] = "tests/performance_reference.py"
    return {
        "id": entry_id or family + ".generate", "family": family, "model": model,
        "operation": "generate", "workload": {"testcase": model}, "baseline": baseline,
    }


def _write_suite(path: Path, entries, **extra):
    value = {
        "schema_version": perf.SUITE_SCHEMA, "name": "fixture-performance",
        "defaults": {"measurement": {"warmup": 2, "iterations": 3}},
        "entries": entries, **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    families = root / "families"
    for owner in ("sample", "other"):
        folder = families / owner / "tests"
        (folder / "manifests").mkdir(parents=True)
        manifest = {
            "name": owner + "-model", "bundle": owner + ".bundle", "family": owner,
            "task": "text_continuation", "precision": "fp32", "hf_id": "fixture/" + owner,
            "testcases": [{"name": owner + "-model", "prompt": "Hello"}],
        }
        (folder / "manifests" / (owner + ".json")).write_text(json.dumps(manifest))
    script = families / "sample/tests/performance_reference.py"
    script.write_text(REFERENCE_SCRIPT, encoding="utf-8")
    canonical = root / "apps/benchmark/performance/release.yaml"
    _write_suite(canonical, [_entry("other", "other-model")])
    owned_suite = families / "sample/tests/performance.yaml"
    _write_suite(owned_suite, [_entry("sample", "sample-model", script=True)])
    monkeypatch.setattr(perf, "REPOSITORY", root)
    monkeypatch.setattr(perf, "MANIFEST_ROOT", families)
    environment = SimpleNamespace(
        bundle_cache=tmp_path / "bundles", runtime_root=tmp_path / "runtime",
        hf_runner=tmp_path / "hf.py", task_runner=tmp_path / "task.py",
        local_files_only=True, references={},
    )
    return SimpleNamespace(root=root, families=families, canonical=canonical,
                           owned_suite=owned_suite, script=script, environment=environment)


def test_family_reference_minimal_real_subprocess(repository, tmp_path):
    _, specs, excluded = perf.load_suite(repository.canonical)
    assert [entry["id"] for entry in specs] == ["other.generate", "sample.generate"]
    assert excluded == set()
    entry, = perf.resolve_entries([specs[-1]], repository.environment)
    output = tmp_path / "reference.json"
    command = perf.baseline_command(entry, repository.environment, output)
    assert Path(command[1]) == repository.script
    assert "--adapter" not in command
    assert command[command.index("--selected-task") + 1] == "text_continuation"
    process = perf.run_command(command, timeout=10, stdout_path=tmp_path / "stdout.log",
                               stderr_path=tmp_path / "stderr.log", verbose=False)
    assert process["exit_code"] == 0
    result = perf._json_file(output, "fixture reference")
    perf._validate_script_result(entry, repository.environment, result)
    assert result["output_summary"] == {
        "token_ids": [5], "text": "fixture", "loaded_models": 1, "invocations": 5,
    }
    assert len(result["samples_ms"]) == 3
    assert perf._output_contract(entry, {"output_summary": {"token_ids": [5]}}, result)[0]


@pytest.mark.parametrize("entry_id", ["sample.generate", "unrelated.workload"])
def test_family_reference_takes_over_own_canonical_entry(repository, tmp_path, entry_id):
    legacy = _entry("sample", "sample-model", script=True, entry_id=entry_id)
    legacy["baseline"].pop("script")
    legacy["baseline"].update(adapter="hf-transformers-vlm", output_contract="exact-token-ids",
                              adapter_options={"legacy_only": True})
    legacy["equivalence_margin_percent"] = 1.25
    _write_suite(repository.canonical, [legacy, _entry("other", "other-model")])
    central_bytes = repository.canonical.read_bytes()
    original, = perf.resolve_entries(
        [perf._load_suite_file(repository.canonical)[1][0]], repository.environment,
    )

    owned = _entry("sample", "sample-model", script=True, entry_id=entry_id)
    owned["baseline"]["output_contract"] = "exact-token-ids"
    owned["equivalence_margin_percent"] = 1.25
    _write_suite(repository.owned_suite, [owned])
    _, specs, excluded = perf.load_suite(repository.canonical)
    assert [item["id"] for item in specs] == [entry_id, "other.generate"]
    assert excluded == set()
    perf._coverage(specs, excluded)
    assert specs[0] == perf._load_suite_file(repository.owned_suite, owner="sample")[1][0]
    assert "adapter" not in specs[0]["baseline"]
    assert "adapter_options" not in specs[0]["baseline"]
    entry, = perf.resolve_entries([specs[0]], repository.environment)
    assert entry.case.testcase_name == original.case.testcase_name
    assert entry.case.request == original.case.request
    assert entry.case.measurement == original.case.measurement
    assert entry.spec["equivalence_margin_percent"] == original.spec["equivalence_margin_percent"]
    assert entry.spec["baseline"]["output_contract"] == original.spec["baseline"]["output_contract"]
    output = tmp_path / "owned-reference.json"
    command = perf.baseline_command(entry, repository.environment, output)
    assert Path(command[1]) == repository.script and "--adapter" not in command
    process = perf.run_command(command, timeout=10, stdout_path=tmp_path / "stdout.log",
                               stderr_path=tmp_path / "stderr.log", verbose=False)
    assert process["exit_code"] == 0
    result = perf._json_file(output, "owned fixture reference")
    perf._validate_script_result(entry, repository.environment, result)
    assert result["output_summary"] == {
        "token_ids": [5], "text": "fixture", "loaded_models": 1, "invocations": 5,
    }
    assert len(result["samples_ms"]) == 3
    assert perf._output_contract(entry, {"output_summary": {"token_ids": [5]}}, result)[0]
    assert repository.canonical.read_bytes() == central_bytes


def test_family_takeover_is_complete_and_keeps_other_rows_in_place(repository):
    legacy = _entry("sample", "sample-model")
    legacy["baseline"]["adapter_options"] = {"central_only": True}
    _write_suite(repository.canonical, [legacy, _entry("other", "other-model")],
                 defaults={"measurement": {"warmup": 11, "iterations": 13},
                           "equivalence_margin_percent": 0.5},
                 additional_profiles=[{"inherit": "sample.generate", "model": "sample-model"}],
                 excluded_profiles=[{"model": "existing-exclusion", "reason": "existing policy"}])
    central_bytes = repository.canonical.read_bytes()
    original = perf._load_suite_file(repository.canonical)[1]
    owned = _entry("sample", "sample-model", script=True)
    owned["baseline"]["adapter_options"] = {"owned_only": False}
    added = {**deepcopy(owned), "id": "sample.additional"}
    _write_suite(repository.owned_suite, [added, owned])
    _, rows, excluded = perf.load_suite(repository.canonical)
    assert [row["id"] for row in rows] == [
        "sample.generate", "other.generate", "sample.generate@sample-model", "sample.additional",
    ]
    assert rows[0] == perf._load_suite_file(repository.owned_suite, owner="sample")[1][1]
    assert rows[0]["measurement"] == {"warmup": 2, "iterations": 3}
    assert rows[0]["equivalence_margin_percent"] == 5.0
    assert rows[0]["baseline"]["adapter_options"] == {"owned_only": False}
    assert rows[1:3] == original[1:3]
    assert excluded == {"existing-exclusion"}
    assert repository.canonical.read_bytes() == central_bytes


def test_family_takeover_uses_owner_not_entry_id_prefix(repository):
    central = _entry("other", "other-model", entry_id="sample.generate")
    _write_suite(repository.canonical, [central])
    with pytest.raises(perf.PerfMatrixError, match="duplicate suite entry"):
        perf.load_suite(repository.canonical)


@pytest.mark.parametrize("canonical", [False, True])
def test_family_suite_rejects_duplicate_derived_ids(repository, canonical):
    profile = {"inherit": "sample.generate", "model": "sample-model"}
    _write_suite(repository.owned_suite, [_entry("sample", "sample-model", script=True)],
                 additional_profiles=[profile, profile])
    with pytest.raises(perf.PerfMatrixError, match="duplicate suite entry"):
        perf.load_suite(repository.canonical if canonical else repository.owned_suite)


def test_family_additional_profiles_preserve_existing_chained_inheritance(repository):
    _write_suite(repository.owned_suite, [_entry("sample", "sample-model", script=True)],
                 additional_profiles=[
                     {"inherit": "sample.generate", "model": "sample-model"},
                     {"inherit": "sample.generate@sample-model", "model": "sample-model"},
                 ])
    rows = perf.load_suite(repository.canonical)[1]
    assert [row["id"] for row in rows] == [
        "other.generate", "sample.generate", "sample.generate@sample-model",
        "sample.generate@sample-model@sample-model",
    ]
    assert rows[-1]["baseline"] == rows[-2]["baseline"] == rows[-3]["baseline"]
    assert rows[-1]["workload"] == {"testcase": "sample-model"}
    assert rows[-1]["measurement"] == {"warmup": 2, "iterations": 3}


def test_family_can_take_over_parent_and_derived_entry(repository, tmp_path):
    owned = _entry("sample", "sample-model", script=True)
    owned["equivalence_margin_percent"] = 1.25
    owned["baseline"]["output_contract"] = "exact-token-ids"
    legacy = deepcopy(owned)
    legacy["baseline"].pop("script")
    legacy["baseline"]["adapter"] = "hf-transformers-vlm"
    profiles = [{"inherit": "sample.generate", "model": "sample-model"}]
    _write_suite(repository.canonical, [legacy, _entry("other", "other-model")],
                 additional_profiles=profiles)
    central_bytes = repository.canonical.read_bytes()
    original = perf.resolve_entries(perf._load_suite_file(repository.canonical)[1], repository.environment)
    _write_suite(repository.owned_suite, [owned], additional_profiles=profiles)
    _, rows, excluded = perf.load_suite(repository.canonical)
    assert [row["id"] for row in rows] == ["sample.generate", "other.generate", "sample.generate@sample-model"]
    perf._coverage(rows, excluded)
    actual = perf.resolve_entries(rows, repository.environment)
    for index in (0, 2):
        entry = actual[index]
        assert entry.case.request == original[index].case.request
        assert entry.case.testcase_name == original[index].case.testcase_name
        assert entry.case.measurement == original[index].case.measurement
        assert entry.spec["equivalence_margin_percent"] == 1.25
        assert entry.spec["baseline"]["output_contract"] == "exact-token-ids"
        assert "adapter" not in entry.spec["baseline"]
        output = tmp_path / f"reference-{index}.json"
        command = perf.baseline_command(entry, repository.environment, output)
        assert Path(command[1]) == repository.script
        process = perf.run_command(command, timeout=10, stdout_path=tmp_path / f"stdout-{index}.log",
                                   stderr_path=tmp_path / f"stderr-{index}.log", verbose=False)
        assert process["exit_code"] == 0
        result = perf._json_file(output, "owned parent/derived reference")
        perf._validate_script_result(entry, repository.environment, result)
        assert result["output_summary"] == {
            "token_ids": [5], "text": "fixture", "loaded_models": 1, "invocations": 5,
        }
        assert len(result["samples_ms"]) == 3
        assert perf._output_contract(entry, {"output_summary": {"token_ids": [5]}}, result)[0]
    assert repository.canonical.read_bytes() == central_bytes


def test_different_family_suites_cannot_claim_the_same_new_id(repository):
    _write_suite(repository.owned_suite, [
        _entry("sample", "sample-model", script=True, entry_id="new.workload"),
    ])
    _write_suite(repository.families / "other/tests/performance.yaml", [
        _entry("other", "other-model", entry_id="new.workload"),
    ])
    with pytest.raises(perf.PerfMatrixError, match="duplicate suite entry"):
        perf.load_suite(repository.canonical)


def test_family_takeover_cannot_hide_missing_ready_model_coverage(repository):
    _write_suite(repository.canonical, [_entry("sample", "sample-model"), _entry("other", "other-model")])
    path = repository.families / "sample/tests/manifests/alternate.json"
    value = json.loads(path.with_name("sample.json").read_text())
    value.update(name="sample-alternate", testcases=[{"name": "sample-alternate", "prompt": "Hello"}])
    path.write_text(json.dumps(value))
    _write_suite(repository.owned_suite, [
        _entry("sample", "sample-alternate", script=True, entry_id="sample.generate"),
    ])
    _, rows, excluded = perf.load_suite(repository.canonical)
    with pytest.raises(perf.PerfMatrixError, match="omits ready models: sample-model"):
        perf._coverage(rows, excluded)


def test_owned_reference_activates_only_its_declared_excluded_model(repository, tmp_path):
    manifest = repository.families / "sample/tests/manifests/still-excluded.json"
    value = json.loads(manifest.with_name("sample.json").read_text())
    value.update(name="sample-still-excluded",
                 testcases=[{"name": "sample-still-excluded", "prompt": "Hello"}])
    manifest.write_text(json.dumps(value))
    _write_suite(repository.canonical, [_entry("other", "other-model")], excluded_profiles=[
        {"model": "sample-model", "reason": "no reference workload yet"},
        {"model": "sample-still-excluded", "reason": "independent existing exclusion"},
    ])
    central_bytes = repository.canonical.read_bytes()
    owned = _entry("sample", "sample-model", script=True)
    owned["baseline"]["output_contract"] = "exact-token-ids"
    owned["equivalence_margin_percent"] = 1.25
    _write_suite(repository.owned_suite, [owned])
    _, rows, excluded = perf.load_suite(repository.canonical)
    assert excluded == {"sample-still-excluded"}
    perf._coverage(rows, excluded)
    entry, = perf.resolve_entries([rows[-1]], repository.environment)
    assert entry.model.name == "sample-model"
    assert entry.case.request == {"prompt": "Hello"}
    assert entry.case.measurement.warmup == 2 and entry.case.measurement.iterations == 3
    assert entry.spec["equivalence_margin_percent"] == 1.25
    output = tmp_path / "activated-reference.json"
    command = perf.baseline_command(entry, repository.environment, output)
    assert Path(command[1]) == repository.script
    process = perf.run_command(command, timeout=10, stdout_path=tmp_path / "stdout.log",
                               stderr_path=tmp_path / "stderr.log", verbose=False)
    assert process["exit_code"] == 0
    result = perf._json_file(output, "activated owned reference")
    perf._validate_script_result(entry, repository.environment, result)
    assert result["output_summary"] == {
        "token_ids": [5], "text": "fixture", "loaded_models": 1, "invocations": 5,
    }
    assert len(result["samples_ms"]) == 3
    assert perf._output_contract(entry, {"output_summary": {"token_ids": [5]}}, result)[0]
    assert repository.canonical.read_bytes() == central_bytes
    repository.owned_suite.unlink()
    _, rows, excluded = perf.load_suite(repository.canonical)
    assert excluded == {"sample-model", "sample-still-excluded"}
    perf._coverage(rows, excluded)
    assert repository.canonical.read_bytes() == central_bytes


def test_owned_entry_cannot_activate_another_familys_excluded_model(repository):
    _write_suite(repository.canonical, [_entry("sample", "sample-model")], excluded_profiles=[
        {"model": "other-model", "reason": "existing exclusion"},
    ])
    central_bytes = repository.canonical.read_bytes()
    _write_suite(repository.owned_suite, [_entry("sample", "other-model", script=True)])
    with pytest.raises(perf.PerfMatrixError, match="another family's manifest"):
        perf.load_suite(repository.canonical)
    assert repository.canonical.read_bytes() == central_bytes


def test_central_configured_and_excluded_model_still_fails_without_owned_entry(repository):
    repository.owned_suite.unlink()
    _write_suite(repository.canonical, [
        _entry("sample", "sample-model"), _entry("other", "other-model"),
    ], excluded_profiles=[{"model": "sample-model", "reason": "conflicting central policy"}])
    _, rows, excluded = perf.load_suite(repository.canonical)
    assert excluded == {"sample-model"}
    with pytest.raises(perf.PerfMatrixError, match="both configured and excluded: sample-model"):
        perf._coverage(rows, excluded)


def _resolved(repository):
    _, entries, _ = perf.load_suite(repository.canonical)
    entry, = perf.resolve_entries([entries[-1]], repository.environment)
    return entry


@pytest.mark.parametrize("prepare_only", [False, True])
@pytest.mark.parametrize("overrides", [
    {},
    {"prompt": "false\nquoted: 'yes' # text = retained", "config": {
        "seed": 2**60 + 3, "temperature": 1e-12, "enabled": False, "suffix": "",
        "ids": [0, -1, 2], "weights": [1e-15, -1e20], "labels": ["yes", "null", "a=b"],
    }},
    {"prompt": "new input", "config": {}},
    {"prompt": "null", "config": {"empty": [], "optional": None}},
])
def test_candidate_workload_overrides_round_trip_through_original_cli(
    repository, tmp_path, prepare_only, overrides
):
    from trtmc_benchmark import cli as benchmark_cli

    original = _resolved(repository)
    expected_native = deepcopy({**original.case.request, **overrides})
    expected_reference = {
        name: value for name, value in expected_native.items() if name != "config"
    }
    expected_reference.update(expected_native.get("config", {}))
    spec = {**original.spec, "workload": {
        **original.spec["workload"], "request": overrides,
    }}
    environment = SimpleNamespace(
        **vars(repository.environment), trtmc_bench=tmp_path / "bench",
        worker=tmp_path / "worker", bundle_roots=(),
    )
    entry, = perf.resolve_entries([spec], environment)
    command = perf.candidate_command(
        entry, environment, None if prepare_only else tmp_path / "candidate",
        prepare_only=prepare_only,
    )
    arguments = benchmark_cli.build_parser().parse_args(command[1:])
    actual, = benchmark_cli._resolve_cases(
        arguments, {}, perf.ManifestCatalog(perf.MANIFEST_ROOT),
        SimpleNamespace(provisional_path=lambda model: entry.case.bundle_path),
        environment.runtime_root,
    )
    reference = perf.baseline_command(entry, environment, tmp_path / "reference.json")
    reference_request = json.loads(reference[reference.index("--request-json") + 1])
    # Native keeps Config nested; references use the same explicit values flat.
    assert json.dumps(actual.worker_request()["request"], sort_keys=True) == json.dumps(
        expected_native, sort_keys=True
    )
    assert json.dumps(reference_request, sort_keys=True) == json.dumps(
        expected_reference, sort_keys=True
    )
    assert actual.request == entry.case.request
    assert actual.operation == entry.case.operation
    assert actual.measurement == entry.case.measurement
    assert actual.effective_task == entry.case.effective_task
    request_sets = [value for value in arguments.sets if value.startswith("request.")]
    assert len(request_sets) == len(overrides)
    assert spec["workload"]["request"] == overrides


@pytest.mark.parametrize("prepare_only", [False, True])
def test_candidate_and_reference_select_the_same_prepared_document(
    repository, tmp_path, prepare_only
):
    from trtmc_benchmark import cli as benchmark_cli

    original = tmp_path / "original.yaml"
    original.write_text("version: 1\n", encoding="utf-8")
    prepared = tmp_path / "prepared request.b2rq"
    prepared.write_bytes(b"B2RQ\x00prepared fixture")
    source = tmp_path / "source request.json"
    source.write_text("{}", encoding="utf-8")
    manifest_path = repository.families / "sample/tests/manifests/sample.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task"] = "molecular_document_to_structure"
    manifest["testcases"] = [{
        "name": "sample-model", "document_path": str(original),
        "config": {"seed": 9, "include_confidence": False},
    }]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_bytes = manifest_path.read_bytes()
    _, specs, _ = perf.load_suite(repository.canonical)
    spec = {**specs[-1], "operation": "predict_structure", "workload": {
        "testcase": "sample-model", "request": {
            "document_path": str(prepared), "input_encoding": "b2rq",
            "source_path": str(source), "config": {"seed": 0, "include_confidence": True},
        },
    }}
    expected_native = deepcopy(spec["workload"]["request"])
    expected_reference = {
        "document_path": str(prepared), "input_encoding": "b2rq", "source_path": str(source),
        "seed": 0, "include_confidence": True,
    }
    environment = SimpleNamespace(
        **vars(repository.environment), trtmc_bench=tmp_path / "bench",
        worker=tmp_path / "worker", bundle_roots=(),
    )
    entry, = perf.resolve_entries([spec], environment)
    command = perf.candidate_command(
        entry, environment, None if prepare_only else tmp_path / "candidate",
        prepare_only=prepare_only,
    )
    arguments = benchmark_cli.build_parser().parse_args(command[1:])
    actual, = benchmark_cli._resolve_cases(
        arguments, {}, perf.ManifestCatalog(perf.MANIFEST_ROOT),
        SimpleNamespace(provisional_path=lambda model: entry.case.bundle_path),
        environment.runtime_root,
    )
    reference = perf.baseline_command(entry, environment, tmp_path / "reference.json")
    reference_request = json.loads(reference[reference.index("--request-json") + 1])
    assert json.dumps(actual.worker_request()["request"], sort_keys=True) == json.dumps(
        expected_native, sort_keys=True
    )
    assert json.dumps(reference_request, sort_keys=True) == json.dumps(
        expected_reference, sort_keys=True
    )
    assert actual.request == spec["workload"]["request"] == expected_native
    assert actual.request["document_path"] != str(original)
    assert Path(actual.request["document_path"]).read_bytes() == prepared.read_bytes()
    assert actual.effective_task == "molecular_document_to_structure"
    assert actual.measurement == entry.case.measurement
    assert manifest_path.read_bytes() == manifest_bytes


def _result(entry):
    return {
        "schema_version": "trtmc.perf-baseline/v1", "status": "completed",
        "model": entry.model.hf_id, "family": entry.model.family,
        "operation": entry.spec["operation"], "case_name": entry.spec["id"],
        "selected_task": entry.case.effective_task, "precision": entry.reference_precision,
        "mode": entry.spec["baseline"]["mode"],
        "measurement": {"warmup": 2, "iterations": 3},
        "measurement_policy": dict(entry.baseline_timing), **entry.baseline_timing,
        "samples_ms": [1.0, 2.0, 3.0], "metrics": {"latency_ms": {"p50": 2.0}},
        "output_summary": {"token_ids": [5]},
    }


def test_explicit_standalone_suite_does_not_discover_family_files(repository, tmp_path):
    standalone = tmp_path / "custom.yaml"
    _write_suite(standalone, [_entry("other", "other-model")])
    assert len(perf.load_suite(standalone)[1]) == 1
    assert len(perf.load_suite(repository.owned_suite)[1]) == 1
    assert len(perf.load_suite(repository.canonical)[1]) == 2


def test_discovery_absent_is_noop_and_keeps_central_exclusions(repository):
    baseline = perf._load_suite_file(repository.canonical)
    repository.owned_suite.unlink()
    assert perf.load_suite(repository.canonical) == baseline
    _write_suite(repository.canonical, [_entry("other", "other-model")],
                 excluded_profiles=[{"model": "existing-exclusion", "reason": "existing policy"}])
    assert perf.load_suite(repository.canonical)[2] == {"existing-exclusion"}


def test_same_model_can_have_independent_secondary_workloads(repository):
    path = repository.families / "sample/tests/manifests/sample.json"
    manifest = json.loads(path.read_text())
    manifest["testcases"].append({
        "name": "conditional", "selected_task": "conditional_text_generation", "prompt": "Hello",
    })
    path.write_text(json.dumps(manifest))
    first = _entry("sample", "sample-model", script=True)
    second = {**deepcopy(first), "id": "sample.conditional", "workload": {"testcase": "conditional"}}
    _write_suite(repository.owned_suite, [first, second])
    entries = perf.load_suite(repository.canonical)[1]
    assert [row["id"] for row in entries] == ["other.generate", "sample.generate", "sample.conditional"]
    resolved = perf.resolve_entries(entries[1:], repository.environment)
    assert [item.case.effective_task for item in resolved] == [
        "text_continuation", "conditional_text_generation",
    ]
    assert all(item.case.worker_request()["expected_task"] == "text_continuation" for item in resolved)


def test_local_additional_profiles_keep_original_expansion(repository):
    path = repository.families / "sample/tests/manifests/sample.json"
    manifest = json.loads(path.read_text())
    manifest.update(name="sample-alternate", testcases=[{"name": "sample-alternate", "prompt": "Hello"}])
    path.with_name("alternate.json").write_text(json.dumps(manifest))
    _write_suite(repository.owned_suite, [_entry("sample", "sample-model", script=True)],
                 additional_profiles=[{"inherit": "sample.generate", "model": "sample-alternate"}])
    rows = perf.load_suite(repository.canonical)[1]
    assert rows[-1]["id"] == "sample.generate@sample-alternate"
    assert rows[-1]["workload"]["testcase"] == "sample-alternate"
    assert rows[-1]["measurement"] == {"warmup": 2, "iterations": 3}


@pytest.mark.parametrize("entries,extra,match", [
    ([_entry("sample", "sample-model", script=True, entry_id="other.generate")], {}, "duplicate"),
    ([_entry("sample", "sample-model", script=True)] * 2, {}, "duplicate"),
    ([_entry("other", "other-model")], {}, "another family"),
    ([_entry("sample", "other-model", script=True)], {}, "another family's manifest"),
    ([_entry("sample", "missing-model", script=True)], {}, "owned model/testcase"),
    ([_entry("sample", "sample-model", script=True)], {"excluded_profiles": []}, "exclusions"),
    ([_entry("sample", "sample-model", script=True)],
     {"additional_profiles": [{"inherit": "other.generate", "model": "sample-model"}]},
     "inherits unknown"),
])
def test_family_suite_rejects_foreign_duplicate_or_policy_changes(repository, entries, extra, match):
    _write_suite(repository.owned_suite, entries, **extra)
    with pytest.raises(perf.PerfMatrixError, match=match):
        perf.load_suite(repository.canonical)


def test_direct_family_suite_still_enforces_ownership(repository):
    _write_suite(repository.owned_suite, [_entry("other", "other-model")])
    with pytest.raises(perf.PerfMatrixError, match="another family"):
        perf.load_suite(repository.owned_suite)


def test_owned_manifest_cannot_be_outside_family(repository, tmp_path):
    manifest = json.loads((repository.families / "sample/tests/manifests/sample.json").read_text())
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(manifest))
    entry = _entry("sample", str(outside), script=True)
    entry["workload"]["testcase"] = "sample-model"
    _write_suite(repository.owned_suite, [entry])
    with pytest.raises(perf.PerfMatrixError, match="another family's manifest"):
        perf.load_suite(repository.canonical)


def test_family_testcase_must_exist(repository):
    entry = _entry("sample", "sample-model", script=True)
    entry["workload"]["testcase"] = "missing"
    _write_suite(repository.owned_suite, [entry])
    with pytest.raises(perf.PerfMatrixError, match="owned model/testcase"):
        perf.load_suite(repository.canonical)


@pytest.mark.parametrize("script", ["", "../outside.py", "tests/../script.py", "/tmp/script.py",
                                   "tests\\script.py", "tests/reference.sh", "tests/missing.py", None])
def test_reference_script_must_be_existing_owned_python_file(repository, script):
    entry = _entry("sample", "sample-model", script=True)
    entry["baseline"]["script"] = script
    _write_suite(repository.owned_suite, [entry])
    with pytest.raises(perf.PerfMatrixError, match="script"):
        perf.load_suite(repository.canonical)


@pytest.mark.parametrize("kind", ["file", "directory", "suite", "owner"])
def test_family_script_and_suite_reject_symlinks(repository, kind):
    if kind == "owner":
        family = repository.families / "sample"
        target = repository.root / "sample-target"
        family.rename(target)
        family.symlink_to(target, target_is_directory=True)
    elif kind == "directory":
        tests = repository.families / "sample/tests"
        target = tests.with_name("real-tests")
        tests.rename(target)
        tests.symlink_to(target, target_is_directory=True)
    else:
        path = repository.owned_suite if kind == "suite" else repository.script
        target = path.with_name("actual-" + path.name)
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(perf.PerfMatrixError, match="symlinks"):
        perf.load_suite(repository.canonical)


@pytest.mark.parametrize("patch", [{"adapter": "hf-transformers-embedding"}, {"runner": "hf-transformers"}])
def test_script_and_adapter_are_exclusive(repository, patch):
    entry = _entry("sample", "sample-model", script=True)
    entry["baseline"].update(patch)
    _write_suite(repository.owned_suite, [entry])
    with pytest.raises(perf.PerfMatrixError, match="script requires task-reference without adapter"):
        perf.load_suite(repository.canonical)


@pytest.mark.parametrize("field", ["schema_version", "status", "model", "family", "operation",
                                  "case_name", "selected_task", "precision", "mode"])
def test_reference_result_identity_must_match_execution(repository, field):
    entry = _resolved(repository)
    result = _result(entry)
    result[field] = "wrong"
    with pytest.raises(perf.PerfMatrixError, match=field):
        perf._validate_script_result(entry, repository.environment, result)


@pytest.mark.parametrize("samples", [[], [1.0], [1.0] * 4, [1.0, 0.0, 2.0],
                                    [1.0, -1.0, 2.0], [True, 1.0, 2.0],
                                    [float("inf"), 1.0, 2.0], [float("nan"), 1.0, 2.0],
                                    ["1", 1.0, 2.0], None])
def test_reference_requires_actual_positive_sample_count(repository, samples):
    entry = _resolved(repository)
    result = {**_result(entry), "samples_ms": samples}
    with pytest.raises(perf.PerfMatrixError, match="one finite positive sample"):
        perf._validate_script_result(entry, repository.environment, result)


@pytest.mark.parametrize("field", ["warmup", "iterations"])
@pytest.mark.parametrize("value", [True, 0, "3", None])
def test_reference_measurement_cannot_drift(repository, field, value):
    entry = _resolved(repository)
    result = _result(entry)
    result["measurement"][field] = value
    with pytest.raises(perf.PerfMatrixError, match=field):
        perf._validate_script_result(entry, repository.environment, result)


@pytest.mark.parametrize("field", ["timing_scope", "input_preparation_included", "asset_loading_included"])
@pytest.mark.parametrize("location", ["top", "policy"])
def test_reference_timing_must_be_reported_consistently(repository, field, location):
    entry = _resolved(repository)
    result = _result(entry)
    target = result if location == "top" else result["measurement_policy"]
    target[field] = "task-pipeline-call-wall" if field == "timing_scope" else True
    with pytest.raises(perf.PerfMatrixError, match=field):
        perf._validate_script_result(entry, repository.environment, result)


def test_reference_does_not_rewrite_wrong_median_or_missing_output(repository):
    entry = _resolved(repository)
    result = _result(entry)
    result["metrics"]["latency_ms"]["p50"] = 100.0
    with pytest.raises(perf.PerfMatrixError, match="does not match its samples"):
        perf._validate_script_result(entry, repository.environment, result)
    assert result["metrics"]["latency_ms"]["p50"] == 100.0
    result = {**_result(entry), "output_summary": None}
    with pytest.raises(perf.PerfMatrixError, match="no output summary"):
        perf._validate_script_result(entry, repository.environment, result)


def test_reference_selected_task_and_token_ids_change_actual_output(repository, tmp_path):
    entry = _resolved(repository)
    case = replace(entry.case, selected_task="conditional_text_generation", request={"token_ids": [8, 9]})
    entry = replace(entry, case=case)
    output = tmp_path / "reference.json"
    command = perf.baseline_command(entry, repository.environment, output)
    payload = json.loads(command[command.index("--request-json") + 1])
    assert payload == {"token_ids": [8, 9]}
    assert entry.case.worker_request()["expected_task"] == "text_continuation"
    process = perf.run_command(command, timeout=10, stdout_path=tmp_path / "stdout.log",
                               stderr_path=tmp_path / "stderr.log", verbose=False)
    assert process["exit_code"] == 0
    result = perf._json_file(output, "fixture reference")
    perf._validate_script_result(entry, repository.environment, result)
    assert result["output_summary"]["token_ids"] == [3]


def test_reference_script_is_revalidated_before_launch(repository):
    entry = _resolved(repository)
    repository.script.unlink()
    with pytest.raises(perf.PerfMatrixError, match="missing"):
        perf.baseline_command(entry, repository.environment, repository.root / "out.json")


def test_existing_adapter_command_keeps_existing_arguments(repository):
    entry = _resolved(repository)
    baseline = {key: value for key, value in entry.spec["baseline"].items() if key != "script"}
    baseline["adapter"] = "hf-transformers-embedding"
    entry = replace(entry, spec={**entry.spec, "baseline": baseline})
    command = perf.baseline_command(entry, repository.environment, repository.root / "out.json")
    assert command[:4] == [sys.executable, str(repository.environment.task_runner),
                           "--adapter", "hf-transformers-embedding"]
    assert "--selected-task" not in command
    assert command[command.index("--manifest") + 1] == str(entry.model.manifest_path)


@pytest.mark.parametrize("failure", [None, "process", "identity"])
def test_execute_entry_validates_actual_script_process(repository, tmp_path, monkeypatch, failure):
    entry = _resolved(repository)
    candidate = tmp_path / "candidate.py"
    cell = {
        "status": "completed", "metrics": {"latency_ms": {"p50": 2.0}},
        "samples_ms": [1.0, 2.0, 3.0], "output_summary": {"token_ids": [5]},
        "timing_scope": "public_task_call_wall", "asset_loading_included": False,
    }
    payload = {"schema_version": "trtmc.benchmark-run/v2", "cells": [cell]}
    candidate.write_text(
        '"""Synthetic candidate fixture, not model performance."""\n'
        'from pathlib import Path\nimport sys\n'
        'output = Path(sys.argv[1])\noutput.mkdir(parents=True, exist_ok=True)\n'
        f'(output / "result.json").write_text({json.dumps(payload)!r})\n'
    )
    monkeypatch.setattr(perf, "candidate_command", lambda _entry, _env, output, **_kwargs:
                        [sys.executable, str(candidate), str(output)])
    if failure == "process":
        repository.script.write_text('raise RuntimeError("fixture failure")\n')
    elif failure == "identity":
        repository.script.write_text(REFERENCE_SCRIPT.replace(
            '"case_name": args.case_name', '"case_name": "foreign-case"'))
    environment = SimpleNamespace(**vars(repository.environment), scratch_root=tmp_path / "scratch",
                                  timeout_seconds=10, hf_cache_mode="shared", bundle_retention="retain")
    if failure:
        expected = "reference command failed" if failure == "process" else "case_name"
        with pytest.raises(perf.PerfMatrixError, match=expected):
            perf._execute_entry(entry, environment, tmp_path / "run", no_build=True, verbose=False, attempt=1)
    else:
        row = perf._execute_entry(entry, environment, tmp_path / "run", no_build=True, verbose=False, attempt=1)
        assert row["reference"]["output_summary"]["loaded_models"] == 1
        assert row["reference"]["output_summary"]["invocations"] == 5
        # The existing ten-sample performance gate is intentionally unchanged.
        assert row["status"] == "white"
        assert row["comparison"]["reason"] == "timing stability requires ten valid samples per side"


def test_check_prepare_run_and_resume_share_family_suite_discovery(repository, tmp_path, monkeypatch):
    environment = repository.environment
    environment.name = "fixture-environment"
    environment.results_root = tmp_path / "results"
    environment_path = tmp_path / "environment.yaml"
    preflight_calls = []
    executions = []
    preparations = []

    def preflight(entries, env, *, require_runtime):
        preflight_calls.append(([entry["id"] for entry in entries], require_runtime))
        return perf.resolve_entries(entries, env)

    def run_rows(directory, result, entries, env, **kwargs):
        executions.append([entry.spec["id"] for entry in entries])
        return 0

    def prepare(entries, env, output, **kwargs):
        preparations.append([entry.spec["id"] for entry in entries])
        return 0

    monkeypatch.setattr(perf, "load_environment", lambda _path: environment)
    monkeypatch.setattr(perf, "preflight", preflight)
    monkeypatch.setattr(perf, "_run_rows", run_rows)
    monkeypatch.setattr(perf, "prepare_entries", prepare)
    common = [str(repository.canonical), "--environment", str(environment_path)]
    assert perf.main(["check", *common]) == 0
    assert perf.main(["prepare", *common, "--output", str(tmp_path / "prepared.json")]) == 0
    assert perf.main(["run", *common]) == 0
    run_directory, = environment.results_root.iterdir()
    assert perf.main(["resume", str(run_directory)]) == 0
    expected = ["other.generate", "sample.generate"]
    assert preflight_calls == [(expected, True), (expected, False), (expected, True), (expected, True)]
    assert preparations == [expected]
    assert executions == [expected, expected]
    # A disappeared owned entry cannot silently vanish during resume.
    repository.owned_suite.unlink()
    assert perf.main(["resume", str(run_directory)]) == 2
