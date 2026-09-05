# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

import tools.perf_matrix as perf
from apps.benchmark.performance.baselines import (
    lance_reference,
    sana_wm_reference,
    task_reference,
)


REPO = Path(__file__).resolve().parents[4]
SUITE = REPO / "apps/benchmark/performance/release.yaml"


def _environment(tmp_path: Path) -> tuple[Path, perf.Environment]:
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("bench", "worker", "hf.py", "task.py"):
        path = tools / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in (
        "libtrtmc_runtime.so",
        "libtrtmc_backend_trt.so",
        "libtrtmc_model_gpt2.so",
        "libtrtmc_model_lance.so",
    ):
        (runtime / name).write_bytes(b"")
    references = {}
    for name in perf.REFERENCE_FIELDS:
        path = tmp_path / name
        path.mkdir()
        references[name] = str(path)
    value = {
        "schema_version": perf.ENVIRONMENT_SCHEMA,
        "name": "test",
        "tools": {
            "trtmc_bench": str(tools / "bench"),
            "trtmc_worker": str(tools / "worker"),
            "hf_transformers_runner": str(tools / "hf.py"),
            "task_reference_runner": str(tools / "task.py"),
        },
        "references": references,
        "storage": {
            "results_root": str(tmp_path / "results"),
            "scratch_root": str(tmp_path / "scratch"),
            "bundle_cache": str(tmp_path / "bundles"),
            "bundle_roots": [],
            "runtime_root": str(runtime),
            "bundle_retention": "retain",
        },
        "execution": {"local_files_only": True, "timeout_seconds": 10},
    }
    path = tmp_path / "environment.yaml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path, perf.load_environment(path)


def _fake_measurement_runner(
    environment,
    entry,
    *,
    candidate_samples=(),
    candidate_tokens=(),
    candidate_exit_codes=(),
    record_bundle=False,
):
    state = {"candidate_runs": 0, "commands": [], "environments": []}

    def run_command(arguments, *, stdout_path, stderr_path, env=None, **_kwargs):
        state["commands"].append(list(arguments))
        state["environments"].append(dict(env or {}))
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        output = Path(arguments[arguments.index("--output") + 1])
        if Path(arguments[0]) == environment.trtmc_bench:
            index = state["candidate_runs"]
            state["candidate_runs"] += 1
            exit_code = candidate_exit_codes[index] if index < len(candidate_exit_codes) else 0
            if exit_code:
                return {"argv": list(arguments), "exit_code": exit_code}
            samples = candidate_samples[index] if index < len(candidate_samples) else [10.0] * 10
            tokens = candidate_tokens[index] if index < len(candidate_tokens) else [1, 2]
            bundles = (
                [{"model": entry.model.name, "bundle": str(entry.case.bundle_path)}]
                if record_bundle
                else []
            )
            output.mkdir(parents=True)
            (output / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": "trtmc.benchmark-run/v2",
                        "status": "completed",
                        "preparation": {"bundles": bundles},
                        "cells": [
                            {
                                "status": "completed",
                                "metrics": {"latency_ms": {"p50": float(np.median(samples))}},
                                "samples_ms": samples,
                                "output_summary": {
                                    "token_ids": tokens,
                                    "output_tokens": len(tokens),
                                },
                                "timing_scope": "public_task_call_wall",
                                "asset_loading_included": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        else:
            output.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "precision": entry.reference_precision,
                        "metrics": {"latency_ms": {"p50": 10.1}},
                        "samples_ms": [10.1] * 10,
                        "output_summary": {"token_ids": [1, 2], "output_tokens": 2},
                        "measurement_policy": dict(entry.baseline_timing),
                    }
                ),
                encoding="utf-8",
            )
        return {
            "argv": list(arguments),
            "exit_code": 0,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    return state, run_command


def test_environment_enforces_storage_root_and_per_entry_cache_policy(tmp_path: Path) -> None:
    environment_path, _ = _environment(tmp_path)
    value = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    storage_root = tmp_path / "managed"
    storage_root.mkdir()
    value["storage"]["storage_root"] = str(storage_root)
    value["execution"].update({"hf_cache_mode": "per_entry", "hf_cache_retention": "delete_always"})
    environment_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    environment = perf.load_environment(environment_path)

    assert environment.storage_root == storage_root
    assert environment.hf_cache_mode == "per_entry"
    assert environment.hf_cache_retention == "delete_always"
    with pytest.raises(perf.PerfMatrixError, match="results_root must stay below storage_root"):
        perf.preflight((), environment, require_runtime=False)


def test_per_entry_hf_cache_is_private_and_follows_retention(tmp_path: Path, monkeypatch) -> None:
    environment_path, _ = _environment(tmp_path)
    value = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    value["execution"].update(
        {"hf_cache_mode": "per_entry", "hf_cache_retention": "delete_on_pass"}
    )
    environment_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    environment = perf.load_environment(environment_path)
    monkeypatch.setenv("HF_HUB_CACHE", "/shared/hub")
    monkeypatch.setenv("HF_MODULES_CACHE", "/shared/modules")
    monkeypatch.setenv("TRANSFORMERS_CACHE", "/shared/transformers")
    work = environment.scratch_root / "entry" / "attempt-1"
    (work / "hf-cache").mkdir(parents=True)

    command_environment = perf._entry_command_environment(environment, work)
    assert command_environment["HF_HOME"] == str((work / "hf-cache").resolve())
    assert "HF_HUB_CACHE" not in command_environment
    assert "HF_MODULES_CACHE" not in command_environment
    assert "TRANSFORMERS_CACHE" not in command_environment
    assert perf._cleanup_entry_work(work, environment, passed=False)["status"] == "retained"
    assert perf._cleanup_entry_work(work, environment, passed=True)["status"] == "deleted"
    assert not work.exists()


def test_shared_hf_cache_cannot_be_deleted(tmp_path: Path) -> None:
    environment_path, _ = _environment(tmp_path)
    value = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    value["execution"].update({"hf_cache_mode": "shared", "hf_cache_retention": "delete_always"})
    environment_path.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(perf.PerfMatrixError, match="shared Hugging Face cache"):
        perf.load_environment(environment_path)


def test_checked_in_environments_have_no_dead_gpu_headroom_setting() -> None:
    root = REPO / "apps/benchmark/performance/environments"
    for path in root.glob("*.yaml"):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "minimum_gpu_free_fraction" not in value["execution"], path


def test_explicit_empty_model_selection_fails_closed(tmp_path: Path) -> None:
    selection = tmp_path / "selection.json"
    selection.write_text('{"families": []}\n', encoding="utf-8")

    with pytest.raises(perf.PerfMatrixError, match="matches no release entries"):
        perf.select_entries(
            [{"id": "a", "family": "alpha", "model": "model-a"}], model_selection=selection
        )


@pytest.mark.parametrize("entry_id", (".", ".."))
def test_entry_slug_cannot_escape_its_root(entry_id: str) -> None:
    assert perf._entry_slug(entry_id) == "entry"


def test_release_suite_expands_profiles_and_covers_ready_catalog() -> None:
    name, entries, excluded = perf.load_suite(SUITE)
    assert name == "release-family-performance"
    profile = next(entry for entry in entries if entry["id"] == "gpt2.generate@gpt2-125m")
    assert profile["workload"]["testcase"] == "gpt2-125m"
    vision_ids = {
        "timm_densenet.classify",
        "timm_efficientnet.classify",
        "timm_inception.classify",
        "timm_mnasnet.classify",
        "timm_mobilenetv3.classify",
        "timm_repvgg.classify",
        "timm_resnet.classify",
        "timm_vgg.classify",
        "timm_vit.classify",
    }
    vision_entries = {entry["id"]: entry for entry in entries if entry["id"] in vision_ids}
    assert set(vision_entries) == vision_ids
    assert all(
        entry["baseline"]["adapter"] == "hf-transformers-vision"
        for entry in vision_entries.values()
    )
    perf._coverage(entries, excluded)


@pytest.mark.parametrize(
    (
        "family",
        "expected_scope",
        "input_preparation_included",
        "calls_after_load",
        "calls_after_invoke",
    ),
    [
        ("bert", "task-pipeline-call-wall", True, [], ["tokenize", "model"]),
        ("eagle_vlm", "task-model-call-wall", False, ["tokenize"], ["tokenize", "model"]),
    ],
)
def test_embedding_reference_measures_the_family_timing_contract(
    monkeypatch,
    family,
    expected_scope,
    input_preparation_included,
    calls_after_load,
    calls_after_invoke,
) -> None:
    calls: list[str] = []

    class FakeTensor:
        shape = (1, 2)
        dtype = "fp32"

        def to(self, *_args, **_kwargs):
            return self

        def unsqueeze(self, _dimension):
            return self

        def sum(self, **_kwargs):
            return self

        def clamp(self, **_kwargs):
            return self

        def numel(self):
            return 2

        def isfinite(self):
            return self

        def all(self):
            return self

        def item(self):
            return True

        def __mul__(self, _other):
            return self

        def __truediv__(self, _other):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, *_args, **_kwargs):
            calls.append("tokenize")
            return {"input_ids": FakeTensor(), "attention_mask": FakeTensor()}

    class FakeModel:
        config = SimpleNamespace(_commit_hash="model-revision")

        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def eval(self):
            return self

        def to(self, *_args, **_kwargs):
            return self

        def __call__(self, **_kwargs):
            calls.append("model")
            return SimpleNamespace(last_hidden_state=FakeTensor())

    fake_torch = ModuleType("torch")
    fake_torch.device = lambda value: value
    fake_torch.float16 = "fp16"
    fake_torch.float32 = "fp32"
    fake_torch.bfloat16 = "bf16"
    fake_torch.inference_mode = nullcontext
    fake_torch.ones = lambda *_args, **_kwargs: FakeTensor()
    fake_torch.nn = SimpleNamespace(
        functional=SimpleNamespace(normalize=lambda value, **_kwargs: value)
    )
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoModel = FakeModel
    fake_transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    arguments = SimpleNamespace(
        family=family,
        model="sentence-transformers/all-MiniLM-L6-v2",
        precision="fp32",
        revision="model-revision",
        trust_remote_code=False,
        local_files_only=True,
    )

    session = task_reference.LOADERS["hf-transformers-embedding"](
        arguments,
        {"prompt": "The quick brown fox"},
        {},
    )

    assert calls == calls_after_load
    assert session.timing_scope == expected_scope
    assert session.input_preparation_included is input_preparation_included
    assert session.asset_loading_included is False
    assert session.invoke()["embedding_vectors"] == 1
    assert calls == calls_after_invoke


def test_check_resolves_selected_entry_with_one_runtime_root(tmp_path: Path, capsys) -> None:
    environment_path, _ = _environment(tmp_path)
    assert (
        perf.main(
            [
                "check",
                str(SUITE),
                "--environment",
                str(environment_path),
                "--entry",
                "gpt2.generate",
            ]
        )
        == 0
    )
    assert "Ready: 1" in capsys.readouterr().out


def test_candidate_and_reference_commands_use_current_contract(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "gpt2.generate"]
    resolved = perf.resolve_entries(selected, environment)[0]
    candidate = perf.candidate_command(resolved, environment, tmp_path / "candidate", no_build=True)
    assert "--runtime-root" in candidate
    assert "--operation" in candidate
    assert "--no-build" in candidate
    reference = perf.baseline_command(resolved, environment, tmp_path / "reference.json")
    assert "--case-name" in reference
    assert "--task" in reference
    assert ("--revision" in reference) is bool(resolved.model.hf_revision)


def test_lerobot_reference_is_family_owned_and_has_a_closed_contract(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    source = Path(environment.references["lerobot_repo"])
    entrypoint = source / "lerobot/common/policies/act/modeling_act.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.write_text("", encoding="utf-8")
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "lerobot_act.control"]
    resolved = perf.resolve_entries(selected, environment)[0]
    command = perf.baseline_command(resolved, environment, tmp_path / "reference.json")
    parsed = task_reference.build_parser().parse_args(command[2:])
    assert parsed.adapter == "pytorch-lerobot-act"
    assert json.loads(parsed.adapter_options_json) == {"source_root": str(source)}

    candidate = {
        "action_steps": 100,
        "action_dim": 14,
        "action_values": 1400,
        "within_training_bounds": True,
    }
    reference = {
        "action_steps": 100,
        "action_dim": 14,
        "action_values": 1400,
        "finite": True,
    }
    assert perf._output_contract(
        resolved,
        {"output_summary": candidate},
        {"output_summary": reference},
    ) == (True, "", None)


def test_comparison_preserves_output_gate_and_three_performance_states(
    tmp_path: Path,
) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "gpt2.generate"]
    entry = perf.resolve_entries(selected, environment)[0]

    def value(candidate_ms: float, reference_ms: float, tokens: list[int]):
        candidate = {
            "metrics": {"latency_ms": {"p50": candidate_ms}},
            "output_summary": {"token_ids": tokens, "output_tokens": len(tokens)},
            "timing_scope": "public_task_call_wall",
            "asset_loading_included": False,
        }
        reference = {
            "status": "completed",
            "precision": entry.reference_precision,
            "metrics": {"latency_ms": {"p50": reference_ms}},
            "output_summary": {"token_ids": tokens, "output_tokens": len(tokens)},
            "measurement_policy": dict(entry.baseline_timing),
        }
        return candidate, reference

    candidate, reference = value(10.0, 12.0, [1, 2])
    assert perf.compare(entry, candidate, reference)[0] == "green"
    candidate, reference = value(10.0, 10.2, [1, 2])
    assert perf.compare(entry, candidate, reference)[0] == "yellow"
    candidate, reference = value(12.0, 10.0, [1, 2])
    assert perf.compare(entry, candidate, reference)[0] == "red"
    reference["output_summary"]["token_ids"] = [9]
    assert perf.compare(entry, candidate, reference)[0] == "contract-mismatch"


@pytest.mark.parametrize(
    ("samples", "status"),
    (
        ([100.0, 101.0, 99.0, 100.0, 100.0, 101.0, 100.0, 99.0, 100.0, 100.0], "stable"),
        ([3.7, 3.4, 3.0, 2.7, 2.3, 1.9, 1.6, 1.4, 1.2, 1.0], "unstable"),
        ([10.0, 11.0], "not_evaluated"),
    ),
)
def test_timing_stability_preserves_the_ten_sample_contract(samples, status) -> None:
    assert perf._timing_stability(samples)["status"] == status


@pytest.mark.parametrize(
    ("second_samples", "expected_status", "stability_status"),
    (([10.0] * 10, "yellow", "stable_after_retry"), (None, "white", "measurement_inconclusive")),
)
def test_unstable_measurement_is_retried_once(
    tmp_path: Path,
    monkeypatch,
    second_samples,
    expected_status,
    stability_status,
) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(entry for entry in entries if entry["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    falling = [3.7, 3.4, 3.0, 2.7, 2.3, 1.9, 1.6, 1.4, 1.2, 1.0]
    second = falling if second_samples is None else second_samples
    state, run_command = _fake_measurement_runner(
        environment,
        entry,
        candidate_samples=(falling, second),
    )
    monkeypatch.setattr(perf, "run_command", run_command)
    row = perf._execute_entry(
        entry,
        environment,
        tmp_path / "run",
        no_build=True,
        verbose=False,
        attempt=1,
    )

    assert len(state["commands"]) == 4
    assert row["status"] == expected_status
    assert row["measurement_stability"]["status"] == stability_status
    assert set(row["commands"]) == {
        "candidate",
        "reference",
        "candidate_measurement_2",
        "reference_measurement_2",
    }


def test_scratch_is_run_scoped_and_success_cleans_all_entry_attempts(
    tmp_path: Path, monkeypatch
) -> None:
    _, environment = _environment(tmp_path)
    environment = replace(
        environment,
        hf_cache_mode="per_entry",
        hf_cache_retention="delete_on_pass",
    )
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    run = tmp_path / "run-a"
    entry_work = environment.scratch_root / "run-a" / "gpt2.generate"
    (entry_work / "attempt-1" / "hf-cache").mkdir(parents=True)
    state, run_command = _fake_measurement_runner(environment, entry)
    monkeypatch.setattr(perf, "run_command", run_command)

    row = perf._execute_entry(
        entry,
        environment,
        run,
        no_build=True,
        verbose=False,
        attempt=2,
    )

    assert row["status"] == "yellow"
    assert not entry_work.exists()
    expected_cache = str((entry_work / "attempt-2" / "hf-cache").resolve())
    assert {value["HF_HOME"] for value in state["environments"]} == {expected_cache}


def test_existing_artifact_attempt_is_skipped_in_one_execution(tmp_path: Path, monkeypatch) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    run = tmp_path / "run"
    (run / "artifacts" / "gpt2.generate" / "attempt-1").mkdir(parents=True)
    _, run_command = _fake_measurement_runner(environment, entry)
    monkeypatch.setattr(perf, "run_command", run_command)

    row = perf._execute_entry(
        entry,
        environment,
        run,
        no_build=True,
        verbose=False,
        attempt=1,
    )

    assert row["attempts"] == 2
    assert row["artifact_dir"] == "artifacts/gpt2.generate/attempt-2"


def test_failed_command_records_the_scanned_artifact_attempt(tmp_path: Path, monkeypatch) -> None:
    entry = SimpleNamespace(
        spec={"id": "first", "operation": "generate"},
        model=SimpleNamespace(name="model", family="family"),
        case=SimpleNamespace(testcase_name="case"),
    )
    run = tmp_path / "run"
    artifact_root = run / "artifacts" / "first"
    (artifact_root / "attempt-1").mkdir(parents=True)
    attempts = []

    def execute(_entry, _environment, _run, *, attempt, **_kwargs):
        attempts.append(attempt)
        (artifact_root / f"attempt-{attempt}").mkdir(exist_ok=True)
        raise perf.PerfMatrixError("candidate command failed")

    results = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "running",
        "selected_entry_ids": ["first"],
        "rows": [],
    }
    monkeypatch.setattr(perf, "_execute_entry", execute)
    monkeypatch.setattr(perf, "_write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(perf, "write_report", lambda *_args, **_kwargs: {})

    assert (
        perf._run_rows(
            run,
            results,
            (entry,),
            SimpleNamespace(),
            no_build=True,
            verbose=False,
        )
        == 1
    )
    assert results["rows"][0]["attempts"] == 2

    assert (
        perf._run_rows(
            run,
            results,
            (entry,),
            SimpleNamespace(),
            no_build=True,
            verbose=False,
        )
        == 1
    )
    assert attempts == [2, 3]
    assert results["rows"][0]["attempts"] == 3


def test_second_measurement_contract_mismatch_discards_first_stability(
    tmp_path: Path, monkeypatch
) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    falling = [3.7, 3.4, 3.0, 2.7, 2.3, 1.9, 1.6, 1.4, 1.2, 1.0]
    _, run_command = _fake_measurement_runner(
        environment,
        entry,
        candidate_samples=(falling, [10.0] * 10),
        candidate_tokens=([1, 2], [9]),
    )
    monkeypatch.setattr(perf, "run_command", run_command)

    row = perf._execute_entry(
        entry,
        environment,
        tmp_path / "run",
        no_build=True,
        verbose=False,
        attempt=1,
    )

    assert row["status"] == "contract-mismatch"
    assert "measurement_stability" not in row


def test_failed_remeasurement_is_not_treated_as_a_pass(tmp_path: Path, monkeypatch) -> None:
    _, environment = _environment(tmp_path)
    environment = replace(environment, bundle_retention="delete_on_pass")
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    entry.case.bundle_path.parent.mkdir(parents=True)
    entry.case.bundle_path.write_bytes(b"bundle")
    falling = [3.7, 3.4, 3.0, 2.7, 2.3, 1.9, 1.6, 1.4, 1.2, 1.0]
    _, run_command = _fake_measurement_runner(
        environment,
        entry,
        candidate_samples=(falling,),
        candidate_exit_codes=(0, 1),
        record_bundle=True,
    )
    monkeypatch.setattr(perf, "run_command", run_command)

    with pytest.raises(perf.PerfMatrixError, match="candidate command failed"):
        perf._execute_entry(
            entry,
            environment,
            tmp_path / "run",
            no_build=True,
            verbose=False,
            attempt=1,
        )

    assert entry.case.bundle_path.is_file()


def test_delete_always_cleans_declared_bundle_when_candidate_process_fails(
    tmp_path: Path, monkeypatch
) -> None:
    _, environment = _environment(tmp_path)
    environment = replace(environment, bundle_retention="delete_always")
    _, entries, _ = perf.load_suite(SUITE)
    spec = next(value for value in entries if value["id"] == "gpt2.generate")
    entry = perf.resolve_entries((spec,), environment)[0]
    entry.case.bundle_path.parent.mkdir(parents=True)
    entry.case.bundle_path.write_bytes(b"bundle")
    _, run_command = _fake_measurement_runner(
        environment,
        entry,
        candidate_exit_codes=(1,),
    )
    monkeypatch.setattr(perf, "run_command", run_command)

    with pytest.raises(perf.PerfMatrixError, match="candidate command failed"):
        perf._execute_entry(
            entry,
            environment,
            tmp_path / "run",
            no_build=True,
            verbose=False,
            attempt=1,
        )

    assert not entry.case.bundle_path.exists()

    external_bundle = tmp_path / "external.bundle"
    external_bundle.write_bytes(b"external")
    external_entry = replace(
        entry,
        case=entry.case.with_values(bundle_path=external_bundle),
    )
    _, run_command = _fake_measurement_runner(
        environment,
        external_entry,
        candidate_exit_codes=(1,),
    )
    monkeypatch.setattr(perf, "run_command", run_command)
    with pytest.raises(perf.PerfMatrixError, match="candidate command failed"):
        perf._execute_entry(
            external_entry,
            environment,
            tmp_path / "external-run",
            no_build=True,
            verbose=False,
            attempt=1,
        )
    assert external_bundle.is_file()


def test_prepare_aggregates_public_builder_receipts(tmp_path: Path, monkeypatch) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "gpt2.generate"]
    entry = perf.resolve_entries(selected, environment)[0]

    def run_command(arguments, *, stdout_path, stderr_path, **_kwargs):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(
            json.dumps(
                {
                    "bundles": [
                        {
                            "model": "distilgpt2",
                            "bundle": str(tmp_path / "distilgpt2.bundle"),
                            "status": "built",
                            "included_in_performance_metrics": False,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return {
            "argv": list(arguments),
            "exit_code": 0,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    monkeypatch.setattr(perf, "run_command", run_command)
    output = tmp_path / "preparation.json"
    assert perf.prepare_entries((entry,), environment, output, verbose=False) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == perf.PREPARATION_SCHEMA
    assert len(receipt["bundles"]) == 1


def test_report_uses_selected_ids_and_shows_pending_and_stability(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    results = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "completed",
        "suite": "test",
        "environment": "test",
        "selected_entry_ids": ["gpt2.generate", "pending.generate"],
        "rows": [
            {
                "id": "gpt2.generate",
                "model": "distilgpt2",
                "operation": "generate",
                "status": "yellow",
                "measurement_stability": {"status": "stable_after_retry"},
                "comparison": {
                    "candidate_p50_ms": 1.0,
                    "reference_p50_ms": 1.01,
                },
            }
        ],
    }
    report = perf.write_report(run, results)
    assert report["summary"]["comparable"] == 1
    assert report["summary"]["selected"] == 2
    assert report["summary"]["pending"] == 1
    assert (run / "report.json").is_file()
    html = (run / "report.html").read_text(encoding="utf-8")
    assert "pending: 1" in html
    assert "stable_after_retry" in html


def test_multi_entry_progress_publishes_only_completed_rows(tmp_path: Path, monkeypatch) -> None:
    entries = tuple(
        SimpleNamespace(
            spec={"id": entry_id, "operation": "generate"},
            model=SimpleNamespace(name=entry_id, family="gpt2"),
            case=SimpleNamespace(testcase_name=entry_id),
        )
        for entry_id in ("first", "second")
    )
    results = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "running",
        "selected_entry_ids": ["first", "second"],
        "rows": [],
    }
    snapshots = []

    def execute(entry, *_args, attempt, **_kwargs):
        return {"id": entry.spec["id"], "status": "green", "attempts": attempt}

    def write_json(path, value):
        if path.name == "results.json":
            snapshots.append([row["id"] for row in value["rows"]])

    monkeypatch.setattr(perf, "_execute_entry", execute)
    monkeypatch.setattr(perf, "_write_json", write_json)
    monkeypatch.setattr(perf, "write_report", lambda *_args, **_kwargs: {})

    assert (
        perf._run_rows(
            tmp_path / "run",
            results,
            entries,
            SimpleNamespace(),
            no_build=True,
            verbose=False,
        )
        == 0
    )
    assert snapshots[:2] == [["first"], ["first", "second"]]


def test_contract_mismatch_is_finished_but_keeps_run_non_green(tmp_path: Path, monkeypatch) -> None:
    entry = SimpleNamespace(spec={"id": "first"})
    results = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "running",
        "selected_entry_ids": ["first"],
        "rows": [{"id": "first", "status": "contract-mismatch", "attempts": 1}],
    }
    monkeypatch.setattr(
        perf,
        "_execute_entry",
        lambda *_args, **_kwargs: pytest.fail("finished contract mismatch was rerun"),
    )
    monkeypatch.setattr(perf, "_write_json", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(perf, "write_report", lambda *_args, **_kwargs: {})

    assert (
        perf._run_rows(
            tmp_path / "run",
            results,
            (entry,),
            SimpleNamespace(),
            no_build=True,
            verbose=False,
        )
        == 1
    )
    assert results["status"] == "failed"


@pytest.mark.parametrize("stored_ids", (None, ["removed.entry"]))
def test_resume_fails_when_stored_selection_is_missing(tmp_path: Path, capsys, stored_ids) -> None:
    environment_path, _ = _environment(tmp_path)
    run = tmp_path / "resume-run"
    run.mkdir()
    results = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "failed",
        "suite_path": str(SUITE),
        "environment_path": str(environment_path),
        "rows": [],
    }
    if stored_ids is not None:
        results["selected_entry_ids"] = stored_ids
    (run / "results.json").write_text(json.dumps(results), encoding="utf-8")

    assert perf.main(["resume", str(run)]) == 2
    expected = (
        "matrix results has no selected entry IDs"
        if stored_ids is None
        else "selected entries are missing from the suite: removed.entry"
    )
    assert expected in capsys.readouterr().err


def test_run_executes_candidate_then_reference_and_publishes_report(
    tmp_path: Path, monkeypatch
) -> None:
    environment_path, environment = _environment(tmp_path)

    def run_command(arguments, *, stdout_path, stderr_path, **_kwargs):
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        output = Path(arguments[arguments.index("--output") + 1])
        if Path(arguments[0]) == environment.trtmc_bench:
            output.mkdir(parents=True)
            (output / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": "trtmc.benchmark-run/v2",
                        "status": "completed",
                        "preparation": {"bundles": []},
                        "cells": [
                            {
                                "status": "completed",
                                "metrics": {"latency_ms": {"p50": 10.0}},
                                "samples_ms": [10.0] * 10,
                                "output_summary": {
                                    "token_ids": [1, 2],
                                    "output_tokens": 2,
                                },
                                "timing_scope": "public_task_call_wall",
                                "asset_loading_included": False,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
        else:
            output.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "precision": "fp32",
                        "metrics": {"latency_ms": {"p50": 10.1}},
                        "samples_ms": [10.1] * 10,
                        "output_summary": {
                            "token_ids": [1, 2],
                            "output_tokens": 2,
                        },
                        "measurement_policy": {
                            "timing_scope": "public_operation_call_wall",
                            "input_preparation_included": True,
                            "asset_loading_included": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
        return {
            "argv": list(arguments),
            "exit_code": 0,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }

    monkeypatch.setattr(perf, "run_command", run_command)
    assert (
        perf.main(
            [
                "run",
                str(SUITE),
                "--environment",
                str(environment_path),
                "--entry",
                "gpt2.generate",
            ]
        )
        == 0
    )
    runs = list((tmp_path / "results").iterdir())
    assert len(runs) == 1
    result = json.loads((runs[0] / "results.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["rows"][0]["status"] == "yellow"
    assert (runs[0] / "report.json").is_file()


def test_reference_runner_dependencies_are_baseline_owned() -> None:
    root = REPO / "apps/benchmark/performance/baselines"
    required = {
        "audio_reference.py",
        "elf_reference.py",
        "lance_reference.py",
        "reference_support.py",
        "sana_wm_reference.py",
    }
    assert all((root / name).is_file() for name in required)
    source = (root / "task_reference.py").read_text(encoding="utf-8")
    assert "from tools." not in source
    assert 'REPOSITORY / "tools/' not in source
    assert "tests/e2e" not in source
    assert "tensorrt_model_connect.families" not in source


def test_lance_command_requires_explicit_checkout_without_a_commit_gate(
    tmp_path: Path,
) -> None:
    _, environment = _environment(tmp_path)
    checkout = Path(environment.references["lance_repo"])
    (checkout / "inference_lance.py").write_text("", encoding="utf-8")
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "lance.generate"]
    resolved = perf.resolve_entries(selected, environment)[0]
    command = perf.baseline_command(resolved, environment, tmp_path / "reference.json")
    parsed = task_reference.build_parser().parse_args(command[2:])
    options = json.loads(parsed.adapter_options_json)
    assert parsed.adapter == "upstream-lance"
    assert options == {
        "model_subdir": "Lance_3B",
        "reference_repo": str(checkout),
        "resolution": "image_768res",
        "vit_subdir": "Qwen2.5-VL-ViT",
    }

    direct = lance_reference.build_parser().parse_args(
        [
            "--reference-repo",
            str(checkout),
            "--model",
            "model",
            "--image",
            str(tmp_path / "image.png"),
            "--prompt",
            "describe",
            "--max-new-tokens",
            "4",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--output",
            str(tmp_path / "output.json"),
        ]
    )
    assert direct.reference_repo == checkout


def test_check_fails_fast_when_selected_reference_input_is_missing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    environment_path, _ = _environment(tmp_path)
    value = yaml.safe_load(environment_path.read_text(encoding="utf-8"))
    value["references"]["lance_repo"] = "${TRTMC_TEST_UNSET_LANCE_REPO}"
    environment_path.write_text(yaml.safe_dump(value), encoding="utf-8")
    monkeypatch.delenv("TRTMC_TEST_UNSET_LANCE_REPO", raising=False)
    assert (
        perf.main(
            [
                "check",
                str(SUITE),
                "--environment",
                str(environment_path),
                "--entry",
                "lance.generate",
            ]
        )
        == 2
    )
    assert "TRTMC_TEST_UNSET_LANCE_REPO" in capsys.readouterr().err


def test_timeseries_entries_use_current_forecast_request_schema(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["operation"] == "solve"]
    assert len(selected) == 5
    resolved = perf.resolve_entries(selected, environment)
    for entry in resolved:
        assert entry.case.request["past_values"]
        assert not ({"branch_input", "field_input", "trunk_input"} & entry.case.request.keys())
    timesfm = next(entry for entry in resolved if entry.model.family == "timesfm")
    assert timesfm.case.request["frequency"] == 2
    source = (REPO / "apps/benchmark/performance/baselines/task_reference.py").read_text(
        encoding="utf-8"
    )
    assert '_numeric_values(request, "past_values")' in source
    assert '_numeric_values(request, "branch_input")' not in source
    assert '_numeric_values(request, "field_input")' not in source


def test_qwen3_omni_preserves_thinker_and_talker_limits(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "qwen3_omni.generate_audio"]
    resolved = perf.resolve_entries(selected, environment)[0]
    assert resolved.case.request["max_new_tokens"] == 16
    assert resolved.case.request["talker_max_new_tokens"] == 32
    command = perf.baseline_command(resolved, environment, tmp_path / "reference.json")
    request = json.loads(command[command.index("--request-json") + 1])
    assert request["max_new_tokens"] == 16
    assert request["talker_max_new_tokens"] == 32
    worker = (REPO / "apps/benchmark/native/benchmark_worker.cpp").read_text(encoding="utf-8")
    assert 'optional_value<std::int32_t>(request, "talker_max_new_tokens", 0)' in worker


def test_sana_reference_reports_materialized_video_shape() -> None:
    video = np.stack(
        [
            np.zeros((24, 32, 3), dtype=np.uint8),
            np.ones((24, 32, 3), dtype=np.uint8),
        ]
    )
    summary = sana_wm_reference.media_summary(video)
    assert summary == {
        "media_type": "video",
        "media_count": 2,
        "num_frames": 2,
        "height": 24,
        "width": 32,
        "channels": 3,
    }
    assert perf._media_shape(summary) == (2, 24, 32, 3)
    source = (REPO / "apps/benchmark/performance/baselines/task_reference.py").read_text(
        encoding="utf-8"
    )
    assert 'request.get("action"' in source


def test_sana_world_request_preserves_official_camera_controls(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = [entry for entry in entries if entry["id"] == "sana_wm.generate_image"]
    request = perf.resolve_entries(selected, environment)[0].case.request
    assert request["translation_speed"] == 0.055
    assert request["rotation_speed_deg"] == 1.2
    assert request["fps"] == 16
    assert request["flow_shift"] == 9.8
    assert request["no_action_overlay"] is True


def test_sana_reference_calls_official_pipeline_with_exact_workload(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {"generate": []}
    reference_repo = tmp_path / "Sana"
    reference_repo.mkdir()
    model_dir = tmp_path / "model"
    (model_dir / "dit").mkdir(parents=True)
    (model_dir / "refiner/text_encoder").mkdir(parents=True)
    (model_dir / "config.yaml").write_text("{}\n", encoding="utf-8")
    (model_dir / "dit/sana_wm_1600m_720p.safetensors").write_bytes(b"weights")
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("drive forward", encoding="utf-8")
    intrinsics = tmp_path / "intrinsics.npy"
    intrinsics.write_bytes(b"intrinsics")
    output = tmp_path / "result.json"

    class FakeImage:
        def convert(self, mode):
            captured["image_mode"] = mode
            return self

    pil = ModuleType("PIL")
    pil.Image = SimpleNamespace(open=lambda path: FakeImage())
    monkeypatch.setitem(sys.modules, "PIL", pil)

    synchronize_calls = []
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: synchronize_calls.append(True),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)

    pyrallis = ModuleType("pyrallis")

    def parse_config(**kwargs):
        captured["parse"] = kwargs
        return "config"

    pyrallis.parse = parse_config
    monkeypatch.setitem(sys.modules, "pyrallis", pyrallis)

    class RefinerSettings:
        def __init__(self, **kwargs):
            captured["refiner"] = kwargs

    class GenerationParams:
        def __init__(self, **kwargs):
            captured["generation"] = kwargs

    class Pipeline:
        def __init__(self, **kwargs):
            captured["pipeline"] = kwargs

        def generate(self, *args):
            captured["generate"].append(args)
            return {
                "video": np.zeros((321, 24, 32, 3), dtype=np.uint8),
                "c2w": "camera",
            }

    trajectory = np.zeros((321, 4, 4), dtype=np.float32)
    official = SimpleNamespace(
        InferenceConfig=object,
        RefinerSettings=RefinerSettings,
        GenerationParams=GenerationParams,
        SanaWMPipeline=Pipeline,
        action_string_to_c2w=lambda action, **kwargs: (
            captured.update(action=(action, kwargs)) or trajectory
        ),
        _snap_num_frames=lambda value, **kwargs: value,
        resize_and_center_crop=lambda value: ("cropped", (1, 1), (2, 2), (0, 0)),
        load_intrinsics=lambda path, frames: (
            captured.update(load_intrinsics=(path, frames)) or "raw-intrinsics"
        ),
        transform_intrinsics_for_crop=lambda value, *sizes: (
            captured.update(transform_intrinsics=(value, sizes)) or "intrinsics"
        ),
        apply_overlay=lambda *_: (_ for _ in ()).throw(
            AssertionError("no-action-overlay must skip overlay")
        ),
    )
    monkeypatch.setattr(sana_wm_reference, "_official_module", lambda path: official)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sana_wm_reference.py",
            "--reference-repo",
            str(reference_repo),
            "--image",
            str(image),
            "--model-dir",
            str(model_dir),
            "--prompt",
            str(prompt),
            "--action",
            "w-320",
            "--intrinsics",
            str(intrinsics),
            "--num_frames",
            "321",
            "--fps",
            "16",
            "--step",
            "60",
            "--cfg_scale",
            "5.0",
            "--flow_shift",
            "9.8",
            "--seed",
            "42",
            "--refiner_seed",
            "42",
            "--translation_speed",
            "0.055",
            "--rotation_speed_deg",
            "1.2",
            "--no_action_overlay",
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--output",
            str(output),
        ],
    )

    assert sana_wm_reference.main() == 0
    assert captured["action"] == (
        "w-320",
        {"translation_speed": 0.055, "rotation_speed_deg": 1.2},
    )
    assert captured["refiner"] == {
        "root": model_dir / "refiner",
        "gemma_root": model_dir / "refiner/text_encoder",
        "seed": 42,
    }
    assert captured["generation"] == {
        "num_frames": 321,
        "fps": 16,
        "step": 60,
        "cfg_scale": 5.0,
        "flow_shift": 9.8,
        "seed": 42,
    }
    assert len(captured["generate"]) == 3
    assert len(synchronize_calls) == 5
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["output_summary"]["num_frames"] == 321


def test_sana_reference_requires_action_from_current_request(tmp_path: Path) -> None:
    checkout = tmp_path / "Sana"
    checkout.mkdir()
    arguments = SimpleNamespace(
        manifest=REPO / "families/sana_wm/tests/manifests/sana-wm-bidirectional.json"
    )
    with pytest.raises(ValueError, match="non-empty action"):
        task_reference._run_sana_wm(
            arguments,
            {"prompt": "drive", "image_path": "assets/demo_0.png"},
            {"reference_repo": str(checkout)},
        )


def test_sana_task_reference_uses_one_explicit_official_command(
    monkeypatch, tmp_path: Path
) -> None:
    checkout = tmp_path / "Sana"
    checkout.mkdir()
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    captured = {}

    def run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "samples_ms": [1.0, 2.0],
                    "output_summary": {
                        "media_type": "video",
                        "media_count": 321,
                        "num_frames": 321,
                        "height": 704,
                        "width": 1280,
                        "channels": 3,
                    },
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(task_reference.subprocess, "run", run)
    arguments = SimpleNamespace(
        manifest=REPO / "families/sana_wm/tests/manifests/sana-wm-bidirectional.json",
        warmup=1,
        iterations=2,
    )
    result = task_reference._run_sana_wm(
        arguments,
        {
            "prompt": "drive forward",
            "image_path": "assets/demo_0.png",
            "action": "w-80,jw-40,w-40,lw-60,w-100",
            "translation_speed": 0.055,
            "rotation_speed_deg": 1.2,
            "num_frames": 321,
            "fps": 16,
            "num_steps": 60,
            "cfg_scale": 5.0,
            "flow_shift": 9.8,
            "seed": 42,
            "no_action_overlay": True,
        },
        {
            "reference_repo": str(checkout),
            "model_dir": str(model_dir),
            "intrinsics": "assets/demo_0_intrinsics.npy",
        },
    )

    command = captured["command"]
    assert command[command.index("--reference-repo") + 1] == str(checkout.resolve())
    assert command[command.index("--translation_speed") + 1] == "0.055"
    assert command[command.index("--rotation_speed_deg") + 1] == "1.2"
    assert command[command.index("--num_frames") + 1] == "321"
    assert command[command.index("--fps") + 1] == "16"
    assert command[command.index("--flow_shift") + 1] == "9.8"
    assert command[command.index("--refiner_seed") + 1] == "42"
    assert command[command.index("--warmup") + 1] == "1"
    assert command[command.index("--iterations") + 1] == "2"
    assert "--no_action_overlay" in command
    assert "env" not in captured["kwargs"]
    assert result[0] == [1.0, 2.0]
    assert result[1]["num_frames"] == 321


def test_output_contracts_are_closed_and_semantic(tmp_path: Path) -> None:
    _, environment = _environment(tmp_path)
    _, entries, _ = perf.load_suite(SUITE)
    selected = {
        entry["id"]: perf.resolve_entries([entry], environment)[0]
        for entry in entries
        if entry["id"]
        in {
            "canary.transcribe",
            "chronos_bolt.solve",
            "segformer.segment",
            "timm_vit.classify",
        }
    }
    assert perf._contract_name(selected["canary.transcribe"]) == "transcription-text"
    assert perf._contract_name(selected["chronos_bolt.solve"]) == "forecast-shape"
    assert perf._contract_name(selected["segformer.segment"]) == "segmentation-shape"
    assert perf._contract_name(selected["timm_vit.classify"]) == "classification-top-class"

    forecast = selected["chronos_bolt.solve"]
    candidate = {"output_summary": {"forecast_elements": 12, "shape": [1, 4, 3]}}
    reference = {"output_summary": {"element_count": 12, "shape": [1, 3, 4]}}
    assert perf._output_contract(forecast, candidate, reference)[0] is True
    reference["output_summary"]["element_count"] = 11
    assert perf._output_contract(forecast, candidate, reference)[0] is False

    bad_spec = {
        **forecast.spec,
        "baseline": {**forecast.spec["baseline"], "output_contract": "misspelled"},
    }
    bad = perf.ResolvedEntry(
        bad_spec,
        forecast.model,
        forecast.case,
        forecast.manifest,
        forecast.reference_precision,
        forecast.baseline_timing,
    )
    with pytest.raises(perf.PerfMatrixError, match="unsupported output contract"):
        perf._contract_name(bad)
