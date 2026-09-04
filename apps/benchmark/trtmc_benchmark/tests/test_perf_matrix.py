# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from contextlib import nullcontext
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


def test_report_is_derived_from_rows(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    results = {
        "schema_version": perf.RESULT_SCHEMA,
        "status": "completed",
        "suite": "test",
        "environment": "test",
        "rows": [
            {
                "id": "gpt2.generate",
                "model": "distilgpt2",
                "operation": "generate",
                "status": "yellow",
                "comparison": {
                    "candidate_p50_ms": 1.0,
                    "reference_p50_ms": 1.01,
                },
            }
        ],
    }
    report = perf.write_report(run, results)
    assert report["summary"]["comparable"] == 1
    assert (run / "report.json").is_file()
    assert (run / "report.html").is_file()


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
    frames = [
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.ones((24, 32, 3), dtype=np.uint8),
    ]
    summary = sana_wm_reference.media_summary(SimpleNamespace(frames=[frames]))
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
