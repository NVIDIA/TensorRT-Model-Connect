# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from array import array
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import model_benchmark, prepare_coco_detection_dataset
from qualification_tests.benchmark_qualification.performance.references import hf_transformers, generic_reference
from qualification_tests.benchmark_qualification.performance.references.timing_contracts import timing_contract
from qualification_tests.benchmark_qualification import accuracy as qualification_accuracy
from qualification_tests.benchmark_qualification.performance import (
    qualification as qualification_performance,
)
from qualification_tests.benchmark_qualification import runtime as qualification_runtime
from qualification_tests.benchmark_qualification.catalog import (
    QualificationCase,
    QualificationError,
    discover,
    load_benchmark,
)
from qualification_tests.benchmark_qualification.datasets import Dataset, resolve_dataset
from qualification_tests.benchmark_qualification.references import hf_encoder, hf_text_generation
from qualification_tests.benchmark_qualification.runtime import (
    RuntimeContext,
    prepare_bundle,
    reference_environment_options,
    reference_python,
    run_command,
    write_model_descriptor,
)


REPOSITORY = Path(__file__).resolve().parents[2]


def _example_case(tmp_path: Path, *, kind: str = "accuracy") -> QualificationCase:
    return QualificationCase(
        kind=kind,
        model="example-model",
        family="example",
        name="parity",
        benchmark="example_benchmark",
        candidate={
            "family": "example",
            "checkpoint": "example/model",
            "task": "example_task",
            "precision": "fp32",
            "build": {},
        },
        values={},
        source=tmp_path / "families/example/tests/benchmark/example-model.yaml",
        reference_requirements=None,
    )


def test_qualification_yaml_rejects_duplicate_explicit_keys(tmp_path: Path) -> None:
    profile = tmp_path / "families/example/tests/benchmark/example-model.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        "schema_version: trtmc.qualification/v1\nmodel: example-model\nmodel: replaced-model\n",
        encoding="utf-8",
    )

    with pytest.raises(QualificationError, match="duplicate key 'model'"):
        discover(tmp_path)


def test_qualification_yaml_preserves_merge_keys(tmp_path: Path) -> None:
    profile = tmp_path / "families/example/tests/benchmark/example-model.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        "schema_version: trtmc.qualification/v1\n"
        "model: example-model\n"
        "candidate: {family: example, checkpoint: example/model, revision: "
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, task: example_task, "
        "precision: fp32}\n"
        "reference: &reference {runner: hf-transformers, mode: hf-eager}\n"
        "accuracy: []\n"
        "performance:\n"
        "  - name: parity\n"
        "    benchmark: example_performance\n"
        "    reference: {<<: *reference, precision: fp32}\n",
        encoding="utf-8",
    )

    cases = discover(tmp_path)

    assert len(cases) == 1
    assert cases[0].values["reference"] == {
        "runner": "hf-transformers",
        "mode": "hf-eager",
        "precision": "fp32",
    }


def test_reference_environment_options_are_family_declared(tmp_path: Path, monkeypatch) -> None:
    case = _example_case(tmp_path)
    context = SimpleNamespace()
    monkeypatch.setattr(
        qualification_runtime,
        "reference_environment_paths",
        lambda *_args: {"reference_repo": "/reference/example"},
    )

    assert reference_environment_options(case, context, {"variant": "official"}) == {
        "variant": "official",
        "reference_repo": "/reference/example",
    }
    with pytest.raises(QualificationError, match="duplicate.*reference_repo"):
        reference_environment_options(
            case,
            context,
            {"reference_repo": "/different/reference"},
        )


def test_qualification_summary_is_written_as_report_json(tmp_path: Path) -> None:
    summary = {
        "schema_version": "trtmc.qualification-summary/v1",
        "status": "passed",
        "cases": [
            {
                "model": "example-model",
                "kind": "accuracy",
                "case": "example-model/accuracy/parity",
                "status": "passed",
            }
        ],
    }

    model_benchmark._write_summary(tmp_path, summary)

    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == summary
    assert not (tmp_path / "summary.json").exists()
    html_report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert 'href="report.json"' in html_report
    assert "summary.json" not in html_report


def test_family_configs_auto_discover_without_a_central_model_registry() -> None:
    cases = discover(REPOSITORY)
    assert cases
    assert len({case.id for case in cases}) == len(cases)
    for case in cases:
        relative = case.source.relative_to(REPOSITORY / "families")
        assert relative.parts[0] == case.family
        assert relative.parts[1:3] == ("tests", "benchmark")
        assert case.kind in {"accuracy", "performance"}
    assert not any("l0" in case.model.lower() for case in cases)


def test_qualification_profiles_match_family_manifest_tasks_when_present() -> None:
    profiles = {}
    for case in discover(REPOSITORY):
        manifest = case.source.parent.parent / "manifests" / f"{case.model}.json"
        if (
            case.kind == "accuracy"
            and case.benchmark == "imagenette_classification"
            and manifest.is_file()
        ):
            profiles[case.model] = (case, manifest)

    assert profiles
    for model, (case, manifest) in profiles.items():
        declared = json.loads(manifest.read_text(encoding="utf-8"))
        assert case.candidate["task"] == declared["task"], model


def test_l0_configs_outside_the_benchmark_folder_are_not_discovered(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "families/example/tests/l0/example.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        "schema_version: trtmc.qualification/v1\n"
        "model: example\n"
        "candidate:\n"
        "  family: example\n"
        "  checkpoint: example/model\n"
        "  revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "  task: text_generation\n"
        "  precision: fp16\n"
        "accuracy: []\n"
        "performance: []\n",
        encoding="utf-8",
    )

    assert discover(tmp_path) == ()


@pytest.mark.parametrize("revision", [None, "main", "abc1234"])
def test_trusted_remote_code_requires_an_immutable_revision(
    tmp_path: Path, revision: str | None
) -> None:
    profile = tmp_path / "families/example/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    revision_line = "" if revision is None else f"  revision: {revision}\n"
    profile.write_text(
        "schema_version: trtmc.qualification/v1\n"
        "model: example\n"
        "candidate:\n"
        "  family: example\n"
        "  checkpoint: example/model\n"
        f"{revision_line}"
        "  task: text_generation\n"
        "  precision: fp16\n"
        "  trust_remote_code: true\n"
        "accuracy: []\n"
        "performance: []\n",
        encoding="utf-8",
    )

    with pytest.raises(QualificationError, match="immutable 40-character revision"):
        discover(tmp_path)


@pytest.mark.parametrize("revision", [None, "main", "abc1234"])
def test_remote_checkpoint_requires_an_immutable_revision(
    tmp_path: Path, revision: str | None
) -> None:
    profile = tmp_path / "families/example/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    revision_line = "" if revision is None else f"  revision: {revision}\n"
    profile.write_text(
        "schema_version: trtmc.qualification/v1\n"
        "model: example\n"
        "candidate:\n"
        "  family: example\n"
        "  checkpoint: example/model\n"
        f"{revision_line}"
        "  task: text_generation\n"
        "  precision: fp16\n"
        "accuracy: []\n"
        "performance: []\n",
        encoding="utf-8",
    )

    with pytest.raises(QualificationError, match="remote checkpoint requires an immutable"):
        discover(tmp_path)


@pytest.mark.parametrize("model_directory", ["/tmp/model", "../model"])
def test_candidate_model_directory_must_stay_in_the_family_environment(
    tmp_path: Path, model_directory: str
) -> None:
    profile = tmp_path / "families/example/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        "schema_version: trtmc.qualification/v1\n"
        "model: example\n"
        "candidate:\n"
        "  family: example\n"
        "  checkpoint: example/model\n"
        f"  model_directory: {model_directory}\n"
        "  task: classification\n"
        "  precision: fp16\n"
        "accuracy: []\n"
        "performance: []\n",
        encoding="utf-8",
    )

    with pytest.raises(QualificationError, match="must stay inside the family environment"):
        discover(tmp_path)


def test_one_model_file_owns_multiple_cases_without_testcase_indirection(
    tmp_path: Path,
) -> None:
    profile = tmp_path / "families/example/tests/benchmark/example-model.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        "schema_version: trtmc.qualification/v1\n"
        "model: example-model\n"
        "candidate:\n"
        "  family: example\n"
        "  checkpoint: example/model\n"
        "  revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "  task: text_generation\n"
        "  precision: fp16\n"
        "accuracy:\n"
        "  - name: continuation\n"
        "    benchmark: example_accuracy\n"
        "performance:\n"
        "  - name: generate\n"
        "    benchmark: example_performance\n"
        "    operation: generate\n",
        encoding="utf-8",
    )

    cases = discover(tmp_path)

    assert {case.kind for case in cases} == {"accuracy", "performance"}
    assert {case.name for case in cases} == {"continuation", "generate"}
    for case in cases:
        assert case.source.parent.name == "benchmark"
        assert "testcase" not in case.values
        assert "testcase" not in str(case.values)
        assert case.candidate["checkpoint"] == "example/model"


def test_family_environment_hook_is_discovered_with_the_model_profiles(tmp_path: Path) -> None:
    profile = tmp_path / "families/example/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        "schema_version: trtmc.qualification/v1\n"
        "model: example\n"
        "candidate:\n"
        "  family: example\n"
        "  checkpoint: example/model\n"
        "  revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "  task: text_generation\n"
        "  precision: fp16\n"
        "accuracy:\n"
        "  - name: continuation\n"
        "    benchmark: example_accuracy\n"
        "performance: []\n",
        encoding="utf-8",
    )
    hook = profile.parent / "prepare_environment.py"
    hook.write_text("# family hook\n", encoding="utf-8")

    (case,) = discover(tmp_path)

    assert case.environment_hook == hook.resolve()


def test_accuracy_forwards_declared_reference_model_load_options(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile = tmp_path / "families/example/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    runner = profile.parent / "reference.py"
    runner.write_text("# family reference\n", encoding="utf-8")
    case = QualificationCase(
        kind="accuracy",
        model="example-model",
        family="example",
        name="continuation",
        benchmark="mmlu_continuation",
        candidate={
            "family": "example",
            "checkpoint": "example/model",
            "revision": "a" * 40,
            "task": "text_generation",
            "precision": "fp16",
            "trust_remote_code": True,
            "build": {"max_sequence_length": 16},
        },
        values={
            "samples": 1,
            "prompt_token_limit": 8,
            "truncation_side": "left",
            "reference": {
                "command": "reference.py",
                "model": "example/reference-model",
                "revision": "b" * 40,
                "trust_remote_code": False,
                "precision": "fp16",
                "experts_implementation": "batched_mm",
            },
            "request": {"max_new_tokens": 1, "temperature": 0.0},
            "gate": {"min_pass_rate": 1.0, "allowed_failures": 0},
        },
        source=profile,
        reference_requirements=None,
    )
    dataset_path = tmp_path / "mmlu.json"
    dataset_path.write_text(
        json.dumps({"requests": [{"id": "sample", "prompt": "Question"}]}),
        encoding="utf-8",
    )
    dataset = Dataset("mmlu-five-shot", dataset_path, "provided", "digest")
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    captured: dict[str, object] = {}
    captured_command: list[str] = []

    def reference(command, *_args, **_kwargs):
        captured_command.extend(command)
        request = Path(command[command.index("--request") + 1])
        output = Path(command[command.index("--output") + 1])
        captured.update(json.loads(request.read_text(encoding="utf-8")))
        output.write_text(
            json.dumps(
                {"samples": [{"sample_id": "sample", "prompt": "Question", "token_ids": [1]}]}
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        qualification_accuracy,
        "load_benchmark",
        lambda *_args: {
            "kind": "accuracy",
            "selection": {"method": "first"},
            "metric": {"name": "exact_token_ids"},
        },
    )
    monkeypatch.setattr(
        qualification_accuracy, "reference_python", lambda *_args: Path(sys.executable)
    )
    monkeypatch.setattr(qualification_accuracy, "run_command", reference)
    monkeypatch.setattr(
        qualification_accuracy,
        "_candidate_outputs",
        lambda *_args: ([{"token_ids": [1]}], tmp_path / "model.bundle"),
    )

    result = qualification_accuracy.run_accuracy(case, context)

    assert result["status"] == "passed"
    assert captured["model"] == "example/reference-model"
    assert captured["revision"] == "b" * 40
    assert captured["trust_remote_code"] is False
    assert captured["experts_implementation"] == "batched_mm"
    assert runner in [Path(argument) for argument in captured_command]


def test_hf_accuracy_reference_uses_requested_expert_implementation(monkeypatch) -> None:
    monkeypatch.setenv("TRTMC_QUALIFICATION_LOCAL_FILES_ONLY", "1")

    assert hf_text_generation._model_load_options(
        {"revision": "revision", "experts_implementation": "batched_mm"}
    ) == {
        "revision": "revision",
        "local_files_only": True,
        "experts_implementation": "batched_mm",
    }
    assert hf_text_generation._model_load_options({"trust_remote_code": True}) == {
        "local_files_only": True,
        "trust_remote_code": True,
    }
    assert hf_text_generation._precision_load_options("fp16") == {"torch_dtype": "fp16"}


def test_image_classification_accuracy_compares_top1_and_gold_accuracy(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "Imagenette"
    image_root = dataset_root / "images"
    image_root.mkdir(parents=True)
    for name in ("a.jpeg", "b.jpeg"):
        (image_root / name).write_bytes(b"fixture")
    dataset_path = dataset_root / "manifest.json"
    dataset_path.write_text(
        json.dumps(
            {
                "requests": [
                    {"id": "a", "image": "images/a.jpeg", "label": 1},
                    {"id": "b", "image": "images/b.jpeg", "label": 3},
                ]
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "families/example/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    case = QualificationCase(
        kind="accuracy",
        model="example",
        family="example",
        name="imagenette-parity",
        benchmark="imagenette_classification",
        candidate={
            "family": "example",
            "checkpoint": "example/model",
            "task": "classification",
            "precision": "fp16",
            "build": {},
        },
        values={
            "samples": 2,
            "reference": {"precision": "fp32", "batch_size": 2},
            "gate": {
                "min_top1_agreement": 1.0,
                "max_top1_accuracy_drop_from_hf": 0.0,
            },
        },
        source=profile,
        reference_requirements=None,
    )
    dataset = Dataset("imagenette-validation", dataset_path, "provided", "digest")
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    captured: dict[str, object] = {}

    def reference(command, *_args, **_kwargs):
        request = Path(command[command.index("--request") + 1])
        output = Path(command[command.index("--output") + 1])
        captured.update(json.loads(request.read_text(encoding="utf-8")))
        output.write_text(
            json.dumps(
                {
                    "samples": [
                        {"sample_id": "a", "top_class": 1},
                        {"sample_id": "b", "top_class": 2},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        qualification_accuracy,
        "load_benchmark",
        lambda *_args: {"metric": {"name": "image_classification_top1_parity"}},
    )
    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        qualification_accuracy, "reference_python", lambda *_args: Path(sys.executable)
    )
    monkeypatch.setattr(qualification_accuracy, "run_command", reference)
    monkeypatch.setattr(
        qualification_accuracy,
        "_candidate_outputs",
        lambda *_args: (
            [{"top_class": 1}, {"top_class": 2}],
            tmp_path / "example.bundle",
        ),
    )

    result = qualification_accuracy.run_accuracy(case, context)

    assert result["status"] == "passed"
    assert result["metrics"] == {
        "samples": 2,
        "top1_agreement": 1.0,
        "reference_top1_accuracy": 0.5,
        "candidate_top1_accuracy": 0.5,
        "top1_accuracy_drop_from_hf": 0.0,
    }
    assert captured["batch_size"] == 2
    assert captured["samples"][0]["image_path"] == str(image_root / "a.jpeg")


def test_performance_resolves_profile_owned_relative_assets(tmp_path: Path) -> None:
    profile = tmp_path / "families/example/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    image = profile.parent.parent / "data/test.jpeg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fixture")
    case = QualificationCase(
        kind="performance",
        model="example",
        family="example",
        name="classify",
        benchmark="image_classification_performance",
        candidate={
            "family": "example",
            "checkpoint": "example/model",
            "task": "classification",
            "precision": "fp16",
            "build": {},
        },
        values={},
        source=profile,
        reference_requirements=None,
    )

    resolved = qualification_performance._resolve_family_assets(
        case, {"image_path": "../data/test.jpeg", "batch_size": 1}
    )

    assert resolved == {"image_path": str(image.resolve()), "batch_size": 1}


def test_robot_action_accuracy_compares_complete_action_chunk(tmp_path: Path, monkeypatch) -> None:
    data = tmp_path / "families/example/tests/data"
    data.mkdir(parents=True)
    image = data / "image.png"
    state = data / "state.f32"
    image.write_bytes(b"fixture")
    state.write_bytes(b"state")
    dataset_path = data / "requests.json"
    dataset_path.write_text(
        json.dumps({"requests": [{"id": "recorded", "image": image.name, "state": state.name}]}),
        encoding="utf-8",
    )
    profile = tmp_path / "families/example/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    runner = profile.parent / "reference.py"
    runner.write_text("# family reference\n", encoding="utf-8")
    case = QualificationCase(
        kind="accuracy",
        model="example-control",
        family="example",
        name="recorded-action-parity",
        benchmark="robot_action_parity",
        candidate={
            "family": "example",
            "checkpoint": "example/control",
            "task": "robot_control",
            "precision": "fp32",
            "build": {},
        },
        values={
            "samples": 1,
            "reference": {"command": "reference.py", "precision": "fp32"},
            "gate": {
                "action_max_abs_error": 0.00005,
                "action_mean_abs_error": 0.000005,
                "action_rmse": 0.00001,
                "min_sample_pass_rate": 1.0,
            },
        },
        source=profile,
        reference_requirements=None,
    )
    dataset = Dataset("robot-action-recorded-observation", dataset_path, "family", "digest")
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    reference = {
        "action_steps": 2,
        "action_dim": 2,
        "action_values": 4,
        "actions": [0.1, -0.2, 0.3, -0.4],
        "finite": True,
    }
    candidate = {
        **reference,
        "within_training_bounds": True,
    }

    monkeypatch.setattr(
        qualification_accuracy,
        "load_benchmark",
        lambda *_args: {"metric": {"name": "robot_action_parity"}},
    )
    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        qualification_accuracy,
        "_family_image_reference",
        lambda *_args: [reference],
    )
    monkeypatch.setattr(
        qualification_accuracy,
        "_candidate_outputs",
        lambda *_args: ([candidate], tmp_path / "act.bundle"),
    )

    result = qualification_accuracy.run_accuracy(case, context)

    assert result["status"] == "passed"
    assert result["metrics"]["sample_pass_rate"] == 1.0
    assert result["metrics"]["action_max_abs_error"] == pytest.approx(0.0)


def test_coco_object_detection_accuracy_uses_ground_truth_map(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "COCO2017_object_detection"
    dataset_root.mkdir()
    image = dataset_root / "image.jpeg"
    image.write_bytes(b"fixture")
    dataset_path = dataset_root / "manifest.json"
    dataset_path.write_text(
        json.dumps(
            {
                "sampling": "test fixture",
                "requests": [
                    {
                        "id": "image",
                        "image": "image.jpeg",
                        "annotations": [
                            {
                                "bbox_xyxy": [10.0, 20.0, 50.0, 70.0],
                                "category_id": 3,
                                "category_index": 2,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "families/detr/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    runner = profile.parent / "reference.py"
    runner.write_text("# family reference\n", encoding="utf-8")
    case = QualificationCase(
        kind="accuracy",
        model="example",
        family="detr",
        name="coco2017-object-detection",
        benchmark="coco2017_object_detection",
        candidate={
            "family": "detr",
            "checkpoint": "facebook/example",
            "task": "object_detection",
            "precision": "fp16",
            "build": {},
        },
        values={
            "samples": 1,
            "label_space": "coco-category-id",
            "request": {"score_threshold": 0.5},
            "reference": {"command": "reference.py", "precision": "fp32"},
            "gate": {
                "max_map_50_95_drop": 0.02,
                "max_map_50_drop": 0.02,
            },
        },
        source=profile,
        reference_requirements=None,
    )
    dataset = Dataset("coco2017-object-detection", dataset_path, "provided", "digest")
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )

    def reference(command, *_args, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "sample_id": "image",
                            "boxes": [[10.0, 20.0, 50.0, 70.0]],
                            "scores": [0.9],
                            "class_ids": [3],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        qualification_accuracy,
        "load_benchmark",
        lambda *_args: {"metric": {"name": "coco_object_detection_accuracy"}},
    )
    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        qualification_accuracy, "reference_python", lambda *_args: Path(sys.executable)
    )
    monkeypatch.setattr(qualification_accuracy, "run_command", reference)
    monkeypatch.setattr(
        qualification_accuracy,
        "_candidate_outputs",
        lambda *_args: (
            [
                {
                    "boxes": [10.0, 20.0, 50.0, 70.0],
                    "scores": [0.89],
                    "class_ids": [3],
                }
            ],
            tmp_path / "example.bundle",
        ),
    )

    result = qualification_accuracy.run_accuracy(case, context)

    assert result["status"] == "passed"
    assert result["metrics"]["candidate_map_50_95"] == pytest.approx(1.0)
    assert result["metrics"]["reference_map_50_95"] == pytest.approx(1.0)
    assert result["metrics"]["map_50_95_drop"] == pytest.approx(0.0)


def test_prepare_coco_detection_dataset_is_balanced_and_keeps_real_boxes(
    tmp_path: Path,
) -> None:
    coco_root = tmp_path / "source"
    (coco_root / "annotations").mkdir(parents=True)
    (coco_root / "val2017").mkdir()
    images = [
        {"id": 11, "file_name": "000000000011.jpg", "width": 100, "height": 100},
        {"id": 22, "file_name": "000000000022.jpg", "width": 100, "height": 100},
    ]
    annotations = [
        {
            "id": 101,
            "image_id": 11,
            "category_id": 1,
            "bbox": [10, 20, 30, 40],
            "area": 1200,
            "iscrowd": 0,
        },
        {
            "id": 202,
            "image_id": 22,
            "category_id": 3,
            "bbox": [1, 2, 10, 20],
            "area": 200,
            "iscrowd": 0,
        },
    ]
    categories = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}]
    (coco_root / "annotations" / "instances_val2017.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}),
        encoding="utf-8",
    )
    for image in images:
        (coco_root / "val2017" / image["file_name"]).write_bytes(b"fixture")

    output = prepare_coco_detection_dataset.prepare(coco_root, tmp_path / "output", 2)
    manifest = json.loads(output.read_text(encoding="utf-8"))

    assert [request["image_id"] for request in manifest["requests"]] == [11, 22]
    assert manifest["requests"][0]["annotations"] == [
        {
            "id": 101,
            "category_id": 1,
            "category_index": 0,
            "bbox_xyxy": [10.0, 20.0, 40.0, 60.0],
            "area": 1200.0,
        }
    ]
    assert manifest["requests"][1]["annotations"][0]["category_index"] == 1


def test_prompted_segmentation_masks_match_independent_of_order() -> None:
    first = {
        "num_masks": 2,
        "height": 2,
        "width": 2,
        "mask_kind": "logits",
        "masks": [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
    }
    second = {
        "num_masks": 2,
        "height": 2,
        "width": 2,
        "mask_kind": "binary",
        "masks": [0, 1, 1, 0, 1, 0, 0, 1],
    }

    _, _, candidate = qualification_accuracy._binary_masks(first, "candidate")
    _, _, reference = qualification_accuracy._binary_masks(second, "reference")

    assert qualification_accuracy._match_masks(candidate, reference) == [1.0, 1.0]


def test_text_prompted_instances_match_masks_boxes_and_scores() -> None:
    summary = {
        "num_masks": 1,
        "height": 2,
        "width": 2,
        "mask_kind": "binary",
        "masks": [1, 0, 0, 1],
        "iou_scores": [0.9],
        "boxes": [[0.0, 0.0, 2.0, 2.0]],
        "box_coordinates": "original_image_pixels_xyxy",
    }
    parsed = qualification_accuracy._instance_masks(summary, "test")
    matches = qualification_accuracy._match_instances(parsed, parsed)

    assert matches == [{"mask_iou": 1.0, "box_iou": 1.0, "score_abs_error": 0.0}]


def test_vision_language_text_distance_is_normalized() -> None:
    left = qualification_accuracy._normalized_answer("  Red\ncar ")
    right = qualification_accuracy._normalized_answer("red car")

    assert qualification_accuracy._normalized_edit_distance(left, right) == 0.0
    assert qualification_accuracy._normalized_edit_distance(left, "blue car") > 0.15


def test_ocr_accuracy_uses_dataset_questions_and_reports_gold_matches(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "OCRBench_v2/unified"
    image_root = dataset_root / "images"
    image_root.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (image_root / name).write_bytes(b"fixture")
    dataset_path = dataset_root / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "a",
                        "question": "What is the setting?",
                        "media": [{"type": "image", "path": "images/a.jpg"}],
                        "answer": {"primary": "enabled", "aliases": ["enabled", "on"]},
                    },
                    {
                        "id": "b",
                        "question": "Which application?",
                        "media": [{"type": "image", "path": "images/b.jpg"}],
                        "answer": {"primary": "Facebook", "aliases": ["Facebook"]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "families/example/tests/benchmark/example-ocr.yaml"
    profile.parent.mkdir(parents=True)
    runner = profile.parent / "reference.py"
    runner.write_text("# family reference\n", encoding="utf-8")
    case = QualificationCase(
        kind="accuracy",
        model="example-ocr",
        family="example",
        name="ocrbench-v2-parity",
        benchmark="ocrbench_v2_parity",
        candidate={
            "family": "example",
            "checkpoint": "example/ocr",
            "revision": "a" * 40,
            "task": "vision_language_generation",
            "precision": "fp16",
            "build": {},
        },
        values={
            "samples": 2,
            "request": {"max_new_tokens": 16},
            "reference": {"command": "reference.py", "precision": "bf16"},
            "gate": {"max_normalized_edit_distance": 0.1, "min_sample_pass_rate": 1.0},
        },
        source=profile,
        reference_requirements=None,
    )
    dataset = Dataset("ocrbench-v2", dataset_path, "provided", "digest")
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path,
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    captured_reference: dict[str, object] = {}
    captured_candidate: list[dict[str, object]] = []

    def reference(command, *_args, **_kwargs):
        request = Path(command[command.index("--request") + 1])
        output = Path(command[command.index("--output") + 1])
        captured_reference.update(json.loads(request.read_text(encoding="utf-8")))
        output.write_text(
            json.dumps(
                {
                    "samples": [
                        {"sample_id": "a", "text": "enabled"},
                        {"sample_id": "b", "text": "Facebook"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def candidate(_case, _context, _output, _operation, requests):
        captured_candidate.extend(requests)
        return ([{"text": "enabled"}, {"text": "facebook"}], tmp_path / "model.bundle")

    monkeypatch.setattr(
        qualification_accuracy,
        "load_benchmark",
        lambda *_args: {"metric": {"name": "ocr_text_parity"}},
    )
    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        qualification_accuracy, "reference_python", lambda *_args: Path(sys.executable)
    )
    monkeypatch.setattr(qualification_accuracy, "run_command", reference)
    monkeypatch.setattr(qualification_accuracy, "_candidate_outputs", candidate)

    result = qualification_accuracy.run_accuracy(case, context)

    assert result["status"] == "passed"
    assert result["metrics"]["candidate_normalized_gold_match_rate"] == 1.0
    assert result["metrics"]["reference_normalized_gold_match_rate"] == 1.0
    assert captured_reference["samples"][0]["prompt"] == "What is the setting?"
    assert captured_candidate[1]["request"]["prompt"] == "Which application?"


def test_localization_accuracy_matches_unordered_boxes_and_points() -> None:
    first_kind, first_boxes = qualification_accuracy._localization_values(
        "<ref>cars</ref><box><10><20><100><200></box><box><300><400><500><600></box>"
    )
    second_kind, second_boxes = qualification_accuracy._localization_values(
        "<ref>cars</ref><box><300><400><500><600></box><box><10><20><100><200></box>"
    )
    first_point_kind, first_points = qualification_accuracy._localization_values(
        "<ref>cars</ref><point><100><200></point>"
    )
    second_point_kind, second_points = qualification_accuracy._localization_values(
        "<ref>cars</ref><point><106><208></point>"
    )

    assert first_kind == second_kind == "box"
    assert qualification_accuracy._localization_alignment(first_boxes, second_boxes, "box") == 1.0
    assert first_point_kind == second_point_kind == "point"
    assert (
        qualification_accuracy._localization_alignment(first_points, second_points, "point") == 10.0
    )
    with pytest.raises(QualificationError, match="reference tag"):
        qualification_accuracy._localization_values("<box><10><20><100><200></box>")


def test_reranking_order_is_stable_and_score_validation_is_strict() -> None:
    assert qualification_accuracy._reranking_order([0.7, 0.2, 0.7]) == [0, 2, 1]
    assert qualification_accuracy._reranking_scores({"scores": [0.7, -0.2]}, "test") == [
        0.7,
        -0.2,
    ]
    with pytest.raises(QualificationError, match="non-finite"):
        qualification_accuracy._reranking_scores({"scores": [float("nan")]}, "test")


def test_semantic_segmentation_accuracy_compares_pixel_and_class_iou(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "Imagenette"
    dataset_root.mkdir()
    image = dataset_root / "image.jpeg"
    image.write_bytes(b"fixture")
    dataset_path = dataset_root / "manifest.json"
    dataset_path.write_text(
        json.dumps({"requests": [{"id": "image", "image": "image.jpeg"}]}),
        encoding="utf-8",
    )
    profile = tmp_path / "families/example/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    runner = profile.parent / "reference.py"
    runner.write_text("# family reference\n", encoding="utf-8")
    case = QualificationCase(
        kind="accuracy",
        model="example",
        family="example",
        name="segmentation-parity",
        benchmark="imagenette_segmentation_parity",
        candidate={
            "family": "example",
            "checkpoint": "example/segmentation",
            "task": "segmentation",
            "precision": "fp16",
            "build": {},
        },
        values={
            "samples": 1,
            "reference": {"command": "reference.py", "precision": "fp32"},
            "gate": {
                "min_pixel_accuracy": 0.75,
                "min_mean_iou": 0.58,
                "min_sample_pass_rate": 1.0,
            },
        },
        source=profile,
        reference_requirements=None,
    )
    dataset = Dataset("imagenette-validation", dataset_path, "provided", "digest")
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )

    def reference(command, *_args, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "sample_id": "image",
                            "height": 2,
                            "width": 2,
                            "mask": [0, 0, 1, 1],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        qualification_accuracy,
        "load_benchmark",
        lambda *_args: {"metric": {"name": "semantic_segmentation_parity"}},
    )
    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        qualification_accuracy, "reference_python", lambda *_args: Path(sys.executable)
    )
    monkeypatch.setattr(qualification_accuracy, "run_command", reference)
    monkeypatch.setattr(
        qualification_accuracy,
        "_candidate_outputs",
        lambda *_args: (
            [{"height": 2, "width": 2, "mask": [0, 0, 1, 0]}],
            tmp_path / "example.bundle",
        ),
    )

    result = qualification_accuracy.run_accuracy(case, context)

    assert result["status"] == "passed"
    assert result["metrics"]["mean_pixel_accuracy"] == 0.75
    assert result["metrics"]["mean_iou"] >= 0.58


def test_image_feature_accuracy_compares_vectors_and_knn_utility(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_root = tmp_path / "beans"
    (dataset_root / "images/train").mkdir(parents=True)
    (dataset_root / "images/test").mkdir(parents=True)
    (dataset_root / "queries").mkdir()
    for split in ("train", "test"):
        for index in range(2):
            (dataset_root / f"images/{split}/{index}.jpg").write_bytes(b"fixture")
    (dataset_root / "bank.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"image": "images/train/0.jpg", "label": 0, "source_index": 0},
                    {"image": "images/train/1.jpg", "label": 1, "source_index": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    (dataset_root / "queries/test.json").write_text(
        json.dumps(
            {
                "samples": [
                    {"image": "../images/test/0.jpg", "label": 0, "source_index": 0},
                    {"image": "../images/test/1.jpg", "label": 1, "source_index": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset_path = dataset_root / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "inputs": {
                            "bank_manifest": "bank.json",
                            "query_manifest": "queries/test.json",
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "families/example/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    runner = profile.parent / "reference.py"
    runner.write_text("# family reference\n", encoding="utf-8")
    case = QualificationCase(
        kind="accuracy",
        model="example",
        family="example",
        name="beans-knn-parity",
        benchmark="beans_image_feature_knn",
        candidate={
            "family": "example",
            "checkpoint": "example/features",
            "task": "image_features",
            "precision": "fp16",
            "build": {"max_sequence_length": 1},
        },
        values={
            "bank_samples_per_class": 1,
            "query_samples_per_class": 1,
            "knn_k": 1,
            "knn_temperature": 0.07,
            "reference": {"command": "reference.py", "precision": "fp32"},
            "gate": {
                "min_pooler_cosine": 0.999,
                "min_vector_pass_rate": 1.0,
                "min_knn_top1_agreement": 1.0,
                "max_knn_accuracy_drop_from_hf": 0.0,
            },
        },
        source=profile,
        reference_requirements=None,
    )
    dataset = Dataset("image-feature-knn-v1", dataset_path, "provided", "digest")
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9]]

    def reference(command, *_args, **_kwargs):
        request = Path(command[command.index("--request") + 1])
        output = Path(command[command.index("--output") + 1])
        samples = json.loads(request.read_text(encoding="utf-8"))["samples"]
        output.write_text(
            json.dumps(
                {
                    "samples": [
                        {"sample_id": sample["sample_id"], "pooler_output": vector}
                        for sample, vector in zip(samples, vectors, strict=True)
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        qualification_accuracy,
        "load_benchmark",
        lambda *_args: {"metric": {"name": "image_feature_knn_parity"}},
    )
    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        qualification_accuracy, "reference_python", lambda *_args: Path(sys.executable)
    )
    monkeypatch.setattr(qualification_accuracy, "run_command", reference)
    monkeypatch.setattr(
        qualification_accuracy,
        "_candidate_outputs",
        lambda *_args: (
            [{"pooler_output": vector} for vector in vectors],
            tmp_path / "example.bundle",
        ),
    )

    result = qualification_accuracy.run_accuracy(case, context)

    assert result["status"] == "passed"
    assert result["metrics"]["vector_pass_rate"] == 1.0
    assert result["metrics"]["knn_top1_agreement"] == 1.0
    assert result["metrics"]["candidate_knn_top1_accuracy"] == 1.0


def test_timm_classification_uses_a_generic_reference_adapter() -> None:
    assert "timm-classification" in generic_reference.ADAPTERS
    assert generic_reference.LOADERS["timm-classification"] is generic_reference._load_vision


def test_glm_asr_performance_uses_its_family_reference() -> None:
    case = next(
        value
        for value in discover(REPOSITORY)
        if value.model == "glmasr-nano-fp16" and value.kind == "performance"
    )
    assert case.values["reference"]["script"] == "tests/benchmark/reference.py"
    assert "hf-chat-asr" not in generic_reference.ADAPTERS


def test_speech_accuracy_reports_reference_and_labeled_wer(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "datasets"
    audio_root = data_root / "speech/audio"
    audio_root.mkdir(parents=True)
    for name in ("a.flac", "b.flac"):
        (audio_root / name).write_bytes(b"fixture")
    dataset_path = data_root / "speech/dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "id": "a",
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "audio", "audio": "speech/audio/a.flac"}],
                            }
                        ],
                        "reference": "hello world",
                    },
                    {
                        "id": "b",
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "audio", "audio": "speech/audio/b.flac"}],
                            }
                        ],
                        "reference": "one two",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "families/example/tests/benchmark/example.yaml"
    profile.parent.mkdir(parents=True)
    runner = profile.parent / "reference.py"
    runner.write_text("# family reference\n", encoding="utf-8")
    case = QualificationCase(
        kind="accuracy",
        model="example",
        family="example",
        name="librispeech-wer",
        benchmark="librispeech_transcription",
        candidate={
            "family": "example",
            "checkpoint": "org/example",
            "task": "transcription",
            "precision": "fp16",
            "build": {},
        },
        values={
            "samples": 2,
            "request": {"max_new_tokens": 32},
            "reference": {"command": "reference.py", "precision": "fp32"},
            "gate": {
                "max_wer_to_reference": 0.3,
                "max_wer_increase_from_reference": 0.3,
            },
        },
        source=profile,
        reference_requirements=None,
    )
    dataset = Dataset("speech", dataset_path, "provided", "digest")
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=data_root,
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )

    def reference(command, *_args, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        wavs = [tmp_path / "a.wav", tmp_path / "b.wav"]
        for wav in wavs:
            wav.write_bytes(b"fixture")
        output.write_text(
            json.dumps(
                {
                    "samples": [
                        {"sample_id": "a", "text": "hello world", "audio_path": str(wavs[0])},
                        {"sample_id": "b", "text": "one two", "audio_path": str(wavs[1])},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        qualification_accuracy,
        "load_benchmark",
        lambda *_args: {"metric": {"name": "speech_transcription_wer_parity"}},
    )
    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        qualification_accuracy, "reference_python", lambda *_args: Path(sys.executable)
    )
    monkeypatch.setattr(qualification_accuracy, "run_command", reference)
    monkeypatch.setattr(
        qualification_accuracy,
        "_candidate_outputs",
        lambda *_args: (
            [{"text": "hello world"}, {"text": "one three"}],
            tmp_path / "example.bundle",
        ),
    )

    result = qualification_accuracy.run_accuracy(case, context)

    assert result["status"] == "passed"
    assert result["metrics"] == {
        "samples": 2,
        "reference_wer": 0.0,
        "candidate_wer": 0.25,
        "wer_increase_from_reference": 0.25,
        "wer_to_reference": 0.25,
    }


def test_accuracy_forwards_seq2seq_reference_contract_and_nested_dataset_input(
    tmp_path: Path, monkeypatch
) -> None:
    dataset_path = tmp_path / "translation.json"
    dataset_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "sample_id": "translation-0",
                        "inputs": {"prompt": "The house is wonderful."},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    case = QualificationCase(
        kind="accuracy",
        model="translation-model",
        family="translation",
        name="translation-parity",
        benchmark="translation_parity",
        candidate={
            "family": "translation",
            "checkpoint": "org/model",
            "task": "text_generation",
            "precision": "fp16",
            "build": {},
        },
        values={
            "samples": 1,
            "prompt_token_limit": 96,
            "reference": {
                "precision": "fp16",
                "task": "seq2seq-lm",
                "output_token_policy": "strip-start-and-eos",
                "source_language_placement": "replace-final-unk",
            },
            "request": {
                "max_new_tokens": 128,
                "source_language": "eng_Latn",
                "source_language_token_id": 256047,
                "target_language": "fra_Latn",
                "forced_bos_token_id": 256057,
            },
            "gate": {"min_pass_rate": 1.0, "allowed_failures": 0},
        },
        source=tmp_path / "translation.yaml",
        reference_requirements=None,
    )
    dataset = Dataset("translation", dataset_path, "provided", "digest")
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    captured_reference: dict[str, object] = {}
    captured_candidate: list[dict[str, object]] = []

    def reference(command, *_args, **_kwargs):
        request = Path(command[command.index("--request") + 1])
        output = Path(command[command.index("--output") + 1])
        captured_reference.update(json.loads(request.read_text(encoding="utf-8")))
        output.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "sample_id": "translation-0",
                            "prompt": "The house is wonderful.",
                            "token_ids": [4, 5],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def candidate(_case, _context, _output, _operation, requests):
        captured_candidate.extend(requests)
        return ([{"token_ids": [4, 5]}], tmp_path / "translation.bundle")

    monkeypatch.setattr(
        qualification_accuracy,
        "load_benchmark",
        lambda *_args: {"metric": {"name": "exact_token_ids"}},
    )
    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        qualification_accuracy, "reference_python", lambda *_args: Path(sys.executable)
    )
    monkeypatch.setattr(qualification_accuracy, "run_command", reference)
    monkeypatch.setattr(qualification_accuracy, "_candidate_outputs", candidate)

    result = qualification_accuracy.run_accuracy(case, context)

    assert result["status"] == "passed"
    assert captured_reference["task"] == "seq2seq-lm"
    assert captured_reference["output_token_policy"] == "strip-start-and-eos"
    assert captured_reference["generation"]["source_language_placement"] == ("replace-final-unk")
    assert captured_reference["samples"] == [
        {"sample_id": "translation-0", "prompt": "The house is wonderful."}
    ]
    assert captured_candidate[0]["request"]["forced_bos_token_id"] == 256057


def test_hf_accuracy_reference_selects_the_seq2seq_model_class() -> None:
    causal = object()
    seq2seq = object()

    assert hf_text_generation._model_class("causal-lm", causal, seq2seq) is causal
    assert hf_text_generation._model_class("seq2seq-lm", causal, seq2seq) is seq2seq
    with pytest.raises(ValueError, match="unsupported reference task"):
        hf_text_generation._model_class("encoder", causal, seq2seq)


def test_hf_accuracy_reference_applies_explicit_translation_languages() -> None:
    class Tokenizer:
        src_lang = None
        unk_token_id = 0

        @staticmethod
        def convert_tokens_to_ids(value: str) -> int:
            return {"eng_Latn": 256047, "fra_Latn": 256057}.get(value, 0)

        @staticmethod
        def convert_ids_to_tokens(value: int) -> str:
            return {256047: "eng_Latn", 256057: "fra_Latn"}[value]

    tokenizer = Tokenizer()
    controls, source_token_id = hf_text_generation._translation_controls(
        tokenizer,
        {
            "source_language": "eng_Latn",
            "source_language_token_id": 256047,
            "target_language": "fra_Latn",
            "forced_bos_token_id": 256057,
        },
    )

    assert tokenizer.src_lang == "eng_Latn"
    assert controls == {"forced_bos_token_id": 256057}
    assert source_token_id is None


@pytest.mark.parametrize("runner", [hf_text_generation, hf_transformers])
def test_hf_translation_supports_replace_final_unknown_tokenizer(runner) -> None:
    import torch

    class GenericTranslationTokenizer:
        unk_token_id = 3

        @staticmethod
        def convert_tokens_to_ids(value: str) -> int:
            return {"eng_Latn": 256047, "fra_Latn": 256057}.get(value, 3)

        @staticmethod
        def convert_ids_to_tokens(value: int) -> str:
            return {256047: "eng_Latn", 256057: "fra_Latn"}[value]

    tokenizer = GenericTranslationTokenizer()
    request = {
        "source_language": "eng_Latn",
        "source_language_token_id": 256047,
        "target_language": "fra_Latn",
        "forced_bos_token_id": 256057,
    }
    with pytest.raises(ValueError, match="source_language_placement"):
        runner._translation_controls(tokenizer, request)

    controls, source_token_id = runner._translation_controls(
        tokenizer,
        {
            **request,
            "source_language_placement": "replace-final-unk",
        },
    )
    encoded = {
        "input_ids": torch.tensor([[17, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    runner._apply_source_language(encoded, source_token_id, tokenizer)

    assert controls == {"forced_bos_token_id": 256057}
    assert source_token_id == 256047
    assert encoded["input_ids"].tolist() == [[17, 2, 256047]]


@pytest.mark.parametrize(
    ("row", "expected_status", "expected_error"),
    [
        (
            {
                "status": "contract-mismatch",
                "comparison": {"reason": "generated token ids differ"},
                "reference_attempts": [],
            },
            "failed",
            None,
        ),
        (
            {"status": "white", "error": "candidate command failed"},
            "error",
            "candidate command failed",
        ),
        (
            {
                "status": "white",
                "error": "reference command and configured fallback failed",
                "reference_attempts": [
                    {
                        "mode": "torch-compile",
                        "exit_code": 1,
                        "fallback_reason": "reference command failed",
                    }
                ],
            },
            "error",
            "reference command and configured fallback failed",
        ),
    ],
)
def test_performance_distinguishes_model_contract_failures_from_execution_errors(
    tmp_path: Path,
    monkeypatch,
    row: dict[str, object],
    expected_status: str,
    expected_error: str | None,
) -> None:
    case = QualificationCase(
        kind="performance",
        model="example-model",
        family="example",
        name="generate",
        benchmark="text_generation_performance",
        candidate={
            "family": "example",
            "checkpoint": "example/model",
            "task": "text_generation",
            "precision": "fp16",
            "build": {},
        },
        values={
            "operation": "generate",
            "request": {"prompt": "Hello", "max_new_tokens": 1},
            "measurement": {"warmup": 1, "iterations": 10},
            "reference": {
                "runner": "hf-transformers",
                "mode": "hf-eager",
                "precision": "fp32",
                "output_contract": "exact-token-ids",
            },
        },
        source=tmp_path / "example.yaml",
        reference_requirements=None,
    )
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=tmp_path / "runtime",
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=tmp_path / "worker",
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    definition = {
        "reference_timing": {
            "timing_scope": "public_operation_call_wall",
            "input_preparation_included": True,
            "asset_loading_included": False,
        },
        "stability": {
            "samples": 10,
            "max_half_median_change_percent": 5.0,
            "median_band_percent": 5.0,
            "minimum_samples_within_band": 8,
            "retries": 1,
        },
    }

    def run_single_case(*_args, **_kwargs):
        return {"id": "qualification.example.generate", **row}

    monkeypatch.setattr(qualification_performance, "load_benchmark", lambda *_: definition)
    monkeypatch.setattr(
        qualification_performance,
        "require_candidate",
        lambda *_: (context.worker, context.runtime_root),
    )
    monkeypatch.setattr(
        qualification_performance, "reference_python", lambda *_: Path(sys.executable)
    )
    monkeypatch.setattr(qualification_performance, "run_case", run_single_case)

    result = qualification_performance.run_performance(case, context)

    assert result["status"] == expected_status
    assert result.get("error") == expected_error


def test_hf_accuracy_reference_normalizes_seq2seq_control_tokens() -> None:
    assert hf_text_generation._normalize_seq2seq_tokens(
        [2, 256057, 1034, 248075, 2],
        decoder_start_token_id=2,
        eos_token_id=2,
        policy="strip-start-and-eos",
    ) == [256057, 1034, 248075]
    assert hf_text_generation._normalize_seq2seq_tokens(
        [0, 17, 1],
        decoder_start_token_id=0,
        eos_token_id=1,
        policy="strip-start",
    ) == [17, 1]


def test_sts_samples_expand_pairs_with_family_owned_prompt_prefix(tmp_path: Path) -> None:
    dataset = tmp_path / "sts.jsonl"
    dataset.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "genre": "captions",
                        "score": 2.5,
                        "sentence1": "A girl is styling her hair.",
                        "sentence2": "A girl is brushing her hair.",
                    }
                ),
                json.dumps(
                    {
                        "genre": "news",
                        "score": 4.0,
                        "sentence1": "One sentence.",
                        "sentence2": "Another sentence.",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert qualification_accuracy._sts_samples(dataset, 1, "query: ") == [
        {
            "sample_id": "stsbenchmark-000000-a",
            "pair_id": "stsbenchmark-000000",
            "pair_side": "sentence1",
            "score": 2.5,
            "prompt": "query: A girl is styling her hair.",
        },
        {
            "sample_id": "stsbenchmark-000000-b",
            "pair_id": "stsbenchmark-000000",
            "pair_side": "sentence2",
            "score": 2.5,
            "prompt": "query: A girl is brushing her hair.",
        },
    ]


def test_encoder_embedding_comparison_restores_pre_refactor_gates() -> None:
    reference = [
        {
            "sample_id": "pair-a",
            "pair_id": "pair",
            "pair_side": "sentence1",
            "score": 5.0,
            "vector": [1.0, 0.0],
        },
        {
            "sample_id": "pair-b",
            "pair_id": "pair",
            "pair_side": "sentence2",
            "score": 5.0,
            "vector": [0.8, 0.6],
        },
    ]
    candidate = [{"values": [1.0, 0.0]}, {"values": [0.8, 0.6]}]

    result = qualification_accuracy._compare_encoder_embeddings(
        reference,
        candidate,
        {
            "min_vector_cosine": 0.999,
            "min_vector_pass_rate": 1.0,
            "max_pair_cosine_abs_delta": 0.02,
        },
    )

    assert result["status"] == "passed"
    assert result["metrics"]["vector_pass_rate"] == 1.0
    assert result["metrics"]["max_pair_cosine_abs_delta"] == pytest.approx(0.0)
    assert result["metrics"]["hf_sts_spearman"] is None
    assert result["metrics"]["candidate_sts_spearman"] is None


@pytest.mark.parametrize(
    ("task", "expected_mode", "expected_operation"),
    [("encoding", "cls", "encode"), ("embedding", "embedding", "embed")],
)
def test_encoder_accuracy_uses_task_semantics_without_model_specific_runner(
    tmp_path: Path,
    monkeypatch,
    task: str,
    expected_mode: str,
    expected_operation: str,
) -> None:
    dataset_path = tmp_path / "sts.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "genre": "captions",
                "score": 3.0,
                "sentence1": "Sentence one.",
                "sentence2": "Sentence two.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    case = QualificationCase(
        kind="accuracy",
        model="encoder-model",
        family="encoder",
        name="stsbenchmark-parity",
        benchmark="stsbenchmark_embedding_parity",
        candidate={
            "family": "encoder",
            "checkpoint": "org/model",
            "task": task,
            "precision": "fp16",
            "build": {"max_sequence_length": 128},
        },
        values={
            "samples": 1,
            "reference": {"precision": "fp32"},
            "gate": {
                "min_vector_cosine": 0.999,
                "min_vector_pass_rate": 1.0,
                "max_pair_cosine_abs_delta": 0.02,
            },
        },
        source=tmp_path / "encoder.yaml",
        reference_requirements=None,
    )
    dataset = Dataset("stsbenchmark-test", dataset_path, "provided", "digest")
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    captured_reference: dict[str, object] = {}
    captured_candidate: dict[str, object] = {}

    def reference(command, *_args, **_kwargs):
        request = Path(command[command.index("--request") + 1])
        output = Path(command[command.index("--output") + 1])
        captured_reference.update(json.loads(request.read_text(encoding="utf-8")))
        samples = captured_reference["samples"]
        output.write_text(
            json.dumps(
                {
                    "samples": [
                        {**sample, "vector": [1.0, float(index)]}
                        for index, sample in enumerate(samples)
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def candidate(_case, _context, _output, operation, requests):
        captured_candidate.update({"operation": operation, "requests": requests})
        return (
            [{"values": [1.0, float(index)]} for index, _request in enumerate(requests)],
            tmp_path / "encoder.bundle",
        )

    monkeypatch.setattr(
        qualification_accuracy, "reference_python", lambda *_args: Path(sys.executable)
    )
    monkeypatch.setattr(qualification_accuracy, "run_command", reference)
    monkeypatch.setattr(qualification_accuracy, "_candidate_outputs", candidate)

    result = qualification_accuracy._encoder_embedding_parity(
        case, context, dataset, tmp_path / "output"
    )

    assert result["status"] == "passed"
    assert captured_reference["mode"] == expected_mode
    assert captured_candidate["operation"] == expected_operation
    assert [request["request"] for request in captured_candidate["requests"]] == [
        {"prompt": "Sentence one."},
        {"prompt": "Sentence two."},
    ]


def test_hf_encoder_rejects_unknown_vector_mode() -> None:
    with pytest.raises(ValueError, match="unsupported encoder vector mode"):
        hf_encoder._vector_mode("mean")


def test_hf_encoder_resolves_family_declared_transformers_classes() -> None:
    auto_model = object()
    auto_tokenizer = object()
    custom_model = object()
    custom_tokenizer = object()
    transformers = SimpleNamespace(
        AutoModel=auto_model,
        AutoTokenizer=auto_tokenizer,
        CustomEncoder=custom_model,
        CustomTokenizer=custom_tokenizer,
    )

    assert hf_encoder._reference_classes(transformers, "auto", "auto") == (
        auto_model,
        auto_tokenizer,
    )
    assert hf_encoder._reference_classes(
        transformers,
        "transformers.CustomEncoder",
        "transformers.CustomTokenizer",
    ) == (custom_model, custom_tokenizer)
    with pytest.raises(ValueError, match="unsupported Transformers class"):
        hf_encoder._reference_classes(
            transformers,
            "dpr-context-encoder",
            "transformers.CustomTokenizer",
        )


def test_shared_definitions_own_dataset_and_metric_not_models() -> None:
    for case in discover(REPOSITORY):
        definition = load_benchmark(REPOSITORY, case)
        assert "models" not in definition
        assert definition["kind"] == case.kind
        if case.kind == "accuracy":
            assert definition["metric"]["name"]
            if definition["metric"]["name"] == "task_output_parity":
                assert definition["input"]["adapter"] == "configured_request"
                assert "dataset" not in definition
            else:
                assert definition["dataset"]["id"]


def test_fixed_dataset_selection_uses_declared_indices() -> None:
    rows = [f"sample-{index}" for index in range(8)]

    selected, receipt = qualification_accuracy._select_dataset_rows(
        rows,
        {"selection": {"method": "fixed-indices", "indices": [6, 2, 4]}},
        2,
        "example",
    )

    assert selected == ["sample-6", "sample-2"]
    assert receipt == {"method": "fixed-indices", "indices": [6, 2]}


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        ([1, 1], "must be unique"),
        ([1, True], "nonnegative integers"),
        ([1, 8], "out of range"),
    ],
)
def test_fixed_dataset_selection_rejects_invalid_indices(indices: list[int], message: str) -> None:
    with pytest.raises(QualificationError, match=message):
        qualification_accuracy._select_dataset_rows(
            list(range(8)),
            {"selection": {"method": "fixed-indices", "indices": indices}},
            2,
            "example",
        )


@pytest.mark.parametrize(
    "benchmark",
    ["mmlu_continuation", "librispeech_transcription", "ocrbench_v2_parity"],
)
def test_slow_dataset_benchmarks_keep_ten_fixed_representative_samples(
    benchmark: str,
) -> None:
    case = next(case for case in discover(REPOSITORY) if case.benchmark == benchmark)
    selection = load_benchmark(REPOSITORY, case)["selection"]

    assert selection["method"] == "fixed-indices"
    assert len(selection["indices"]) == 10
    assert len(set(selection["indices"])) == 10


def test_shared_performance_definitions_own_complete_reference_timing() -> None:
    for case in discover(REPOSITORY):
        if case.kind != "performance":
            continue
        definition = load_benchmark(REPOSITORY, case)
        declared = definition["reference_timing"]

        assert set(declared) == {
            "timing_scope",
            "input_preparation_included",
            "asset_loading_included",
        }
        assert timing_contract(runner=str(case.values["reference"]["runner"]), declared=declared)


def test_internal_automation_is_separate_from_the_installed_benchmark() -> None:
    old_application = REPOSITORY / "apps/benchmark/qualification"
    assert list(old_application.rglob("*.py")) == []
    assert list((REPOSITORY / "tools/benchmark_qualification").rglob("*.py")) == []
    user_sources = [
        path.read_text(encoding="utf-8")
        for path in (REPOSITORY / "apps/benchmark").rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    assert not any("benchmark_qualification" in source for source in user_sources)


def test_candidate_descriptor_is_generated_from_public_build_inputs(tmp_path: Path) -> None:
    case = QualificationCase(
        kind="accuracy",
        model="example-model",
        family="example",
        name="continuation",
        benchmark="example_accuracy",
        candidate={
            "family": "example",
            "checkpoint": "example/model",
            "revision": "a" * 40,
            "task": "text_generation",
            "precision": "fp16",
            "trust_remote_code": True,
            "build": {"max_sequence_length": 1024},
        },
        values={},
        source=tmp_path / "example.yaml",
        reference_requirements=None,
    )

    descriptor = write_model_descriptor(case, tmp_path, {"prompt": "hello"})
    value = json.loads(descriptor.read_text(encoding="utf-8"))

    assert value["hf_id"] == "example/model"
    assert value["hf_revision"] == "a" * 40
    assert value["task"] == "text_generation"
    assert value["max_sequence_length"] == 1024
    assert value["trust_remote_code"] is True
    assert "tests/manifests" not in descriptor.read_text(encoding="utf-8")


def test_candidate_descriptor_preserves_family_selected_task(tmp_path: Path) -> None:
    case = QualificationCase(
        kind="performance",
        model="geometry-model",
        family="geometry",
        name="geometry",
        benchmark="geometry_performance",
        candidate={
            "family": "geometry",
            "checkpoint": "example/geometry",
            "task": "legacy_geometry",
            "selected_task": "image_to_metric_geometry",
            "precision": "fp32",
            "build": {},
        },
        values={},
        source=tmp_path / "geometry.yaml",
        reference_requirements=None,
    )

    descriptor = write_model_descriptor(case, tmp_path, {"image_path": "/data/image.png"})
    testcase = json.loads(descriptor.read_text(encoding="utf-8"))["testcases"][0]

    assert testcase["selected_task"] == "image_to_metric_geometry"
    assert testcase["image_path"] == "/data/image.png"


def test_candidate_descriptor_uses_family_environment_model_directory(
    tmp_path: Path, monkeypatch
) -> None:
    environment = tmp_path / "family-environment"
    python = environment / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    model_directory = environment / "prepared/model"
    model_directory.mkdir(parents=True)
    case = QualificationCase(
        kind="accuracy",
        model="source-model",
        family="source_family",
        name="parity",
        benchmark="source_parity",
        candidate={
            "family": "source_family",
            "checkpoint": "example/source-model",
            "model_directory": "prepared/model",
            "task": "classification",
            "precision": "fp16",
            "build": {},
        },
        values={},
        source=tmp_path / "source-model.yaml",
        reference_requirements=None,
    )
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=False,
        verbose=False,
    )
    monkeypatch.setattr(
        "qualification_tests.benchmark_qualification.runtime.reference_python", lambda *_args: python
    )

    descriptor = write_model_descriptor(
        case, tmp_path, {"image_path": "/data/image.png"}, context=context
    )

    assert json.loads(descriptor.read_text(encoding="utf-8"))["hf_id"] == str(
        model_directory.resolve()
    )


@pytest.mark.parametrize("artifact_path", ["../outside.npy", "absolute"])
def test_accuracy_rejects_candidate_artifacts_outside_the_cell_directory(
    tmp_path: Path, monkeypatch, artifact_path: str
) -> None:
    case = _example_case(tmp_path)
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    outside = tmp_path / "outside.npy"
    reported_path = str(outside.resolve()) if artifact_path == "absolute" else artifact_path

    monkeypatch.setattr(
        qualification_accuracy,
        "require_candidate",
        lambda _context: (tmp_path / "worker", tmp_path / "runtime"),
    )
    monkeypatch.setattr(
        qualification_accuracy,
        "write_model_descriptor",
        lambda *_args, **_kwargs: tmp_path / "model.json",
    )
    monkeypatch.setattr(
        qualification_accuracy,
        "prepare_bundle",
        lambda *_args, **_kwargs: tmp_path / "model.bundle",
    )
    def candidate(command, *_args, **_kwargs):
        candidate_output = Path(command[command.index("--output") + 1])
        candidate_output.mkdir(parents=True)
        (candidate_output / "result.json").write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "status": "completed",
                            "artifact_dir": "artifacts/case",
                            "output_summary": {"mask_artifact": reported_path},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(qualification_accuracy, "run_command", candidate)

    output = tmp_path / "qualification"
    output.mkdir()
    with pytest.raises(QualificationError, match="unsafe artifact path"):
        qualification_accuracy._candidate_outputs(
            case,
            context,
            output,
            "segment",
            [{"request": {"image_path": "/data/image.jpg"}}],
        )


@pytest.mark.parametrize(
    ("task", "public_request", "expected_inputs", "expected_controls"),
    [
        (
            "stereo_disparity",
            {
                "left_image_path": "/data/left.png",
                "right_image_path": "/data/right.png",
                "height": 700,
                "width": 700,
            },
            {"left_image": "/data/left.png", "right_image": "/data/right.png"},
            {"height": 700, "width": 700},
        ),
        (
            "robot_control",
            {"image_path": "/data/observation.png", "state_path": "/data/state.f32"},
            {"image": "/data/observation.png", "state": "/data/state.f32"},
            {},
        ),
    ],
)
def test_candidate_descriptor_uses_manifest_input_contract(
    tmp_path: Path,
    task: str,
    public_request: dict[str, object],
    expected_inputs: dict[str, str],
    expected_controls: dict[str, object],
) -> None:
    case = QualificationCase(
        kind="performance",
        model="example-model",
        family="example",
        name="case",
        benchmark="example_performance",
        candidate={
            "family": "example",
            "checkpoint": "example/model",
            "task": task,
            "precision": "fp16",
            "build": {},
        },
        values={},
        source=tmp_path / "example.yaml",
        reference_requirements=None,
    )

    descriptor = write_model_descriptor(case, tmp_path, public_request)
    testcase = json.loads(descriptor.read_text(encoding="utf-8"))["testcases"][0]

    assert testcase["inputs"] == expected_inputs
    for name, value in expected_controls.items():
        assert testcase[name] == value
    assert not (set(public_request) & set(testcase["inputs"]))


def test_internal_subprocesses_can_import_repository_packages(tmp_path: Path) -> None:
    completed = run_command(
        [
            sys.executable,
            "-c",
            "import tensorrt_model_connect; import trtmc_benchmark",
        ],
        tmp_path,
        "repository-imports",
        timeout=30,
        verbose=False,
        env={"PATH": os.environ["PATH"]},
    )

    assert completed.returncode == 0, completed.stderr


def test_family_reference_environment_inherits_parent_venv_packages(
    tmp_path: Path, monkeypatch
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("transformers==4.46.3\n", encoding="utf-8")
    case = QualificationCase(
        kind="accuracy",
        model="custom-environment-model",
        family="example",
        name="continuation",
        benchmark="mmlu_continuation",
        candidate={
            "family": "example",
            "checkpoint": "example/model",
            "task": "text_generation",
            "precision": "fp16",
            "build": {},
        },
        values={},
        source=tmp_path / "example.yaml",
        reference_requirements=requirements,
        reference_build_isolation=False,
    )
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=False,
        verbose=False,
    )
    parent_packages = tmp_path / "parent-venv/site-packages"
    parent_packages.mkdir(parents=True)
    commands = []
    environments = []
    timeouts = []

    def complete(command, *_args, **_kwargs):
        commands.append(command)
        environments.append(_kwargs.get("env"))
        timeouts.append(_kwargs.get("timeout"))
        if command[1:3] == ["-m", "venv"]:
            environment = Path(command[-1])
            (environment / "bin").mkdir(parents=True)
            (environment / "bin/python").write_text("", encoding="utf-8")
            child_packages = environment / (
                f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            )
            child_packages.mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("qualification_tests.benchmark_qualification.runtime.run_command", complete)
    monkeypatch.setattr("site.getsitepackages", lambda: [str(parent_packages)])

    python = reference_python(case, context)

    inherited = python.parents[1] / (
        f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/"
        "trtmc-parent-environment.pth"
    )
    assert inherited.read_text(encoding="utf-8") == f"{parent_packages.resolve()}\n"
    pip_command = next(command for command in commands if command[1:3] == ["-m", "pip"])
    assert "--no-build-isolation" in pip_command
    pip_index = commands.index(pip_command)
    assert environments[pip_index]["MAX_JOBS"] == "4"
    assert timeouts[pip_index] == 7200


def test_family_environment_hook_runs_after_requirements_install(
    tmp_path: Path, monkeypatch
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example==1\n", encoding="utf-8")
    hook = tmp_path / "prepare_environment.py"
    hook.write_text("# family hook\n", encoding="utf-8")
    case = QualificationCase(
        kind="accuracy",
        model="custom-environment-model",
        family="example",
        name="continuation",
        benchmark="mmlu_continuation",
        candidate={
            "family": "example",
            "checkpoint": "example/model",
            "task": "text_generation",
            "precision": "fp16",
            "build": {},
        },
        values={},
        source=tmp_path / "example.yaml",
        reference_requirements=requirements,
        environment_hook=hook,
    )
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=False,
        verbose=False,
    )
    commands: list[list[str]] = []
    parent_packages = tmp_path / "parent/site-packages"
    parent_packages.mkdir(parents=True)

    def complete(command, *_args, **_kwargs):
        commands.append(list(command))
        if command[1:3] == ["-m", "venv"]:
            environment = Path(command[-1])
            (environment / "bin").mkdir(parents=True)
            (environment / "bin/python").write_text("", encoding="utf-8")
            child = environment / (
                f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            )
            child.mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("qualification_tests.benchmark_qualification.runtime.run_command", complete)
    monkeypatch.setattr("site.getsitepackages", lambda: [str(parent_packages)])

    python = reference_python(case, context)

    assert commands[-1] == [str(python), str(hook)]


def test_bundle_preparation_uses_the_selected_runtime(tmp_path: Path, monkeypatch) -> None:
    case = QualificationCase(
        kind="accuracy",
        model="example-model",
        family="example",
        name="continuation",
        benchmark="example_accuracy",
        candidate={
            "family": "example",
            "checkpoint": "example/model",
            "task": "text_generation",
            "precision": "fp16",
            "bundle": "example.bundle",
            "build": {"max_sequence_length": 16},
        },
        values={},
        source=tmp_path / "example.yaml",
        reference_requirements=None,
    )
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    bundle = tmp_path / "example.bundle"
    bundle.write_bytes(b"bundle")
    descriptor = tmp_path / "candidate-model.json"
    descriptor.write_text("{}\n", encoding="utf-8")
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=runtime_root,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    captured: list[str] = []

    def complete(command, *_args, **_kwargs):
        captured.extend(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"bundles": [{"model": case.model, "bundle": str(bundle)}]}),
            stderr="",
        )

    monkeypatch.setattr("qualification_tests.benchmark_qualification.runtime.run_command", complete)

    assert prepare_bundle(case, context, tmp_path, descriptor) == bundle
    assert captured[0] == str(context.trtmc_bench)
    assert captured[captured.index("--runtime-root") + 1] == str(runtime_root)


def test_manual_dataset_path_is_supplied_by_the_internal_invocation(tmp_path: Path) -> None:
    dataset = tmp_path / "private.jsonl"
    dataset.write_text('{"input":"example"}\n', encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "cache",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={"private": dataset},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )

    resolved = resolve_dataset(
        {
            "dataset": {
                "id": "private",
                "sha256": digest,
                "source": {"mode": "manual"},
            }
        },
        context,
    )

    assert resolved.path == dataset
    assert resolved.receipt() == {
        "id": "private",
        "source_mode": "provided",
        "sha256": digest,
    }


def test_public_dataset_download_is_pinned_and_cached(tmp_path: Path, monkeypatch) -> None:
    payload = b'{"input":"public"}\n'
    digest = hashlib.sha256(payload).hexdigest()
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "cache",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )

    def download(_url, destination):
        Path(destination).write_bytes(payload)

    monkeypatch.setattr("urllib.request.urlretrieve", download)
    definition = {
        "dataset": {
            "id": "public",
            "sha256": digest,
            "source": {
                "mode": "download",
                "url": "https://example.invalid/public.jsonl",
            },
        }
    }

    resolved = resolve_dataset(definition, context)

    assert resolved.path == tmp_path / "cache/public/public.jsonl"
    assert resolved.path.read_bytes() == payload
    assert resolve_dataset(definition, context).path == resolved.path


def test_family_dataset_stays_with_its_discovered_model_profile(tmp_path: Path) -> None:
    profile = tmp_path / "families/stereo/tests/benchmark/stereo.yaml"
    dataset = tmp_path / "families/stereo/tests/data/pairs.json"
    profile.parent.mkdir(parents=True)
    dataset.parent.mkdir(parents=True)
    dataset.write_text('{"requests": []}\n', encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    context = RuntimeContext(
        repository=tmp_path,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "cache",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    definition = {
        "dataset": {
            "id": "family-pairs",
            "sha256": digest,
            "source": {"mode": "family", "path": "../data/pairs.json"},
        }
    }

    resolved = resolve_dataset(definition, context, profile)

    assert resolved.path == dataset
    assert resolved.receipt()["source_mode"] == "family"
    definition["dataset"]["source"]["path"] = "../../../../outside.json"
    with pytest.raises(QualificationError, match="inside its family"):
        resolve_dataset(definition, context, profile)


def test_stereo_accuracy_compares_complete_disparity_artifacts(tmp_path: Path, monkeypatch) -> None:
    profile = tmp_path / "families/stereo/tests/benchmark/stereo.yaml"
    reference_script = profile.parent / "reference.py"
    reference_script.parent.mkdir(parents=True)
    reference_script.write_text("# family reference\n", encoding="utf-8")
    data = tmp_path / "data"
    data.mkdir()
    for name in ("left.png", "right.png"):
        (data / name).write_bytes(b"fixture")
    dataset_path = data / "pairs.json"
    dataset_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "id": "pair",
                        "left_image": "left.png",
                        "right_image": "right.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = Dataset("stereo", dataset_path, "provided", "digest")
    case = QualificationCase(
        kind="accuracy",
        model="stereo-model",
        family="stereo",
        name="parity",
        benchmark="stereo_disparity_parity",
        candidate={
            "family": "stereo",
            "checkpoint": "example/stereo",
            "task": "stereo_disparity",
            "precision": "fp16",
            "build": {},
        },
        values={
            "samples": 1,
            "request": {"height": 1, "width": 2},
            "reference": {"command": "reference.py", "precision": "fp16"},
            "gate": {
                "min_cosine": 0.99,
                "max_mean_abs_error": 0.2,
                "max_bad_2px_fraction": 0.0,
                "min_sample_pass_rate": 1.0,
            },
        },
        source=profile,
        reference_requirements=None,
    )
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data-root",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    reference_artifact = tmp_path / "reference.f32"
    reference_artifact.write_bytes(array("f", [1.0, 2.0]).tobytes())
    candidate_artifact = tmp_path / "candidate.f32"
    candidate_artifact.write_bytes(array("f", [1.1, 2.1]).tobytes())

    def run_reference(command, *_args, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "sample_id": "pair",
                            "height": 1,
                            "width": 2,
                            "element_count": 2,
                            "disparity_artifact": str(reference_artifact),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        qualification_accuracy,
        "load_benchmark",
        lambda *_args: {"metric": {"name": "stereo_disparity_parity"}},
    )
    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        qualification_accuracy, "reference_python", lambda *_args: Path(sys.executable)
    )
    monkeypatch.setattr(qualification_accuracy, "run_command", run_reference)
    monkeypatch.setattr(
        qualification_accuracy,
        "_candidate_outputs",
        lambda *_args: (
            [
                {
                    "height": 1,
                    "width": 2,
                    "element_count": 2,
                    "disparity_artifact": str(candidate_artifact),
                }
            ],
            tmp_path / "stereo.bundle",
        ),
    )

    result = qualification_accuracy.run_accuracy(case, context)

    assert result["status"] == "passed"
    assert result["metrics"]["sample_pass_rate"] == 1.0
    assert result["metrics"]["max_mean_abs_error"] == pytest.approx(0.1)


def test_metric_geometry_accuracy_compares_complete_task_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    profile = tmp_path / "families/example/tests/benchmark/example-geometry.yaml"
    profile.parent.mkdir(parents=True)
    image = tmp_path / "data/image.jpeg"
    image.parent.mkdir()
    image.write_bytes(b"fixture")
    dataset_path = image.parent / "dataset.json"
    dataset_path.write_text(
        json.dumps({"requests": [{"id": "image", "image": "image.jpeg"}]}),
        encoding="utf-8",
    )
    dataset = Dataset("geometry", dataset_path, "provided", "digest")

    def geometry(root: Path) -> dict[str, object]:
        root.mkdir()
        (root / "points.f32").write_bytes(
            array("f", [1.0, 2.0, 3.0, float("inf"), float("inf"), float("inf")]).tobytes()
        )
        (root / "depth.f32").write_bytes(array("f", [3.0, float("inf")]).tobytes())
        (root / "mask.u8").write_bytes(bytes((1, 0)))
        return {
            "geometry_images": 1,
            "geometry_pixels": 2,
            "height": 1,
            "width": 2,
            "point_shape": [1, 2, 3],
            "valid_pixels": 1,
            "normalized_intrinsics": [
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
                [0.0, 0.0, 1.0],
            ],
            "units": "meters",
            "camera_axes": ["right", "down", "forward"],
            "intrinsics_coordinates": "normalized_uv",
            "points_artifact": str(root / "points.f32"),
            "depth_artifact": str(root / "depth.f32"),
            "valid_mask_artifact": str(root / "mask.u8"),
        }

    candidate = geometry(tmp_path / "candidate")
    reference = geometry(tmp_path / "reference")
    thresholds = {
        "mask_iou": 0.999,
        "depth_absrel_mean": 0.005,
        "depth_rel_l2": 0.02,
        "points_rel_l2": 0.02,
        "points_cosine": 0.99999,
        "intrinsics_max_relative_error": 0.002,
        "point_depth_consistency": 0.00001,
        "min_sample_pass_rate": 1.0,
    }
    case = QualificationCase(
        kind="accuracy",
        model="example-geometry",
        family="example",
        name="geometry",
        benchmark="metric_geometry_parity",
        candidate={
            "family": "example",
            "checkpoint": "example/geometry",
            "task": "monocular_geometry",
            "precision": "fp32",
            "build": {},
        },
        values={
            "samples": 1,
            "request": {"num_tokens": 1800},
            "reference": {"command": "reference.py", "precision": "fp32"},
            "gate": thresholds,
        },
        source=profile,
        reference_requirements=None,
    )
    context = RuntimeContext(
        repository=REPOSITORY,
        artifacts=tmp_path / "artifacts",
        data_root=tmp_path / "data-root",
        environment_root=tmp_path / "envs",
        bundle_cache=tmp_path / "bundles",
        bundle_roots=(),
        runtime_root=None,
        trtmc_bench=tmp_path / "trtmc-bench",
        worker=None,
        datasets={},
        reference_pythons={},
        no_build=True,
        verbose=False,
    )
    monkeypatch.setattr(
        qualification_accuracy,
        "load_benchmark",
        lambda *_args: {"metric": {"name": "metric_geometry_parity"}},
    )
    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
    monkeypatch.setattr(
        qualification_accuracy, "_family_image_reference", lambda *_args: [reference]
    )
    monkeypatch.setattr(
        qualification_accuracy,
        "_candidate_outputs",
        lambda *_args: ([candidate], tmp_path / "example.bundle"),
    )

    result = qualification_accuracy.run_accuracy(case, context)

    assert result["status"] == "passed"
    assert result["metrics"]["samples"] == 1
    assert result["metrics"]["sample_pass_rate"] == 1.0
    assert result["metrics"]["mask_iou"] == 1.0
    assert result["metrics"]["points_cosine"] == pytest.approx(1.0)
    for name in (
        "depth_absrel_mean",
        "depth_rel_l2",
        "points_rel_l2",
        "intrinsics_max_relative_error",
        "point_depth_consistency",
    ):
        assert result["metrics"][name] == 0.0
