# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from apps.benchmark.performance.baselines import hf_transformers
from apps.benchmark.performance.baselines.timing_contracts import timing_contract
from tools.benchmark_qualification import accuracy as qualification_accuracy
from tools.benchmark_qualification.catalog import (
    QualificationCase,
    QualificationError,
    discover,
    load_benchmark,
)
from tools.benchmark_qualification.datasets import Dataset, resolve_dataset
from tools.benchmark_qualification.references import hf_encoder, hf_text_generation
from tools.benchmark_qualification.runtime import (
    RuntimeContext,
    prepare_bundle,
    reference_python,
    run_command,
    write_model_descriptor,
)


REPOSITORY = Path(__file__).resolve().parents[2]


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


def test_accuracy_forwards_declared_reference_model_load_options(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
                "precision": "fp16",
                "experts_implementation": "batched_mm",
            },
            "request": {"max_new_tokens": 1, "temperature": 0.0},
            "gate": {"min_pass_rate": 1.0, "allowed_failures": 0},
        },
        source=tmp_path / "example.yaml",
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

    def reference(command, *_args, **_kwargs):
        request = Path(command[command.index("--request") + 1])
        output = Path(command[command.index("--output") + 1])
        captured.update(json.loads(request.read_text(encoding="utf-8")))
        output.write_text(
            json.dumps(
                {
                    "samples": [
                        {"sample_id": "sample", "prompt": "Question", "token_ids": [1]}
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(qualification_accuracy, "resolve_dataset", lambda *_args: dataset)
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
    assert captured["revision"] == "a" * 40
    assert captured["trust_remote_code"] is True
    assert captured["experts_implementation"] == "batched_mm"


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
    assert hf_text_generation._precision_load_options("fp16") == {
        "torch_dtype": "fp16"
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
def test_hf_translation_supports_transformers5_generic_nllb_tokenizer(runner) -> None:
    import torch

    class GenericNllbTokenizer:
        unk_token_id = 3

        @staticmethod
        def convert_tokens_to_ids(value: str) -> int:
            return {"eng_Latn": 256047, "fra_Latn": 256057}.get(value, 3)

        @staticmethod
        def convert_ids_to_tokens(value: int) -> str:
            return {256047: "eng_Latn", 256057: "fra_Latn"}[value]

    tokenizer = GenericNllbTokenizer()
    controls, source_token_id = runner._translation_controls(
        tokenizer,
        {
            "source_language": "eng_Latn",
            "source_language_token_id": 256047,
            "target_language": "fra_Latn",
            "forced_bos_token_id": 256057,
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
            assert definition["dataset"]["id"]
            assert definition["metric"]["name"]


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
        assert timing_contract(
            runner=str(case.values["reference"]["runner"]), declared=declared
        )


def test_internal_automation_is_separate_from_the_installed_benchmark() -> None:
    old_application = REPOSITORY / "apps/benchmark/qualification"
    assert list(old_application.rglob("*.py")) == []
    internal_sources = [
        path.read_text(encoding="utf-8")
        for path in (REPOSITORY / "tools/benchmark_qualification").rglob("*.py")
    ]
    assert not any("ManifestCatalog" in source for source in internal_sources)
    assert not any("tests/manifests" in source for source in internal_sources)
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

    def complete(command, *_args, **_kwargs):
        if command[1:3] == ["-m", "venv"]:
            environment = Path(command[-1])
            (environment / "bin").mkdir(parents=True)
            (environment / "bin/python").write_text("", encoding="utf-8")
            child_packages = environment / (
                f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
            )
            child_packages.mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("tools.benchmark_qualification.runtime.run_command", complete)
    monkeypatch.setattr("site.getsitepackages", lambda: [str(parent_packages)])

    python = reference_python(case, context)

    inherited = python.parents[1] / (
        f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/"
        "trtmc-parent-environment.pth"
    )
    assert inherited.read_text(encoding="utf-8") == f"{parent_packages.resolve()}\n"


def test_bundle_preparation_uses_the_selected_runtime(
    tmp_path: Path, monkeypatch
) -> None:
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

    monkeypatch.setattr("tools.benchmark_qualification.runtime.run_command", complete)

    assert prepare_bundle(case, context, tmp_path, descriptor) == bundle
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
