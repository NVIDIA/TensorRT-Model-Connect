# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from apps.benchmark.performance.baselines.timing_contracts import timing_contract
from tools.benchmark_qualification import accuracy as qualification_accuracy
from tools.benchmark_qualification.catalog import (
    QualificationCase,
    discover,
    load_benchmark,
    select,
)
from tools.benchmark_qualification.datasets import Dataset, resolve_dataset
from tools.benchmark_qualification.references import hf_text_generation
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


def test_one_model_file_owns_multiple_cases_without_testcase_indirection() -> None:
    cases = select(discover(REPOSITORY), ["gpt2-125m"])

    assert {case.kind for case in cases} == {"accuracy", "performance"}
    assert {case.name for case in cases} == {"mmlu-continuation", "generate-64"}
    for case in cases:
        assert case.source.parent.name == "benchmark"
        assert "testcase" not in case.values
        assert "testcase" not in str(case.values)
        assert case.candidate["checkpoint"] == "openai-community/gpt2"


def test_opt_uses_validated_profile_and_pre_refactor_performance_length() -> None:
    cases = select(discover(REPOSITORY), ["opt-125m"])
    accuracy = next(case for case in cases if case.kind == "accuracy")
    performance = next(case for case in cases if case.kind == "performance")

    assert accuracy.candidate["build"]["max_sequence_length"] == 256
    assert accuracy.values["prompt_token_limit"] == 192
    assert performance.name == "generate-10"
    assert performance.values["request"]["max_new_tokens"] == 10


def test_restored_text_profiles_preserve_pre_refactor_performance_lengths() -> None:
    expected = {
        "codegen-350m": 20,
        "deepseek-v2-lite": 10,
        "deepseek-v2-tiny": 10,
        "distilgpt2": 12,
        "falcon3-1b": 20,
        "gemma-2-2b": 10,
        "glm-4-9b": 20,
        "granite-3.1-2b": 20,
        "internlm2-1.8b": 20,
        "marian-en-ru": 20,
        "minitron-4b-depth": 20,
        "minitron-4b-width": 20,
        "mistral-7b": 10,
        "nemotron-hindi-4b": 20,
        "nllb-200-distilled-600m": 20,
        "olmo2-1b": 8,
        "phi3-mini": 10,
        "qwen3-0.6b-fp16": 10,
        "stablelm2-1.6b": 22,
        "starcoder2-3b": 20,
        "riva-translate-4b": 20,
        "t5-small": 20,
    }

    for model, tokens in expected.items():
        cases = select(discover(REPOSITORY), [model])
        performance = next(case for case in cases if case.kind == "performance")
        assert performance.name == f"generate-{tokens}"
        assert performance.values["request"]["max_new_tokens"] == tokens

    stablelm = select(discover(REPOSITORY), ["stablelm2-1.6b"])[0]
    assert stablelm.candidate["build"]["fp32_layers"] == [23]
    minitron_width = select(discover(REPOSITORY), ["minitron-4b-width"])[0]
    assert minitron_width.candidate["build"] == {
        "max_sequence_length": 131072,
        "dynamic_kv_cache": True,
    }
    minitron_depth_cases = select(discover(REPOSITORY), ["minitron-4b-depth"])
    minitron_depth = next(case for case in minitron_depth_cases if case.kind == "accuracy")
    assert minitron_depth.candidate["build"]["max_sequence_length"] == (
        minitron_depth.values["prompt_token_limit"]
        + minitron_depth.values["request"]["max_new_tokens"]
        + 1
    )
    internlm = select(discover(REPOSITORY), ["internlm2-1.8b"])[0]
    assert internlm.reference_requirements == (
        REPOSITORY / "families/internlm/requirements.txt"
    ).resolve()


@pytest.mark.parametrize(
    ("model", "expected_options"),
    [
        ("deepseek-v2-tiny", {"experts_implementation": "batched_mm"}),
        ("internlm2-1.8b", {"trust_remote_code": True}),
    ],
)
def test_mmlu_forwards_reference_model_load_options(
    tmp_path: Path,
    monkeypatch,
    model: str,
    expected_options: dict[str, object],
) -> None:
    case = next(
        case
        for case in select(discover(REPOSITORY), [model])
        if case.kind == "accuracy"
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
    assert {name: captured[name] for name in expected_options} == expected_options


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
    controls = hf_text_generation._translation_controls(
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
    case = select(discover(REPOSITORY), ["gpt2-125m"])[0]

    descriptor = write_model_descriptor(case, tmp_path, {"prompt": "hello"})
    value = descriptor.read_text(encoding="utf-8")

    assert '"hf_id": "openai-community/gpt2"' in value
    assert '"task": "text_generation"' in value
    assert '"max_sequence_length": 1024' in value
    assert "tests/manifests" not in value

    internlm = select(discover(REPOSITORY), ["internlm2-1.8b"])[0]
    descriptor = write_model_descriptor(internlm, tmp_path, {"prompt": "hello"})
    assert json.loads(descriptor.read_text(encoding="utf-8"))["trust_remote_code"] is True


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
    case = select(discover(REPOSITORY), ["internlm2-1.8b"])[0]
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
    case = next(
        case
        for case in select(discover(REPOSITORY), ["gpt2-125m"])
        if case.kind == "accuracy"
    )
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    bundle = tmp_path / "gpt2.bundle"
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
