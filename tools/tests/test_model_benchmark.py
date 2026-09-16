# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

from tools.benchmark_qualification.catalog import discover, load_benchmark, select
from tools.benchmark_qualification.datasets import resolve_dataset
from tools.benchmark_qualification.runtime import RuntimeContext, write_model_descriptor


REPOSITORY = Path(__file__).resolve().parents[2]


def test_family_configs_auto_discover_both_kinds_without_l0() -> None:
    cases = discover(REPOSITORY)

    assert {(case.model, case.kind) for case in cases} == {
        ("gpt2-125m", "accuracy"),
        ("gpt2-125m", "performance"),
        ("chronos-bolt-tiny-official", "accuracy"),
        ("chronos-bolt-tiny-official", "performance"),
    }
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


def test_shared_definitions_own_dataset_and_metric_not_models() -> None:
    for case in discover(REPOSITORY):
        definition = load_benchmark(REPOSITORY, case)
        assert "models" not in definition
        assert definition["kind"] == case.kind
        if case.kind == "accuracy":
            assert definition["dataset"]["id"]
            assert definition["metric"]["name"]


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
