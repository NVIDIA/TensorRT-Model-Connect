# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from families.gpt2.tests.qualification import executor
from families.gpt2.tests.qualification.executor import (
    _compare,
    _load_samples,
    _request_prompt,
    _run_reference,
    _truncate_prompt,
)
from trtmc_benchmark.qualification import QualificationCatalog


REPOSITORY = Path(__file__).resolve().parents[4]


def test_checked_in_gpt2_accuracy_suite_is_discoverable() -> None:
    plan = QualificationCatalog(REPOSITORY / "families").plan("accuracy", models=["gpt2-125m"])

    assert [(item.suite_id, item.case_id) for item in plan.items] == [
        ("mmlu_continuation_parity", "smoke")
    ]
    assert plan.items[0].definition["implementation"] == "mmlu_continuation_parity"
    assert "device" not in plan.items[0].case


def test_checked_in_gpt2_performance_suite_is_discoverable() -> None:
    plan = QualificationCatalog(REPOSITORY / "families").plan(
        "performance", models=["gpt2-125m"]
    )

    assert [(item.suite_id, item.case_id) for item in plan.items] == [
        ("text_generation_performance", "generate_64")
    ]
    assert plan.items[0].gate_policy == "observation_only"
    assert plan.items[0].case["candidate"]["measurement"] == {
        "warmup": 5,
        "iterations": 20,
    }
    assert "device" not in plan.items[0].case


def test_gpt2_performance_reports_measurements_without_a_device_gate(
    tmp_path: Path, monkeypatch
) -> None:
    item = (
        QualificationCatalog(REPOSITORY / "families")
        .plan("performance", models=["gpt2-125m"])
        .items[0]
    )
    candidate = {
        "measurement_policy": {
            "timing_scope": "public_task_call_wall",
            "load_excluded": True,
            "warmup_excluded": True,
            "telemetry_in_timed_path": False,
        },
        "environment": {"gpus": [{"name": "test-gpu"}]},
        "preparation": {"included_in_performance_metrics": False, "bundles": []},
        "cells": [
            {
                "name": "generate_64",
                "status": "completed",
                "samples_ms": [4.0, 5.0],
                "metrics": {
                    "sample_count": 2,
                    "latency_ms": {"p50": 4.5, "p95": 4.95},
                    "request_throughput_per_s": 222.2,
                    "output_tokens_per_s": 14222.2,
                },
            }
        ],
    }
    monkeypatch.setattr(
        executor,
        "_run_performance_candidate",
        lambda **_kwargs: candidate,
    )

    result = executor._execute(
        {
            "schema_version": "trtmc.qualification-executor-request/v1",
            "environment": {"schema_version": "trtmc.qualification-environment/v1"},
        },
        item.to_json(),
        tmp_path,
    )

    assert result["execution"] == "completed"
    assert result["verdict"] is None
    assert result["details"]["metrics"]["latency_ms"]["p50"] == 4.5
    assert result["details"]["runtime_environment"]["gpus"][0]["name"] == "test-gpu"


def test_gpt2_manifest_covers_the_accuracy_context_window() -> None:
    item = QualificationCatalog(REPOSITORY / "families").plan(
        "accuracy", models=["gpt2-125m"]
    ).items[0]
    manifest = json.loads(item.manifest_path.read_text(encoding="utf-8"))

    assert (
        item.case["prompt"]["token_limit"]
        + item.case["candidate"]["request"]["max_new_tokens"]
        <= manifest["max_sequence_length"]
    )


def test_mmlu_samples_are_selected_without_padding_or_repetition(tmp_path: Path) -> None:
    dataset_root = tmp_path / "data"
    dataset_path = dataset_root / "MMLU_five_shot" / "mmlu_dataset.json"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "id": f"sample-{index}",
                        "subject": "math",
                        "answer": "A",
                        "messages": [{"role": "user", "content": f"question {index}"}],
                    }
                    for index in range(3)
                ]
            }
        ),
        encoding="utf-8",
    )
    definition = {
        "dataset": {"relative_path": "MMLU_five_shot/mmlu_dataset.json"},
        "selection": {"method": "first"},
    }
    environment = {"storage": {"data_root": str(dataset_root)}}

    samples, resolved = _load_samples(definition, {"sample_limit": 10}, environment)

    assert resolved == dataset_path
    assert [sample["sample_id"] for sample in samples] == [
        "sample-0",
        "sample-1",
        "sample-2",
    ]


def test_mmlu_prompt_prefers_the_last_user_message() -> None:
    assert (
        _request_prompt(
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": ""},
                ],
                "prompt": "fallback",
            }
        )
        == "question"
    )


def test_reference_preserves_virtual_environment_python_symlink(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "base-python"
    target.write_text("", encoding="utf-8")
    virtualenv_python = tmp_path / "venv-python"
    virtualenv_python.symlink_to(target)
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "schema_version": "trtmc.gpt2-reference-result/v1",
                    "samples": [],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(executor.subprocess, "run", run)

    _run_reference(
        manifest={"hf_id": "openai-community/gpt2"},
        definition={},
        case={
            "reference": {"implementation": "hf_transformers"},
            "prompt": {},
            "candidate": {"request": {}},
        },
        samples=[],
        environment={
            "tools": {"reference_python": str(virtualenv_python)},
            "execution": {"timeout_seconds": 1},
        },
        item_dir=tmp_path / "result",
    )

    assert commands[0][0] == str(virtualenv_python)


def test_exact_token_gate_preserves_the_old_sample_acceptance_rule() -> None:
    definition = {"scoring": {"implementation": "continuation_token_parity"}}
    case = {"gate": {"min_pass_rate": 0.9, "min_allowed_failures": 1}}
    item = {"gate_policy": "blocking"}
    reference = {
        "samples": [
            {"sample_id": f"sample-{index}", "token_ids": [index], "text": str(index)}
            for index in range(10)
        ]
    }

    def candidate(mismatches: set[int]):
        return {
            "cells": [
                {
                    "name": f"sample-{index}",
                    "output_summary": {
                        "token_ids": [-1] if index in mismatches else [index],
                        "text": str(index),
                    },
                }
                for index in range(10)
            ]
        }

    accepted = _compare(definition, case, item, reference, candidate({0}))
    rejected = _compare(definition, case, item, reference, candidate({0, 1}))

    assert accepted["allowed_failure_count"] == 1
    assert accepted["verdict"] == "pass"
    assert rejected["verdict"] == "fail"
    assert rejected["metrics"]["tie_adjusted_exact_match_rate"] == 0.8


def test_first_divergence_at_an_exact_reference_tie_is_sample_equivalent() -> None:
    definition = {"scoring": {"implementation": "continuation_token_parity"}}
    case = {"gate": {"min_pass_rate": 1.0, "min_allowed_failures": 0}}
    reference = {
        "samples": [
            {
                "sample_id": "tie",
                "token_ids": [10, 20],
                "text": "reference",
                "generated_token_max_score_ids": [[10, 11], [20]],
            }
        ]
    }
    candidate = {
        "cells": [
            {
                "name": "tie",
                "output_summary": {"token_ids": [11, 99], "text": "candidate"},
            }
        ]
    }

    comparison = _compare(
        definition,
        case,
        {"gate_policy": "blocking"},
        reference,
        candidate,
    )

    assert comparison["verdict"] == "pass"
    assert comparison["metrics"]["exact_token_match_rate"] == 0.0
    assert comparison["metrics"]["tie_adjusted_exact_match_rate"] == 1.0
    assert comparison["metrics"]["reference_tie_equivalent_count"] == 1
    assert len(comparison["disagreements"]) == 1


def test_prompt_limit_preserves_the_old_left_truncation() -> None:
    class Tokenizer:
        @staticmethod
        def encode(_prompt, *, add_special_tokens):
            assert not add_special_tokens
            return [1, 2, 3, 4]

        @staticmethod
        def decode(token_ids, **options):
            assert options == {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            }
            return " ".join(str(value) for value in token_ids)

    assert _truncate_prompt(Tokenizer(), "original", 2, "left") == "3 4"
