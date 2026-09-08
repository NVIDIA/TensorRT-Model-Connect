# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from families.gpt2.tests.qualification.executor import (
    _compare,
    _load_samples,
    _request_prompt,
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
