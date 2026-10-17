# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model-card sampling case must call the original sampled generation path."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from families.lfm2.tests import lfm25_contract_validation as contract


@pytest.mark.parametrize(
    "name,budget", [("model-card-sampling", 512), ("sampled-answer-completion", 2048)]
)
def test_original_sampling_receives_card_processors_and_seed(name, budget):
    request = dict(contract.cases())[name]
    assert request["messages"] == [{"role": "user", "content": "What is C. elegans?"}]
    expected = {
        "max_new_tokens": budget,
        "temperature": 0.1,
        "top_k": 50,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.1,
        "seed": 42,
    }
    assert request["config"] == expected
    seeds, calls = [], []

    class Generated:
        def __getitem__(self, _):
            return SimpleNamespace(tolist=lambda: [9])

    class Model:
        def generate(self, **kwargs):
            calls.append(kwargs)
            return Generated()

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return "rendered fixture"

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(input_ids=[1, 2])

    torch = SimpleNamespace(
        long="long",
        tensor=lambda values, **kwargs: values,
        inference_mode=nullcontext,
        manual_seed=seeds.append,
    )
    assert contract.reference_case(Model(), Tokenizer(), request, name, torch) == (
        2,
        [9],
    )
    assert seeds == [42]
    assert calls == [
        {
            "input_ids": [[1, 2]],
            "do_sample": True,
            **{key: value for key, value in expected.items() if key != "seed"},
        }
    ]
    calls.clear()
    seeds.clear()
    contract.reference_case(Model(), Tokenizer(), request, "model-card-quickstart", torch)
    assert seeds == []
    assert calls == [
        {
            "input_ids": [[1, 2]],
            "max_new_tokens": budget,
            "do_sample": False,
            "repetition_penalty": 1.0,
        }
    ]
