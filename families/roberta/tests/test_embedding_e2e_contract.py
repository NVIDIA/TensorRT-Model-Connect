# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel

from families.roberta.tests import test_e2e as e2e


@pytest.mark.parametrize("task,command", [("embedding", "embed"), ("encoding", "encode")])
def test_native_e2e_dispatches_the_manifest_task(tmp_path, monkeypatch, task, command):
    calls = []
    monkeypatch.setattr(e2e, "_run_json", lambda *args: calls.append(args) or {"values": [1.0]})
    manifest = {"task": task}
    case = {"prompt": "query: a small encoder"}

    e2e._native(tmp_path, tmp_path, tmp_path, tmp_path, manifest, case, tmp_path)

    assert calls[0][-3:] == (command, "--text", case["prompt"])


@pytest.mark.parametrize("task", ["embedding", "encoding"])
def test_reference_e2e_matches_real_xlm_roberta_outputs(tmp_path, monkeypatch, task):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = XLMRobertaModel(
            XLMRobertaConfig(
                vocab_size=19,
                hidden_size=8,
                intermediate_size=12,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=18,
                type_vocab_size=1,
                pad_token_id=1,
            )
        ).eval()
    model.save_pretrained(tmp_path)
    encoded = {
        "input_ids": torch.tensor([[0, 5, 9, 2, 1, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
    }
    calls = []

    def tokenize(text, **kwargs):
        calls.append((text, kwargs))
        return encoded

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenize)
    tensor_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        args = tuple(
            torch.device("cpu") if isinstance(arg, torch.device) and arg.type == "cuda" else arg
            for arg in args
        )
        return tensor_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    manifest = {"task": task, "max_sequence_length": 8}
    case = {"prompt": "query: encoder reference", "reference_precision": "fp32"}

    actual = e2e._official_reference(tmp_path, manifest, case, tmp_path)

    with torch.no_grad():
        hidden = model(**encoded).last_hidden_state
    mean = hidden[0, :4].mean(dim=0)
    expected = torch.nn.functional.normalize(mean, dim=0) if task == "embedding" else hidden[0, 0]
    np.testing.assert_allclose(actual["values"], expected.numpy(), atol=1e-6)
    assert calls == [
        (case["prompt"], {"return_tensors": "pt", "truncation": True, "max_length": 8})
    ]
    if task == "embedding":
        assert not np.allclose(expected.numpy(), hidden[0, 0].numpy())
        assert not np.allclose(
            expected.numpy(), torch.nn.functional.normalize(hidden[0].mean(dim=0), dim=0).numpy()
        )


@pytest.mark.parametrize(
    "values,accepted", [([3.0, 4.0], False), ([0.6, 0.8], True), ([-0.6, -0.8], False)]
)
def test_embedding_parity_requires_normalization_and_matching_direction(values, accepted):
    def compare():
        e2e._assert_parity(
            {"values": values},
            {"values": [0.6, 0.8]},
            {"task": "embedding"},
            {},
            {"contract_cosine_threshold": 0.999},
        )

    if accepted:
        compare()
    else:
        with pytest.raises(AssertionError):
            compare()


def test_embedding_manifest_is_selected_by_the_existing_e2e_selectors():
    config = SimpleNamespace(
        getoption=lambda option: ["multilingual-e5-large"] if option == "--e2e-model" else None
    )
    selected, enabled = e2e._selected_cases(config)

    assert enabled and set(selected) == {
        "multilingual-e5-large-query-en",
        "multilingual-e5-large-passage-zh",
    }
