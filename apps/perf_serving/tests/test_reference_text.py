# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")

from trtmc_perf_serving.backends.base import BackendError  # noqa: E402
from trtmc_perf_serving.backends.reference.text import (  # noqa: E402
    _generation_kwargs, _language_controls, _place_source_language, encoder_input_limit,
)


class NllbTokenizer:
    src_lang = "eng_Latn"
    unk_token_id = 3
    ids = {"eng_Latn": 256047, "fra_Latn": 256057}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def convert_ids_to_tokens(self, token_id):
        return {value: key for key, value in self.ids.items()}[token_id]


class PairTokenizer:  # a fixed-pair tokenizer without language APIs (Marian-style)
    unk_token_id = 3

    def convert_ids_to_tokens(self, token_id):
        return f"<{token_id}>"


def test_generation_is_greedy_with_one_beam_unless_requested():
    assert _generation_kwargs({"max_new_tokens": 20}) == {"max_new_tokens": 20, "num_beams": 1, "do_sample": False}
    assert _generation_kwargs({"num_beams": 4})["num_beams"] == 4
    assert _generation_kwargs({"repetition_penalty": 1.2})["repetition_penalty"] == 1.2


def test_nllb_generate_requests_keep_their_language_controls():
    tokenizer = NllbTokenizer()
    request = {"source_language": "eng_Latn", "source_language_token_id": 256047, "target_language": "fra_Latn",
               "forced_bos_token_id": 256057}
    controls, manual = _language_controls(tokenizer, request)
    assert controls == {"forced_bos_token_id": 256057} and manual is None and tokenizer.src_lang == "eng_Latn"
    with pytest.raises(BackendError):
        _language_controls(tokenizer, {**request, "forced_bos_token_id": 1})
    assert _language_controls(tokenizer, {}) == ({}, None)


def test_a_source_token_replaces_the_final_unknown_token_for_pair_tokenizers():
    request = {"source_language": "<7>", "source_language_token_id": 7, "source_language_placement": "replace-final-unk"}
    controls, manual = _language_controls(PairTokenizer(), request)
    assert (controls, manual) == ({}, 7)
    inputs = {"input_ids": torch.tensor([[5, 6, 3, 0]]), "attention_mask": torch.tensor([[1, 1, 1, 0]])}
    _place_source_language(inputs, manual, PairTokenizer())
    assert inputs["input_ids"].tolist() == [[5, 6, 7, 0]]


def test_encoder_inputs_are_cut_at_the_declared_limit():
    from types import SimpleNamespace

    assert encoder_input_limit(SimpleNamespace(model_max_length=512), SimpleNamespace(max_position_embeddings=514)) == 512
    unbounded = SimpleNamespace(model_max_length=int(1e30))  # tokenizers without a declared maximum
    assert encoder_input_limit(unbounded, SimpleNamespace(max_position_embeddings=8192)) == 8192
    assert encoder_input_limit(unbounded, SimpleNamespace()) is None
