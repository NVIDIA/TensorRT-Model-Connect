# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json

import sentencepiece as sentencepiece_lib
from tokenizers import Tokenizer

from families.pixart.model import _tokenizer_json_bytes


def test_spiece_model_is_serialized_as_unigram_tokenizer_json(tmp_path) -> None:
    model = io.BytesIO()
    sentencepiece_lib.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello pixart", "hello world"]),
        model_writer=model,
        model_type="unigram",
        vocab_size=24,
        hard_vocab_limit=False,
    )
    model_bytes = model.getvalue()
    (tmp_path / "spiece.model").write_bytes(model_bytes)

    tokenizer_json = _tokenizer_json_bytes(tmp_path)
    payload = json.loads(tokenizer_json)

    assert payload["model"]["type"] == "Unigram"
    assert isinstance(payload["model"]["vocab"], list)
    assert payload["model"]["vocab"]
    text = "hello world"
    sentencepiece = sentencepiece_lib.SentencePieceProcessor(model_proto=model_bytes)
    converted = Tokenizer.from_str(tokenizer_json.decode("utf-8"))
    assert converted.encode(text).ids == sentencepiece.encode(text, out_type=int)
