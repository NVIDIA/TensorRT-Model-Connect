# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FLUX's native CLIP asset preserves checkpoint token IDs and normalization."""

import json

import pytest

from families.flux.model import _clip_tokenizer_bytes


def test_pre_serialized_clip_tokenizer_bytes_are_preserved(tmp_path):
    original = b'{"version":"1.0", "model":{"type":"BPE"}}\n'
    (tmp_path / "tokenizer.json").write_bytes(original)
    assert _clip_tokenizer_bytes(tmp_path) == original


def test_split_clip_assets_keep_vocabulary_merges_special_tokens_and_normalization(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")

    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    symbols = alphabet + [symbol + "</w>" for symbol in alphabet]
    symbols += ["he", "hel", "hell", "hello</w>", "<|startoftext|>", "<|endoftext|>"]
    vocab = {symbol: index for index, symbol in enumerate(symbols)}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\nh e\nhe l\nhel l\nhell o</w>\n")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "CLIPTokenizer", "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>", "model_max_length": 77,
    }))
    reference = transformers.AutoTokenizer.from_pretrained(tmp_path, local_files_only=True, trust_remote_code=False)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    exported = tokenizers.Tokenizer.from_str(_clip_tokenizer_bytes(tmp_path).decode())
    assert exported.encode("HELLO", add_special_tokens=False).ids == [vocab["hello</w>"]]
    for prompt in ["hello", "HELLO, world!", " leading\ntrailing ", "A cat 🚲", "<|endoftext|>"]:
        for special in (False, True):
            assert exported.encode(prompt, add_special_tokens=special).ids == reference.encode(prompt, add_special_tokens=special)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
