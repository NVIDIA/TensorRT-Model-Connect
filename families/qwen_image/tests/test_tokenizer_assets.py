# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tokenizer export preserves the checkpoint's vocabulary and added tokens."""

import json

import pytest

from families.qwen_image.model import _tokenizer_bytes


def test_pre_serialized_tokenizer_bytes_are_preserved(tmp_path):
    original = b'{"version":"1.0", "model":{"type":"BPE"}}\n'
    (tmp_path / "tokenizer.json").write_bytes(original)
    assert _tokenizer_bytes(tmp_path) == original


def test_vocab_merges_snapshot_exports_identical_token_ids_without_cache_writes(tmp_path):
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")

    vocab = {symbol: index for index, symbol in enumerate(sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet()))}
    vocab["<|endoftext|>"] = len(vocab)
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>",
        "added_tokens_decoder": {str(256 + index): {"content": token, "special": True,
            "normalized": False, "lstrip": False, "rstrip": False, "single_word": False}
            for index, token in enumerate(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])},
    }))
    reference = transformers.AutoTokenizer.from_pretrained(tmp_path, local_files_only=True, trust_remote_code=False)
    assert not (tmp_path / "tokenizer.json").exists()
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    exported = tokenizers.Tokenizer.from_str(_tokenizer_bytes(tmp_path).decode())
    assert exported.encode("abc", add_special_tokens=False).ids == [vocab[ch] for ch in "abc"]
    assert exported.encode("<|im_start|>", add_special_tokens=False).ids == [257]
    for prompt in ["a cat", "Hello, 世界!", " leading\ntrailing ", "<|im_start|>user\ncat<|im_end|>"]:
        assert exported.encode(prompt, add_special_tokens=False).ids == reference.encode(prompt, add_special_tokens=False)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
