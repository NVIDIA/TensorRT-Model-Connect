# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The pinned LocateAnything checkpoint exports vocab/merges without tokenizer.json."""

import pytest

pytest.importorskip("tensorrt")
pytest.importorskip("transformers")


def test_vocab_merges_export_preserves_coordinate_and_unicode_token_ids(tmp_path):
    from tokenizers import ByteLevelBPETokenizer, Tokenizer
    from transformers import Qwen2TokenizerFast
    from families.locateanything.model import _native_tokenizer, _tokenizer_runtime_contract
    from families.locateanything.tests.hf_reference import _LocalTokenizer

    markers = ["<ref>", "</ref>", "<box>", "</box>", "<0>", "<20>", "<999>", "<1000>"]
    probes = ["hello", "Locate a single instance that matches the following description: dog.",
              "<ref>dog</ref><box><0><20><999><1000></box>", "找到图片中的狗。"]
    trained = ByteLevelBPETokenizer()
    trained.train_from_iterator(probes, vocab_size=300, special_tokens=markers)
    trained.save_model(str(tmp_path))
    original = Qwen2TokenizerFast(
        vocab_file=str(tmp_path / "vocab.json"), merges_file=str(tmp_path / "merges.txt"),
        additional_special_tokens=markers)
    original.save_pretrained(tmp_path)
    (tmp_path / "tokenizer.json").unlink()

    native = _native_tokenizer(tmp_path)
    restored = Tokenizer.from_str(native.to_str())
    reference = _LocalTokenizer(tmp_path)
    for text in probes:
        assert restored.encode(text, add_special_tokens=False).ids == original.encode(
            text, add_special_tokens=False)
        assert reference.encode(text) == original.encode(text, add_special_tokens=False)
    framing = _tokenizer_runtime_contract(native)
    assert framing == {"tokenizer_add_special_tokens": False,
                       "tokenizer_prefix_ids": [], "tokenizer_suffix_ids": []}
