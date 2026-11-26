# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The smallest tokenizer.json of the layout the native tokenizer loads (as the host test's)."""

from __future__ import annotations

from families.minicpmv4_6 import prompt


def minimal_tokenizer() -> dict:
    """256 byte tokens, two merged tokens and one added token: 259 token ids."""
    vocab = {character: index for index, character in enumerate(prompt._byte_alphabet())}
    vocab.update({"xy": 256, "xyx": 257})
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    split = {"type": "Split", "pattern": {"Regex": prompt._SPLIT_PATTERN}, "behavior": "Isolated", "invert": False}
    return {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": 258, "content": "<|end|>", "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True}],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [split, dict(byte_level)]},
        "post_processor": dict(byte_level), "decoder": dict(byte_level),
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [["xy", "x"], ["x", "y"]]},
    }
