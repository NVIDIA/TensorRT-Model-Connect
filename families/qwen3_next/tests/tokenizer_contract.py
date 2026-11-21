# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small test models using the pinned Qwen3-Coder-Next tokenizer contract."""

import copy

PIPELINE = {'pre_tokenizer': {'type': 'Sequence', 'pretokenizers': [{'type': 'Split', 'pattern': {'Regex': "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, 'behavior': 'Isolated', 'invert': False}, {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': False, 'use_regex': False}]}, 'normalizer': {'type': 'NFC'}, 'post_processor': {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': False, 'use_regex': False}, 'decoder': {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': False, 'use_regex': False}}


def tokenizer_document(vocabulary=None):
    return {**copy.deepcopy(PIPELINE), "added_tokens": [], "model": {
        "type": "BPE", "vocab": {} if vocabulary is None else vocabulary, "merges": [],
        "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
        "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
        "ignore_merges": False,
    }}
