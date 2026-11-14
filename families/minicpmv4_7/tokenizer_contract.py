# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``tokenizer.json`` contract that the native runtime implements (``runtime/bpe_tokenizer.cpp``).

The builder checks it before any engine is built, so a checkpoint whose tokenizer the runtime would
reject fails at once instead of after a full bundle build. Keep both sides in step.
"""

from __future__ import annotations

import json
from pathlib import Path

_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_BASE_VOCAB_SIZE = 248044
_MERGE_COUNT = 247587
_ADDED_TOKEN_COUNT = 100
_SPECIAL_TOKEN_COUNT = 88
FULL_VOCAB_SIZE = _BASE_VOCAB_SIZE + _ADDED_TOKEN_COUNT
# Chat and image protocol tokens: content -> (id, special).
_PROTOCOL_TOKENS = {
    "<|endoftext|>": (248044, True),
    "<|im_start|>": (248045, True),
    "<|im_end|>": (248046, True),
    "<|image_pad|>": (248056, True),
    "<think>": (248068, False),
    "</think>": (248069, False),
    "<image>": (248078, True),
    "</image>": (248079, True),
    "<slice>": (248088, True),
    "</slice>": (248089, True),
    "<image_id>": (248090, True),
    "</image_id>": (248091, True),
}
_PRE_TOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": _SPLIT_PATTERN}, "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    ],
}
_POST_PROCESSOR = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
_DECODER = {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True}
_MODEL_FIELDS = {
    "type": "BPE",
    "dropout": None,
    "unk_token": None,
    "continuing_subword_prefix": "",
    "end_of_word_suffix": "",
    "fuse_unk": False,
    "byte_fallback": False,
    "ignore_merges": False,
}
_ADDED_TOKEN_FLAGS = {"single_word": False, "lstrip": False, "rstrip": False, "normalized": False}


def _same(value, expected) -> bool:
    # JSON equality that keeps booleans, numbers and null apart (Python treats False == 0).
    return json.dumps(value, sort_keys=True) == json.dumps(expected, sort_keys=True)


def _byte_level_alphabet() -> list[str]:
    direct = [*range(33, 127), *range(161, 173), *range(174, 256)]
    displaced = 0
    alphabet = []
    for byte in range(256):
        if byte in direct:
            alphabet.append(chr(byte))
        else:
            alphabet.append(chr(256 + displaced))
            displaced += 1
    return alphabet


def check_tokenizer_contract(model_dir: Path, vocab_size: int) -> None:
    """Raise ``ValueError`` unless ``tokenizer.json`` is exactly what the runtime implements.

    ``vocab_size`` is the decoder's; it must cover exactly the tokenizer's ids, so that every prompt id has a
    decoder row and every generated id can be decoded.
    """
    if vocab_size != FULL_VOCAB_SIZE:
        raise ValueError(f"minicpmv4_7 decoder vocab_size {vocab_size} differs from the tokenizer's {FULL_VOCAB_SIZE} ids")
    path = model_dir / "tokenizer.json"
    if not path.is_file():
        raise ValueError("minicpmv4_7 requires tokenizer.json (the runtime tokenizer reads it)")
    root = json.loads(path.read_text(encoding="utf-8"))

    def require(condition: bool, detail: str) -> None:
        if not condition:
            raise ValueError(f"minicpmv4_7 tokenizer.json {detail}; the runtime implements the released tokenizer only")

    require(isinstance(root, dict) and len(root) == 9 and root.get("version") == "1.0"
            and root.get("truncation") is None and root.get("padding") is None,
            "has an unexpected top-level shape, version, truncation or padding")
    require(_same(root.get("normalizer"), {"type": "NFC"}), "must use the NFC normalizer")
    require(_same(root.get("pre_tokenizer"), _PRE_TOKENIZER),
            "does not hold the Split and ByteLevel pre-tokenizer the runtime implements")
    require(_same(root.get("post_processor"), _POST_PROCESSOR), "must use the ByteLevel post-processor")
    require(_same(root.get("decoder"), _DECODER), "must use the ByteLevel decoder")

    model = root.get("model")
    require(isinstance(model, dict) and len(model) == 10
            and all(_same(model.get(key), value) for key, value in _MODEL_FIELDS.items()),
            "must use a plain byte-level BPE model")
    vocab, merges = model.get("vocab"), model.get("merges")
    require(isinstance(vocab, dict) and len(vocab) == _BASE_VOCAB_SIZE
            and all(type(i) is int for i in vocab.values())
            and sorted(vocab.values()) == list(range(_BASE_VOCAB_SIZE)),
            "has an unexpected BPE vocabulary")
    require(all(token in vocab for token in _byte_level_alphabet()), "misses a byte-level vocabulary token")
    require(isinstance(merges, list) and len(merges) == _MERGE_COUNT
            and all(isinstance(m, list) and len(m) == 2 and all(isinstance(p, str) for p in m) for m in merges)
            and len({(m[0], m[1]) for m in merges}) == _MERGE_COUNT,
            "has an unexpected BPE merge list")

    added = root.get("added_tokens")
    require(isinstance(added, list) and len(added) == _ADDED_TOKEN_COUNT, "has an unexpected added-token count")
    contents: dict[str, tuple[int, bool]] = {}
    for index, entry in enumerate(added):
        require(isinstance(entry, dict) and len(entry) == 7 and type(entry.get("id")) is int
                and entry["id"] == _BASE_VOCAB_SIZE + index
                and isinstance(entry.get("content"), str) and entry["content"] != "" and entry["content"].isascii()
                and all(_same(entry.get(key), value) for key, value in _ADDED_TOKEN_FLAGS.items())
                and type(entry.get("special")) is bool,
                "contains an unsupported added-token entry")
        require(entry["content"] not in vocab and entry["content"] not in contents,
                "contains duplicate added-token content")
        contents[entry["content"]] = (entry["id"], entry["special"])
    require(sum(special for _, special in contents.values()) == _SPECIAL_TOKEN_COUNT,
            "has an unexpected special-token count")
    for content, expected in _PROTOCOL_TOKENS.items():
        require(contents.get(content) == expected, f"does not hold the protocol token {content!r} as {expected}")
