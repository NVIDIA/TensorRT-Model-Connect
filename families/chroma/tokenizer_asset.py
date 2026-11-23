# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint's T5 tokenizer as tokenizers JSON for the native runtime.

The tokenizer is loaded through transformers, as the reference pipeline loads
it, and the native runtime implements exactly the resulting configuration:
no normalizer, whitespace split, ``Metaspace`` with an always-prepended
``U+2581`` and splitting, a Unigram model without byte fallback, and an
appended ``</s>``. Any other configuration is rejected here instead of being
tokenized differently at run time.
"""

from __future__ import annotations

import json
from pathlib import Path

EOS = "</s>"
PAD = "<pad>"
METASPACE = "\u2581"


def validate_tokenizer(data: dict) -> None:
    if data.get("normalizer") is not None:
        raise ValueError("the Chroma runtime tokenizer supports no normalizer")
    expected_pre = {
        "type": "Sequence",
        "pretokenizers": [
            {"type": "WhitespaceSplit"},
            {"type": "Metaspace", "replacement": METASPACE, "prepend_scheme": "always", "split": True},
        ],
    }
    if data.get("pre_tokenizer") != expected_pre:
        raise ValueError(f"unsupported pre-tokenizer: {data.get('pre_tokenizer')!r}")
    post = data.get("post_processor") or {}
    single = [{"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": EOS, "type_id": 0}}]
    if post.get("type") != "TemplateProcessing" or post.get("single") != single:
        raise ValueError(f"unsupported post-processor: {post!r}")
    model = data.get("model") or {}
    if model.get("type") != "Unigram" or model.get("byte_fallback"):
        raise ValueError("the Chroma runtime tokenizer requires a Unigram model without byte fallback")
    vocab = model.get("vocab") or []
    pieces = {piece: index for index, (piece, _score) in enumerate(vocab)}
    if pieces.get(PAD) != 0 or pieces.get(EOS) != 1 or model.get("unk_id") is None:
        raise ValueError("the T5 vocabulary must place <pad> at 0 and </s> at 1 and declare unk_id")
    for token in data.get("added_tokens", []):
        if token.get("normalized") or token.get("lstrip") or token.get("rstrip") or token.get("single_word"):
            raise ValueError(f"unsupported added-token options for {token.get('content')!r}")
        if pieces.get(token["content"]) != token["id"]:
            raise ValueError(f"added token {token['content']!r} disagrees with the vocabulary")


def tokenizer_json(model_dir: Path) -> bytes:
    """Serialize the checkpoint tokenizer after checking the runtime implements it."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("the Chroma tokenizer must load as a tokenizers-backed T5 tokenizer")
    data = json.loads(backend.to_str())
    validate_tokenizer(data)
    return json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
