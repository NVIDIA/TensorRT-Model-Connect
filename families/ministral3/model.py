# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundle builder of the bidirectional Ministral 3 sentence encoder.

The bundle holds one encoder plan for sequences of 1 to ``max_sequence_length``
tokens, the checkpoint's ``tokenizer.json`` and ``runtime.json`` with the
Sentence Transformers prompts. One bundle serves ``text_to_embedding``,
``text_to_pooled_features`` and ``text_to_token_features``; the build task
selects the bundle's primary Task.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ModelConfig
from .weights import load_weights

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

TASKS = ("text_to_embedding", "text_to_pooled_features", "text_to_token_features")
DEFAULT_MAX_SEQUENCE_LENGTH = 8192
OPT_SEQUENCE_LENGTH = 512


# The tokenizer.json contract the native tokenizer implements (runtime/tokenizer.cpp
# refuses any other at load); checking it here fails the build instead.
TEKKEN_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_PRE_TOKENIZER = {
    "type": "Sequence",
    "pretokenizers": [
        {
            "type": "Split",
            "pattern": {"Regex": TEKKEN_PATTERN},
            "behavior": "Isolated",
            "invert": False,
        },
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    ],
}
_BPE_FLAGS = {
    "type": "BPE",
    "dropout": None,
    "unk_token": None,
    "continuing_subword_prefix": None,
    "end_of_word_suffix": None,
    "fuse_unk": False,
    "byte_fallback": False,
    "ignore_merges": True,
}


def _byte_alphabet() -> list[str]:
    """The GPT-2 byte-to-character alphabet of the ByteLevel pre-tokenizer."""
    alphabet, shifted = [], 0
    for byte in range(256):
        if 33 <= byte <= 126 or 161 <= byte <= 172 or byte >= 174:
            alphabet.append(chr(byte))
        else:
            alphabet.append(chr(256 + shifted))
            shifted += 1
    return alphabet


def _merge_parts(merge: object) -> tuple[str, str] | None:
    if isinstance(merge, str):
        parts = merge.split(" ")
        return (parts[0], parts[1]) if len(parts) == 2 and all(parts) else None
    if isinstance(merge, list) and len(merge) == 2 and all(isinstance(part, str) for part in merge):
        return merge[0], merge[1]
    return None


def _vocabulary_problems(model: dict, added_tokens: object, vocab_size: int) -> list[str]:
    """The vocabulary invariants the native tokenizer requires at load."""
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or len(vocab) != vocab_size:
        return ["vocab size"]
    problems = []
    ids = list(vocab.values())
    if not all(type(value) is int for value in ids) or sorted(ids) != list(range(len(ids))):
        problems.append("vocab ids")
    if not all(symbol in vocab for symbol in _byte_alphabet()):
        problems.append("byte alphabet")
    merges = model.get("merges")
    pairs: set[tuple[str, str]] = set()
    for merge in merges if isinstance(merges, list) else (None,):
        parts = _merge_parts(merge)
        if (
            parts is None
            or parts in pairs
            or not all(part in vocab for part in (*parts, parts[0] + parts[1]))
        ):
            problems.append("merges")
            break
        pairs.add(parts)
    if not isinstance(added_tokens, list) or not all(
        isinstance(token, dict)
        and isinstance(token.get("content"), str)
        and token["content"]
        and type(token.get("id")) is int
        and 0 <= token["id"] < vocab_size
        and vocab.get(token["content"]) == token["id"]
        for token in added_tokens
    ):
        problems.append("added token ids")
    return problems


def _check_tokenizer(tokenizer: dict, vocab_size: int) -> None:
    model = tokenizer.get("model") or {}
    post = tokenizer.get("post_processor") or {}
    added = tokenizer.get("added_tokens", [])
    problems = [
        name
        for name, ok in (
            ("normalizer", tokenizer.get("normalizer") is None),
            ("truncation", tokenizer.get("truncation") is None),
            ("padding", tokenizer.get("padding") is None),
            ("pre_tokenizer", tokenizer.get("pre_tokenizer") == _PRE_TOKENIZER),
            (
                "post_processor",
                post.get("type") == "TemplateProcessing"
                and post.get("single") == [{"Sequence": {"id": "A", "type_id": 0}}]
                and post.get("special_tokens") == {},
            ),
            ("model", all(model.get(key, "?") == value for key, value in _BPE_FLAGS.items())),
            (
                "added_tokens",
                isinstance(added, list)
                and all(
                    isinstance(token, dict) and not token.get(key, True)
                    for token in added
                    for key in ("single_word", "lstrip", "rstrip", "normalized")
                ),
            ),
        )
        if not ok
    ]
    problems += _vocabulary_problems(model, added, vocab_size)
    if problems:
        raise ValueError(
            f"ministral3 tokenizer.json differs from the implemented Tekken contract: {problems}"
        )


def _sequence_bounds(requested: int | None, config: ModelConfig) -> tuple[int, int]:
    limit = config.sentence.max_seq_length
    maximum = requested or min(DEFAULT_MAX_SEQUENCE_LENGTH, limit)
    if not 1 <= maximum <= limit:
        raise ValueError(
            f"ministral3 max_sequence_length must lie in [1, {limit}] "
            f"(the checkpoint's Sentence Transformers max_seq_length), got {maximum}"
        )
    return maximum, min(OPT_SEQUENCE_LENGTH, maximum)


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    if request.task not in TASKS:
        raise ValueError(f"ministral3 task must be one of {TASKS}, got {request.task!r}")
    if request.backend != "trt":
        raise ValueError("ministral3 supports only the trt backend")
    precision = str(request.precision).lower()
    if precision not in {"bf16", "fp32"}:
        raise ValueError(f"ministral3 precision must be bf16 or fp32, got {request.precision!r}")
    if request.quantization not in (None, "none"):
        raise NotImplementedError("ministral3 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("ministral3 does not support fp32_layers")
    if request.max_batch_size != 1:
        raise NotImplementedError("ministral3 encodes one sequence per call (max_batch_size 1)")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("ministral3 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise ValueError("ministral3 does not support context parallelism")
    if request.dynamic_kv_cache:
        raise NotImplementedError("ministral3 does not support dynamic_kv_cache")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("ministral3 does not support MTP speculative decoding")
    if request.image_height is not None or request.image_width is not None:
        raise NotImplementedError("ministral3 does not take images")
    if request.video_num_frames is not None:
        raise NotImplementedError("ministral3 does not take video")
    model_dir = Path(request.model_dir)
    config = ModelConfig.from_dir(model_dir)
    max_sequence_length, opt_sequence_length = _sequence_bounds(request.max_sequence_length, config)
    tokenizer = model_dir / "tokenizer.json"
    if not tokenizer.is_file():
        raise ValueError(f"ministral3 checkpoint has no tokenizer.json in {model_dir}")
    _check_tokenizer(json.loads(tokenizer.read_text(encoding="utf-8")), config.vocab_size)

    from .graph import build_engine

    plan = build_engine(
        config,
        load_weights(model_dir, config),
        precision=precision,
        max_sequence_length=max_sequence_length,
        opt_sequence_length=opt_sequence_length,
        verbose=bool(request.verbose),
    )
    writer.set_header(family="ministral3", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "max_sequence_length": max_sequence_length,
            "precision": precision,
            "query_prompt": config.sentence.prompts["query"],
            "document_prompt": config.sentence.prompts["document"],
        },
    )
    writer.add_bytes("tokenizer.json", tokenizer.read_bytes())
