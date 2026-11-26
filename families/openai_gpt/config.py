# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI GPT checkpoint configuration, validated against the graph this family builds.

Hugging Face ``OpenAIGPTLMHeadModel`` always builds post-norm blocks with a fused
QKV projection, scaled causal attention, a ``4 * n_embd`` MLP and an LM head tied to
the token embedding. The fields that size or change that graph are required and
checked here. The dropout, initializer and sequence-summary fields only affect
training or the other heads, and ``n_ctx`` is not read by the model, so they are
not used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARCHITECTURE = "OpenAIGPTLMHeadModel"
MODEL_TYPE = "openai-gpt"
_INT32_MAX = (1 << 31) - 1
# The smallest normal float32. A smaller value becomes zero or a subnormal (which the GPU may
# flush to zero) in the engine; a larger one than float32 can hold becomes infinity.
_FLOAT32_TINY = np.finfo(np.float32).tiny


@dataclass(frozen=True)
class OpenAIGPTConfig:
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    layer_norm_epsilon: float
    # Generation stops at any of these ids; empty for the published checkpoint.
    eos_token_ids: tuple[int, ...]

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def inner_size(self) -> int:
        return 4 * self.n_embd


def _integer(raw: dict, name: str) -> int:
    if name not in raw:
        raise ValueError(f"OpenAI GPT config.json must declare {name!r}")
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _INT32_MAX:
        raise ValueError(f"OpenAI GPT config field {name!r} must be an integer in [1, {_INT32_MAX}]")
    return value


def _positive_float32(raw: dict, name: str) -> float:
    """A positive number that stays a normal, finite float32: the engine stores it as one."""
    if name not in raw:
        raise ValueError(f"OpenAI GPT config.json must declare {name!r}")
    value = raw[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"OpenAI GPT config field {name!r} must be a number")
    with np.errstate(over="ignore"):
        stored = np.float32(value)
    if not math.isfinite(float(value)) or not np.isfinite(stored) or stored < _FLOAT32_TINY:
        raise ValueError(
            f"OpenAI GPT config field {name!r} must be a positive, normal float32 value"
        )
    return float(stored)


def _eos_token_ids(raw: dict, generation: dict, vocab_size: int) -> tuple[int, ...]:
    # transformers generates with generation_config.json when the checkpoint has one.
    source = generation if "eos_token_id" in generation else raw
    value = source.get("eos_token_id")
    if value is None:
        return ()
    values = value if isinstance(value, list) else [value]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise ValueError("OpenAI GPT eos_token_id must be an integer or a list of integers")
    if any(not 0 <= item < vocab_size for item in values):
        raise ValueError(f"OpenAI GPT eos_token_id must be inside the vocabulary [0, {vocab_size})")
    return tuple(dict.fromkeys(values))


def parse_config(raw: dict, generation: dict) -> OpenAIGPTConfig:
    """Validate ``config.json`` and ``generation_config.json`` for the native graph."""
    if raw.get("model_type") != MODEL_TYPE:
        raise ValueError(f"OpenAI GPT requires model_type={MODEL_TYPE!r}, got {raw.get('model_type')!r}")
    if raw.get("architectures") != [ARCHITECTURE]:
        raise ValueError(f"OpenAI GPT requires architectures=[{ARCHITECTURE!r}]")
    if raw.get("afn") != "gelu":
        raise ValueError(f"OpenAI GPT supports only afn='gelu' (gelu_new), got {raw.get('afn')!r}")
    tied = raw.get("tie_word_embeddings", True)
    if tied is not True:
        raise ValueError(f"OpenAI GPT supports only tie_word_embeddings=True, got {tied!r}")
    embd = _integer(raw, "n_embd")
    heads = _integer(raw, "n_head")
    if embd % heads:
        raise ValueError("OpenAI GPT n_embd must be divisible by n_head")
    # The MLP width is 4 * n_embd; it must stay an int32 like every other dimension.
    if 4 * embd > _INT32_MAX:
        raise ValueError("OpenAI GPT n_embd is too large for its 4 * n_embd MLP")
    vocab = _integer(raw, "vocab_size")
    return OpenAIGPTConfig(
        vocab_size=vocab,
        n_positions=_integer(raw, "n_positions"),
        n_embd=embd,
        n_layer=_integer(raw, "n_layer"),
        n_head=heads,
        layer_norm_epsilon=_positive_float32(raw, "layer_norm_epsilon"),
        eos_token_ids=_eos_token_ids(raw, generation, vocab),
    )


def parse_json_object(data: bytes | str, name: str) -> dict:
    """One standard JSON object of valid Unicode text: a repeated key, which the parser would
    silently collapse, NaN or Infinity, which are not JSON, a number past the double range and a
    lone surrogate are refused."""

    def unique(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"OpenAI GPT {name} repeats the key {key!r}")
            result[key] = value
        return result

    def standard(constant: str) -> float:
        # NaN and Infinity are not JSON; the runtime's parser refuses them.
        raise ValueError(f"OpenAI GPT {name} holds the non-standard constant {constant}")

    if isinstance(data, bytes):
        # The bundle keeps the file's bytes, and the runtime parses them as UTF-8.
        try:
            data = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"OpenAI GPT {name} is not UTF-8 text") from error
    value = json.loads(data, object_pairs_hook=unique, parse_constant=standard)
    if not isinstance(value, dict):
        raise ValueError(f"OpenAI GPT {name} must contain one JSON object")
    pending: list[object] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str) and any(0xD800 <= ord(char) <= 0xDFFF for char in item):
            # An escaped lone surrogate is not text; the runtime's parser refuses it.
            raise ValueError(f"OpenAI GPT {name} holds a string that is not valid Unicode")
        elif isinstance(item, float) and not math.isfinite(item):
            # A literal past the double range (1e400) parses as infinity here; the runtime's
            # parser refuses it.
            raise ValueError(f"OpenAI GPT {name} holds a number outside the double range")
    return value


def _read_object(path: Path) -> dict:
    return parse_json_object(path.read_bytes(), path.name)


def load_config(model_dir: Path) -> OpenAIGPTConfig:
    generation_path = model_dir / "generation_config.json"
    generation = _read_object(generation_path) if generation_path.is_file() else {}
    return parse_config(_read_object(model_dir / "config.json"), generation)
