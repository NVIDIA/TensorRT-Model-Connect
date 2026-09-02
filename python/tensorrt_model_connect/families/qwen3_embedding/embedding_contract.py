# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint and request contract owned by the Qwen3-Embedding family."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import ModelConfig


DEFAULT_RETRIEVAL_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
QUERY_INPUT_FORMAT = "Instruct: {instruction}\nQuery:{query}"
_QWEN3_EMBEDDING_06B_ARCHITECTURE = {
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151669,
    "max_position_embeddings": 32768,
    "eos_token_id": 151643,
}


@dataclass(frozen=True)
class Qwen3EmbeddingContract:
    pooling: str
    normalize: bool
    embedding_dimension: int
    input_format: str
    eos_token_id: int


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _pooling_config(config: ModelConfig) -> dict | None:
    model_dir = str(config.raw.get("_model_dir", "")).strip()
    if not model_dir:
        return None
    root = Path(model_dir)
    modules = _read_json(root / "modules.json")
    if not isinstance(modules, list):
        return None

    semantic_modules: list[str] = []
    pooling_path: str | None = None
    for module in modules:
        if not isinstance(module, dict):
            return None
        module_type = str(module.get("type", ""))
        semantic_modules.append(module_type.rsplit(".", 1)[-1])
        if module_type.endswith(".Pooling"):
            pooling_path = str(module.get("path") or "1_Pooling")
    if semantic_modules != ["Transformer", "Pooling", "Normalize"]:
        return None
    if pooling_path is None:
        return None
    pooling_relative = Path(pooling_path)
    if pooling_relative.is_absolute() or ".." in pooling_relative.parts:
        return None

    value = _read_json(root / pooling_relative / "config.json")
    return value if isinstance(value, dict) else None


def detect_qwen3_embedding_contract(
    config: ModelConfig,
) -> Qwen3EmbeddingContract | None:
    """Recognize the sentence-transformers last-token checkpoint contract.

    A normal Qwen3 causal-LM config is intentionally insufficient: the
    official embedding checkpoint publishes the same architecture name.  The
    model-owned ``1_Pooling/config.json`` is the semantic discriminator.
    """

    if str(config.model_type).lower() != "qwen3":
        return None
    architectures = {str(value).lower() for value in config.architectures}
    if "qwen3forcausallm" not in architectures:
        return None
    observed_architecture = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "eos_token_id": config.eos_token_id,
    }
    if observed_architecture != _QWEN3_EMBEDDING_06B_ARCHITECTURE:
        return None

    pooling = _pooling_config(config)
    if pooling is None:
        return None
    enabled_modes = {
        str(key)
        for key, value in pooling.items()
        if str(key).startswith("pooling_mode_") and value is True
    }
    if enabled_modes != {"pooling_mode_lasttoken"}:
        return None

    dimension = pooling.get("word_embedding_dimension")
    if not isinstance(dimension, int) or dimension != config.hidden_size:
        return None
    if config.eos_token_id < 0:
        return None

    return Qwen3EmbeddingContract(
        pooling="last_token",
        normalize=True,
        embedding_dimension=dimension,
        input_format=QUERY_INPUT_FORMAT,
        eos_token_id=config.eos_token_id,
    )


def format_embedding_query(instruction: str, query: str) -> str:
    """Apply the official Qwen3-Embedding query-side input convention."""

    instruction = str(instruction).strip()
    query = str(query).strip()
    if not instruction:
        raise ValueError("Qwen3-Embedding query instruction must be non-empty")
    if not query:
        raise ValueError("Qwen3-Embedding query must be non-empty")
    return QUERY_INPUT_FORMAT.format(instruction=instruction, query=query)


def last_token_indices(attention_mask: Sequence[Sequence[int]]) -> list[int]:
    """Return each row's final valid-token index for either padding side."""

    indices: list[int] = []
    for row in attention_mask:
        valid = [index for index, value in enumerate(row) if int(value) != 0]
        if not valid:
            raise ValueError("Qwen3-Embedding attention mask row has no valid token")
        indices.append(valid[-1])
    return indices
