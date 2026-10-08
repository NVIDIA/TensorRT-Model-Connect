# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialize native full-EOS metadata without mutating original tokenizer assets."""

from __future__ import annotations

import json
from pathlib import Path
import shutil


def _object(path: Path) -> dict:
    """Read a JSON object or raise a preparation error with the source path."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Tokenizer metadata must contain an object: {path.name}")
    return value


def source_chat_template(checkpoint: Path) -> str:
    """Resolve the original standalone template with HF file precedence."""
    metadata = _object(checkpoint / "tokenizer_config.json")
    standalone = checkpoint / "chat_template.jinja"
    template = standalone.read_text(encoding="utf-8") if standalone.exists() else metadata.get("chat_template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Nemotron-H requires an embedded or standalone chat template")
    return template


def native_eos_ids(checkpoint: Path, raw: dict) -> list[int]:
    """Resolve native generation-config precedence and the complete stop vector."""
    generation = checkpoint / "generation_config.json"
    config = _object(generation) if generation.exists() else {}
    eos = config.get("eos_token_id", raw.get("eos_token_id"))
    ids = eos if isinstance(eos, list) else [eos]
    if (not ids or any(type(value) is not int or not 0 <= value < raw["vocab_size"] for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Nemotron-H native EOS must contain unique vocabulary IDs")
    return ids


def eos_token(tokenizer: dict, eos: int) -> str:
    """Resolve one unambiguous original token string for an ID, without constants."""
    by_id: dict[int, set[str]] = {}
    by_token: dict[str, set[int]] = {}
    vocab = tokenizer.get("model", {}).get("vocab", {})
    if not isinstance(vocab, dict):
        raise ValueError("Nemotron-H Edge requires a BPE vocabulary object")
    entries = [(token, index) for token, index in vocab.items()]
    for entry in tokenizer.get("added_tokens", []):
        if not isinstance(entry, dict):
            raise ValueError("Invalid Nemotron-H added token entry")
        entries.append((entry.get("content"), entry.get("id")))
    for token, index in entries:
        if not isinstance(token, str) or type(index) is not int or index < 0:
            raise ValueError("Invalid Nemotron-H tokenizer token/ID mapping")
        by_id.setdefault(index, set()).add(token)
        by_token.setdefault(token, set()).add(index)
    tokens = by_id.get(eos, set())
    if len(tokens) != 1:
        raise ValueError("Native first EOS has missing or ambiguous tokenizer ID mapping")
    token = next(iter(tokens))
    if not token or by_token[token] != {eos}:
        raise ValueError("Native first EOS token maps to multiple IDs")
    return token


def prepare_tokenizer(checkpoint: Path, engine: Path, destination: Path, raw: dict,
                      *, chat_template: str | None = None) -> list[int]:
    """Create explicit derivative metadata; original source/engine files stay intact.

    Returns the full native EOS vector. Invalid source metadata is a
    preparation failure, before bundle publication or runtime execution.
    """
    tokenizer = _object(checkpoint / "tokenizer.json")
    metadata = _object(checkpoint / "tokenizer_config.json")
    if chat_template is not None:
        if not isinstance(chat_template, str) or not chat_template.strip():
            raise ValueError("Explicit Edge chat template must be non-empty")
        metadata["chat_template"] = chat_template
    if not isinstance(metadata.get("chat_template"), str) or not metadata["chat_template"]:
        raise ValueError("Nemotron-H native requires embedded tokenizer chat_template")
    eos = native_eos_ids(checkpoint, raw)
    for token_id in eos:
        eos_token(tokenizer, token_id)
    metadata["eos_token"] = eos_token(tokenizer, eos[0])
    destination.mkdir(parents=True)
    # Runtime token IDs come from the byte-exact retained source vocabulary.
    shutil.copy2(checkpoint / "tokenizer.json", destination / "tokenizer.json")
    shutil.copy2(engine / "processed_chat_template.json", destination / "processed_chat_template.json")
    (destination / "tokenizer_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return eos
