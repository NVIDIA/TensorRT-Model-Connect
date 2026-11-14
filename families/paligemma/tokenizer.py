# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint's tokenizer, flattened for the native runtime and cross-checked.

``tokenizer.model`` (SentencePiece BPE) is converted to the runtime asset. The
added and special tokens follow the transformers 5.2.0 loading rules:
``tokenizer_config.json``'s ``added_tokens_decoder`` when present, otherwise
``added_tokens.json`` and ``tokenizer.json``'s ``added_tokens``; named special
tokens (BOS, EOS, ...), extra special tokens and model-specific ``*_token``
entries are always registered as special; the processor adds ``<image>`` as
special. Decoding skips the special ids. The reference splits
every added token out of a prompt before BPE; SentencePiece matches only its
user-defined pieces that way, so a prefix containing any other added token is
refused natively rather than tokenized differently.
"""

from __future__ import annotations

import json
from pathlib import Path

from .tokenizer_export import convert_paligemma_tokenizer_model, parse_trtmcbpe_asset

IMAGE_TOKEN = "<image>"
# The reference processor adds these location and segmentation tokens as plain
# (non-special) tokens; they must already be pieces of the vocabulary.
EXTRA_TOKENS = tuple(f"<loc{i:04d}>" for i in range(1024)) + tuple(f"<seg{i:03d}>" for i in range(128))
_NAMED_SPECIAL = ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token",
                  "mask_token")
# GemmaTokenizer constructor defaults for named special tokens a config omits.
_GEMMA_SPECIAL_DEFAULTS = {"bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>",
                           "pad_token": "<pad>", "mask_token": "<mask>"}
_USER_DEFINED = 4


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"PaliGemma {path.name} must be a JSON object")
    return document


def _content(value) -> str | None:
    if isinstance(value, dict):
        value = value.get("content")
    return value if isinstance(value, str) and value else None


def _special_source(root: Path, config: dict, legacy: bool) -> dict:
    """The tokenizer init arguments that name special tokens, merged as 5.2.0 merges them."""
    source = dict(config)
    if legacy and (root / "special_tokens_map.json").is_file():
        source.update(_read(root / "special_tokens_map.json"))
    if "additional_special_tokens" in source:
        source.setdefault("extra_special_tokens", source.pop("additional_special_tokens"))
    return source


def _named_special(source: dict) -> list[str]:
    names = [_content(source.get(key, _GEMMA_SPECIAL_DEFAULTS.get(key))) for key in _NAMED_SPECIAL]
    extra = source.get("extra_special_tokens") or []
    names.extend(_content(value) for value in (extra.values() if isinstance(extra, dict) else extra))
    # Any other ``*_token`` key holding a token is a model-specific special token.
    names.extend(_content(value) for key, value in source.items()
                 if key.endswith("_token") and key not in _NAMED_SPECIAL
                 and isinstance(value, (str, dict)))
    return [name for name in names if name is not None]


def _added_tokens(root: Path, config: dict, special_names: set[str]) -> dict[int, tuple[str, bool, bool]]:
    """Map id -> (content, special, strips): the table the reference tokenizer loads."""
    decoder = config.get("added_tokens_decoder")
    entries = {}
    if decoder is not None:
        if not isinstance(decoder, dict):
            raise ValueError("PaliGemma tokenizer_config.json added_tokens_decoder must be an object")
        entries = {int(index): token for index, token in decoder.items()}
    else:
        legacy = _read(root / "added_tokens.json")
        entries = {index: {"content": content, "special": content in special_names}
                   for content, index in legacy.items()}
        added = _read(root / "tokenizer.json").get("added_tokens")
        if not isinstance(added, list):
            raise ValueError("PaliGemma tokenizer.json must list its added tokens")
        entries.update({token.get("id"): token for token in added if isinstance(token, dict)})
    result = {}
    for index, token in entries.items():
        if type(index) is not int or not isinstance(token, dict) or _content(token) is None:
            raise ValueError("PaliGemma tokenizer has an invalid added token")
        strips = any(token.get(flag) is True for flag in ("lstrip", "rstrip", "single_word"))
        result[index] = (_content(token), token.get("special") is True, strips)
    return result


def load_tokenizer(model_dir: Path, image_token_id: int, vocab_size: int) -> tuple[bytes, dict]:
    """Return the runtime asset and the token facts the runtime reads from ``runtime.json``."""
    root = Path(model_dir)
    asset, metadata = convert_paligemma_tokenizer_model((root / "tokenizer.model").read_bytes())
    pieces = parse_trtmcbpe_asset(asset).pieces
    if len(pieces) + 1 > vocab_size or image_token_id >= vocab_size:
        raise ValueError("PaliGemma tokenizer does not fit the embedding table")
    ids = {piece.text: index for index, piece in enumerate(pieces)}
    if IMAGE_TOKEN in ids:
        raise ValueError("PaliGemma vocabulary must not contain the image token")
    missing = [token for token in EXTRA_TOKENS if token not in ids]
    if missing:
        raise ValueError(f"PaliGemma vocabulary lacks processor tokens: {missing[:4]}")
    config = _read(root / "tokenizer_config.json")
    source = _special_source(root, config, "added_tokens_decoder" not in config)
    # The processor writes the configured BOS text into every prompt and the
    # runtime prepends tokenizer.model's BOS id, so both must name one piece.
    bos = _content(source.get("bos_token", _GEMMA_SPECIAL_DEFAULTS["bos_token"]))
    if bos is None or ids.get(bos) != metadata.bos_id:
        raise ValueError("PaliGemma bos_token is not the tokenizer.model BOS piece")
    named = _named_special(source)
    added = _added_tokens(root, config, set(named))
    # Without a listed image token the processor appends it after the vocabulary.
    if added.get(image_token_id, (IMAGE_TOKEN,))[0] != IMAGE_TOKEN or (
            image_token_id not in added and image_token_id != len(pieces)):
        raise ValueError("PaliGemma image token id does not match the tokenizer")
    added[image_token_id] = (IMAGE_TOKEN, True, False)
    for index, (content, _, _) in added.items():
        if index != image_token_id and (index >= len(pieces) or pieces[index].text != content):
            raise ValueError(f"PaliGemma added token {index} {content!r} is not that vocabulary piece")
    special = {index for index, (_, flag, _) in added.items() if flag}
    for name in named:
        if name != IMAGE_TOKEN and name not in ids:
            raise ValueError(f"PaliGemma special token {name!r} is not in the vocabulary")
        special.add(image_token_id if name == IMAGE_TOKEN else ids[name])
    # Whitespace-stripping added tokens also match differently from SentencePiece.
    reserved = sorted(index for index in set(added) | special
                      if index == image_token_id or pieces[index].piece_type != _USER_DEFINED
                      or added.get(index, ("", False, False))[2])
    facts = {
        "bos_token_id": metadata.bos_id,
        "tokenizer_pieces": len(pieces),
        "special_token_ids": sorted(special),
        "reserved_token_ids": reserved,
    }
    return asset, facts
