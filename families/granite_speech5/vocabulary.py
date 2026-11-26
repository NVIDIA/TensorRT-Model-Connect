# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CTC vocabulary: one decoding entry per token id, and the reference's decoding rules.

The reference transcript is ``ParakeetTokenizer.decode(frame_ids, skip_special_tokens=True)``:
repeated frame ids collapse to one, the blank (the tokenizer's pad token) is dropped, special
tokens are skipped, and the ``tokenizer.json`` decoder runs: ``▁`` becomes a space, runs of
``<0xHH>`` byte tokens become their UTF-8 text (one U+FFFD per byte of a run that is not valid
UTF-8), the pieces are joined and one leading space is stripped. Only that decoder is accepted.

The bundle stores each id as ``[kind, value]``: ``[0, ""]`` skipped, ``[1, text]`` text after the
space replacement, ``[2, byte]`` one byte of a byte run. :func:`decode` is the same algorithm as
the runtime's and is checked against the tokenizer in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from .config import read_json

SKIP, TEXT, BYTE = 0, 1, 2
SPACE_MARK = "▁"
REPLACEMENT = "�"
DECODER = {
    "type": "Sequence",
    "decoders": [
        {"type": "Replace", "pattern": {"String": SPACE_MARK}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0},
    ],
}
_HEX = frozenset("0123456789abcdefABCDEF")


@dataclass(frozen=True)
class Vocabulary:
    entries: tuple[tuple[int, str | int], ...]
    blank_id: int

    def to_json(self) -> dict:
        return {"blank_id": self.blank_id, "entries": [list(entry) for entry in self.entries]}


def byte_value(token: str) -> int | None:
    """The byte of a ``<0xHH>`` token as the ByteFallback decoder reads it, else ``None``.

    Like the decoder: exactly six bytes, ``<0x`` and ``>`` around two characters that parse as
    one base-16 ``u8`` (an optional leading ``+``, then ASCII hex digits only).
    """
    raw = token.encode("utf-8")
    if len(raw) != 6 or not raw.startswith(b"<0x") or not raw.endswith(b">"):
        return None
    digits = raw[3:5].decode("utf-8", errors="strict") if raw[3:5].isascii() else ""
    if digits.startswith("+"):
        digits = digits[1:]
    if not digits or any(char not in _HEX for char in digits):
        return None
    return int(digits, 16)


def _tokenizer(model_dir: Path) -> dict:
    tokenizer = read_json(Path(model_dir) / "tokenizer.json")
    model = tokenizer.get("model")
    if not isinstance(model, dict) or model.get("type") != "BPE" or not isinstance(model.get("vocab"), dict):
        raise NotImplementedError("granite_speech5 decodes a BPE tokenizer.json with a vocab object")
    if tokenizer.get("decoder") != DECODER:
        raise NotImplementedError(
            f"granite_speech5 decodes only the {DECODER['decoders']} decoder, not {tokenizer.get('decoder')!r}")
    return tokenizer


def _added_tokens(tokenizer: dict) -> list[tuple[int, str, bool]]:
    raw = tokenizer.get("added_tokens", [])
    if not isinstance(raw, list):
        raise ValueError("granite_speech5 tokenizer.json added_tokens must be a list")
    result, contents = [], set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"granite_speech5 tokenizer.json added token {item!r} is not an object")
        token_id, content, special = item.get("id"), item.get("content"), item.get("special")
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError(f"granite_speech5 added token {content!r} has no valid id: {token_id!r}")
        if not isinstance(content, str) or not content or not isinstance(special, bool):
            raise ValueError(f"granite_speech5 added token {token_id} needs non-empty content and a special flag")
        if content in contents:  # the tokenizer would merge both into one token with one set of flags
            raise ValueError(f"granite_speech5 tokenizer.json repeats the added token {content!r}")
        contents.add(content)
        result.append((token_id, content, special))
    return result


def _token_table(tokenizer: dict) -> tuple[dict[int, str], dict[str, int], set[str]]:
    """Id -> token, token -> id and the special token strings, as the tokenizers library resolves them.

    An added token keeps the id of its content in the model vocabulary; new content takes the next
    id after the vocabulary and the added tokens before it. The id stored in the file is only checked
    against that (the library warns and ignores it), so a file whose ids disagree, or whose new token
    lands on a model id, is refused instead of being decoded differently from the reference.
    """
    vocab = tokenizer["model"]["vocab"]
    pieces: dict[str, int] = {}
    by_id: dict[int, str] = {}
    for piece, token_id in vocab.items():
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError(f"granite_speech5 tokenizer.json gives {piece!r} the id {token_id!r}")
        if token_id in by_id:
            raise ValueError(f"granite_speech5 tokenizer.json gives two pieces the id {token_id}")
        by_id[token_id] = piece
        pieces[piece] = token_id
    size, resolved_ids, special = len(vocab), [], set()
    for stored, content, is_special in _added_tokens(tokenizer):
        if content in pieces:
            resolved = pieces[content]
        else:
            top = max(resolved_ids, default=None)
            resolved = size if top is None or (top < size and size > 0) else top + 1
            if resolved in by_id:
                raise ValueError(f"granite_speech5 added token {content!r} would take the id {resolved} "
                                 f"of the piece {by_id[resolved]!r}")
        if resolved != stored:
            raise ValueError(f"granite_speech5 tokenizer.json stores the added token {content!r} with id "
                             f"{stored}, but the tokenizer gives it {resolved}")
        resolved_ids.append(resolved)
        by_id[resolved], pieces[content] = content, resolved
        if is_special:
            special.add(content)
    return by_id, pieces, special


def read_pieces(model_dir: Path) -> dict[str, int]:
    """Token string -> id, as the tokenizer resolves a token string."""
    return _token_table(_tokenizer(model_dir))[1]


def _library(model_dir: Path):
    """The checkpoint's tokenizer as the tokenizers library loads it; a file it refuses is refused here."""
    from tokenizers import Tokenizer

    try:
        return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    except Exception as error:  # noqa: BLE001 - the library reports every malformed file the same way
        raise ValueError(f"granite_speech5 tokenizer.json does not load in the tokenizers library: {error}") from error


def _check_against_library(library, vocabulary: Vocabulary, by_id: dict[int, str], special: set[str]) -> None:
    """The table must be the library's: every id's token, the special tokens and every id decoded alone."""
    library_special = {token.content for token in library.get_added_tokens_decoder().values() if token.special}
    if library_special != special:
        raise ValueError("granite_speech5 special tokens differ from the tokenizers library's")
    for token_id, (kind, value) in enumerate(vocabulary.entries):
        if library.id_to_token(token_id) != by_id[token_id]:
            raise ValueError(f"granite_speech5 token {token_id} differs from the tokenizers library's "
                             f"({by_id[token_id]!r} != {library.id_to_token(token_id)!r})")
        if library.decode([token_id], skip_special_tokens=True) != decode(vocabulary, [token_id]):
            raise ValueError(f"granite_speech5 decodes token {token_id} ({by_id[token_id]!r}) differently "
                             f"from the tokenizers library")


_SPECIAL_LISTS = ("additional_special_tokens", "extra_special_tokens", "model_specific_special_tokens")
_ADDED_TOKEN_FIELDS = ("content", "single_word", "lstrip", "rstrip", "normalized", "special")


def _registered_special(config: dict, name: str) -> set[str]:
    """The tokens Transformers registers as special from one configuration file.

    Every ``*_token`` key holding a token (``bos_token`` ... ``mask_token`` and model-specific ones such
    as ``image_token``; a boolean like ``add_bos_token`` is not a token) and every entry of
    ``additional_special_tokens``, ``extra_special_tokens`` and ``model_specific_special_tokens``.
    """
    registered: set[str] = set()

    def add(value, key: str) -> None:
        if isinstance(value, dict):
            value = value.get("content")
        if not isinstance(value, str) or not value:
            raise ValueError(f"granite_speech5 {name} {key} is not a token: {value!r}")
        registered.add(value)

    for key, value in config.items():
        if value is None:
            continue
        if key in _SPECIAL_LISTS:
            if not isinstance(value, (list, dict)):
                raise ValueError(f"granite_speech5 {name} {key} must be a list or an object, not {value!r}")
            for item in value.values() if isinstance(value, dict) else value:
                add(item, key)
        elif key.endswith("_token") and isinstance(value, (str, dict)):
            add(value, key)
    return registered


def check_tokenizer_config(model_dir: Path, tokenizer: dict, special: set[str]) -> None:
    """Refuse token metadata that Transformers applies on load and ``tokenizer.json`` does not hold.

    The tokens ``tokenizer_config.json`` and ``special_tokens_map.json`` register as special make the
    reference skip them, and every ``added_tokens_decoder`` entry is added with its flags (a special flag
    skips the token, a normalized flag changes its spacing). The bundle's table comes from
    ``tokenizer.json``, so each registered token must be special there and each entry must be one of its
    added tokens, unchanged; ``added_tokens.json`` is refused.
    """
    model_dir = Path(model_dir)
    if (model_dir / "added_tokens.json").is_file():
        raise NotImplementedError("granite_speech5 does not read added_tokens.json; Transformers would add its "
                                  "tokens to the reference's tokenizer")
    registered: set[str] = set()
    config: dict = {}
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        if (model_dir / name).is_file():
            content = read_json(model_dir / name)
            registered |= _registered_special(content, name)
            config = content if name == "tokenizer_config.json" else config
    extra = registered - special
    if extra:
        raise NotImplementedError(f"granite_speech5 tokenizer configuration makes {sorted(extra)} special, which "
                                  "tokenizer.json does not; the reference transcript would skip them")
    if "added_tokens_decoder" not in config:
        return
    entries = config["added_tokens_decoder"]
    if not isinstance(entries, dict):  # Transformers fails to load a null or any other value too
        raise ValueError("granite_speech5 tokenizer_config.json added_tokens_decoder must be an object")
    added = {item["content"]: item for item in tokenizer.get("added_tokens", [])}
    for key, entry in entries.items():
        if not (key.isascii() and key.isdigit() and (key == "0" or not key.startswith("0"))):
            raise ValueError(f"granite_speech5 tokenizer_config.json added_tokens_decoder key {key!r} "
                             "is not a canonical decimal id")
        content = entry.get("content") if isinstance(entry, dict) else None
        stored = added.get(content)
        expected = None if stored is None else {field: stored.get(field) for field in _ADDED_TOKEN_FIELDS}
        if expected is None or entry != expected or int(key) != stored["id"]:
            raise NotImplementedError(
                f"granite_speech5 tokenizer_config.json added_tokens_decoder entry {key} {entry!r} is not the "
                f"added token tokenizer.json stores ({None if stored is None else stored['id']} {expected!r}); "
                "Transformers applies it on load, so the reference could decode differently")


def read_vocabulary(model_dir: Path, vocab_size: int, blank_id: int) -> Vocabulary:
    """One decoding entry for every id the CTC head can emit, checked against the tokenizers library."""
    tokenizer = _tokenizer(model_dir)  # the layouts this family decodes, refused by name first
    library = _library(model_dir)
    by_id, _, special = _token_table(tokenizer)
    check_tokenizer_config(model_dir, tokenizer, special)
    if len(by_id) < vocab_size:  # checked before the loop below, whose length the head sets
        raise ValueError(f"granite_speech5 tokenizer.json has {len(by_id)} tokens, fewer than the "
                         f"{vocab_size} ids the CTC head emits")
    missing = [token_id for token_id in range(vocab_size) if token_id not in by_id]
    if missing:
        raise ValueError(f"granite_speech5 tokenizer.json has no token for the ids {missing[:8]} "
                         f"of the {vocab_size} the CTC head emits")
    entries = []
    for token_id in range(vocab_size):
        token = by_id[token_id]
        if token in special:  # skip_special_tokens compares token strings
            entries.append((SKIP, ""))
            continue
        text = token.replace(SPACE_MARK, " ")
        value = byte_value(text)
        entries.append((TEXT, text) if value is None else (BYTE, value))
    if not 0 <= blank_id < vocab_size:
        raise ValueError(f"granite_speech5 blank id {blank_id} is outside the vocabulary")
    vocabulary = Vocabulary(entries=tuple(entries), blank_id=blank_id)
    _check_against_library(library, vocabulary, by_id, special)
    return vocabulary


def collapse(frame_ids, blank_id: int) -> list[int]:
    """Greedy CTC: one id per run of equal frame ids, then every blank dropped."""
    return [token_id for token_id, _ in groupby(int(value) for value in frame_ids) if token_id != blank_id]


def decode(vocabulary: Vocabulary, token_ids) -> str:
    """The transcript of collapsed ids with special tokens skipped (the runtime's algorithm)."""
    parts: list[str] = []
    pending = bytearray()

    def flush() -> None:
        if pending:
            try:
                parts.append(pending.decode("utf-8", errors="strict"))
            except UnicodeDecodeError:
                parts.append(REPLACEMENT * len(pending))
            pending.clear()

    for token_id in token_ids:
        kind, value = vocabulary.entries[int(token_id)]
        if kind == SKIP:
            continue  # skipped before the decoder runs, so it does not end a byte run
        if kind == BYTE:
            pending.append(int(value))
            continue
        flush()
        parts.append(str(value))
    flush()
    text = "".join(parts)
    return text[1:] if text.startswith(" ") else text
