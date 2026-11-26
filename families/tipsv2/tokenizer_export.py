# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flatten the checkpoint's SentencePiece model into the asset the TIPSv2 runtime reads.

The checkpoint ships ``tokenizer.model``: a SentencePiece **BPE** model (not Unigram) with byte
fallback, the ``nmt_nfkc`` normalizer, and 32,000 pieces whose scores are negative integer merge
ranks. The runtime must not link protobuf or SentencePiece, so everything it needs is flattened
here into one little-endian record and checked on the way out.

Three parts travel in the asset:

* the **pieces**, with their score and SentencePiece type;
* the **precompiled charsmap** verbatim - the serialized Darts trie that *is* ``nmt_nfkc`` for this
  checkpoint. Substituting a generic Unicode normalizer would change token ids;
* a **lowercase table**, because the checkpoint's own tokenizer lowercases before it normalizes
  (``self.sp.encode(t.lower(), ...)`` in ``text_encoder.py``) and ``str.lower()`` is Python's full
  Unicode mapping, not ASCII folding. The table is read out of the building interpreter so the
  bundle reproduces exactly the reference it was measured against; its Unicode version travels in
  ``runtime.json`` so a bundle says which mapping it carries.

  One code point in all of Unicode does not fit a table, and it is the reason the ``Cased`` and
  ``Case_Ignorable`` sets travel too: U+03A3 GREEK CAPITAL LETTER SIGMA lowercases to the final
  form U+03C2 when a cased character precedes it and none follows, and to U+03C3 otherwise, so
  ``"ΟΣ".lower()`` is ``"ος"`` and not ``"οσ"``. Getting that wrong changes the token ids of every
  Greek word that ends in a sigma.
"""

from __future__ import annotations

import struct
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

MAGIC = b"TRTMCTIP"
ASSET_VERSION = 1

# SentencePiece ModelProto.SentencePiece.Type
PIECE_NORMAL = 1
PIECE_UNKNOWN = 2
PIECE_CONTROL = 3
PIECE_USER_DEFINED = 4
PIECE_UNUSED = 5
PIECE_BYTE = 6
_PIECE_TYPES = (PIECE_NORMAL, PIECE_UNKNOWN, PIECE_CONTROL, PIECE_USER_DEFINED, PIECE_UNUSED,
                PIECE_BYTE)

FLAG_ADD_DUMMY_PREFIX = 1 << 0
FLAG_REMOVE_EXTRA_WHITESPACES = 1 << 1
FLAG_ESCAPE_WHITESPACES = 1 << 2
FLAG_BYTE_FALLBACK = 1 << 3
FLAG_TREAT_WHITESPACE_AS_SUFFIX = 1 << 4
_ALL_FLAGS = (FLAG_ADD_DUMMY_PREFIX | FLAG_REMOVE_EXTRA_WHITESPACES | FLAG_ESCAPE_WHITESPACES
              | FLAG_BYTE_FALLBACK | FLAG_TREAT_WHITESPACE_AS_SUFFIX)

# version, flags, unk, bos, eos, pad, pieces, charsmap bytes, lowercase entries, cased ranges,
# case-ignorable ranges
_HEADER = struct.Struct("<IIiiiiIIIII")
_PIECE_HEADER = struct.Struct("<BfI")
_LOWERCASE_HEADER = struct.Struct("<II")
_RANGE = struct.Struct("<II")

# The three Greek sigmas, which is the whole of the contextual rule.
CAPITAL_SIGMA = 0x03A3
SMALL_SIGMA = "σ"
FINAL_SIGMA = "ς"

# A SentencePiece BPE model cannot have more pieces than this and still be the thing we were
# given; the bound keeps a corrupt length from becoming a huge allocation in the runtime.
MAX_RECORDS = 1_000_000
# Byte-fallback pieces are spelled exactly like this.
_BYTE_PIECES = tuple(f"<0x{value:02X}>" for value in range(256))
_BYTE_PIECE_IDS = {piece: index for index, piece in enumerate(_BYTE_PIECES)}


@dataclass
class TokenizerAsset:
    """What the runtime reads, and what a round-trip of the serialized bytes must reproduce."""

    version: int = ASSET_VERSION
    add_dummy_prefix: bool = True
    remove_extra_whitespaces: bool = True
    escape_whitespaces: bool = True
    byte_fallback: bool = True
    treat_whitespace_as_suffix: bool = False
    unknown_id: int = -1
    bos_id: int = -1
    eos_id: int = -1
    pad_id: int = -1
    pieces: List[Tuple[str, float, int]] = field(default_factory=list)
    charsmap: bytes = b""
    lowercase: List[Tuple[int, str]] = field(default_factory=list)
    cased: List[Tuple[int, int]] = field(default_factory=list)
    case_ignorable: List[Tuple[int, int]] = field(default_factory=list)


def lowercase_table() -> List[Tuple[int, str]]:
    """Every code point whose ``str.lower()`` is not itself, in code-point order.

    1,433 entries on Unicode 15.0, one of them a 1:N mapping (U+0130 lowercases to two code
    points), so the table stores a string rather than a code point per entry. The capital sigma
    is in the table too, with its context-free form; the runtime overrides it with the rule the
    two range sets below describe.
    """
    table: List[Tuple[int, str]] = []
    for code_point in range(sys.maxunicode + 1):
        character = chr(code_point)
        lowered = character.lower()
        if lowered != character:
            table.append((code_point, lowered))
    return table


def _ranges(points: List[int]) -> List[Tuple[int, int]]:
    out: List[List[int]] = []
    for code_point in points:
        if out and code_point == out[-1][1] + 1:
            out[-1][1] = code_point
        else:
            out.append([code_point, code_point])
    return [(first, last) for first, last in out]


def case_tables() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """``(Cased, Case_Ignorable)`` as code-point ranges, read out of this interpreter.

    CPython's final-sigma rule (``handle_capital_sigma`` in ``unicodeobject.c``) walks away from
    the sigma in both directions skipping ``Case_Ignorable`` characters and then asks whether the
    first character it stops on is ``Cased``. Rather than ship a copy of two Unicode property
    files, each code point is asked which of the two it is, by putting it next to a sigma and
    reading what ``str.lower()`` does:

    * ``"ΑΣ" + c`` - nothing cased can follow, so the sigma is final unless ``c`` itself is the
      first non-ignorable character and is cased. A plain sigma means **cased and not ignorable**;
    * ``"ΑΣ" + c + "Α"`` - a cased character follows ``c``. A plain sigma now also means
      **ignorable**, because then the scan skips ``c`` and stops on the cased ``Α``.

    Ignorability is checked before casedness in CPython's scan, which is why the two probes are
    read in that order; a character that is both (U+02B0, for instance) comes out ignorable.
    """
    cased: List[int] = []
    ignorable: List[int] = []
    prefix = "Α" + chr(CAPITAL_SIGMA)
    for code_point in range(sys.maxunicode + 1):
        if 0xD800 <= code_point <= 0xDFFF:
            continue
        character = chr(code_point)
        alone = (prefix + character).lower()[1]
        followed = (prefix + character + "Α").lower()[1]
        if alone == SMALL_SIGMA:
            cased.append(code_point)
        if followed == SMALL_SIGMA and alone == FINAL_SIGMA:
            ignorable.append(code_point)
    return _ranges(cased), _ranges(ignorable)


def _require_bpe_model(trainer) -> None:
    # The merge order lives in the scores; a Unigram model's scores are log probabilities and the
    # same code would silently produce a different segmentation.
    from sentencepiece import sentencepiece_model_pb2 as model_pb2

    name = model_pb2.TrainerSpec.ModelType.Name(trainer.model_type)
    if name != "BPE":
        raise NotImplementedError(
            f"tipsv2 is qualified for a SentencePiece BPE model, not {name}")


def _require_supported_normalizer(model) -> None:
    normalizer = model.normalizer_spec
    if normalizer.normalization_rule_tsv:
        raise NotImplementedError(
            "tipsv2 cannot carry a normalization_rule_tsv; the asset holds a precompiled charsmap")
    if model.HasField("denormalizer_spec") and model.denormalizer_spec.precompiled_charsmap:
        raise NotImplementedError("tipsv2 cannot carry a denormalizer charsmap")
    if not normalizer.precompiled_charsmap:
        # Dropping it would change token ids for every non-ASCII prompt, silently.
        raise NotImplementedError(
            f"tipsv2 expects a precompiled charsmap; normalizer {normalizer.name!r} has none")


def _validate_charsmap(charsmap: bytes) -> None:
    """The three things `PrecompiledCharsMap::load` insists on, checked where the build can stop.

    The charsmap is copied into the asset verbatim: a Darts double-array trie behind a four-byte
    little-endian length, then the replacement strings. Nothing downstream looks inside it again
    until the runtime loads the bundle, so a truncated or misaligned map would otherwise produce
    a bundle that builds, ships and then refuses to open. These are the loader's own conditions,
    not stricter ones.
    """
    if len(charsmap) < 4:
        raise ValueError(f"tipsv2 precompiled charsmap is {len(charsmap)} bytes, too short to "
                         f"hold its own length")
    trie_bytes = int.from_bytes(charsmap[:4], "little")
    if trie_bytes == 0 or trie_bytes % 4 != 0:
        raise ValueError(f"tipsv2 precompiled charsmap states a trie of {trie_bytes} bytes, "
                         f"which is not a positive multiple of four")
    if trie_bytes > len(charsmap) - 4:
        raise ValueError(f"tipsv2 precompiled charsmap states a trie of {trie_bytes} bytes but "
                         f"carries only {len(charsmap) - 4}")


def _validate_pieces(pieces: List[Tuple[str, float, int]]) -> None:
    if not pieces:
        raise ValueError("tipsv2 tokenizer has no pieces")
    if len(pieces) > MAX_RECORDS:
        raise ValueError(f"tipsv2 tokenizer has {len(pieces)} pieces, more than {MAX_RECORDS}")
    seen: Dict[str, int] = {}
    unknown_count = 0
    byte_pieces = set()
    for index, (text, score, piece_type) in enumerate(pieces):
        if not text:
            # An empty piece matches at every position without advancing.
            raise ValueError(f"tipsv2 tokenizer piece {index} is empty")
        if text in seen:
            raise ValueError(
                f"tipsv2 tokenizer pieces {seen[text]} and {index} are both {text!r}")
        seen[text] = index
        if score != score or score in (float("inf"), float("-inf")):
            raise ValueError(f"tipsv2 tokenizer piece {index} has a non-finite score")
        if piece_type not in _PIECE_TYPES:
            raise ValueError(f"tipsv2 tokenizer piece {index} has type {piece_type}")
        if piece_type == PIECE_UNKNOWN:
            unknown_count += 1
        if piece_type == PIECE_BYTE:
            if text not in _BYTE_PIECE_IDS:
                raise ValueError(f"tipsv2 tokenizer byte piece {index} is spelled {text!r}")
            byte_pieces.add(text)
    if unknown_count != 1:
        raise ValueError(
            f"tipsv2 tokenizer must have exactly one unknown piece, not {unknown_count}")
    return byte_pieces


def _validate_special_ids(asset: TokenizerAsset, byte_pieces: set) -> None:
    count = len(asset.pieces)

    def check(identifier: int, expected_type: int, what: str, required: bool) -> None:
        if identifier < 0:
            if required:
                raise ValueError(f"tipsv2 tokenizer has no {what}")
            return
        if identifier >= count:
            raise ValueError(f"tipsv2 tokenizer {what} {identifier} is outside the vocabulary")
        actual = asset.pieces[identifier][2]
        if actual != expected_type:
            raise ValueError(
                f"tipsv2 tokenizer {what} {identifier} has piece type {actual}, "
                f"not {expected_type}")

    check(asset.unknown_id, PIECE_UNKNOWN, "unknown id", True)
    # The checkpoint's tokenizer disables both, but an asset that carried the wrong kind of piece
    # under these names would be wrong for any later use of them.
    check(asset.bos_id, PIECE_CONTROL, "BOS id", False)
    check(asset.eos_id, PIECE_CONTROL, "EOS id", False)
    # The engine derives its padding mask from this id, so it has to be a piece that no prompt can
    # produce - which is what a control piece is.
    check(asset.pad_id, PIECE_CONTROL, "padding id", True)

    enabled = [value for value in (asset.unknown_id, asset.bos_id, asset.eos_id, asset.pad_id)
               if value >= 0]
    if len(set(enabled)) != len(enabled):
        raise ValueError("tipsv2 tokenizer special ids are not distinct")

    if asset.byte_fallback:
        if len(byte_pieces) != 256:
            missing = sorted(set(_BYTE_PIECES) - byte_pieces)
            raise ValueError(
                "tipsv2 tokenizer byte fallback needs one piece per byte; "
                f"{len(missing)} are missing, first {missing[:4]}")
    elif byte_pieces:
        raise ValueError("tipsv2 tokenizer has byte pieces but no byte fallback")


def _validate_lowercase(table: List[Tuple[int, str]]) -> None:
    if len(table) > MAX_RECORDS:
        raise ValueError(f"tipsv2 lowercase table has {len(table)} entries")
    previous = -1
    for code_point, lowered in table:
        if code_point <= previous:
            raise ValueError("tipsv2 lowercase table is not sorted by code point")
        if not 0 <= code_point <= sys.maxunicode:
            raise ValueError(f"tipsv2 lowercase table has code point {code_point}")
        if 0xD800 <= code_point <= 0xDFFF:
            raise ValueError("tipsv2 lowercase table contains a surrogate code point")
        if not lowered:
            raise ValueError(f"tipsv2 lowercase table maps U+{code_point:04X} to nothing")
        previous = code_point


def _validate_case_tables(asset: "TokenizerAsset") -> None:
    """The two tables, each on its own and then against each other.

    They answer one question between them - whether a capital sigma is word-final - and the
    runtime refuses an asset carrying one without the other, because one alone would silently
    give every sigma the same form. The builder has to refuse it here, where the checkpoint that
    produced it is still in hand, rather than let a bundle be written that cannot be opened.
    """
    _validate_ranges(asset.cased, "cased table")
    _validate_ranges(asset.case_ignorable, "case-ignorable table")
    if not asset.cased or not asset.case_ignorable:
        # Both, or the final-sigma rule is not merely weakened but silently off: with no cased
        # table the scan finds nothing cased after a sigma and every one of them lowercases to
        # the non-final form, changing the ids of every Greek word that ends in one.
        raise ValueError(
            "tipsv2 tokenizer asset needs both case tables (cased: "
            f"{len(asset.cased)} ranges, case-ignorable: {len(asset.case_ignorable)}); they "
            "answer one question between them and the runtime refuses any other combination")


def _validate_ranges(ranges: List[Tuple[int, int]], what: str) -> None:
    """Sorted, non-empty, non-touching ranges, so the runtime can binary-search them."""
    if len(ranges) > MAX_RECORDS:
        raise ValueError(f"tipsv2 {what} has {len(ranges)} ranges")
    previous = -2
    for first, last in ranges:
        if not 0 <= first <= last <= sys.maxunicode:
            # One chained comparison: `not 0 <= first and last <= maxunicode` binds as
            # `(not (0 <= first)) and (...)`, which lets an endpoint past U+10FFFF through and
            # produces an asset the runtime then refuses.
            raise ValueError(f"tipsv2 {what} has a range outside Unicode or an empty range")
        if first <= previous + 1:
            # Touching ranges would still be correct but say the producer merged them wrongly,
            # and the runtime's search assumes they are disjoint and ordered.
            raise ValueError(f"tipsv2 {what} ranges are not sorted and separated")
        previous = last


def serialize_tokenizer_asset(asset: TokenizerAsset) -> bytes:
    """Pack the asset the way ``parse_tokenizer_asset`` and the runtime read it."""
    byte_pieces = _validate_pieces(asset.pieces)
    _validate_special_ids(asset, byte_pieces)
    _validate_lowercase(asset.lowercase)
    _validate_case_tables(asset)
    if asset.version != ASSET_VERSION:
        raise ValueError(f"tipsv2 tokenizer asset version {asset.version} is not supported")
    if len(asset.charsmap) > 1 << 30:
        raise ValueError("tipsv2 charsmap is implausibly large")
    _validate_charsmap(asset.charsmap)

    flags = 0
    for enabled, bit in ((asset.add_dummy_prefix, FLAG_ADD_DUMMY_PREFIX),
                         (asset.remove_extra_whitespaces, FLAG_REMOVE_EXTRA_WHITESPACES),
                         (asset.escape_whitespaces, FLAG_ESCAPE_WHITESPACES),
                         (asset.byte_fallback, FLAG_BYTE_FALLBACK),
                         (asset.treat_whitespace_as_suffix, FLAG_TREAT_WHITESPACE_AS_SUFFIX)):
        if enabled:
            flags |= bit

    out = bytearray(MAGIC)
    out += _HEADER.pack(asset.version, flags, asset.unknown_id, asset.bos_id, asset.eos_id,
                        asset.pad_id, len(asset.pieces), len(asset.charsmap),
                        len(asset.lowercase), len(asset.cased), len(asset.case_ignorable))
    for text, score, piece_type in asset.pieces:
        encoded = text.encode("utf-8")
        out += _PIECE_HEADER.pack(piece_type, score, len(encoded))
        out += encoded
    out += asset.charsmap
    for code_point, lowered in asset.lowercase:
        encoded = lowered.encode("utf-8")
        out += _LOWERCASE_HEADER.pack(code_point, len(encoded))
        out += encoded
    for table in (asset.cased, asset.case_ignorable):
        for first, last in table:
            out += _RANGE.pack(first, last)
    return bytes(out)


class _AssetReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def take(self, length: int) -> bytes:
        if length < 0 or length > len(self._data) - self._offset:
            raise ValueError("tipsv2 tokenizer asset is truncated")
        chunk = self._data[self._offset:self._offset + length]
        self._offset += length
        return chunk

    def unpack(self, layout: struct.Struct):
        return layout.unpack(self.take(layout.size))

    def finish(self) -> None:
        if self._offset != len(self._data):
            raise ValueError("tipsv2 tokenizer asset has trailing bytes")


def parse_tokenizer_asset(data: bytes) -> TokenizerAsset:
    """Read back what ``serialize_tokenizer_asset`` wrote, the way the runtime reads it.

    The build step round-trips its own output through this so a packing mistake fails the build
    rather than the engine.
    """
    reader = _AssetReader(data)
    if reader.take(len(MAGIC)) != MAGIC:
        raise ValueError("tipsv2 tokenizer asset has the wrong magic")
    (version, flags, unknown_id, bos_id, eos_id, pad_id, piece_count, charsmap_length,
     lowercase_count, cased_count, case_ignorable_count) = reader.unpack(_HEADER)
    if version != ASSET_VERSION:
        raise ValueError(f"tipsv2 tokenizer asset version {version} is not supported")
    if flags & ~_ALL_FLAGS:
        raise ValueError(f"tipsv2 tokenizer asset has unknown flags {flags:#x}")
    if max(piece_count, lowercase_count, cased_count, case_ignorable_count) > MAX_RECORDS:
        raise ValueError("tipsv2 tokenizer asset record count is invalid")

    pieces: List[Tuple[str, float, int]] = []
    for _ in range(piece_count):
        piece_type, score, length = reader.unpack(_PIECE_HEADER)
        pieces.append((reader.take(length).decode("utf-8"), score, piece_type))
    charsmap = reader.take(charsmap_length)
    lowercase: List[Tuple[int, str]] = []
    for _ in range(lowercase_count):
        code_point, length = reader.unpack(_LOWERCASE_HEADER)
        lowercase.append((code_point, reader.take(length).decode("utf-8")))
    tables = []
    for count in (cased_count, case_ignorable_count):
        tables.append([reader.unpack(_RANGE) for _ in range(count)])
    reader.finish()

    asset = TokenizerAsset(
        version=version,
        add_dummy_prefix=bool(flags & FLAG_ADD_DUMMY_PREFIX),
        remove_extra_whitespaces=bool(flags & FLAG_REMOVE_EXTRA_WHITESPACES),
        escape_whitespaces=bool(flags & FLAG_ESCAPE_WHITESPACES),
        byte_fallback=bool(flags & FLAG_BYTE_FALLBACK),
        treat_whitespace_as_suffix=bool(flags & FLAG_TREAT_WHITESPACE_AS_SUFFIX),
        unknown_id=unknown_id, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
        pieces=pieces, charsmap=charsmap, lowercase=lowercase,
        cased=tables[0], case_ignorable=tables[1])
    byte_pieces = _validate_pieces(asset.pieces)
    _validate_special_ids(asset, byte_pieces)
    _validate_lowercase(asset.lowercase)
    _validate_case_tables(asset)
    return asset


def convert_tokenizer_model(model_bytes: bytes) -> Tuple[bytes, TokenizerAsset]:
    """Convert ``tokenizer.model`` into the runtime asset, and return both."""
    try:
        from sentencepiece import sentencepiece_model_pb2 as model_pb2
    except ImportError as error:  # pragma: no cover - the build environment pins sentencepiece
        raise RuntimeError("tipsv2 needs sentencepiece in the build environment") from error

    model = model_pb2.ModelProto()
    model.ParseFromString(model_bytes)
    _require_bpe_model(model.trainer_spec)
    _require_supported_normalizer(model)

    cased, case_ignorable = case_tables()
    normalizer = model.normalizer_spec
    asset = TokenizerAsset(
        add_dummy_prefix=bool(normalizer.add_dummy_prefix),
        remove_extra_whitespaces=bool(normalizer.remove_extra_whitespaces),
        escape_whitespaces=bool(normalizer.escape_whitespaces),
        byte_fallback=bool(model.trainer_spec.byte_fallback),
        treat_whitespace_as_suffix=bool(model.trainer_spec.treat_whitespace_as_suffix),
        unknown_id=int(model.trainer_spec.unk_id),
        bos_id=int(model.trainer_spec.bos_id),
        eos_id=int(model.trainer_spec.eos_id),
        pad_id=int(model.trainer_spec.pad_id),
        pieces=[(piece.piece, float(piece.score), int(piece.type)) for piece in model.pieces],
        charsmap=bytes(normalizer.precompiled_charsmap),
        lowercase=lowercase_table(),
        cased=cased,
        case_ignorable=case_ignorable,
    )
    if len(asset.pieces) != int(model.trainer_spec.vocab_size):
        raise ValueError(
            f"tipsv2 tokenizer states vocab_size={model.trainer_spec.vocab_size} but ships "
            f"{len(asset.pieces)} pieces")

    data = serialize_tokenizer_asset(asset)
    # Fail the build, not the engine: read it back the way the runtime will.
    round_trip = parse_tokenizer_asset(data)
    if (round_trip.pieces != asset.pieces or round_trip.charsmap != asset.charsmap
            or round_trip.lowercase != asset.lowercase or round_trip.cased != asset.cased
            or round_trip.case_ignorable != asset.case_ignorable):
        raise RuntimeError("tipsv2 tokenizer asset does not survive its own round trip")
    return data, asset


def unicode_version() -> str:
    """The Unicode version of the lowercase table this build carries."""
    return unicodedata.unidata_version
