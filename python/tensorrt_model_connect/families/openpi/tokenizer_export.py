# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export the PaliGemma SentencePiece BPE model for the native OpenPI runtime.

The source file is a build-time-only SentencePiece protobuf.  The resulting
``TRTMCBPE`` asset contains only the data consumed by the dependency-free C++
runtime in ``src/runtime/models/openpi/paligemma_bpe.{h,cpp}``.

PaliGemma uses SentencePiece's score-ordered BPE model with byte fallback.  It
does not use the SentencePiece Unigram model.  The pinned public tokenizer has
an identity normalizer with no precompiled character map.  Version 1 refuses
non-identity normalizers because silently dropping their rules would change
token IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ASSET_MAGIC = b"TRTMCBPE"
ASSET_VERSION = 1

_FLAG_DUMMY_PREFIX = 1 << 0
_FLAG_REMOVE_EXTRA_WHITESPACES = 1 << 1
_FLAG_ESCAPE_WHITESPACES = 1 << 2
_FLAG_BYTE_FALLBACK = 1 << 3
_FLAG_WHITESPACE_AS_SUFFIX = 1 << 4
_KNOWN_FLAGS = (
    _FLAG_DUMMY_PREFIX
    | _FLAG_REMOVE_EXTRA_WHITESPACES
    | _FLAG_ESCAPE_WHITESPACES
    | _FLAG_BYTE_FALLBACK
    | _FLAG_WHITESPACE_AS_SUFFIX
)

_HEADER = struct.Struct("<IIiiiiII")
_PIECE_HEADER = struct.Struct("<BfI")
_RULE_HEADER = struct.Struct("<II")
_MAX_RECORDS = 1_000_000

_NORMAL = 1
_UNKNOWN = 2
_CONTROL = 3
_USER_DEFINED = 4
_UNUSED = 5
_BYTE = 6
_VALID_PIECE_TYPES = frozenset({_NORMAL, _UNKNOWN, _CONTROL, _USER_DEFINED, _UNUSED, _BYTE})


class TokenizerExportError(ValueError):
    """Raised when an input tokenizer cannot be represented exactly."""


class TokenizerExportDependencyError(RuntimeError):
    """Raised when build-time protobuf support is unavailable."""


@dataclass(frozen=True)
class FlatBpePiece:
    """One vocabulary entry in source ID order."""

    text: str
    score: float
    piece_type: int


@dataclass(frozen=True)
class FlatBpeAsset:
    """Parsed ``TRTMCBPE`` v1 data used for build-time verification."""

    version: int
    add_dummy_prefix: bool
    remove_extra_whitespaces: bool
    escape_whitespaces: bool
    byte_fallback: bool
    treat_whitespace_as_suffix: bool
    unknown_id: int
    bos_id: int
    eos_id: int
    pad_id: int
    pieces: tuple[FlatBpePiece, ...]
    normalization_rules: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TokenizerExportMetadata:
    """Hash-bound provenance returned to ``prepare_model_dir`` callers."""

    schema_version: int
    source_model_type: str
    source_sha256: str
    asset_sha256: str
    asset_size: int
    piece_count: int
    piece_type_counts: dict[str, int]
    normalization_name: str
    normalization_rule_count: int
    add_dummy_prefix: bool
    remove_extra_whitespaces: bool
    escape_whitespaces: bool
    byte_fallback: bool
    treat_whitespace_as_suffix: bool
    unknown_id: int
    bos_id: int
    eos_id: int
    pad_id: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready copy for the conversion manifest."""

        return asdict(self)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_sentencepiece_proto_module():
    """Load the build-only protobuf bindings without adding a runtime import."""

    try:
        from sentencepiece import sentencepiece_model_pb2
    except (ImportError, ModuleNotFoundError) as exc:
        raise TokenizerExportDependencyError(
            "OpenPI tokenizer export requires build-time SentencePiece protobuf bindings; "
            "install both 'sentencepiece' and 'protobuf' in the builder environment"
        ) from exc
    return sentencepiece_model_pb2


def _special_piece(
    pieces: Sequence[FlatBpePiece],
    token_id: int,
    *,
    expected_type: int,
    name: str,
    required: bool,
) -> None:
    if token_id < 0:
        if required:
            raise TokenizerExportError(f"PaliGemma tokenizer is missing its {name}")
        return
    if token_id >= len(pieces) or pieces[token_id].piece_type != expected_type:
        raise TokenizerExportError(
            f"PaliGemma tokenizer {name} {token_id} does not reference the required piece type"
        )


def _validate_piece_inventory(
    pieces: Sequence[FlatBpePiece],
    *,
    unknown_id: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    byte_fallback: bool,
) -> None:
    if not pieces:
        raise TokenizerExportError("PaliGemma tokenizer contains no pieces")
    if len(pieces) > _MAX_RECORDS:
        raise TokenizerExportError(
            f"PaliGemma tokenizer exceeds the TRTMCBPE v1 limit of {_MAX_RECORDS} pieces"
        )

    seen: set[str] = set()
    unknown_count = 0
    byte_pieces: set[str] = set()
    for token_id, piece in enumerate(pieces):
        if not piece.text:
            raise TokenizerExportError(f"PaliGemma tokenizer piece {token_id} is empty")
        if piece.text in seen:
            raise TokenizerExportError(
                f"PaliGemma tokenizer piece {token_id} duplicates {piece.text!r}"
            )
        seen.add(piece.text)
        if not math.isfinite(piece.score):
            raise TokenizerExportError(
                f"PaliGemma tokenizer piece {token_id} has a non-finite score"
            )
        if piece.piece_type not in _VALID_PIECE_TYPES:
            raise TokenizerExportError(
                f"PaliGemma tokenizer piece {token_id} has unsupported type {piece.piece_type}"
            )
        encoded = piece.text.encode("utf-8")
        if len(encoded) > 0xFFFFFFFF:
            raise TokenizerExportError(
                f"PaliGemma tokenizer piece {token_id} exceeds the TRTMCBPE string limit"
            )
        if piece.piece_type == _UNKNOWN:
            unknown_count += 1
        elif piece.piece_type == _BYTE:
            byte_pieces.add(piece.text)

    if unknown_count != 1:
        raise TokenizerExportError(
            f"PaliGemma tokenizer must have exactly one unknown piece, found {unknown_count}"
        )
    _special_piece(
        pieces,
        unknown_id,
        expected_type=_UNKNOWN,
        name="unknown id",
        required=True,
    )
    _special_piece(
        pieces,
        bos_id,
        expected_type=_CONTROL,
        name="BOS id",
        required=True,
    )
    _special_piece(
        pieces,
        eos_id,
        expected_type=_CONTROL,
        name="EOS id",
        required=False,
    )
    _special_piece(
        pieces,
        pad_id,
        expected_type=_CONTROL,
        name="padding id",
        required=False,
    )
    enabled_special_ids = [
        token_id for token_id in (unknown_id, bos_id, eos_id, pad_id) if token_id >= 0
    ]
    if len(enabled_special_ids) != len(set(enabled_special_ids)):
        raise TokenizerExportError("PaliGemma tokenizer special token ids are not distinct")

    if byte_fallback:
        expected_bytes = {f"<0x{value:02X}>" for value in range(256)}
        missing = sorted(expected_bytes - byte_pieces)
        unexpected = sorted(byte_pieces - expected_bytes)
        if missing or unexpected or len(byte_pieces) != 256:
            raise TokenizerExportError(
                "PaliGemma byte fallback table is not exact: "
                f"missing={missing[:4]}, unexpected={unexpected[:4]}"
            )
    elif byte_pieces:
        raise TokenizerExportError(
            "PaliGemma tokenizer contains byte pieces while byte_fallback is disabled"
        )


def _flags(
    *,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    byte_fallback: bool,
    treat_whitespace_as_suffix: bool,
) -> int:
    return (
        (_FLAG_DUMMY_PREFIX if add_dummy_prefix else 0)
        | (_FLAG_REMOVE_EXTRA_WHITESPACES if remove_extra_whitespaces else 0)
        | (_FLAG_ESCAPE_WHITESPACES if escape_whitespaces else 0)
        | (_FLAG_BYTE_FALLBACK if byte_fallback else 0)
        | (_FLAG_WHITESPACE_AS_SUFFIX if treat_whitespace_as_suffix else 0)
    )


def _serialize_asset(
    pieces: Sequence[FlatBpePiece],
    *,
    flags: int,
    unknown_id: int,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    normalization_rules: Sequence[tuple[str, str]] = (),
) -> bytes:
    if flags & ~_KNOWN_FLAGS:
        raise TokenizerExportError(f"TRTMCBPE flags contain unknown bits: {flags:#x}")
    if len(normalization_rules) > _MAX_RECORDS:
        raise TokenizerExportError(
            f"TRTMCBPE v1 exceeds its limit of {_MAX_RECORDS} normalization rules"
        )

    output = bytearray(ASSET_MAGIC)
    output.extend(
        _HEADER.pack(
            ASSET_VERSION,
            flags,
            unknown_id,
            bos_id,
            eos_id,
            pad_id,
            len(pieces),
            len(normalization_rules),
        )
    )
    for piece in pieces:
        encoded = piece.text.encode("utf-8")
        output.extend(_PIECE_HEADER.pack(piece.piece_type, piece.score, len(encoded)))
        output.extend(encoded)
    for source, replacement in normalization_rules:
        source_bytes = source.encode("utf-8")
        replacement_bytes = replacement.encode("utf-8")
        if not source_bytes:
            raise TokenizerExportError("TRTMCBPE normalization rule source cannot be empty")
        if len(source_bytes) > 0xFFFFFFFF or len(replacement_bytes) > 0xFFFFFFFF:
            raise TokenizerExportError("TRTMCBPE normalization rule exceeds uint32 string size")
        output.extend(_RULE_HEADER.pack(len(source_bytes), len(replacement_bytes)))
        output.extend(source_bytes)
        output.extend(replacement_bytes)
    return bytes(output)


class _AssetReader:
    def __init__(self, data: bytes) -> None:
        self._data = memoryview(data)
        self._offset = 0

    def read(self, size: int) -> bytes:
        if size < 0 or self._offset + size > len(self._data):
            raise TokenizerExportError("truncated TRTMCBPE asset")
        value = bytes(self._data[self._offset : self._offset + size])
        self._offset += size
        return value

    def unpack(self, layout: struct.Struct) -> tuple[Any, ...]:
        return layout.unpack(self.read(layout.size))

    @property
    def at_end(self) -> bool:
        return self._offset == len(self._data)


def parse_trtmcbpe_asset(data: bytes) -> FlatBpeAsset:
    """Parse and validate bytes using the C++ runtime's exact v1 layout."""

    reader = _AssetReader(data)
    if reader.read(len(ASSET_MAGIC)) != ASSET_MAGIC:
        raise TokenizerExportError("invalid TRTMCBPE asset magic")
    (
        version,
        flags,
        unknown_id,
        bos_id,
        eos_id,
        pad_id,
        piece_count,
        rule_count,
    ) = reader.unpack(_HEADER)
    if version != ASSET_VERSION:
        raise TokenizerExportError(f"unsupported TRTMCBPE asset version {version}")
    if flags & ~_KNOWN_FLAGS:
        raise TokenizerExportError(f"TRTMCBPE asset contains unknown flags {flags:#x}")
    if piece_count > _MAX_RECORDS or rule_count > _MAX_RECORDS:
        raise TokenizerExportError("TRTMCBPE asset record count exceeds its limit")

    pieces: list[FlatBpePiece] = []
    for token_id in range(piece_count):
        piece_type, score, byte_length = reader.unpack(_PIECE_HEADER)
        try:
            text = reader.read(byte_length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TokenizerExportError(f"TRTMCBPE piece {token_id} is not valid UTF-8") from exc
        pieces.append(FlatBpePiece(text=text, score=score, piece_type=piece_type))

    rules: list[tuple[str, str]] = []
    for rule_index in range(rule_count):
        source_length, replacement_length = reader.unpack(_RULE_HEADER)
        try:
            source = reader.read(source_length).decode("utf-8")
            replacement = reader.read(replacement_length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TokenizerExportError(
                f"TRTMCBPE normalization rule {rule_index} is not valid UTF-8"
            ) from exc
        rules.append((source, replacement))
    if not reader.at_end:
        raise TokenizerExportError("TRTMCBPE asset contains trailing bytes")

    byte_fallback = bool(flags & _FLAG_BYTE_FALLBACK)
    _validate_piece_inventory(
        pieces,
        unknown_id=unknown_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        byte_fallback=byte_fallback,
    )
    if any(not source for source, _ in rules):
        raise TokenizerExportError("TRTMCBPE asset contains an empty normalization source")
    if len({source for source, _ in rules}) != len(rules):
        raise TokenizerExportError("TRTMCBPE asset contains duplicate normalization sources")

    return FlatBpeAsset(
        version=version,
        add_dummy_prefix=bool(flags & _FLAG_DUMMY_PREFIX),
        remove_extra_whitespaces=bool(flags & _FLAG_REMOVE_EXTRA_WHITESPACES),
        escape_whitespaces=bool(flags & _FLAG_ESCAPE_WHITESPACES),
        byte_fallback=byte_fallback,
        treat_whitespace_as_suffix=bool(flags & _FLAG_WHITESPACE_AS_SUFFIX),
        unknown_id=unknown_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        pieces=tuple(pieces),
        normalization_rules=tuple(rules),
    )


def convert_paligemma_tokenizer_model(
    model_bytes: bytes,
) -> tuple[bytes, TokenizerExportMetadata]:
    """Convert one raw SentencePiece BPE protobuf to ``TRTMCBPE`` v1 bytes."""

    if not model_bytes:
        raise TokenizerExportError("PaliGemma SentencePiece model is empty")
    sentencepiece_model_pb2 = _load_sentencepiece_proto_module()
    model = sentencepiece_model_pb2.ModelProto()
    try:
        model.ParseFromString(model_bytes)
    except Exception as exc:
        raise TokenizerExportError("PaliGemma tokenizer is not a valid ModelProto") from exc

    trainer = model.trainer_spec
    bpe_model_type = int(sentencepiece_model_pb2.TrainerSpec.BPE)
    if int(trainer.model_type) != bpe_model_type:
        raise TokenizerExportError(
            "OpenPI requires the PaliGemma SentencePiece BPE model; "
            f"source model_type={int(trainer.model_type)} is not BPE ({bpe_model_type})"
        )

    normalizer = model.normalizer_spec
    if normalizer.name != "identity":
        raise TokenizerExportError(
            "TRTMCBPE v1 only supports the pinned PaliGemma identity normalizer; "
            f"source normalizer is {normalizer.name!r}"
        )
    if normalizer.precompiled_charsmap:
        raise TokenizerExportError(
            "TRTMCBPE v1 cannot safely flatten a non-empty precompiled_charsmap; "
            "refusing to drop tokenizer normalization semantics"
        )
    if normalizer.normalization_rule_tsv:
        raise TokenizerExportError("TRTMCBPE v1 does not accept an external normalization_rule_tsv")
    if model.HasField("denormalizer_spec"):
        raise TokenizerExportError(
            "TRTMCBPE v1 does not represent a SentencePiece denormalizer_spec"
        )

    pieces = tuple(
        FlatBpePiece(
            text=piece.piece,
            score=float(piece.score),
            piece_type=int(piece.type),
        )
        for piece in model.pieces
    )
    unknown_id = int(trainer.unk_id)
    bos_id = int(trainer.bos_id)
    eos_id = int(trainer.eos_id)
    pad_id = int(trainer.pad_id)
    byte_fallback = bool(trainer.byte_fallback)
    _validate_piece_inventory(
        pieces,
        unknown_id=unknown_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        byte_fallback=byte_fallback,
    )

    add_dummy_prefix = bool(normalizer.add_dummy_prefix)
    remove_extra_whitespaces = bool(normalizer.remove_extra_whitespaces)
    escape_whitespaces = bool(normalizer.escape_whitespaces)
    treat_whitespace_as_suffix = bool(trainer.treat_whitespace_as_suffix)
    flags = _flags(
        add_dummy_prefix=add_dummy_prefix,
        remove_extra_whitespaces=remove_extra_whitespaces,
        escape_whitespaces=escape_whitespaces,
        byte_fallback=byte_fallback,
        treat_whitespace_as_suffix=treat_whitespace_as_suffix,
    )
    # The pinned tokenizer's normalizer is identity and has no rules.  Keep the
    # v1 rule count explicit so a future non-identity model cannot be mistaken
    # for an exact conversion.
    normalization_rules: tuple[tuple[str, str], ...] = ()
    asset_bytes = _serialize_asset(
        pieces,
        flags=flags,
        unknown_id=unknown_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        normalization_rules=normalization_rules,
    )

    parsed = parse_trtmcbpe_asset(asset_bytes)
    if parsed.pieces != pieces:
        raise TokenizerExportError("TRTMCBPE serializer failed its piece round-trip check")

    type_names = {
        _NORMAL: "normal",
        _UNKNOWN: "unknown",
        _CONTROL: "control",
        _USER_DEFINED: "user_defined",
        _UNUSED: "unused",
        _BYTE: "byte",
    }
    counts = Counter(piece.piece_type for piece in pieces)
    metadata = TokenizerExportMetadata(
        schema_version=ASSET_VERSION,
        source_model_type="BPE",
        source_sha256=_sha256(model_bytes),
        asset_sha256=_sha256(asset_bytes),
        asset_size=len(asset_bytes),
        piece_count=len(pieces),
        piece_type_counts={type_names[kind]: counts.get(kind, 0) for kind in sorted(type_names)},
        normalization_name=normalizer.name,
        normalization_rule_count=0,
        add_dummy_prefix=add_dummy_prefix,
        remove_extra_whitespaces=remove_extra_whitespaces,
        escape_whitespaces=escape_whitespaces,
        byte_fallback=byte_fallback,
        treat_whitespace_as_suffix=treat_whitespace_as_suffix,
        unknown_id=unknown_id,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    return asset_bytes, metadata


def _atomic_write(path: Path, data: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing tokenizer asset: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def export_paligemma_bpe_model(
    source_model: str | Path,
    output_asset: str | Path,
    *,
    overwrite: bool = False,
) -> TokenizerExportMetadata:
    """Convert ``source_model`` and atomically write one native tokenizer asset.

    ``prepare_model_dir`` can call this helper with its raw tokenizer path and a
    staging destination such as ``tokenizer.model``.  The
    returned metadata is ready to embed under the manifest's tokenizer entry.
    """

    source = Path(source_model).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PaliGemma SentencePiece model does not exist: {source}")
    model_bytes = source.read_bytes()
    asset_bytes, metadata = convert_paligemma_tokenizer_model(model_bytes)
    output = Path(output_asset).expanduser().resolve()
    _atomic_write(output, asset_bytes, overwrite=overwrite)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw PaliGemma SentencePiece .model")
    parser.add_argument("--output", required=True, help="TRTMCBPE v1 asset to create")
    parser.add_argument(
        "--metadata-output",
        help="optional JSON file for hash-bound exporter metadata",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing output asset")
    args = parser.parse_args(argv)

    metadata = export_paligemma_bpe_model(args.input, args.output, overwrite=args.force)
    payload = {
        "input": str(Path(args.input).expanduser().resolve()),
        "output": str(Path(args.output).expanduser().resolve()),
        **metadata.to_dict(),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.metadata_output:
        metadata_path = Path(args.metadata_output).expanduser().resolve()
        _atomic_write(metadata_path, rendered.encode("utf-8"), overwrite=args.force)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
