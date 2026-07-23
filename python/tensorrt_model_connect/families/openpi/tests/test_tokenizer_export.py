# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import hashlib
import json
import struct
from pathlib import Path

import pytest

from tensorrt_model_connect.families.openpi import tokenizer_export


def _add_piece(model, text: str, score: float, piece_type: int) -> None:
    piece = model.pieces.add()
    piece.piece = text
    piece.score = score
    piece.type = piece_type


def _synthetic_bpe_model():
    try:
        pb = tokenizer_export._load_sentencepiece_proto_module()
    except tokenizer_export.TokenizerExportDependencyError as exc:
        pytest.skip(str(exc))
    model = pb.ModelProto()
    trainer = model.trainer_spec
    trainer.model_type = pb.TrainerSpec.BPE
    trainer.byte_fallback = True
    trainer.treat_whitespace_as_suffix = False
    trainer.unk_id = 3
    trainer.bos_id = 2
    trainer.eos_id = 1
    trainer.pad_id = 0

    normalizer = model.normalizer_spec
    normalizer.name = "identity"
    normalizer.add_dummy_prefix = False
    normalizer.remove_extra_whitespaces = False
    normalizer.escape_whitespaces = True

    piece_type = pb.ModelProto.SentencePiece
    _add_piece(model, "<pad>", 0.0, piece_type.CONTROL)
    _add_piece(model, "<eos>", 0.0, piece_type.CONTROL)
    _add_piece(model, "<bos>", 0.0, piece_type.CONTROL)
    _add_piece(model, "<unk>", 0.0, piece_type.UNKNOWN)
    _add_piece(model, "<robot>", 0.0, piece_type.USER_DEFINED)
    for value in range(256):
        _add_piece(model, f"<0x{value:02X}>", 0.0, piece_type.BYTE)
    _add_piece(model, "h", -2.0, piece_type.NORMAL)
    _add_piece(model, "i", -3.0, piece_type.NORMAL)
    _add_piece(model, "hi", 7.5, piece_type.NORMAL)
    return model


def _serialize(model) -> bytes:
    return model.SerializeToString(deterministic=True)


def test_synthetic_sentencepiece_bpe_round_trips_exact_v1_layout() -> None:
    model_bytes = _serialize(_synthetic_bpe_model())
    asset_bytes, metadata = tokenizer_export.convert_paligemma_tokenizer_model(model_bytes)
    parsed = tokenizer_export.parse_trtmcbpe_asset(asset_bytes)

    assert asset_bytes.startswith(b"TRTMCBPE")
    expected_flags = (1 << 2) | (1 << 3)
    expected_piece_count = 4 + 1 + 256 + 3
    assert asset_bytes[8:40] == struct.pack(
        "<IIiiiiII",
        1,
        expected_flags,
        3,
        2,
        1,
        0,
        expected_piece_count,
        0,
    )
    assert parsed.version == 1
    assert not parsed.add_dummy_prefix
    assert not parsed.remove_extra_whitespaces
    assert parsed.escape_whitespaces
    assert parsed.byte_fallback
    assert not parsed.treat_whitespace_as_suffix
    assert (parsed.pad_id, parsed.eos_id, parsed.bos_id, parsed.unknown_id) == (0, 1, 2, 3)
    assert len(parsed.pieces) == expected_piece_count
    assert parsed.pieces[0] == tokenizer_export.FlatBpePiece("<pad>", 0.0, 3)
    assert parsed.pieces[4] == tokenizer_export.FlatBpePiece("<robot>", 0.0, 4)
    assert parsed.pieces[5].text == "<0x00>"
    assert parsed.pieces[260].text == "<0xFF>"
    assert parsed.pieces[-1] == tokenizer_export.FlatBpePiece("hi", 7.5, 1)
    assert parsed.normalization_rules == ()

    assert metadata.source_model_type == "BPE"
    assert metadata.source_sha256 == hashlib.sha256(model_bytes).hexdigest()
    assert metadata.asset_sha256 == hashlib.sha256(asset_bytes).hexdigest()
    assert metadata.asset_size == len(asset_bytes)
    assert metadata.piece_count == expected_piece_count
    assert metadata.piece_type_counts == {
        "normal": 3,
        "unknown": 1,
        "control": 3,
        "user_defined": 1,
        "unused": 0,
        "byte": 256,
    }
    assert metadata.normalization_name == "identity"
    assert metadata.normalization_rule_count == 0

    repeated_bytes, repeated_metadata = tokenizer_export.convert_paligemma_tokenizer_model(
        model_bytes
    )
    assert repeated_bytes == asset_bytes
    assert repeated_metadata == metadata


def test_file_export_and_cli_produce_hash_bound_metadata(tmp_path, capsys) -> None:
    model_path = tmp_path / "paligemma_tokenizer.model"
    model_path.write_bytes(_serialize(_synthetic_bpe_model()))
    asset_path = tmp_path / "assets" / "paligemma_tokenizer.trtmcbpe"

    metadata = tokenizer_export.export_paligemma_bpe_model(model_path, asset_path)
    assert asset_path.is_file()
    assert metadata.asset_sha256 == hashlib.sha256(asset_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        tokenizer_export.export_paligemma_bpe_model(model_path, asset_path)

    metadata_path = tmp_path / "tokenizer_export.json"
    assert (
        tokenizer_export.main(
            [
                "--input",
                str(model_path),
                "--output",
                str(asset_path),
                "--metadata-output",
                str(metadata_path),
                "--force",
            ]
        )
        == 0
    )
    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert stdout_payload["asset_sha256"] == metadata.asset_sha256
    assert stdout_payload["source_model_type"] == "BPE"
    assert Path(stdout_payload["output"]) == asset_path.resolve()


def test_non_bpe_model_is_rejected() -> None:
    model = _synthetic_bpe_model()
    pb = tokenizer_export._load_sentencepiece_proto_module()
    model.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    with pytest.raises(tokenizer_export.TokenizerExportError, match="not BPE"):
        tokenizer_export.convert_paligemma_tokenizer_model(_serialize(model))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda model: setattr(model.normalizer_spec, "name", "nmt_nfkc"), "identity normalizer"),
        (
            lambda model: setattr(model.normalizer_spec, "precompiled_charsmap", b"rules"),
            "precompiled_charsmap",
        ),
        (
            lambda model: setattr(model.normalizer_spec, "normalization_rule_tsv", "rules.tsv"),
            "normalization_rule_tsv",
        ),
        (
            lambda model: setattr(model.denormalizer_spec, "name", "identity"),
            "denormalizer_spec",
        ),
    ],
)
def test_unsupported_normalizer_semantics_are_never_dropped(mutation, message) -> None:
    model = _synthetic_bpe_model()
    mutation(model)
    with pytest.raises(tokenizer_export.TokenizerExportError, match=message):
        tokenizer_export.convert_paligemma_tokenizer_model(_serialize(model))


def test_incomplete_byte_fallback_table_is_rejected() -> None:
    model = _synthetic_bpe_model()
    del model.pieces[5]
    with pytest.raises(tokenizer_export.TokenizerExportError, match="byte fallback table"):
        tokenizer_export.convert_paligemma_tokenizer_model(_serialize(model))


def test_corrupt_source_and_flat_assets_are_rejected() -> None:
    model = _synthetic_bpe_model()
    with pytest.raises(tokenizer_export.TokenizerExportError, match="valid ModelProto"):
        tokenizer_export.convert_paligemma_tokenizer_model(b"\xff\xff\xff")

    asset, _ = tokenizer_export.convert_paligemma_tokenizer_model(_serialize(model))
    with pytest.raises(tokenizer_export.TokenizerExportError, match="magic"):
        tokenizer_export.parse_trtmcbpe_asset(b"X" + asset[1:])
    with pytest.raises(tokenizer_export.TokenizerExportError, match="truncated"):
        tokenizer_export.parse_trtmcbpe_asset(asset[:-1])
    with pytest.raises(tokenizer_export.TokenizerExportError, match="trailing"):
        tokenizer_export.parse_trtmcbpe_asset(asset + b"x")
    excessive_count = bytearray(asset)
    struct.pack_into("<I", excessive_count, 32, 1_000_001)
    with pytest.raises(tokenizer_export.TokenizerExportError, match="record count"):
        tokenizer_export.parse_trtmcbpe_asset(bytes(excessive_count))


def test_missing_build_dependencies_have_actionable_error(monkeypatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "sentencepiece" or name.startswith("sentencepiece."):
            raise ModuleNotFoundError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(
        tokenizer_export.TokenizerExportDependencyError,
        match="install both 'sentencepiece' and 'protobuf'",
    ):
        tokenizer_export._load_sentencepiece_proto_module()
