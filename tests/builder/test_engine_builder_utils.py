# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thin model resolver and its shared leaf primitives."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from tensorrt_model_connect import bundle_writer, trt_compat
from tensorrt_model_connect.engine_builder import (
    _is_hf_model_dir,
    _resolve_model,
    build,
)
from tensorrt_model_connect.hf_snapshot import (
    GENERIC_HF_ALLOW_PATTERNS,
    hf_snapshot_allow_patterns,
)
from tensorrt_model_connect.tokenizer_conversion import (
    detect_tokenizer_add_special_tokens,
    detect_tokenizer_special_frame,
    ensure_tokenizer_json,
    prepare_tokenizer_special_frame,
)


@pytest.mark.parametrize("filename", ["config.json", "model_index.json"])
def test_hf_model_directory_entrypoints(tmp_path, filename: str) -> None:
    (tmp_path / filename).write_text("{}", encoding="utf-8")
    assert _is_hf_model_dir(tmp_path)
    assert _resolve_model(str(tmp_path)) == str(tmp_path)


def test_non_model_directory_is_rejected_by_probe(tmp_path) -> None:
    (tmp_path / "other.txt").write_text("data", encoding="utf-8")
    assert not _is_hf_model_dir(tmp_path)


def test_tokenizer_config_fallback_detects_special_tokens(tmp_path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"add_bos_token": True}), encoding="utf-8"
    )
    assert detect_tokenizer_add_special_tokens(tmp_path)


def test_tokenizer_runtime_detection_beats_stale_config(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"add_bos_token": False}), encoding="utf-8"
    )

    class Tokenizer:
        def encode(self, _text, add_special_tokens=True):
            return [1, 2] if add_special_tokens else [2]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: Tokenizer()
            )
        ),
    )
    assert detect_tokenizer_add_special_tokens(tmp_path)
    assert detect_tokenizer_special_frame(tmp_path) == ([1], [])


def test_prepare_frame_preserves_source_contract_during_conversion(
    tmp_path,
    monkeypatch,
) -> None:
    state = {"converted": False}

    class SourceTokenizer:
        def encode(self, _text, add_special_tokens=True):
            return [10]

        def save_pretrained(self, path):
            state["converted"] = True
            (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class ConvertedTokenizer:
        def encode(self, _text, add_special_tokens=True):
            return [2, 10] if add_special_tokens else [10]

    def load(*_args, **_kwargs):
        return ConvertedTokenizer() if state["converted"] else SourceTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=load)
        ),
    )
    assert prepare_tokenizer_special_frame(tmp_path) == ([], [])
    assert state["converted"]


def test_ensure_tokenizer_rebuilds_undersized_wordpiece(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "config.json").write_text('{"vocab_size": 2}', encoding="utf-8")
    (tmp_path / "vocab.txt").write_text("[UNK]\nhello\n", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text(
        '{"model":{"type":"WordPiece","vocab":{"[UNK]":0}}}',
        encoding="utf-8",
    )

    class Tokenizer:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text(
                '{"model":{"type":"WordPiece","vocab":{"[UNK]":0,"hello":1}}}',
                encoding="utf-8",
            )

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: Tokenizer()
            )
        ),
    )
    ensure_tokenizer_json(tmp_path)
    assert "hello" in (tmp_path / "tokenizer.json").read_text(encoding="utf-8")


def test_family_tokenizer_converter_is_an_explicit_callable(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    RuntimeError("no fast tokenizer")
                )
            )
        ),
    )
    calls = []

    def convert(path, *, previous_error=None):
        calls.append((Path(path), previous_error))
        (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")
        return True

    ensure_tokenizer_json(tmp_path, family_ensure=convert)
    assert calls[0][0] == tmp_path
    assert "fast tokenizer conversion failed" in calls[0][1]


@pytest.mark.parametrize(
    ("version", "abi"),
    [("10.15.0.6", "10.15"), ("11.0", "11.0"), ("unknown", "")],
)
def test_bundle_tensorrt_abi(version: str, abi: str) -> None:
    assert bundle_writer.tensorrt_abi(version) == abi


def test_bundle_tensorrt_version_uses_selected_backend(monkeypatch) -> None:
    monkeypatch.setattr(trt_compat, "tensorrt_version", lambda: "11.1.0")
    assert bundle_writer.tensorrt_version() == "11.1.0"


def test_snapshot_allowlist_combines_generic_and_family_files() -> None:
    combined = set(hf_snapshot_allow_patterns())
    assert set(GENERIC_HF_ALLOW_PATTERNS) <= combined
    for required in ("config.json", "tokenizer.json", "*.nemo"):
        assert required in combined


def test_unknown_model_family_fails_closed(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "totally_unsupported_model_xyz",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "vocab_size": 32,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No model family owns"):
        build(str(tmp_path), str(tmp_path / "out.bundle"))
