# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from tensorrt_model_connect.engine_builder import _ensure_tokenizer_json
from tensorrt_model_connect.families.internlm.tokenizer_json import (
    ensure_tokenizer_json,
)


def _write_valid_bpe_tokenizer(path):
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {
                    "type": "BPE",
                    "vocab": {"a": 0},
                    "merges": [],
                }
            }
        ),
        encoding="utf-8",
    )


def test_ensure_tokenizer_json_defaults_to_untrusted_fast_tokenizer(
    monkeypatch,
    tmp_path,
):
    calls = {}

    class FakeTokenizer:
        is_fast = True

        def save_pretrained(self, path):
            _write_valid_bpe_tokenizer(Path(path))

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.update(path=path, **kwargs)
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    assert ensure_tokenizer_json(tmp_path)
    assert calls == {
        "path": str(tmp_path),
        "trust_remote_code": False,
        "use_fast": True,
    }


def test_ensure_tokenizer_json_forwards_explicit_remote_code_trust(
    monkeypatch,
    tmp_path,
):
    calls = {}

    class FakeTokenizer:
        is_fast = True

        def save_pretrained(self, path):
            _write_valid_bpe_tokenizer(Path(path))

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.update(path=path, **kwargs)
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    assert ensure_tokenizer_json(tmp_path, trust_remote_code=True)
    assert calls == {
        "path": str(tmp_path),
        "trust_remote_code": True,
        "use_fast": True,
    }


def test_ensure_tokenizer_json_overwrites_malformed_existing_file(
    monkeypatch,
    tmp_path,
):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    calls = []

    class FakeTokenizer:
        is_fast = True

        def save_pretrained(self, path):
            calls.append(("save", path))
            _write_valid_bpe_tokenizer(Path(path))

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert not tokenizer_path.exists()
            calls.append(("load", path, kwargs))
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    assert ensure_tokenizer_json(tmp_path)
    assert calls[0] == (
        "load",
        str(tmp_path),
        {"trust_remote_code": False, "use_fast": True},
    )
    assert calls[1][0] == "save"
    assert Path(calls[1][1]).parent.parent == tmp_path
    assert json.loads(tokenizer_path.read_text())["model"]["type"] == "BPE"


def test_ensure_tokenizer_json_shortcuts_only_native_compatible_existing_file(
    monkeypatch,
    tmp_path,
):
    _write_valid_bpe_tokenizer(tmp_path)

    class UnexpectedAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("valid tokenizer.json must not be regenerated")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=UnexpectedAutoTokenizer),
    )

    assert ensure_tokenizer_json(tmp_path)


def test_ensure_tokenizer_json_rejects_malformed_regenerated_file(
    monkeypatch,
    tmp_path,
):
    tokenizer_path = tmp_path / "tokenizer.json"
    original = b'{"malformed":"original bytes"}'
    tokenizer_path.write_bytes(original)

    class FakeTokenizer:
        is_fast = True

        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    assert not ensure_tokenizer_json(tmp_path)
    assert tokenizer_path.read_bytes() == original


@pytest.mark.parametrize(
    "symlink_kind",
    ("relative", "absolute-to-temporary-file"),
)
def test_ensure_tokenizer_json_rejects_generated_symlink_and_restores_original(
    monkeypatch,
    tmp_path,
    symlink_kind,
):
    tokenizer_path = tmp_path / "tokenizer.json"
    original = b'{"malformed":"must survive generated symlink"}'
    tokenizer_path.write_bytes(original)

    class SymlinkTokenizer:
        is_fast = True

        @staticmethod
        def save_pretrained(path):
            generated_dir = Path(path)
            real_path = generated_dir / "real-tokenizer.json"
            real_path.write_text(
                json.dumps({
                    "model": {
                        "type": "BPE",
                        "vocab": {"a": 0},
                        "merges": [],
                    },
                }),
                encoding="utf-8",
            )
            link_target = (
                real_path.name
                if symlink_kind == "relative"
                else real_path.resolve()
            )
            (generated_dir / "tokenizer.json").symlink_to(link_target)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert not tokenizer_path.exists()
            assert not tokenizer_path.is_symlink()
            return SymlinkTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    assert not ensure_tokenizer_json(tmp_path)
    assert tokenizer_path.read_bytes() == original
    assert not tokenizer_path.is_symlink()
    assert not (tmp_path / "real-tokenizer.json").exists()
    assert not list(tmp_path.glob(".internlm-tokenizer-repair-*"))


def test_ensure_tokenizer_json_preserves_malformed_file_when_loader_fails(
    monkeypatch,
    tmp_path,
):
    tokenizer_path = tmp_path / "tokenizer.json"
    original = b'{"malformed":"must survive"}'
    tokenizer_path.write_bytes(original)

    class FailingAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert not tokenizer_path.exists()
            raise ValueError("slow tokenizer assets are incomplete")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FailingAutoTokenizer),
    )

    assert not ensure_tokenizer_json(tmp_path)
    assert tokenizer_path.read_bytes() == original


def test_required_repair_preserves_original_across_both_failed_generators(
    monkeypatch,
    tmp_path,
):
    tokenizer_path = tmp_path / "tokenizer.json"
    original = b'{"malformed":"original required-tokenizer bytes"}'
    tokenizer_path.write_bytes(original)
    use_fast_values = []

    class InvalidTokenizer:
        is_fast = True

        @staticmethod
        def save_pretrained(path):
            assert Path(path) != tmp_path
            (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert Path(path) == tmp_path
            assert not tokenizer_path.exists()
            use_fast_values.append(kwargs["use_fast"])
            return InvalidTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    with pytest.raises(RuntimeError, match="refusing to write a bundle"):
        _ensure_tokenizer_json(
            tmp_path,
            plugin=types.SimpleNamespace(
                ensure_tokenizer_json=ensure_tokenizer_json,
            ),
        )

    assert use_fast_values == [False, True]
    assert tokenizer_path.read_bytes() == original


@pytest.mark.parametrize(
    "invalid_trust",
    ("false", 1, None),
    ids=("string-false", "integer-one", "none"),
)
def test_ensure_tokenizer_json_rejects_non_boolean_remote_code_trust(
    tmp_path,
    invalid_trust,
):
    with pytest.raises(TypeError, match="trust_remote_code must be a bool"):
        ensure_tokenizer_json(
            tmp_path,
            trust_remote_code=invalid_trust,
        )


def test_required_internlm_tokenizer_generation_fails_closed_by_default(
    monkeypatch,
    tmp_path,
):
    trust_values = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(tmp_path)
            trust_values.append(kwargs["trust_remote_code"])
            raise ValueError("custom tokenizer code is not trusted")

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    with pytest.raises(RuntimeError) as exc_info:
        _ensure_tokenizer_json(
            tmp_path,
            plugin=types.SimpleNamespace(
                ensure_tokenizer_json=ensure_tokenizer_json,
            ),
        )

    message = str(exc_info.value)
    assert "refusing to write a bundle" in message
    assert "Review any repository-provided tokenizer code" in message
    assert "trust_remote_code=True" in message
    assert "build(model_revision=...)" in message
    assert "immutable local snapshot" in message
    assert trust_values == [False, False]
    assert not (tmp_path / "tokenizer.json").exists()
