# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types

import pytest

from tensorrt_model_connect.engine_builder import _ensure_tokenizer_json
from tensorrt_model_connect.families.internlm.tokenizer_json import (
    ensure_tokenizer_json,
)


def test_ensure_tokenizer_json_defaults_to_untrusted_fast_tokenizer(
    monkeypatch,
    tmp_path,
):
    calls = {}

    class FakeTokenizer:
        is_fast = True

        def save_pretrained(self, path):
            (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

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
            (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

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
