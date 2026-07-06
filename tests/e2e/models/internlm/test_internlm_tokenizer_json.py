# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types

from tensorrt_model_connect.families.internlm.tokenizer_json import (
    ensure_tokenizer_json,
)


def test_ensure_tokenizer_json_uses_trusted_fast_tokenizer(monkeypatch, tmp_path):
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
        "trust_remote_code": True,
        "use_fast": True,
    }
