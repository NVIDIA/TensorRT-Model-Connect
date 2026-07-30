# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
import types

from tensorrt_model_connect.families.starcoder2.tokenizer_json import (
    tokenizer_json_bundle_override,
)


def test_tokenizer_json_bundle_override_uses_hf_runtime_backend(
    monkeypatch,
    tmp_path,
):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(json.dumps({
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Digits", "individual_digits": True},
                {"type": "ByteLevel"},
            ],
        },
    }))
    canonical = {
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": False,
        },
    }
    calls = {}

    class FakeBackendTokenizer:
        def to_str(self):
            return json.dumps(canonical)

    class FakeTokenizer:
        backend_tokenizer = FakeBackendTokenizer()

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

    payload = tokenizer_json_bundle_override(tmp_path)
    assert json.loads(payload) == canonical
    assert json.loads(tokenizer_path.read_text()) != canonical
    assert calls == {
        "path": str(tmp_path),
        "local_files_only": True,
        "use_fast": True,
    }
