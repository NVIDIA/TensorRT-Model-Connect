# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ConvBERT tokenizer bundle regression coverage."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("tensorrt", reason="TensorRT is required for family builder tests")

from families.convbert import model


def test_runtime_contract_materializes_missing_fast_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Backend:
        def save(self, path: str) -> None:
            Path(path).write_text('{"model":{"type":"WordPiece"}}', encoding="utf-8")

    class Tokenizer:
        backend_tokenizer = Backend()

        @staticmethod
        def encode(_text: str, add_special_tokens: bool = True) -> list[int]:
            return [101, 42, 102] if add_special_tokens else [42]

    auto = SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=auto))

    contract = model._tokenizer_runtime_contract(tmp_path)

    assert (tmp_path / "tokenizer.json").is_file()
    assert contract["tokenizer_prefix_ids"] == [101]
    assert contract["tokenizer_suffix_ids"] == [102]
