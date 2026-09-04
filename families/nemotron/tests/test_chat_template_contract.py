# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from families.nemotron.model import _BUNDLE_FILES, _chat_template


def test_build_materializes_one_canonical_chat_template(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "{{ messages }}"}),
        encoding="utf-8",
    )

    assert _chat_template(tmp_path) == b"{{ messages }}"
    assert "tokenizer_config.json" not in _BUNDLE_FILES
    assert "chat_template.jinja" not in _BUNDLE_FILES


def test_build_rejects_missing_chat_template(tmp_path: Path) -> None:
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="requires chat_template"):
        _chat_template(tmp_path)
