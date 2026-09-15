# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the checkpoint's chat template available to the native runtime."""

import json
from pathlib import Path

import pytest


def test_tokenizer_config_template_preserves_full_jinja(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt")
    from families.llama.model import _chat_template

    template = "{%- for message in messages %}{{ message.content }}多语言{% endfor %}"
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template}), encoding="utf-8"
    )
    assert _chat_template(tmp_path) == template.encode("utf-8")


def test_explicit_template_keeps_checkpoint_precedence(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt")
    from families.llama.model import _chat_template

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": ["unsupported inline template"]}), encoding="utf-8"
    )
    explicit = b"{{ messages[0].content }}\n"
    (tmp_path / "chat_template.jinja").write_bytes(explicit)
    assert _chat_template(tmp_path) == explicit


def test_tokenizer_config_requires_an_object(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt")
    from families.llama.model import _chat_template

    (tmp_path / "tokenizer_config.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain one JSON object"):
        _chat_template(tmp_path)


@pytest.mark.parametrize("template", [[], {"default": "named template"}, False, 7])
def test_unsupported_inline_template_cannot_fall_back_to_raw_prompt(
    tmp_path: Path, template
) -> None:
    pytest.importorskip("tensorrt")
    from families.llama.model import _chat_template

    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": template}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="chat_template must be a string"):
        _chat_template(tmp_path)


@pytest.mark.parametrize("document", [{}, {"chat_template": None}])
def test_absent_or_null_template_keeps_plain_checkpoint_support(
    tmp_path: Path, document: dict
) -> None:
    pytest.importorskip("tensorrt")
    from families.llama.model import _chat_template

    (tmp_path / "tokenizer_config.json").write_text(json.dumps(document), encoding="utf-8")
    assert _chat_template(tmp_path) is None
