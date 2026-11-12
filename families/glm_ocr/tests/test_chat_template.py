# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build refuses a chat template that renders differently from the native renderer."""

from __future__ import annotations

import pytest

from families.glm_ocr.chat_template import TEMPLATE_CASES, check_chat_template

pytest.importorskip("jinja2")


def _template(first: str, other: str) -> str:
    """A template that renders ``first`` for a one-message conversation and ``other`` otherwise."""
    return "{% if messages | length == 1 %}" + first + "{% else %}" + other + "{% endif %}"


def test_a_template_with_the_native_rendering_is_accepted(tmp_path) -> None:
    path = tmp_path / "chat_template.jinja"
    path.write_text(_template(TEMPLATE_CASES[0][1], TEMPLATE_CASES[1][1]), encoding="utf-8")
    check_chat_template(path)


@pytest.mark.parametrize("index", [0, 1])
def test_a_template_that_renders_differently_is_refused(tmp_path, index) -> None:
    rendered = [TEMPLATE_CASES[0][1], TEMPLATE_CASES[1][1]]
    rendered[index] = rendered[index].replace("<|assistant|>\n", "<|assistant|>\n<think></think>\n", 1)
    path = tmp_path / "chat_template.jinja"
    path.write_text(_template(*rendered), encoding="utf-8")
    with pytest.raises(ValueError, match="chat template renders"):
        check_chat_template(path)


def test_a_template_that_cannot_render_is_refused(tmp_path) -> None:
    path = tmp_path / "chat_template.jinja"
    path.write_text("{{ raise_exception('unsupported') }}", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot render"):
        check_chat_template(path)


def test_a_missing_template_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="no chat_template.jinja"):
        check_chat_template(tmp_path / "chat_template.jinja")
