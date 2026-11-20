# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The build refuses a chat template that renders differently from the native renderer."""

from __future__ import annotations

import pytest

from families.glm_ocr.chat_template import check_chat_template, probe_conversations, render_native

_SINGLE = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Text Recognition:"}]}]
_TURNS = [
    {"role": "system", "content": [{"type": "text", "text": "You are OCR."}]},
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "A"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "  spaced  \n"}]},
    {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}, {"type": "image"}]},
]
# The prompts tests/cpp/test_runtime.cpp expects from the runtime for the same conversations.
_SINGLE_PROMPT = "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>Text Recognition:<|assistant|>\n"
_TURNS_PROMPT = ("[gMASK]<sop><|system|>\nYou are OCR.<|user|>\n<|begin_of_image|><|image|><|end_of_image|>A"
                 "<|assistant|>\n<think></think>\nspaced<|user|>\nab<|begin_of_image|><|image|><|end_of_image|>"
                 "<|assistant|>\n")

# A template written from the native rules (not the checkpoint's).
_FAITHFUL = (
    "[gMASK]<sop>{% for m in messages %}"
    "{% if m.role == 'system' %}<|system|>\n{% for p in m.content %}{{ p.text }}{% endfor %}"
    "{% elif m.role == 'user' %}<|user|>\n{% for p in m.content %}"
    "{% if p.type == 'image' %}<|begin_of_image|><|image|><|end_of_image|>{% else %}{{ p.text }}{% endif %}"
    "{% endfor %}"
    "{% elif m.role == 'assistant' %}<|assistant|>\n<think></think>"
    "{% set c = m.content | map(attribute='text') | join('') %}{% if c.strip() %}{{ '\\n' + c.strip() }}{% endif %}"
    "{% endif %}{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def _write(tmp_path, text: str):
    path = tmp_path / "chat_template.jinja"
    path.write_text(text, encoding="utf-8")
    return path


def test_the_native_renderer_matches_the_runtime() -> None:
    assert render_native(_SINGLE) == _SINGLE_PROMPT
    assert render_native(_TURNS) == _TURNS_PROMPT


def test_the_probe_conversations_cover_every_rule() -> None:
    conversations = probe_conversations()
    roles = [[m["role"] for m in c] for c in conversations]
    assert all(r[-1] == "user" for r in roles)
    assert any(r[0] == "system" for r in roles) and any("system" in r[1:] for r in roles)
    assert any("assistant" in r for r in roles) and any(("user", "user") in zip(r, r[1:]) for r in roles)
    parts = [p for c in conversations for m in c if m["role"] == "user" for p in m["content"]]
    assert any(p["type"] == "image" for p in parts) and any(p["type"] == "text" for p in parts)


def test_a_template_with_the_native_rules_is_accepted(tmp_path) -> None:
    pytest.importorskip("jinja2")
    check_chat_template(_write(tmp_path, _FAITHFUL))


def test_a_template_that_only_matches_fixed_prompts_is_refused(tmp_path) -> None:
    pytest.importorskip("jinja2")
    fixed = "{% if messages | length == 1 %}" + _SINGLE_PROMPT + "{% else %}" + _TURNS_PROMPT + "{% endif %}"
    with pytest.raises(ValueError, match="chat template renders"):
        check_chat_template(_write(tmp_path, fixed))


@pytest.mark.parametrize(("rule", "changed"), [
    ("assistant think block", ("<think></think>", "")),
    ("assistant strip", ("c.strip()", "c")),
    ("system prefix", ("<|system|>\n", "<|system|>")),
    ("generation prompt", ("{% if add_generation_prompt %}<|assistant|>\n", "{% if add_generation_prompt %}<|assistant|>")),
])
def test_a_template_that_changes_one_rule_is_refused(tmp_path, rule, changed) -> None:
    pytest.importorskip("jinja2")
    template = _FAITHFUL.replace(*changed)
    assert template != _FAITHFUL, rule
    with pytest.raises(ValueError, match="chat template renders"):
        check_chat_template(_write(tmp_path, template))


def test_a_template_that_cannot_render_is_refused(tmp_path) -> None:
    pytest.importorskip("jinja2")
    with pytest.raises(ValueError, match="cannot render"):
        check_chat_template(_write(tmp_path, "{{ raise_exception('unsupported') }}"))


def test_a_missing_template_is_refused(tmp_path) -> None:
    pytest.importorskip("jinja2")
    with pytest.raises(ValueError, match="no chat_template.jinja"):
        check_chat_template(tmp_path / "chat_template.jinja")
