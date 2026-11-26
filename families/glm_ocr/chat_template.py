# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint's chat template must render what the native prompt renderer renders.

``render_native`` restates the rules of ``runtime/prompt.cpp`` (the host-parity tests compare
the two). ``check_chat_template`` renders the checkpoint's template as Transformers renders
chat templates on generated conversations that exercise every rule and compares each prompt.
Messages use the Transformers form: a role and a list of ``{"type": "text", "text": ...}`` or
``{"type": "image"}`` parts.
"""

from __future__ import annotations

import random
from pathlib import Path

_IMAGE = "<|begin_of_image|><|image|><|end_of_image|>"

# Text pieces the generated conversations draw from: plain words, leading and trailing white
# space, line breaks, tabs, non-ASCII text, Jinja-like and special-token-like text.
_PIECES = (
    "Text Recognition:", "Formula Recognition:", "Table Recognition:", "A", "ab", "  padded  ",
    "\n", "two\nlines", "\ttab", " ", "x\r\ny", "Ünïcödé", "数学公式", "{{ x }}", "{% if %}",
    "<|user|>", "</tool_call>", "/nothink", "", "　", "trailing \n\n",
)


def render_native(messages: list[dict]) -> str:
    """The prompt ``runtime/prompt.cpp`` renders for ``messages`` (an assistant reply follows)."""
    prompt = "[gMASK]<sop>"
    for message in messages:
        parts = message["content"]
        if message["role"] == "system":
            prompt += "<|system|>\n" + "".join(part["text"] for part in parts)
        elif message["role"] == "user":
            prompt += "<|user|>\n" + "".join(
                _IMAGE if part["type"] == "image" else part["text"] for part in parts)
        else:
            content = "".join(part["text"] for part in parts).strip()
            prompt += "<|assistant|>\n<think></think>" + ("\n" + content if content else "")
    return prompt + "<|assistant|>\n"


def probe_conversations(count: int = 200, seed: int = 191) -> list[list[dict]]:
    """Conversations the native renderer accepts, covering every rendering rule.

    Each one ends with a user turn; system turns appear first and occasionally later, users
    may follow users, user turns mix text and image parts in any order, and assistant replies
    carry white space that the template strips.
    """
    rng = random.Random(seed)

    def text() -> dict:
        return {"type": "text", "text": "".join(rng.choice(_PIECES) for _ in range(rng.randint(1, 3)))}

    conversations: list[list[dict]] = []
    for _ in range(count):
        messages: list[dict] = []
        if rng.random() < 0.5:
            messages.append({"role": "system", "content": [text() for _ in range(rng.randint(1, 2))]})
        for turn in range(rng.randint(1, 3)):
            if turn and rng.random() < 0.15:
                messages.append({"role": "system", "content": [text()]})
            user = [text() if rng.random() < 0.5 else {"type": "image"} for _ in range(rng.randint(1, 4))]
            messages.append({"role": "user", "content": user})
            if rng.random() < 0.7:
                assistant = [text() for _ in range(rng.randint(1, 2))]
                messages.append({"role": "assistant", "content": assistant})
        if messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": [{"type": "image"}, text()]})
        conversations.append(messages)
    return conversations


def check_chat_template(path: Path) -> None:
    """Refuse a checkpoint whose chat template renders differently from the native renderer.

    The template is rendered as Transformers renders chat templates (sandboxed Jinja with
    ``trim_blocks``, ``lstrip_blocks`` and loop controls).
    """
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    if not path.is_file():
        raise ValueError("glm_ocr: checkpoint has no chat_template.jinja")
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    try:
        template = environment.from_string(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - any template failure means an unsupported template
        raise ValueError(f"glm_ocr cannot render this checkpoint's chat template: {error}") from error
    for messages in probe_conversations():
        expected = render_native(messages)
        try:
            actual = template.render(messages=messages, add_generation_prompt=True)
        except Exception as error:  # noqa: BLE001
            raise ValueError(f"glm_ocr cannot render this checkpoint's chat template: {error}") from error
        if actual != expected:
            raise ValueError(f"glm_ocr: chat template renders {actual!r} where the native renderer "
                             f"produces {expected!r} (messages {messages!r})")
