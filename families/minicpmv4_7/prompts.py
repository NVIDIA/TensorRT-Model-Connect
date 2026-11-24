# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider chat-template recipes for the native message formatter, verified against the checkpoint template.

The runtime formats conversations without a template engine. At build time the checkpoint's own template
is rendered on sentinel messages to extract the fixed strings around each role, and the extracted recipe is
then checked against the template on representative conversations, including structured image content.
"""

from __future__ import annotations

IMAGE_PLACEHOLDER = "<|image_pad|>"


def _split(text: str, marker: str) -> list[str]:
    if text.count(marker) != 1:
        raise ValueError("MiniCPM-V 4.7 chat template does not preserve content markers")
    return text.split(marker)


def _without_prefix(text: str, prefix: str) -> str:
    if not text.startswith(prefix):
        raise ValueError("MiniCPM-V 4.7 chat template has context-dependent formatting")
    return text[len(prefix):]


def _without_suffix(text: str, suffix: str) -> str:
    if not text.endswith(suffix):
        raise ValueError("MiniCPM-V 4.7 chat template has context-dependent formatting")
    return text[:-len(suffix)] if suffix else text


def content_text(content) -> str:
    """Template ``render_content``: string content as is, parts joined by newlines, images as placeholders."""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        parts.append(IMAGE_PLACEHOLDER if item.get("type") == "image" else item["text"])
    return "\n".join(parts)


def render_recipe(recipe: dict, messages: list[dict], *, generation: bool = True) -> str:
    """Builder-side oracle for the runtime formatter (``runtime/prompt.cpp``)."""
    contents = [content_text(message["content"]).strip() for message in messages]
    last_user = -1
    for i, message in enumerate(messages):
        text = contents[i]
        if message["role"] == "user" and not (text.startswith("<tool_response>") and text.endswith("</tool_response>")):
            last_user = i
    if last_user < 0:
        raise ValueError("MiniCPM-V 4.7 requires a user query")
    if messages[0]["role"] != "system":
        output = recipe["default_prefix"]
    elif contents[0]:
        output = recipe["system"]["prefix"] + contents[0] + recipe["system"]["suffix"]
    else:
        output = recipe["empty_system_prefix"]
    for i, message in enumerate(messages):
        role, text = message["role"], contents[i]
        if role == "system":
            if i:
                raise ValueError("System message must be first")
            continue
        if role == "user":
            output += recipe["user"]["prefix"] + text + recipe["user"]["suffix"]
            continue
        if role != "assistant":
            raise ValueError("Unsupported MiniCPM-V 4.7 role")
        reasoning = message.get("reasoning_content", "").strip()
        if "reasoning_content" not in message and "</think>" in text:
            reasoning = text.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip()
            text = text.split("</think>")[-1].lstrip("\n")
        form = recipe["assistant_recent" if i > last_user else "assistant_history"]
        output += form["prefix"]
        if form["include_reasoning"] and reasoning:
            output += form["reasoning_prefix"] + reasoning + form["reasoning_suffix"]
        else:
            output += form["empty_reasoning"]
        output += text + form["suffix"]
    return output + (recipe["generation"] if generation else "")


def provider_recipes(tokenizer) -> dict:
    """Extract and verify one recipe per ``enable_thinking`` value from the checkpoint template."""
    markers = [f"MINICPMV47_PROMPT_SENTINEL_{i}_END" for i in range(4)]
    a, b, c, reasoning = markers

    def message(role, content, **kwargs):
        return {"role": role, "content": content, **kwargs}

    result = {}
    for thinking in (False, True):
        def render(messages, generation=False):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation,
                                                 enable_thinking=thinking)

        single = render([message("user", a)])
        full_prefix, user_suffix = _split(single, a)
        doubled = render([message("user", a), message("user", b)])
        middle = _split(_split(doubled, a)[1], b)[0]
        user_prefix = _without_prefix(middle, user_suffix)
        default = _without_suffix(full_prefix, user_prefix)
        system = _without_suffix(render([message("system", c), message("user", a)]), user_prefix + a + user_suffix)
        system_prefix, system_suffix = _split(system, c)
        recipe = {
            "default_prefix": default,
            "user": {"prefix": user_prefix, "suffix": user_suffix},
            "system": {"prefix": system_prefix, "suffix": system_suffix},
            "empty_system_prefix": _without_suffix(render([message("system", ""), message("user", a)]),
                                                   user_prefix + a + user_suffix),
            "generation": _without_prefix(render([message("user", a)], True), single),
        }
        for historical in (False, True):
            forms = []
            for with_reasoning in (True, False):
                kwargs = {"reasoning_content": reasoning} if with_reasoning else {}
                messages = [message("user", a), message("assistant", b, **kwargs)]
                if historical:
                    messages.append(message("user", c))
                assistant = _without_prefix(render(messages), single)
                if historical:
                    assistant = _without_suffix(assistant, user_prefix + c + user_suffix)
                forms.append(_split(assistant, b))
            (with_prefix, suffix), (without_prefix, without_suffix) = forms
            if suffix != without_suffix:
                raise ValueError("MiniCPM-V 4.7 assistant suffix depends on reasoning")
            includes = reasoning in with_prefix
            start, between = _split(with_prefix, reasoning) if includes else (with_prefix, "")
            common = 0
            while common < min(len(start), len(without_prefix)) and start[common] == without_prefix[common]:
                common += 1
            recipe["assistant_history" if historical else "assistant_recent"] = {
                "prefix": start[:common], "include_reasoning": includes,
                "reasoning_prefix": start[common:], "reasoning_suffix": between,
                "empty_reasoning": without_prefix[common:], "suffix": suffix}
        image_text = [{"type": "image"}, {"type": "text", "text": "Describe this."}]
        cases = [
            [message("user", " \nHello, 世界! ")],
            [message("system", ""), message("user", "")],
            [message("system", " Follow instructions. "), message("user", a)],
            [message("user", a), message("assistant", b)],
            [message("user", a), message("assistant", b), message("user", c)],
            [message("user", a), message("assistant", f"<think>{reasoning}</think>{b}")],
            [message("user", a), message("assistant", f"<think>\n{reasoning}\n</think>\n\n{b}"), message("user", c)],
            [message("user", a), message("assistant", b, reasoning_content=reasoning), message("user", c)],
            [message("user", a), message("assistant", b, reasoning_content=reasoning)],
            [message("user", a), message("user", "<tool_response>value</tool_response>"), message("assistant", b)],
            [message("user", image_text)],
            [message("user", [{"type": "image"}, {"type": "image"}, {"type": "text", "text": " Compare. "}])],
            [message("system", "Be brief."), message("user", image_text), message("assistant", b),
             message("user", [{"type": "text", "text": "And now?"}, {"type": "image"}])],
        ]
        for messages in cases:
            for generation in (False, True):
                if render_recipe(recipe, messages, generation=generation) != render(messages, generation):
                    raise ValueError("MiniCPM-V 4.7 prompt recipe does not match the checkpoint template")
        result[str(thinking).lower()] = recipe
    return result
