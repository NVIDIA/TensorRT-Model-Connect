# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned native media bundle metadata and verified provider prompt recipes."""

from __future__ import annotations


def _split(text, marker):
    if text.count(marker) != 1:
        raise ValueError(
            "Flash-Next native media provider template does not preserve content markers"
        )
    return text.split(marker)


def _without_prefix(text, prefix):
    if not text.startswith(prefix):
        raise ValueError(
            "Flash-Next native media provider template has context-dependent formatting"
        )
    return text[len(prefix) :]


def _without_suffix(text, suffix):
    if not text.endswith(suffix):
        raise ValueError(
            "Flash-Next native media provider template has context-dependent formatting"
        )
    return text[: -len(suffix)] if suffix else text


def render_recipe(recipe, messages, *, generation=True):
    """Builder-side oracle for the small admitted C++ message formatter."""
    last_user = -1
    for i, message in enumerate(messages):
        content = message["content"].strip()
        if message["role"] == "user" and not (
            content.startswith("<tool_response>") and content.endswith("</tool_response>")
        ):
            last_user = i
    if last_user < 0:
        raise ValueError("Flash-Next native media requires a user query")
    system = messages[0]["content"].strip() if messages[0]["role"] == "system" else None
    output = (
        recipe["default_prefix"]
        if system is None
        else (
            recipe["system"]["prefix"] + system + recipe["system"]["suffix"]
            if system
            else recipe["empty_system_prefix"]
        )
    )
    for i, message in enumerate(messages):
        role, content = message["role"], message["content"].strip()
        if role == "system":
            if i:
                raise ValueError("System message must be first")
            continue
        if role not in {"user", "assistant"}:
            raise ValueError("Unsupported Flash-Next native media role")
        if role == "user":
            output += recipe["user"]["prefix"] + content + recipe["user"]["suffix"]
            continue
        reasoning = message.get("reasoning_content", "").strip()
        if (
            recipe["parse_assistant_thinking"]
            and "reasoning_content" not in message
            and "</think>" in content
        ):
            reasoning = (
                content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n").strip()
            )
            content = content.split("</think>")[-1].lstrip("\n")
        form = recipe["assistant_recent" if i > last_user else "assistant_history"]
        output += form["prefix"]
        if form["include_reasoning"]:
            output += reasoning + form["reasoning_suffix"]
        output += content + form["suffix"]
    return output + (recipe["generation"] if generation else "")


def provider_recipes(tokenizer):
    """Extract only validated no-tool message forms, not a version-specific template."""
    markers = [f"FLASH_NEXT_MEDIA_SENTINEL_{i}_END" for i in range(4)]
    a, b, c, reasoning = markers

    def message(role, content, **kwargs):
        return {"role": role, "content": content, **kwargs}

    result = {}
    for thinking in (False, True):

        def render(messages, generation=False):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=generation, enable_thinking=thinking
            )

        single = render([message("user", a)])
        full_prefix, user_suffix = _split(single, a)
        doubled = render([message("user", a), message("user", b)])
        middle = _split(_split(doubled, a)[1], b)[0]
        user_prefix = _without_prefix(middle, user_suffix)
        default = _without_suffix(full_prefix, user_prefix)
        system = _without_suffix(
            render([message("system", c), message("user", a)]), user_prefix + a + user_suffix
        )
        system_prefix, system_suffix = _split(system, c)
        recipe = {
            "default_prefix": default,
            "user": {"prefix": user_prefix, "suffix": user_suffix},
            "system": {"prefix": system_prefix, "suffix": system_suffix},
            "empty_system_prefix": _without_suffix(
                render([message("system", ""), message("user", a)]), user_prefix + a + user_suffix
            ),
            "generation": _without_prefix(render([message("user", a)], True), single),
        }
        for historical in (False, True):
            messages = [message("user", a), message("assistant", b, reasoning_content=reasoning)]
            if historical:
                messages.append(message("user", c))
            assistant = _without_prefix(render(messages), single)
            if historical:
                assistant = _without_suffix(assistant, user_prefix + c + user_suffix)
            prefix, suffix = _split(assistant, b)
            includes = reasoning in prefix
            start, between = _split(prefix, reasoning) if includes else (prefix, "")
            recipe["assistant_history" if historical else "assistant_recent"] = {
                "prefix": start,
                "include_reasoning": includes,
                "reasoning_suffix": between,
                "suffix": suffix,
            }
        probe = [message("user", a), message("assistant", f"<think>{reasoning}</think>{b}")]
        recipe["parse_assistant_thinking"] = False
        actual = render(probe)
        if render_recipe(recipe, probe, generation=False) != actual:
            recipe["parse_assistant_thinking"] = True
            if render_recipe(recipe, probe, generation=False) != actual:
                raise ValueError("Unsupported Flash-Next native assistant reasoning formatting")
        cases = [
            [message("user", " \nHello, 世界!\u2003")],
            [message("system", ""), message("user", "")],
            [message("system", " Follow instructions. "), message("user", a)],
            [message("user", a), message("assistant", b)],
            [message("user", a), message("assistant", b), message("user", c)],
            probe,
            probe + [message("user", c)],
            [
                message("user", a),
                message("assistant", b, reasoning_content=reasoning),
                message("user", c),
            ],
            [message("user", a), message("assistant", b, reasoning_content=reasoning)],
            [
                message("user", a),
                message("user", "<tool_response>value</tool_response>"),
                message("assistant", b),
            ],
            [message("user", "<|vision_start|><|image_pad|><|vision_end|>Describe this.")],
            [message("user", "<|vision_start|><|video_pad|><|vision_end|>Describe this.")],
        ]
        for messages in cases:
            for generation in (False, True):
                if render_recipe(recipe, messages, generation=generation) != render(
                    messages, generation
                ):
                    raise ValueError(
                        "Flash-Next native provider prompt recipe does not match checkpoint template"
                    )
        for kind in ("image", "video"):
            structured = [
                message("user", [{"type": kind}, {"type": "text", "text": "Describe this."}])
            ]
            expanded = [
                message("user", f"<|vision_start|><|{kind}_pad|><|vision_end|>Describe this.")
            ]
            if render(structured, True) != render_recipe(recipe, expanded):
                raise ValueError(
                    "Flash-Next native provider media placeholders differ from checkpoint template"
                )
        result[str(thinking).lower()] = recipe
    return result
