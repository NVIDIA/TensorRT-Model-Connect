# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint's chat template must render what the native prompt renderer renders."""

from __future__ import annotations

from pathlib import Path


# Conversations and the exact prompts the native renderer (runtime/prompt.cpp) produces for them.
TEMPLATE_CASES = (
    ([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Text Recognition:"}]}],
     "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>Text Recognition:<|assistant|>\n"),
    ([{"role": "system", "content": [{"type": "text", "text": "You are OCR."}]},
      {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "A"}]},
      {"role": "assistant", "content": [{"type": "text", "text": "  spaced  \n"}]},
      {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"},
                                   {"type": "image"}]}],
     "[gMASK]<sop><|system|>\nYou are OCR.<|user|>\n<|begin_of_image|><|image|><|end_of_image|>A"
     "<|assistant|>\n<think></think>\nspaced<|user|>\nab<|begin_of_image|><|image|><|end_of_image|>"
     "<|assistant|>\n"),
)


def check_chat_template(path: Path) -> None:
    """Refuse a checkpoint whose chat template renders differently from the native renderer.

    The template is rendered as Transformers renders chat templates (sandboxed Jinja with
    ``trim_blocks``, ``lstrip_blocks`` and loop controls) on conversations that cover every
    rendering rule of the runtime: system, user and assistant turns, text and image parts.
    """
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    if not path.is_file():
        raise ValueError("glm_ocr: checkpoint has no chat_template.jinja")
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    try:
        template = environment.from_string(path.read_text(encoding="utf-8"))
        rendered = [template.render(messages=messages, add_generation_prompt=True)
                    for messages, _ in TEMPLATE_CASES]
    except Exception as error:  # noqa: BLE001 - any template failure means an unsupported template
        raise ValueError(f"glm_ocr cannot render this checkpoint's chat template: {error}") from error
    for (_, expected), actual in zip(TEMPLATE_CASES, rendered, strict=True):
        if actual != expected:
            raise ValueError(f"glm_ocr: chat template renders {actual!r} where the native renderer "
                             f"produces {expected!r}")
