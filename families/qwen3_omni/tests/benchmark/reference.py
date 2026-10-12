#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the family-owned Qwen3-Omni text reference."""

from __future__ import annotations

import sys
from typing import Any, Mapping, Sequence

from qualification_tests.benchmark_qualification.performance import reference_harness


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' }}
{%- if message.content is string %}
{{- message.content }}
{%- else %}
{%- for item in message.content %}
{%- if item.type == 'text' %}{{- item.text }}{%- endif %}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}"""


def _load(
    arguments: Any,
    request: Mapping[str, Any],
    options: Mapping[str, Any],
) -> reference_harness.Session:
    if arguments.mode != "hf-eager":
        raise ValueError("Qwen3-Omni reference requires hf-eager mode")
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    common = {
        "trust_remote_code": arguments.trust_remote_code,
        "local_files_only": arguments.local_files_only,
    }
    if arguments.revision:
        common["revision"] = arguments.revision
    processor = Qwen3OmniMoeProcessor.from_pretrained(arguments.model, **common)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        arguments.precision
    ]
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        arguments.model,
        torch_dtype=dtype,
        device_map=str(options.get("device_map", "cuda:0")),
        enable_audio_output=False,
        **common,
    ).eval()
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": str(request.get("prompt", ""))}]},
    ]
    processor_template = getattr(processor, "chat_template", None)
    tokenizer_template = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
    template = (
        processor_template
        if isinstance(processor_template, str) and processor_template.strip()
        else tokenizer_template
        if isinstance(tokenizer_template, str) and tokenizer_template.strip()
        else CHAT_TEMPLATE
    )
    inputs = processor.apply_chat_template(
        conversation,
        chat_template=template,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    max_new_tokens = int(request.get("max_new_tokens", 16))

    def invoke() -> Mapping[str, Any]:
        with torch.inference_mode():
            text_ids = model.generate(
                **inputs,
                thinker_max_new_tokens=max_new_tokens,
                thinker_do_sample=False,
                return_audio=False,
            )
        generated = text_ids[:, int(inputs["input_ids"].shape[-1]) :]
        token_ids = [int(token) for token in generated[0].detach().cpu().tolist()]
        return {
            "text": processor.batch_decode(generated, skip_special_tokens=True)[0].strip(),
            "token_ids": token_ids,
            "output_tokens": len(token_ids),
        }

    return reference_harness.Session(invoke, "transformers")


def main(argv: Sequence[str] | None = None) -> int:
    return reference_harness.run(
        list(sys.argv[1:] if argv is None else argv), description=__doc__, load=_load
    )


if __name__ == "__main__":
    raise SystemExit(main())
