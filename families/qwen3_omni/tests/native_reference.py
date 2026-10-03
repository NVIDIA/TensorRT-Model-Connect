# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni's native pipeline for the trtmc-perf-serve reference backend (``generate``): Transformers'
Qwen3OmniMoeForConditionalGeneration without audio output, the prompt wrapped in the model's fixed system prompt
and chat template (the request's own chat flag stays off: TRTMC applies the same fixed contract), greedy thinker
decoding, as the family's benchmark reference runs it."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from trtmc_perf_serving.backends.base import Invocation
from trtmc_perf_serving.backends.reference.common import ReferenceSpec, invocation, required, timed

SYSTEM_PROMPT = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
                 "auditory and visual inputs, as well as generating text and speech.")
CHAT_TEMPLATE = """{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\\n' }}
{%- if message.content is string %}
{{- message.content }}
{%- else %}
{%- for item in message.content %}
{%- if item.type == 'text' %}{{- item.text }}{%- endif %}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\\n' }}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"""


class Adapter:
    def __init__(self, spec: ReferenceSpec) -> None:
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.spec = spec
        options = {"trust_remote_code": spec.trust_remote_code, **({"revision": spec.revision} if spec.revision else {})}
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(spec.model, **options)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            spec.model, torch_dtype=spec.dtype, device_map=spec.device, enable_audio_output=False, **options).eval()
        templates = (getattr(self.processor, "chat_template", None),
                     getattr(getattr(self.processor, "tokenizer", None), "chat_template", None))
        self.template = next((value for value in templates if isinstance(value, str) and value.strip()), CHAT_TEMPLATE)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        conversation = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user", "content": [{"type": "text", "text": str(required(request, "prompt"))}]}]
        inputs = self.processor.apply_chat_template(conversation, chat_template=self.template, add_generation_prompt=True,
                                                    tokenize=True, return_dict=True, return_tensors="pt",
                                                    padding=True).to(self.model.device)
        steps = int(request.get("max_new_tokens", 16))
        text_ids, model_ms = timed(lambda: self.model.generate(**inputs, thinker_max_new_tokens=steps,
                                                               thinker_do_sample=False, return_audio=False))
        generated = text_ids[:, int(inputs["input_ids"].shape[-1]):]
        token_ids = [int(token) for token in generated[0].detach().cpu().tolist()]
        return invocation({"text": self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip(),
                           "token_ids": token_ids, "output_tokens": len(token_ids)}, model_ms)
