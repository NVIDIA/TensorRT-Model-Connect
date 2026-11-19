# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-Edge's Transformers model for ``generate`` with an image, as tests/test_e2e.py runs it: the chat
template with the image before the question and thinking off, greedy decoding of the requested new tokens."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import AutoProcessor, Cosmos3EdgeForConditionalGeneration

        self.spec, self.host = spec, host
        self.processor = AutoProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = host.maybe_compile(Cosmos3EdgeForConditionalGeneration.from_pretrained(
            spec.model, **spec.model_kwargs(), device_map=spec.device_map).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        prompt = str(self.host.required(request, "prompt"))
        image_path = request.get("image_path")
        parts = ([{"type": "image"}] if image_path else []) + [{"type": "text", "text": prompt}]
        text = self.processor.apply_chat_template([{"role": "user", "content": parts}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False)
        media = {"images": [self.host.load_image(str(image_path))]} if image_path else {}
        inputs = self.processor(text=[text], return_tensors="pt", **media).to(self.spec.device)
        new_tokens = int(request.get("max_new_tokens", 64))
        output, model_ms = self.host.timed(lambda: self.model.generate(**inputs, do_sample=False,
                                                                       max_new_tokens=new_tokens))
        prompt_tokens = int(inputs["input_ids"].shape[1])
        token_ids = output[0, prompt_tokens:].tolist()
        decoded = self.processor.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return self.host.invocation({"output_tokens": len(token_ids), "token_ids": token_ids, "text": decoded},
                                    model_ms, prompt_tokens=prompt_tokens)
