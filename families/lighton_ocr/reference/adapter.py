# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LightOnOCR's Transformers model for ``generate``, as tests/original_reference.py runs it: LightOnOcrProcessor's
chat template of the image alone (a prompt, when a request states one, after it), greedy decoding, and the tokens
after the prompt decoded without special tokens. The generic image-text adapter cannot: the image-text-to-text
auto class loads this checkpoint without its vision tower and projector weights (it renames none of them)."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

        self.spec, self.host = spec, host
        self.processor = LightOnOcrProcessor.from_pretrained(
            spec.model, **spec.pretrained_kwargs(), **dict(spec.options.get("processor_kwargs", {})))
        model, loading = LightOnOcrForConditionalGeneration.from_pretrained(
            spec.model, **spec.model_kwargs(), output_loading_info=True)
        incomplete = {key: value for key, value in loading.items()
                      if key in ("missing_keys", "unexpected_keys", "mismatched_keys") and value}
        if incomplete:
            raise host.Error(f"LightOnOCR checkpoint loaded incompletely: {sorted(incomplete)}")
        self.model = host.maybe_compile(model.to(spec.device, spec.dtype).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        image = self.host.load_image(str(self.host.required(request, "image_path")))
        prompt = str(request.get("prompt") or "")
        content: list[dict[str, Any]] = [{"type": "image", "image": image}]
        if prompt:
            content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.spec.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.spec.dtype)
        limit = int(request.get("max_new_tokens", 64))

        def generate() -> Any:
            with torch.inference_mode():
                return self.model.generate(**inputs, max_new_tokens=limit, do_sample=False)[0]

        output, model_ms = self.host.timed(generate)
        token_ids = output[inputs["input_ids"].shape[1]:].cpu().tolist()
        text = self.processor.decode(token_ids, skip_special_tokens=True)
        return self.host.invocation({"text": text, "token_ids": token_ids, "output_tokens": len(token_ids)}, model_ms)
