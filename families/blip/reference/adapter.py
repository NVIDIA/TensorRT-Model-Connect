# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BLIP's Transformers captioner for ``generate``, as tests/test_e2e.py runs it: the Python processor, the prompt
as a caption prefix (none when empty), greedy decoding, and the continuation after the prefix decoded without
special tokens (the runtime's text). The generic image-text adapter cannot: the processor has no chat template."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.spec, self.host = spec, host
        self.processor = BlipProcessor.from_pretrained(spec.model, **{**spec.pretrained_kwargs(), "use_fast": False})
        self.model = host.maybe_compile(BlipForConditionalGeneration.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        image = self.host.load_image(str(self.host.required(request, "image_path")))
        prompt = str(request.get("prompt") or "")
        inputs = self.processor(images=image, text=prompt or None, return_tensors="pt").to(self.spec.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.spec.dtype)
        limit = int(request.get("max_new_tokens", 32))

        def generate() -> Any:
            with torch.inference_mode():
                return self.model.generate(**inputs, max_new_tokens=limit, do_sample=False, num_beams=1)[0]

        output, model_ms = self.host.timed(generate)
        # The prompt's ids end in SEP, which generation drops; without a prompt the output starts with BOS.
        prefix = inputs["input_ids"].shape[1] - 1 if "input_ids" in inputs else 1
        token_ids = output[prefix:].cpu().tolist()
        text = self.processor.decode(token_ids, skip_special_tokens=True)
        return self.host.invocation({"text": text, "token_ids": token_ids, "output_tokens": len(token_ids)}, model_ms)
