# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PaliGemma's Transformers model for ``generate``, as tests/test_e2e.py runs it: the processor's task-prefix
prompt (image tokens, BOS, the prefix and a newline), greedy decoding, and the generated tokens after the prompt
decoded without special tokens. The generic image-text adapter cannot: the processor has no chat template."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

        self.spec, self.host = spec, host
        self.processor = PaliGemmaProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = host.maybe_compile(PaliGemmaForConditionalGeneration.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        image = self.host.load_image(str(self.host.required(request, "image_path")))
        prompt = str(self.host.required(request, "prompt"))
        inputs = self.processor(images=[[image]], text=[prompt], return_tensors="pt").to(self.spec.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.spec.dtype)
        limit = int(request.get("max_new_tokens", 64))

        def generate() -> Any:
            with torch.inference_mode():
                return self.model.generate(**inputs, max_new_tokens=limit, do_sample=False)[0]

        output, model_ms = self.host.timed(generate)
        token_ids = output[inputs["input_ids"].shape[1]:].cpu().tolist()
        text = self.processor.decode(token_ids, skip_special_tokens=True)
        return self.host.invocation({"text": text, "token_ids": token_ids, "output_tokens": len(token_ids)}, model_ms)
