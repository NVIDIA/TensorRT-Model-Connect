# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi-4 Multimodal's native pipeline for the trtmc-perf-serve reference backend (``generate``): the
checkpoint's own remote code (processor and causal LM) with eager attention, the image placed as
``<|image_1|>`` in the chat template, greedy generation, as the family's benchmark reference runs it. Text-only
requests go through the chat template without an image."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


IMAGE_MARKER = "<|image_1|>"


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        self.spec, self.torch = spec, torch
        options = {"trust_remote_code": True, **({"revision": spec.revision} if spec.revision else {})}
        self.processor = AutoProcessor.from_pretrained(spec.model, **options)
        config = AutoConfig.from_pretrained(spec.model, **options)
        config._attn_implementation = "eager"
        config._attn_implementation_internal = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(spec.model, torch_dtype=spec.dtype, config=config,
                                                          attn_implementation="eager", **options).eval().to(spec.device)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        prompt = str(request.get("prompt", ""))
        image_path = request.get("image_path")
        image = self.host.load_image(str(image_path)).convert("RGB") if image_path else None
        content = f"{IMAGE_MARKER}{prompt}" if image is not None else prompt
        rendered = self.processor.tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                                tokenize=False, add_generation_prompt=True)
        if image is not None and IMAGE_MARKER not in rendered:
            raise self.host.Error("the Phi-4 Multimodal chat template lost the image placeholder")
        encoded = self.processor(text=rendered, images=image, return_tensors="pt") if image is not None else \
            self.processor(text=rendered, return_tensors="pt")
        inputs = {name: value.to(device=self.spec.device,
                                 dtype=self.spec.dtype if value.is_floating_point() else value.dtype)
                  for name, value in encoded.items() if hasattr(value, "to")}
        steps = int(request.get("max_new_tokens", 128))
        generated, model_ms = self.host.timed(
            lambda: self.model.generate(**inputs, max_new_tokens=steps, do_sample=False, num_beams=1))
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        sequence = generated[0]
        tokens = sequence[prompt_tokens:] if sequence.shape[0] > prompt_tokens else sequence
        token_ids = [int(token) for token in tokens.detach().cpu().tolist()]
        return self.host.invocation({"text": self.processor.decode(tokens, skip_special_tokens=True),
                                     "token_ids": token_ids, "output_tokens": len(token_ids)}, model_ms)
