# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-OCR's native pipeline for the trtmc-perf-serve reference backend (``generate``): the checkpoint's
own remote code (``model.infer``, which reads the image file itself) at bf16 with eager attention, base size
1024, image size 768, crop mode, as the family's benchmark reference runs it. The generic image-text adapter
cannot: the checkpoint ships a tokenizer, no processor."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Mapping



class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        if spec.precision != "bf16":
            raise self.host.Error("the official DeepSeek-OCR reference runs at bf16")
        from transformers import AutoModel, AutoTokenizer

        self.spec = spec
        options = {"trust_remote_code": True, **({"revision": spec.revision} if spec.revision else {})}
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model, **options)
        self.model = AutoModel.from_pretrained(spec.model, use_safetensors=True, torch_dtype=spec.dtype,
                                               attn_implementation="eager", **options).to(spec.device).eval()
        self.scratch = tempfile.TemporaryDirectory(prefix="deepseek-ocr-")

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        prompt = str(request.get("prompt", ""))
        if "<image>" not in prompt:
            prompt = f"<image>\n{prompt}"
        image = str(self.host.required(request, "image_path"))
        text, model_ms = self.host.timed(lambda: self.model.infer(
            self.tokenizer, prompt=prompt, image_file=image, output_path=self.scratch.name, base_size=1024,
            image_size=768, crop_mode=True, save_results=False, eval_mode=True))
        text = str(text or "")
        token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return self.host.invocation({"text": text, "token_ids": token_ids, "output_tokens": len(token_ids)}, model_ms)
