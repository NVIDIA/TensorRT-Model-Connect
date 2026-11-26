# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TrOCR's Transformers model for ``generate``, as tests/common.py runs it: the Pillow image processor
(``ViTImageProcessor``, whose bilinear filter the runtime reproduces), greedy decoding of the one text line, and
the tokens after the decoder start decoded without special tokens. The generic image-text adapter cannot: a vision
encoder-decoder has no processor chat template and no image-text-to-text class."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import AutoTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

        self.spec, self.host = spec, host
        self.processor = ViTImageProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = host.maybe_compile(VisionEncoderDecoderModel.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        if request.get("prompt"):
            raise self.host.Error("TrOCR takes no prompt")
        image = self.host.load_image(str(self.host.required(request, "image_path")))
        pixels = self.processor(images=image, return_tensors="pt").pixel_values.to(self.spec.device, self.spec.dtype)
        limit = int(request.get("max_new_tokens", 32))

        def generate() -> Any:
            with torch.inference_mode():
                return self.model.generate(pixels, max_new_tokens=limit, do_sample=False, num_beams=1)[0]

        output, model_ms = self.host.timed(generate)
        token_ids = output.cpu().tolist()
        start = self.model.generation_config.decoder_start_token_id
        token_ids = token_ids[1:] if token_ids and token_ids[0] == start else token_ids
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return self.host.invocation({"text": text, "token_ids": token_ids, "output_tokens": len(token_ids)}, model_ms)
