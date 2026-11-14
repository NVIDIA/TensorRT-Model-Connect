# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLIP's Transformers model for ``relevance``, as tests/test_e2e.py runs it: every query of the request
against the one image, the image and text embeddings, each pair scored as the logit scale times their cosine (the runtime's logit)."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.spec, self.host = spec, host
        self.processor = CLIPProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = host.maybe_compile(CLIPModel.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        image = self.host.load_image(str(self.host.required(request, "image_path")))
        queries = [str(query) for query in self.host.required(request, "queries")]
        batch = self.processor(text=queries, images=image, padding="max_length", max_length=77, truncation=True,
                               return_tensors="pt")
        pixels = batch["pixel_values"].to(self.spec.device, self.spec.dtype)
        tokens = {key: batch[key].to(self.spec.device) for key in ("input_ids", "attention_mask") if key in batch}

        def score() -> Any:
            def pooled(output: Any) -> Any:
                return output if torch.is_tensor(output) else output.pooler_output

            image_vector = pooled(self.model.get_image_features(pixel_values=pixels))[0].float()
            texts = pooled(self.model.get_text_features(**tokens)).float()
            image = image_vector / image_vector.norm()
            texts = texts / texts.norm(dim=-1, keepdim=True)
            scale = float(self.model.logit_scale.exp())
            return scale * (texts @ image)

        scores, model_ms = self.host.timed(score)
        values = scores.cpu().tolist()
        return self.host.invocation({"scored_pairs": len(values), "scores": values, "queries": queries,
                                     "top_class": max(range(len(values)), key=values.__getitem__)}, model_ms)
