# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP 2 NaFlex's Transformers model for ``relevance``, as tests/test_e2e.py runs it: the Pillow processor
(``use_fast=False``, which the runtime reproduces), the image alone at its native aspect ratio with the patch mask
and spatial shape, every query of the request padded to the 64-token window, and each pair scored as the sigmoid of
the scaled cosine plus the bias (the runtime's probability)."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import AutoModel, AutoProcessor

        self.spec, self.host = spec, host
        self.processor = AutoProcessor.from_pretrained(spec.model, **{**spec.pretrained_kwargs(), "use_fast": False})
        self.model = host.maybe_compile(AutoModel.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        image = self.host.load_image(str(self.host.required(request, "image_path")))
        queries = [str(query) for query in self.host.required(request, "queries")]
        vision = self.processor(images=[image], return_tensors="pt")
        pixels = {"pixel_values": vision["pixel_values"].to(self.spec.device, self.spec.dtype),
                  "pixel_attention_mask": vision["pixel_attention_mask"].to(self.spec.device),
                  "spatial_shapes": vision["spatial_shapes"].to(self.spec.device)}
        text = self.processor(text=queries, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
        tokens = {key: text[key].to(self.spec.device) for key in ("input_ids", "attention_mask") if key in text}

        def score() -> Any:
            def pooled(output: Any) -> Any:
                return output if torch.is_tensor(output) else output.pooler_output

            image_vector = pooled(self.model.get_image_features(**pixels))[0].float()
            texts = pooled(self.model.get_text_features(**tokens)).float()
            image_unit = image_vector / image_vector.norm()
            texts = texts / texts.norm(dim=-1, keepdim=True)
            scale = float(self.model.logit_scale.exp())
            return torch.sigmoid(scale * (texts @ image_unit) + float(self.model.logit_bias))

        scores, model_ms = self.host.timed(score)
        values = scores.cpu().tolist()
        return self.host.invocation({"scored_pairs": len(values), "scores": values, "queries": queries,
                                     "top_class": max(range(len(values)), key=values.__getitem__)}, model_ms)
