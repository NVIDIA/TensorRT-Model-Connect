# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chinese-CLIP's Transformers model for ``relevance``, as tests/test_e2e.py runs it: the Python processor
(``use_fast=False``), lowercasing as the runtime does, every query of the request against the one image with the
text padded to the 64-token window, and each pair scored as the logit scale times the cosine of the projected
embeddings (the runtime's logit)."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor

        self.spec, self.host = spec, host
        # The checkpoint ships no tokenizer config, and Transformers' BertTokenizer does not lowercase by default:
        # its vocabulary is lowercase, so a capital such as "English" would become [UNK]. The runtime lowercases.
        self.processor = ChineseCLIPProcessor.from_pretrained(spec.model, **{**spec.pretrained_kwargs(),
                                                                            "use_fast": False, "do_lower_case": True})
        self.model = host.maybe_compile(ChineseCLIPModel.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        image = self.host.load_image(str(self.host.required(request, "image_path")))
        queries = [str(query) for query in self.host.required(request, "queries")]
        batch = self.processor(text=queries, images=image, padding="max_length", max_length=64, truncation=True,
                               return_tensors="pt")
        pixels = batch["pixel_values"].to(self.spec.device, self.spec.dtype)
        tokens = {key: batch[key].to(self.spec.device) for key in ("input_ids", "attention_mask")}

        def score() -> Any:
            def embedded(output: Any, name: str) -> Any:
                if torch.is_tensor(output):
                    return output
                vector = getattr(output, name, None)
                return output.pooler_output if vector is None else vector

            image_vector = embedded(self.model.get_image_features(pixel_values=pixels), "image_embeds")[0].float()
            texts = embedded(self.model.get_text_features(**tokens), "text_embeds").float()
            image_unit = image_vector / image_vector.norm()
            texts = texts / texts.norm(dim=-1, keepdim=True)
            return float(self.model.logit_scale.exp()) * (texts @ image_unit)

        scores, model_ms = self.host.timed(score)
        values = scores.cpu().tolist()
        return self.host.invocation({"scored_pairs": len(values), "scores": values, "queries": queries,
                                     "top_class": max(range(len(values)), key=values.__getitem__)}, model_ms)
