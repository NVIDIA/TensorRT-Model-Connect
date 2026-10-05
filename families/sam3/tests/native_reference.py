# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAM3's native pipeline for the trtmc-perf-serve reference backend (``segment_prompted`` with a text
prompt): Transformers' Sam3Model and Sam3Processor, instance masks at the processor's 0.5 thresholds."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


THRESHOLD = 0.5
MASK_THRESHOLD = 0.5


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        from transformers import Sam3Model, Sam3Processor

        self.spec = spec
        self.processor = Sam3Processor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = Sam3Model.from_pretrained(spec.model, dtype=spec.dtype,
                                               **spec.pretrained_kwargs()).to(spec.device).eval()

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        image = self.host.load_image(str(self.host.required(request, "image_path"))).convert("RGB")
        encoded = self.processor(images=image, text=str(self.host.required(request, "prompt")), return_tensors="pt")
        inputs = {name: value.to(self.spec.device) if hasattr(value, "to") else value for name, value in encoded.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.spec.dtype)
        outputs, model_ms = self.host.timed(lambda: self.model(**inputs))
        result = self.processor.post_process_instance_segmentation(
            outputs, threshold=THRESHOLD, mask_threshold=MASK_THRESHOLD,
            target_sizes=inputs["original_sizes"].cpu().tolist())[0]
        masks = result["masks"].detach().float().cpu()
        scores = result["scores"].detach().float().cpu().reshape(-1)
        boxes = result["boxes"].detach().float().cpu().reshape(-1, 4)
        height, width = (int(value) for value in masks.shape[-2:]) if masks.ndim == 3 else (image.height, image.width)
        count = int(masks.shape[0]) if masks.ndim == 3 else 0
        return self.host.invocation({"segmented_images": 1, "num_masks": count, "height": height, "width": width,
                                     "mask_kind": "binary", "masks": masks.reshape(-1).tolist(),
                                     "iou_scores": scores.tolist(), "boxes": boxes.tolist(),
                                     "box_coordinates": "original_image_pixels_xyxy"}, model_ms)
