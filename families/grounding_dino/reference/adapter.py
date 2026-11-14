# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounding DINO's Transformers model for ``detect_prompted``, as tests/test_e2e.py runs it: the image
resized to the bundle's fixed square first (the runtime does the same), the processor's own resize off,
the grounded post-processing at the request's box and text thresholds, boxes in source pixels."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self.spec, self.host = spec, host
        self.processor = AutoProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = host.maybe_compile(GroundingDinoForObjectDetection.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)
        width, height = spec.options["image_size"]  # the manifest's image_width and image_height
        self.size = (int(width), int(height))

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        from PIL import Image

        image = self.host.load_image(str(self.host.required(request, "image_path")))
        resized = image.resize(self.size, Image.BILINEAR)
        encoding = self.processor(images=resized, text=str(self.host.required(request, "prompt")),
                                  return_tensors="pt", do_resize=False)
        batch = {key: value.to(self.spec.device, self.spec.dtype) if value.is_floating_point()
                 else value.to(self.spec.device) for key, value in encoding.items()}
        outputs, model_ms = self.host.timed(lambda: self.model(**batch))
        result = self.processor.post_process_grounded_object_detection(
            outputs, batch["input_ids"], threshold=float(request.get("box_threshold", 0.35)),
            text_threshold=float(request.get("text_threshold", 0.25)),
            target_sizes=[(image.height, image.width)])[0]
        boxes = result["boxes"].float().cpu()
        return self.host.invocation({
            "detected_images": 1, "detections": int(boxes.shape[0]), "boxes": boxes.flatten().tolist(),
            "scores": result["scores"].float().cpu().tolist(), "labels": [str(label) for label in result["text_labels"]],
            "coordinates": "xyxy", "units": "pixels", "image_width": image.width, "image_height": image.height,
        }, model_ms)
