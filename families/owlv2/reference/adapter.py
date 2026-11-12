# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OWLv2's Transformers model for ``detect_prompted``, as tests/test_e2e.py runs it: one query per
'.'-separated phrase of the prompt, the grounded post-processing at the request's score threshold, boxes
scaled by the padded square's side (the longer image edge), so in source pixels."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import AutoProcessor, Owlv2ForObjectDetection

        self.spec, self.host = spec, host
        self.processor = AutoProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = host.maybe_compile(Owlv2ForObjectDetection.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        image = self.host.load_image(str(self.host.required(request, "image_path")))
        phrases = [phrase.strip() for phrase in str(self.host.required(request, "prompt")).split(".") if phrase.strip()]
        encoding = self.processor(images=image, text=[phrases], return_tensors="pt")
        batch = {key: value.to(self.spec.device, self.spec.dtype) if value.is_floating_point()
                 else value.to(self.spec.device) for key, value in encoding.items()}
        outputs, model_ms = self.host.timed(lambda: self.model(**batch))
        side = max(image.height, image.width)
        result = self.processor.post_process_grounded_object_detection(
            outputs=outputs, threshold=float(request.get("score_threshold", 0.1)),
            target_sizes=torch.tensor([[side, side]]))[0]
        boxes = result["boxes"].float().cpu()
        return self.host.invocation({
            "detected_images": 1, "detections": int(boxes.shape[0]), "boxes": boxes.flatten().tolist(),
            "scores": result["scores"].float().cpu().tolist(),
            "labels": [phrases[int(label)] for label in result["labels"].cpu().tolist()],
            "coordinates": "xyxy", "units": "pixels", "image_width": image.width, "image_height": image.height,
        }, model_ms)
