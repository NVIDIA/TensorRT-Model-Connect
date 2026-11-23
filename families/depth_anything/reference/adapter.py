# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth Anything V2's Transformers model for ``depth``, as tests/test_e2e.py runs it: the image resized to the
checkpoint's native square (bilinear, aspect ratio not kept, as the engine does), the predicted depth resampled to
the source image size (nearest)."""

from pathlib import Path
from typing import Any, Mapping


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.spec, self.host = spec, host
        self.processor = AutoImageProcessor.from_pretrained(spec.model, **spec.pretrained_kwargs())
        self.model = host.maybe_compile(AutoModelForDepthEstimation.from_pretrained(
            spec.model, **spec.model_kwargs()).to(spec.device, spec.dtype).eval(), spec)
        self.size = int(self.model.config.backbone_config.image_size)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import torch

        image = self.host.load_image(str(self.host.required(request, "image_path")))
        encoding = self.processor(images=image, return_tensors="pt", size={"height": self.size, "width": self.size},
                                  keep_aspect_ratio=False, resample=2)
        pixels = encoding["pixel_values"].to(self.spec.device, self.spec.dtype)
        depth, model_ms = self.host.timed(lambda: self.model(pixel_values=pixels).predicted_depth)
        resampled = torch.nn.functional.interpolate(depth.unsqueeze(1).float(), size=(image.height, image.width),
                                                    mode="nearest").squeeze()
        return self.host.invocation({"depth_images": 1, "depth_pixels": int(resampled.numel()),
                                     "height": image.height, "width": image.width,
                                     "depth": resampled.flatten().cpu().tolist()}, model_ms)
