# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official BiRefNet foreground masks with the family's E2E preprocessing and output contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        if spec.mode != "eager":
            raise host.Error("BiRefNet reference supports only eager mode")
        from transformers import AutoModelForImageSegmentation

        self.spec, self.host = spec, host
        self.size = int(spec.options["input_size"])
        self.model = AutoModelForImageSegmentation.from_pretrained(
            spec.model, **spec.model_kwargs()
        ).eval().to(device=spec.device, dtype=spec.dtype)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        image = self.host.load_image(str(self.host.required(request, "image_path")))
        resized = np.asarray(image.resize((self.size, self.size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        pixels = torch.from_numpy(((resized - mean) / std).transpose(2, 0, 1)[None]).to(
            device=self.spec.device, dtype=self.spec.dtype
        )
        outputs, model_ms = self.host.timed(lambda: self.model(pixels))
        logits = outputs[-1]
        while isinstance(logits, (list, tuple)):
            logits = logits[-1]
        if tuple(logits.shape) != (1, 1, self.size, self.size) or not torch.isfinite(logits).all():
            raise self.host.Error("BiRefNet returned invalid or non-finite mask logits")
        probability = torch.sigmoid(logits.float())[0, 0].detach().cpu().numpy()
        coarse = (probability > 0.5).astype(np.int32)
        rows = (np.arange(image.height) * self.size // image.height).clip(0, self.size - 1)
        columns = (np.arange(image.width) * self.size // image.width).clip(0, self.size - 1)
        mask = coarse[rows[:, None], columns[None, :]]
        return self.host.invocation({"height": image.height, "width": image.width, "num_masks": 1,
                                     "mask_pixels": int(mask.size), "mask": mask.reshape(-1).tolist()}, model_ms)
