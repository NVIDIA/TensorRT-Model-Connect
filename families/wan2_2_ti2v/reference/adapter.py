# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2-TI2V-5B's native image-to-video pipeline for ``generate_image``: the Diffusers export of the same weights
through ``WanImageToVideoPipeline`` (the TI2V model conditions the first latent frame on the image), with the
request's prompt, negative prompt, size, frame count, steps, guidance and seed; frames THWC in 0-1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        import torch
        from diffusers import WanImageToVideoPipeline

        self.spec, self.host, self.torch = spec, host, torch
        kwargs = {"revision": spec.revision} if spec.revision else {}
        self.pipe = WanImageToVideoPipeline.from_pretrained(spec.model, torch_dtype=spec.dtype, **kwargs).to(spec.device)
        self.pipe.set_progress_bar_config(disable=True)
        host.maybe_compile(self.pipe.transformer, spec)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        host = self.host
        image = host.load_image(str(host.required(request, "image_path"))).convert("RGB")
        height, width = int(host.required(request, "height")), int(host.required(request, "width"))
        if image.size != (width, height):
            raise host.Error(f"the initial image is {image.size[0]}x{image.size[1]}, not the video's {width}x{height}")
        kwargs: dict[str, Any] = {
            "image": image, "prompt": str(host.required(request, "prompt")), "height": height, "width": width,
            "num_frames": int(host.required(request, "num_frames")), "output_type": "np",
            "generator": self.torch.Generator(self.spec.device).manual_seed(max(int(request.get("seed", 0)), 0))}
        if request.get("negative_prompt"):
            kwargs["negative_prompt"] = str(request["negative_prompt"])
        if int(request.get("num_steps", -1)) > 0:
            kwargs["num_inference_steps"] = int(request["num_steps"])
        if float(request.get("guidance_scale", -1.0)) >= 0:
            kwargs["guidance_scale"] = float(request["guidance_scale"])
        result, model_ms = host.timed(lambda: self.pipe(**kwargs))
        video = np.asarray(result.frames[0])
        return host.invocation({**host.tensor_observation(video, artifact_base), "media_type": "video",
                                "num_frames": int(video.shape[0]), "height": int(video.shape[1]),
                                "width": int(video.shape[2])}, model_ms)
