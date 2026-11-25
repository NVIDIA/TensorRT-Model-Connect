# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable Video Diffusion's native pipeline for ``generate_image``, as tests/test_e2e.py runs it: Diffusers'
``StableVideoDiffusionPipeline`` on the initial image alone (an image-only model: a stated prompt is refused, as the
runtime does), at the image's size and the bundle's frame count, with the request's steps, seed and conditioning
controls. The noise follows the runtime's stream: a CPU generator seeded with ``seed`` draws the float32
conditioning-image noise and then the float32 latents, cast to the pipeline dtype. Frames THWC in 0-1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

# The runtime's conditioning controls (its Task config) that the pipeline also takes.
CONTROLS = {"min_guidance_scale": float, "max_guidance_scale": float, "fps": int, "motion_bucket_id": int,
            "noise_aug_strength": float}


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        import torch
        from diffusers import StableVideoDiffusionPipeline

        self.spec, self.host, self.torch = spec, host, torch
        kwargs = {"revision": spec.revision} if spec.revision else {}
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(spec.model, torch_dtype=spec.dtype,
                                                                 **kwargs).to(spec.device)
        self.pipe.set_progress_bar_config(disable=True)
        host.maybe_compile(self.pipe.unet, spec)
        self.frames = int(spec.options["num_frames"])  # the bundle's frame count (its manifest's video_num_frames)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        torch, host = self.torch, self.host
        if str(request.get("prompt") or ""):
            raise host.Error("Stable Video Diffusion is conditioned on the initial image only; pass an empty prompt")
        image = host.load_image(str(host.required(request, "image_path"))).convert("RGB")
        width, height = image.size
        seed = int(request.get("seed", 0))
        generator = torch.Generator("cpu").manual_seed(seed)
        torch.randn((1, 3, height, width), generator=generator)  # the conditioning-image noise draw
        factor = self.pipe.vae_scale_factor
        latents = torch.randn((1, self.frames, self.pipe.unet.config.in_channels // 2, height // factor,
                               width // factor), generator=generator).to(self.spec.dtype)
        kwargs: dict[str, Any] = {
            "image": image, "height": height, "width": width, "num_frames": self.frames,
            "decode_chunk_size": self.frames, "generator": torch.Generator("cpu").manual_seed(seed),
            "latents": latents, "output_type": "np"}
        if int(request.get("num_steps", -1)) > 0:
            kwargs["num_inference_steps"] = int(request["num_steps"])
        kwargs.update({name: convert(request[name]) for name, convert in CONTROLS.items() if name in request})
        with torch.no_grad():
            result, model_ms = host.timed(lambda: self.pipe(**kwargs))
        video = np.asarray(result.frames[0], dtype=np.float32)
        return host.invocation({**host.tensor_observation(video, artifact_base), "media_type": "video",
                                "num_frames": int(video.shape[0]), "height": int(video.shape[1]),
                                "width": int(video.shape[2])}, model_ms)
