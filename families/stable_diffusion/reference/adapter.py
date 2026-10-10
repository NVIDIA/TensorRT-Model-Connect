# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SD1.5's official Native pipeline, with the family's DDIM sampler and caller noise."""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


@lru_cache(maxsize=None)
def _geometry(snapshot: str) -> tuple[int, int]:
    root = Path(snapshot)
    unet = json.loads((root / "unet/config.json").read_text())
    vae = json.loads((root / "vae/config.json").read_text())
    return int(unet["in_channels"]), 2 ** (len(vae["block_out_channels"]) - 1)


class Adapter:
    @staticmethod
    def latent_shape(snapshot: Path, request: Mapping[str, Any]) -> tuple[int, ...]:
        channels, scale = _geometry(str(snapshot))
        height, width = int(request["height"]), int(request["width"])
        if height <= 0 or width <= 0 or height % scale or width % scale:
            raise ValueError("stable_diffusion requires positive height/width divisible by the VAE scale")
        return (1, channels, height // scale, width // scale)

    def __init__(self, spec: Any, host: Any) -> None:
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        self.spec, self.host = spec, host
        self.pipe = StableDiffusionPipeline.from_pretrained(
            spec.model, torch_dtype=spec.dtype, safety_checker=None, requires_safety_checker=False,
            **spec.pretrained_kwargs()).to(spec.device)
        self.pipe.scheduler = DDIMScheduler.from_pretrained(
            spec.model, subfolder="scheduler", **spec.pretrained_kwargs())
        self.pipe.set_progress_bar_config(disable=True)
        if spec.mode == "compile":
            import torch

            self.pipe.unet.forward = torch.compile(self.pipe.unet.forward)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        import numpy as np
        import torch

        height, width = int(request["height"]), int(request["width"])
        shape = (1, self.pipe.unet.config.in_channels,
                 height // self.pipe.vae_scale_factor, width // self.pipe.vae_scale_factor)
        flat = np.fromfile(str(self.host.required(request, "initial_latents_path")), dtype=np.float32)
        if flat.size != int(np.prod(shape)):
            raise self.host.Error(f"SD initial latents contain {flat.size} floats; expected {shape}")
        latents = torch.from_numpy(flat.reshape(shape)).to(device=self.spec.device, dtype=self.spec.dtype)
        kwargs = dict(prompt=str(self.host.required(request, "prompt")), height=height, width=width,
                      num_inference_steps=int(request["num_steps"]), guidance_scale=float(request["guidance_scale"]),
                      negative_prompt=request.get("negative_prompt") or None, latents=latents, output_type="np")
        output, ms = self.host.timed(lambda: self.pipe(**kwargs).images)
        return self.host.invocation(self.host.tensor_observation(np.asarray(output), artifact_base), ms,
                                    sampler="DDIM", safety_checker=False)
