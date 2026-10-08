# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3's native pipeline for the trtmc-perf-serve reference backend (``generate_image``, video): Diffusers'
ModularPipeline (``t2va`` workflow) from the pinned snapshot with its components at the reference precision, a
CPU generator seeded from the request, as the family's E2E reference runs it. The weights (351 GB) exceed one GPU,
so with ``options.cpu_offload`` a ComponentsManager moves components onto the GPU as they run. The released weights
are CFG-distilled: the pipeline takes no guidance input, and the request's guidance fields are not passed."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np



class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        import torch
        from diffusers import ComponentsManager, ModularPipeline
        from huggingface_hub import snapshot_download

        self.spec, self.torch = spec, torch
        directory = snapshot_download(spec.model, revision=spec.revision)
        manager = None
        if spec.options.get("cpu_offload"):
            manager = ComponentsManager()
            manager.enable_auto_cpu_offload(device=spec.device)
        pipeline = ModularPipeline.from_pretrained(directory, workflow="t2va", components_manager=manager,
                                                   local_files_only=True)
        pipeline.load_components(dtype=spec.dtype, pretrained_model_name_or_path=directory, local_files_only=True)
        self.pipeline = pipeline if manager is not None else pipeline.to(spec.device)

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        kwargs: dict[str, Any] = {
            "prompt": str(self.host.required(request, "prompt")),
            "height": int(self.host.required(request, "height")), "width": int(self.host.required(request, "width")),
            "num_inference_steps": int(self.host.required(request, "num_steps")),
            "generator": self.torch.Generator().manual_seed(max(int(request.get("seed", 0)), 0))}
        if int(request.get("num_frames", 1)) > 1:
            kwargs["num_frames"] = int(request["num_frames"])
        if request.get("negative_prompt"):
            kwargs["negative_prompt"] = str(request["negative_prompt"])
        videos, model_ms = self.host.timed(lambda: self.pipeline(output="videos", output_type="np", **kwargs))
        frames = np.asarray(videos[0])
        return self.host.invocation({**self.host.tensor_observation(frames, artifact_base), "media_type": "video",
                                     "num_frames": int(frames.shape[0]), "height": int(frames.shape[1]),
                                     "width": int(frames.shape[2])}, model_ms)
