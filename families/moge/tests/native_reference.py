# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoGe-2's native pipeline for the trtmc-perf-serve reference backend (``infer_geometry``): the pinned official
source (native_prepare.py) through the family's reference loader, writing the geometry files the TRTMC worker
writes (``<artifact>.depth.f32``, ``.points.f32``, ``.mask.u8``) and reporting their paths."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from families.moge.tests.reference_support import OfficialReference

CHECKPOINT = "model.pt"


class Adapter:
    def __init__(self, spec: Any, host: Any) -> None:
        self.host = host
        from huggingface_hub import hf_hub_download

        self.spec = spec
        source = Path(sys.prefix) / "trtmc-reference/MoGe"
        checkpoint = hf_hub_download(spec.model, CHECKPOINT, revision=spec.revision)
        self.reference = OfficialReference(source, Path(checkpoint))

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Any:
        # Decoded before the timed call (preloaded), as the official read_image does: RGB in [0, 1].
        image = self.host.load_image(str(self.host.required(request, "image_path")))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        fov_x = float(request["fov_x"]) if request.get("fov_x") is not None else None
        tokens = int(request.get("num_tokens", 1800))
        arrays, model_ms = self.host.timed(lambda: self.reference.infer(pixels, num_tokens=tokens, fov_x=fov_x))
        depth = np.asarray(arrays["depth"], dtype="<f4")
        height, width = depth.shape
        paths = {"points_artifact": Path(f"{artifact_base}.points.f32"), "depth_artifact": Path(f"{artifact_base}.depth.f32"),
                 "valid_mask_artifact": Path(f"{artifact_base}.mask.u8")}
        files = {paths["points_artifact"]: np.asarray(arrays["points"], dtype="<f4"), paths["depth_artifact"]: depth,
                 paths["valid_mask_artifact"]: np.asarray(arrays["mask"], dtype=np.uint8)}
        observation = {"geometry_images": 1, "height": height, "width": width,
                       "valid_pixels": int(np.asarray(arrays["mask"]).astype(bool).sum()), "units": "meters",
                       **{name: str(path.resolve()) for name, path in paths.items()}}
        # written after the timed call
        return self.host.invocation(self.host.deferred_files(observation, files), model_ms)
