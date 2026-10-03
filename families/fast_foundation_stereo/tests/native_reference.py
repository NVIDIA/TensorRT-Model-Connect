# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast-FoundationStereo's native pipeline for the trtmc-perf-serve reference backend (``disparity``): the
pinned official source and serialized checkpoint (native_prepare.py) on the 700x700 profile (max disparity 192,
8 refinement iterations, padded to multiples of 32, autocast as upstream), writing the disparity file the TRTMC
worker writes (``<artifact>.disparity.f32``)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from trtmc_perf_serving.backends.base import BackendError, Invocation
from trtmc_perf_serving.backends.reference.common import (ReferenceSpec, deferred_files, invocation, load_image, required,
                                                           timed)

PROFILE = {"height": 700, "width": 700, "max_disp": 192, "valid_iters": 8}
WEIGHTS = "weights/23-36-37/model_best_bp2_serialize.pth"


class Adapter:
    def __init__(self, spec: ReferenceSpec) -> None:
        import torch

        self.spec = spec
        root = Path(sys.prefix) / "trtmc-reference/Fast-FoundationStereo"
        if not (root / WEIGHTS).is_file():
            raise BackendError(f"the Fast-FoundationStereo reference is not prepared: {root}")
        previous = Path.cwd()
        try:  # upstream imports and the pickled model resolve their modules relative to the checkout
            os.chdir(root)
            sys.path.insert(0, str(root))
            from core.utils.utils import InputPadder
            from Utils import AMP_DTYPE

            model = torch.load(root / WEIGHTS, map_location="cpu", weights_only=False)
        finally:
            os.chdir(previous)
        model.args.max_disp, model.args.valid_iters, model.args.normalize = PROFILE["max_disp"], PROFILE["valid_iters"], True
        self.model, self.padder, self.amp_dtype = model.cuda().eval(), InputPadder, AMP_DTYPE

    def _image(self, path: str) -> np.ndarray:
        pixels = np.asarray(load_image(path), dtype=np.uint8)  # decoded before the timed call (preloaded)
        if pixels.shape != (PROFILE["height"], PROFILE["width"], 3):
            raise BackendError(f"stereo images must be {PROFILE['height']}x{PROFILE['width']} RGB, got {pixels.shape}")
        return pixels

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        import torch

        for key, value in PROFILE.items():
            if int(request.get(key, value)) != value:
                raise BackendError(f"the reference runs the 700x700 profile ({key}={value})")
        left = torch.as_tensor(self._image(str(required(request, "left_image_path")))).cuda().float()[None].permute(0, 3, 1, 2)
        right = torch.as_tensor(self._image(str(required(request, "right_image_path")))).cuda().float()[None].permute(0, 3, 1, 2)
        padder = self.padder(left.shape, divis_by=32, force_square=False)
        left, right = padder.pad(left, right)

        def run() -> Any:
            with torch.amp.autocast("cuda", enabled=True, dtype=self.amp_dtype):
                return self.model.forward(left, right, iters=PROFILE["valid_iters"], test_mode=True,
                                          optimize_build_volume="pytorch1")

        disparity, model_ms = timed(run)
        values = np.clip(padder.unpad(disparity.float()).cpu().numpy().reshape(PROFILE["height"], PROFILE["width"]),
                         0, None).astype("<f4", copy=False)
        artifact = Path(f"{artifact_base}.disparity.f32")
        observation = {"height": PROFILE["height"], "width": PROFILE["width"], "element_count": int(values.size),
                       "disparity_artifact": str(artifact.resolve())}
        return invocation(deferred_files(observation, {artifact: values}), model_ms)  # written after the timed call
