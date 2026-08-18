# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the pinned official Fast-FoundationStereo PyTorch implementation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tensorrt_model_connect.families.fast_foundation_stereo.prepare_model import (
    configure_official_model_args,
    install_official_io_import_shims,
)


def _load_stereo_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if pixels.shape != (700, 700, 3):
        raise ValueError(f"Stereo fixture must be 700x700 RGB, got {pixels.shape}")
    return pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-image", type=Path, required=True)
    parser.add_argument("--right-image", type=Path, required=True)
    parser.add_argument("--valid-iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Fast-FoundationStereo reference")

    model_root = arguments.model_root.resolve()
    os.chdir(model_root)
    sys.path.insert(0, str(model_root))
    install_official_io_import_shims()
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE

    checkpoint = model_root / "weights/23-36-37/model_best_bp2_serialize.pth"
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configure_official_model_args(
        model,
        max_disparity=arguments.max_disp,
        valid_iters=arguments.valid_iters,
    )
    model = model.cuda().eval()

    left = _load_stereo_image(arguments.left_image)
    right = _load_stereo_image(arguments.right_image)
    left_tensor = torch.as_tensor(left, device="cuda").float()[None].permute(0, 3, 1, 2)
    right_tensor = torch.as_tensor(right, device="cuda").float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(left_tensor.shape, divis_by=32, force_square=False)
    left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)
    with (
        torch.inference_mode(),
        torch.amp.autocast(
            "cuda",
            enabled=True,
            dtype=AMP_DTYPE,
        ),
    ):
        disparity = model.forward(
            left_tensor,
            right_tensor,
            iters=arguments.valid_iters,
            test_mode=True,
            optimize_build_volume="pytorch1",
        )
    output = padder.unpad(disparity.float()).cpu().numpy().reshape(700, 700)
    output = np.clip(output, 0, None).astype(np.float32)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(arguments.output, output, allow_pickle=False)


if __name__ == "__main__":
    main()
