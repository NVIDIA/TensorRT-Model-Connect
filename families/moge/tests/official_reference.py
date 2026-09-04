# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the pinned official MoGe-2 reference with an explicit token budget."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import types
from pathlib import Path

import numpy as np


def _math_sdpa(torch):
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except (ImportError, AttributeError):
        return contextlib.nullcontext()


def _install_utils3d_shim(torch) -> None:
    """Provide the two utils3d operations used by the pinned v2 infer path."""

    def intrinsics_from_focal_center(fx, fy, cx, cy):
        fx, fy, cx, cy = torch.broadcast_tensors(
            torch.as_tensor(fx),
            torch.as_tensor(fy),
            torch.as_tensor(cx, device=torch.as_tensor(fx).device),
            torch.as_tensor(cy, device=torch.as_tensor(fx).device),
        )
        matrix = torch.zeros((*fx.shape, 3, 3), dtype=fx.dtype, device=fx.device)
        matrix[..., 0, 0] = fx
        matrix[..., 1, 1] = fy
        matrix[..., 0, 2] = cx
        matrix[..., 1, 2] = cy
        matrix[..., 2, 2] = 1.0
        return matrix

    def depth_map_to_point_map(depth, *, intrinsics):
        height, width = depth.shape[-2:]
        u = (torch.arange(width, dtype=depth.dtype, device=depth.device) + 0.5) / width
        v = (torch.arange(height, dtype=depth.dtype, device=depth.device) + 0.5) / height
        u, v = torch.meshgrid(u, v, indexing="xy")
        while u.ndim < depth.ndim:
            u = u.unsqueeze(0)
            v = v.unsqueeze(0)
        x = (u - intrinsics[..., 0, 2, None, None]) / intrinsics[..., 0, 0, None, None]
        y = (v - intrinsics[..., 1, 2, None, None]) / intrinsics[..., 1, 1, None, None]
        return torch.stack((x * depth, y * depth, depth), dim=-1)

    shim = types.ModuleType("utils3d_moge")
    shim.pt = types.SimpleNamespace(
        intrinsics_from_focal_center=intrinsics_from_focal_center,
        depth_map_to_point_map=depth_map_to_point_map,
    )
    sys.modules["utils3d_moge"] = shim
    # The pinned geometry_numpy module imports cv2 at module import time, but
    # the v2 torch inference path used here never calls it.
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-tokens", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.num_tokens != 1800:
        parser.error("the qualified MoGe reference contract requires --num-tokens 1800")

    import torch
    from PIL import Image

    _install_utils3d_shim(torch)
    sys.path.insert(0, str(arguments.source_root.resolve()))
    from moge.model.v2 import MoGeModel

    checkpoint = torch.load(
        arguments.checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if set(checkpoint) != {"model_config", "model"}:
        raise ValueError("MoGe checkpoint has an unexpected top-level contract")
    model = MoGeModel(**checkpoint["model_config"])
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise ValueError(f"MoGe state mismatch: missing={missing}, unexpected={unexpected}")
    # Select the upstream deployment-compatible resize and positional
    # interpolation semantics matched by the native TensorRT graph.
    model.onnx_compatible_mode = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    image = np.asarray(Image.open(arguments.image).convert("RGB"), dtype=np.float32).copy()
    image_tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).to(device)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    with _math_sdpa(torch):
        output = model.infer(
            image_tensor,
            num_tokens=arguments.num_tokens,
            use_fp16=False,
            force_projection=True,
            apply_mask=True,
        )
    required = {"points", "depth", "intrinsics", "mask"}
    if set(output) != required:
        raise ValueError(f"MoGe reference returned {sorted(output)}, expected {sorted(required)}")

    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in output.items()}
    height, width = arrays["depth"].shape
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        points=arrays["points"].astype(np.float32, copy=False),
        depth=arrays["depth"].astype(np.float32, copy=False),
        intrinsics=arrays["intrinsics"].astype(np.float32, copy=False),
        mask=arrays["mask"].astype(np.uint8, copy=False),
        height=np.asarray(height, dtype=np.int32),
        width=np.asarray(width, dtype=np.int32),
    )
    print(json.dumps({"height": height, "width": width, "num_tokens": 1800}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
