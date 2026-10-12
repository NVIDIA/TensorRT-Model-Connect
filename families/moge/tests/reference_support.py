# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned pinned MoGe-2 reference loading and inference."""

from __future__ import annotations

import contextlib
from pathlib import Path
import sys
import types

import numpy as np


def _math_sdpa(torch):
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([SDPBackend.MATH])
    except (ImportError, AttributeError):
        return contextlib.nullcontext()


def _install_utils3d_shim(torch) -> None:
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
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))


class OfficialReference:
    def __init__(self, source_root: Path, checkpoint: Path):
        import torch

        _install_utils3d_shim(torch)
        source_root = source_root.resolve()
        sys.path.insert(0, str(source_root))
        from moge.model.v2 import MoGeModel

        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if set(state) != {"model_config", "model"}:
            raise ValueError("MoGe checkpoint has an unexpected top-level contract")
        model = MoGeModel(**state["model_config"])
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        if missing or unexpected:
            raise ValueError(f"MoGe state mismatch: missing={missing}, unexpected={unexpected}")
        model.onnx_compatible_mode = True
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().float().to(self.device)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def infer(self, pixels: np.ndarray, *, num_tokens: int, fov_x: float | None = None):
        if num_tokens != 1800:
            raise ValueError("the qualified MoGe reference requires num_tokens=1800")
        tensor = self.torch.from_numpy(pixels).permute(2, 0, 1).to(self.device)
        options = {
            "num_tokens": num_tokens,
            "use_fp16": False,
            "force_projection": True,
            "apply_mask": True,
        }
        if fov_x is not None:
            options["fov_x"] = fov_x
        with _math_sdpa(self.torch):
            output = self.model.infer(tensor, **options)
        required = {"points", "depth", "intrinsics", "mask"}
        if set(output) != required:
            raise ValueError(
                f"MoGe reference returned {sorted(output)}, expected {sorted(required)}"
            )
        return {name: tensor.detach().cpu().numpy() for name, tensor in output.items()}


def read_image(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
