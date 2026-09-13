# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read an official YOLOX state dictionary without unpickling model code."""

from pathlib import Path

import numpy as np
import torch

from .support import ARCHIVES


class Checkpoint:
    def __init__(self, state: dict[str, torch.Tensor], *, image_size: int = 640) -> None:
        if not isinstance(state, dict) or not state:
            raise ValueError("YOLOX checkpoint must contain a non-empty model state dictionary")
        self.state = state
        self.image_size = image_size
        self.used: set[str] = set()
        for name, tensor in state.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError("YOLOX model state must contain named tensors")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"YOLOX checkpoint contains non-finite values: {name}")

    @classmethod
    def open(cls, model_dir: Path) -> "Checkpoint":
        paths = [model_dir / name for name in ARCHIVES if (model_dir / name).is_file()]
        if len(paths) != 1:
            raise ValueError("YOLOX model directory must contain exactly one official checkpoint")
        path = paths[0]
        archive = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(archive, dict) or "model" not in archive:
            raise ValueError("YOLOX checkpoint must contain a model state dictionary")
        # Training/evaluation resolution is not stored in a state dictionary.
        image_size = 416 if path.name in {"yolox_nano.pth", "yolox_tiny.pth"} else 640
        return cls(archive["model"], image_size=image_size)

    def tensor(self, name: str) -> np.ndarray:
        if name not in self.state:
            raise ValueError(f"YOLOX checkpoint is missing {name}")
        self.used.add(name)
        return self.state[name].detach().float().numpy()

    def assert_consumed(self) -> None:
        unused = set(self.state) - self.used
        unused = {name for name in unused if not name.endswith(".bn.num_batches_tracked")}
        if unused:
            raise ValueError(f"Unsupported YOLOX checkpoint tensors: {sorted(unused)}")
