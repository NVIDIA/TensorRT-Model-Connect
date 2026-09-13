# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the official YOLOX-s state dictionary without unpickling model code."""

from pathlib import Path

import numpy as np
import torch


class Checkpoint:
    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        if not isinstance(state, dict) or not state:
            raise ValueError("YOLOX checkpoint must contain a non-empty model state dictionary")
        self.state = state
        self.used: set[str] = set()
        for name, tensor in state.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError("YOLOX model state must contain named tensors")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"YOLOX checkpoint contains non-finite values: {name}")

    @classmethod
    def open(cls, model_dir: Path) -> "Checkpoint":
        archive = torch.load(model_dir / "yolox_s.pth", map_location="cpu", weights_only=True)
        if not isinstance(archive, dict) or "model" not in archive:
            raise ValueError("YOLOX checkpoint must contain a model state dictionary")
        return cls(archive["model"])

    def tensor(self, name: str) -> np.ndarray:
        if name not in self.state:
            raise ValueError(f"YOLOX checkpoint is missing {name}")
        self.used.add(name)
        return self.state[name].detach().float().numpy()

    def assert_consumed(self) -> None:
        unused = set(self.state) - self.used
        unused = {name for name in unused if not name.endswith(".bn.num_batches_tracked")}
        if unused:
            raise ValueError(f"Unsupported YOLOX-s checkpoint tensors: {sorted(unused)}")
