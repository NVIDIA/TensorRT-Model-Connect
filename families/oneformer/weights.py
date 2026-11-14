# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the OneFormer family's exact pytorch_model.bin checkpoint.

This family requires torch in the build environment to deserialize the
Hugging Face PyTorch state dict (loaded with ``weights_only=True``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Weights:
    """Name-checked access to the checkpoint tensors as NumPy arrays."""

    def __init__(self, state: dict):
        self._state = state
        self._used: set[str] = set()

    def __contains__(self, name: str) -> bool:
        return name in self._state

    def names(self) -> list[str]:
        return list(self._state)

    def get(self, name: str, dtype=np.float32) -> np.ndarray:
        tensor = self._state.get(name)
        if tensor is None:
            raise KeyError(f"oneformer checkpoint is missing tensor {name}")
        self._used.add(name)
        if dtype is None:
            return tensor.detach().cpu().numpy()
        return tensor.detach().cpu().float().numpy().astype(dtype, copy=False)

    def unused(self) -> list[str]:
        return sorted(set(self._state) - self._used)


def _open_torch_checkpoint(model_dir: Path) -> Weights:
    """Open the family's required pytorch_model.bin checkpoint."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("oneformer requires torch in the build environment") from error

    path = Path(model_dir) / "pytorch_model.bin"
    if not path.is_file():
        raise FileNotFoundError(f"oneformer requires the PyTorch checkpoint: {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"oneformer checkpoint is not a state dict: {path}")
    return Weights(state)
