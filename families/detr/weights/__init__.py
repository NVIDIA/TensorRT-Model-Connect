# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the DETR family's exact pytorch_model.bin checkpoint.

This family requires torch in the build environment to deserialize the
Hugging Face PyTorch state dict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class WeightDict(dict):
    """Concrete mapping returned by this family's checkpoint loader."""


def _target_np_dtype(precision: str) -> np.dtype:
    if precision in ("fp16", "bf16"):
        return np.float16
    if precision == "fp32":
        return np.float32
    raise ValueError(f"Unsupported detr precision: {precision}")


class _TorchCheckpointReader:
    """One exact PyTorch state-dict reader."""

    def __init__(self, state: dict):
        self._state = state

    def keys(self):
        return list(self._state)

    def get_tensor(self, name):
        return self._state[name]


class _ReaderCollection(list):
    """Checkpoint readers with one exact tensor-to-reader index."""

    def __init__(self, readers: list, *, tensor_map: dict[str, object] | None = None):
        super().__init__(readers)
        if tensor_map is None:
            tensor_map = {name: reader for reader in readers for name in reader.keys()}
        self.tensor_map = tensor_map


def _open_torch_checkpoint(model_dir: Path) -> _ReaderCollection:
    """Open the family's required pytorch_model.bin checkpoint."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "This family requires torch in the build environment"
        ) from error

    path = model_dir / "pytorch_model.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Required PyTorch checkpoint is missing: {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return _ReaderCollection([_TorchCheckpointReader(state)])


def _has_tensor(readers: _ReaderCollection, name: str) -> bool:
    return name in readers.tensor_map


def _to_numpy_fp32(tensor) -> np.ndarray:
    """Copy a required CPU Torch checkpoint tensor to NumPy float32."""
    return tensor.detach().float().cpu().numpy()


def _load_tensor(readers: _ReaderCollection, name: str) -> np.ndarray:
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Tensor not found: {name}")
    return _to_numpy_fp32(reader.get_tensor(name))
