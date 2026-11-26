# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the tensors of a DPT checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CHECKPOINT = "pytorch_model.bin"
# Before 2.10.0, torch.load(weights_only=True) can run code from a crafted file
# (CVE-2026-24747, GHSA-63cw-57p8-fm3p).
MIN_TORCH = "2.10.0"


def tensor_arrays(state, source: str) -> dict[str, np.ndarray]:
    """Float32 copies of a state dict's tensors.

    Every entry must be a named floating-point tensor whose values stay finite in
    float32: integer, boolean or complex tensors are not weights of this graph, and a
    NaN or infinity (also one that the conversion from float64 creates) would reach
    the engine.
    """
    import torch

    if not isinstance(state, dict):
        raise ValueError(f"{source} does not hold a state dict")
    arrays = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError(f"{source} holds an entry that is not a named tensor: {name!r}")
        if not value.is_floating_point():
            raise ValueError(f"{source}: tensor {name} is {value.dtype}, not floating point")
        array = value.detach().to(torch.float32).numpy()
        if not np.isfinite(array).all():
            raise ValueError(f"{source}: tensor {name} holds a value that is not finite in float32")
        arrays[name] = array
    return arrays


def as_engine_weights(values, dtype) -> np.ndarray:
    """``values`` as the engine stores them; a value that does not fit that dtype is refused.

    A finite float32 weight can still overflow fp16 (100000 becomes infinity).
    """
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.ascontiguousarray(values, dtype=dtype)
    if not np.isfinite(array).all():
        raise ValueError(f"a DPT weight is not finite in {np.dtype(dtype).name}; build fp32")
    return array


def load_tensors(model_dir: str | Path) -> dict[str, np.ndarray]:
    """The checkpoint's state dict as float32 arrays.

    The published repository ships only a PyTorch pickle. ``weights_only=True``
    restricts unpickling to tensors and plain containers.
    """
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("dpt requires torch in the build environment") from error
    from packaging.version import Version

    installed = str(torch.__version__).split("+")[0]
    if Version(installed) < Version(MIN_TORCH):
        raise RuntimeError(f"dpt reads checkpoints with torch {MIN_TORCH} or later; found {installed}")
    path = Path(model_dir) / CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(f"dpt requires the checkpoint file {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return tensor_arrays(state, str(path))
