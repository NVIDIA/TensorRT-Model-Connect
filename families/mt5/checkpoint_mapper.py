# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the published mT5 checkpoint without rewriting its weights."""

import numpy as np


class WeightDict(dict):
    pass


def _open_torch_checkpoint(directory):
    """Read the pinned checkpoint's published PyTorch state dictionary."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("mT5 checkpoint loading requires torch") from exc
    from packaging.version import Version

    if Version(torch.__version__.split("+")[0]) < Version("2.10.0"):
        raise RuntimeError("mT5 checkpoint loading requires torch>=2.10.0 (GHSA-63cw-57p8-fm3p)")
    return {
        name: value.float().numpy()
        for name, value in torch.load(
            directory / "pytorch_model.bin", map_location="cpu", weights_only=True
        ).items()
    }


def _load_tensor(weights, name):
    return np.asarray(weights[name], dtype=np.float32)


def _has_tensor(weights, name):
    return name in weights


def _transpose_2d(value, name):
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank 2")
    return np.ascontiguousarray(value.T, dtype=np.float32)
