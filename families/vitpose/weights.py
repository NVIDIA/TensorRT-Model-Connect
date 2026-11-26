# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint's tensors: the header first, so the configuration is checked before any weight is read."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as config_module
from .config import VitPoseConfig


def _weights_path(model_dir: Path) -> Path:
    path = Path(model_dir) / config_module.WEIGHTS_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"vitpose checkpoint is missing {config_module.WEIGHTS_FILE}: {path}"
        )
    return path


def read_header(model_dir: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Name -> (safetensors dtype, shape) of every stored tensor, without reading tensor data."""

    from safetensors import safe_open

    header = {}
    with safe_open(str(_weights_path(model_dir)), framework="np") as handle:
        for name in handle.keys():
            tensor = handle.get_slice(name)
            header[name] = (tensor.get_dtype(), tuple(int(value) for value in tensor.get_shape()))
    return header


def load(model_dir: Path, cfg: VitPoseConfig) -> dict[str, np.ndarray]:
    """The float32 arrays of a checkpoint whose header `config.resolve` accepted for `cfg`."""

    from safetensors import safe_open

    header = read_header(model_dir)
    config_module.check_tensors(cfg, header)
    weights = {}
    with safe_open(str(_weights_path(model_dir)), framework="np") as handle:
        for name in header:
            array = handle.get_tensor(name)
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"vitpose {name} must hold finite float32 values")
            weights[name] = array
    return weights
