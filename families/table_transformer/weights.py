# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the exact published Table Transformer safetensors state dict."""

from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


class WeightDict(dict):
    """Owned arrays retained until engine serialization completes."""


def _open_checkpoint(model_dir: Path):
    return load_file(model_dir / "model.safetensors")


def _has_tensor(readers, name):
    return name in readers


def _load_tensor(readers, name):
    return readers[name].astype(np.float32)


def _target_np_dtype(precision):
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported Table Transformer precision: {precision}")
    return np.float16 if precision == "fp16" else np.float32
