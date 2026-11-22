# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read D3 safetensors without executing checkpoint code."""
from contextlib import ExitStack
import json
from pathlib import Path
import numpy as np
import ml_dtypes  # noqa: F401 -- registers BF16 for NumPy safetensors readers
from safetensors import safe_open


def _open_safetensors(root: Path, **_unused):
    index = root / "model.safetensors.index.json"
    files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    stack = ExitStack()
    try:
        readers = [stack.enter_context(safe_open(root / f, framework="numpy")) for f in files]
    except BaseException:
        stack.close()
        raise
    return stack, readers


def _load_tensor(readers, key):
    for reader in readers:
        if key in reader.keys():
            if reader.get_slice(key).get_dtype() not in {"BF16", "F32"}:
                raise ValueError(f"D3 requires BF16/FP32 source tensors: {key}")
            value = reader.get_tensor(key).astype(np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite D3 source tensor: {key}")
            return value
    raise ValueError(f"missing D3 tensor: {key}")


def load_projection(readers, name):
    return np.ascontiguousarray(_load_tensor(readers, name + ".weight").T)
