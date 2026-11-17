# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a component's safetensors weights as NumPy arrays in their stored dtype."""

from __future__ import annotations

import json
from pathlib import Path

# Registers NumPy bfloat16, which safetensors needs to return BF16 tensors.
import ml_dtypes
import numpy as np
from safetensors import safe_open

BF16 = np.dtype(ml_dtypes.bfloat16)


class ComponentWeights:
    """Lazy, read-only access to one Diffusers / Transformers component directory."""

    def __init__(self, component_dir: str | Path, prefix: str = "") -> None:
        root = Path(component_dir)
        self._prefix = prefix
        self._handles: dict[Path, object] = {}
        index = next(
            (
                root / name
                for name in (
                    "diffusion_pytorch_model.safetensors.index.json",
                    "model.safetensors.index.json",
                )
                if (root / name).is_file()
            ),
            None,
        )
        if index is not None:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"flux3: {index} has no weight_map")
            self._files = {name: root / shard for name, shard in weight_map.items()}
        else:
            single = next(
                (
                    root / name
                    for name in ("diffusion_pytorch_model.safetensors", "model.safetensors")
                    if (root / name).is_file()
                ),
                None,
            )
            if single is None:
                raise FileNotFoundError(f"flux3: no safetensors weights in {root}")
            with safe_open(str(single), framework="numpy") as handle:
                names = list(handle.keys())
            self._files = {name: single for name in names}

    def _handle(self, path: Path):
        if path not in self._handles:
            self._handles[path] = safe_open(str(path), framework="numpy")
        return self._handles[path]

    def has(self, name: str) -> bool:
        return self._prefix + name in self._files

    def get(self, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
        """One tensor in its stored dtype; ``shape`` is checked when given."""

        full = self._prefix + name
        path = self._files.get(full)
        if path is None:
            raise KeyError(f"flux3: missing weight {full}")
        array = np.ascontiguousarray(self._handle(path).get_tensor(full))
        if shape is not None and tuple(array.shape) != tuple(shape):
            raise ValueError(f"flux3: {full} has shape {array.shape}, expected {shape}")
        return array

    def names(self) -> list[str]:
        return sorted(
            name[len(self._prefix) :] for name in self._files if name.startswith(self._prefix)
        )
