# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy, contained access to the HunYuanVL safetensors checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import load_json

_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})


def _contained(root: Path, name: str) -> Path:
    relative = Path(name)
    if (
        not name
        or relative.is_absolute()
        or relative.as_posix() != name
        or "\\" in name
        or "\0" in name
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"HunYuanVL shard name is not a plain relative path: {name!r}")
    path = root / relative
    target = path.resolve()
    # A Hub snapshot links each file to its repository's blobs/ directory.
    blobs = root.parent.parent / "blobs" if root.parent.name == "snapshots" else None
    if not target.is_file() or not (
        target.is_relative_to(root)
        or (blobs is not None and target.is_relative_to(blobs.resolve()))
    ):
        raise ValueError(f"HunYuanVL missing or non-contained checkpoint shard: {name}")
    return target


class CheckpointWeights:
    """Tensor lookup by checkpoint key; values are returned as FP32 NumPy arrays."""

    def __init__(self, model_dir: Path):
        self.root = Path(model_dir).resolve()
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            mapping = load_json(index).get("weight_map")
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("HunYuanVL safetensors index needs a nonempty weight_map")
        else:
            from safetensors import safe_open

            path = _contained(self.root, "model.safetensors")
            with safe_open(path, framework="pt", device="cpu") as stream:
                mapping = {key: "model.safetensors" for key in stream.keys()}
        self.paths = {}
        for key, shard in mapping.items():
            if not isinstance(key, str) or not isinstance(shard, str):
                raise ValueError("HunYuanVL weight_map entries must be strings")
            if shard not in self.paths:
                self.paths[shard] = _contained(self.root, shard)
        self.mapping = mapping

    def check(self, shapes: dict, optional=frozenset()) -> None:
        """Require every tensor in ``shapes`` with that shape and a floating dtype, from the
        safetensors headers alone; keys in ``optional`` may be absent."""
        from safetensors import safe_open

        by_shard = {}
        for key in shapes:
            if key in self.mapping:
                by_shard.setdefault(self.mapping[key], []).append(key)
            elif key not in optional:
                raise ValueError(f"HunYuanVL checkpoint has no tensor {key!r}")
        for shard, keys in by_shard.items():
            with safe_open(self.paths[shard], framework="pt", device="cpu") as stream:
                for key in keys:
                    piece = stream.get_slice(key)
                    if piece.get_dtype() not in _FLOAT_DTYPES:
                        raise ValueError(f"HunYuanVL tensor {key!r} is {piece.get_dtype()}")
                    if tuple(piece.get_shape()) != tuple(shapes[key]):
                        raise ValueError(
                            f"HunYuanVL tensor {key!r} has shape {tuple(piece.get_shape())}, "
                            f"the configuration needs {tuple(shapes[key])}"
                        )

    def __contains__(self, key: str) -> bool:
        return key in self.mapping

    def __getitem__(self, key: str) -> np.ndarray:
        from safetensors import safe_open

        if key not in self.mapping:
            raise KeyError(f"HunYuanVL checkpoint has no tensor {key!r}")
        # BF16 widens exactly to FP32; the graph owns the computation dtype.
        with safe_open(self.paths[self.mapping[key]], framework="pt", device="cpu") as stream:
            return stream.get_tensor(key).float().numpy()
