# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Depth Anything V2 safetensors loading helpers.

The runtime graph consumes explicit NumPy arrays and never creates or imports
an ONNX graph. Depth Anything V2 publishes a standard Transformers
`DepthAnythingForDepthEstimation` checkpoint: a `backbone.*` DINOv2 encoder, a
`neck.*` DPT reassemble/fusion stage, and a `head.*` depth-prediction head, all
in one `model.safetensors` for the small and base sizes. The sharded-index
fallback below exists for the larger published sizes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


class WeightDict(dict[str, np.ndarray]):
    """Family-owned logical weight map."""


class _Readers(list):
    def __init__(self, readers: list, tensor_map: dict[str, object]):
        super().__init__(readers)
        self.tensor_map = tensor_map


def open_checkpoint(model_dir: str | Path) -> _Readers:
    root = Path(model_dir)
    single = root / "model.safetensors"
    if single.is_file():
        reader = safe_open(str(single), framework="pt", device="cpu")
        return _Readers([reader], {name: reader for name in reader.keys()})

    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Depth Anything V2 requires model.safetensors or "
            f"model.safetensors.index.json in {root}"
        )
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map", {})
    by_file = {
        shard: safe_open(str(root / shard), framework="pt", device="cpu")
        for shard in sorted(set(weight_map.values()))
    }
    return _Readers(
        list(by_file.values()),
        {name: by_file[shard] for name, shard in weight_map.items()},
    )


def has_tensor(readers: _Readers, name: str) -> bool:
    return name in readers.tensor_map


def load_tensor(readers: _Readers, name: str) -> np.ndarray:
    reader = readers.tensor_map.get(name)
    if reader is None:
        raise KeyError(f"Depth Anything V2 tensor not found: {name}")
    return reader.get_tensor(name).detach().float().numpy()


def encoder_layer_count(readers: _Readers) -> int:
    """Count `backbone.encoder.layer.<n>` blocks present in the checkpoint."""

    indices = set()
    for name in readers.tensor_map:
        prefix = "backbone.encoder.layer."
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix) :]
        index = rest.split(".", 1)[0]
        if index.isdigit():
            indices.add(int(index))
    if not indices:
        raise ValueError("Depth Anything V2 checkpoint has no backbone encoder layers")
    if sorted(indices) != list(range(len(indices))):
        raise ValueError(f"Depth Anything V2 backbone layer indices are not contiguous: {sorted(indices)}")
    return len(indices)
