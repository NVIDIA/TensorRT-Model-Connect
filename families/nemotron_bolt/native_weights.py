# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read only the current layer's tensors; never instantiate a dense 74B checkpoint."""

import json
import math
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors import safe_open


class Weights:
    def __init__(self, root):
        self.root = Path(root)
        index = self.root / "model.safetensors.index.json"
        if index.is_file():
            self.index = json.loads(index.read_text())["weight_map"]
        else:
            single = self.root / "model.safetensors"
            with safe_open(single, framework="numpy") as reader:
                self.index = dict.fromkeys(reader.keys(), single.name)
        self.readers = {}

    def tensor(self, key):
        name = self.index[key]
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Unsafe Bolt shard name")
        path = self.root / name
        if name not in self.readers:
            # safe_open validates the file layout, offsets and non-overlap. Its numpy
            # get_tensor cannot read FP8 on NumPy versions without built-in float8.
            checked = safe_open(path, framework="numpy")
            with path.open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                if size > 100_000_000:
                    raise ValueError("Excessive safetensors header")
                metadata = json.loads(stream.read(size))
            self.readers[name] = (
                checked,
                metadata,
                size + 8,
                np.memmap(path, dtype=np.uint8, mode="r"),
            )
        checked, header, base, storage = self.readers[name]
        view = checked.get_slice(key)
        shape = tuple(view.get_shape())
        dtype = np.dtype(
            {
                "BF16": ml_dtypes.bfloat16,
                "F8_E4M3": ml_dtypes.float8_e4m3fn,
                "F32": "<f4",
                "F16": "<f2",
                "U8": "u1",
                "I32": "<i4",
            }[view.get_dtype()]
        )
        lo, hi = header[key]["data_offsets"]
        if sys.byteorder != "little" or not (0 <= lo <= hi <= storage.size - base):
            raise ValueError(f"Invalid tensor storage for {key}")
        if hi - lo != math.prod(shape) * dtype.itemsize:
            raise ValueError(f"Invalid tensor size for {key}")
        return storage[base + lo : base + hi].view(dtype).reshape(shape)

    __getitem__ = tensor

    def f32(self, key):
        a = self.tensor(key)
        if a.dtype not in (np.float32, np.float16, ml_dtypes.bfloat16):
            raise ValueError(f"Expected original floating weights for {key}")
        return np.ascontiguousarray(a, dtype=np.float32)

    def experts(self, prefix, count):
        return {
            "gate": self.f32(prefix + "gate.weight"),
            "correction": self.f32(prefix + "gate.e_score_correction_bias"),
            "latent_in": self.tensor(prefix + "fc1_latent_proj.weight"),
            "latent_out": self.tensor(prefix + "fc2_latent_proj.weight"),
            **{
                role: np.stack(
                    [self.tensor(f"{prefix}experts.{i}.{role}_proj.weight") for i in range(count)]
                )
                for role in ("up", "down")
            },
            **{
                "shared_" + role: self.tensor(f"{prefix}shared_experts.{role}_proj.weight")
                for role in ("up", "down")
            },
        }
