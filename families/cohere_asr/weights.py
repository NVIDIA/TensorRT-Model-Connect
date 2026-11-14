# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One explicit safetensors checkpoint format, with original BF16 values expanded to FP32."""

from pathlib import Path
import numpy as np
from safetensors import safe_open


class Weights:
    def __init__(self, directory: Path):
        self.reader = safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu")
        self.names = set(self.reader.keys())
        self.cache = {}

    def __contains__(self, name):
        return name in self.names

    def __getitem__(self, name):
        if name not in self.cache:
            self.cache[name] = (
                self.reader.get_tensor(name).float().numpy().astype(np.float32, copy=False)
            )
        return self.cache[name]
