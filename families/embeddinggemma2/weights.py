# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np


def load_weights(model_dir, config):
    from safetensors import safe_open
    import torch

    directory = Path(model_dir)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        names = {mapping.get("language_model." + key) for key in config.weight_shapes()}
        if None in names:
            missing = [
                key for key in config.weight_shapes() if "language_model." + key not in mapping
            ]
            raise ValueError(f"EmbeddingGemma2 missing text weights: {missing}")
        files = [directory / name for name in sorted(names)]
    else:
        files = [directory / "model.safetensors"]
    with ExitStack() as stack:
        readers = [
            stack.enter_context(safe_open(path, framework="pt", device="cpu")) for path in files
        ]
        locations = {key: reader for reader in readers for key in reader.keys()}
        result = {}
        for key, shape in config.weight_shapes().items():
            full_key = "language_model." + key
            if full_key not in locations:
                raise ValueError(f"EmbeddingGemma2 missing text weight: {full_key}")
            tensor = locations[full_key].get_tensor(full_key)
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"EmbeddingGemma2 {full_key} has shape {tuple(tensor.shape)}, expected {shape}"
                )
            result[key] = np.ascontiguousarray(tensor.to(torch.float32).numpy())
        return result
