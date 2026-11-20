# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read Muse source tensors once per use, retaining their packed storage formats."""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from .quantization import Projection


def canonical_module(name: str) -> str:
    for prefix in ("model.language_model.", "language_model.", "model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name.replace(".self_attn.output_gate_proj", ".self_attn.gate_proj")


class Checkpoint:
    """Family-local indexed safetensors reader; never calibrates or rewrites weights."""

    def __init__(self, model_dir: Path):
        self.path = Path(model_dir)
        descriptor = json.loads((self.path / "hf_quant_config.json").read_text(encoding="utf-8"))
        layers = descriptor.get("quantization", {}).get("quantized_layers")
        if not isinstance(layers, dict):
            raise ValueError("Muse requires an explicit per-module quantization descriptor")
        self.formats = {}
        for name, policy in layers.items():
            name = canonical_module(name)
            if name in self.formats:
                raise ValueError(f"Duplicate Muse quantization policy: {name}")
            if not isinstance(policy, dict) or policy.get("quant_algo") not in {"NVFP4", "MXFP8"}:
                raise ValueError(f"Unsupported Muse quantization policy: {name}")
            if policy["quant_algo"] == "NVFP4" and policy.get("group_size") != 16:
                raise ValueError("Muse NVFP4 uses groups of 16 input channels")
            self.formats[name] = policy["quant_algo"]
        index = json.loads((self.path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.index = index["weight_map"]
        self.tensor_names = {}
        for name in self.index:
            canonical = canonical_module(name)
            if canonical in self.tensor_names:
                raise ValueError(f"Duplicate canonical Muse tensor: {canonical}")
            self.tensor_names[canonical] = name
        self.stack = ExitStack()
        self.readers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.stack.__exit__(*args)

    def tensor(self, name: str):
        from safetensors import safe_open

        key = self.tensor_names.get(canonical_module(name))
        if key is None:
            raise KeyError(f"Muse checkpoint tensor missing: {name}")
        shard = self.index[key]
        if Path(shard).name != shard:
            raise ValueError("Muse weight index must refer to local shard filenames")
        if shard not in self.readers:
            self.readers[shard] = self.stack.enter_context(
                safe_open(self.path / shard, framework="pt", device="cpu")
            )
        return self.readers[shard].get_tensor(key)

    def vector(self, name: str, width: int) -> np.ndarray:
        value = self.tensor(name)
        if tuple(value.shape) != (width,):
            raise ValueError(f"Muse vector shape mismatch: {name}: {tuple(value.shape)}")
        result = np.ascontiguousarray(value.float().numpy())
        if not np.isfinite(result).all():
            raise ValueError(f"Muse vector contains nonfinite values: {name}")
        return result

    def projection(self, name: str, output_width: int, input_width: int) -> Projection:
        import torch

        name = canonical_module(name)
        algorithm = self.formats.get(name, "dense")
        tensor = self.tensor(name + ".weight")
        expected = (output_width, input_width // 2 if algorithm == "NVFP4" else input_width)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"Muse {name} weight shape {tuple(tensor.shape)} != {expected}")
        if algorithm == "dense":
            if tensor.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
                raise ValueError(f"Undeclared packed Muse projection: {name}")
            return Projection(algorithm, np.ascontiguousarray(tensor.float().numpy()),
                              output_width, input_width)
        if algorithm == "NVFP4":
            if input_width % 16 or tensor.dtype not in {torch.uint8, torch.int8}:
                raise ValueError(f"Muse {name} has invalid NVFP4 storage")
            blocks = self.tensor(name + ".weight_scale")
            if blocks.dtype != torch.float8_e4m3fn or tuple(blocks.shape) != (output_width, input_width // 16):
                raise ValueError(f"Muse {name} has invalid NVFP4 block scales")
            global_scale = self.tensor(name + ".weight_scale_2")
            if global_scale.numel() != 1:
                raise ValueError("Muse NVFP4 global weight scale must be scalar")
            activation = None if name == "embed_tokens" else self.tensor(name + ".input_scale")
            if activation is not None and activation.numel() != 1:
                raise ValueError("Muse NVFP4 input scale must be scalar")
            return Projection(algorithm, tensor.contiguous().view(torch.uint8).numpy(),
                              output_width, input_width, blocks.contiguous().view(torch.uint8).numpy(),
                              float(global_scale.item()), None if activation is None else float(activation.item()))
        blocks = self.tensor(name + ".weight_scale_inv")
        if input_width % 32 or tensor.dtype != torch.float8_e4m3fn or blocks.dtype != torch.uint8:
            raise ValueError(f"Muse {name} has invalid MXFP8 storage")
        if tuple(blocks.shape) != (output_width, input_width // 32) or bool((blocks == 255).any()):
            raise ValueError(f"Muse {name} has invalid E8M0 block scales")
        return Projection(algorithm, tensor.contiguous().view(torch.uint8).numpy(), output_width,
                          input_width, blocks.contiguous().numpy())
