# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and validate the OpenAI GPT safetensors checkpoint.

The published checkpoint stores the bare ``OpenAIGPTModel`` names (``tokens_embed``,
``positions_embed``, ``h.<i>.*``) in float32; the LM head is the token embedding.
Projections keep the Hugging Face ``Conv1D`` layout ``[in, out]`` (``y = x @ W + b``).
Every tensor's name, dtype and shape is checked in the file header before any data
is read, every tensor is consumed, and no value may be NaN or infinite.

``h.<i>.attn.bias`` is the stored copy of a causal-mask buffer that transformers
recomputes and never loads; it is the one stored tensor this family does not read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import OpenAIGPTConfig

_STORED_DTYPE = "F32"


@dataclass(frozen=True)
class LayerWeights:
    attn_weight: np.ndarray  # c_attn [n_embd, 3 * n_embd]: query, key, value columns
    attn_bias: np.ndarray  # [3 * n_embd]
    attn_proj_weight: np.ndarray  # [n_embd, n_embd]
    attn_proj_bias: np.ndarray  # [n_embd]
    ln_1_weight: np.ndarray  # [n_embd]
    ln_1_bias: np.ndarray
    fc_weight: np.ndarray  # [n_embd, 4 * n_embd]
    fc_bias: np.ndarray  # [4 * n_embd]
    mlp_proj_weight: np.ndarray  # [4 * n_embd, n_embd]
    mlp_proj_bias: np.ndarray  # [n_embd]
    ln_2_weight: np.ndarray  # [n_embd]
    ln_2_bias: np.ndarray


@dataclass(frozen=True)
class OpenAIGPTWeights:
    tokens_embed: np.ndarray  # [vocab_size, n_embd], also the LM head
    positions_embed: np.ndarray  # [n_positions, n_embd]
    layers: tuple[LayerWeights, ...]


def expected_shapes(config: OpenAIGPTConfig) -> dict[str, tuple[int, ...]]:
    """Every tensor the graph reads, with the shape ``config.json`` requires."""
    embd, inner = config.n_embd, config.inner_size
    shapes = {
        "tokens_embed.weight": (config.vocab_size, embd),
        "positions_embed.weight": (config.n_positions, embd),
    }
    for index in range(config.n_layer):
        prefix = f"h.{index}"
        shapes.update(
            {
                f"{prefix}.attn.c_attn.weight": (embd, 3 * embd),
                f"{prefix}.attn.c_attn.bias": (3 * embd,),
                f"{prefix}.attn.c_proj.weight": (embd, embd),
                f"{prefix}.attn.c_proj.bias": (embd,),
                f"{prefix}.ln_1.weight": (embd,),
                f"{prefix}.ln_1.bias": (embd,),
                f"{prefix}.mlp.c_fc.weight": (embd, inner),
                f"{prefix}.mlp.c_fc.bias": (inner,),
                f"{prefix}.mlp.c_proj.weight": (inner, embd),
                f"{prefix}.mlp.c_proj.bias": (embd,),
                f"{prefix}.ln_2.weight": (embd,),
                f"{prefix}.ln_2.bias": (embd,),
            }
        )
    return shapes


def _mask_buffers(config: OpenAIGPTConfig) -> set[str]:
    return {f"h.{index}.attn.bias" for index in range(config.n_layer)}


def check_header(names_to_header: dict[str, tuple[str, tuple[int, ...]]], config: OpenAIGPTConfig) -> None:
    """Validate the stored names, dtypes and shapes against ``config`` before reading data."""
    # Two embeddings and twelve tensors per layer: the file must hold that many before a name is
    # built for each declared layer. Extra tensors are named by the check below.
    count, layers = len(names_to_header), config.n_layer
    if count < 2 + 12 * layers:
        raise ValueError(
            f"OpenAI GPT checkpoint has {count} tensors; n_layer={layers} needs {2 + 12 * layers}"
        )
    expected = expected_shapes(config)
    missing = sorted(set(expected) - set(names_to_header))
    if missing:
        raise ValueError("OpenAI GPT checkpoint is missing tensors: " + ", ".join(missing[:8]))
    unused = sorted(set(names_to_header) - set(expected) - _mask_buffers(config))
    if unused:
        raise ValueError(
            "OpenAI GPT checkpoint has tensors this graph does not use: " + ", ".join(unused[:8])
        )
    for name, shape in expected.items():
        dtype, stored = names_to_header[name]
        if dtype != _STORED_DTYPE:
            raise ValueError(f"OpenAI GPT tensor {name} must be stored as F32, got {dtype}")
        if stored != shape:
            raise ValueError(f"OpenAI GPT tensor {name} must have shape {shape}, got {stored}")


def load_weights(model_dir: Path, config: OpenAIGPTConfig) -> OpenAIGPTWeights:
    """Read every tensor of the graph from ``model.safetensors``."""
    from safetensors import safe_open

    path = model_dir / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"OpenAI GPT needs model.safetensors in {model_dir}")
    with safe_open(str(path), framework="numpy") as reader:
        header = {}
        for name in reader.keys():
            view = reader.get_slice(name)
            header[name] = (view.get_dtype(), tuple(view.get_shape()))
        check_header(header, config)

        def tensor(name: str) -> np.ndarray:
            value = np.ascontiguousarray(reader.get_tensor(name), dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"OpenAI GPT tensor {name} holds NaN or infinite values")
            return value

        layers = []
        for index in range(config.n_layer):
            prefix = f"h.{index}"
            layers.append(
                LayerWeights(
                    attn_weight=tensor(f"{prefix}.attn.c_attn.weight"),
                    attn_bias=tensor(f"{prefix}.attn.c_attn.bias"),
                    attn_proj_weight=tensor(f"{prefix}.attn.c_proj.weight"),
                    attn_proj_bias=tensor(f"{prefix}.attn.c_proj.bias"),
                    ln_1_weight=tensor(f"{prefix}.ln_1.weight"),
                    ln_1_bias=tensor(f"{prefix}.ln_1.bias"),
                    fc_weight=tensor(f"{prefix}.mlp.c_fc.weight"),
                    fc_bias=tensor(f"{prefix}.mlp.c_fc.bias"),
                    mlp_proj_weight=tensor(f"{prefix}.mlp.c_proj.weight"),
                    mlp_proj_bias=tensor(f"{prefix}.mlp.c_proj.bias"),
                    ln_2_weight=tensor(f"{prefix}.ln_2.weight"),
                    ln_2_bias=tensor(f"{prefix}.ln_2.bias"),
                )
            )
        return OpenAIGPTWeights(
            tokens_embed=tensor("tokens_embed.weight"),
            positions_embed=tensor("positions_embed.weight"),
            layers=tuple(layers),
        )
