# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact Cosmos3-Nano checkpoint loading."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def transformer_safetensor_paths(component_dir: str | Path) -> tuple[Path, ...]:
    """Resolve the seven official transformer shards from their exact index."""

    root = Path(component_dir)
    index_path = root / "diffusion_pytorch_model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Cosmos3 transformer index is missing: {index_path}")
    weight_map = read_json(index_path).get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Cosmos3 transformer index has no weight_map: {index_path}")
    raw_names = tuple(weight_map.values())
    if not all(isinstance(name, str) and name for name in raw_names):
        raise ValueError(f"Cosmos3 transformer index has invalid shard names: {index_path}")
    names = sorted(set(raw_names))
    expected_names = [
        f"diffusion_pytorch_model-{index:05d}-of-00007.safetensors" for index in range(1, 8)
    ]
    if names != expected_names:
        raise ValueError("Cosmos3 transformer index must reference the seven official shards")
    paths = tuple(root / name for name in names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cosmos3 transformer shards are missing: " + ", ".join(missing))
    return paths


def iter_transformer_tensors(component_dir: str | Path) -> Iterator[tuple[str, Any]]:
    """Stream the official transformer checkpoint shard by shard on CPU."""

    from safetensors import safe_open

    seen: set[str] = set()
    for path in transformer_safetensor_paths(component_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in seen:
                    raise ValueError(f"Duplicate Cosmos3 tensor {key!r}")
                seen.add(key)
                yield key, handle.get_tensor(key)


def load_transformer_state_dict(component_dir: str | Path) -> dict[str, Any]:
    return dict(iter_transformer_tensors(component_dir))


def load_vae_decoder_weights(component_dir: str | Path) -> dict[str, Any]:
    """Load decoder tensors from the exact official VAE file."""

    from safetensors import safe_open

    root = Path(component_dir)
    weights_path = root / "diffusion_pytorch_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Cosmos3 VAE weights are missing: {weights_path}")
    selected: dict[str, Any] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if name.startswith("decoder.") or name.startswith("post_quant_conv."):
                selected[name] = handle.get_tensor(name)
    if not selected:
        raise ValueError(f"Cosmos3 VAE contains no decoder tensors: {weights_path}")

    config = read_json(root / "config.json")
    for field in ("latents_mean", "latents_std"):
        values = config.get(field)
        if not isinstance(values, list) or len(values) != 48:
            raise ValueError(f"Cosmos3 VAE config requires 48-value {field}")
        selected[f"_{field}"] = values
    return selected
