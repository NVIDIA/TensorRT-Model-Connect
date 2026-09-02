# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-Nano Diffusers checkpoint discovery and tensor loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def component_safetensor_paths(component_dir: str | Path) -> tuple[Path, ...]:
    """Resolve one Diffusers component's sharded safetensors deterministically."""

    root = Path(component_dir)
    index_candidates = (
        root / "diffusion_pytorch_model.safetensors.index.json",
        root / "model.safetensors.index.json",
    )
    for index_path in index_candidates:
        if index_path.is_file():
            weight_map = read_json(index_path).get("weight_map", {})
            paths = tuple(root / name for name in sorted(set(weight_map.values())))
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing checkpoint shards: " + ", ".join(missing))
            return paths

    paths = tuple(
        sorted(
            {
                *root.glob("diffusion_pytorch_model*.safetensors"),
                *root.glob("model*.safetensors"),
            }
        )
    )
    if not paths:
        raise FileNotFoundError(f"No safetensor weights found in {root}")
    return paths


def iter_component_tensors(component_dir: str | Path) -> Iterator[tuple[str, Any]]:
    """Stream CPU tensors shard-by-shard without duplicating the full checkpoint."""

    from safetensors import safe_open

    seen: set[str] = set()
    for path in component_safetensor_paths(component_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in seen:
                    raise ValueError(f"Duplicate Cosmos3 tensor {key!r}")
                seen.add(key)
                yield key, handle.get_tensor(key)


def load_component_state_dict(component_dir: str | Path) -> dict[str, Any]:
    """Materialize a component state dict when a builder needs all weights."""

    return dict(iter_component_tensors(component_dir))


def load_vae_decoder_weights(component_dir: str | Path) -> dict[str, Any]:
    """Load only recurrent decoder tensors and normalization metadata."""

    root = Path(component_dir)
    config = read_json(root / "config.json")
    state = load_component_state_dict(root)
    selected = {
        name: value
        for name, value in state.items()
        if name.startswith("decoder.") or name.startswith("post_quant_conv.")
    }
    for field in ("latents_mean", "latents_std"):
        values = config.get(field)
        if not isinstance(values, list) or len(values) != 48:
            raise ValueError(f"Cosmos3 VAE config requires 48-value {field}")
        selected[f"_{field}"] = values
    return selected
