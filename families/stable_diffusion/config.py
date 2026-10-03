# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the Stable Diffusion component configs this family builds against."""

from __future__ import annotations

import json
from pathlib import Path

# diffusers supplies these when a component config stays silent.
_CLIP_LAYER_NORM_EPS = 1e-5
_VAE_SCALING_FACTOR = 0.18215


def _read(model_dir: Path, component: str) -> dict:
    path = Path(model_dir) / component / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {component} config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(model_dir: str | Path, *, latent_size: int) -> dict:
    """The subset of the three component configs the builders need."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing model_index.json: {index_path}")

    unet = _read(model_dir, "unet")
    vae = _read(model_dir, "vae")
    text = _read(model_dir, "text_encoder")

    if int(unet["in_channels"]) != int(vae["latent_channels"]):
        raise ValueError("Stable Diffusion unet and vae disagree on the latent width")

    scheduler_path = model_dir / "scheduler" / "scheduler_config.json"
    scheduler = json.loads(scheduler_path.read_text(encoding="utf-8")) if scheduler_path.is_file() else {}

    return {
        "latent_size": int(latent_size),
        "image_size": int(latent_size) * 8,
        "unet": {
            "in_channels": int(unet["in_channels"]),
            "sample_size": int(latent_size),
            "block_out_channels": [int(v) for v in unet["block_out_channels"]],
            "layers_per_block": int(unet["layers_per_block"]),
            "attention_head_dim": int(unet["attention_head_dim"]),
            "cross_attention_dim": int(unet["cross_attention_dim"]),
            "norm_num_groups": int(unet.get("norm_num_groups", 32)),
            "context_length": int(text["max_position_embeddings"]),
        },
        "vae": {
            "block_out_channels": [int(v) for v in vae["block_out_channels"]],
            "layers_per_block": int(vae["layers_per_block"]),
            "norm_num_groups": int(vae.get("norm_num_groups", 32)),
            "latent_channels": int(vae["latent_channels"]),
        },
        "text_encoder": {
            "hidden_size": int(text["hidden_size"]),
            "num_hidden_layers": int(text["num_hidden_layers"]),
            "num_attention_heads": int(text["num_attention_heads"]),
            "max_position_embeddings": int(text["max_position_embeddings"]),
            "layer_norm_eps": float(text.get("layer_norm_eps") or _CLIP_LAYER_NORM_EPS),
        },
        "scaling_factor": float(vae.get("scaling_factor") or _VAE_SCALING_FACTOR),
        "num_train_timesteps": int(scheduler.get("num_train_timesteps", 1000)),
        "beta_start": float(scheduler.get("beta_start", 0.00085)),
        "beta_end": float(scheduler.get("beta_end", 0.012)),
        "beta_schedule": str(scheduler.get("beta_schedule", "scaled_linear")),
        "steps_offset": int(scheduler.get("steps_offset", 1)),
    }
