# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the Stable Diffusion component configs this family builds against."""

from __future__ import annotations

import json
from pathlib import Path

# diffusers supplies these when a component config stays silent.
_CLIP_LAYER_NORM_EPS = 1e-5
_VAE_SCALING_FACTOR = 0.18215
# (original_h, original_w, crop_top, crop_left, target_h, target_w)
_ADDITION_TIME_IDS = 6


def _heads(unet: dict):
    """Head counts per level.

    The field is named ``attention_head_dim`` but diffusers reads it as the
    number of heads, falling back to it when ``num_attention_heads`` is unset.
    SD 1.5 states one number, SDXL states one per level.
    """
    value = unet.get("num_attention_heads") or unet["attention_head_dim"]
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return int(value)


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
    # SDXL adds a second, larger text encoder and keeps the first one's
    # tokenizer length, so the context length still comes from text_encoder.
    second_dir = model_dir / "text_encoder_2"
    text_2 = _read(model_dir, "text_encoder_2") if second_dir.is_dir() else None

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
            "attention_head_dim": _heads(unet),
            "cross_attention_dim": int(unet["cross_attention_dim"]),
            "norm_num_groups": int(unet.get("norm_num_groups", 32)),
            "context_length": int(text["max_position_embeddings"]),
            "use_linear_projection": bool(unet.get("use_linear_projection", False)),
            "addition_embed_type": unet.get("addition_embed_type"),
            "addition_time_embed_dim": int(unet.get("addition_time_embed_dim") or 0),
            "addition_time_ids": _ADDITION_TIME_IDS,
            "pooled_projection_dim": int(text_2["projection_dim"]) if text_2 else 0,
            "projection_class_embeddings_input_dim":
                int(unet.get("projection_class_embeddings_input_dim") or 0),
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
        "text_encoder_2": {
            "hidden_size": int(text_2["hidden_size"]),
            "num_hidden_layers": int(text_2["num_hidden_layers"]),
            "num_attention_heads": int(text_2["num_attention_heads"]),
            "max_position_embeddings": int(text_2["max_position_embeddings"]),
            "projection_dim": int(text_2["projection_dim"]),
            "hidden_act": str(text_2.get("hidden_act", "gelu")),
            "layer_norm_eps": float(text_2.get("layer_norm_eps") or _CLIP_LAYER_NORM_EPS),
        } if text_2 else None,
        # SDXL's VAE overflows in fp16, which is what force_upcast records.
        "vae_force_upcast": bool(vae.get("force_upcast", False)),
        "scaling_factor": float(vae.get("scaling_factor") or _VAE_SCALING_FACTOR),
        "num_train_timesteps": int(scheduler.get("num_train_timesteps", 1000)),
        "beta_start": float(scheduler.get("beta_start", 0.00085)),
        "beta_end": float(scheduler.get("beta_end", 0.012)),
        "beta_schedule": str(scheduler.get("beta_schedule", "scaled_linear")),
        "steps_offset": int(scheduler.get("steps_offset", 1)),
    }
