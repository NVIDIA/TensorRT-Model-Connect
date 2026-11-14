# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and check the Diffusers component configs this family builds against.

Only the configuration the family implements is accepted; anything else is
refused here, before weights are read.
"""

from __future__ import annotations

import json
from pathlib import Path

PIPELINE = "StableVideoDiffusionPipeline"
COMPONENTS = {
    "unet": ("diffusers", "UNetSpatioTemporalConditionModel"),
    "vae": ("diffusers", "AutoencoderKLTemporalDecoder"),
    "image_encoder": ("transformers", "CLIPVisionModelWithProjection"),
    "scheduler": ("diffusers", "EulerDiscreteScheduler"),
    "feature_extractor": ("transformers", "CLIPImageProcessor"),
}
_DOWN_BLOCKS = {"CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal"}
_UP_BLOCKS = {"CrossAttnUpBlockSpatioTemporal", "UpBlockSpatioTemporal"}
# The pipeline resizes the conditioning image to this CLIP input size itself.
CLIP_IMAGE_SIZE = 224


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing Stable Video Diffusion config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(f"stable_video_diffusion: {message}")


def _per_level(value, levels: int, name: str) -> list[int]:
    if isinstance(value, int):
        return [value] * levels
    _require(isinstance(value, list) and len(value) == levels, f"{name} must name every level")
    return [int(item) for item in value]


def _unet(raw: dict) -> dict:
    channels = [int(value) for value in raw["block_out_channels"]]
    levels = len(channels)
    down = list(raw["down_block_types"])
    up = list(raw["up_block_types"])
    _require(len(down) == levels and len(up) == levels, "block types must match the level count")
    _require(set(down) <= _DOWN_BLOCKS and set(up) <= _UP_BLOCKS, "unknown UNet block types")
    _require(isinstance(raw["layers_per_block"], int), "layers_per_block must be one integer")
    _require(raw.get("transformer_layers_per_block", 1) == 1,
             "one transformer layer per attention block is built")
    _require(isinstance(raw["cross_attention_dim"], int), "cross_attention_dim must be one integer")
    heads = _per_level(raw["num_attention_heads"], levels, "num_attention_heads")
    _require(all(width % count == 0 for width, count in zip(channels, heads)),
             "attention heads must divide the level widths")
    width = int(raw["addition_time_embed_dim"])
    projection = int(raw["projection_class_embeddings_input_dim"])
    _require(projection % width == 0, "the added time ids must fill the projection input")
    return {
        "in_channels": int(raw["in_channels"]),
        "out_channels": int(raw["out_channels"]),
        "block_out_channels": channels,
        "down_block_types": down,
        "up_block_types": up,
        "layers_per_block": int(raw["layers_per_block"]),
        "num_attention_heads": heads,
        "cross_attention_dim": int(raw["cross_attention_dim"]),
        "addition_time_embed_dim": width,
        # fps - 1, motion_bucket_id and noise_aug_strength for the published checkpoints.
        "added_time_ids": projection // width,
        "num_frames": int(raw["num_frames"]),
    }


def _vae(raw: dict) -> dict:
    _require(all(kind == "DownEncoderBlock2D" for kind in raw["down_block_types"]),
             "the VAE encoder must use DownEncoderBlock2D blocks")
    _require(len(raw["down_block_types"]) == len(raw["block_out_channels"]),
             "VAE block types must match the level count")
    _require(int(raw["in_channels"]) == 3 and int(raw["out_channels"]) == 3, "the VAE must be RGB")
    return {
        "in_channels": 3,
        "block_out_channels": [int(value) for value in raw["block_out_channels"]],
        "layers_per_block": int(raw["layers_per_block"]),
        "latent_channels": int(raw["latent_channels"]),
        "scaling_factor": float(raw["scaling_factor"]),
        "force_upcast": bool(raw.get("force_upcast", True)),
    }


def _image_encoder(raw: dict) -> dict:
    _require(raw.get("model_type") == "clip_vision_model", "the image encoder must be CLIP vision")
    _require(raw.get("hidden_act") == "gelu", "the CLIP MLP must use the exact GELU")
    _require(int(raw["image_size"]) == CLIP_IMAGE_SIZE, "the CLIP input must be 224 x 224")
    _require(int(raw.get("num_channels", 3)) == 3, "the CLIP input must be RGB")
    return {
        "image_size": int(raw["image_size"]),
        "patch_size": int(raw["patch_size"]),
        "hidden_size": int(raw["hidden_size"]),
        "num_hidden_layers": int(raw["num_hidden_layers"]),
        "num_attention_heads": int(raw["num_attention_heads"]),
        "projection_dim": int(raw["projection_dim"]),
        "layer_norm_eps": float(raw["layer_norm_eps"]),
    }


def _scheduler(raw: dict) -> dict:
    _require(raw.get("prediction_type") == "v_prediction", "the scheduler must predict v")
    _require(raw.get("timestep_type") == "continuous", "the scheduler must use continuous timesteps")
    _require(raw.get("use_karras_sigmas") is True, "the scheduler must use Karras sigmas")
    _require(not raw.get("use_exponential_sigmas") and not raw.get("use_beta_sigmas"),
             "only Karras sigmas are built")
    _require(raw.get("final_sigmas_type", "zero") == "zero", "the schedule must end at sigma 0")
    _require(not raw.get("rescale_betas_zero_snr"), "zero-SNR rescaling is not built")
    _require(raw.get("sigma_min") is not None and raw.get("sigma_max") is not None,
             "the Karras range must be explicit")
    spacing = raw.get("timestep_spacing", "linspace")
    _require(spacing in {"linspace", "leading", "trailing"}, "unknown timestep_spacing")
    return {
        "sigma_min": float(raw["sigma_min"]),
        "sigma_max": float(raw["sigma_max"]),
        # EulerDiscreteScheduler.init_noise_sigma: the largest sigma for linspace
        # and trailing spacing, sqrt(sigma_max^2 + 1) otherwise.
        "init_noise_sigma_includes_unit": spacing not in {"linspace", "trailing"},
    }


def _feature_extractor(raw: dict) -> dict:
    mean = [float(value) for value in raw["image_mean"]]
    std = [float(value) for value in raw["image_std"]]
    _require(len(mean) == 3 and len(std) == 3, "the CLIP normalization must have three channels")
    return {"image_mean": mean, "image_std": std}


def resolve(model_dir: str | Path) -> dict:
    """The subset of the component configs the builders and runtime need."""
    model_dir = Path(model_dir)
    index = _read(model_dir / "model_index.json")
    _require(index.get("_class_name") == PIPELINE, f"model_index.json must name {PIPELINE}")
    for component, declared in COMPONENTS.items():
        _require(list(index.get(component) or []) == list(declared),
                 f"{component} must be {declared[1]}")
    scheduler = _read(model_dir / "scheduler" / "scheduler_config.json")
    _require(scheduler.get("_class_name") == COMPONENTS["scheduler"][1],
             "the scheduler config must be EulerDiscreteScheduler")
    resolved = {
        "unet": _unet(_read(model_dir / "unet" / "config.json")),
        "vae": _vae(_read(model_dir / "vae" / "config.json")),
        "image_encoder": _image_encoder(_read(model_dir / "image_encoder" / "config.json")),
        "scheduler": _scheduler(scheduler),
        "feature_extractor": _feature_extractor(
            _read(model_dir / "feature_extractor" / "preprocessor_config.json")),
    }
    _require(resolved["unet"]["in_channels"] == 2 * resolved["vae"]["latent_channels"],
             "the UNet input must be the noisy latent plus the image latent")
    _require(resolved["unet"]["out_channels"] == resolved["vae"]["latent_channels"],
             "the UNet must predict one latent")
    _require(resolved["unet"]["cross_attention_dim"] == resolved["image_encoder"]["projection_dim"],
             "the UNet must attend to the CLIP image embedding")
    return resolved
