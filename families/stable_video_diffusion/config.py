# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and check the Diffusers component configs this family builds against.

Only the configuration the family implements is accepted; anything else is
refused here, before weights are read.
"""

from __future__ import annotations

import json
import math
import struct
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
# The graph builders normalize in this many groups (diffusers' norm_num_groups for these models).
NORM_GROUPS = 32


def _unique_keys(pairs: list) -> dict:
    """A JSON object whose keys are unique: json.loads would otherwise keep the last value."""
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"stable_video_diffusion: a config repeats the key {key!r}")
        document[key] = value
    return document


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing Stable Video Diffusion config: {path}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(f"stable_video_diffusion: {message}")


def _integer(raw: dict, key: str, minimum: int = 1) -> int:
    """A JSON integer (not a float, string or boolean) of at least ``minimum``."""
    value = raw[key]
    _require(type(value) is int and value >= minimum,
             f"{key} must be an integer of at least {minimum}")
    return value


def _integers(values, name: str) -> list[int]:
    """A non-empty JSON list of positive integers."""
    _require(isinstance(values, list) and len(values) > 0
             and all(type(value) is int and value > 0 for value in values),
             f"{name} must list positive integers")
    return list(values)


def _names(values, name: str) -> list[str]:
    """A JSON list of strings."""
    _require(isinstance(values, list) and all(isinstance(value, str) for value in values),
             f"{name} must list block type names")
    return list(values)


def _finite(value) -> bool:
    """A JSON number (not a string or boolean) that is finite as a double."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _float32(value) -> float | None:
    """The value as the runtime stores it, or None when it is not a finite float32."""
    if type(value) not in (int, float):
        return None
    try:
        stored = struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return None
    return stored if math.isfinite(stored) else None


def _positive_float32(value) -> bool:
    stored = _float32(value)
    return stored is not None and stored > 0.0


def _sampler_sigma(value) -> bool:
    """A sigma the float32 sampler can use: positive, with finite sigma^2 + 1 and 1 / sigma."""
    stored = _float32(value)
    return (stored is not None and stored > 0.0 and _float32(stored * stored + 1.0) is not None
            and _float32(1.0 / stored) is not None)


# The runtime's bicubic resize of the conditioning image (cubic convolution, A = -0.75, no
# clamp) can leave [0, 1] by up to 0.4453125 on either side: per axis, at most 1.1875 positive
# and 0.1875 negative weight.
_RESIZED_PIXEL_BOUNDS = (-0.5, 1.5)


def _normalizable(mean, std) -> bool:
    """(pixel - mean) / std stays finite in float32 for every resized pixel, computed as the
    runtime does: with the stored float32 mean and std, and two separate float32 operations."""
    stored_mean, stored_std = _float32(mean), _float32(std)
    if stored_mean is None or stored_std is None or not stored_std > 0.0:
        return False
    for pixel in _RESIZED_PIXEL_BOUNDS:
        centered = _float32(pixel - stored_mean)
        if centered is None or _float32(centered / stored_std) is None:
            return False
    return True


def _per_level(value, levels: int, name: str) -> list[int]:
    if type(value) is int:
        return [value] * levels
    _require(isinstance(value, list) and len(value) == levels, f"{name} must name every level")
    _require(all(type(item) is int for item in value), f"{name} must be integers")
    return list(value)


def _unet(raw: dict) -> dict:
    channels = _integers(raw["block_out_channels"], "block_out_channels")
    levels = len(channels)
    down = _names(raw["down_block_types"], "down_block_types")
    up = _names(raw["up_block_types"], "up_block_types")
    _require(len(down) == levels and len(up) == levels, "block types must match the level count")
    _require(set(down) <= _DOWN_BLOCKS and set(up) <= _UP_BLOCKS, "unknown UNet block types")
    layers = _integer(raw, "layers_per_block")
    transformer_layers = raw.get("transformer_layers_per_block", 1)
    _require(type(transformer_layers) is int and transformer_layers == 1,
             "one transformer layer per attention block is built")
    cross_attention_dim = _integer(raw, "cross_attention_dim")
    heads = _per_level(raw["num_attention_heads"], levels, "num_attention_heads")
    _require(all(count > 0 and width % count == 0 for width, count in zip(channels, heads)),
             "attention heads must divide the level widths")
    _require(all(width % NORM_GROUPS == 0 for width in channels),
             f"the UNet level widths must be multiples of the {NORM_GROUPS} normalization groups")
    width = _integer(raw, "addition_time_embed_dim", minimum=0)
    projection = _integer(raw, "projection_class_embeddings_input_dim", minimum=0)
    # The runtime sends fps - 1, motion_bucket_id and noise_aug_strength; each is embedded as
    # [cos | sin] halves.
    _require(width > 0 and width % 2 == 0 and projection == 3 * width,
             "the UNet must take three added time ids of one even embedding width each")
    frames = _integer(raw, "num_frames")
    _require(frames < 2**31, "num_frames must fit the runtime's 32-bit frame count")
    return {
        "in_channels": _integer(raw, "in_channels"),
        "out_channels": _integer(raw, "out_channels"),
        "block_out_channels": channels,
        "down_block_types": down,
        "up_block_types": up,
        "layers_per_block": layers,
        "num_attention_heads": heads,
        "cross_attention_dim": cross_attention_dim,
        "addition_time_embed_dim": width,
        # fps - 1, motion_bucket_id and noise_aug_strength for the published checkpoints.
        "added_time_ids": projection // width,
        "num_frames": frames,
    }


def _vae(raw: dict) -> dict:
    down = _names(raw["down_block_types"], "VAE down_block_types")
    _require(all(kind == "DownEncoderBlock2D" for kind in down),
             "the VAE encoder must use DownEncoderBlock2D blocks")
    channels = _integers(raw["block_out_channels"], "VAE block_out_channels")
    _require(len(down) == len(channels),
             "VAE block types must match the level count")
    _require(len(channels) == 4,
             "the VAE must have four levels: the bundle uses an eightfold latent scale")
    _require(all(width % NORM_GROUPS == 0 for width in channels),
             f"the VAE level widths must be multiples of the {NORM_GROUPS} normalization groups")
    _require(_integer(raw, "in_channels") == 3 and _integer(raw, "out_channels") == 3,
             "the VAE must be RGB")
    scaling_factor = raw["scaling_factor"]
    # The decoder input is the latent times 1 / scaling_factor in float32.
    _require(_finite(scaling_factor) and scaling_factor > 0
             and _positive_float32(1.0 / scaling_factor),
             "the VAE scaling factor and its float32 reciprocal must be finite and positive")
    force_upcast = raw.get("force_upcast", True)
    _require(type(force_upcast) is bool, "force_upcast must be true or false")
    return {
        "in_channels": 3,
        "block_out_channels": channels,
        "layers_per_block": _integer(raw, "layers_per_block"),
        "latent_channels": _integer(raw, "latent_channels"),
        "scaling_factor": float(scaling_factor),
        "force_upcast": force_upcast,
    }


def _image_encoder(raw: dict) -> dict:
    _require(raw.get("model_type") == "clip_vision_model", "the image encoder must be CLIP vision")
    _require(raw.get("hidden_act") == "gelu", "the CLIP MLP must use the exact GELU")
    _require(_integer(raw, "image_size") == CLIP_IMAGE_SIZE, "the CLIP input must be 224 x 224")
    num_channels = raw.get("num_channels", 3)
    _require(type(num_channels) is int and num_channels == 3, "the CLIP input must be RGB")
    patch_size = _integer(raw, "patch_size")
    _require(CLIP_IMAGE_SIZE % patch_size == 0, "the CLIP patch size must divide the 224 input")
    hidden_size = _integer(raw, "hidden_size")
    heads = _integer(raw, "num_attention_heads")
    _require(hidden_size % heads == 0, "the CLIP attention heads must divide the hidden size")
    # The layer norms take epsilon as a float32 attribute.
    epsilon = raw["layer_norm_eps"]
    _require(_positive_float32(epsilon),
             "the CLIP layer norm epsilon must be positive and finite in float32")
    return {
        "image_size": CLIP_IMAGE_SIZE,
        "patch_size": patch_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": _integer(raw, "num_hidden_layers"),
        "num_attention_heads": heads,
        "projection_dim": _integer(raw, "projection_dim"),
        "layer_norm_eps": float(epsilon),
    }


def _scheduler(raw: dict) -> dict:
    _require(raw.get("prediction_type") == "v_prediction", "the scheduler must predict v")
    _require(raw.get("timestep_type") == "continuous", "the scheduler must use continuous timesteps")
    _require(raw.get("use_karras_sigmas") is True, "the scheduler must use Karras sigmas")
    _require(raw.get("use_exponential_sigmas", False) is False
             and raw.get("use_beta_sigmas", False) is False, "only Karras sigmas are built")
    _require(raw.get("final_sigmas_type", "zero") == "zero", "the schedule must end at sigma 0")
    _require(raw.get("rescale_betas_zero_snr", False) is False, "zero-SNR rescaling is not built")
    _require(raw.get("sigma_min") is not None and raw.get("sigma_max") is not None,
             "the Karras range must be explicit")
    sigma_min, sigma_max = raw["sigma_min"], raw["sigma_max"]
    # The sampler steps through the range in float32, with sigma^2 + 1 and 1 / sigma.
    _require(_sampler_sigma(sigma_min) and _sampler_sigma(sigma_max) and sigma_min < sigma_max,
             "the Karras range needs sigma_min < sigma_max, each positive with sigma^2 + 1 and "
             "1 / sigma finite in float32")
    spacing = raw.get("timestep_spacing", "linspace")
    _require(spacing in {"linspace", "leading", "trailing"}, "unknown timestep_spacing")
    return {
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        # EulerDiscreteScheduler.init_noise_sigma: the largest sigma for linspace
        # and trailing spacing, sqrt(sigma_max^2 + 1) otherwise.
        "init_noise_sigma_includes_unit": spacing not in {"linspace", "trailing"},
    }


def _feature_extractor(raw: dict) -> dict:
    mean, std = raw["image_mean"], raw["image_std"]
    _require(isinstance(mean, list) and isinstance(std, list) and len(mean) == 3 and len(std) == 3,
             "the CLIP normalization must have three channels")
    _require(all(_normalizable(m, s) for m, s in zip(mean, std)),
             "the CLIP normalization needs finite means and positive standard deviations "
             "that keep normalized pixels finite in float32")
    return {"image_mean": [float(value) for value in mean],
            "image_std": [float(value) for value in std]}


def resolve(model_dir: str | Path) -> dict:
    """The subset of the component configs the builders and runtime need."""
    model_dir = Path(model_dir)
    index = _read(model_dir / "model_index.json")
    _require(index.get("_class_name") == PIPELINE, f"model_index.json must name {PIPELINE}")
    for component, declared in COMPONENTS.items():
        _require(index.get(component) == list(declared), f"{component} must be {declared[1]}")
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
