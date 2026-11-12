# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read and check the configuration of one ``StableDiffusion3Pipeline`` snapshot.

The model configuration the builders and the runtime rely on is checked here,
and the tokenizer files in ``prompt_tokenizers``, so an unsupported checkpoint
fails before any weight is read.
"""

from __future__ import annotations

import json
from pathlib import Path

# Component -> accepted (library, class) entries of model_index.json. The text
# encoders are CLIP-L, CLIP-G (OpenCLIP bigG) and T5, in that order. transformers
# 5 saves the tokenizers under their unified class names.
_COMPONENTS = {
    "transformer": {("diffusers", "SD3Transformer2DModel")},
    "vae": {("diffusers", "AutoencoderKL")},
    "scheduler": {("diffusers", "FlowMatchEulerDiscreteScheduler")},
    "text_encoder": {("transformers", "CLIPTextModelWithProjection")},
    "text_encoder_2": {("transformers", "CLIPTextModelWithProjection")},
    "text_encoder_3": {("transformers", "T5EncoderModel")},
    "tokenizer": {("transformers", "CLIPTokenizer"), ("transformers", "CLIPTokenizerFast")},
    "tokenizer_2": {("transformers", "CLIPTokenizer"), ("transformers", "CLIPTokenizerFast")},
    "tokenizer_3": {("transformers", "T5TokenizerFast"), ("transformers", "T5Tokenizer")},
}
# IP-Adapter components; a text-to-image snapshot leaves them empty.
_UNUSED_COMPONENTS = ("image_encoder", "feature_extractor")

CLIP_ACTIVATIONS = ("quick_gelu", "gelu")
# The pipeline rejects longer T5 prompts.
MAX_T5_LENGTH = 512
DEFAULT_T5_LENGTH = 256


def _read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"stable_diffusion_3 needs {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NotImplementedError(f"stable_diffusion_3: {message}")


def _check_pipeline(model_index: dict) -> None:
    _require(model_index.get("_class_name") == "StableDiffusion3Pipeline",
             f"model_index.json names {model_index.get('_class_name')!r}")
    for component, accepted in _COMPONENTS.items():
        _require(tuple(model_index.get(component) or ()) in accepted,
                 f"component {component} is {model_index.get(component)!r}, expected one of "
                 f"{sorted(accepted)}")
    for component in _UNUSED_COMPONENTS:
        value = model_index.get(component)
        _require(value is None or all(item is None for item in value),
                 f"component {component} ({value!r}) is not supported")


def _check_scheduler(scheduler: dict) -> dict:
    for flag in ("use_dynamic_shifting", "use_karras_sigmas", "use_exponential_sigmas",
                 "use_beta_sigmas", "invert_sigmas", "stochastic_sampling"):
        _require(not scheduler.get(flag, False), f"scheduler option {flag} is not supported")
    _require(not scheduler.get("shift_terminal"), "scheduler option shift_terminal is not supported")
    steps = int(scheduler.get("num_train_timesteps", 1000))
    shift = float(scheduler.get("shift", 1.0))
    _require(steps > 1 and shift > 0.0, "scheduler constants are out of range")
    return {"num_train_timesteps": steps, "shift": shift}


def _check_transformer(transformer: dict) -> dict:
    heads = int(transformer["num_attention_heads"])
    head_dim = int(transformer["attention_head_dim"])
    layers = int(transformer["num_layers"])
    dual = tuple(int(index) for index in transformer.get("dual_attention_layers") or ())
    qk_norm = transformer.get("qk_norm")
    _require(qk_norm in (None, "rms_norm"), f"qk_norm={qk_norm!r} is not supported")
    _require(all(0 <= index < layers for index in dual), "dual_attention_layers out of range")
    _require(int(transformer["caption_projection_dim"]) == heads * head_dim,
             "caption_projection_dim must equal the transformer width")
    _require(int(transformer.get("out_channels") or transformer["in_channels"])
             == int(transformer["in_channels"]), "in_channels and out_channels differ")
    _require(transformer.get("pos_embed_max_size") is not None,
             "a transformer without pos_embed_max_size is not supported")
    return {
        "num_layers": layers,
        "num_heads": heads,
        "head_dim": head_dim,
        "width": heads * head_dim,
        "patch_size": int(transformer["patch_size"]),
        "in_channels": int(transformer["in_channels"]),
        "joint_attention_dim": int(transformer["joint_attention_dim"]),
        "pooled_projection_dim": int(transformer["pooled_projection_dim"]),
        "pos_embed_max_size": int(transformer["pos_embed_max_size"]),
        "dual_attention_layers": dual,
        "qk_norm": qk_norm,
    }


def _check_vae(vae: dict) -> dict:
    _require(vae.get("use_post_quant_conv") is False, "a VAE with post_quant_conv is not supported")
    _require(vae.get("act_fn") == "silu", f"VAE act_fn={vae.get('act_fn')!r}")
    _require(all(kind == "UpDecoderBlock2D" for kind in vae["up_block_types"]),
             "VAE up blocks must all be UpDecoderBlock2D")
    _require(vae.get("mid_block_add_attention", True) is True,
             "a VAE without mid-block attention is not supported")
    _require(vae.get("latents_mean") is None and vae.get("latents_std") is None,
             "a VAE with latents_mean / latents_std is not supported")
    _require(vae.get("shift_factor") is not None, "the VAE has no shift_factor")
    return {
        "latent_channels": int(vae["latent_channels"]),
        "block_out_channels": tuple(int(value) for value in vae["block_out_channels"]),
        "layers_per_block": int(vae["layers_per_block"]),
        "norm_num_groups": int(vae["norm_num_groups"]),
        "out_channels": int(vae["out_channels"]),
        "scaling_factor": float(vae["scaling_factor"]),
        "shift_factor": float(vae["shift_factor"]),
        # Each up block but the last doubles the resolution.
        "scale_factor": 2 ** (len(vae["block_out_channels"]) - 1),
    }


def _check_clip(clip: dict, name: str) -> dict:
    _require(clip.get("model_type") == "clip_text_model", f"{name} is not a CLIP text model")
    _require(clip.get("hidden_act") in CLIP_ACTIVATIONS,
             f"{name} hidden_act={clip.get('hidden_act')!r}")
    width = int(clip["hidden_size"])
    heads = int(clip["num_attention_heads"])
    _require(width % heads == 0, f"{name} heads do not divide the width")
    return {
        "width": width,
        "heads": heads,
        "layers": int(clip["num_hidden_layers"]),
        "intermediate": int(clip["intermediate_size"]),
        "projection_dim": int(clip["projection_dim"]),
        "length": int(clip["max_position_embeddings"]),
        "vocab_size": int(clip["vocab_size"]),
        "eps": float(clip["layer_norm_eps"]),
        "activation": clip["hidden_act"],
        # Selects transformers' pooling rule; legacy configs say 2.
        "eos_token_id": int(clip["eos_token_id"]),
    }


def _check_t5(t5: dict) -> dict:
    _require(t5.get("model_type") == "t5", "text_encoder_3 is not T5")
    _require(t5.get("feed_forward_proj") == "gated-gelu"
             and t5.get("dense_act_fn", "gelu_new") == "gelu_new",
             "only the gated-gelu T5 v1.1 feed-forward is supported")
    _require(int(t5["d_kv"]) * int(t5["num_heads"]) == int(t5["d_model"]),
             "T5 heads must tile d_model")
    return {
        "width": int(t5["d_model"]),
        "heads": int(t5["num_heads"]),
        "head_dim": int(t5["d_kv"]),
        "layers": int(t5["num_layers"]),
        "intermediate": int(t5["d_ff"]),
        "num_buckets": int(t5["relative_attention_num_buckets"]),
        "max_distance": int(t5["relative_attention_max_distance"]),
        "eps": float(t5["layer_norm_epsilon"]),
        "vocab_size": int(t5["vocab_size"]),
        "pad_token_id": int(t5["pad_token_id"]),
        "eos_token_id": int(t5["eos_token_id"]),
    }


def default_image_size(model_dir: Path) -> int:
    """The pipeline's default edge: the transformer sample size in pixels."""
    root = Path(model_dir)
    transformer = _read(root / "transformer" / "config.json")
    vae = _read(root / "vae" / "config.json")
    return int(transformer["sample_size"]) * 2 ** (len(vae["block_out_channels"]) - 1)


def resolve(model_dir: Path, *, height: int, width: int, max_sequence_length: int) -> dict:
    """Return the checked configuration for one build geometry."""
    root = Path(model_dir)
    _check_pipeline(_read(root / "model_index.json"))
    scheduler = _check_scheduler(_read(root / "scheduler" / "scheduler_config.json"))
    transformer = _check_transformer(_read(root / "transformer" / "config.json"))
    vae = _check_vae(_read(root / "vae" / "config.json"))
    clip_l = _check_clip(_read(root / "text_encoder" / "config.json"), "text_encoder")
    clip_g = _check_clip(_read(root / "text_encoder_2" / "config.json"), "text_encoder_2")
    t5 = _check_t5(_read(root / "text_encoder_3" / "config.json"))

    _require(clip_l["length"] == clip_g["length"], "the two CLIP encoders differ in length")
    _require(transformer["in_channels"] == vae["latent_channels"],
             "the transformer and the VAE disagree on latent channels")
    _require(transformer["joint_attention_dim"] == t5["width"],
             "joint_attention_dim must equal the T5 width")
    _require(clip_l["width"] + clip_g["width"] <= t5["width"],
             "the CLIP hidden states must fit into the T5 width")
    _require(clip_l["projection_dim"] + clip_g["projection_dim"]
             == transformer["pooled_projection_dim"],
             "the pooled CLIP projections must fill pooled_projection_dim")
    _require(1 <= max_sequence_length <= MAX_T5_LENGTH,
             f"max_sequence_length must be in [1, {MAX_T5_LENGTH}]")

    multiple = vae["scale_factor"] * transformer["patch_size"]
    if height % multiple or width % multiple:
        raise ValueError(f"stable_diffusion_3 image height and width must be multiples of {multiple}")
    limit = transformer["pos_embed_max_size"] * multiple
    if height > limit or width > limit:
        raise ValueError(f"stable_diffusion_3 images are limited to {limit} pixels per side")

    for name in ("tokenizer", "tokenizer_2"):
        directory = root / name
        if not (directory / "tokenizer.json").is_file() and not (
                (directory / "vocab.json").is_file() and (directory / "merges.txt").is_file()):
            raise FileNotFoundError(f"{directory} has neither tokenizer.json nor vocab/merges")
    if not (root / "tokenizer_3" / "tokenizer.json").is_file():
        raise FileNotFoundError(f"{root / 'tokenizer_3'} has no tokenizer.json")

    return {
        "scheduler": scheduler,
        "transformer": transformer,
        "vae": vae,
        "clip_l": clip_l,
        "clip_g": clip_g,
        "t5": t5,
        "height": int(height),
        "width": int(width),
        "latent_height": int(height) // vae["scale_factor"],
        "latent_width": int(width) // vae["scale_factor"],
        "t5_length": int(max_sequence_length),
    }
