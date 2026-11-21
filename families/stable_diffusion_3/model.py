# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Stable Diffusion 3 / 3.5 text-to-image bundle.

Five engines go into the bundle: the CLIP-L and CLIP-G (OpenCLIP bigG) text
encoders, the T5 encoder, the MMDiT denoiser and the VAE decoder. The three
tokenizers and the constants the runtime needs to assemble the prompt
embedding and to run the flow-matching sampler go with them.

The image size and the T5 sequence length are fixed when the bundle is built.
"""

from __future__ import annotations

from pathlib import Path

from . import config as config_module
from . import prompt_tokenizers
from .clip_builder import build_clip_engine
from .mmdit_builder import TIMESTEP_CHANNELS, build_transformer_engine
from .t5_builder import build_t5_engine
from .vae_builder import build_vae_engine
from .weights import load_component

# StableDiffusion3Pipeline's own defaults.
_DEFAULT_STEPS = 28
_DEFAULT_GUIDANCE = 7.0
# Classifier-free guidance denoises the unconditional and the conditional
# sample in one transformer call.
_GUIDED_BATCH = 2


def _write_plan(writer, name: str, plan) -> None:
    # A serialized plan exposes the buffer protocol; write it without a copy.
    with writer.open_section(name) as section:
        section.write(plan)


def build(request, writer) -> None:
    """Build one Stable Diffusion 3 text-to-image bundle."""
    if request.task != "text_to_image":
        raise ValueError("stable_diffusion_3 supports only task=text_to_image")
    if request.backend != "trt":
        raise ValueError("stable_diffusion_3 supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("stable_diffusion_3 does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("stable_diffusion_3 generates one image per request")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("stable_diffusion_3 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("stable_diffusion_3 does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("stable_diffusion_3 does not support video_num_frames")
    if request.quantization is not None:
        raise NotImplementedError("stable_diffusion_3 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("stable_diffusion_3 does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("stable_diffusion_3 does not support MTP speculative decoding")
    precision = str(request.precision).lower()
    if precision not in {"bf16", "fp32"}:
        raise ValueError(f"stable_diffusion_3 builds bf16 or fp32, not {request.precision!r}")

    model_dir = Path(request.model_dir)
    default_size = config_module.default_image_size(model_dir)
    height = int(request.image_height or default_size)
    width = int(request.image_width or default_size)
    t5_length = int(request.max_sequence_length or config_module.DEFAULT_T5_LENGTH)
    cfg = config_module.resolve(model_dir, height=height, width=width,
                                max_sequence_length=t5_length)
    verbose = bool(request.verbose)
    latent_h, latent_w = cfg["latent_height"], cfg["latent_width"]
    text_tokens = cfg["clip_l"]["length"] + t5_length

    # Tokenizers first: a missing tokenizer fails before any engine is built.
    clip_l_json, clip_l_ids = prompt_tokenizers.clip_tokenizer(model_dir, "tokenizer")
    clip_g_json, clip_g_ids = prompt_tokenizers.clip_tokenizer(model_dir, "tokenizer_2")
    t5_json = prompt_tokenizers.t5_tokenizer(model_dir, cfg["t5"])

    writer.set_header(family="stable_diffusion_3", task=request.task, backend=request.backend)
    _write_plan(writer, "clip_l.plan", build_clip_engine(
        load_component(model_dir, "text_encoder"), cfg["clip_l"], precision=precision,
        verbose=verbose))
    _write_plan(writer, "clip_g.plan", build_clip_engine(
        load_component(model_dir, "text_encoder_2"), cfg["clip_g"], precision=precision,
        verbose=verbose))
    _write_plan(writer, "t5.plan", build_t5_engine(
        load_component(model_dir, "text_encoder_3"), cfg["t5"], t5_length,
        precision=precision, verbose=verbose))
    _write_plan(writer, "transformer.plan", build_transformer_engine(
        load_component(model_dir, "transformer"), cfg["transformer"], latent_h=latent_h,
        latent_w=latent_w, text_tokens=text_tokens, max_batch=_GUIDED_BATCH,
        precision=precision, verbose=verbose))
    _write_plan(writer, "vae.plan", build_vae_engine(
        load_component(model_dir, "vae", prefix="decoder."), cfg["vae"], latent_h=latent_h,
        latent_w=latent_w, precision=precision, verbose=verbose))
    writer.add_bytes("clip_l_tokenizer.json", clip_l_json)
    writer.add_bytes("clip_g_tokenizer.json", clip_g_json)
    writer.add_bytes("t5_tokenizer.json", t5_json)
    writer.add_json("runtime.json", {
        "height": height,
        "width": width,
        "latent_height": latent_h,
        "latent_width": latent_w,
        "latent_channels": cfg["vae"]["latent_channels"],
        "clip_length": cfg["clip_l"]["length"],
        "clip_l_width": cfg["clip_l"]["width"],
        "clip_g_width": cfg["clip_g"]["width"],
        "clip_l_projection": cfg["clip_l"]["projection_dim"],
        "clip_g_projection": cfg["clip_g"]["projection_dim"],
        "clip_l_tokens": clip_l_ids,
        "clip_g_tokens": clip_g_ids,
        "clip_l_config_eos": cfg["clip_l"]["eos_token_id"],
        "clip_g_config_eos": cfg["clip_g"]["eos_token_id"],
        "t5_length": t5_length,
        "t5_eos_id": cfg["t5"]["eos_token_id"],
        "t5_pad_id": cfg["t5"]["pad_token_id"],
        "joint_attention_dim": cfg["transformer"]["joint_attention_dim"],
        "pooled_projection_dim": cfg["transformer"]["pooled_projection_dim"],
        "timestep_channels": TIMESTEP_CHANNELS,
        "num_train_timesteps": cfg["scheduler"]["num_train_timesteps"],
        "shift": cfg["scheduler"]["shift"],
        "scaling_factor": cfg["vae"]["scaling_factor"],
        "shift_factor": cfg["vae"]["shift_factor"],
        "max_batch": _GUIDED_BATCH,
        "default_num_steps": _DEFAULT_STEPS,
        "default_guidance_scale": _DEFAULT_GUIDANCE,
        "precision": precision,
    })
