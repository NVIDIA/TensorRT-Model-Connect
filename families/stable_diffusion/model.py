# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Stable Diffusion text-to-image bundle.

Three engines go into the bundle — the CLIP text encoder, the UNet denoiser and
the VAE decoder — plus the tokenizer and the scheduler constants the runtime
needs to reproduce the sampling loop.

This is the repository's first UNet denoiser; every other diffusion family
builds a DiT. See ``unet_builder`` for the skip-stack bookkeeping that makes it
different.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt
from safetensors import safe_open

from . import config as config_module
from . import text_encoder_builder, unet_builder, vae_builder

_DEFAULT_LATENT = 64  # 512x512, the resolution SD 1.5 was trained at
_DEFAULT_STEPS = 25
_DEFAULT_GUIDANCE = 7.5


def _load_component(model_dir: Path, component: str, filename: str) -> dict:
    path = Path(model_dir) / component / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing {component} weights: {path}")
    weights: dict = {}
    with safe_open(str(path), framework="numpy") as reader:
        for key in reader.keys():
            weights[key] = reader.get_tensor(key)
    return weights


def _compile(populate, *, precision: str, verbose: bool) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    work_np = np.float16 if precision == "fp16" else np.float32
    work_trt = trt.float16 if precision == "fp16" else trt.float32
    populate(network, work_np, work_trt)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("Stable Diffusion TensorRT engine build failed")
    return bytes(plan)


def build_text_encoder_engine(weights, cfg, *, precision, verbose=False) -> bytes:
    def populate(network, work_np, work_trt):
        ids = network.add_input(
            "input_ids", trt.int32, (1, cfg["max_position_embeddings"]))
        out = text_encoder_builder.build_text_encoder(network, ids, weights, cfg, work_np)
        if out.dtype != trt.float32:
            out = network.add_cast(out, trt.float32).get_output(0)
        out.name = "last_hidden_state"
        network.mark_output(out)
    return _compile(populate, precision=precision, verbose=verbose)


def build_unet_engine(weights, cfg, *, precision, verbose=False) -> bytes:
    return _compile(
        lambda network, work_np, work_trt: unet_builder.build_unet(
            network, weights, cfg, work_np, work_trt),
        precision=precision, verbose=verbose)


def build_vae_engine(weights, cfg, latent_size, *, precision, verbose=False) -> bytes:
    def populate(network, work_np, work_trt):
        latents = network.add_input(
            "latents", trt.float32, (1, cfg["latent_channels"], latent_size, latent_size))
        source = latents
        if source.dtype != work_trt:
            source = network.add_cast(source, work_trt).get_output(0)
        out = vae_builder.build_decoder(network, source, weights, cfg, work_np)
        if out.dtype != trt.float32:
            out = network.add_cast(out, trt.float32).get_output(0)
        out.name = "image"
        network.mark_output(out)
    return _compile(populate, precision=precision, verbose=verbose)


def _tokenizer_bytes(model_dir: Path) -> bytes:
    """The runtime reads a fast-tokenizer JSON; SD 1.5 ships the legacy pair.

    Checkpoints that already carry ``tokenizer.json`` are used as-is. The older
    ``vocab.json`` plus ``merges.txt`` layout is converted here rather than in
    the runtime, so the bundle always holds one format.
    """
    directory = Path(model_dir) / "tokenizer"
    path = directory / "tokenizer.json"
    if path.is_file():
        return path.read_bytes()
    if not (directory / "vocab.json").is_file() or not (directory / "merges.txt").is_file():
        raise FileNotFoundError(
            f"{directory} has neither tokenizer.json nor a vocab.json/merges.txt pair")
    from transformers import CLIPTokenizerFast

    return CLIPTokenizerFast.from_pretrained(str(directory)).backend_tokenizer.to_str().encode()


def _alphas_cumprod(cfg: dict) -> list:
    """The scheduler constants, precomputed so the runtime owns no beta policy."""
    steps = cfg["num_train_timesteps"]
    if cfg["beta_schedule"] == "scaled_linear":
        betas = np.linspace(cfg["beta_start"] ** 0.5, cfg["beta_end"] ** 0.5,
                            steps, dtype=np.float64) ** 2
    elif cfg["beta_schedule"] == "linear":
        betas = np.linspace(cfg["beta_start"], cfg["beta_end"], steps, dtype=np.float64)
    else:
        raise NotImplementedError(
            f"stable_diffusion does not support beta_schedule={cfg['beta_schedule']!r}")
    return np.cumprod(1.0 - betas).astype(np.float32).tolist()


def build(request, writer) -> None:
    """Build one Stable Diffusion image-generation bundle."""
    if request.task != "image_generation":
        raise ValueError("stable_diffusion supports only task=image_generation")
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("stable_diffusion supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("stable_diffusion does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("stable_diffusion does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("stable_diffusion does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("stable_diffusion does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("stable_diffusion does not support video_num_frames")
    if request.max_sequence_length is not None:
        # CLIP's context length is fixed by the checkpoint, so it cannot be overridden.
        raise NotImplementedError("stable_diffusion does not support max_sequence_length")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("stable_diffusion does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("stable_diffusion does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("stable_diffusion does not support MTP speculative decoding")

    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported stable_diffusion precision: {precision}")

    height = int(request.image_height or 0) or _DEFAULT_LATENT * 8
    width = int(request.image_width or 0) or _DEFAULT_LATENT * 8
    if height != width:
        raise NotImplementedError("stable_diffusion builds a square image only")
    if height % 64:
        raise ValueError("stable_diffusion image size must be a multiple of 64")
    latent = height // 8

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir, latent_size=latent)
    verbose = bool(request.verbose)

    text_weights = _load_component(model_dir, "text_encoder", "model.safetensors")
    unet_weights = _load_component(model_dir, "unet", "diffusion_pytorch_model.safetensors")
    vae_weights = _load_component(model_dir, "vae", "diffusion_pytorch_model.safetensors")

    writer.set_header(family="stable_diffusion", task=request.task, backend=request.backend)
    writer.add_bytes("text_encoder.plan", build_text_encoder_engine(
        text_weights, cfg["text_encoder"], precision=precision, verbose=verbose))
    writer.add_bytes("unet.plan", build_unet_engine(
        unet_weights, cfg["unet"], precision=precision, verbose=verbose))
    writer.add_bytes("vae.plan", build_vae_engine(
        vae_weights, cfg["vae"], latent, precision=precision, verbose=verbose))
    writer.add_bytes("tokenizer.json", _tokenizer_bytes(model_dir))
    writer.add_json(
        "runtime.json",
        {
            "latent_size": latent,
            "latent_channels": cfg["vae"]["latent_channels"],
            "image_size": height,
            "context_length": cfg["text_encoder"]["max_position_embeddings"],
            "context_width": cfg["unet"]["cross_attention_dim"],
            "scaling_factor": cfg["scaling_factor"],
            "num_train_timesteps": cfg["num_train_timesteps"],
            "steps_offset": cfg["steps_offset"],
            "default_num_steps": _DEFAULT_STEPS,
            "default_guidance_scale": _DEFAULT_GUIDANCE,
            "alphas_cumprod": _alphas_cumprod(cfg),
        },
    )
