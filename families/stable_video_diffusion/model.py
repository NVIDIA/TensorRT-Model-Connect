# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one Stable Video Diffusion image-to-video bundle.

Four engines go into the bundle: the CLIP image encoder, the VAE image encoder,
the spatio-temporal UNet for the classifier-free-guidance pair and the temporal
VAE decoder. ``runtime.json`` carries the shapes, the conditioning
normalization, the Karras sigma range and the pipeline's call defaults; the
runtime owns preprocessing, noise, the Euler sampler and guidance.

Precision follows the published pipeline: ``fp16`` runs the image encoder,
UNet and decoder in FP16 and the VAE encoder in FP32 when the VAE config sets
``force_upcast``, as the pipeline upcasts only the encode; ``fp32`` runs all
four in FP32. FP16 means FP16 activations and FP16-rounded weights with FP32
accumulation in every linear and convolution layer (see ``graph``): those
layers take FP32 copies of their FP16 operands, so the weights are serialized
as FP32 and the products run as TF32, which represents FP16 values exactly.
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt
from safetensors import safe_open

from . import clip_builder, unet_builder, vae_builder
from . import config as config_module

_DEFAULT_HEIGHT = 576
_DEFAULT_WIDTH = 1024
# StableVideoDiffusionPipeline.__call__ defaults.
_CALL_DEFAULTS = {
    "num_steps": 25,
    "min_guidance_scale": 1.0,
    "max_guidance_scale": 3.0,
    "fps": 7,
    "motion_bucket_id": 127,
    "noise_aug_strength": 0.02,
    "seed": 0,
}
_PRECISIONS = {"fp16": trt.float16, "fp32": trt.float32}


def _load_weights(model_dir: Path, component: str, filename: str) -> dict:
    path = model_dir / component / filename
    if not path.is_file():
        raise FileNotFoundError(f"missing {component} weights: {path}")
    weights = {}
    with safe_open(str(path), framework="numpy") as reader:
        for key in reader.keys():
            weights[key] = reader.get_tensor(key)
    return weights


def _compile(populate, *, work, verbose: bool) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    # An FP16 network's linear and convolution layers take FP32 copies of FP16 operands,
    # which TF32 represents exactly: FP32 accumulation at tensor-core speed. An FP32
    # network keeps full FP32 products.
    if work == trt.float16:
        builder_config.set_flag(trt.BuilderFlag.TF32)
    else:
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    populate(network)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("Stable Video Diffusion TensorRT engine build failed")
    return bytes(plan)


def _image_encoder_plan(model_dir: Path, cfg: dict, work, verbose: bool) -> bytes:
    weights = _load_weights(model_dir, "image_encoder", "model.safetensors")
    return _compile(lambda network: clip_builder.build_image_encoder(
        network, weights, cfg["image_encoder"], work), work=work, verbose=verbose)


def _vae_plans(model_dir: Path, cfg: dict, height: int, width: int, frames: int, encoder_work,
               work, verbose: bool) -> tuple[bytes, bytes]:
    weights = _load_weights(model_dir, "vae", "diffusion_pytorch_model.safetensors")
    encoder = _compile(lambda network: vae_builder.build_encoder(
        network, weights, cfg["vae"], height, width, encoder_work), work=encoder_work,
        verbose=verbose)
    decoder = _compile(lambda network: vae_builder.build_decoder(
        network, weights, cfg["vae"], frames, height // 8, width // 8, work), work=work,
        verbose=verbose)
    return encoder, decoder


def _unet_plan(model_dir: Path, cfg: dict, frames: int, height: int, width: int, work,
               verbose: bool) -> bytes:
    weights = _load_weights(model_dir, "unet", "diffusion_pytorch_model.safetensors")
    return _compile(lambda network: unet_builder.build_unet(
        network, weights, cfg["unet"], frames, height // 8, width // 8, work), work=work,
        verbose=verbose)


def _runtime_config(cfg: dict, *, height: int, width: int, frames: int, precision: str) -> dict:
    vae = cfg["vae"]
    scheduler = cfg["scheduler"]
    return {
        "precision": precision,
        "height": height,
        "width": width,
        "num_frames": frames,
        "latent_channels": vae["latent_channels"],
        "latent_height": height // 8,
        "latent_width": width // 8,
        "cross_attention_dim": cfg["unet"]["cross_attention_dim"],
        "added_time_ids": cfg["unet"]["added_time_ids"],
        "clip_image_size": cfg["image_encoder"]["image_size"],
        "clip_image_mean": cfg["feature_extractor"]["image_mean"],
        "clip_image_std": cfg["feature_extractor"]["image_std"],
        "scaling_factor": vae["scaling_factor"],
        "sigma_min": scheduler["sigma_min"],
        "sigma_max": scheduler["sigma_max"],
        "init_noise_sigma_includes_unit": scheduler["init_noise_sigma_includes_unit"],
        "defaults": dict(_CALL_DEFAULTS),
    }


def build(request, writer) -> None:
    """Build one Stable Video Diffusion initial-image-to-video bundle."""
    if request.task != "initial_image_text_to_video":
        raise ValueError("stable_video_diffusion supports only task=initial_image_text_to_video")
    if request.backend != "trt":
        raise ValueError("stable_video_diffusion supports only backend=trt")
    precision = str(request.precision).lower()
    if precision not in _PRECISIONS:
        raise ValueError(f"stable_video_diffusion supports precision fp16 or fp32, got {precision!r}")
    if request.max_sequence_length is not None:
        raise NotImplementedError("stable_video_diffusion has no text sequence to size")
    if request.max_batch_size != 1:
        raise NotImplementedError("stable_video_diffusion generates one video per request")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("stable_video_diffusion does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("stable_video_diffusion does not support context parallelism")
    if request.quantization not in (None, "none"):
        raise NotImplementedError(
            f"stable_video_diffusion does not support quantization={request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("stable_video_diffusion does not support fp32_layers")
    if request.dynamic_kv_cache:
        raise NotImplementedError("stable_video_diffusion does not support dynamic_kv_cache")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("stable_video_diffusion does not support MTP speculative decoding")

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir)
    height = int(request.image_height or _DEFAULT_HEIGHT)
    width = int(request.image_width or _DEFAULT_WIDTH)
    frames = int(request.video_num_frames or cfg["unet"]["num_frames"])
    levels = len(cfg["unet"]["block_out_channels"])
    multiple = 8 * 2 ** (levels - 1)
    if height % multiple or width % multiple:
        raise ValueError(
            f"stable_video_diffusion needs height and width that are multiples of {multiple}")
    verbose = bool(request.verbose)
    work = _PRECISIONS[precision]
    # The pipeline upcasts only the VAE encode when force_upcast is set.
    encoder_work = trt.float32 if cfg["vae"]["force_upcast"] else work

    writer.set_header(family="stable_video_diffusion", task=request.task, backend=request.backend)
    writer.add_bytes("image_encoder.plan", _image_encoder_plan(model_dir, cfg, work, verbose))
    encoder_plan, decoder_plan = _vae_plans(model_dir, cfg, height, width, frames, encoder_work,
                                            work, verbose)
    writer.add_bytes("vae_encoder.plan", encoder_plan)
    writer.add_bytes("vae_decoder.plan", decoder_plan)
    writer.add_bytes("unet.plan", _unet_plan(model_dir, cfg, frames, height, width, work, verbose))
    writer.add_json("runtime.json", _runtime_config(
        cfg, height=height, width=width, frames=frames, precision=precision))
