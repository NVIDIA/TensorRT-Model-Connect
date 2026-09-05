# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the exact native Cosmos3-Nano text-to-video bundle."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checkpoint_mapper import read_json, transformer_safetensor_paths
from .model_config import (
    COSMOS3_NANO,
    COSMOS3_NANO_NEGATIVE_PROMPT,
    select_generation_profile,
    validate_transformer_config,
    validate_vae_config,
)

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def _require_checkpoint(model_dir: Path) -> tuple[Path, Path, Path]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Cosmos3 model directory does not exist: {model_dir}")

    model_index_path = model_dir / "model_index.json"
    transformer_dir = model_dir / "transformer"
    vae_dir = model_dir / "vae"
    tokenizer_dir = model_dir / "text_tokenizer"
    required = (
        model_index_path,
        transformer_dir / "config.json",
        transformer_dir / "diffusion_pytorch_model.safetensors.index.json",
        vae_dir / "config.json",
        vae_dir / "diffusion_pytorch_model.safetensors",
        tokenizer_dir / "tokenizer.json",
        tokenizer_dir / "tokenizer_config.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete Cosmos3-Nano checkpoint; missing: " + ", ".join(missing)
        )

    model_index = read_json(model_index_path)
    if model_index.get("_class_name") != "Cosmos3OmniDiffusersPipeline":
        raise ValueError("Cosmos3 requires the Cosmos3OmniDiffusersPipeline checkpoint")
    validate_transformer_config(read_json(transformer_dir / "config.json"))
    validate_vae_config(read_json(vae_dir / "config.json"))
    transformer_safetensor_paths(transformer_dir)
    return transformer_dir, vae_dir, tokenizer_dir


def _build_denoiser(transformer_dir: Path, *, context_parallel_size: int, verbose: bool) -> bytes:
    from .transformer_builder import build_cosmos3_transformer_engine

    return build_cosmos3_transformer_engine(
        str(transformer_dir),
        context_parallel_size=context_parallel_size,
        verbose=verbose,
    )


def _load_vae_weights(vae_dir: Path) -> dict[str, Any]:
    from .vae_step_builder import load_vae_step_weights

    return load_vae_step_weights(vae_dir)


def _build_vae(weights: dict[str, Any], *, first_frame_only: bool, verbose: bool) -> bytes:
    from .vae_step_builder import Cosmos3VaeStepProfile, build_vae_step_engine

    return build_vae_step_engine(
        weights,
        profile=Cosmos3VaeStepProfile(COSMOS3_NANO.latent_height, COSMOS3_NANO.latent_width),
        first_frame_only=first_frame_only,
        verbose=verbose,
    )


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one fixed-profile Cosmos3-Nano bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("cosmos3 does not support dynamic_kv_cache")

    if request.task != "image_generation":
        raise ValueError("cosmos3 supports only task=image_generation")
    if request.precision != "bf16":
        raise ValueError("Cosmos3-Nano requires precision=bf16")
    if request.max_batch_size != 1:
        raise ValueError("Cosmos3-Nano requires max_batch_size=1")
    if request.tensor_parallel_size != 1:
        raise ValueError("Cosmos3-Nano does not use tensor parallelism")
    if request.context_parallel_size not in (1, 2):
        raise ValueError("Cosmos3-Nano context_parallel_size must be 1 or 2")
    if request.quantization is not None:
        raise ValueError("Cosmos3-Nano does not support quantization")
    if request.fp32_layers:
        raise ValueError("Cosmos3-Nano does not support fp32_layers")

    profile = select_generation_profile(
        {
            "video_height": request.image_height or COSMOS3_NANO.video_height,
            "video_width": request.image_width or COSMOS3_NANO.video_width,
            "video_num_frames": request.video_num_frames or COSMOS3_NANO.video_num_frames,
            "frame_rate": COSMOS3_NANO.frame_rate,
            "num_inference_steps": COSMOS3_NANO.num_inference_steps,
            "guidance_scale": COSMOS3_NANO.guidance_scale,
            "flow_shift": COSMOS3_NANO.flow_shift,
        }
    )
    text_seq_len = request.max_sequence_length or profile.max_text_seq_len
    if text_seq_len != profile.max_text_seq_len:
        raise ValueError("Cosmos3-Nano requires max_sequence_length=4096")

    model_dir = Path(request.model_dir)
    transformer_dir, vae_dir, tokenizer_dir = _require_checkpoint(model_dir)

    writer.set_header(family="cosmos3", task=request.task, backend=request.backend)
    denoiser = _build_denoiser(
        transformer_dir,
        context_parallel_size=request.context_parallel_size,
        verbose=request.verbose,
    )
    writer.add_bytes("denoiser.plan", denoiser)
    del denoiser

    vae_weights = _load_vae_weights(vae_dir)
    recurrent_vae = _build_vae(vae_weights, first_frame_only=False, verbose=request.verbose)
    writer.add_bytes("vae.plan", recurrent_vae)
    del recurrent_vae
    first_frame_vae = _build_vae(vae_weights, first_frame_only=True, verbose=request.verbose)
    writer.add_bytes("vae.first_frame.plan", first_frame_vae)
    del first_frame_vae, vae_weights

    writer.add_bytes("tokenizer.json", (tokenizer_dir / "tokenizer.json").read_bytes())
    writer.add_bytes(
        "tokenizer_config.json", (tokenizer_dir / "tokenizer_config.json").read_bytes()
    )
    writer.add_json(
        "runtime.json",
        {
            "negative_prompt": COSMOS3_NANO_NEGATIVE_PROMPT,
            "num_inference_steps": profile.num_inference_steps,
            "guidance_scale": profile.guidance_scale,
            "flow_shift": profile.flow_shift,
            "seed": profile.seed,
            "video_height": profile.video_height,
            "video_width": profile.video_width,
            "video_num_frames": profile.video_num_frames,
            "frame_rate": profile.frame_rate,
            "text_seq_len": text_seq_len,
            "context_parallel_size": request.context_parallel_size,
        },
    )
