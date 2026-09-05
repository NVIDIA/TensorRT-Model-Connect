# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V-5B family build."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .model_config import (
    OFFICIAL_NEGATIVE_PROMPT,
    select_generation_profile,
    validate_native_config,
)


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


class _Wan22TI2VModel:
    def load_weights(self, model_dir: str, config) -> dict:
        root = Path(model_dir)
        config_path = root / "config.json"
        if not config_path.exists():
            raise ValueError(f"Wan2.2 TI2V requires native config.json in {root}")
        native_config = json.loads(config_path.read_text())
        validate_native_config(native_config)

        required = {
            "_vae_checkpoint": root / "Wan2.2_VAE.pth",
            "_text_encoder_checkpoint": root / "models_t5_umt5-xxl-enc-bf16.pth",
            "_tokenizer_dir": root / "google" / "umt5-xxl",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Incomplete Wan2.2-TI2V-5B checkpoint; missing: " + ", ".join(missing)
            )
        tokenizer_json = required["_tokenizer_dir"] / "tokenizer.json"
        if not tokenizer_json.is_file():
            raise FileNotFoundError(
                f"Incomplete Wan2.2-TI2V-5B checkpoint; missing: {tokenizer_json}"
            )

        return {key: str(path) for key, path in required.items()}

    def build_components(
        self,
        model_dir: str,
        config,
        weights: dict,
        *,
        precision: str = "bf16",
        verbose: bool = False,
        fp8_scales: dict | None = None,
    ) -> dict:
        from .trt_builder import build_wan22_components

        return build_wan22_components(
            model_dir,
            config=config,
            weights=weights,
            precision=precision,
            verbose=verbose,
            fp8_scales=fp8_scales,
        )

    def get_diffusion_config(self, config) -> dict:
        raw = config.raw
        arch = select_generation_profile(raw)
        seed = int(raw.get("seed", 42))
        if not 0 <= seed <= 2_147_483_647:
            raise ValueError("Wan2.2-TI2V-5B bundle seed must be between 0 and 2147483647")
        return {
            "num_inference_steps": arch.num_inference_steps,
            "guidance_scale": arch.guidance_scale,
            "flow_shift": arch.flow_shift,
            "video_height": arch.video_height,
            "video_width": arch.video_width,
            "video_num_frames": arch.video_num_frames,
            "frame_rate": arch.frame_rate,
            "negative_prompt": str(raw.get("negative_prompt", OFFICIAL_NEGATIVE_PROMPT)),
            "text_seq_len": arch.text_seq_len,
            "seed": seed,
        }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Wan2.2 TI2V image-generation bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("wan2_2_ti2v does not support dynamic_kv_cache")

    if request.max_sequence_length is not None:
        raise NotImplementedError("wan2_2_ti2v does not support max_sequence_length")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "image_generation":
        raise ValueError("wan2_2_ti2v supports only task=image_generation")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("Wan2.2 TI2V requires tensor_parallel_size=1")
    if request.max_batch_size != 1:
        raise NotImplementedError("Wan2.2 TI2V requires max_batch_size=1")
    if request.fp32_layers:
        raise NotImplementedError("Wan2.2 TI2V does not support fp32_layers")
    if request.precision not in {"bf16", "bfloat16"}:
        raise ValueError("Wan2.2 TI2V requires precision=bf16")

    from .model_config import WAN22_TI2V_5B, WAN22_TI2V_5B_L0

    requested_shape = (
        int(request.image_width or WAN22_TI2V_5B.video_width),
        int(request.image_height or WAN22_TI2V_5B.video_height),
        int(request.video_num_frames or WAN22_TI2V_5B.video_num_frames),
    )
    profiles = {
        (profile.video_width, profile.video_height, profile.video_num_frames): profile
        for profile in (WAN22_TI2V_5B, WAN22_TI2V_5B_L0)
    }
    if requested_shape not in profiles:
        raise ValueError("Wan2.2 TI2V supports only 1280x704/121 frames or 672x384/5 frames")
    profile = profiles[requested_shape]
    config = SimpleNamespace(
        raw={
            "video_width": profile.video_width,
            "video_height": profile.video_height,
            "video_num_frames": profile.video_num_frames,
            "num_inference_steps": profile.num_inference_steps,
            "guidance_scale": profile.guidance_scale,
            "flow_shift": profile.flow_shift,
            "frame_rate": profile.frame_rate,
        }
    )
    model_dir = Path(request.model_dir)
    model = _Wan22TI2VModel()
    weights = model.load_weights(str(model_dir), config)
    if request.quantization in {None, "none"}:
        fp8_scales = None
    elif request.quantization == "fp8":
        from .fp8_profile import load_precomputed_fp8_scales

        fp8_scales = load_precomputed_fp8_scales(config)
    else:
        raise NotImplementedError(
            f"Wan2.2 TI2V does not support quantization={request.quantization!r}"
        )
    components = model.build_components(
        str(model_dir),
        config,
        weights,
        precision=request.precision,
        verbose=request.verbose,
        fp8_scales=fp8_scales,
    )
    text_encoders = components["text_encoders"]
    if len(text_encoders) != 1:
        raise RuntimeError("Wan2.2 TI2V must produce exactly one text encoder")

    writer.set_header(family="wan2_2_ti2v", task=request.task, backend=request.backend)
    writer.add_bytes("text_encoder.0.plan", text_encoders[0][1])
    writer.add_bytes("denoiser.plan", components["denoiser"])
    writer.add_bytes("vae.plan", components["vae_decoder"])
    writer.add_bytes("vae.first_frame.plan", components["vae_decoder_first_frame"])
    writer.add_bytes("tokenizer.json", components["tokenizer_json"])
    runtime = model.get_diffusion_config(config)
    runtime.update(
        {
            "easycache_enabled": False,
            "easycache_threshold": 0.02,
            "easycache_first_exact_steps": 7,
            "easycache_last_exact_steps": 2,
            "easycache_max_consecutive_reuse": 1,
            "late_cfg_enabled": False,
        }
    )
    writer.add_json("runtime.json", runtime)
