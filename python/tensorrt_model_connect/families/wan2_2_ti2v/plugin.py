# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-Model-Connect plugin for Wan2.2 TI2V-5B."""

from __future__ import annotations

import json
from pathlib import Path

from .model_config import (
    OFFICIAL_NEGATIVE_PROMPT,
    select_generation_profile,
    validate_native_config,
)


WAN22_MODEL_OWNED_BUNDLE_SECTIONS = (
    "text_encoder_0_plan",
    "denoiser_plan",
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
    "tokenizer.json",
)
WAN22_EAGER_BUNDLE_SECTIONS = ("tokenizer.json", "config.json")
WAN22_LAZY_BUNDLE_SECTIONS = (
    "text_encoder_0_plan",
    "denoiser_plan",
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
)


class Wan22TI2VPlugin:
    name = "wan2_2_ti2v"
    default_build_precision = "bf16"
    runtime_strategy = "diffusion_wan2_2_ti2v"
    pipeline_classes = ("WanModel",)

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in {
            "ti2v",
            "wan2_2_ti2v",
        }

    def load_weights(self, model_dir: str, config, **_kwargs) -> dict:
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

    def build_engine(
        self,
        config,
        weights: dict,
        _max_cache_length: int,
        *,
        precision: str = "bf16",
        verbose: bool = False,
        **kwargs,
    ) -> bytes:
        del config, weights, _max_cache_length, precision, verbose, kwargs
        raise NotImplementedError("Wan2.2 TI2V uses build_components(), not build_engine()")

    def fp8_precomputed_scales(self, model_dir: str, config) -> dict:
        """Load the packaged scale profile after fail-closed qualification."""

        from .fp8_profile import load_packaged_fp8_scales

        return load_packaged_fp8_scales(model_dir, config)

    def build_components(
        self,
        model_dir: str,
        config,
        weights: dict,
        *,
        precision: str = "bf16",
        verbose: bool = False,
        **kwargs,
    ) -> dict:
        from .trt_builder import build_wan22_components

        return build_wan22_components(
            model_dir,
            config=config,
            weights=weights,
            precision=precision,
            verbose=verbose,
            **kwargs,
        )

    def diffusion_bundle_sections(
        self, components: dict, *, parallel_config=None
    ) -> list[tuple[str, bytes]]:
        del parallel_config
        payloads = {
            "text_encoder_0_plan": components["text_encoders"][0][1],
            "denoiser_plan": components["denoiser"],
            "vae_decoder_plan": components["vae_decoder"],
            "vae_decoder_first_frame_plan": components["vae_decoder_first_frame"],
            "tokenizer.json": components["tokenizer_json"],
        }
        return [(name, payloads[name]) for name in WAN22_MODEL_OWNED_BUNDLE_SECTIONS]

    def diffusion_bundle_config(self, config, *, components: dict) -> dict:
        del components
        result = self.get_diffusion_config(config)
        result["bundle_loading"] = {
            "mode": "staged",
            "eager_sections": list(WAN22_EAGER_BUNDLE_SECTIONS),
            "lazy_sections": list(WAN22_LAZY_BUNDLE_SECTIONS),
        }
        return result

    def diffusion_tokenizer_add_special_tokens(
        self, model_dir_path, *, detect_tokenizer_add_special_tokens
    ) -> bool:
        del model_dir_path, detect_tokenizer_add_special_tokens
        return False

    def diffusion_tokenizer_bundle_sections(
        self, model_dir_path, *, ensure_tokenizer_json
    ) -> list[tuple[str, bytes]]:
        # tokenizer.json is already emitted with the model-owned sections.
        del model_dir_path, ensure_tokenizer_json
        return []

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


plugin = Wan22TI2VPlugin()
