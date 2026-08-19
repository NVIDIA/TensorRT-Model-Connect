# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-Model-Connect plugin for Cosmos3-Nano text-to-video."""

from __future__ import annotations

from pathlib import Path

from .checkpoint_mapper import read_json
from .model_config import (
    COSMOS3_NANO,
    select_generation_profile,
    validate_transformer_config,
    validate_vae_config,
)


COSMOS3_COMMON_BUNDLE_SECTIONS = (
    "vae_decoder_plan",
    "vae_decoder_first_frame_plan",
    "tokenizer.json",
    "tokenizer_config.json",
)


class Cosmos3Plugin:
    name = "cosmos3"
    default_build_precision = "bf16"
    runtime_strategy = "diffusion_cosmos3"
    pipeline_classes = (
        "Cosmos3OmniDiffusersPipeline",
        "Cosmos3OmniPipeline",
        "Cosmos3OmniModularPipeline",
    )

    def matches(self, model_type: str) -> bool:
        return model_type.lower() in {
            "cosmos3",
            "cosmos3_nano",
            "cosmos3-nano",
        }

    def load_weights(self, model_dir: str, config, **_kwargs) -> dict:
        del config
        root = Path(model_dir)
        model_index = root / "model_index.json"
        transformer_dir = root / "transformer"
        vae_dir = root / "vae"
        transformer_config = transformer_dir / "config.json"
        vae_config = vae_dir / "config.json"
        required = (model_index, transformer_config, vae_config)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Incomplete Cosmos3-Nano Diffusers checkpoint; missing: " + ", ".join(missing)
            )

        validate_transformer_config(read_json(transformer_config))
        validate_vae_config(read_json(vae_config))

        tokenizer_dir = root / "text_tokenizer"
        if not tokenizer_dir.is_dir():
            tokenizer_dir = root / "tokenizer"
        tokenizer_json = tokenizer_dir / "tokenizer.json"
        tokenizer_config_json = tokenizer_dir / "tokenizer_config.json"
        negative_prompt = root / "assets" / "negative_prompt.json"
        tokenizer_assets = (tokenizer_json, tokenizer_config_json, negative_prompt)
        missing_tokenizer_assets = [str(path) for path in tokenizer_assets if not path.is_file()]
        if missing_tokenizer_assets:
            raise FileNotFoundError(
                "Incomplete Cosmos3-Nano Diffusers checkpoint; missing: "
                + ", ".join(missing_tokenizer_assets)
            )

        return {
            "_model_format": "diffusers",
            "_root_dir": str(root),
            "_transformer_dir": str(transformer_dir),
            "_vae_dir": str(vae_dir),
            "_tokenizer_dir": str(tokenizer_dir),
            "_transformer_config": read_json(transformer_config),
            "_vae_config": read_json(vae_config),
        }

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
        raise NotImplementedError("Cosmos3-Nano uses build_components(), not build_engine()")

    def build_components(
        self,
        model_dir: str,
        config,
        weights: dict,
        *,
        precision: str = "bf16",
        verbose: bool = False,
        parallel_config=None,
        **kwargs,
    ) -> dict:
        from .trt_builder import build_cosmos3_components

        return build_cosmos3_components(
            model_dir,
            config=config,
            weights=weights,
            precision=precision,
            verbose=verbose,
            parallel_config=parallel_config,
            **kwargs,
        )

    def diffusion_bundle_sections(
        self, components: dict, *, parallel_config=None
    ) -> list[tuple[str, bytes]]:
        from tensorrt_model_connect.parallel_config import (
            context_denoiser_section,
            normalize_parallel_config,
        )

        parallel = normalize_parallel_config(parallel_config)
        denoiser_section = context_denoiser_section() if parallel.cp_enabled else "denoiser_plan"
        sections = [(denoiser_section, components["denoiser"])]
        payloads = {
            "vae_decoder_plan": components["vae_decoder"],
            "vae_decoder_first_frame_plan": components["vae_decoder_first_frame"],
            "tokenizer.json": components["tokenizer_json"],
            "tokenizer_config.json": components["tokenizer_config_json"],
        }
        sections.extend((name, payloads[name]) for name in COSMOS3_COMMON_BUNDLE_SECTIONS)
        return sections

    def diffusion_bundle_config(self, config, *, components: dict) -> dict:
        bundle_config = self.get_diffusion_config(config)
        bundle_config["negative_prompt"] = components["negative_prompt"]
        return bundle_config

    def diffusion_tokenizer_add_special_tokens(
        self, model_dir_path, *, detect_tokenizer_add_special_tokens
    ) -> bool:
        del model_dir_path, detect_tokenizer_add_special_tokens
        return False

    def diffusion_tokenizer_bundle_sections(
        self, model_dir_path, *, ensure_tokenizer_json
    ) -> list[tuple[str, bytes]]:
        del model_dir_path, ensure_tokenizer_json
        return []

    def get_diffusion_config(self, config) -> dict:
        profile = select_generation_profile(config.raw)
        seed = int(config.raw.get("seed", profile.seed))
        if not 0 <= seed <= 2_147_483_647:
            raise ValueError("Cosmos3-Nano seed must be between 0 and 2147483647")
        return {
            "num_inference_steps": profile.num_inference_steps,
            "guidance_scale": profile.guidance_scale,
            "flow_shift": profile.flow_shift,
            "video_height": profile.video_height,
            "video_width": profile.video_width,
            "video_num_frames": profile.video_num_frames,
            "frame_rate": profile.frame_rate,
            "text_seq_len": COSMOS3_NANO.max_text_seq_len,
            "seed": seed,
        }


plugin = Cosmos3Plugin()
