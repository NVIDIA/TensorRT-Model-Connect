# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-Model-Connect family plugin for MiniMaxAI/MiniMax-H3."""

from __future__ import annotations

import gc
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .checkpoint import (
    load_selected_component_state_dict,
    numpy_state,
    validate_component_key_partition,
)
from .config import (
    BASE_GENERATION_PROFILE,
    FASTH3_DENSE_4STEP_GENERATION_PROFILE,
    FASTH3_DENSE_4STEP_MODEL_ID,
    FASTH3_VSA_4STEP_GENERATION_PROFILE,
    FASTH3_VSA_4STEP_MODEL_ID,
    SOL_ENGINE_1344X768_124F,
    MiniMaxH3GenerationProfile,
    default_workspace_limit_bytes,
)

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


SUPPORTS_WEIGHT_STREAMING = True


def _read_json_object(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid MiniMax-H3 {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"MiniMax-H3 {label} must be a JSON object: {path}")
    return value


def _generation_profile(root: Path) -> MiniMaxH3GenerationProfile:
    """Resolve and validate a checkpoint-owned base or distilled schedule."""

    contract_path = root / "fastvideo_inference.json"
    if not contract_path.exists():
        BASE_GENERATION_PROFILE.validate()
        return BASE_GENERATION_PROFILE

    contract = _read_json_object(contract_path, label="FastVideo inference contract")
    model_id = contract.get("model_id")
    profiles = {
        FASTH3_DENSE_4STEP_MODEL_ID: (
            FASTH3_DENSE_4STEP_GENERATION_PROFILE,
            {
                "attention_backend": "FLASH_ATTN",
                "checkpoint_step": 1000,
            },
        ),
        FASTH3_VSA_4STEP_MODEL_ID: (
            FASTH3_VSA_4STEP_GENERATION_PROFILE,
            {
                "attention_backend": "VIDEO_SPARSE_ATTN_H3",
                "checkpoint_step": 1300,
                "vsa_tile_size": 64,
                "vsa_sparsity": 0.9,
                "vsa_kernel": "sm100a",
            },
        ),
    }
    if model_id not in profiles:
        raise ValueError(f"Unsupported FastH3 model_id: {model_id!r}")
    profile, variant_contract = profiles[model_id]
    expected = {
        "schema_version": "fasth3-inference-contract-v1",
        "model_id": model_id,
        "guidance_scale": 1.0,
        "num_inference_steps": 5,
        "transformer_forwards": 4,
        "task": "t2av",
        "dmd_denoising_steps": [999, 749, 500, 250],
        **variant_contract,
    }
    mismatches = {
        name: (contract.get(name), value)
        for name, value in expected.items()
        if contract.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported FastH3 inference contract: {mismatches}")

    scheduler_paths = {
        "video": root / "scheduler" / "scheduler_config.json",
        "audio": root / "audio_scheduler" / "scheduler_config.json",
    }
    expected_shifts = {
        "video": profile.video_scheduler_shift,
        "audio": profile.audio_scheduler_shift,
    }
    for name, path in scheduler_paths.items():
        scheduler = _read_json_object(path, label=f"{name} scheduler config")
        if scheduler.get("_class_name") != "MiniMaxH3Scheduler":
            raise ValueError(f"FastH3 {name} scheduler is not MiniMaxH3Scheduler")
        if scheduler.get("shift") != expected_shifts[name]:
            raise ValueError(f"FastH3 {name} scheduler shift must be {expected_shifts[name]}")

    profile.validate()
    return profile


def _fixed_profile(raw: dict):
    expected = {
        "text_rows": SOL_ENGINE_1344X768_124F.text_rows,
        "text_rows_min": SOL_ENGINE_1344X768_124F.min_text_rows,
        "text_rows_opt": SOL_ENGINE_1344X768_124F.opt_text_rows,
        "text_rows_max": SOL_ENGINE_1344X768_124F.text_rows,
        "audio_rows": SOL_ENGINE_1344X768_124F.audio_rows,
        "video_rows": SOL_ENGINE_1344X768_124F.video_rows,
        "padded_sequence_length": SOL_ENGINE_1344X768_124F.padded_sequence_length,
    }
    mismatches = {
        name: (raw[name], value)
        for name, value in expected.items()
        if name in raw and int(raw[name]) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported MiniMax-H3 packed-row profile: {mismatches}")
    explicit_flag = raw.get("first_block_cache")
    mode = raw.get(
        "denoiser_cache_mode",
        "first_block" if explicit_flag is True else "monolithic",
    )
    if mode not in ("monolithic", "first_block"):
        raise ValueError(f"Unsupported MiniMax-H3 denoiser_cache_mode: {mode!r}")
    if explicit_flag is not None and not isinstance(explicit_flag, bool):
        raise ValueError("MiniMax-H3 first_block_cache must be a boolean")
    mode_flag = mode == "first_block"
    if explicit_flag is not None and explicit_flag != mode_flag:
        raise ValueError("MiniMax-H3 cache mode and first_block_cache flag disagree")
    if not mode_flag:
        return SOL_ENGINE_1344X768_124F
    return replace(SOL_ENGINE_1344X768_124F, first_block_cache=True)


class _MiniMaxH3Model:
    def load_weights(self, model_dir: str, config) -> dict:
        del config
        root = Path(model_dir)
        required_dirs = ("transformer", "text_encoder", "vae", "audio_vae", "tokenizer")
        missing = [str(root / name) for name in required_dirs if not (root / name).is_dir()]
        if missing:
            raise FileNotFoundError(
                "Incomplete MiniMax-H3 Diffusers checkpoint: " + ", ".join(missing)
            )
        transformer_config = json.loads((root / "transformer" / "config.json").read_text())
        expected = {
            "hidden_size": 5376,
            "num_layers": 50,
            "num_attention_heads": 56,
            "attention_head_dim": 128,
            "ffn_dim": 14336,
        }
        mismatches = {
            name: (transformer_config.get(name), value)
            for name, value in expected.items()
            if transformer_config.get(name) != value
        }
        if mismatches:
            raise ValueError(f"Unsupported MiniMax-H3 transformer architecture: {mismatches}")
        return {
            "_model_dir": str(root),
            "_transformer_dir": str(root / "transformer"),
            "_text_encoder_dir": str(root / "text_encoder"),
            "_vae_dir": str(root / "vae"),
            "_audio_vae_dir": str(root / "audio_vae"),
            "_tokenizer_dir": str(root / "tokenizer"),
            "_generation_profile": _generation_profile(root),
        }

    def build_components(
        self,
        model_dir: str,
        config,
        weights: dict,
        *,
        precision: str = "bf16",
        verbose: bool = False,
    ) -> dict:
        del model_dir
        if precision.lower() != "bf16":
            raise ValueError("MiniMax-H3 native builds require BF16 checkpoint weights")
        raw = getattr(config, "raw", {})
        profile = _fixed_profile(raw)
        profile.validate()
        weight_streaming = raw.get("weight_streaming", False)
        if not isinstance(weight_streaming, bool):
            raise ValueError("MiniMax-H3 weight_streaming must be a boolean")
        generation_profile = weights["_generation_profile"]
        if generation_profile.uses_vsa and profile.first_block_cache:
            raise ValueError("FastH3 VSA does not support the split FirstBlockCache graph")
        workspace_limits = default_workspace_limit_bytes(
            first_block_cache=profile.first_block_cache
        )
        from .adaln_builder import build_adaln_precompute_engine
        from .adaln_builder import checkpoint_keys as adaln_checkpoint_keys
        from .dit_builder import (
            build_dit_engine,
            build_dit_finish_engine,
            build_dit_head_engine,
            build_dit_tail_engine,
            checkpoint_keys as dit_checkpoint_keys,
            finish_checkpoint_keys,
            head_checkpoint_keys,
            tail_checkpoint_keys,
        )
        from .text_encoder_builder import (
            build_text_encoder_engine,
            checkpoint_keys as text_encoder_checkpoint_keys,
        )

        if profile.first_block_cache:
            denoiser_specs = (
                (
                    "denoiser_head",
                    "denoiser_head.plan",
                    build_dit_head_engine,
                    head_checkpoint_keys(profile, generation_profile),
                ),
                (
                    "denoiser_tail",
                    "denoiser_tail.plan",
                    build_dit_tail_engine,
                    tail_checkpoint_keys(profile, generation_profile),
                ),
                (
                    "denoiser_finish",
                    "denoiser_finish.plan",
                    build_dit_finish_engine,
                    finish_checkpoint_keys(profile, generation_profile),
                ),
            )
            checkpoint_groups = (
                adaln_checkpoint_keys(profile),
                *(spec[3] for spec in denoiser_specs),
            )
        else:
            denoiser_specs = (
                (
                    "denoiser",
                    "denoiser.plan",
                    build_dit_engine,
                    dit_checkpoint_keys(profile, generation_profile),
                ),
            )
            checkpoint_groups = (
                adaln_checkpoint_keys(profile),
                dit_checkpoint_keys(profile, generation_profile),
            )
        validate_component_key_partition(weights["_transformer_dir"], checkpoint_groups)

        text_state = load_selected_component_state_dict(
            weights["_text_encoder_dir"], text_encoder_checkpoint_keys()
        )
        text_weights = numpy_state(text_state)
        del text_state
        text_encoder_plan = build_text_encoder_engine(
            text_weights,
            sequence_length=profile.text_rows,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits["text_encoder.plan"],
            weight_streaming=weight_streaming,
        )
        del text_weights
        gc.collect()

        adaln_state = load_selected_component_state_dict(
            weights["_transformer_dir"], adaln_checkpoint_keys(profile)
        )
        adaln_weights = numpy_state(adaln_state)
        del adaln_state
        adaln_plan = build_adaln_precompute_engine(
            adaln_weights,
            profile,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits["adaln_precompute.plan"],
        )
        del adaln_weights
        gc.collect()

        denoiser_components = {}
        for component_name, filename, denoiser_builder, selected_keys in denoiser_specs:
            dit_state = load_selected_component_state_dict(
                weights["_transformer_dir"], selected_keys
            )
            dit_weights = numpy_state(dit_state)
            del dit_state
            denoiser_options = {
                "generation_profile": generation_profile,
                "verbose": verbose,
                "consume_weights": True,
                "workspace_bytes": workspace_limits[filename],
            }
            if filename == "denoiser.plan":
                denoiser_options["weight_streaming"] = weight_streaming
            denoiser_plan = denoiser_builder(
                dit_weights,
                profile,
                **denoiser_options,
            )
            del dit_weights
            gc.collect()
            denoiser_components[component_name] = denoiser_plan

        from .vae_builder import (
            build_vae_tile_decoder_engine,
            checkpoint_keys as vae_checkpoint_keys,
        )

        vae_state = load_selected_component_state_dict(weights["_vae_dir"], vae_checkpoint_keys())
        vae_weights = numpy_state(vae_state)
        del vae_state
        vae_decoder_plan = build_vae_tile_decoder_engine(
            vae_weights,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits["vae_tile_decoder.plan"],
        )
        del vae_weights
        gc.collect()

        from .audio_vae_builder import (
            build_audio_vae_decoder_engine,
            checkpoint_keys as audio_vae_checkpoint_keys,
        )

        audio_config = _read_json_object(
            Path(weights["_audio_vae_dir"]) / "config.json", label="audio VAE config"
        )
        expected_audio = {
            "_class_name": "AutoencoderKLMiniMaxH3Audio",
            "latent_dim": 2048,
            "latent_channels": 32,
            "decoder_dim": 1024,
            "decoder_rates": [5, 5, 2, 2, 2, 2, 2],
            "decoder_kernel_sizes": [9, 9, 4, 4, 4, 4, 4],
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "sampling_rate": 32000,
        }
        audio_mismatches = {
            name: (audio_config.get(name), value)
            for name, value in expected_audio.items()
            if audio_config.get(name) != value
        }
        if audio_mismatches:
            raise ValueError(f"Unsupported MiniMax-H3 audio VAE architecture: {audio_mismatches}")
        latent_mean = audio_config.get("latents_mean")
        latent_std = audio_config.get("latents_std")
        if (
            not isinstance(latent_mean, list)
            or not isinstance(latent_std, list)
            or len(latent_mean) != 32
            or len(latent_std) != 32
        ):
            raise ValueError("MiniMax-H3 audio VAE must declare 32 latent mean/std values")
        audio_state = load_selected_component_state_dict(
            weights["_audio_vae_dir"], audio_vae_checkpoint_keys()
        )
        audio_weights = numpy_state(audio_state)
        del audio_state
        audio_vae_decoder_plan = build_audio_vae_decoder_engine(
            audio_weights,
            latent_mean,
            latent_std,
            verbose=verbose,
            consume_weights=True,
            workspace_bytes=workspace_limits["audio_vae_decoder.plan"],
        )
        del audio_weights
        gc.collect()
        tokenizer_json = (Path(weights["_tokenizer_dir"]) / "tokenizer.json").read_bytes()

        return {
            "text_encoder": text_encoder_plan,
            "adaln_precompute": adaln_plan,
            **denoiser_components,
            "vae_decoder": vae_decoder_plan,
            "audio_vae_decoder": audio_vae_decoder_plan,
            "profile": profile,
            "generation_profile": generation_profile,
            # Text/VAE paths remain explicit so follow-on native component
            # builders cannot silently substitute a different checkpoint.
            "vae_dir": weights["_vae_dir"],
            "audio_vae_dir": weights["_audio_vae_dir"],
            "tokenizer_dir": weights["_tokenizer_dir"],
            "tokenizer_json": tokenizer_json,
        }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one MiniMax-H3 synchronized audio/video bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("minimax_h3 does not support dynamic_kv_cache")

    if request.max_sequence_length is not None:
        raise NotImplementedError("minimax_h3 does not support max_sequence_length")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "text_to_audio_video":
        raise ValueError("minimax_h3 supports only task=text_to_audio_video")
    if request.precision != "bf16":
        raise ValueError("MiniMax-H3 requires precision=bf16")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("MiniMax-H3 requires tensor_parallel_size=1")
    if request.max_batch_size != 1:
        raise NotImplementedError("MiniMax-H3 requires max_batch_size=1")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("MiniMax-H3 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("MiniMax-H3 does not support fp32_layers")
    if int(request.image_height or 768) != 768 or int(request.image_width or 1344) != 1344:
        raise ValueError("MiniMax-H3 requires image_height=768 and image_width=1344")
    if int(request.video_num_frames or 124) != 124:
        raise ValueError("MiniMax-H3 requires video_num_frames=124")

    config = SimpleNamespace(
        raw={"weight_streaming": request.weight_streaming_budget_bytes is not None}
    )
    model_dir = Path(request.model_dir)
    model = _MiniMaxH3Model()
    weights = model.load_weights(str(model_dir), config)
    components = model.build_components(
        str(model_dir),
        config,
        weights,
        precision=request.precision,
        verbose=request.verbose,
    )
    profile = components["profile"]
    generation_profile = components["generation_profile"]
    if profile.first_block_cache:
        raise RuntimeError("MiniMax-H3 minimal build uses the monolithic denoiser profile")

    writer.set_header(family="minimax_h3", task=request.task, backend=request.backend)
    writer.add_bytes("text_encoder.plan", components["text_encoder"])
    writer.add_bytes("adaln.plan", components["adaln_precompute"])
    writer.add_bytes("denoiser.plan", components["denoiser"])
    writer.add_bytes("vae.plan", components["vae_decoder"])
    writer.add_bytes("audio_vae.plan", components["audio_vae_decoder"])
    writer.add_bytes("tokenizer.json", components["tokenizer_json"])
    writer.add_json(
        "runtime.json",
        {
            "height": 768,
            "width": 1344,
            "num_frames": 124,
            "fps": 24,
            "audio_sample_rate": 32000,
            "audio_channels": 2,
            "audio_samples_per_channel": 165600,
            "generation_profile": generation_profile.name,
            "num_inference_steps": generation_profile.num_inference_steps,
            "transformer_forwards": generation_profile.transformer_forwards,
            "video_scheduler_shift": generation_profile.video_scheduler_shift,
            "audio_scheduler_shift": generation_profile.audio_scheduler_shift,
            "dmd_denoising_steps": list(generation_profile.dmd_denoising_steps),
            "attention_backend": generation_profile.attention_backend,
            "vsa_tile_size": generation_profile.vsa_tile_size,
            "vsa_sparsity": generation_profile.vsa_sparsity,
            "vsa_kernel": generation_profile.vsa_kernel,
            "seed": 0,
            "first_block_cache": False,
            "denoiser_cache_mode": "monolithic",
            "first_block_cache_threshold": 0.025,
            "weight_streaming_budget_bytes": request.weight_streaming_budget_bytes,
            "text_rows": profile.text_rows,
            "text_rows_min": profile.min_text_rows,
            "text_rows_opt": profile.opt_text_rows,
            "text_rows_max": profile.text_rows,
            "audio_rows": profile.audio_rows,
            "video_rows": profile.video_rows,
            "padded_sequence_length": profile.padded_sequence_length,
            "max_timestep_count": profile.max_timestep_count,
            "context_parallel_size": profile.context_parallel_size,
            "vae_tile_batch": 28,
            "vae_tile_size": 256,
            "vae_tile_overlap": 64,
        },
    )
