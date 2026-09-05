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
    SOL_ENGINE_1344X768_124F,
    default_workspace_limit_bytes,
)

if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


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
                    head_checkpoint_keys(profile),
                ),
                (
                    "denoiser_tail",
                    "denoiser_tail.plan",
                    build_dit_tail_engine,
                    tail_checkpoint_keys(profile),
                ),
                (
                    "denoiser_finish",
                    "denoiser_finish.plan",
                    build_dit_finish_engine,
                    finish_checkpoint_keys(profile),
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
                    dit_checkpoint_keys(profile),
                ),
            )
            checkpoint_groups = (
                adaln_checkpoint_keys(profile),
                dit_checkpoint_keys(profile),
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
            denoiser_plan = denoiser_builder(
                dit_weights,
                profile,
                verbose=verbose,
                consume_weights=True,
                workspace_bytes=workspace_limits[filename],
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
        tokenizer_json = (Path(weights["_tokenizer_dir"]) / "tokenizer.json").read_bytes()

        return {
            "text_encoder": text_encoder_plan,
            "adaln_precompute": adaln_plan,
            **denoiser_components,
            "vae_decoder": vae_decoder_plan,
            "profile": profile,
            # Text/VAE paths remain explicit so follow-on native component
            # builders cannot silently substitute a different checkpoint.
            "vae_dir": weights["_vae_dir"],
            "audio_vae_dir": weights["_audio_vae_dir"],
            "tokenizer_dir": weights["_tokenizer_dir"],
            "tokenizer_json": tokenizer_json,
        }


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one MiniMax-H3 image-generation bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("minimax_h3 does not support dynamic_kv_cache")

    if request.max_sequence_length is not None:
        raise NotImplementedError("minimax_h3 does not support max_sequence_length")

    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "image_generation":
        raise ValueError("minimax_h3 supports only task=image_generation")
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

    config = SimpleNamespace(raw={})
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
    if profile.first_block_cache:
        raise RuntimeError("MiniMax-H3 minimal build uses the monolithic denoiser profile")

    writer.set_header(family="minimax_h3", task=request.task, backend=request.backend)
    writer.add_bytes("text_encoder.plan", components["text_encoder"])
    writer.add_bytes("adaln.plan", components["adaln_precompute"])
    writer.add_bytes("denoiser.plan", components["denoiser"])
    writer.add_bytes("vae.plan", components["vae_decoder"])
    writer.add_bytes("tokenizer.json", components["tokenizer_json"])
    writer.add_json(
        "runtime.json",
        {
            "height": 768,
            "width": 1344,
            "num_frames": 124,
            "fps": 24,
            "num_inference_steps": 50,
            "seed": 0,
            "first_block_cache": False,
            "denoiser_cache_mode": "monolithic",
            "first_block_cache_threshold": 0.025,
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
