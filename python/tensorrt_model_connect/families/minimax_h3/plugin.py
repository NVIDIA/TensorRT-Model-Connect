# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-Model-Connect family plugin for MiniMaxAI/MiniMax-H3."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from .checkpoint import (
    load_component_state_dict,
    load_selected_component_state_dict,
    numpy_state,
    require_keys,
)
from .config import SOL_ENGINE_1344X768_124F


class MiniMaxH3Plugin:
    name = "minimax_h3"
    default_build_precision = "bf16"
    runtime_strategy = "diffusion_minimax_h3"
    pipeline_classes = ("MiniMaxH3ModularPipeline", "MiniMaxH3Pipeline")

    def matches(self, model_type: str) -> bool:
        return model_type.lower().replace("-", "_") in {
            "minimax_h3",
            "minimaxh3",
            "minimaxh3modularpipeline",
            "minimaxh3pipeline",
        }

    def load_weights(self, model_dir: str, config, **_kwargs) -> dict:
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

    def build_engine(self, *_args, **_kwargs) -> bytes:
        raise NotImplementedError("MiniMax-H3 uses build_components(), not build_engine()")

    def build_components(
        self,
        model_dir: str,
        config,
        weights: dict,
        *,
        precision: str = "bf16",
        verbose: bool = False,
        parallel_config=None,
        **_kwargs,
    ) -> dict:
        del model_dir
        if precision.lower() != "bf16":
            raise ValueError("MiniMax-H3 native builds require BF16 checkpoint weights")
        cp_size = int(getattr(parallel_config, "cp_size", 1))
        mode = str(getattr(parallel_config, "mode", "single"))
        if mode != "context_parallel" or cp_size not in {4, 8}:
            raise ValueError(
                "MiniMax-H3 requires parallel.mode=context_parallel and cp_size=4 or 8"
            )

        raw = getattr(config, "raw", {})
        profile = replace(
            SOL_ENGINE_1344X768_124F,
            context_parallel_size=cp_size,
            text_rows=int(raw.get("text_rows", SOL_ENGINE_1344X768_124F.text_rows)),
            audio_rows=int(raw.get("audio_rows", SOL_ENGINE_1344X768_124F.audio_rows)),
            video_rows=int(raw.get("video_rows", SOL_ENGINE_1344X768_124F.video_rows)),
            padded_sequence_length=int(
                raw.get(
                    "padded_sequence_length",
                    SOL_ENGINE_1344X768_124F.padded_sequence_length,
                )
            ),
        )
        profile.validate()
        from .adaln_builder import build_adaln_precompute_engine
        from .dit_builder import build_dit_engine
        from .text_encoder_builder import (
            build_text_encoder_engine,
            checkpoint_keys as text_encoder_checkpoint_keys,
        )

        text_state = load_selected_component_state_dict(
            weights["_text_encoder_dir"], text_encoder_checkpoint_keys()
        )
        text_weights = numpy_state(text_state)
        del text_state
        text_encoder_plan = build_text_encoder_engine(
            text_weights, sequence_length=profile.text_rows, verbose=verbose
        )
        del text_weights

        state = load_component_state_dict(weights["_transformer_dir"])
        require_keys(
            state,
            (
                "proj_in.weight",
                "audio_proj_in.weight",
                "context_embedder.weight",
                "time_embedder.linear_1.weight",
                "transformer_blocks.49.attn.to_q.weight",
                "transformer_blocks.49.adaln_proj.linear.weight",
                "norm_out.norm.weight",
                "proj_out.weight",
                "audio_proj_out.weight",
            ),
        )
        transformer_weights = numpy_state(state)
        del state
        adaln_plan = build_adaln_precompute_engine(transformer_weights, profile, verbose=verbose)
        denoiser_plan = build_dit_engine(transformer_weights, profile, verbose=verbose)
        del transformer_weights

        from .vae_builder import (
            build_vae_tile_decoder_engine,
            checkpoint_keys as vae_checkpoint_keys,
        )

        vae_state = load_selected_component_state_dict(weights["_vae_dir"], vae_checkpoint_keys())
        vae_weights = numpy_state(vae_state)
        del vae_state
        vae_decoder_plan = build_vae_tile_decoder_engine(vae_weights, verbose=verbose)
        tokenizer_json = (Path(weights["_tokenizer_dir"]) / "tokenizer.json").read_bytes()

        return {
            "text_encoder": text_encoder_plan,
            "adaln_precompute": adaln_plan,
            "denoiser": denoiser_plan,
            "vae_decoder": vae_decoder_plan,
            "profile": profile,
            # Text/VAE paths remain explicit so follow-on native component
            # builders cannot silently substitute a different checkpoint.
            "vae_dir": weights["_vae_dir"],
            "audio_vae_dir": weights["_audio_vae_dir"],
            "tokenizer_dir": weights["_tokenizer_dir"],
            "tokenizer_json": tokenizer_json,
        }

    def diffusion_bundle_sections(
        self, components: dict, *, parallel_config=None
    ) -> list[tuple[str, bytes]]:
        del parallel_config
        return [
            ("text_encoder_plan", components["text_encoder"]),
            ("adaln_precompute_plan", components["adaln_precompute"]),
            ("denoiser_plan_cp", components["denoiser"]),
            ("vae_tile_decoder_plan", components["vae_decoder"]),
            ("tokenizer.json", components["tokenizer_json"]),
        ]

    def diffusion_bundle_config(self, config, *, components: dict) -> dict:
        raw = getattr(config, "raw", {})
        profile = components["profile"]
        return {
            "checkpoint_revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
            "height": int(raw.get("video_height", 768)),
            "width": int(raw.get("video_width", 1344)),
            "num_frames": int(raw.get("video_num_frames", 124)),
            "fps": 24,
            "num_inference_steps": int(raw.get("num_inference_steps", 50)),
            "seed": int(raw.get("seed", 0)),
            "text_rows": profile.text_rows,
            "audio_rows": profile.audio_rows,
            "video_rows": profile.video_rows,
            "padded_sequence_length": profile.padded_sequence_length,
            "max_timestep_count": profile.max_timestep_count,
            "context_parallel_size": profile.context_parallel_size,
            "vae_tile_batch": 7,
            "vae_tile_size": 256,
            "vae_tile_overlap": 64,
        }

    def diffusion_tokenizer_add_special_tokens(self, *_args, **_kwargs) -> bool:
        return False

    def diffusion_tokenizer_bundle_sections(self, *_args, **_kwargs):
        # tokenizer.json is emitted with the model-owned engine sections.
        return []


plugin = MiniMaxH3Plugin()
