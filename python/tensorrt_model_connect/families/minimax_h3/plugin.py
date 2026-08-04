# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-Model-Connect family plugin for MiniMaxAI/MiniMax-H3."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

from .checkpoint import (
    load_selected_component_state_dict,
    numpy_state,
    validate_component_key_partition,
)
from .config import SOL_ENGINE_1344X768_124F
from .provenance import (
    builder_source_sha256,
    checkpoint_snapshot_record,
    validate_source_revision,
)


def _build_source_revision() -> str:
    for name in ("TRTMC_MINIMAX_H3_SOURCE_REVISION", "GITHUB_SHA"):
        revision = os.environ.get(name, "").strip().lower()
        if revision:
            return validate_source_revision(revision)
    raise ValueError(
        "MiniMax-H3 native builds require TRTMC_MINIMAX_H3_SOURCE_REVISION "
        "(or GITHUB_SHA in GitHub Actions)"
    )


def _fixed_profile(raw: dict):
    expected = {
        "text_rows": SOL_ENGINE_1344X768_124F.text_rows,
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
    return SOL_ENGINE_1344X768_124F


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
        if mode != "single" or cp_size != 1:
            raise ValueError("MiniMax-H3 requires parallel.mode=single and cp_size=1")

        raw = getattr(config, "raw", {})
        profile = _fixed_profile(raw)
        profile.validate()
        source_revision = _build_source_revision()
        snapshot = checkpoint_snapshot_record(Path(weights["_model_dir"]))
        from .adaln_builder import build_adaln_precompute_engine
        from .adaln_builder import checkpoint_keys as adaln_checkpoint_keys
        from .dit_builder import build_dit_engine, checkpoint_keys as dit_checkpoint_keys
        from .text_encoder_builder import (
            build_text_encoder_engine,
            checkpoint_keys as text_encoder_checkpoint_keys,
        )

        validate_component_key_partition(
            weights["_transformer_dir"],
            (adaln_checkpoint_keys(profile), dit_checkpoint_keys(profile)),
        )

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
        )
        del adaln_weights
        gc.collect()

        dit_state = load_selected_component_state_dict(
            weights["_transformer_dir"], dit_checkpoint_keys(profile)
        )
        dit_weights = numpy_state(dit_state)
        del dit_state
        denoiser_plan = build_dit_engine(
            dit_weights,
            profile,
            verbose=verbose,
            consume_weights=True,
        )
        del dit_weights
        gc.collect()

        from .vae_builder import (
            build_vae_tile_decoder_engine,
            checkpoint_keys as vae_checkpoint_keys,
        )

        vae_state = load_selected_component_state_dict(weights["_vae_dir"], vae_checkpoint_keys())
        vae_weights = numpy_state(vae_state)
        del vae_state
        vae_decoder_plan = build_vae_tile_decoder_engine(
            vae_weights, verbose=verbose, consume_weights=True
        )
        tokenizer_json = (Path(weights["_tokenizer_dir"]) / "tokenizer.json").read_bytes()

        plan_sha256 = {
            "text_encoder.plan": hashlib.sha256(text_encoder_plan).hexdigest(),
            "adaln_precompute.plan": hashlib.sha256(adaln_plan).hexdigest(),
            "denoiser.plan": hashlib.sha256(denoiser_plan).hexdigest(),
            "vae_tile_decoder.plan": hashlib.sha256(vae_decoder_plan).hexdigest(),
        }

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
            "provenance": {
                "source_revision": source_revision,
                "builder_source_sha256": builder_source_sha256(),
                "checkpoint_inventory_sha256": snapshot["inventory_sha256"],
                "plan_sha256": plan_sha256,
            },
        }

    def diffusion_bundle_sections(
        self, components: dict, *, parallel_config=None
    ) -> list[tuple[str, bytes]]:
        del parallel_config
        return [
            ("text_encoder_plan", components["text_encoder"]),
            ("adaln_precompute_plan", components["adaln_precompute"]),
            ("denoiser_plan", components["denoiser"]),
            ("vae_tile_decoder_plan", components["vae_decoder"]),
            ("tokenizer.json", components["tokenizer_json"]),
        ]

    def diffusion_bundle_config(self, config, *, components: dict) -> dict:
        raw = getattr(config, "raw", {})
        profile = components["profile"]
        fixed_request = {
            "video_height": 768,
            "video_width": 1344,
            "video_num_frames": 124,
            "num_inference_steps": 50,
        }
        mismatches = {
            name: (raw[name], value)
            for name, value in fixed_request.items()
            if name in raw and int(raw[name]) != value
        }
        if mismatches:
            raise ValueError(f"Unsupported MiniMax-H3 runtime profile: {mismatches}")
        provenance = components.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("MiniMax-H3 components are missing exact build provenance")
        return {
            "checkpoint_revision": "48d93ede732756e404a3b1b2f3b3a9b5a22f6cfc",
            **provenance,
            "height": 768,
            "width": 1344,
            "num_frames": 124,
            "fps": 24,
            "num_inference_steps": 50,
            "seed": int(raw.get("seed", 0)),
            "bundle_loading": {
                "mode": "staged",
                "eager_sections": ["tokenizer.json", "config.json"],
                "lazy_sections": [
                    "text_encoder_plan",
                    "adaln_precompute_plan",
                    "denoiser_plan",
                    "vae_tile_decoder_plan",
                ],
            },
            "text_rows": profile.text_rows,
            "audio_rows": profile.audio_rows,
            "video_rows": profile.video_rows,
            "padded_sequence_length": profile.padded_sequence_length,
            "max_timestep_count": profile.max_timestep_count,
            "context_parallel_size": profile.context_parallel_size,
            "vae_tile_batch": 28,
            "vae_tile_size": 256,
            "vae_tile_overlap": 64,
        }

    def diffusion_tokenizer_add_special_tokens(self, *_args, **_kwargs) -> bool:
        return False

    def diffusion_tokenizer_bundle_sections(self, *_args, **_kwargs):
        # tokenizer.json is emitted with the model-owned engine sections.
        return []


plugin = MiniMaxH3Plugin()
