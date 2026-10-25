# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit, experimental original-H3 image Ref2VA bundle construction.

Build every component from its own checkpoint partition, stage one plan at a
time, and retain exact reference geometry. This does not declare trained-model
qualification or add reference support to automatic family discovery.
"""

from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .checkpoint import (
    load_selected_component_state_dict,
    numpy_state,
    validate_component_key_partition,
)
from .config import Ref2VAPackedProfile, SOL_ENGINE_1344X768_124F

TASK = "references_text_to_audio_video"
SCHEMA = "minimax_h3.image_ref2va/v1"
CONTINUATION_SCHEMA = "minimax_h3.image_ref2va_continuation/v1"
PLAN_NAMES = (
    "vision.plan",
    "text_encoder.plan",
    "vae_encoder.plan",
    "vae_reference_posterior.plan",
    "adaln.plan",
    "denoiser.plan",
    "vae.plan",
    "audio_vae.plan",
)
TEXT_ROWS = 32768
REFERENCE_ROWS = (4096, 5504, 16384)
REFERENCE_CAPACITIES = {
    "standard": (TEXT_ROWS, REFERENCE_ROWS),
    # Six 9:16 references retain their released 3648x2048 preparation: 43,776
    # rows in both the visual language input and the separate DiT condition.
    # The maximum also covers six references up to 2:1, subject to text budget.
    "six_portrait": (65536, (4096, 43776, 49152)),
}
REFERENCE_PIXELS = (16384, 22016, 65536)
WORKSPACE_BYTES = 8 << 30


def _require(config, expected, name):
    mismatches = {
        key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Unsupported MiniMax-H3 {name} configuration: {mismatches}")


def _statistics(config, channels, name):
    result = {}
    for key in ("latents_mean", "latents_std"):
        values = config.get(key)
        if (
            not isinstance(values, list)
            or len(values) != channels
            or any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                or (key == "latents_std" and x <= 0)
                for x in values
            )
        ):
            raise ValueError(f"MiniMax-H3 {name} requires original finite {channels}-channel {key}")
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            fp32 = np.asarray(values, dtype=np.float32)
        if not np.isfinite(fp32).all() or (key == "latents_std" and np.any(fp32 <= 0)):
            raise ValueError(f"MiniMax-H3 {name} statistics must remain valid at the FP32 boundary")
        result[key] = values
    return result


def _configs(root):
    raw = {
        name: (root / name / "config.json").read_bytes()
        for name in ("transformer_ref", "text_encoder", "vae", "audio_vae")
    }
    configs = {name: json.loads(value) for name, value in raw.items()}
    _require(
        configs["transformer_ref"],
        dict(
            hidden_size=5376,
            num_layers=50,
            num_refiner_layers=2,
            num_attention_heads=56,
            attention_head_dim=128,
            ffn_dim=14336,
            in_channels=24,
            audio_in_channels=32,
            patch_size=[1, 2, 2],
            text_dim=5120,
            freq_dim=256,
            time_embed_hidden_dim=5376,
            time_embed_dim=2688,
            rope_freq_dim=16,
            rope_theta=10000.0,
            norm_eps=1e-5,
            qk_norm_eps=1e-5,
            final_norm_eps=1e-5,
        ),
        "transformer_ref",
    )
    text = configs["text_encoder"]
    _require(
        text,
        dict(
            model_type="qwen3_vl",
            image_token_id=151655,
            vision_start_token_id=151652,
            vision_end_token_id=151653,
        ),
        "Qwen",
    )
    _require(
        text.get("text_config", {}),
        dict(
            hidden_size=5120,
            num_hidden_layers=64,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=25600,
            vocab_size=151936,
            rms_norm_eps=1e-6,
            rope_theta=5000000,
            hidden_act="silu",
            attention_bias=False,
            rope_scaling=dict(
                mrope_interleaved=True, mrope_section=[24, 20, 20], rope_type="default"
            ),
        ),
        "Qwen language",
    )
    _require(
        text.get("vision_config", {}),
        dict(
            depth=27,
            hidden_size=1152,
            intermediate_size=4304,
            num_heads=16,
            in_channels=3,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=5120,
            num_position_embeddings=2304,
            deepstack_visual_indexes=[8, 16, 24],
            hidden_act="gelu_pytorch_tanh",
        ),
        "Qwen vision",
    )
    _require(
        configs["vae"],
        dict(
            in_channels=3,
            out_channels=3,
            latent_channels=24,
            block_out_channels=[128, 256, 256, 512, 512, 1024],
            layers_per_block=2,
            spatial_downsample_factors=[2, 2, 2, 2, 1, 1],
            temporal_downsample_factors=[1, 2, 2, 1, 1, 1],
            norm_num_groups=32,
            norm_eps=1e-6,
            spatial_padding_mode="reflect",
            decoder_num_layers=36,
            decoder_num_attention_heads=32,
            decoder_attention_head_dim=64,
            decoder_num_register_tokens=4,
            decoder_ffn_mult=4,
            decoder_rope_theta=100.0,
            decoder_rope_dim_ratio=0.75,
            decoder_norm_eps=1e-5,
            clip_length=17,
            token_drop=3,
        ),
        "video VAE",
    )
    _require(
        configs["audio_vae"],
        dict(
            latent_dim=2048,
            latent_channels=32,
            decoder_dim=1024,
            decoder_rates=[5, 5, 2, 2, 2, 2, 2],
            decoder_kernel_sizes=[9, 9, 4, 4, 4, 4, 4],
            num_attention_heads=8,
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5]] * 3,
            sampling_rate=32000,
        ),
        "audio VAE",
    )
    _statistics(configs["vae"], 24, "video VAE")
    _statistics(configs["audio_vae"], 32, "audio VAE")
    return configs, {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}


def _target(request, memory_preset, reference_capacity="standard", continuation=False):
    if not isinstance(continuation, bool):
        raise ValueError("MiniMax-H3 continuation must be a boolean")
    if (
        request.family != "minimax_h3"
        or request.task != TASK
        or request.backend != "trt"
        or request.precision != "bf16"
    ):
        raise ValueError(
            "Original image Ref2VA requires family=minimax_h3, its reference AV task, backend=trt and precision=bf16"
        )
    if memory_preset not in ("minimal", "original"):
        raise ValueError("MiniMax-H3 memory_preset must be minimal or original")
    if not isinstance(reference_capacity, str) or reference_capacity not in REFERENCE_CAPACITIES:
        raise ValueError("MiniMax-H3 reference_capacity must be standard or six_portrait")
    text_rows, condition_rows = REFERENCE_CAPACITIES[reference_capacity]
    if (
        request.max_sequence_length is not None
        or request.dynamic_kv_cache
        or request.max_batch_size != 1
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.mtp_seq_len is not None
    ):
        raise ValueError("Original image Ref2VA does not support the requested build overrides")
    height = 1344 if request.image_height is None else request.image_height
    width = 768 if request.image_width is None else request.image_width
    for value in (height, width):
        if isinstance(value, bool) or not isinstance(value, int) or value < 256 or value % 32:
            raise ValueError(
                "H3 output dimensions must be integers >=256 aligned to 32; no resizing is implicit"
            )
    output_frames = 226 if continuation else 124
    if request.video_num_frames not in (None, output_frames):
        raise ValueError(
            f"This original image Ref2VA profile requires {output_frames} output frames"
        )
    video_rows = 37 * (height // 32) * (width // 32)
    target = replace(
        SOL_ENGINE_1344X768_124F,
        video_rows=video_rows,
        text_rows=text_rows,
        padded_sequence_length=video_rows + 414 + text_rows,
    )
    # The first window has no guide. Later windows append seven video latent
    # frames and 37 audio latents per stereo channel to the static references.
    # Keep the minimum video count unchanged so the first window also fits.
    guide_video = 7 * (height // 32) * (width // 32) if continuation else 0
    if continuation:
        condition_rows = (
            condition_rows[0],
            condition_rows[1] + guide_video,
            condition_rows[2] + guide_video,
        )
    reference = Ref2VAPackedProfile(
        (1, 128, text_rows), condition_rows, (0, 74, 74) if continuation else (0, 0, 0)
    )
    reference.validate(target)
    return height, width, target, reference


def runtime_metadata(
    height,
    width,
    target,
    reference,
    configs,
    config_sha256,
    *,
    memory_preset,
    reference_capacity,
    continuation,
):
    """Describe the exact family profile, shared by full and incremental builds."""
    batch = 1 if memory_preset == "minimal" else 28
    return dict(
        schema_version=CONTINUATION_SCHEMA if continuation else SCHEMA,
        workflow="image_ref2va_continuation" if continuation else "image_ref2va",
        weights_variant="transformer_ref",
        memory_preset=memory_preset,
        vae_tile_batch=batch,
        height=height,
        width=width,
        num_frames=226 if continuation else 124,
        fps=24,
        num_inference_steps=50,
        denoiser_forwards=49,
        latent_frames=37,
        latent_height=height // 16,
        latent_width=width // 16,
        audio_latent_frames=207,
        audio_sampling_rate=32000,
        audio_channels=2,
        max_text_rows=reference.text_rows[2],
        max_sequence_rows=reference.sequence_rows(target)[2],
        text_rows=list(reference.text_rows),
        condition_video_rows=list(reference.condition_video_rows),
        condition_audio_rows=list(reference.condition_audio_rows),
        reference_pixels=list(REFERENCE_PIXELS),
        layers=50,
        hidden_size=5376,
        text_width=5120,
        context_parallel_size=1,
        first_block_cache=False,
        config_sha256=config_sha256,
        **(
            dict(
                continuation_window_frames=124,
                continuation_overlap_frames=22,
                continuation_windows=2,
                decoded_latent_frames=67,
                decoded_audio_latent_frames=377,
                total_denoiser_forwards=98,
            )
            if continuation
            else {}
        ),
        **_statistics(configs["vae"], 24, "video VAE"),
        **({"reference_capacity": reference_capacity} if reference_capacity != "standard" else {}),
    )


def build(
    request, writer, *, memory_preset="minimal", reference_capacity="standard", continuation=False
):
    """Build the explicit experimental profile; callers own writer commit/abort.

    Minimal selects staged modules and a static batch-1 video plan; original
    selects resident modules and batch 28. Both use the same trained partitions,
    reference resolution, precision boundaries, target geometry and 49 forwards.
    Reference capacity changes operand profiles, not reference preparation,
    model math or checkpoint partitions. Standard retains the original budget;
    six_portrait requires newly built language/DiT plans, not a metadata edit.
    The complete trained path must still pass its owning qualification gates.

    Continuation is an explicit two-window profile, not a metadata upgrade of
    an existing bundle. It retains two complete 124-frame/49-forward windows,
    discards the second window's 22-frame overlap in latent space, and decodes
    the complete 226-frame history. Canonical images and the prompt are reused
    in both windows; cross-request state and prompt changes are not enabled.
    """
    height, width, target, reference = _target(
        request, memory_preset, reference_capacity, continuation
    )
    root = Path(request.model_dir)
    configs, config_sha256 = _configs(root)
    tokenizer = (root / "tokenizer/tokenizer.json").read_bytes()
    parsed = json.loads(tokenizer)
    if parsed.get("model", {}).get("type") != "BPE":
        raise ValueError("MiniMax-H3 reference conditioning requires its BPE tokenizer")
    tokens = {row["content"]: row["id"] for row in parsed.get("added_tokens", [])}
    _require(
        tokens,
        {"<|vision_start|>": 151652, "<|vision_end|>": 151653, "<|image_pad|>": 151655},
        "reference tokenizer",
    )

    from . import adaln_builder as adaln, audio_vae_builder as audio, dit_builder as dit
    from . import reference_posterior_builder as posterior, text_encoder_builder as text
    from . import vae_builder as video, vae_encoder_builder as encoder, vision_builder as vision

    validate_component_key_partition(
        root / "transformer_ref", [adaln.checkpoint_keys(target), dit.checkpoint_keys(target)]
    )
    validate_component_key_partition(
        root / "vae", [encoder.checkpoint_keys(), video.checkpoint_keys()]
    )
    # Keep each original plan only once on disk. This changes the writer's
    # staging mechanics, not any section bytes, model settings or runtime math.
    writer.enable_direct_staging()
    writer.set_header(family="minimax_h3", task=TASK, backend=request.backend)

    def stage(name, component, keys, builder, *args, **kwargs):
        state = load_selected_component_state_dict(root / component, keys)
        weights = numpy_state(state)
        del state
        payload = builder(
            weights,
            *args,
            verbose=request.verbose,
            consume_weights=True,
            workspace_bytes=WORKSPACE_BYTES,
            **kwargs,
        )
        del weights
        gc.collect()
        writer.add_bytes(name, payload)
        del payload
        gc.collect()

    stage(
        "vision.plan",
        "text_encoder",
        vision.checkpoint_keys(),
        vision.build_vision_encoder_engine,
        patch_rows=REFERENCE_PIXELS,
        precision="bf16",
    )
    stage(
        "text_encoder.plan",
        "text_encoder",
        text.checkpoint_keys(),
        text.build_text_encoder_engine,
        sequence_length=reference.text_rows[2],
        reference=True,
    )
    stage(
        "vae_encoder.plan",
        "vae",
        encoder.checkpoint_keys(),
        encoder.build_vae_tile_encoder_engine,
        max_frames=1,
    )
    payload = posterior.build_reference_posterior_engine(latent_pixels=REFERENCE_PIXELS)
    writer.add_bytes("vae_reference_posterior.plan", payload)
    del payload
    gc.collect()
    stage(
        "adaln.plan",
        "transformer_ref",
        adaln.checkpoint_keys(target),
        adaln.build_adaln_precompute_engine,
        target,
        input_timesteps=True,
    )
    stage(
        "denoiser.plan",
        "transformer_ref",
        dit.checkpoint_keys(target),
        dit.build_dit_engine,
        target,
        reference_profile=reference,
    )
    batch = 1 if memory_preset == "minimal" else 28
    stage(
        "vae.plan",
        "vae",
        video.checkpoint_keys(),
        video.build_vae_tile_decoder_engine,
        batch_size=batch,
    )
    stage(
        "audio_vae.plan",
        "audio_vae",
        audio.checkpoint_keys(),
        audio.build_audio_vae_decoder_engine,
        configs["audio_vae"]["latents_mean"],
        configs["audio_vae"]["latents_std"],
        **({"latent_frames": 377} if continuation else {}),
    )
    writer.add_bytes("tokenizer.json", tokenizer)
    writer.add_json(
        "runtime.json",
        runtime_metadata(
            height,
            width,
            target,
            reference,
            configs,
            config_sha256,
            memory_preset=memory_preset,
            reference_capacity=reference_capacity,
            continuation=continuation,
        ),
    )
