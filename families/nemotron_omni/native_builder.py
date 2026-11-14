# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete native Omni engine composition, independent of Edge installation."""

from pathlib import Path

import numpy as np

from .audio_graph import build_audio
from .checkpoint_mapper import ComponentWeights, _open_safetensors
from .config import OmniConfig
from .decoder import OmniDecoder
from .prepared_quantization import PreparedQuantContext, checkpoint_policy
from .preprocessing import mel_filters
from .visual_graph import build_patch, build_visual


def build_native(request, writer, *, max_audio_frames=1600, max_visual_patches=1024, verbose=False):
    """Build actual decoder, image, temporal-video and acoustic networks."""
    if (
        request.max_batch_size != 1
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.dynamic_kv_cache
        or request.fp32_layers
        or request.graph_transform is not None
        or request.mtp_seq_len is not None
    ):
        raise ValueError(
            "Native Omni currently implements one non-speculative request on one TensorRT device"
        )
    if any(
        value is not None
        for value in (request.image_height, request.image_width, request.video_num_frames)
    ):
        raise ValueError(
            "Omni uses dynamic family-owned media profiles, not fixed generic media dimensions"
        )
    root = Path(request.model_dir)
    config = OmniConfig.from_dir(root)
    precision = request.precision.lower()
    if request.backend != "trt" or precision not in {"fp16", "fp32"}:
        raise ValueError("Native Omni requires TensorRT with fp16 or fp32 computation")
    capacity = 2048 if request.max_sequence_length is None else request.max_sequence_length
    if type(capacity) is not int or not 1 <= capacity <= config.decoder.max_position_embeddings:
        raise ValueError("Omni decoder capacity must be within the checkpoint context limit")
    if type(max_audio_frames) is not int or max_audio_frames < 8:
        raise ValueError("Omni acoustic capacity must be at least eight frames")
    if type(max_visual_patches) is not int or max_visual_patches < 4 or max_visual_patches % 4:
        raise ValueError("Omni visual capacity must contain complete 2x2 patch groups")
    policy = checkpoint_policy(root)
    if policy is None or policy.request_format != "nvfp4":
        raise ValueError("Omni integration requires its documented prepared NVFP4 checkpoint")
    if request.quantization not in {None, "none", "nvfp4"}:
        raise ValueError("Omni preserves prepared NVFP4/FP8 weights; it does not recalibrate them")
    readers = _open_safetensors(root, policy=policy)
    decoder = OmniDecoder()
    quantization = PreparedQuantContext()
    weights = decoder.load_weights(
        root,
        config.decoder,
        quant_ctx=quantization,
        prepared_policy=policy,
        readers=readers,
        precision=precision,
    )
    state = decoder.get_bundle_config_overrides(config.decoder)
    state["fp8_kv_cache"] = bool(weights["_fp8_kv_cache"])
    writer.add_bytes(
        "decoder.plan",
        decoder.build_engine(
            config.decoder,
            weights,
            capacity,
            precision=precision,
            quant_ctx=quantization,
            verbose=verbose,
        ),
    )
    with writer.open_section("embeddings.f32") as output:
        weights["embedding"].astype("<f4", copy=False).tofile(output)
    del weights
    component = ComponentWeights(readers)
    for video, name in ((False, "image_patch.plan"), (True, "video_patch.plan")):
        writer.add_bytes(
            name,
            build_patch(
                config,
                component,
                max_visual_patches,
                video=video,
                precision=precision,
                verbose=verbose,
            ),
        )
    writer.add_bytes(
        "visual.plan",
        build_visual(config, component, max_visual_patches, precision=precision, verbose=verbose),
    )
    writer.add_bytes(
        "audio.plan",
        build_audio(config, component, max_audio_frames, precision=precision, verbose=verbose),
    )
    position = component["vision_model.radio_model.model.patch_generator.pos_embed"]
    if (
        position.ndim != 3
        or position.shape[0] != 1
        or position.shape[2] != config.raw["vit_hidden_size"]
    ):
        raise ValueError("Omni visual position table is malformed")
    side = int(np.sqrt(position.shape[1]))
    if side * side != position.shape[1]:
        raise ValueError("Omni visual position table must have a square source grid")
    writer.add_bytes("visual_positions.f32", position.astype("<f4").tobytes())
    writer.add_bytes("mel_filters.f32", mel_filters().astype("<f4", copy=False).tobytes())
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "generation_config.json",
    ):
        writer.add_bytes(name, (root / name).read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "format": 1,
            "config": config.raw,
            "precision": precision,
            "state": state,
            "max_cache_length": capacity,
            "max_audio_frames": max_audio_frames,
            "max_visual_patches": max_visual_patches,
            "visual_position_side": side,
        },
    )
