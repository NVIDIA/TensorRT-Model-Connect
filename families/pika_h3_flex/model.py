# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex text-to-audio/video, with its released FL2VA adapter."""

import gc
import json
from contextlib import ExitStack
from pathlib import Path

import tensorrt as trt

from . import audio_decoder, denoiser, video_decoder
from .graph import Graph, Weights
from .text import _text_layer


def build(request, writer):
    if request.task != "text_to_audio_video" or request.backend != "trt":
        raise ValueError("Pika H3 Flex requires text_to_audio_video with backend=trt")
    if (
        request.precision != "bf16"
        or request.quantization not in (None, "none")
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.fp32_layers
        or request.dynamic_kv_cache
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError("Pika H3 Flex supports one BF16 request with FP32 media decoders")
    height, width = request.image_height or 480, request.image_width or 832
    frames = request.video_num_frames or 120
    if (height, width) != (480, 832) or frames not in (120, 124):
        raise NotImplementedError(
            "The Pika H3 Flex profile is a five-second 832x480 video (124 aligned frames)"
        )
    length = request.max_sequence_length or 128
    if not 1 <= length <= 128:
        raise ValueError("Pika H3 Flex supports 1–128 prompt tokens")
    directory = Path(request.model_dir)
    cfg = {
        name: json.loads((directory / name / "config.json").read_text())
        for name in ("text_encoder", "transformer", "vae", "audio_vae")
    }
    tc, dc, vc, ac = [cfg[name] for name in ("text_encoder", "transformer", "vae", "audio_vae")]
    tc = tc["text_config"]
    index = json.loads((directory / "modular_model_index.json").read_text())
    adapter_config = json.loads((directory / "fl2va/manifest.json").read_text())
    if index.get("_class_name") != "PikaH3FlexModularPipeline":
        raise ValueError("Expected the Pika H3 Flex checkpoint")
    if (
        tc["hidden_size"] != 5120
        or tc["num_hidden_layers"] != 64
        or tc["head_dim"] != 128
        or dc["hidden_size"] != 5376
        or dc["num_layers"] != 50
        or dc["patch_size"] != [1, 2, 2]
        or vc["decoder_num_layers"] != 36
        or vc["decoder_num_attention_heads"] != 32
        or vc["decoder_attention_head_dim"] != 64
        or vc["decoder_num_register_tokens"] != 4
        or vc["clip_length"] != 17
        or vc["token_drop"] != 3
        or ac["decoder_dim"] != 1024
        or ac["latent_channels"] != 32
        or ac["sampling_rate"] != 32000
        or ac["decoder_rates"] != [5, 5, 2, 2, 2, 2, 2]
    ):
        raise NotImplementedError("Unsupported Pika H3 Flex component topology")
    if (
        adapter_config["rank"] != 128
        or adapter_config["alpha"] != 128
        or adapter_config["task"] != "fl2va"
    ):
        raise NotImplementedError(
            "Pika H3 Flex requires its released scale-one rank-128 FL2VA adapter"
        )
    schedules = [
        json.loads((directory / name / "scheduler_config.json").read_text())
        for name in ("scheduler", "audio_scheduler")
    ]
    for schedule, shift in zip(schedules, (12.0, 3.0)):
        if (
            schedule["shift"] != shift
            or schedule.get("shift_provided_sigmas") is not True
            or schedule.get("stochastic_sampling") is not True
        ):
            raise NotImplementedError(
                "Pika H3 Flex requires its two shifted stochastic six-step schedules"
            )
    if adapter_config["sampling"]["knots"] != [1000, 750, 500, 250, 100, 40, 0]:
        raise NotImplementedError("Unsupported Pika denoising grid")
    # The source aligns 120 frames to 17*7+5=124, i.e. 5*7+2 video
    # latent frames and round(124/24*40)=207 audio latents per channel.
    video_rows, audio_rows = 37 * 15 * 26, 207 * 2
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    writer.set_header(family="pika_h3_flex", task=request.task, backend=request.backend)

    def emit(weights, name, populate, bounds=(1, min(64, length), length)):
        graph = Graph(weights, logger, *bounds)
        populate(graph)
        with writer.open_section(name + ".plan") as section:
            graph.save(section)
        del graph
        gc.collect()

    with ExitStack() as stack:
        tw, dw, vw, aw = [
            Weights(directory / name, stack)
            for name in ("text_encoder", "transformer", "vae", "audio_vae")
        ]
        adapter = Weights(directory / "fl2va", stack)
        if len(adapter.index) != 624:
            raise ValueError("The Pika FL2VA adapter is incomplete")
        for key in adapter.index:
            base = key.replace(".lora_A.default.weight", ".weight").replace(
                ".lora_B.default.weight", ".weight"
            )
            if base not in dw.index:
                raise ValueError(f"Unknown Pika LoRA target: {key}")
        dw.adapter = adapter

        def embedding(g):
            ids = g.net.add_input("input_ids", trt.int32, (-1,))
            g.output(
                g.gather(g.const(tw.read("model.language_model.embed_tokens.weight")), ids),
                "result",
            )

        emit(tw, "text.embedding", embedding)
        # hidden_states[50] is before the remaining 14 layers and final norm.
        for layer in range(50):
            emit(tw, f"text.layer.{layer}", lambda g, layer=layer: _text_layer(g, tc, layer))
        emit(dw, "denoise.prepare", lambda g: denoiser._prepare(g, dc, video_rows, audio_rows))
        emit(dw, "denoise.time", lambda g: denoiser._time(g, dc))
        bounds = tuple(video_rows + audio_rows + n for n in (1, min(64, length), length))
        for layer in range(50):
            emit(
                dw,
                f"denoise.layer.{layer}",
                lambda g, layer=layer: denoiser._denoise_layer(g, dc, layer),
                bounds,
            )
        emit(dw, "denoise.head", lambda g: denoiser._denoise_head(g, dc), bounds)
        emit(vw, "video.input", lambda g: video_decoder.input_graph(g, vc))
        for layer in range(36):
            emit(
                vw,
                f"video.layer.{layer}",
                lambda g, layer=layer: video_decoder.layer_graph(g, vc, layer),
            )
        emit(vw, "video.head", lambda g: video_decoder.head_graph(g, vc))
        emit(aw, "audio.input", lambda g: audio_decoder.input_graph(g, 207))
        audio_frames = 207
        for stage, rate in enumerate(ac["decoder_rates"]):
            emit(
                aw,
                f"audio.stage.{stage}",
                lambda g, stage=stage, audio_frames=audio_frames: audio_decoder.stage_graph(
                    g, ac, stage, audio_frames
                ),
            )
            audio_frames *= rate
        emit(aw, "audio.head", lambda g: audio_decoder.head_graph(g, audio_frames))
    writer.add_bytes("tokenizer.json", (directory / "tokenizer/tokenizer.json").read_bytes())
    writer.add_json(
        "runtime.json",
        dict(
            height=480,
            width=832,
            frames=124,
            latent_frames=37,
            audio_latents=207,
            max_prompt_tokens=length,
            text=tc,
            denoiser=dc,
            vae=vc,
            audio_vae=ac,
            knots=adapter_config["sampling"]["knots"],
            video_shift=12.0,
            audio_shift=3.0,
        ),
    )
