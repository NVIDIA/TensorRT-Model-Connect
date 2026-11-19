# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal-prompt P-H3-Image: Qwen encoder, joint denoiser and tiled ViT VAE."""

import gc
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import tensorrt as trt

from .graph import Graph, Weights


def _text_layer(g, cfg, index):
    width, heads, kv, dim = (
        cfg[k] for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim")
    )
    eps = cfg["rms_norm_eps"]
    name = f"model.language_model.layers.{index}"
    hidden = g.net.add_input("hidden", trt.bfloat16, (-1, width))
    cos = g.net.add_input("cos", trt.bfloat16, (1, -1, 1, dim))
    sin = g.net.add_input("sin", trt.bfloat16, (1, -1, 1, dim))
    mask = g.net.add_input("mask", trt.bfloat16, (1, 1, -1, -1))
    normal = g.norm(hidden, name + ".input_layernorm", eps, round_before_affine=True)
    projections = []
    for kind in ("q", "k", "v"):
        value = g.reshape(
            g.linear(normal, name + f".self_attn.{kind}_proj"),
            (1, -1, heads if kind == "q" else kv, dim),
        )
        if kind != "v":
            value = g.norm(value, name + f".self_attn.{kind}_norm", eps, round_before_affine=True)
            value = g.rope(value, cos, sin, dim)
        if kind != "q":
            value = g.gather(value, g.const(np.repeat(np.arange(kv), heads // kv), trt.int32), 2)
        projections.append(g.transpose(value, (0, 2, 1, 3)))
    query, key, value = projections
    scores = g.mm(query, key, True)
    scores = g.cast(g.mul(g.cast(scores, trt.float32), dim**-0.5), trt.bfloat16)
    scores = g.add(scores, mask)
    probability = g.net.add_softmax(g.cast(scores, trt.float32))
    probability.axes = 8
    attended = g.mm(g.cast(probability.get_output(0), trt.bfloat16), value)
    attended = g.reshape(g.transpose(attended, (0, 2, 1, 3)), (-1, heads * dim))
    hidden = g.add(hidden, g.linear(attended, name + ".self_attn.o_proj"))
    normal = g.norm(hidden, name + ".post_attention_layernorm", eps, round_before_affine=True)
    gated = g.mul(
        g.silu(g.linear(normal, name + ".mlp.gate_proj")), g.linear(normal, name + ".mlp.up_proj")
    )
    g.output(g.add(hidden, g.linear(gated, name + ".mlp.down_proj")), "result")


def _prepare(g, cfg, rows):
    width = cfg["hidden_size"]
    context = g.net.add_input("context", trt.bfloat16, (-1, cfg["text_dim"]))
    video = g.net.add_input("video", trt.float32, (rows, 96))
    audio = g.net.add_input("audio", trt.float32, (4, 32))
    text = g.linear(context, "context_embedder")
    for index in range(cfg["num_refiner_layers"]):
        prefix = f"token_refiner.refiner_blocks.{index}"
        normal = g.norm(text, prefix + ".norm1", cfg["norm_eps"])
        text = g.add(
            text,
            g.attention(
                normal,
                prefix + ".attn",
                cfg["num_attention_heads"],
                cfg["attention_head_dim"],
                cfg["qk_norm_eps"],
            ),
        )
        normal = g.norm(text, prefix + ".norm2", cfg["norm_eps"])
        text = g.add(text, g.ff(normal, prefix + ".ff", cfg["ffn_dim"]))
    text = g.norm(text, "token_refiner.final_norm", cfg["final_norm_eps"])
    packed = g.concat(
        [
            text,
            g.cast(g.linear(audio, "audio_proj_in"), trt.bfloat16),
            g.cast(g.linear(video, "proj_in"), trt.bfloat16),
        ],
        0,
    )
    assert packed.shape[-1] == width
    g.output(packed, "result")


def _time(g, cfg):
    timestep = g.net.add_input("timestep", trt.float32, (1, 1))
    half = cfg["freq_dim"] // 2
    frequency = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float32) / half).astype(
        np.float32
    )
    angles = g.mul(timestep, g.const(frequency.reshape(1, -1), trt.float32))
    cosine = g.net.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sine = g.net.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    embedded = g.linear(
        g.silu(g.linear(g.concat([cosine, sine], 1), "time_embedder.linear_1")),
        "time_embedder.linear_2",
    )
    # Every AdaLN projection applies SiLU in FP32, then converts to BF16.
    g.output(g.cast(g.silu(embedded), trt.bfloat16), "result")


def _denoise_layer(g, cfg, index):
    width = cfg["hidden_size"]
    prefix = f"transformer_blocks.{index}"
    hidden = g.net.add_input("hidden", trt.bfloat16, (-1, width))
    time = g.net.add_input("time", trt.bfloat16, (1, cfg["time_embed_dim"]))
    tags = g.net.add_input("tags", trt.int32, (-1,))
    cos = g.net.add_input("cos", trt.bfloat16, (1, -1, 1, cfg["rope_freq_dim"] * 6))
    sin = g.net.add_input("sin", trt.bfloat16, (1, -1, 1, cfg["rope_freq_dim"] * 6))
    table = g.reshape(g.linear(time, prefix + ".adaln_proj.linear"), (3, 6 * width))
    table = g.gather(table, tags)
    shift1, scale1, gate1, shift2, scale2, gate2 = [
        g.channels(table, i * width, width) for i in range(6)
    ]
    normal = g.add(
        g.mul(g.norm(hidden, prefix + ".norm1", cfg["norm_eps"]), g.add(scale1, 1)), shift1
    )
    attention = g.attention(
        normal,
        prefix + ".attn",
        cfg["num_attention_heads"],
        cfg["attention_head_dim"],
        cfg["qk_norm_eps"],
        cos=cos,
        sin=sin,
        rotary_dim=cfg["rope_freq_dim"] * 6,
    )
    hidden = g.add(hidden, g.mul(gate1, attention))
    normal = g.add(
        g.mul(g.norm(hidden, prefix + ".norm2", cfg["norm_eps"]), g.add(scale2, 1)), shift2
    )
    g.output(g.add(hidden, g.mul(gate2, g.ff(normal, prefix + ".ff", cfg["ffn_dim"]))), "result")


def _denoise_head(g, cfg):
    width = cfg["hidden_size"]
    hidden = g.net.add_input("hidden", trt.bfloat16, (-1, width))
    time = g.net.add_input("time", trt.bfloat16, (1, cfg["time_embed_dim"]))
    projected = g.linear(time, "norm_out.linear")
    shift, scale = g.channels(projected, 0, width), g.channels(projected, width, width)
    normal = g.norm(hidden, "norm_out.norm", cfg["final_norm_eps"])
    hidden = g.cast(g.add(g.mul(normal, g.add(scale, 1)), shift), trt.float32)
    g.output(g.linear(hidden, "proj_out"), "video")
    g.output(g.linear(hidden, "audio_proj_out"), "audio")


def _vae_input(g, cfg):
    # The upstream default uses 256px tiles, with boundaries aligned to the
    # 16px latent stride. A tile contains 16x16 latent pixels.
    latents = g.net.add_input("latents", trt.float32, (256, 24))
    hidden = g.cast(latents, trt.bfloat16)
    weight = g.weights.read("post_quant_conv.weight").reshape(24, 24)
    hidden = g.affine_bias(
        g.mm(hidden, g.const(weight), True),
        g.const(g.weights.read("post_quant_conv.bias").reshape(1, 24)),
    )
    hidden = g.linear(hidden, "decoder.proj_in")
    registers = g.const(
        g.weights.read("decoder.register_tokens").reshape(cfg["decoder_num_register_tokens"], -1)
    )
    cls = g.const(
        np.zeros((1, cfg["decoder_num_attention_heads"] * cfg["decoder_attention_head_dim"]))
    )
    g.output(g.concat([hidden, registers, cls], 0), "result")


def _vae_layer(g, cfg, index):
    width = cfg["decoder_num_attention_heads"] * cfg["decoder_attention_head_dim"]
    prefix = f"decoder.transformer_blocks.{index}"
    hidden = g.net.add_input("hidden", trt.bfloat16, (261, width))
    # Coordinates are constants for the upstream default 256px tile.
    grid = np.stack(
        np.meshgrid(
            np.array([2 * (0.5 / 7) - 1], np.float32),
            2 * (np.arange(16, dtype=np.float32) + 0.5) / 16 - 1,
            2 * (np.arange(16, dtype=np.float32) + 0.5) / 16 - 1,
            indexing="ij",
        ),
        -1,
    ).reshape(256, 3)
    grid = np.concatenate([grid, np.zeros((5, 3), np.float32)])
    rotary_dim = int(cfg["decoder_attention_head_dim"] * cfg["decoder_rope_dim_ratio"])
    frequency = 1 / cfg["decoder_rope_theta"] ** np.arange(0, 1, 6 / rotary_dim, dtype=np.float32)
    angles = (2 * np.float32(np.pi) * grid[..., None] * frequency).reshape(261, -1)
    angles = np.concatenate([angles, angles], -1).reshape(1, 261, 1, rotary_dim)
    cos, sin = g.const(np.cos(angles)), g.const(np.sin(angles))
    normal = g.norm(hidden, prefix + ".norm1", cfg["decoder_norm_eps"])
    attention = g.attention(
        normal,
        prefix + ".attn",
        cfg["decoder_num_attention_heads"],
        cfg["decoder_attention_head_dim"],
        cfg["decoder_norm_eps"],
        cos=cos,
        sin=sin,
        rotary_dim=rotary_dim,
        bias=True,
        affine=False,
    )
    hidden = g.add(
        hidden, g.mul(attention, g.const(g.weights.read(prefix + ".scale1").reshape(1, -1)))
    )
    normal = g.norm(hidden, prefix + ".norm2", cfg["decoder_norm_eps"])
    forward = g.ff(normal, prefix + ".ff", width * cfg["decoder_ffn_mult"], bias=True)
    g.output(
        g.add(hidden, g.mul(forward, g.const(g.weights.read(prefix + ".scale2").reshape(1, -1)))),
        "result",
    )


def _vae_head(g, cfg):
    width = cfg["decoder_num_attention_heads"] * cfg["decoder_attention_head_dim"]
    hidden = g.net.add_input("hidden", trt.bfloat16, (261, width))
    normal = g.norm(hidden, "decoder.norm_out", cfg["decoder_norm_eps"], layer=True)
    g.output(g.linear(g.cast(normal, trt.float32), "decoder.proj_out"), "result")


def build(request, writer):
    if request.backend != "trt" or request.task != "text_to_image":
        raise ValueError("P-H3-Image requires text_to_image with backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("p_h3_image does not support dynamic_kv_cache")
    if (
        request.precision != "bf16"
        or request.quantization not in (None, "none")
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.fp32_layers
        or request.dynamic_kv_cache
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError("P-H3-Image supports one BF16 image with mixed FP32 modules")
    height, width = request.image_height or 1024, request.image_width or 2048
    if height % 256 or width % 256 or height * width < 2 * 1024**2 or height * width > 9 * 1024**2:
        raise ValueError("P-H3-Image profiles require 2–9 megapixels and 256px aligned sides")
    length = request.max_sequence_length or 256
    if length < 1 or length > 256:
        raise ValueError("P-H3-Image supports up to 256 prompt tokens")
    directory = Path(request.model_dir)
    cfg = {
        name: json.loads((directory / name / "config.json").read_text())
        for name in ("text_encoder", "text_connector", "transformer", "vae")
    }
    tc, dc, vc = cfg["text_encoder"]["text_config"], cfg["transformer"], cfg["vae"]
    index = json.loads((directory / "modular_model_index.json").read_text())
    if index.get("_class_name") != "PH3ImageModularPipeline" or tc["model_type"] != "qwen3_vl_text":
        raise ValueError("Expected the P-H3-Image Qwen3-VL checkpoint")
    if (
        dc["patch_size"] != [1, 2, 2]
        or dc["in_channels"] != 24
        or dc["audio_in_channels"] != 32
        or vc["decoder_num_register_tokens"] != 4
        or vc["decoder_attention_head_dim"] != 64
    ):
        raise NotImplementedError("Unsupported P-H3-Image component layout")
    scheduler = json.loads((directory / "scheduler/scheduler_config.json").read_text())
    audio_scheduler = json.loads((directory / "audio_scheduler/scheduler_config.json").read_text())
    if scheduler != audio_scheduler:
        raise NotImplementedError("P-H3-Image requires matching image and audio schedules")
    if (
        scheduler.get("stochastic_sampling", False)
        or scheduler.get("shift_provided_sigmas") is not False
        or scheduler.get("euler_update_form", "blend") != "blend"
    ):
        raise NotImplementedError(
            "P-H3-Image requires the checkpoint's fixed deterministic blend schedule"
        )
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    writer.set_header(family="p_h3_image", task=request.task, backend=request.backend)

    def emit(weights, name, populate, bounds=(1, min(64, length), length)):
        graph = Graph(weights, logger, *bounds)
        populate(graph)
        with writer.open_section(name + ".plan") as section:
            graph.save(section)
        del graph
        gc.collect()

    with ExitStack() as stack:
        weights = {name: Weights(directory / name, stack) for name in cfg}
        tw, cw, dw, vw = [
            weights[name] for name in ("text_encoder", "text_connector", "transformer", "vae")
        ]

        def embedding(g):
            ids = g.net.add_input("input_ids", trt.int32, (-1,))
            g.output(
                g.gather(g.const(tw.read("model.language_model.embed_tokens.weight")), ids),
                "result",
            )

        emit(tw, "text.embedding", embedding)
        for layer in range(tc["num_hidden_layers"]):
            emit(tw, f"text.layer.{layer}", lambda g, layer=layer: _text_layer(g, tc, layer))

        def final_norm(g):
            hidden = g.net.add_input("hidden", trt.bfloat16, (-1, tc["hidden_size"]))
            g.output(
                g.norm(
                    hidden,
                    "model.language_model.norm",
                    tc["rms_norm_eps"],
                    round_before_affine=True,
                ),
                "result",
            )

        emit(tw, "text.norm", final_norm)

        def connector(g):
            hidden = g.net.add_input(
                "hidden",
                trt.bfloat16,
                (-1, len(cfg["text_connector"]["hidden_state_indices"]) * tc["hidden_size"]),
            )
            normalized = g.div(
                g.op(
                    g.cast(hidden, trt.float32),
                    g.const(cw.read("input_mean").reshape(1, -1), trt.float32),
                    trt.ElementWiseOperation.SUB,
                ),
                g.const(cw.read("input_std").reshape(1, -1), trt.float32),
            )
            projected = g.add(
                g.mul(
                    g.linear(normalized, "proj"),
                    g.const(cw.read("output_std").reshape(1, -1), trt.float32),
                ),
                g.const(cw.read("output_mean").reshape(1, -1), trt.float32),
            )
            # The native owner replaces row zero with the checkpoint sink.
            g.output(g.cast(projected, trt.bfloat16), "result")

        emit(cw, "text.connector", connector)
        writer.add_bytes(
            "text.sink.f32", np.asarray(cw.read("sink_embedding"), dtype="<f4").tobytes()
        )
        rows = height * width // 1024
        emit(dw, "denoise.prepare", lambda g: _prepare(g, dc, rows))
        emit(dw, "denoise.time", lambda g: _time(g, dc))
        for layer in range(dc["num_layers"]):
            emit(
                dw,
                f"denoise.layer.{layer}",
                lambda g, layer=layer: _denoise_layer(g, dc, layer),
                (rows + 5, rows + min(64, length) + 4, rows + length + 4),
            )
        emit(
            dw,
            "denoise.head",
            lambda g: _denoise_head(g, dc),
            (rows + 5, rows + min(64, length) + 4, rows + length + 4),
        )
        emit(vw, "vae.input", lambda g: _vae_input(g, vc))
        for layer in range(vc["decoder_num_layers"]):
            emit(vw, f"vae.layer.{layer}", lambda g, layer=layer: _vae_layer(g, vc, layer))
        emit(vw, "vae.head", lambda g: _vae_head(g, vc))
    writer.add_bytes("tokenizer.json", (directory / "tokenizer/tokenizer.json").read_bytes())
    writer.add_json(
        "runtime.json",
        dict(
            height=height,
            width=width,
            max_prompt_tokens=length,
            text=tc,
            connector=cfg["text_connector"],
            denoiser=dc,
            vae=vc,
            sigmas=index["sample_sigmas"],
        ),
    )
