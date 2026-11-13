# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import tensorrt as trt


def _prepare(g, cfg, rows, audio_rows):
    width = cfg["hidden_size"]
    context = g.net.add_input("context", trt.bfloat16, (-1, cfg["text_dim"]))
    video = g.net.add_input("video", trt.float32, (rows, 96))
    audio = g.net.add_input("audio", trt.float32, (audio_rows, 32))
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
    timestep = g.net.add_input("timestep", trt.float32, (-1, 1))
    g.profiles["timestep"] = ((1, 1), (2, 1), (2, 1))
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
    time = g.net.add_input("time", trt.bfloat16, (-1, cfg["time_embed_dim"]))
    g.profiles["time"] = tuple((n, cfg["time_embed_dim"]) for n in (1, 2, 2))
    tags = g.net.add_input("adaln_indices", trt.int32, (-1,))
    cos = g.net.add_input("cos", trt.bfloat16, (1, -1, 1, cfg["rope_freq_dim"] * 6))
    sin = g.net.add_input("sin", trt.bfloat16, (1, -1, 1, cfg["rope_freq_dim"] * 6))
    table = g.reshape(g.linear(time, prefix + ".adaln_proj.linear"), (-1, 6 * width))
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
    time = g.net.add_input("time", trt.bfloat16, (-1, cfg["time_embed_dim"]))
    g.profiles["time"] = tuple((n, cfg["time_embed_dim"]) for n in (1, 2, 2))
    indices = g.net.add_input("time_indices", trt.int32, (-1,))
    projected = g.linear(time, "norm_out.linear")
    shift, scale = g.channels(projected, 0, width), g.channels(projected, width, width)
    shift, scale = g.gather(shift, indices), g.gather(scale, indices)
    normal = g.norm(hidden, "norm_out.norm", cfg["final_norm_eps"])
    hidden = g.cast(g.add(g.mul(normal, g.add(scale, 1)), shift), trt.float32)
    g.output(g.linear(hidden, "proj_out"), "video")
    g.output(g.linear(hidden, "audio_proj_out"), "audio")
