# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import tensorrt as trt


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
