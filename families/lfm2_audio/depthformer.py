# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eight-codebook RQ transformer with native TensorRT cache updates."""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .backbone import _active_rope_cache, add_native_kv_attention
from .graph import Graph
from .native_kv_attention_builder import add_active_prefix_causal_masks


def build_depthformer(weights, *, precision="bf16", verbose=False, fp32_attention=False):
    g = Graph(weights, precision, verbose)
    if fp32_attention:
        g.config.clear_flag(trt.BuilderFlag.TF32)
    embedding = g.net.add_input("input_embedding", trt.float32, (1, 2048))
    previous_embedding = g.net.add_input("previous_embedding", trt.float32, (1, 1024))
    position = g.net.add_input("position_id", trt.int32, (1,))
    index = g.net.add_input("cache_write_indices", trt.int32, (1,))
    length = g.net.add_input("key_value_lengths", trt.int32, (1,))
    x = g.linear(g.cast(embedding, g.dtype), "depth_linear")
    x = g.reshape(x, (8, 1024))
    x = g.net.add_gather(x, position, 0).get_output(0)
    x = g.add(x, g.cast(previous_embedding, g.dtype))
    masks = add_active_prefix_causal_masks(g.net, x, index, length, 8)
    cos, sin = _active_rope_cache(
        g.net, position, head_dim=32, rope_theta=1000000, work_dtype=trt.float32
    )
    presents = []

    def rotate(t, heads):
        h = g.cast(g.reshape(t, (1, heads, 1, 32)), trt.float32)
        rotated = g.net.add_rotary_embedding(h, cos, sin, True, 32).get_output(0)
        return g.cast(g.reshape(rotated, (1, heads * 32)), g.dtype)

    for i in range(6):
        prefix = f"depthformer.layers.{i}"
        h = g.norm(x, prefix + ".operator_norm", rms=True)
        h = g.linear(h, prefix + ".operator.qkv_proj", bias=False)
        q = g.slice(h, (0, 0), (1, 1024))
        k = g.slice(h, (0, 1024), (1, 256))
        v = g.slice(h, (0, 1280), (1, 256))
        q = g.norm(
            g.reshape(q, (1, 32, 32)), prefix + ".operator.bounded_attention.q_layernorm", rms=True
        )
        k = g.norm(
            g.reshape(k, (1, 8, 32)), prefix + ".operator.bounded_attention.k_layernorm", rms=True
        )
        q, k = rotate(q, 32), rotate(k, 8)
        cache_k = g.net.add_input(f"cache_k_{i}", g.dtype, (1, 8, 8, 32))
        cache_v = g.net.add_input(f"cache_v_{i}", g.dtype, (1, 8, 8, 32))
        att = add_native_kv_attention(
            g.net,
            q,
            k,
            v,
            cache_k,
            cache_v,
            index,
            masks,
            num_heads=32,
            num_kv_heads=8,
            head_dim=32,
            work_dtype=g.dtype,
            tag=prefix,
            fp32_attention=fp32_attention,
        )
        presents.append(att)
        x = g.add(x, g.linear(att["context"], prefix + ".operator.out_proj", bias=False))
        h = g.norm(x, prefix + ".ffn_norm", rms=True)
        a = g.silu(g.linear(h, prefix + ".feed_forward.w1", bias=False))
        b = g.linear(h, prefix + ".feed_forward.w3", bias=False)
        x = g.add(x, g.linear(g.mul(a, b), prefix + ".feed_forward.w2", bias=False))
    gamma = g.constant(
        np.stack([weights[f"depth_embeddings.{i}.embedding_norm.weight"] for i in range(8)]).astype(
            np.float32
        ),
        dtype=g.dtype,
    )
    gamma = g.net.add_gather(gamma, position, 0).get_output(0)
    fp = g.cast(x, trt.float32)
    mean = g.net.add_reduce(g.mul(fp, fp), trt.ReduceOperation.AVG, 2, True).get_output(0)
    norm = g.op(
        fp,
        g.unary(g.add(mean, g.scalar(1e-5, 2, trt.float32)), trt.UnaryOperation.SQRT),
        trt.ElementWiseOperation.DIV,
    )
    norm = g.cast(g.mul(norm, g.cast(gamma, trt.float32)), g.dtype)
    projection = g.constant(
        np.stack([weights[f"depth_embeddings.{i}.to_logits.weight"] for i in range(8)]).astype(
            np.float32
        ),
        dtype=g.dtype,
    )
    projection = g.net.add_gather(projection, position, 0).get_output(0)
    projection = g.reshape(projection, (2049, 1024))
    g.output(g.matmul(norm, projection, transpose=True), "logits")
    for i, att in enumerate(presents):
        for kind in ("k", "v"):
            out = att["present_" + kind]
            out.name = f"present_{kind}_{i}"
            g.net.mark_output(out)
    return g.finish()
