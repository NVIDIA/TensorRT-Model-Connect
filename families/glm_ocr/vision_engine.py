# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM-OCR vision tower, 2x2 downsample and patch merger as one TensorRT engine.

One call encodes one image. Inputs are the processor's flattened patches
``pixel_patches [N, C*T*P*P]`` in its merge-window order and the axial rotary
table of each patch, ``rotary_cos`` / ``rotary_sin [N, head_dim / 2]`` (the
height frequencies followed by the width frequencies). The output is
``image_embeds [N / 4, out_hidden_size]`` in float32, one row per image token
of the prompt. Attention spans the whole image, as for one entry of the
Transformers ``cu_seqlens``.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint, GlmOcrConfig
from .graph import DTYPES, Net, build_engine

_PREFIX = "model.visual"
# Tactic selection point: a typical cropped document region.
_OPT_PATCHES = 2048
# torch.nn.LayerNorm default, used by the merger's post-projection norm.
_LAYER_NORM_EPS = 1e-5


def _rotate(net: Net, tensor: trt.ITensor, cos: trt.ITensor, sin: trt.ITensor) -> trt.ITensor:
    """Axial rotary embedding of ``[N, heads, head_dim]`` in FP32 (rotate-half layout)."""
    head_dim = tensor.shape[-1]
    half = head_dim // 2
    wide = net.cast(tensor, trt.float32)
    first = net.gather(wide, net.indices(np.arange(half)), 2)
    second = net.gather(wide, net.indices(np.arange(half, head_dim)), 2)
    out_first = net.sub(net.mul(first, cos), net.mul(second, sin))
    out_second = net.add(net.mul(second, cos), net.mul(first, sin))
    return net.cast(net.concat([out_first, out_second], 2), tensor.dtype)


def _block(net: Net, x: trt.ITensor, index: int, ck: Checkpoint, config: GlmOcrConfig,
           cos: trt.ITensor, sin: trt.ITensor) -> trt.ITensor:
    v = config.vision
    p = f"{_PREFIX}.blocks.{index}"
    width, heads, head_dim = v.hidden_size, v.num_heads, v.head_dim

    h = net.rms_norm(x, ck.get(f"{p}.norm1.weight"), v.rms_norm_eps)
    qkv_w, qkv_b = ck.get(f"{p}.attn.qkv.weight"), ck.get(f"{p}.attn.qkv.bias")
    projected = []
    for part in range(3):
        rows = slice(part * width, (part + 1) * width)
        flat = net.linear(h, qkv_w[rows], qkv_b[rows])
        projected.append(net.reshape(flat, (0, heads, head_dim)))
    query, key, value = projected
    query = net.rms_norm(query, ck.get(f"{p}.attn.q_norm.weight"), v.rms_norm_eps)
    key = net.rms_norm(key, ck.get(f"{p}.attn.k_norm.weight"), v.rms_norm_eps)
    query = _rotate(net, query, cos, sin)
    key = _rotate(net, key, cos, sin)

    def heads_first(tensor: trt.ITensor) -> trt.ITensor:
        return net.reshape(tensor, (1, -1, heads, head_dim), second_transpose=(0, 2, 1, 3))

    context = net.attention(heads_first(query), heads_first(key), heads_first(value),
                            causal=False, scale=head_dim ** -0.5)
    context = net.transpose_reshape(context, (0, 2, 1, 3), (-1, width))
    attended = net.linear(context, ck.get(f"{p}.attn.proj.weight"), ck.get(f"{p}.attn.proj.bias"))
    x = net.add(x, attended)

    h = net.rms_norm(x, ck.get(f"{p}.norm2.weight"), v.rms_norm_eps)
    gate = net.linear(h, ck.get(f"{p}.mlp.gate_proj.weight"), ck.get(f"{p}.mlp.gate_proj.bias"))
    up = net.linear(h, ck.get(f"{p}.mlp.up_proj.weight"), ck.get(f"{p}.mlp.up_proj.bias"))
    down = net.linear(net.mul(net.silu(gate), up),
                      ck.get(f"{p}.mlp.down_proj.weight"), ck.get(f"{p}.mlp.down_proj.bias"))
    return net.add(x, down)


def _merge(net: Net, x: trt.ITensor, ck: Checkpoint, config: GlmOcrConfig) -> trt.ITensor:
    """RMSNorm, 2x2 strided convolution and the gated merger, over merge windows of 4 rows."""
    v = config.vision
    x = net.rms_norm(x, ck.get(f"{_PREFIX}.post_layernorm.weight"), v.rms_norm_eps)
    # Four consecutive patch rows form one merge window in (row, column) order, so the
    # window flattens to [kh, kw, channel], the convolution weight read in that order.
    windows = net.reshape(x, (-1, v.merge_area * v.hidden_size))
    kernel = ck.get(f"{_PREFIX}.downsample.weight").transpose(0, 2, 3, 1)
    merged = net.linear(windows, kernel.reshape(v.out_hidden_size, -1),
                        ck.get(f"{_PREFIX}.downsample.bias"))

    m = f"{_PREFIX}.merger"
    y = net.linear(merged, ck.get(f"{m}.proj.weight"))
    y = net.gelu(net.layer_norm(y, ck.get(f"{m}.post_projection_norm.weight"),
                                ck.get(f"{m}.post_projection_norm.bias"), _LAYER_NORM_EPS))
    gate = net.linear(y, ck.get(f"{m}.gate_proj.weight"))
    up = net.linear(y, ck.get(f"{m}.up_proj.weight"))
    return net.linear(net.mul(net.silu(gate), up), ck.get(f"{m}.down_proj.weight"))


def build_vision_engine(ck: Checkpoint, config: GlmOcrConfig, precision: str,
                        max_patches: int, *, verbose: bool) -> bytes:
    v = config.vision
    half = v.head_dim // 2

    def populate(net: Net) -> None:
        network = net.network
        patches = network.add_input("pixel_patches", trt.float32, (-1, v.patch_dim))
        cos = network.add_input("rotary_cos", trt.float32, (-1, half))
        sin = network.add_input("rotary_sin", trt.float32, (-1, half))
        cos3 = net.reshape(cos, (0, 1, half))
        sin3 = net.reshape(sin, (0, 1, half))
        embed = ck.get(f"{_PREFIX}.patch_embed.proj.weight").reshape(v.hidden_size, v.patch_dim)
        x = net.linear(net.cast(patches, net.dtype), embed,
                       ck.get(f"{_PREFIX}.patch_embed.proj.bias"))
        for index in range(v.depth):
            x = _block(net, x, index, ck, config, cos3, sin3)
        out = net.cast(_merge(net, x, ck, config), trt.float32)
        out.name = "image_embeds"
        network.mark_output(out)

    low = v.merge_area
    opt = min(max(_OPT_PATCHES, low), max_patches)
    profile = {
        "pixel_patches": ((low, v.patch_dim), (opt, v.patch_dim), (max_patches, v.patch_dim)),
        "rotary_cos": ((low, half), (opt, half), (max_patches, half)),
        "rotary_sin": ((low, half), (opt, half), (max_patches, half)),
    }
    return build_engine(populate, DTYPES[precision], [profile], verbose=verbose)
