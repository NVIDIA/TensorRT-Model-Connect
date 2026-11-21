# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SAME autoencoder decoder (latents -> stereo audio) as one TensorRT network.

It reproduces ``AudioAutoencoder.decode`` of the reference for the single-stage
SAME decoder of Stable Audio 3:

* the softnorm bottleneck scales the latents by ``running_std``;
* each latent becomes one token followed by ``stride`` copies of the learned
  ``new_tokens``; the transformer blocks attend within a band of one latent group
  on each side (flash-attention ``window_size=(17, 17)``), use DynamicTanh norms
  and qk-norms, differential attention and a GLU feed-forward whose last blocks
  use ``sin(pi x)``;
* the new tokens are kept, mapped by a weight-normed 1x1 convolution and unpatched
  into ``audio_channels x (stride * patch_size)`` samples per latent.

The decoder is deterministic. The official ``generate()`` adds noise here at
inference (the bottleneck's ``noise_regularize``, 1e-3 x running_std, and the
decoder's ``mask_noise``, 0.1 on the new tokens); both are left out on purpose,
as Stability AI's own TensorRT and MLX ports of this decoder do, and the E2E
oracle switches them off the same way. One call decodes the whole latent
sequence; the reference's chunked decode trims more context than the decoder's
receptive field, so it computes the same values.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import weight_norm
from .graph import MASKED, Graph, apply_rotary, rotary_tables

LATENTS = "latents"
AUDIO = "audio"

_PREFIX = "layers.3."
_BLOCK = [
    "pre_norm.alpha", "pre_norm.gamma", "pre_norm.beta", "ff_norm.alpha", "ff_norm.gamma", "ff_norm.beta",
    "self_attn.to_qkv.weight", "self_attn.to_out.weight",
    "self_attn.q_norm.alpha", "self_attn.q_norm.gamma", "self_attn.q_norm.beta",
    "self_attn.k_norm.alpha", "self_attn.k_norm.gamma", "self_attn.k_norm.beta",
    "ff.ff.0.proj.weight", "ff.ff.0.proj.bias", "ff.ff.2.weight", "ff.ff.2.bias", "rope.inv_freq",
]


def weight_names(decoder: dict) -> list[str]:
    names = ["layers.1.weight", "layers.1.bias", _PREFIX + "new_tokens", _PREFIX + "mapping.weight_g",
             _PREFIX + "mapping.weight_v", _PREFIX + "mapping.bias"]
    for block in range(decoder["depth"]):
        names += [f"{_PREFIX}transformers.{block}.{name}" for name in _BLOCK]
    return names


def _band_mask(g: Graph, groups, max_groups: int, group: int, window: int):
    """Additive [T, 1, group, 3 * group] mask of the banded attention.

    Group ``n`` of ``group`` queries sees the key groups ``n - 1, n, n + 1``. A query
    at offset ``i`` keeps keys at offsets ``j`` with ``i <= j <= i + 2 * window``
    (|query - key| <= window); the missing neighbours of the first and last group
    are masked out.
    """
    span = 3 * group
    rows = np.arange(group)[:, None]
    cols = np.arange(span)[None, :]
    band = np.where((cols >= rows + group - window) & (cols <= rows + group + window), 0.0, MASKED)
    band_t = g.const(band.astype(np.float32), trt.float32, shape=(1, 1, group, span))

    index = g.reshape(g.slice(g.const(np.arange(max_groups, dtype=np.int32), trt.int32), (0,), (groups,)),
                      [groups, 1, 1, 1])
    last = g.add(g.cast(groups, trt.int32), g.const(np.array([-1], dtype=np.int32), trt.int32))
    is_first = g.n.add_elementwise(index, g.const(np.zeros((1, 1, 1, 1), dtype=np.int32), trt.int32),
                                   trt.ElementWiseOperation.EQUAL).get_output(0)
    is_last = g.n.add_elementwise(index, g.reshape(last, [1, 1, 1, 1]),
                                  trt.ElementWiseOperation.EQUAL).get_output(0)
    offsets = g.const(np.arange(span, dtype=np.int32), trt.int32, shape=(1, 1, 1, span))
    left = g.n.add_elementwise(offsets, g.const(np.full((1, 1, 1, 1), group, dtype=np.int32), trt.int32),
                               trt.ElementWiseOperation.LESS).get_output(0)
    right = g.n.add_elementwise(offsets, g.const(np.full((1, 1, 1, 1), 2 * group - 1, dtype=np.int32), trt.int32),
                                trt.ElementWiseOperation.GREATER).get_output(0)
    missing = g.n.add_elementwise(
        g.n.add_elementwise(is_first, left, trt.ElementWiseOperation.AND).get_output(0),
        g.n.add_elementwise(is_last, right, trt.ElementWiseOperation.AND).get_output(0),
        trt.ElementWiseOperation.OR).get_output(0)
    edge = g.n.add_select(missing, g.const(np.full((1, 1, 1, 1), MASKED, dtype=np.float32), trt.float32),
                          g.const(np.zeros((1, 1, 1, 1), dtype=np.float32), trt.float32)).get_output(0)
    return g.add(edge, band_t)


def _group_queries(g: Graph, x, groups, heads: int, group: int, head_dim: int):
    """[1, H, T * group, D] -> [T, H, group, D]."""
    return g.transpose(g.reshape(x, [heads, groups, group, head_dim]), (1, 0, 2, 3))


def _group_keys(g: Graph, x, groups, heads: int, group: int, head_dim: int):
    """[1, H, T * group, D] -> [T, H, 3 * group, D]: each group with both neighbours."""
    grouped = g.reshape(x, [heads, groups, group, head_dim])
    zero = g.const(np.zeros((heads, 1, group, head_dim), dtype=np.float32), x.dtype)
    padded = g.concat([zero, grouped, zero], 1)
    shifted = [g.slice(padded, (0, offset, 0, 0), (heads, groups, group, head_dim)) for offset in range(3)]
    return g.transpose(g.concat(shifted, 2), (1, 0, 2, 3))


def _block(g: Graph, x, weights: dict, dec: dict, block: int, rope, tokens, groups, mask):
    prefix = f"{_PREFIX}transformers.{block}."

    def w(name):
        return weights[prefix + name]

    heads, head_dim, width, group = dec["heads"], dec["head_dim"], dec["width"], dec["group"]
    h = g.dynamic_tanh(x, w("pre_norm.alpha"), w("pre_norm.gamma"), w("pre_norm.beta"))
    q, k, v, q_diff, k_diff = g.split_last(g.linear(h, w("self_attn.to_qkv.weight")), 5)
    q, k, v, q_diff, k_diff = (g.transpose(g.reshape(item, [1, tokens, heads, head_dim]), (0, 2, 1, 3))
                               for item in (q, k, v, q_diff, k_diff))
    q, q_diff = (g.dynamic_tanh(item, w("self_attn.q_norm.alpha"), w("self_attn.q_norm.gamma"),
                                w("self_attn.q_norm.beta")) for item in (q, q_diff))
    k, k_diff = (g.dynamic_tanh(item, w("self_attn.k_norm.alpha"), w("self_attn.k_norm.gamma"),
                                w("self_attn.k_norm.beta")) for item in (k, k_diff))
    cos, sin = rope
    q, q_diff, k, k_diff = (apply_rotary(g, item, cos, sin, dec["rope_dim"], head_dim, v.dtype)
                            for item in (q, q_diff, k, k_diff))
    queries = [_group_queries(g, item, groups, heads, group, head_dim) for item in (q, q_diff)]
    keys = [_group_keys(g, item, groups, heads, group, head_dim) for item in (k, k_diff)]
    values = _group_keys(g, v, groups, heads, group, head_dim)
    scale = head_dim ** -0.5
    out = g.sub(g.attention(queries[0], keys[0], values, scale, mask),
                g.attention(queries[1], keys[1], values, scale, mask))  # [T, H, group, D]
    out = g.reshape(g.transpose(out, (0, 2, 1, 3)), [1, tokens, width])
    x = g.add(x, g.linear(out, w("self_attn.to_out.weight")))

    h = g.dynamic_tanh(x, w("ff_norm.alpha"), w("ff_norm.gamma"), w("ff_norm.beta"))
    value, gate = g.split_last(g.linear(h, w("ff.ff.0.proj.weight"), w("ff.ff.0.proj.bias")), 2)
    if dec["sinusoidal"][block]:
        # torch.sin(3.14159265359 * x): the product is formed in FP32 and rounded to
        # the trunk dtype, then sin is evaluated in FP32 and rounded again.
        angle = g.cast(g.mul_scalar(g.cast(gate, trt.float32), 3.14159265359), gate.dtype)
        gate = g.cast(g.unary(g.cast(angle, trt.float32), trt.UnaryOperation.SIN), gate.dtype)
    else:
        gate = g.silu(gate)
    return g.add(x, g.linear(g.mul(value, gate), w("ff.ff.2.weight"), w("ff.ff.2.bias")))


def populate(network: trt.INetworkDefinition, weights: dict, running_std: np.ndarray, cfg: dict,
             work: trt.DataType, max_latents: int) -> Graph:
    dec = cfg["decoder"]
    g = Graph(network, work)
    latent_dim, width, stride, group = dec["latent_dim"], dec["width"], dec["stride"], dec["group"]
    latents = network.add_input(LATENTS, trt.float32, (1, latent_dim, -1))
    groups = g.dim(latents, 2)
    tokens = g.int_mul(groups, group)

    x = g.mul(g.cast(latents, work), g.const(running_std, shape=(1, 1, 1)))
    x = g.linear(g.transpose(x, (0, 2, 1)), weights["layers.1.weight"], weights["layers.1.bias"])  # [1, T, W]
    x = g.reshape(x, [groups, 1, width])
    zeros = g.slice(g.const(np.zeros((max_latents, 1, 1), dtype=np.float32)), (0, 0, 0), (groups, 1, 1))
    new_tokens = np.repeat(weights[_PREFIX + "new_tokens"].reshape(1, 1, width), stride, axis=1)
    x = g.concat([x, g.add(zeros, g.const(new_tokens))], 1)  # [T, group, W]
    x = g.reshape(x, [1, tokens, width])

    inv_freq = weights[f"{_PREFIX}transformers.0.rope.inv_freq"]
    for block in range(1, dec["depth"]):
        if not np.array_equal(weights[f"{_PREFIX}transformers.{block}.rope.inv_freq"], inv_freq):
            raise ValueError("stable_audio_3 decoder blocks disagree on their rotary frequencies")
    tables = rotary_tables(g, inv_freq, max_latents * group)
    rope = tuple(g.reshape(g.slice(table, (0, 0), (tokens, dec["rope_dim"])), [1, 1, tokens, dec["rope_dim"]])
                 for table in tables)
    mask = _band_mask(g, groups, max_latents, group, dec["window"])
    for block in range(dec["depth"]):
        x = _block(g, x, weights, dec, block, rope, tokens, groups, mask)

    # Keep the new tokens of every group, map 1536 -> 512 channels, unpatch to audio.
    x = g.slice(g.reshape(x, [groups, group, width]), (0, 1, 0), (groups, stride, width))
    mapping = weight_norm(weights[_PREFIX + "mapping.weight_g"], weights[_PREFIX + "mapping.weight_v"])[:, :, 0]
    x = g.linear(g.reshape(x, [1, g.int_mul(groups, stride), width]), mapping, weights[_PREFIX + "mapping.bias"])
    channels, patch = dec["audio_channels"], dec["patch_size"]
    # [1, L, (c h)] -> [1, c, L, h] -> [1, c, L * h]
    x = g.transpose(g.reshape(x, [1, g.int_mul(groups, stride), channels, patch]), (0, 2, 1, 3))
    audio = g.cast(g.reshape(x, [1, channels, g.int_mul(groups, stride * patch)]), trt.float32)
    audio.name = AUDIO
    network.mark_output(audio)
    return g
