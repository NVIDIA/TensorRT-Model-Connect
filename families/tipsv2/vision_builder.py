# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TIPSv2 vision tower: a DINOv2-shaped ViT from pixels to one class embedding.

Four details separate it from the CLIP- and SigLIP-shaped towers it otherwise resembles, and each
one produces a same-shaped wrong answer if carried over:

* the **register token is inserted after the position table is added**, between the class row and
  the patch rows. A tower that concatenates it first, or that gives it a position row, reads the
  whole table off by one;
* the QKV projection is **one packed tensor** and the attention is unmasked;
* each residual branch is multiplied by a learned **LayerScale** vector before it is added;
* the MLP activation is the **error-function** GELU.

The class row of the final layer norm is what the text tower is scored against. The register and
patch rows are computed - the final norm runs over the whole sequence - but only the class row is
marked as an output, because the three tasks this family serves use nothing else.
"""

from __future__ import annotations

import numpy as np

from . import graph as g

PREFIX = "vision_encoder"


def _packed_qkv(weights, layer: str, width: int):
    """Split DINOv2's single ``[3 * width, width]`` projection into q, k and v.

    The reference reshapes the packed output to ``(..., 3, heads, head_dim)`` and takes index 0,
    1, 2, so the packing is q, k, v along the first axis - not interleaved per head.
    """
    packed_weight = np.asarray(weights[f"{layer}.attn.qkv.weight"])
    packed_bias = np.asarray(weights[f"{layer}.attn.qkv.bias"])
    if packed_weight.shape != (3 * width, width) or packed_bias.shape != (3 * width,):
        raise ValueError(f"tipsv2 {layer}.attn.qkv is not a packed [3 * width, width] tensor")
    return (np.split(packed_weight, 3, axis=0), np.split(packed_bias, 3, axis=0))


def _embed_tokens(network, pixel_values, weights, cfg, *, dtype):
    """Pixels to the ``[1 + registers + patches, width]`` sequence the blocks run over."""
    width = cfg["hidden_size"]
    grid = cfg["grid"]
    patch = cfg["patch_size"]
    registers = cfg["num_register_tokens"]

    # The engine's input is float32 whatever it computes in, so the cast happens here rather than
    # at the bundle's edge.
    source = g.to_dtype(network, pixel_values, dtype)
    patches = g.add_conv2d(network, source,
                           weights[f"{PREFIX}.patch_embed.proj.weight"],
                           weights[f"{PREFIX}.patch_embed.proj.bias"],
                           stride=(patch, patch), dtype=dtype)
    flat = network.add_shuffle(patches)
    flat.reshape_dims = (width, grid * grid)
    flat.second_transpose = (1, 0)

    class_token = g.add_constant(network, (1, width),
                                 np.asarray(weights[f"{PREFIX}.cls_token"]).reshape(1, width),
                                 dtype=dtype)
    with_class = network.add_concatenation([class_token, flat.get_output(0)])
    with_class.axis = 0
    hidden = with_class.get_output(0)

    # One position row for the class token and one per patch. The reference interpolates this
    # table for an input whose grid differs; this family is built for the trained grid only
    # (config.resolve refuses any other), so the table is added as stored.
    positions = g.add_constant(network, (cfg["num_positions"], width),
                               np.asarray(weights[f"{PREFIX}.pos_embed"]).reshape(
                                   cfg["num_positions"], width), dtype=dtype)
    hidden = g.add_residual(network, hidden, positions, dtype)

    # The registers come after the position add, so they carry no position of their own, and they
    # sit between the class row and the patch rows.
    register_tokens = g.add_constant(
        network, (registers, width),
        np.asarray(weights[f"{PREFIX}.register_tokens"]).reshape(registers, width), dtype=dtype)
    class_slice = network.add_slice(hidden, (0, 0), (1, width), (1, 1)).get_output(0)
    patch_slice = network.add_slice(hidden, (1, 0), (cfg["num_positions"] - 1, width),
                                    (1, 1)).get_output(0)
    with_registers = network.add_concatenation([class_slice, register_tokens, patch_slice])
    with_registers.axis = 0
    return with_registers.get_output(0)


def _layer_scaled(network, branch, gamma, width, *, dtype):
    """LayerScale: a residual branch times its trained per-channel gamma.

    Gamma is exactly one on the released so400m checkpoint, which makes this a no-op there, but
    it is a trained vector in the architecture and is applied from the weights rather than
    assumed away.

    **The branch is held finite first.** This product is the one place in the tower where an
    FP16 overflow stops being recoverable: a gamma entry of zero times an infinity is a NaN, and
    a NaN passes through every clamp downstream, so the residual stream would carry it to the
    embedding. Clamped, an overflowed channel scales to a large finite number instead, which the
    next layer norm can still work with.
    """
    branch = g.clamp_to_finite(network, branch, dtype)
    return g.add_prod(network, branch, g.add_constant(network, (1, width), gamma, dtype=dtype))


def build_vision_tower(network, pixel_values, weights, cfg, *, dtype=np.float32):
    """``[1, 3, H, W]`` pixels to the unnormalised class embedding ``[width]``."""
    width = cfg["hidden_size"]
    hidden = _embed_tokens(network, pixel_values, weights, cfg, dtype=dtype)

    for index in range(cfg["num_layers"]):
        layer = f"{PREFIX}.blocks.{index}"
        (q_w, k_w, v_w), (q_b, k_b, v_b) = _packed_qkv(weights, layer, width)

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.norm1.weight"],
                                  weights[f"{layer}.norm1.bias"], cfg["eps"])
        attention = g.add_attention(
            network,
            g.add_linear(network, normed, q_w, q_b, dtype=dtype),
            g.add_linear(network, normed, k_w, k_b, dtype=dtype),
            g.add_linear(network, normed, v_w, v_b, dtype=dtype),
            num_heads=cfg["num_heads"], head_dim=cfg["head_dim"], dtype=dtype)
        attention = g.add_linear(network, attention, weights[f"{layer}.attn.proj.weight"],
                                 weights[f"{layer}.attn.proj.bias"], dtype=dtype)
        attention = _layer_scaled(network, attention, weights[f"{layer}.ls1.gamma"], width,
                                  dtype=dtype)
        hidden = g.add_residual(network, hidden, attention, dtype)

        normed = g.add_layer_norm(network, hidden, weights[f"{layer}.norm2.weight"],
                                  weights[f"{layer}.norm2.bias"], cfg["eps"])
        inner = g.add_linear(network, normed, weights[f"{layer}.mlp.fc1.weight"],
                             weights[f"{layer}.mlp.fc1.bias"], dtype=dtype)
        inner = g.add_gelu_exact(network, inner, dtype=dtype)
        inner = g.add_linear(network, inner, weights[f"{layer}.mlp.fc2.weight"],
                             weights[f"{layer}.mlp.fc2.bias"], dtype=dtype)
        inner = _layer_scaled(network, inner, weights[f"{layer}.ls2.gamma"], width, dtype=dtype)
        hidden = g.add_residual(network, hidden, inner, dtype)
        if int(hidden.shape[-1]) != width:
            raise ValueError("tipsv2 vision block changed the sequence width")

    hidden = g.add_layer_norm(network, hidden, weights[f"{PREFIX}.norm.weight"],
                              weights[f"{PREFIX}.norm.bias"], cfg["eps"])
    # Row 0 of the normalised sequence: the class token. ``head`` is nn.Identity in the reference.
    class_row = network.add_slice(hidden, (0, 0), (1, width), (1, 1)).get_output(0)
    flat = network.add_shuffle(class_row)
    flat.reshape_dims = (width,)
    # Both engines hand the runtime FP32 whatever they computed in, so the bundle's output
    # contract does not change with the build precision and the runtime's L2 norm starts from the
    # widest value the engine has.
    embedding, _ = g.as_float32(network, flat.get_output(0))
    return embedding
