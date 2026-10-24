# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The twelve post-norm blocks.

One block, written the way the reference writes it::

    hidden = norm1(out_proj(attn(Wqkv(hidden))) + hidden)
    hidden = norm2(fc2(fc11(hidden) * silu(fc12(hidden))) + hidden)

Both norms come **after** the residual add. Reading the same weights
pre-norm - normalising on the way into each branch instead - produces a
model that runs, converges on plausible vectors, and is wrong; the config's
``prenorm: false`` is checked in ``config.py`` for that reason.
"""

from __future__ import annotations

import numpy as np

from . import graph as g


def _split_qkv(packed: np.ndarray, hidden: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``Wqkv`` into q, k and v.

    The reference reshapes the projection output as ``(three h d)`` with
    ``three`` outermost, so the three matrices are contiguous blocks of rows
    in that order. Splitting on the head axis instead - ``(h three d)`` -
    gives three matrices of the right shape built from the wrong rows.
    """
    packed = np.asarray(packed)
    if packed.shape[0] != 3 * hidden:
        raise ValueError(f"nomic_bert packed QKV has {packed.shape[0]} rows, expected {3 * hidden}")
    return packed[:hidden], packed[hidden:2 * hidden], packed[2 * hidden:]


def build_layers(network, hidden, weights, cfg, *, mask, cos, sin, dtype=np.float32):
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    width = cfg["hidden_size"]

    for index in range(cfg["num_layers"]):
        prefix = f"encoder.layers.{index}."
        query, key, value = _split_qkv(weights[f"{prefix}attn.Wqkv.weight"], width)
        attended = g.add_attention(
            network,
            g.add_linear(network, hidden, query, dtype=dtype),
            g.add_linear(network, hidden, key, dtype=dtype),
            g.add_linear(network, hidden, value, dtype=dtype),
            mask, num_heads=num_heads, head_dim=head_dim, cos=cos, sin=sin)
        projected = g.add_linear(network, attended, weights[f"{prefix}attn.out_proj.weight"],
                                 dtype=dtype)
        hidden = g.add_layer_norm(network, g.add_sum(network, projected, hidden),
                                  weights[f"{prefix}norm1.weight"],
                                  weights[f"{prefix}norm1.bias"], cfg["eps"])

        mlp = g.add_gated_mlp(network, hidden,
                              weights[f"{prefix}mlp.fc11.weight"],
                              weights[f"{prefix}mlp.fc12.weight"],
                              weights[f"{prefix}mlp.fc2.weight"], dtype=dtype)
        hidden = g.add_layer_norm(network, g.add_sum(network, mlp, hidden),
                                  weights[f"{prefix}norm2.weight"],
                                  weights[f"{prefix}norm2.bias"], cfg["eps"])
    return hidden
