# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sine position embeddings for the vision tokens, plus the level embedding.

Shaped like the one ``families/deformable_detr`` builds, but **two constants
differ and copying that file would be wrong**:

* ``temperature`` is **20**, not 10000;
* there is **no ``- 0.5`` offset** in the normalisation. Deformable DETR uses
  ``(embed - 0.5) / last``; this model uses ``embed / last``. Dropping that
  offset was a *failing* negative control in that family - here it is the
  correct rule.

Shared with that family: 128 features per axis, scale 2*pi, the y half
concatenated before the x half, and interleaved sin/cos rather than split
halves. With a fixed input size and no padding the mask is all ones, so these
are constants computed once at build time.
"""

from __future__ import annotations

import numpy as np

_FEATURES = 128
_TEMPERATURE = 20.0
_SCALE = 2.0 * np.pi
_EPS = 1e-6


def sine_embedding(height: int, width: int, dtype=np.float32) -> np.ndarray:
    """Return ``(1, height * width, 256)`` sine position embeddings."""
    # An all-ones mask makes the cumulative sums the 1-based row and column
    # indices, which is what the reference reduces to with nothing padded.
    y_embed = np.tile(np.arange(1, height + 1, dtype=np.float64)[:, None], (1, width))
    x_embed = np.tile(np.arange(1, width + 1, dtype=np.float64)[None, :], (height, 1))
    # No -0.5 here. That offset belongs to the DETR lineage, not this one.
    y_embed = y_embed / (height + _EPS) * _SCALE
    x_embed = x_embed / (width + _EPS) * _SCALE

    index = np.arange(_FEATURES, dtype=np.float64)
    dim_t = _TEMPERATURE ** (2.0 * np.floor(index / 2.0) / _FEATURES)

    def interleave(embed):
        raw = embed[:, :, None] / dim_t
        return np.stack((np.sin(raw[:, :, 0::2]), np.cos(raw[:, :, 1::2])), axis=3).reshape(
            height, width, _FEATURES)

    # y first, then x.
    pos = np.concatenate((interleave(y_embed), interleave(x_embed)), axis=2)
    return pos.reshape(1, height * width, 2 * _FEATURES).astype(dtype)


def build_level_embeddings(shapes, level_embed, dtype=np.float32) -> np.ndarray:
    """Flattened ``(1, sum(h * w), 256)`` positions, each offset by its level."""
    if len(shapes) != len(level_embed):
        raise ValueError("grounding_dino level count does not match level_embed")
    pieces = []
    for level, (height, width) in enumerate(shapes):
        embedding = sine_embedding(height, width, dtype)
        pieces.append(embedding + np.asarray(level_embed[level], dtype=dtype).reshape(1, 1, -1))
    return np.concatenate(pieces, axis=1)
