# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rotary tables, built once as weights rather than computed per forward.

The engine runs at one fixed sequence length, so the cosines and sines the
reference recomputes on every call are constants here - the same trade
``families/dinov2`` makes with its interpolated position table. Nothing about
the values changes; they move from runtime to build time.

Two details decide whether this is right, and both are read from the
checkpoint rather than assumed:

* ``rotary_emb_base`` is **1000** here, not the 10000 nearly every other
  rotary model uses. A wrong base rotates every position by the wrong angle.
* ``rotary_emb_interleaved`` is false, which means the GPT-NeoX split-half
  convention: the first half of a head pairs with the second half, and the
  table is **tiled** ``[c0..c31, c0..c31]`` rather than interleaved
  ``[c0, c0, c1, c1, ...]``. The two layouts have the same shape and
  different values, so a mix-up is not a build error.
"""

from __future__ import annotations

import numpy as np


def tables(seq_len: int, head_dim: int, base: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cos, sin)``, each ``[seq_len, head_dim]``.

    ``inv_freq = 1 / base ** (arange(0, head_dim, 2) / head_dim)`` and
    ``freqs = outer(positions, inv_freq)``, exactly as the reference builds
    them, in float64 so the table does not inherit a float32 rounding of the
    exponent before the cosine is taken.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"nomic_bert needs an even head_dim, got {head_dim}")
    exponent = np.arange(0, head_dim, 2, dtype=np.float64) / float(head_dim)
    inv_freq = 1.0 / (float(base) ** exponent)
    freqs = np.outer(np.arange(seq_len, dtype=np.float64), inv_freq)
    # Tiled, not interleaved: this is the non-interleaved branch, and the
    # reference's ``repeat(cos, "... d -> ... 1 (2 d)")`` is a tile.
    cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1)
    sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1)
    return cos.astype(np.float32), sin.astype(np.float32)
