# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rotary tables, built once as weights rather than computed per forward.

Three things here were read out of the reference rather than assumed, and
each of them was measured against the module the checkpoint actually loads.

**The checkpoint ships no ``inv_freq``.** The reference registers it with
``persistent=False``, so it is absent from ``model.safetensors`` - the key
list was checked. There are therefore no shipped values to prefer over a
recomputation; the recomputation *is* the reference's own, and it has to
match it step for step.

**The frequencies are NTK-rescaled, statically.** ``rope_scaling`` is
``{"type": "ntk", "factor": 2.0}``, and the reference's
``NTKScalingRotaryEmbedding`` builds its cache for
``max_position_embeddings * factor``. That length is greater than
``max_position_embeddings``, so its ``seq_len > self.max_position_embeddings``
branch fires once, at construction, and leaves::

    base     = rope_theta * factor          # 500000 * 2 = 1000000
    inv_freq = 1 / base ** (arange(0, d, 2) / d) / factor ** (2 / d)

The result does **not** depend on the input length - the branch is taken
when the cache is built, not when a sequence arrives - so it is a build-time
table here, unlike the dynamic-NTK rotary that ``families/nomic_bert``
refuses. Ignoring the scaling entirely and using ``rope_theta`` as the base
is a table of the right shape with the wrong values.

**The table is built in float32, not float64.** The reference computes
``inv_freq`` and ``freqs`` in float32 throughout. Measured against the
module's own ``cos_cached``, this float32 reconstruction is exact to 1 ulp
(5.96e-08 over all 16384 rows), while the float64 construction
``families/nomic_bert/rope.py`` uses - which is the more accurate table -
differs from the reference by 3.9e-05 at 512 positions and 1.3e-03 at
16384. Honestly stated: at 512 positions that table difference does **not**
reach the pooled embedding. It was built on purpose and measured, and the
[CLS] vector moved by 2.86e-06, which is the correct build's own noise. So
float32 is used because it is the reference's arithmetic and costs nothing,
not because anything downstream has been shown to notice. At the 8192
positions this checkpoint also claims, the table error is 30x larger and
has not been measured through the model.
"""

from __future__ import annotations

import numpy as np


def inverse_frequencies(head_dim: int, theta: float, scaling_factor: float = 1.0) -> np.ndarray:
    """``inv_freq``, in float32, the way the reference builds it."""
    if head_dim % 2 != 0:
        raise ValueError(f"gte needs an even head_dim, got {head_dim}")
    if scaling_factor < 1.0:
        raise ValueError(f"gte rotary scaling factor must be at least 1.0, got {scaling_factor}")
    exponent = np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)
    if scaling_factor > 1.0:
        # The NTK branch: the base is multiplied by the factor and every
        # frequency is then divided by factor ** (2 / head_dim).
        base = np.float32(theta) * np.float32(scaling_factor)
        inv_freq = np.float32(1.0) / np.power(base, exponent, dtype=np.float32)
        return (inv_freq / np.float32(scaling_factor ** (2.0 / head_dim))).astype(np.float32)
    inv_freq = np.float32(1.0) / np.power(np.float32(theta), exponent, dtype=np.float32)
    return inv_freq.astype(np.float32)


def tables(seq_len: int, head_dim: int, theta: float,
           scaling_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(cos, sin)``, each ``[seq_len, head_dim]``, in float32.

    ``freqs = outer(positions, inv_freq)``, then ``cat((freqs, freqs), -1)``
    before the cosine - the **tiled**, split-half (GPT-NeoX) layout that
    ``rotate_half`` in the reference pairs column ``d`` with column
    ``d + head_dim / 2``. The interleaved layout has the same shape and
    different values, so a mix-up is not a build error.
    """
    inv_freq = inverse_frequencies(head_dim, theta, scaling_factor)
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.einsum("i,j->ij", positions, inv_freq).astype(np.float32)
    cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1).astype(np.float32)
    sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1).astype(np.float32)
    return cos, sin
