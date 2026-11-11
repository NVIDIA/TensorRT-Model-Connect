# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token oracles of the SmolVLM E2E.

``first_divergence`` compares two free-running greedy sequences. ``reference_disagreements``
compares native tokens with the reference's next-token logits computed on the native
prefix (teacher forcing): every native token must be the reference's own greedy choice
there, or come within a tolerance of it. Unlike a comparison of free-running texts, it
checks every generated token and does not cascade after a near-tie.
"""

from __future__ import annotations

import numpy as np


def first_divergence(actual, expected):
    """Index of the first differing token, the shorter length for a strict prefix, else None."""
    for index, (a, b) in enumerate(zip(actual, expected)):
        if a != b:
            return index
    return None if len(actual) == len(expected) else min(len(actual), len(expected))


def reference_disagreements(native_tokens, step_logits):
    """Steps where the native token is not the reference's greedy token given the native prefix.

    ``step_logits[t]`` holds the reference logits that predict ``native_tokens[t]`` from the
    prompt and ``native_tokens[:t]``. Each disagreement reports the reference's preference for
    its own token over the native one, in logits (never negative).
    """
    if len(step_logits) != len(native_tokens):
        raise ValueError("one row of reference logits per native token is required")
    rows = []
    for step, (token, logits) in enumerate(zip(native_tokens, step_logits)):
        if not np.isfinite(logits).all():
            raise ValueError(f"reference logits at step {step} are not finite")
        preferred = int(logits.argmax())
        if logits[token] < logits[preferred]:
            rows.append({"step": step, "reference_token": preferred, "native_token": int(token),
                         "logit_gap": float(logits[preferred] - logits[token])})
    return rows
