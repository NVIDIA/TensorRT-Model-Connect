# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic histories with a nearly flat first normalization patch."""

import numpy as np


def histories(context_length: int, patch_length: int) -> tuple[dict, ...]:
    result = []
    for length, frequency, sign, variation in (
        (context_length, 0, 1, 0),
        (3 * patch_length, 2, 1, 0),
        (context_length, 1, 1, 1e-5),
        (context_length, 0, -1, 0),
    ):
        first = 1 + np.linspace(-variation, variation, patch_length, dtype=np.float32)
        rest = sign * (1000 + 50 * np.sin(np.arange(length - patch_length, dtype=np.float32) / 7))
        values = np.concatenate((first, rest)).astype(np.float32)
        result.append({"past_values": values.tolist(), "frequency": frequency})
    return tuple(result)
