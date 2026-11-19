# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the upstream speech generation stopping rule."""

import numpy as np


def assert_complete_eos(trace, threshold, minimum_frames, tail_frames):
    values = np.asarray(trace).reshape(-1)
    if not np.isfinite(values).all():
        raise AssertionError("reference EOS trace is nonfinite")
    eligible = np.flatnonzero((values > threshold) & (np.arange(values.size) >= minimum_frames))
    if not eligible.size:
        raise AssertionError("reference did not reach an eligible EOS crossing")
    first = int(eligible[0])
    if values.size != first + 1 + tail_frames:
        raise AssertionError("reference did not complete its EOS tail")
    return first
