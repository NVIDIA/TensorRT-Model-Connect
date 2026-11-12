# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small immutable preprocessing assets for the owning native C++ runtime."""

import numpy as np


def mel_filters():
    """128 Slaney-normalized triangular filters for 512-point, 16 kHz audio."""
    maximum = 15.0 + np.log(8000.0 / 1000.0) / (np.log(6.4) / 27.0)
    mel = np.linspace(0.0, maximum, 130)
    hz = mel * (200.0 / 3.0)
    logarithmic = mel >= 15.0
    hz[logarithmic] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (mel[logarithmic] - 15.0))
    frequencies = np.fft.rfftfreq(512, d=1.0 / 16000.0)
    differences = np.diff(hz)
    ramps = hz[:, None] - frequencies[None, :]
    weights = np.zeros((128, 257), dtype=np.float32)
    for index in range(128):
        weights[index] = np.maximum(
            0.0,
            np.minimum(
                -ramps[index] / differences[index], ramps[index + 2] / differences[index + 1]
            ),
        )
    weights *= (2.0 / (hz[2:] - hz[:-2]))[:, None]
    return weights
