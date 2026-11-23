# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write one independently readable section per Flash-Next routed expert."""

from functools import lru_cache
import math

import ml_dtypes
import numpy as np


def write_expert(checkpoint, writer, layer, expert):
    """Preserve packed nibbles, original FP8 block scales and both calibrated scalars."""
    with writer.open_section(f"expert.{layer}.{expert}.bin") as out:
        out.write(b"FNE2")
        for name, rows, width in (("gate", 640, 2560), ("up", 640, 2560), ("down", 2560, 640)):
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{name}_proj."
            q = checkpoint.tensor(prefix + "weight", np.uint8, (rows, width // 2))
            s = checkpoint.tensor(
                prefix + "weight_scale", ml_dtypes.float8_e4m3fn, (rows, width // 16)
            )
            bits = s.view(np.uint8)
            if np.any((bits >= 127) & (bits != 128)):
                raise ValueError(f"Invalid NVFP4 block scale: {prefix}")
            out.write(memoryview(q).cast("B"))
            out.write(memoryview(bits).cast("B"))
            for suffix in ("weight_scale_2", "input_scale"):
                scale = checkpoint.tensor(prefix + suffix, np.float32, ())
                if not math.isfinite(scale.item()) or scale.item() <= 0:
                    raise ValueError(f"Invalid NVFP4 calibration: {prefix}{suffix}")
                out.write(np.asarray(scale, dtype="<f4").tobytes())
                if suffix == "input_scale":
                    out.write(memoryview(activation_limits(scale.item())).cast("B"))


@lru_cache(maxsize=32)
def activation_limits(global_scale):
    """First admissible FP16 magnitude at each calibrated ModelOpt rounding boundary.

    The input is FP16 and ModelOpt divides by the effective scale in FP32.
    Comparing to a product rounded to FP32 is not equivalent at half-ULP
    ties. These small build-time quantizer parameters preserve the original
    comparison exactly, even on backends that lower division to reciprocal.
    """
    scale = np.float32(global_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid Flash-Next NVFP4 activation calibration")
    # All nonnegative finite FP16 values in order, plus a sentinel beyond the finite domain.
    values = np.arange(0x7C00, dtype=np.uint16).view(np.float16).astype(np.float32)
    boundaries = np.append(values, np.float32(65536))
    with np.errstate(over="ignore", under="ignore"):
        block = np.clip(values / (np.float32(6) * scale), 2**-9, 448)
        codes = block.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
        limits = np.full((128, 8), 65536, np.float32)
        limits[:, 0] = boundaries[np.searchsorted(codes, np.arange(128), side="left")]
        positive = (
            np.arange(1, 127, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
        )
        for code, effective in enumerate(positive * scale, start=1):
            scaled = values / effective
            for i, mid in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0), start=1):
                which = "left" if i % 2 == 0 else "right"
                limits[code, i] = boundaries[np.searchsorted(scaled, mid, side=which)]
    limits.setflags(write=False)
    return limits
