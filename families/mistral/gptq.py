# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The prepared Mistral GPTQ-v1 W8A16 checkpoint's on-disk representation."""

from dataclasses import dataclass

import numpy as np


def is_prepared_w8a16(config: dict) -> bool:
    quant = config.get("quantization_config") or {}
    if quant.get("quant_method") != "gptq":
        return False
    if not (
        quant.get("bits") == 8
        and quant.get("sym") is True
        and quant.get("group_size") == -1
        and quant.get("desc_act") is False
        and quant.get("checkpoint_format", quant.get("format", "gptq")) == "gptq"
    ):
        raise ValueError("Mistral requires symmetric per-output-channel GPTQ-v1 W8A16")
    return True


@dataclass(frozen=True)
class W8A16Projection:
    values: np.ndarray  # signed INT8 [input, output]
    scales: np.ndarray  # FP16 [output]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape


def unpack_projection(qweight, qzeros, scales, g_idx, in_features, out_features):
    """Unpack four input rows per word, without rounding or floating-point expansion."""
    if (
        in_features % 4
        or out_features % 4
        or qweight.dtype != np.int32
        or qweight.shape != (in_features // 4, out_features)
        or qzeros.dtype != np.int32
        or qzeros.shape != (1, out_features // 4)
        or scales.dtype != np.float16
        or scales.shape != (1, out_features)
        or g_idx.dtype != np.int32
        or g_idx.shape != (in_features,)
    ):
        raise ValueError("Invalid Mistral GPTQ W8A16 projection layout")
    # Legacy GPTQ stores zero_point - 1 in each output byte. Only zero=128
    # admits an exact signed-byte representation with no additive correction.
    if not np.all(qzeros == 0x7F7F7F7F) or np.any(g_idx != 0):
        raise ValueError("Mistral W8A16 requires zero point 128 and one input group")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Mistral W8A16 requires positive finite channel scales")
    words = np.ascontiguousarray(qweight, dtype="<i4")
    lanes = words.view(np.uint8).reshape(in_features // 4, out_features, 4)
    values = np.empty((in_features, out_features), dtype=np.int8)
    for lane in range(4):
        np.bitwise_xor(lanes[:, :, lane], np.uint8(128), out=values[lane::4].view(np.uint8))
    return W8A16Projection(values, np.ascontiguousarray(scales[0]))
