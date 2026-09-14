# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for timm XCiT."""

from tensorrt_model_connect.model_support import family_support


_ARCHITECTURES = (
    "xcit_large_24_p16_224",
    "xcit_large_24_p16_384",
    "xcit_large_24_p8_224",
    "xcit_large_24_p8_384",
    "xcit_medium_24_p16_224",
    "xcit_medium_24_p16_384",
    "xcit_medium_24_p8_224",
    "xcit_medium_24_p8_384",
    "xcit_nano_12_p16_224",
    "xcit_nano_12_p16_384",
    "xcit_nano_12_p8_224",
    "xcit_nano_12_p8_384",
    "xcit_small_12_p16_224",
    "xcit_small_12_p16_384",
    "xcit_small_12_p8_224",
    "xcit_small_12_p8_384",
    "xcit_small_24_p16_224",
    "xcit_small_24_p16_384",
    "xcit_small_24_p8_224",
    "xcit_small_24_p8_384",
    "xcit_tiny_12_p16_224",
    "xcit_tiny_12_p16_384",
    "xcit_tiny_12_p8_224",
    "xcit_tiny_12_p8_384",
    "xcit_tiny_24_p16_224",
    "xcit_tiny_24_p16_384",
    "xcit_tiny_24_p8_224",
    "xcit_tiny_24_p8_384",
)

describe = family_support(
    model_types=("timm_xcit",) + _ARCHITECTURES,
    architectures=_ARCHITECTURES,
    tasks=("classification",),
    default_task="classification",
)
