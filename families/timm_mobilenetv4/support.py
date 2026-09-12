# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for timm_mobilenetv4."""

from tensorrt_model_connect.model_support import family_support


# Only the pure-convolution widths are claimed. The `hybrid` widths add a
# Mobile MQA attention block and the `aa` and `blur` widths move their stride
# into an anti-aliasing blur pool; neither is built here, so neither is
# claimed, and a directory holding one is left for a future family.
describe = family_support(
    architectures=(
        "mobilenetv4_conv_small_050",
        "mobilenetv4_conv_small",
        "mobilenetv4_conv_medium",
        "mobilenetv4_conv_large",
    ),
    tasks=("classification",),
    default_task="classification",
)
