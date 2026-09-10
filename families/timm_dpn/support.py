# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for timm DPN."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("timm_dpn", "dpn68", "dpn68b", "dpn92", "dpn98", "dpn107", "dpn131"),
    architectures=("dpn68", "dpn68b", "dpn92", "dpn98", "dpn107", "dpn131"),
    tasks=("classification",),
    default_task="classification",
)
