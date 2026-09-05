# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for nemotron_labs_diffusion."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("nemotron_labs_diffusion", "nemotronlabsdiffusion"),
    tasks=("text_generation",),
    default_task="text_generation",
)
