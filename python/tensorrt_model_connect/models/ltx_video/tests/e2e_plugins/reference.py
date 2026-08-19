# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ltx_video model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_diffusers import HfDiffusersReference


class LtxVideoHfDiffusersReference(HfDiffusersReference):
    """ltx_video local reference for hf_diffusers."""

reference = LtxVideoHfDiffusersReference()
