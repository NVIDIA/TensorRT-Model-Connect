# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""parakeet_tdt model-owned E2E reference plugins."""

from __future__ import annotations

from .references.parakeet_tdt_hf import HfTransformersReference


class ParakeetTDTHfTransformersReference(HfTransformersReference):
    """parakeet_tdt local reference for hf_transformers."""

reference = ParakeetTDTHfTransformersReference()
