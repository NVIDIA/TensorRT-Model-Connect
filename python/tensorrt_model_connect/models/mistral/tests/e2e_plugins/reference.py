# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mistral model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference


class MistralHfTransformersReference(HfTransformersReference):
    """mistral local reference for hf_transformers."""

reference = MistralHfTransformersReference()
