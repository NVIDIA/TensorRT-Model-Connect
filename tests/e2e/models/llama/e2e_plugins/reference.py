# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""llama model-owned E2E reference plugins."""

from __future__ import annotations

from .references.hf_transformers import HfTransformersReference
from .references.invariant_only import InvariantOnlyReference


class LlamaHfTransformersReference(HfTransformersReference):
    """llama local reference for hf_transformers."""


class LlamaInvariantOnlyReference(InvariantOnlyReference):
    """llama local reference for invariant_only."""


reference = [LlamaHfTransformersReference(), LlamaInvariantOnlyReference()]
