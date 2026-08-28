# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""qwen3_omni model-owned E2E reference plugin."""

from __future__ import annotations

from .references.invariant_only import InvariantOnlyReference
from .references.torch_reference import TorchReference


class Qwen3OmniTorchReference(TorchReference):
    """Pinned official HF WAV evidence with an invariant-only gate."""


class Qwen3OmniInvariantOnlyReference(InvariantOnlyReference):
    """Native Thinker output is checked without an external model."""


reference = [Qwen3OmniTorchReference(), Qwen3OmniInvariantOnlyReference()]
