# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""qwen3_omni model-owned E2E reference plugin."""

from __future__ import annotations

from .references.torch_reference import TorchReference


class Qwen3OmniTorchReference(TorchReference):
    """Pinned official HF WAV evidence with an invariant-only gate."""


reference = Qwen3OmniTorchReference()
