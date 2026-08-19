# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""magpie_tts model-owned E2E reference plugins."""

from __future__ import annotations

from .references.nemo_reference import NemoReference


class MagpieTtsNemoReference(NemoReference):
    """magpie_tts local reference for nemo."""

reference = MagpieTtsNemoReference()
