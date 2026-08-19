# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""patchtsmixer model-owned E2E reference plugins."""

from __future__ import annotations

from .references.torch_reference import TorchReference


class PatchtsmixerTorchReferenceReference(TorchReference):
    """patchtsmixer local reference for torch_reference."""

reference = PatchtsmixerTorchReferenceReference()
