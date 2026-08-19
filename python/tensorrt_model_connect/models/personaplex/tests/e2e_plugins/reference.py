# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""personaplex model-owned E2E reference plugins."""

from __future__ import annotations

from .references.official_personaplex import OfficialPersonaPlexReference
from .references.torch_reference import TorchReference


class PersonaplexTorchReferenceReference(TorchReference):
    """personaplex local reference for torch_reference."""


class PersonaplexOfficialReference(OfficialPersonaPlexReference):
    """Pinned official PersonaPlex greedy reference."""


reference = [
    PersonaplexTorchReferenceReference(),
    PersonaplexOfficialReference(),
]
