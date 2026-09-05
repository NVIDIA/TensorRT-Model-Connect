# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for MoGe."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support


_IDENTIFIED = family_support(
    model_types=("moge", "moge2", "moge-2", "MoGeModelV2"),
    architectures=("MoGeModelV2",),
    tasks=("monocular_geometry",),
    default_task="monocular_geometry",
)
_SUPPORT = FamilySupport(
    tasks=("monocular_geometry",),
    default_task="monocular_geometry",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    support = _IDENTIFIED(metadata)
    if support is not None:
        return support
    if metadata.config or metadata.model_index:
        return None
    return _SUPPORT if "model.pt" in metadata.files else None
