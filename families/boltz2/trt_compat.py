# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small family-owned TensorRT compatibility boundary."""

from __future__ import annotations

from typing import Any


def _module() -> Any:
    import tensorrt

    return tensorrt


def tensorrt_version() -> str:
    return str(getattr(_module(), "__version__", ""))


def network_creation_flags(*, strongly_typed: bool = True, explicit_batch: bool = False) -> int:
    flags = 0
    creation = getattr(_module(), "NetworkDefinitionCreationFlag", None)
    if creation is None:
        return flags
    if strongly_typed and hasattr(creation, "STRONGLY_TYPED"):
        flags |= 1 << int(creation.STRONGLY_TYPED)
    if explicit_batch and hasattr(creation, "EXPLICIT_BATCH"):
        flags |= 1 << int(creation.EXPLICIT_BATCH)
    return flags


def get_trt() -> Any:
    return _module()
