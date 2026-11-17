# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for flux3.

The FLUX 3 Image Klein package is a Diffusers modular repository: it ships no
``model_index.json`` and its root ``config.json`` is Black Forest Labs' own
package description without ``model_type`` or ``architectures``. Identity is
therefore checked on that exact root JSON shape plus the presence of the
component config files the build reads (``build()`` validates their contents).
"""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata

_SUPPORT = FamilySupport(
    tasks=("text_to_image",),
    default_task="text_to_image",
    default_precision="bf16",
)
_MODELS = frozenset({"flux-3-image-klein"})
_SENTINEL_FILES = frozenset(
    {
        "modular_model_index.json",
        "transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
    }
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    """Match the step-distilled FLUX 3 Image Klein package exactly."""

    config = metadata.config
    if config.get("format_version") != 1 or config.get("model") not in _MODELS:
        return None
    dit = config.get("dit")
    if not isinstance(dit, dict) or dit.get("flow_map") is not True:
        return None
    if not _SENTINEL_FILES.issubset(metadata.files):
        return None
    return _SUPPORT
