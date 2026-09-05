# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for ELF."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support


_STATIC = family_support(
    model_types=("elf", "elf_flow", "elfflow", "embedded_language_flow"),
    tasks=("text_generation",),
    default_task="text_generation",
)
_CHECKPOINTS = tuple(
    family_support(
        required_files=(
            config,
            "checkpoint_0/_CHECKPOINT_METADATA",
            "checkpoint_0/manifest.ocdbt",
        ),
        tasks=("text_generation",),
        default_task="text_generation",
    )
    for config in ("ELF-B-de-en.yml", "ELF-B-owt.yml", "ELF-B-xsum.yml")
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    if support := _STATIC(metadata):
        return support
    return next(
        (support for match in _CHECKPOINTS if (support := match(metadata)) is not None),
        None,
    )
