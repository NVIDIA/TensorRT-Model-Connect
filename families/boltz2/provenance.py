# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned public provenance for the initial Boltz-2 qualification target."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PinnedArtifact:
    """One revision-owned file required by the pinned Boltz-2 workflow."""

    filename: str
    size_bytes: int


@dataclass(frozen=True)
class Boltz2ReferenceConfiguration:
    """Inference controls shared by reference and TensorRT qualification."""

    precision: str
    recycling_steps: int
    sampling_steps: int
    diffusion_samples: int
    max_msa_sequences: int
    seed: int
    output_format: str


@dataclass(frozen=True)
class Boltz2Provenance:
    source_repository: str
    source_revision: str
    source_tag: str
    source_license: str
    checkpoint_repository: str
    checkpoint_revision: str
    checkpoint_license: str
    structure_checkpoint: PinnedArtifact
    molecular_archive: PinnedArtifact
    reference_configuration: Boltz2ReferenceConfiguration


PINNED_BOLTZ2 = Boltz2Provenance(
    source_repository="https://github.com/jwohlwend/boltz.git",
    source_revision="cb04aeccdd480fd4db707f0bbafde538397fa2ac",
    source_tag="v2.2.1",
    source_license="MIT",
    checkpoint_repository="boltz-community/boltz-2",
    checkpoint_revision="6fdef46d763fee7fbb83ca5501ccceff43b85607",
    checkpoint_license="MIT",
    structure_checkpoint=PinnedArtifact(
        filename="boltz2_conf.ckpt",
        size_bytes=2_286_561_469,
    ),
    molecular_archive=PinnedArtifact(
        filename="mols.tar",
        size_bytes=1_855_662_080,
    ),
    reference_configuration=Boltz2ReferenceConfiguration(
        precision="bf16",
        recycling_steps=3,
        sampling_steps=200,
        diffusion_samples=1,
        max_msa_sequences=8,
        seed=42,
        output_format="mmcif",
    ),
)
