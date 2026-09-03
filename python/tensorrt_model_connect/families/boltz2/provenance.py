# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned public provenance for the initial Boltz-2 qualification target."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PinnedArtifact:
    """One immutable file required by the pinned Boltz-2 workflow."""

    filename: str
    sha256: str
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
class PinnedQualificationFixture:
    """One immutable request/MSA pair used for qualification evidence."""

    name: str
    request_path: str
    request_sha256: str
    msa_path: str
    msa_sha256: str
    token_count: int
    padded_atom_count: int
    valid_atom_count: int


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
    affinity_checkpoint: PinnedArtifact
    molecular_archive: PinnedArtifact
    upstream_example_path: str
    qualification_request_path: str
    qualification_request_sha256: str
    qualification_msa_path: str
    qualification_msa_sha256: str
    reusable_profile_fixture: PinnedQualificationFixture
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
        sha256="090e82ac8c92f5e943fa1b39e7410a44027bea7243c0bbb3caa67a77fc1428e1",
        size_bytes=2_286_561_469,
    ),
    # Upstream v2.2.1 downloads this artifact while preparing every Boltz-2
    # cache, even though the initial monomer request does not run affinity.
    affinity_checkpoint=PinnedArtifact(
        filename="boltz2_aff.ckpt",
        sha256="dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e",
        size_bytes=2_062_139_170,
    ),
    molecular_archive=PinnedArtifact(
        filename="mols.tar",
        sha256="39e076d96dbec6b4e86982bbda16f3a53a2a60c9bdc17828d88f6f9a0c7d1fd7",
        size_bytes=1_855_662_080,
    ),
    upstream_example_path="examples/prot_custom_msa.yaml",
    qualification_request_path="tests/e2e/models/boltz2/data/protein_monomer.yaml",
    qualification_request_sha256=(
        "e6df7de1d6bfce519133e394c88468565b5944e53c8887da5706183147b63bea"
    ),
    qualification_msa_path="tests/e2e/models/boltz2/data/protein_monomer.a3m",
    qualification_msa_sha256=(
        "217e3be7a2aaae68e66ebd2ae60fa4f604720a029f645708c9dacbe1d067d84f"
    ),
    reusable_profile_fixture=PinnedQualificationFixture(
        name="protein_monomer_variant",
        request_path=(
            "tests/e2e/models/boltz2/data/protein_monomer_variant/"
            "protein_monomer_variant.yaml"
        ),
        request_sha256=(
            "dc604d4b30e40fd72d32aca2e7fad072b6fb7eff780e3c587e9437f8a812f826"
        ),
        msa_path=(
            "tests/e2e/models/boltz2/data/protein_monomer_variant/"
            "protein_monomer_variant.a3m"
        ),
        msa_sha256=(
            "a661423168b21c7379a8521a1fe67eed52cfbe7c33d76367fc02519245027945"
        ),
        token_count=117,
        padded_atom_count=928,
        valid_atom_count=899,
    ),
    reference_configuration=Boltz2ReferenceConfiguration(
        precision="bf16",
        recycling_steps=3,
        sampling_steps=200,
        diffusion_samples=1,
        max_msa_sequences=1024,
        seed=42,
        output_format="mmcif",
    ),
)
