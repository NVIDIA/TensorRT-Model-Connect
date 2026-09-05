# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned public provenance for the OpenFold3 FP16 qualification target."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PinnedArtifact:
    """One immutable file required by the pinned workflow."""

    filename: str
    url: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class OpenFold3ReferenceConfiguration:
    """Inference controls shared by reference and TensorRT qualification."""

    precision: str
    recycling_steps: int
    sampling_steps: int
    diffusion_samples: int
    seed: int
    output_format: str


@dataclass(frozen=True)
class OpenFold3Provenance:
    """Immutable source, checkpoint, and configuration identity."""

    source_repository: str
    source_revision: str
    source_tag: str
    source_license: str
    checkpoint_name: str
    checkpoint_license: str
    checkpoint: PinnedArtifact
    chemical_components: PinnedArtifact
    upstream_example_path: str
    reference_configuration: OpenFold3ReferenceConfiguration


PINNED_OPENFOLD3 = OpenFold3Provenance(
    source_repository="https://github.com/aqlaboratory/openfold-3.git",
    source_revision="c4771653c5d0a3ebb0b3af71b05efd64bc44ee86",
    source_tag="v0.5.0",
    source_license="Apache-2.0",
    checkpoint_name="openbind-2025-06-30-174k",
    checkpoint_license="Apache-2.0",
    checkpoint=PinnedArtifact(
        filename="of3-ob-2025-06-30-174k.pt",
        url=(
            "https://openfold3-data.s3.amazonaws.com/openfold3-parameters/of3-ob-2025-06-30-174k.pt"
        ),
        sha256="bd43301c011d5f87580d3e8b548658869433e4488399feb03035ba248f8e29e4",
        size_bytes=2_287_872_989,
    ),
    chemical_components=PinnedArtifact(
        filename="components.bcif",
        url="https://openfold3-data.s3.amazonaws.com/components.bcif",
        sha256="473d845c8b250b188dbed9bf505ae206692a178a2a7c4869bf8f9de707ffcc0c",
        size_bytes=63_393_643,
    ),
    upstream_example_path="examples/example_inference_inputs/query_ubiquitin.json",
    reference_configuration=OpenFold3ReferenceConfiguration(
        precision="fp16-mixed",
        recycling_steps=3,
        sampling_steps=200,
        diffusion_samples=1,
        seed=42,
        output_format="mmcif",
    ),
)
