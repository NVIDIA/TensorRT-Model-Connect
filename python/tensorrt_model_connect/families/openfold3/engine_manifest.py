# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable multi-engine binding contract for native OpenFold3 inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Final


PAIRFORMER_SEGMENT_SIZE: Final = 6
DIFFUSION_SEGMENT_SIZE: Final = 6


@dataclass(frozen=True)
class EngineSpec:
    role: str
    section: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


PAIRFORMER_ENGINE_SPECS: Final = tuple(
    EngineSpec(
        "pairformer",
        f"openfold3_pairformer_{start:02d}_{start + 6:02d}_plan",
        ("s", "z", "token_mask"),
        ("s_out", "z_out"),
    )
    for start in range(0, 48, 6)
)
DIFFUSION_ENGINE_SPECS: Final = tuple(
    EngineSpec(
        "diffusion_token",
        f"openfold3_diffusion_token_{start:02d}_{start + 6:02d}_plan",
        ("a", "single_condition", "pair_condition", "token_mask"),
        ("a_out",),
    )
    for start in range(0, 24, 6)
)
COMPONENT_ENGINE_SPECS: Final = (
    EngineSpec(
        "input_embedder",
        "engine_plan",
        (
            "ref_pos",
            "ref_mask",
            "ref_element",
            "ref_charge",
            "ref_atom_name_chars",
            "ref_space_uid",
            "atom_mask",
            "atom_to_token_index",
            "restype",
            "profile",
            "deletion_mean",
            "relpos",
            "token_bonds",
        ),
        ("s_input", "s_init", "z_init"),
    ),
    EngineSpec(
        "trunk_cycle",
        "openfold3_trunk_cycle_plan",
        (
            "s_input",
            "s_init",
            "z_init",
            "s_previous",
            "z_previous",
            "token_mask",
            "msa",
            "has_deletion",
            "deletion_value",
            "msa_mask",
        ),
        ("s", "z"),
    ),
    EngineSpec(
        "diffusion_conditioning",
        "openfold3_diffusion_conditioning_plan",
        ("noise_level", "s_input", "s_trunk", "z_trunk", "relpos", "token_mask"),
        ("s_conditioned", "z_conditioned"),
    ),
    EngineSpec(
        "diffusion_score_input",
        "openfold3_diffusion_score_input_plan",
        (
            "ref_pos",
            "ref_mask",
            "ref_element",
            "ref_charge",
            "ref_atom_name_chars",
            "ref_space_uid",
            "atom_mask",
            "atom_to_token_index",
            "noisy_positions",
            "noise_level",
            "s_conditioned",
            "s_trunk",
            "z_conditioned",
        ),
        (
            "token_representation",
            "atom_representation",
            "atom_conditioning",
            "atom_pair_conditioning",
        ),
    ),
    EngineSpec(
        "diffusion_score_output",
        "openfold3_diffusion_score_output_plan",
        (
            "a",
            "atom_representation",
            "atom_conditioning",
            "atom_pair_conditioning",
            "atom_to_token_index",
            "atom_mask",
            "noisy_positions",
            "noise_level",
        ),
        ("denoised_positions",),
    ),
    EngineSpec(
        "confidence",
        "openfold3_confidence_plan",
        (
            "s_input",
            "s",
            "z",
            "positions",
            "representative_atom_map",
            "atom_head_index",
            "token_mask",
        ),
        (
            "pae_logits",
            "pde_logits",
            "plddt_logits",
            "experimentally_resolved_logits",
            "distogram_logits",
        ),
    ),
)
ALL_ENGINE_SPECS: Final = (
    *COMPONENT_ENGINE_SPECS[:2],
    *PAIRFORMER_ENGINE_SPECS,
    *COMPONENT_ENGINE_SPECS[2:4],
    *DIFFUSION_ENGINE_SPECS,
    *COMPONENT_ENGINE_SPECS[4:],
)


def graph_manifest_json(
    *,
    token_count: int,
    atom_count: int,
    padded_atom_count: int,
    tensorrt_version: str,
    precision: str = "fp16",
) -> bytes:
    if token_count <= 0 or atom_count <= 0 or padded_atom_count < atom_count:
        raise ValueError("invalid OpenFold3 graph manifest shape")
    if precision not in {"fp16", "bf16"}:
        raise ValueError(f"unsupported OpenFold3 graph precision: {precision}")
    sections = [spec.section for spec in ALL_ENGINE_SPECS]
    if len(sections) != len(set(sections)):
        raise ValueError("OpenFold3 graph engine section names must be unique")
    document = {
        "schema_version": 1,
        "family": "openfold3",
        "source_revision": "c4771653c5d0a3ebb0b3af71b05efd64bc44ee86",
        "precision": f"{precision}-mixed",
        "token_count": token_count,
        "atom_count": atom_count,
        "padded_atom_count": padded_atom_count,
        "msa_depth": 1,
        "template_mode": "four_identical_disabled_search_placeholders",
        "recycling_passes": 4,
        "sampling_steps": 200,
        "engines": [asdict(spec) for spec in ALL_ENGINE_SPECS],
        "tensorrt_version": tensorrt_version,
    }
    return json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
