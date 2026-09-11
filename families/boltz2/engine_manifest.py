# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable multi-engine section contract for the native Boltz-2 runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Final

from .checkpoint import PINNED_PAIRFORMER


PAIRFORMER_SEGMENT_SIZE: Final = 8
ATOM_ATTENTION_FENCE_NAMES: Final = (
    "query",
    "key",
    "value",
    "logits",
    "scores",
    "probabilities",
    "attended",
    "gate",
    "gated_output",
    "projected_output",
    "output_projection",
    "result",
)


def atom_attention_fence_outputs(prefix: str) -> tuple[str, ...]:
    """Return the explicit outputs that prevent an incorrect TRT 11.2 fusion."""

    return tuple(
        f"{prefix}_{layer}_{name}"
        for layer in range(3)
        for name in ATOM_ATTENTION_FENCE_NAMES
    )


@dataclass(frozen=True)
class EngineSpec:
    role: str
    section: str
    first_block: int
    block_count: int
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class ComponentEngineSpec:
    role: str
    section: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def pairformer_engine_specs() -> tuple[EngineSpec, ...]:
    """Return the ordered, non-overlapping native runtime section contract."""

    return tuple(
        EngineSpec(
            role="pairformer",
            section=f"boltz2_pairformer_{start:02d}_{start + PAIRFORMER_SEGMENT_SIZE:02d}_plan",
            first_block=start,
            block_count=PAIRFORMER_SEGMENT_SIZE,
            inputs=("s", "z", "token_mask"),
            outputs=("s_out", "z_out"),
        )
        for start in range(0, PINNED_PAIRFORMER.num_blocks, PAIRFORMER_SEGMENT_SIZE)
    )


PAIRFORMER_ENGINE_SPECS: Final = pairformer_engine_specs()

COMPONENT_ENGINE_SPECS: Final = (
    ComponentEngineSpec(
        role="input_embedder",
        section="engine_plan",
        inputs=(
            "ref_pos",
            "ref_space_uid",
            "ref_charge",
            "ref_element",
            "ref_atom_name_chars",
            "atom_to_token",
            "atom_pad_mask",
            "res_type",
            "profile",
            "deletion_mean",
            "method_feature",
            "modified",
            "cyclic_period",
            "mol_type",
        ),
        outputs=("s_inputs",),
    ),
    ComponentEngineSpec(
        role="trunk_init",
        section="boltz2_trunk_init_plan",
        inputs=("s_inputs", "recycle_s", "recycle_z", "trunk_features"),
        outputs=("s", "z", "relative_position_encoding"),
    ),
    ComponentEngineSpec(
        role="msa",
        section="boltz2_msa_plan",
        inputs=("z", "s_inputs", "msa_features", "token_mask"),
        outputs=("z_out",),
    ),
    ComponentEngineSpec(
        role="diffusion_conditioning",
        section="boltz2_diffusion_conditioning_plan",
        inputs=("s_trunk", "z_trunk", "relative_position_encoding", "atom_features"),
        outputs=("q", "c", "atom_enc_bias", "atom_dec_bias", "token_trans_bias"),
    ),
    ComponentEngineSpec(
        role="diffusion_score_input",
        section="boltz2_diffusion_score_input_plan",
        inputs=("trunk_state", "static_conditioning", "r_noisy", "time", "atom_features"),
        outputs=(
            "a",
            "single_condition",
            "q_skip",
            "c_skip",
            *atom_attention_fence_outputs("encoder_fence"),
        ),
    ),
    *tuple(
        ComponentEngineSpec(
            role="diffusion_token_transformer",
            section=f"boltz2_diffusion_token_{start:02d}_{start + 6:02d}_plan",
            inputs=("a", "single_condition", "token_trans_bias", "token_mask"),
            outputs=("a_out",),
        )
        for start in range(0, 24, 6)
    ),
    ComponentEngineSpec(
        role="diffusion_score_output",
        section="boltz2_diffusion_score_output_plan",
        inputs=("a", "q_skip", "c_skip", "atom_dec_bias", "atom_features"),
        outputs=("r_update", *atom_attention_fence_outputs("decoder_fence")),
    ),
    ComponentEngineSpec(
        role="confidence",
        section="boltz2_confidence_plan",
        inputs=("s_inputs", "s", "z", "x_pred", "confidence_features"),
        outputs=(
            "pae_logits",
            "pde_logits",
            "plddt_logits",
            "resolved_logits",
            "representative_distance",
            "pdistogram",
            "pbfactor",
        ),
    ),
)

ALL_ENGINE_SPECS: Final = (
    *COMPONENT_ENGINE_SPECS[:3],
    *PAIRFORMER_ENGINE_SPECS,
    *COMPONENT_ENGINE_SPECS[3:],
)


def validate_engine_specs(specs=PAIRFORMER_ENGINE_SPECS) -> None:
    """Fail closed unless specs cover every checkpoint block exactly once."""

    if len({spec.section for spec in specs}) != len(specs):
        raise ValueError("Boltz-2 engine section names must be unique")
    covered = [
        block
        for spec in specs
        for block in range(spec.first_block, spec.first_block + spec.block_count)
    ]
    expected = list(range(PINNED_PAIRFORMER.num_blocks))
    if covered != expected:
        raise ValueError("Boltz-2 Pairformer engine specs must cover blocks 0 through 63 once")


def graph_manifest_json(
    *,
    token_count: int,
    tensorrt_version: str,
    atom_count: int = 928,
    sampling_steps: int = 200,
) -> bytes:
    """Serialize graph roles and bindings for bundle inspection and native loading."""

    validate_engine_specs()
    if token_count <= 0:
        raise ValueError("Boltz-2 graph manifest token_count must be positive")
    if not tensorrt_version:
        raise ValueError("Boltz-2 graph manifest requires a TensorRT version")
    if atom_count <= 0 or sampling_steps <= 0:
        raise ValueError("Boltz-2 graph manifest shape and sampling counts must be positive")
    sections = [spec.section for spec in ALL_ENGINE_SPECS]
    if len(sections) != len(set(sections)):
        raise ValueError("Boltz-2 graph manifest engine section names must be unique")
    document = {
        "schema_version": 1,
        "family": "boltz2",
        "precision": "bf16-mixed",
        "token_count": token_count,
        "atom_count": atom_count,
        "recycling_passes": 4,
        "sampling_steps": sampling_steps,
        "tensorrt_version": tensorrt_version,
        "engines": [asdict(spec) for spec in ALL_ENGINE_SPECS],
    }
    return json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
