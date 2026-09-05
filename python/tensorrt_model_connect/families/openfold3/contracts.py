# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenFold3-owned request, output, and qualification contracts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Final


class MoleculeKind(str, Enum):
    """Molecule types understood by the pinned upstream query format."""

    PROTEIN = "PROTEIN"
    DNA = "DNA"
    RNA = "RNA"
    LIGAND = "LIGAND"


@dataclass(frozen=True)
class SequenceInput:
    """One polymer sequence and its chain identifiers."""

    kind: MoleculeKind
    chain_ids: tuple[str, ...]
    sequence: str


@dataclass(frozen=True)
class OpenFold3Request:
    """Supported semantic subset of an OpenFold3 query."""

    query_name: str
    sequences: tuple[SequenceInput, ...]
    use_msas: bool
    use_main_msas: bool
    use_paired_msas: bool
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    seed: int = 42

    @property
    def token_count(self) -> int:
        """Return the number of polymer tokens before atomization."""

        return sum(len(sequence.sequence) * len(sequence.chain_ids) for sequence in self.sequences)


@dataclass(frozen=True)
class OpenFold3QualificationProfile:
    """Fail-closed bounds for an OpenFold3 mixed-precision profile."""

    precision: str = "fp16"
    min_tokens: int = 1
    max_tokens: int = 128
    msa_depth: int = 1
    template_count: int = 0
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1


@dataclass(frozen=True)
class OpenFold3Confidence:
    """Native confidence and ranking values for one diffusion sample."""

    average_plddt: float
    gpde: float
    ptm: float
    iptm: float
    sample_ranking_score: float | None
    plddt: tuple[float, ...]
    pde: tuple[float, ...]
    pae: tuple[float, ...]


INITIAL_FP16_PROFILE: Final = OpenFold3QualificationProfile()
INITIAL_BF16_PROFILE: Final = OpenFold3QualificationProfile(precision="bf16")
_PROTEIN_ALPHABET: Final = frozenset("ACDEFGHIKLMNPQRSTVWY")


def parse_query_json(text: str) -> OpenFold3Request:
    """Parse the exact upstream JSON subset accepted by the first profile."""

    try:
        document = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("OpenFold3 query must be valid JSON") from error
    if (
        not isinstance(document, dict)
        or "queries" not in document
        or not set(document) <= {"seeds", "queries"}
    ):
        raise ValueError("OpenFold3 query accepts only 'seeds' and required 'queries'")
    seeds = document.get("seeds", [42])
    queries = document["queries"]
    if seeds != [42]:
        raise ValueError("the qualified OpenFold3 profile requires seeds=[42]")
    if not isinstance(queries, dict) or len(queries) != 1:
        raise ValueError("the qualified OpenFold3 profile accepts exactly one query")
    query_name, raw_query = next(iter(queries.items()))
    if not isinstance(query_name, str) or not query_name:
        raise ValueError("OpenFold3 query names must be non-empty strings")
    if not isinstance(raw_query, dict):
        raise ValueError("OpenFold3 query entry must be an object")
    allowed_query_fields = {"chains", "use_msas", "use_main_msas", "use_paired_msas"}
    unknown_query_fields = set(raw_query) - allowed_query_fields
    if unknown_query_fields:
        raise ValueError(
            "unsupported OpenFold3 query fields: " + ", ".join(sorted(unknown_query_fields))
        )
    raw_chains = raw_query.get("chains")
    if not isinstance(raw_chains, list) or not raw_chains:
        raise ValueError("OpenFold3 query chains must be a non-empty list")
    sequences: list[SequenceInput] = []
    for chain in raw_chains:
        if not isinstance(chain, dict):
            raise ValueError("OpenFold3 chains must be objects")
        allowed_chain_fields = {"molecule_type", "chain_ids", "sequence"}
        unknown_chain_fields = set(chain) - allowed_chain_fields
        if unknown_chain_fields:
            raise ValueError(
                "unsupported OpenFold3 chain fields: " + ", ".join(sorted(unknown_chain_fields))
            )
        try:
            raw_kind = chain.get("molecule_type")
            kind = MoleculeKind(raw_kind.upper() if isinstance(raw_kind, str) else raw_kind)
        except ValueError as error:
            raise ValueError("unsupported OpenFold3 molecule_type") from error
        chain_ids = chain.get("chain_ids")
        sequence = chain.get("sequence")
        if isinstance(chain_ids, str):
            chain_ids = [chain_ids]
        if (
            not isinstance(chain_ids, list)
            or not chain_ids
            or any(not isinstance(chain_id, str) or not chain_id for chain_id in chain_ids)
            or not isinstance(sequence, str)
        ):
            raise ValueError("OpenFold3 chain_ids and sequence are invalid")
        sequences.append(SequenceInput(kind, tuple(chain_ids), sequence))
    request = OpenFold3Request(
        query_name=query_name,
        sequences=tuple(sequences),
        # The pinned upstream example omits these fields.  In this profile an
        # omission means the same fail-closed, query-only path as explicit
        # false: the MSA server and template search are disabled by the
        # family-owned preprocessor.
        use_msas=raw_query.get("use_msas", False),
        use_main_msas=raw_query.get("use_main_msas", False),
        use_paired_msas=raw_query.get("use_paired_msas", False),
    )
    validate_request(request)
    return request


def validate_request(
    request: OpenFold3Request,
    *,
    profile: OpenFold3QualificationProfile = INITIAL_FP16_PROFILE,
) -> None:
    """Reject requests outside the explicitly qualified inference envelope."""

    if len(request.sequences) != 1 or len(request.sequences[0].chain_ids) != 1:
        raise ValueError("the qualified OpenFold3 profile accepts one polymer chain")
    sequence = request.sequences[0]
    if sequence.kind is not MoleculeKind.PROTEIN:
        raise ValueError("the qualified OpenFold3 profile accepts protein sequences only")
    if not sequence.sequence or sequence.sequence != sequence.sequence.upper():
        raise ValueError("OpenFold3 protein sequences must be non-empty and uppercase")
    invalid = sorted(set(sequence.sequence) - _PROTEIN_ALPHABET)
    if invalid:
        raise ValueError(f"invalid OpenFold3 protein symbols: {''.join(invalid)}")
    if request.use_msas or request.use_main_msas or request.use_paired_msas:
        raise ValueError("the qualified OpenFold3 profile uses a query-only MSA")
    if not profile.min_tokens <= request.token_count <= profile.max_tokens:
        raise ValueError(
            "OpenFold3 token count is outside the qualified profile: "
            f"{request.token_count} not in [{profile.min_tokens}, {profile.max_tokens}]"
        )
    if request.recycling_steps != profile.recycling_steps:
        raise ValueError(f"OpenFold3 requires recycling_steps={profile.recycling_steps}")
    if request.sampling_steps != profile.sampling_steps:
        raise ValueError(f"OpenFold3 requires sampling_steps={profile.sampling_steps}")
    if request.diffusion_samples != profile.diffusion_samples:
        raise ValueError(f"OpenFold3 requires diffusion_samples={profile.diffusion_samples}")
    if request.seed != 42:
        raise ValueError("the qualified OpenFold3 profile requires seed 42")


def validate_confidence(
    confidence: OpenFold3Confidence, *, atom_count: int, token_count: int
) -> None:
    """Validate shape, range, and finiteness of native confidence output."""

    scalars = {
        "average_plddt": confidence.average_plddt,
        "gpde": confidence.gpde,
        "ptm": confidence.ptm,
        "iptm": confidence.iptm,
    }
    for name, value in scalars.items():
        if not math.isfinite(value):
            raise ValueError(f"OpenFold3 {name} must be finite")
    if confidence.sample_ranking_score is not None and not math.isfinite(
        confidence.sample_ranking_score
    ):
        raise ValueError("OpenFold3 sample_ranking_score must be finite when provided")
    if not 0.0 <= confidence.average_plddt <= 100.0:
        raise ValueError("OpenFold3 average_plddt must be in [0, 100]")
    if not 0.0 <= confidence.gpde <= 32.0:
        raise ValueError("OpenFold3 gpde must be in [0, 32]")
    for name in ("ptm", "iptm"):
        if not 0.0 <= scalars[name] <= 1.0:
            raise ValueError(f"OpenFold3 {name} must be in [0, 1]")
    if len(confidence.plddt) != atom_count:
        raise ValueError("OpenFold3 plddt length does not match atom count")
    if len(confidence.pde) != token_count * token_count:
        raise ValueError("OpenFold3 pde shape does not match token count")
    if len(confidence.pae) != token_count * token_count:
        raise ValueError("OpenFold3 pae shape does not match token count")
    arrays = confidence.plddt + confidence.pde + confidence.pae
    if any(not math.isfinite(value) for value in arrays):
        raise ValueError("OpenFold3 confidence arrays must contain only finite values")
