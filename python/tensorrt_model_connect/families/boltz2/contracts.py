# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boltz-2 request, output, and qualification contracts.

This module deliberately contains no graph code. It is the family-owned source
of truth shared by the builder, native runtime tests, reference runner, and
documentation while the TensorRT graph remains independently replaceable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Final, TypeAlias


class PolymerKind(str, Enum):
    PROTEIN = "protein"
    DNA = "dna"
    RNA = "rna"


class StructureFormat(str, Enum):
    MMCIF = "mmcif"
    PDB = "pdb"


@dataclass(frozen=True)
class ResidueModification:
    position: int
    ccd: str


@dataclass(frozen=True)
class SequenceInput:
    kind: PolymerKind
    chain_ids: tuple[str, ...]
    sequence: str
    msa_path: PurePosixPath | None = None
    modifications: tuple[ResidueModification, ...] = ()
    cyclic: bool = False


@dataclass(frozen=True)
class LigandInput:
    chain_ids: tuple[str, ...]
    smiles: str | None = None
    ccd: str | None = None


@dataclass(frozen=True)
class TemplateInput:
    path: PurePosixPath
    chain_ids: tuple[str, ...] = ()
    template_ids: tuple[str, ...] = ()
    force: bool = False
    threshold_angstrom: float | None = None


@dataclass(frozen=True)
class AtomReference:
    chain_id: str
    residue_index: int
    atom_name: str


@dataclass(frozen=True)
class TokenReference:
    chain_id: str
    residue_or_atom: int | str


@dataclass(frozen=True)
class BondConstraint:
    atom1: AtomReference
    atom2: AtomReference


@dataclass(frozen=True)
class ContactConstraint:
    token1: TokenReference
    token2: TokenReference
    max_distance_angstrom: float = 6.0
    force: bool = False


@dataclass(frozen=True)
class PocketConstraint:
    binder_chain_id: str
    contacts: tuple[TokenReference, ...]
    max_distance_angstrom: float = 6.0
    force: bool = False


Boltz2Constraint: TypeAlias = BondConstraint | ContactConstraint | PocketConstraint


@dataclass(frozen=True)
class Boltz2Request:
    sequences: tuple[SequenceInput, ...]
    ligands: tuple[LigandInput, ...] = ()
    templates: tuple[TemplateInput, ...] = ()
    constraints: tuple[Boltz2Constraint, ...] = ()
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    seed: int = 42
    output_format: StructureFormat = StructureFormat.MMCIF

    @property
    def token_count(self) -> int:
        return sum(len(item.sequence) * len(item.chain_ids) for item in self.sequences)


@dataclass(frozen=True)
class Boltz2Confidence:
    confidence_score: float
    ptm: float
    iptm: float
    ligand_iptm: float
    protein_iptm: float
    complex_plddt: float
    complex_iplddt: float
    plddt: tuple[float, ...]


@dataclass(frozen=True)
class Boltz2QualificationProfile:
    precision: str = "bf16"
    # Each bundle has static plans, but the build accepts the bounded sequence
    # lengths exercised by the qualification and variable-length E2E fixtures.
    min_tokens: int = 1
    opt_tokens: int = 117
    max_tokens: int = 117
    min_msa_depth: int = 1
    opt_msa_depth: int = 1
    max_msa_depth: int = 1
    min_padded_atoms: int = 32
    max_padded_atoms: int = 928
    atom_window_queries: int = 32
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1


@dataclass(frozen=True)
class Boltz2InputSupport:
    """Fail-closed input envelope for the first BF16 qualification."""

    polymer_kinds: frozenset[PolymerKind]
    custom_a3m: bool
    paired_msa_csv: bool
    msa_server: bool
    templates: bool
    ligands: bool
    constraints: bool
    modifications: bool
    cyclic_polymers: bool
    affinity: bool


INITIAL_BF16_PROFILE: Final = Boltz2QualificationProfile()
INITIAL_BF16_INPUT_SUPPORT: Final = Boltz2InputSupport(
    polymer_kinds=frozenset({PolymerKind.PROTEIN}),
    custom_a3m=True,
    paired_msa_csv=False,
    msa_server=False,
    templates=False,
    ligands=False,
    constraints=False,
    modifications=False,
    cyclic_polymers=False,
    affinity=False,
)


_ALPHABETS: Final = {
    PolymerKind.PROTEIN: frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO"),
    PolymerKind.DNA: frozenset("ACGTN"),
    PolymerKind.RNA: frozenset("ACGUN"),
}


def validate_request(
    request: Boltz2Request,
    *,
    profile: Boltz2QualificationProfile = INITIAL_BF16_PROFILE,
    support: Boltz2InputSupport = INITIAL_BF16_INPUT_SUPPORT,
) -> None:
    """Reject requests outside the initial, explicitly qualified envelope."""

    if not request.sequences:
        raise ValueError("Boltz-2 requires at least one polymer sequence")
    seen_chain_ids: set[str] = set()
    for item in request.sequences:
        if item.kind not in support.polymer_kinds:
            raise ValueError(
                f"{item.kind.value} polymers are outside the qualified Boltz-2 input profile"
            )
        if not item.chain_ids:
            raise ValueError("each Boltz-2 sequence requires at least one chain ID")
        if not item.sequence:
            raise ValueError("Boltz-2 sequences must not be empty")
        if item.sequence != item.sequence.upper():
            raise ValueError("Boltz-2 polymer sequences must use uppercase residue symbols")
        invalid = sorted(set(item.sequence.upper()) - _ALPHABETS[item.kind])
        if invalid:
            raise ValueError(
                f"invalid {item.kind.value} residue symbols: {''.join(invalid)}"
            )
        for chain_id in item.chain_ids:
            if not chain_id or any(character.isspace() for character in chain_id):
                raise ValueError("Boltz-2 chain IDs must be non-empty and contain no whitespace")
            if chain_id in seen_chain_ids:
                raise ValueError(f"duplicate Boltz-2 chain ID: {chain_id}")
            seen_chain_ids.add(chain_id)
        if item.msa_path is None:
            raise ValueError("the qualified Boltz-2 profile requires a custom A3M path")
        if not support.custom_a3m or item.msa_path.suffix.lower() != ".a3m":
            raise ValueError("the qualified Boltz-2 profile accepts single-chain A3M input only")
        if item.msa_path.is_absolute() or ".." in item.msa_path.parts:
            raise ValueError(
                "Boltz-2 A3M paths must be relative and remain inside the request root"
            )
        if item.modifications and not support.modifications:
            raise ValueError("polymer modifications are outside the qualified Boltz-2 profile")
        if item.cyclic and not support.cyclic_polymers:
            raise ValueError("cyclic polymers are outside the qualified Boltz-2 profile")

    if request.ligands and not support.ligands:
        raise ValueError("ligands are outside the qualified Boltz-2 profile")
    if request.templates and not support.templates:
        raise ValueError("templates are outside the qualified Boltz-2 profile")
    if request.constraints and not support.constraints:
        raise ValueError("constraints are outside the qualified Boltz-2 profile")
    if len(request.sequences) != 1 or len(request.sequences[0].chain_ids) != 1:
        raise ValueError("the qualified Boltz-2 profile accepts exactly one protein chain")

    if not profile.min_tokens <= request.token_count <= profile.max_tokens:
        raise ValueError(
            "Boltz-2 token count is outside the qualified BF16 profile: "
            f"{request.token_count} not in [{profile.min_tokens}, {profile.max_tokens}]"
        )
    if request.recycling_steps != profile.recycling_steps:
        raise ValueError(
            f"Boltz-2 qualification requires recycling_steps={profile.recycling_steps}"
        )
    if request.sampling_steps != profile.sampling_steps:
        raise ValueError(
            f"Boltz-2 qualification requires sampling_steps={profile.sampling_steps}"
        )
    if request.diffusion_samples != profile.diffusion_samples:
        raise ValueError(
            f"Boltz-2 qualification requires diffusion_samples={profile.diffusion_samples}"
        )
    if request.seed < 0 or request.seed > 2_147_483_647:
        raise ValueError("Boltz-2 seed must be in [0, 2147483647]")
    if request.output_format is not StructureFormat.MMCIF:
        raise ValueError("the qualified Boltz-2 profile supports mmCIF output only")


def parse_request_yaml(text: str) -> Boltz2Request:
    """Parse the supported YAML subset without accepting ignored fields."""

    import yaml

    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError("Boltz-2 request must be a YAML mapping")
    if any(not isinstance(key, str) for key in document):
        raise ValueError("Boltz-2 request field names must be strings")
    unknown = set(document) - {"version", "sequences"}
    if unknown:
        raise ValueError(f"unsupported Boltz-2 request fields: {', '.join(sorted(unknown))}")
    if document.get("version") != 1:
        raise ValueError("Boltz-2 request version must be 1")
    raw_sequences = document.get("sequences")
    if not isinstance(raw_sequences, list):
        raise ValueError("Boltz-2 request sequences must be a list")
    sequences: list[SequenceInput] = []
    for raw_entry in raw_sequences:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {"protein"}:
            raise ValueError("the qualified Boltz-2 request accepts protein entries only")
        protein = raw_entry["protein"]
        if not isinstance(protein, dict):
            raise ValueError("Boltz-2 protein entry must be a mapping")
        if any(not isinstance(key, str) for key in protein):
            raise ValueError("Boltz-2 protein field names must be strings")
        unknown_protein = set(protein) - {"id", "sequence", "msa"}
        if unknown_protein:
            raise ValueError(
                "unsupported Boltz-2 protein fields: "
                + ", ".join(sorted(unknown_protein))
            )
        chain_id = protein.get("id")
        sequence = protein.get("sequence")
        msa = protein.get("msa")
        if not all(isinstance(value, str) for value in (chain_id, sequence, msa)):
            raise ValueError("Boltz-2 protein id, sequence, and msa must be strings")
        sequences.append(
            SequenceInput(
                kind=PolymerKind.PROTEIN,
                chain_ids=(chain_id,),
                sequence=sequence,
                msa_path=PurePosixPath(msa),
            )
        )
    request = Boltz2Request(sequences=tuple(sequences))
    validate_request(request)
    return request


def validate_a3m(
    text: str,
    *,
    expected_query: str | None = None,
    profile: Boltz2QualificationProfile = INITIAL_BF16_PROFILE,
) -> tuple[str, ...]:
    """Validate an A3M document and return its aligned sequence rows.

    Lowercase insertion characters are accepted and removed when comparing the
    query row with the requested polymer sequence, matching A3M semantics.
    """

    rows: list[str] = []
    current: list[str] = []
    saw_header = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if len(line) == 1:
                raise ValueError(f"A3M header on line {line_number} is empty")
            if saw_header:
                if not current:
                    raise ValueError(f"A3M record before line {line_number} has no sequence")
                rows.append("".join(current))
                current = []
            saw_header = True
            continue
        if not saw_header:
            raise ValueError(f"A3M sequence data appears before a header on line {line_number}")
        if any(not (character.isalpha() or character in "-.") for character in line):
            raise ValueError(f"A3M sequence on line {line_number} contains invalid symbols")
        current.append(line)
    if saw_header:
        if not current:
            raise ValueError("last A3M record has no sequence")
        rows.append("".join(current))
    if not rows:
        raise ValueError("A3M document contains no records")
    if not profile.min_msa_depth <= len(rows) <= profile.max_msa_depth:
        raise ValueError(
            "Boltz-2 MSA depth is outside the qualified BF16 profile: "
            f"{len(rows)} not in [{profile.min_msa_depth}, {profile.max_msa_depth}]"
        )

    if expected_query is not None:
        query = "".join(character for character in rows[0] if not character.islower())
        query = query.replace("-", "").replace(".", "").upper()
        if query != expected_query.upper():
            raise ValueError("A3M query row does not match the requested polymer sequence")
    return tuple(rows)


def validate_confidence(confidence: Boltz2Confidence, *, token_count: int) -> None:
    """Validate the model-owned structure confidence output contract."""

    scalars = {
        "confidence_score": confidence.confidence_score,
        "ptm": confidence.ptm,
        "iptm": confidence.iptm,
        "ligand_iptm": confidence.ligand_iptm,
        "protein_iptm": confidence.protein_iptm,
        "complex_plddt": confidence.complex_plddt,
        "complex_iplddt": confidence.complex_iplddt,
    }
    for name, value in scalars.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Boltz-2 {name} must be in [0, 1]")
    if len(confidence.plddt) != token_count:
        raise ValueError(
            f"Boltz-2 plddt length {len(confidence.plddt)} does not match token count {token_count}"
        )
    if any(not 0.0 <= value <= 1.0 for value in confidence.plddt):
        raise ValueError("Boltz-2 plddt values must be in [0, 1]")
