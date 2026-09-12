# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boltz-2 request and profile contracts.

This module deliberately contains no graph code. It is the family-owned source
of truth shared by the builder and native runtime tests.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Final


class PolymerKind(str, Enum):
    PROTEIN = "protein"
    DNA = "dna"
    RNA = "rna"


class StructureFormat(str, Enum):
    MMCIF = "mmcif"


@dataclass(frozen=True)
class PolymerModification:
    ccd: str
    position: int


@dataclass(frozen=True)
class SequenceInput:
    kind: PolymerKind
    chain_ids: tuple[str, ...]
    sequence: str
    msa_path: PurePosixPath | None = None
    cyclic: bool = False
    modifications: tuple[PolymerModification, ...] = ()


@dataclass(frozen=True)
class LigandInput:
    chain_ids: tuple[str, ...]
    smiles: str | None = None
    ccd: tuple[str, ...] = ()


@dataclass(frozen=True)
class AffinityProperty:
    binder: str


@dataclass(frozen=True)
class BondConstraint:
    atom1: tuple[str, int, str]
    atom2: tuple[str, int, str]


TokenSpec = tuple[str, int | str]


@dataclass(frozen=True)
class PocketConstraint:
    binder: str
    contacts: tuple[TokenSpec, ...]
    max_distance: float = 6.0
    force: bool = False


@dataclass(frozen=True)
class ContactConstraint:
    token1: TokenSpec
    token2: TokenSpec
    max_distance: float = 6.0
    force: bool = False


Constraint = BondConstraint | PocketConstraint | ContactConstraint
EntityInput = SequenceInput | LigandInput


@dataclass(frozen=True)
class TemplateInput:
    path: PurePosixPath
    format: str
    chain_ids: tuple[str, ...] | None = None
    template_chain_ids: tuple[str, ...] | None = None
    force: bool = False
    threshold: float | None = None


@dataclass(frozen=True)
class Boltz2Request:
    sequences: tuple[EntityInput, ...]
    templates: tuple[TemplateInput, ...] = ()
    affinity: AffinityProperty | None = None
    constraints: tuple[Constraint, ...] = ()
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    seed: int = 42
    output_format: StructureFormat = StructureFormat.MMCIF

    @property
    def token_count(self) -> int:
        count = 0
        for item in self.sequences:
            if isinstance(item, SequenceInput):
                count += len(item.sequence) * len(item.chain_ids)
            else:
                count += max(1, len(item.ccd)) * len(item.chain_ids)
        return count


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
    max_templates: int = 4
    min_padded_atoms: int = 32
    max_padded_atoms: int = 928
    atom_window_queries: int = 32
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1


INITIAL_BF16_PROFILE: Final = Boltz2QualificationProfile(
    opt_msa_depth=8,
    max_msa_depth=8,
)


_PROTEIN_ALPHABET: Final = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")
_DNA_ALPHABET: Final = frozenset("ACGTN")
_RNA_ALPHABET: Final = frozenset("ACGUN")
_POLYMER_ALPHABETS: Final = {
    PolymerKind.PROTEIN: _PROTEIN_ALPHABET,
    PolymerKind.DNA: _DNA_ALPHABET,
    PolymerKind.RNA: _RNA_ALPHABET,
}


def _safe_relative_path(path: PurePosixPath, label: str) -> None:
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Boltz-2 {label} paths must remain inside the request root")


def _validate_chain_ids(chain_ids: tuple[str, ...], seen: set[str]) -> None:
    if not chain_ids:
        raise ValueError("each Boltz-2 entity requires at least one chain ID")
    for chain_id in chain_ids:
        if (
            not chain_id
            or not chain_id.isascii()
            or any(not (character.isalnum() or character == "_") for character in chain_id)
        ):
            raise ValueError(
                "Boltz-2 chain IDs must contain only ASCII letters, digits, or underscore"
            )
        if chain_id in seen:
            raise ValueError(f"duplicate Boltz-2 chain ID: {chain_id}")
        seen.add(chain_id)


def _validate_ccd(value: str, label: str) -> None:
    if (
        not 1 <= len(value) <= 5
        or not value.isascii()
        or not value.isalnum()
        or value != value.upper()
    ):
        raise ValueError(f"Boltz-2 {label} must be 1-5 uppercase ASCII letters/digits")


def _validate_token_spec(spec: TokenSpec, entities: dict[str, EntityInput], label: str) -> None:
    chain_id, position = spec
    entity = entities.get(chain_id)
    if entity is None:
        raise ValueError(f"Boltz-2 {label} references unknown chain {chain_id!r}")
    if isinstance(entity, LigandInput):
        if not isinstance(position, str) or not position or len(position) > 4:
            raise ValueError(f"Boltz-2 {label} ligand tokens require an atom name")
    elif (
        not isinstance(position, int)
        or isinstance(position, bool)
        or not 1 <= position <= len(entity.sequence)
    ):
        raise ValueError(f"Boltz-2 {label} polymer token is outside its sequence")


def validate_request(
    request: Boltz2Request,
    *,
    profile: Boltz2QualificationProfile = INITIAL_BF16_PROFILE,
) -> None:
    """Reject requests outside the initial, explicitly qualified envelope."""

    if not request.sequences:
        raise ValueError("Boltz-2 requires at least one sequence entity")
    seen_chain_ids: set[str] = set()
    protein_chain_ids: set[str] = set()
    entities: dict[str, EntityInput] = {}
    entity_options: dict[
        tuple[PolymerKind, str],
        tuple[PurePosixPath | None, bool, tuple[PolymerModification, ...]],
    ] = {}
    for item in request.sequences:
        _validate_chain_ids(item.chain_ids, seen_chain_ids)
        entities.update(dict.fromkeys(item.chain_ids, item))
        if isinstance(item, LigandInput):
            if (item.smiles is None) == (not item.ccd):
                raise ValueError("Boltz-2 ligands require exactly one of smiles or ccd")
            if item.smiles is not None and not item.smiles.strip():
                raise ValueError("Boltz-2 ligand SMILES must not be empty")
            for ccd in item.ccd:
                _validate_ccd(ccd, "ligand CCD names")
            continue
        if not item.sequence:
            raise ValueError("Boltz-2 sequences must not be empty")
        if item.sequence != item.sequence.upper():
            raise ValueError("Boltz-2 polymer sequences must use uppercase residue symbols")
        invalid = sorted(set(item.sequence.upper()) - _POLYMER_ALPHABETS[item.kind])
        if invalid:
            raise ValueError(f"invalid {item.kind.value} residue symbols: {''.join(invalid)}")
        for chain_id in item.chain_ids:
            if item.kind is PolymerKind.PROTEIN:
                protein_chain_ids.add(chain_id)
        if item.kind is PolymerKind.PROTEIN:
            if item.msa_path is not None:
                if item.msa_path.suffix.lower() not in {".a3m", ".csv"}:
                    raise ValueError("Boltz-2 protein MSA must be A3M, CSV, or 'empty'")
                _safe_relative_path(item.msa_path, "MSA")
        elif item.msa_path is not None:
            raise ValueError("Boltz-2 DNA and RNA entries do not accept an MSA")
        for modification in item.modifications:
            _validate_ccd(modification.ccd, "CCD modification names")
            if not 1 <= modification.position <= len(item.sequence):
                raise ValueError("Boltz-2 modification position is outside its polymer sequence")
        entity = (item.kind, item.sequence)
        options = (item.msa_path, item.cyclic, item.modifications)
        previous_options = entity_options.setdefault(entity, options)
        if previous_options != options:
            raise ValueError(
                "polymers with the same type and sequence must share MSA, cyclic, "
                "and modification settings"
            )
    if request.affinity is not None:
        binder = entities.get(request.affinity.binder)
        if not isinstance(binder, LigandInput):
            raise ValueError("Boltz-2 affinity binder must name one ligand chain")
        if len(binder.chain_ids) != 1:
            raise ValueError("Boltz-2 affinity does not support ligand copies")
        if len(binder.ccd) > 1:
            raise ValueError("Boltz-2 affinity does not support multi-residue ligands")
        if not protein_chain_ids:
            raise ValueError("Boltz-2 affinity requires at least one protein receptor")
    if len(request.templates) > profile.max_templates:
        raise ValueError(f"Boltz-2 accepts at most {profile.max_templates} templates per request")
    template_names: set[str] = set()
    for template in request.templates:
        _safe_relative_path(template.path, "template")
        if template.format not in {"cif", "pdb"}:
            raise ValueError("Boltz-2 templates must use CIF or PDB format")
        if template.chain_ids is not None:
            unsupported = set(template.chain_ids) - protein_chain_ids
            if unsupported:
                raise ValueError(
                    "Boltz-2 template references non-protein or unknown request chains: "
                    + ", ".join(sorted(unsupported))
                )
        if template.path.stem in template_names:
            raise ValueError("Boltz-2 template filenames must have unique stems")
        template_names.add(template.path.stem)
        if (
            template.chain_ids is not None
            and template.template_chain_ids is not None
            and len(template.chain_ids) != len(template.template_chain_ids)
        ):
            raise ValueError("Boltz-2 template chain mappings must have equal lengths")
        if template.force or template.threshold is not None:
            raise ValueError("Boltz-2 forced template potentials are not supported by this profile")
    for constraint in request.constraints:
        if isinstance(constraint, BondConstraint):
            for label, atom in (("bond atom1", constraint.atom1), ("bond atom2", constraint.atom2)):
                chain_id, residue, atom_name = atom
                entity = entities.get(chain_id)
                if entity is None:
                    raise ValueError(f"Boltz-2 {label} references unknown chain {chain_id!r}")
                residue_count = (
                    len(entity.ccd) if isinstance(entity, LigandInput) else len(entity.sequence)
                )
                residue_count = max(1, residue_count)
                if not 1 <= residue <= residue_count or not atom_name or len(atom_name) > 4:
                    raise ValueError(f"Boltz-2 {label} is outside its entity")
            continue
        if not math.isfinite(constraint.max_distance) or constraint.max_distance <= 0:
            raise ValueError("Boltz-2 contact distances must be finite and positive")
        if constraint.force:
            raise ValueError("Boltz-2 forced contact potentials are not supported by this profile")
        if isinstance(constraint, PocketConstraint):
            if not isinstance(entities.get(constraint.binder), LigandInput):
                raise ValueError("Boltz-2 pocket binder must name a ligand chain")
            if not constraint.contacts:
                raise ValueError("Boltz-2 pocket constraints require at least one contact")
            for contact in constraint.contacts:
                _validate_token_spec(contact, entities, "pocket contact")
        else:
            _validate_token_spec(constraint.token1, entities, "contact token1")
            _validate_token_spec(constraint.token2, entities, "contact token2")
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
        raise ValueError(f"Boltz-2 qualification requires sampling_steps={profile.sampling_steps}")
    if request.diffusion_samples != profile.diffusion_samples:
        raise ValueError(
            f"Boltz-2 qualification requires diffusion_samples={profile.diffusion_samples}"
        )
    if request.seed < 0 or request.seed > 2_147_483_647:
        raise ValueError("Boltz-2 seed must be in [0, 2147483647]")
    if request.output_format is not StructureFormat.MMCIF:
        raise ValueError("the qualified Boltz-2 profile supports mmCIF output only")


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        return tuple(value)
    raise ValueError(f"Boltz-2 {label} must be a string or non-empty list of strings")


def _modifications(value: object) -> tuple[PolymerModification, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Boltz-2 polymer modifications must be a list")
    result = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"ccd", "position"}:
            raise ValueError("Boltz-2 modifications require exactly ccd and position")
        if (
            not isinstance(item["ccd"], str)
            or not isinstance(item["position"], int)
            or isinstance(item["position"], bool)
        ):
            raise ValueError("Boltz-2 modification ccd must be a string and position an integer")
        result.append(PolymerModification(item["ccd"], item["position"]))
    return tuple(result)


def _token_spec(value: object, label: str) -> TokenSpec:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not isinstance(value[0], str)
        or not isinstance(value[1], (int, str))
        or isinstance(value[1], bool)
    ):
        raise ValueError(f"Boltz-2 {label} must be [chain, residue-index-or-atom-name]")
    return value[0], value[1]


def _atom_spec(value: object, label: str) -> tuple[str, int, str]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not isinstance(value[0], str)
        or not isinstance(value[1], int)
        or isinstance(value[1], bool)
        or not isinstance(value[2], str)
    ):
        raise ValueError(f"Boltz-2 {label} must be [chain, residue-index, atom-name]")
    return value[0], value[1], value[2]


def _distance_and_force(value: dict[object, object]) -> tuple[float, bool]:
    distance = value.get("max_distance", 6.0)
    force = value.get("force", False)
    if (
        not isinstance(distance, (int, float))
        or isinstance(distance, bool)
        or not isinstance(force, bool)
    ):
        raise ValueError("Boltz-2 constraint max_distance/force values are invalid")
    return float(distance), force


def _parse_constraints(value: object) -> tuple[Constraint, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Boltz-2 constraints must be a list")
    result: list[Constraint] = []
    for entry in value:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("each Boltz-2 constraint requires exactly one constraint type")
        kind, fields = next(iter(entry.items()))
        if not isinstance(fields, dict) or any(not isinstance(key, str) for key in fields):
            raise ValueError(f"Boltz-2 {kind} constraint must be a mapping")
        if kind == "bond":
            if set(fields) != {"atom1", "atom2"}:
                raise ValueError("Boltz-2 bond constraints require exactly atom1 and atom2")
            result.append(
                BondConstraint(
                    _atom_spec(fields["atom1"], "bond atom1"),
                    _atom_spec(fields["atom2"], "bond atom2"),
                )
            )
        elif kind == "pocket":
            if set(fields) - {"binder", "contacts", "max_distance", "force"}:
                raise ValueError("unsupported Boltz-2 pocket constraint fields")
            binder, contacts = fields.get("binder"), fields.get("contacts")
            if not isinstance(binder, str) or not isinstance(contacts, list):
                raise ValueError("Boltz-2 pocket constraints require binder and contacts")
            distance, force = _distance_and_force(fields)
            result.append(
                PocketConstraint(
                    binder,
                    tuple(_token_spec(item, "pocket contact") for item in contacts),
                    distance,
                    force,
                )
            )
        elif kind == "contact":
            if set(fields) - {"token1", "token2", "max_distance", "force"}:
                raise ValueError("unsupported Boltz-2 contact constraint fields")
            if "token1" not in fields or "token2" not in fields:
                raise ValueError("Boltz-2 contact constraints require token1 and token2")
            distance, force = _distance_and_force(fields)
            result.append(
                ContactConstraint(
                    _token_spec(fields["token1"], "contact token1"),
                    _token_spec(fields["token2"], "contact token2"),
                    distance,
                    force,
                )
            )
        else:
            raise ValueError(f"unsupported Boltz-2 constraint type: {kind}")
    return tuple(result)


def parse_request_yaml(text: str) -> Boltz2Request:
    """Parse the supported Boltz YAML/JSON subset without ignored fields."""

    import yaml

    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError("Boltz-2 request must be a YAML/JSON mapping")
    if any(not isinstance(key, str) for key in document):
        raise ValueError("Boltz-2 request field names must be strings")
    unknown = set(document) - {"version", "sequences", "templates", "properties", "constraints"}
    if unknown:
        raise ValueError(f"unsupported Boltz-2 request fields: {', '.join(sorted(unknown))}")
    if document.get("version") != 1:
        raise ValueError("Boltz-2 request version must be 1")
    raw_sequences = document.get("sequences")
    if not isinstance(raw_sequences, list):
        raise ValueError("Boltz-2 request sequences must be a list")
    sequences: list[EntityInput] = []
    for raw_entry in raw_sequences:
        if not isinstance(raw_entry, dict) or len(raw_entry) != 1:
            raise ValueError("each Boltz-2 sequence entry requires exactly one entity type")
        raw_kind, polymer = next(iter(raw_entry.items()))
        if raw_kind == "ligand":
            if not isinstance(polymer, dict) or set(polymer) - {"id", "smiles", "ccd"}:
                raise ValueError("Boltz-2 ligand entries accept only id and one of smiles/ccd")
            chain_ids = _string_tuple(polymer.get("id"), "ligand id")
            smiles = polymer.get("smiles")
            raw_ccd = polymer.get("ccd")
            if smiles is not None and not isinstance(smiles, str):
                raise ValueError("Boltz-2 ligand smiles must be a string")
            ccd = _string_tuple(raw_ccd, "ligand ccd") if raw_ccd is not None else ()
            sequences.append(LigandInput(chain_ids, smiles, ccd))
            continue
        try:
            kind = PolymerKind(raw_kind)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Boltz-2 sequence entries must be protein, dna, rna, or ligand"
            ) from error
        if not isinstance(polymer, dict):
            raise ValueError(f"Boltz-2 {kind.value} entry must be a mapping")
        if any(not isinstance(key, str) for key in polymer):
            raise ValueError(f"Boltz-2 {kind.value} field names must be strings")
        allowed = {"id", "sequence", "cyclic", "modifications"}
        if kind is PolymerKind.PROTEIN:
            allowed.add("msa")
        unknown_polymer = set(polymer) - allowed
        if unknown_polymer:
            raise ValueError(
                f"unsupported Boltz-2 {kind.value} fields: " + ", ".join(sorted(unknown_polymer))
            )
        chain_ids = _string_tuple(polymer.get("id"), f"{kind.value} id")
        sequence = polymer.get("sequence")
        if not isinstance(sequence, str):
            raise ValueError(f"Boltz-2 {kind.value} sequence must be a string")
        cyclic = polymer.get("cyclic", False)
        if not isinstance(cyclic, bool):
            raise ValueError(f"Boltz-2 {kind.value} cyclic must be a boolean")
        raw_msa = polymer.get("msa")
        if kind is PolymerKind.PROTEIN:
            if not isinstance(raw_msa, str):
                raise ValueError("Boltz-2 protein msa must be an A3M/CSV path or 'empty'")
            msa_path = None if raw_msa == "empty" else PurePosixPath(raw_msa)
        else:
            msa_path = None
        sequences.append(
            SequenceInput(
                kind=kind,
                chain_ids=chain_ids,
                sequence=sequence,
                msa_path=msa_path,
                cyclic=cyclic,
                modifications=_modifications(polymer.get("modifications")),
            )
        )
    raw_templates = document.get("templates", [])
    if not isinstance(raw_templates, list):
        raise ValueError("Boltz-2 templates must be a list")
    templates = []
    for raw_template in raw_templates:
        if not isinstance(raw_template, dict):
            raise ValueError("Boltz-2 template entries must be mappings")
        unknown_template = set(raw_template) - {
            "cif",
            "pdb",
            "chain_id",
            "template_id",
            "force",
            "threshold",
        }
        paths = set(raw_template) & {"cif", "pdb"}
        if unknown_template or len(paths) != 1:
            raise ValueError(
                "Boltz-2 templates require one CIF/PDB path and supported mapping fields"
            )
        format_name = paths.pop()
        raw_path = raw_template[format_name]
        if not isinstance(raw_path, str):
            raise ValueError("Boltz-2 template path must be a string")
        force = raw_template.get("force", False)
        threshold = raw_template.get("threshold")
        if not isinstance(force, bool) or (
            threshold is not None
            and (not isinstance(threshold, (int, float)) or isinstance(threshold, bool))
        ):
            raise ValueError("Boltz-2 template force/threshold values are invalid")
        templates.append(
            TemplateInput(
                path=PurePosixPath(raw_path),
                format=format_name,
                chain_ids=(
                    _string_tuple(raw_template["chain_id"], "template chain_id")
                    if "chain_id" in raw_template
                    else None
                ),
                template_chain_ids=(
                    _string_tuple(raw_template["template_id"], "template template_id")
                    if "template_id" in raw_template
                    else None
                ),
                force=force,
                threshold=float(threshold) if threshold is not None else None,
            )
        )
    raw_properties = document.get("properties", [])
    if not isinstance(raw_properties, list):
        raise ValueError("Boltz-2 properties must be a list")
    affinity: AffinityProperty | None = None
    for raw_property in raw_properties:
        if (
            not isinstance(raw_property, dict)
            or set(raw_property) != {"affinity"}
            or not isinstance(raw_property["affinity"], dict)
            or set(raw_property["affinity"]) != {"binder"}
            or not isinstance(raw_property["affinity"]["binder"], str)
        ):
            raise ValueError("Boltz-2 properties support only affinity with one binder")
        if affinity is not None:
            raise ValueError("Boltz-2 supports exactly one affinity property per request")
        affinity = AffinityProperty(raw_property["affinity"]["binder"])
    request = Boltz2Request(
        sequences=tuple(sequences),
        templates=tuple(templates),
        affinity=affinity,
        constraints=_parse_constraints(document.get("constraints")),
    )
    validate_request(request)
    return request


def _validate_msa_rows(rows: list[str], expected_query: str | None) -> None:
    aligned_widths = set()
    for row in rows:
        for character in row:
            if character == "-" or character in _PROTEIN_ALPHABET:
                continue
            if "a" <= character <= "z":
                continue
            raise ValueError("Boltz-2 MSA sequences contain an unsupported residue symbol")
        aligned_widths.add(sum(not character.islower() for character in row))
    if len(aligned_widths) != 1:
        raise ValueError("Boltz-2 MSA rows must have one aligned width")
    if expected_query is not None:
        query = "".join(character for character in rows[0] if not character.islower())
        if query.replace("-", "").upper() != expected_query.upper():
            raise ValueError("MSA query row does not match the requested polymer sequence")
        if aligned_widths != {len(expected_query)}:
            raise ValueError("Boltz-2 MSA aligned width must match the requested sequence")


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
        if line.startswith("#"):
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
        if current:
            raise ValueError("Boltz-2 A3M records must use exactly one sequence line")
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

    _validate_msa_rows(rows, expected_query)
    return tuple(rows)


def validate_csv_msa(
    text: str,
    *,
    expected_query: str | None = None,
    profile: Boltz2QualificationProfile = INITIAL_BF16_PROFILE,
) -> tuple[str, ...]:
    """Validate Boltz paired/unpaired CSV MSA rows."""

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or tuple(sorted(reader.fieldnames)) != ("key", "sequence"):
        raise ValueError("Boltz-2 CSV MSA requires exactly key and sequence columns")
    rows: list[str] = []
    for line_number, row in enumerate(reader, start=2):
        if None in row:
            raise ValueError(f"Boltz-2 CSV MSA line {line_number} has extra columns")
        key = row.get("key")
        sequence = row.get("sequence")
        if sequence is None or not sequence.strip():
            raise ValueError(f"Boltz-2 CSV MSA sequence on line {line_number} is empty")
        sequence = sequence.strip()
        if key:
            normalized_key = key.strip()
            if not normalized_key.lstrip("-").isdigit() or not (
                -(2**31) <= int(normalized_key) < 2**31
            ):
                raise ValueError(
                    f"Boltz-2 CSV MSA key on line {line_number} must be an INT32 integer"
                )
        rows.append(sequence)
    if not profile.min_msa_depth <= len(rows) <= profile.max_msa_depth:
        raise ValueError(
            "Boltz-2 MSA depth is outside the qualified BF16 profile: "
            f"{len(rows)} not in [{profile.min_msa_depth}, {profile.max_msa_depth}]"
        )
    _validate_msa_rows(rows, expected_query)
    return tuple(rows)
