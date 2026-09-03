# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from tensorrt_model_connect.families.boltz2.contracts import (
    AtomReference,
    Boltz2Confidence,
    Boltz2QualificationProfile,
    Boltz2Request,
    BondConstraint,
    LigandInput,
    PolymerKind,
    ResidueModification,
    SequenceInput,
    StructureFormat,
    TemplateInput,
    parse_request_yaml,
    validate_a3m,
    validate_confidence,
    validate_request,
)
from tensorrt_model_connect.families.boltz2.provenance import PINNED_BOLTZ2


EXAMPLE_SEQUENCE = (
    "QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKAWKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG"
)


def _request(**kwargs) -> Boltz2Request:
    return Boltz2Request(
        sequences=(
            SequenceInput(
                PolymerKind.PROTEIN,
                ("A",),
                EXAMPLE_SEQUENCE,
                PurePosixPath("examples/msa/seq2.a3m"),
            ),
        ),
        **kwargs,
    )


def _request_with_sequence(**changes) -> Boltz2Request:
    request = _request()
    return replace(request, sequences=(replace(request.sequences[0], **changes),))


def test_pinned_provenance_is_immutable_and_complete() -> None:
    assert PINNED_BOLTZ2.source_tag == "v2.2.1"
    assert len(PINNED_BOLTZ2.source_revision) == 40
    assert len(PINNED_BOLTZ2.checkpoint_revision) == 40
    assert len(PINNED_BOLTZ2.structure_checkpoint.sha256) == 64
    assert len(PINNED_BOLTZ2.affinity_checkpoint.sha256) == 64
    assert len(PINNED_BOLTZ2.molecular_archive.sha256) == 64
    assert PINNED_BOLTZ2.source_license == "MIT"
    assert PINNED_BOLTZ2.checkpoint_license == "MIT"
    assert PINNED_BOLTZ2.structure_checkpoint.size_bytes == 2_286_561_469
    assert PINNED_BOLTZ2.reference_configuration.precision == "bf16"
    assert PINNED_BOLTZ2.qualification_request_path.endswith("protein_monomer.yaml")
    assert PINNED_BOLTZ2.reusable_profile_fixture.token_count == 117
    assert PINNED_BOLTZ2.reusable_profile_fixture.padded_atom_count == 928


def test_bounded_profile_declares_atom_window_envelope() -> None:
    profile = Boltz2QualificationProfile()
    assert profile.min_padded_atoms == 32
    assert profile.max_padded_atoms == 928
    assert profile.atom_window_queries == 32


def test_qualification_input_digests_match_provenance() -> None:
    root = Path(__file__).resolve().parents[5]
    request = root / PINNED_BOLTZ2.qualification_request_path
    msa = root / PINNED_BOLTZ2.qualification_msa_path
    assert hashlib.sha256(request.read_bytes()).hexdigest() == (
        PINNED_BOLTZ2.qualification_request_sha256
    )
    assert hashlib.sha256(msa.read_bytes()).hexdigest() == (
        PINNED_BOLTZ2.qualification_msa_sha256
    )
    validate_a3m(msa.read_text(encoding="utf-8"), expected_query=EXAMPLE_SEQUENCE)

    fixture = PINNED_BOLTZ2.reusable_profile_fixture
    request = root / fixture.request_path
    msa = root / fixture.msa_path
    assert hashlib.sha256(request.read_bytes()).hexdigest() == fixture.request_sha256
    assert hashlib.sha256(msa.read_bytes()).hexdigest() == fixture.msa_sha256
    parsed = parse_request_yaml(request.read_text(encoding="utf-8"))
    assert parsed.token_count == fixture.token_count
    validate_a3m(
        msa.read_text(encoding="utf-8"),
        expected_query=parsed.sequences[0].sequence,
    )


def test_request_contract_accepts_a_shorter_protein_before_shape_selection() -> None:
    request = _request_with_sequence(sequence=EXAMPLE_SEQUENCE[:73])
    validate_request(request)
    assert request.token_count == 73


def test_request_yaml_parser_accepts_only_executable_subset() -> None:
    request = parse_request_yaml(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        "      sequence: ACDE\n"
        "      msa: protein_monomer.a3m\n"
    )
    assert request.token_count == 4
    with pytest.raises(ValueError, match="unsupported Boltz-2 protein fields"):
        parse_request_yaml(
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            "      sequence: ACDE\n"
            "      msa: protein_monomer.a3m\n"
            "      cyclic: true\n"
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"recycling_steps": 2}, "recycling_steps=3"),
        ({"sampling_steps": 50}, "sampling_steps=200"),
        ({"diffusion_samples": 2}, "diffusion_samples=1"),
        ({"seed": -1}, "seed must be"),
    ],
)
def test_unqualified_sampling_profiles_fail_closed(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_request(_request(**kwargs))


def test_request_rejects_duplicate_chain_ids() -> None:
    request = Boltz2Request(
        sequences=(
            SequenceInput(PolymerKind.PROTEIN, ("A",), "ACDE", PurePosixPath("a.a3m")),
            SequenceInput(PolymerKind.PROTEIN, ("A",), "FGHI", PurePosixPath("b.a3m")),
        )
    )
    with pytest.raises(ValueError, match="duplicate.*A"):
        validate_request(request)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        (
            _request_with_sequence(kind=PolymerKind.DNA, sequence="ACGT"),
            "dna polymers",
        ),
        (
            _request_with_sequence(
                modifications=(ResidueModification(1, "MSE"),)
            ),
            "modifications",
        ),
        (
            _request(ligands=(LigandInput(("B",), smiles="CCO"),)),
            "ligands",
        ),
        (
            _request(templates=(TemplateInput(PurePosixPath("template.cif")),)),
            "templates",
        ),
        (
            _request(
                constraints=(
                    BondConstraint(
                        AtomReference("A", 1, "CA"),
                        AtomReference("A", 2, "N"),
                    ),
                ),
            ),
            "constraints",
        ),
        (
            _request(output_format=StructureFormat.PDB),
            "mmCIF",
        ),
    ],
)
def test_unqualified_input_features_fail_closed(case, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_request(case)


def test_request_requires_safe_relative_a3m_path() -> None:
    with pytest.raises(ValueError, match="custom A3M"):
        validate_request(_request_with_sequence(msa_path=None))
    with pytest.raises(ValueError, match="remain inside"):
        validate_request(_request_with_sequence(msa_path=PurePosixPath("../outside.a3m")))


def test_a3m_accepts_insertions_and_matches_query() -> None:
    rows = validate_a3m(
        ">query\nACdDE-F\n>hit\nAC-DEF\n",
        expected_query="ACDEF",
        profile=Boltz2QualificationProfile(opt_msa_depth=2, max_msa_depth=2),
    )
    assert rows == ("ACdDE-F", "AC-DEF")


@pytest.mark.parametrize(
    "text",
    ["ACDE", ">query\n", ">\nACDE", ">query\nACD*E"],
)
def test_a3m_rejects_malformed_documents(text) -> None:
    with pytest.raises(ValueError):
        validate_a3m(text)


def test_confidence_contract_checks_ranges_and_token_axis() -> None:
    valid = Boltz2Confidence(0.8, 0.7, 0.0, 0.0, 0.0, 0.8, 0.8, (0.7, 0.9))
    validate_confidence(valid, token_count=2)

    with pytest.raises(ValueError, match="confidence_score"):
        validate_confidence(
            Boltz2Confidence(1.1, 0.7, 0.0, 0.0, 0.0, 0.8, 0.8, (0.7, 0.9)),
            token_count=2,
        )
    with pytest.raises(ValueError, match="plddt length"):
        validate_confidence(valid, token_count=3)
