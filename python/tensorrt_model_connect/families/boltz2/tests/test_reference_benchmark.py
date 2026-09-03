# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from tensorrt_model_connect.families.boltz2 import reference_benchmark
from tensorrt_model_connect.families.boltz2.reference_benchmark import (
    _kabsch_rmsd,
    _lddt,
    _masked_coords,
    _native_mmcif_coords,
    _native_qualification,
    _qualification_fixture_name,
    compare_native,
)
from tensorrt_model_connect.families.boltz2.provenance import PINNED_BOLTZ2


def _qualification_metrics(**overrides) -> dict:
    metrics = {
        "all_outputs_finite": True,
        "atom_count": 899,
        "token_count": 117,
        "lddt": 0.55,
        "kabsch_rmsd_angstrom": 9.0,
        "plddt_mean_abs": 0.10,
        "confidence_score_abs": 0.10,
        "complex_plddt_abs": 0.10,
        "ptm_abs": 0.10,
    }
    metrics.update(overrides)
    return metrics


def test_seeded_batch_pins_rng_before_preprocessing(monkeypatch) -> None:
    calls = []
    expected = object()
    monkeypatch.setattr(reference_benchmark, "_seed", lambda: calls.append("seed"))
    monkeypatch.setattr(
        reference_benchmark,
        "_load_batch",
        lambda processed, mols: calls.append((processed, mols)) or expected,
    )

    result = reference_benchmark._load_seeded_batch(
        Path("processed"), Path("molecules")
    )

    assert result is expected
    assert calls == ["seed", (Path("processed"), Path("molecules"))]


def test_reference_accepts_only_pinned_request_msa_pairs() -> None:
    assert _qualification_fixture_name(
        PINNED_BOLTZ2.qualification_request_sha256,
        PINNED_BOLTZ2.qualification_msa_sha256,
    ) == "protein_monomer"
    fixture = PINNED_BOLTZ2.reusable_profile_fixture
    assert _qualification_fixture_name(
        fixture.request_sha256, fixture.msa_sha256
    ) == fixture.name
    with pytest.raises(RuntimeError, match="digest pair mismatch"):
        _qualification_fixture_name(
            PINNED_BOLTZ2.qualification_request_sha256,
            fixture.msa_sha256,
        )


def test_kabsch_rmsd_removes_rigid_transform() -> None:
    reference = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    candidate = reference @ rotation + np.asarray([4.0, -3.0, 2.0])

    assert _kabsch_rmsd(reference, candidate) == pytest.approx(0.0, abs=1e-12)


def test_lddt_is_one_for_rigid_transform() -> None:
    reference = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    candidate = reference + np.asarray([10.0, -7.0, 4.0])

    assert _lddt(reference, candidate) == pytest.approx(1.0)


def test_masked_coords_rejects_non_finite_values(tmp_path) -> None:
    archive_path = tmp_path / "prediction.npz"
    np.savez(
        archive_path,
        coords=np.asarray([[[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0]]]),
        atom_mask=np.asarray([[True, True]]),
    )

    with np.load(archive_path) as archive:
        with pytest.raises(ValueError, match="non-finite"):
            _masked_coords(archive)


def test_native_mmcif_coords_reads_family_rows(tmp_path) -> None:
    path = tmp_path / "prediction.cif"
    path.write_text(
        "data_boltz2\n#\nloop_\n"
        "ATOM 1 C CA ALA A 1 1.0 2.0 3.0 1.00 75.0 1\n"
        "ATOM 2 O O ALA A 1 -1.0 -2.0 -3.0 1.00 75.0 1\n",
        encoding="utf-8",
    )

    assert _native_mmcif_coords(path) == pytest.approx(
        np.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    )


def test_native_qualification_uses_frozen_fixed_profile_gate() -> None:
    result = _native_qualification(_qualification_metrics())

    assert result["passed"]
    assert all(result["checks"].values())

    result = _native_qualification(_qualification_metrics(lddt=0.549))
    assert not result["passed"]
    assert not result["checks"]["lddt"]


def test_compare_native_emits_structural_and_confidence_metrics(tmp_path) -> None:
    reference = tmp_path / "reference.npz"
    coords = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]
    )
    np.savez(
        reference,
        coords=coords,
        atom_mask=np.ones((1, 4), dtype=bool),
        plddt=np.asarray([0.5, 0.6, 0.7, 0.8]),
        confidence_score=np.asarray([0.7]),
        complex_plddt=np.asarray([0.65]),
        ptm=np.asarray([0.75]),
    )
    candidate = tmp_path / "candidate.cif"
    rows = [
        f"ATOM {index} C CA ALA A {index} {x} {y} {z} 1.00 70.0 1"
        for index, (x, y, z) in enumerate(coords[0], start=1)
    ]
    candidate.write_text("data_boltz2\n#\nloop_\n" + "\n".join(rows) + "\n")
    metadata = tmp_path / "candidate.metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "plddt": [0.5, 0.6, 0.7, 0.8],
                "confidence_score": 0.7,
                "complex_plddt": 0.65,
                "ptm": 0.75,
            }
        )
    )
    output = tmp_path / "comparison.json"

    compare_native(
        argparse.Namespace(
            reference_npz=reference,
            candidate_mmcif=candidate,
            candidate_metadata=metadata,
            output_json=output,
        )
    )

    result = json.loads(output.read_text())
    assert result["all_outputs_finite"]
    assert result["lddt"] == pytest.approx(1.0)
    assert result["kabsch_rmsd_angstrom"] == pytest.approx(0.0, abs=1e-12)
    assert result["confidence_score_abs"] == pytest.approx(0.0)
