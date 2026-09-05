# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare native OpenFold3 structure/confidence output with the pinned reference."""

from __future__ import annotations

import argparse
import json
import math
import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Atom:
    """One atom-site row required for qualification."""

    chain: str
    residue: str
    name: str
    coordinates: np.ndarray

    @property
    def key(self) -> tuple[str, str, str]:
        return self.chain, self.residue, self.name


def _column(headers: list[str], *names: str) -> int:
    for name in names:
        if name in headers:
            return headers.index(name)
    raise ValueError(f"mmCIF atom loop is missing one of: {', '.join(names)}")


def read_atom_site(path: Path) -> list[Atom]:
    """Read the atom-site loop emitted by OpenFold3 or the native runtime."""

    lines = path.read_text(encoding="utf-8").splitlines()
    for loop_index, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        headers: list[str] = []
        cursor = loop_index + 1
        while cursor < len(lines) and lines[cursor].lstrip().startswith("_atom_site."):
            headers.append(lines[cursor].strip())
            cursor += 1
        if not headers:
            continue
        chain = _column(headers, "_atom_site.label_asym_id", "_atom_site.auth_asym_id")
        residue = _column(headers, "_atom_site.label_seq_id", "_atom_site.auth_seq_id")
        atom_name = _column(headers, "_atom_site.label_atom_id", "_atom_site.auth_atom_id")
        x = _column(headers, "_atom_site.Cartn_x")
        y = _column(headers, "_atom_site.Cartn_y")
        z = _column(headers, "_atom_site.Cartn_z")
        atoms: list[Atom] = []
        while cursor < len(lines):
            row = lines[cursor].strip()
            cursor += 1
            if not row or row.startswith("#") or row == "loop_" or row.startswith("_"):
                break
            fields = shlex.split(row)
            if len(fields) != len(headers):
                raise ValueError(f"unsupported multiline mmCIF atom row in {path}")
            atoms.append(
                Atom(
                    fields[chain],
                    fields[residue],
                    fields[atom_name],
                    np.asarray([fields[x], fields[y], fields[z]], dtype=np.float64),
                )
            )
        if atoms:
            return atoms
    raise ValueError(f"no atom-site loop found in {path}")


def _matched_coordinates(
    candidate: list[Atom], reference: list[Atom], *, atom_name: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    reference_by_key = {atom.key: atom.coordinates for atom in reference}
    candidate_points: list[np.ndarray] = []
    reference_points: list[np.ndarray] = []
    for atom in candidate:
        if atom_name is not None and atom.name != atom_name:
            continue
        point = reference_by_key.get(atom.key)
        if point is not None:
            candidate_points.append(atom.coordinates)
            reference_points.append(point)
    if len(candidate_points) < 3:
        raise ValueError("fewer than three corresponding atoms are available")
    return np.stack(candidate_points), np.stack(reference_points)


def aligned_rmsd(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Return rigid-body-aligned RMSD in Angstroms."""

    candidate = candidate - candidate.mean(axis=0, keepdims=True)
    reference = reference - reference.mean(axis=0, keepdims=True)
    left, _, right_t = np.linalg.svd(candidate.T @ reference)
    if np.linalg.det(left @ right_t) < 0:
        left[:, -1] *= -1
    aligned = candidate @ (left @ right_t)
    return float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=-1))))


def distance_mae(candidate: np.ndarray, reference: np.ndarray, cutoff: float = 15.0) -> float:
    """Return pair-distance MAE for reference pairs below *cutoff*."""

    candidate_distance = np.linalg.norm(candidate[:, None] - candidate[None, :], axis=-1)
    reference_distance = np.linalg.norm(reference[:, None] - reference[None, :], axis=-1)
    mask = np.triu(reference_distance < cutoff, k=1)
    if not np.any(mask):
        raise ValueError("no atom pairs fall within the qualification cutoff")
    return float(np.mean(np.abs(candidate_distance[mask] - reference_distance[mask])))


def _finite_array(document: dict, name: str) -> np.ndarray:
    values = np.asarray(document[name], dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"confidence field {name!r} is empty or non-finite")
    return values


def confidence_metrics(
    native: dict, reference_full: dict, reference_aggregated: dict
) -> dict[str, float]:
    """Calculate aligned scalar and array confidence differences."""

    result: dict[str, float] = {}
    for name in ("plddt", "pde", "pae"):
        candidate = _finite_array(native, name)
        reference = _finite_array(reference_full, name)
        if candidate.size != reference.size:
            raise ValueError(f"confidence field {name!r} extent differs")
        candidate = candidate.ravel()
        reference = reference.ravel()
        result[f"{name}_mae"] = float(np.mean(np.abs(candidate - reference)))
        result[f"{name}_max_abs"] = float(np.max(np.abs(candidate - reference)))
        result[f"{name}_pearson"] = float(np.corrcoef(candidate.ravel(), reference.ravel())[0, 1])
    result["average_plddt_abs"] = abs(
        float(native["average_plddt"]) - float(reference_aggregated["avg_plddt"])
    )
    result["gpde_abs"] = abs(float(native["gpde"]) - float(reference_aggregated["gpde"]))
    result["ptm_abs"] = abs(float(native["ptm"]) - float(reference_aggregated["ptm"]))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("confidence comparison produced a non-finite metric")
    return result


def _metric(document: dict, dotted_name: str) -> float:
    value: object = document
    for component in dotted_name.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"qualification metric {dotted_name!r} is missing")
        value = value[component]
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"qualification metric {dotted_name!r} is not finite")
    return float(value)


def apply_thresholds(result: dict, thresholds: dict) -> list[str]:
    """Return human-readable failures for a closed min/max threshold document."""

    if set(thresholds) != {"schema_version", "minimum", "maximum"}:
        raise ValueError("qualification thresholds require schema_version/minimum/maximum")
    failures: list[str] = []
    for name, limit in thresholds["maximum"].items():
        value = _metric(result, name)
        if value > float(limit):
            failures.append(f"{name}={value} exceeds {limit}")
    for name, limit in thresholds["minimum"].items():
        value = _metric(result, name)
        if value < float(limit):
            failures.append(f"{name}={value} is below {limit}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-structure", type=Path, required=True)
    parser.add_argument("--native-confidence", type=Path, required=True)
    parser.add_argument("--reference-structure", type=Path, required=True)
    parser.add_argument("--reference-confidence", type=Path, required=True)
    parser.add_argument("--reference-aggregated", type=Path, required=True)
    parser.add_argument("--experimental-structure", type=Path)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    native_atoms = read_atom_site(arguments.native_structure)
    reference_atoms = read_atom_site(arguments.reference_structure)
    native, reference = _matched_coordinates(native_atoms, reference_atoms)
    native_ca, reference_ca = _matched_coordinates(native_atoms, reference_atoms, atom_name="CA")
    result: dict[str, object] = {
        "schema_version": 1,
        "native_atom_count": len(native_atoms),
        "reference_atom_count": len(reference_atoms),
        "matched_atom_count": len(native),
        "structural_parity": {
            "all_atom_aligned_rmsd_angstrom": aligned_rmsd(native, reference),
            "ca_aligned_rmsd_angstrom": aligned_rmsd(native_ca, reference_ca),
            "all_atom_pair_distance_mae_angstrom": distance_mae(native, reference),
        },
        "confidence_parity": confidence_metrics(
            json.loads(arguments.native_confidence.read_text(encoding="utf-8")),
            json.loads(arguments.reference_confidence.read_text(encoding="utf-8")),
            json.loads(arguments.reference_aggregated.read_text(encoding="utf-8")),
        ),
    }
    if arguments.experimental_structure:
        experimental_atoms = read_atom_site(arguments.experimental_structure)
        predicted_ca, experimental_ca = _matched_coordinates(
            native_atoms, experimental_atoms, atom_name="CA"
        )
        result["experimental_accuracy"] = {
            "matched_ca_count": len(predicted_ca),
            "ca_aligned_rmsd_angstrom": aligned_rmsd(predicted_ca, experimental_ca),
        }
    thresholds = json.loads(arguments.thresholds.read_text(encoding="utf-8"))
    failures = apply_thresholds(result, thresholds)
    result["thresholds"] = {
        "path": str(arguments.thresholds),
        "passed": not failures,
        "failures": failures,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
