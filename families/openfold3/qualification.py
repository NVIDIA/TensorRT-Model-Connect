# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned structure matching for OpenFold3 qualification."""

from __future__ import annotations

from dataclasses import dataclass

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


def matched_coordinates(
    candidate: list[Atom], reference: list[Atom], *, atom_name: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return corresponding coordinates after rejecting ambiguous atom keys."""

    reference_by_key: dict[tuple[str, str, str], np.ndarray] = {}
    for atom in reference:
        if atom.key in reference_by_key:
            raise ValueError(f"duplicate reference atom key: {atom.key}")
        reference_by_key[atom.key] = atom.coordinates
    candidate_points: list[np.ndarray] = []
    reference_points: list[np.ndarray] = []
    candidate_keys: set[tuple[str, str, str]] = set()
    for atom in candidate:
        if atom_name is not None and atom.name != atom_name:
            continue
        if atom.key in candidate_keys:
            raise ValueError(f"duplicate candidate atom key: {atom.key}")
        candidate_keys.add(atom.key)
        point = reference_by_key.get(atom.key)
        if point is not None:
            candidate_points.append(atom.coordinates)
            reference_points.append(point)
    if len(candidate_points) < 3:
        raise ValueError("fewer than three corresponding atoms are available")
    return np.stack(candidate_points), np.stack(reference_points)
