# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structure comparison for the ESMFold2 tests: readers of the native outputs and the parity metrics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def read_mmcif_atoms(path: Path) -> tuple[list[tuple[str, str, int]], np.ndarray, np.ndarray]:
    """``(atom ids, coordinates [n, 3], B-factors [n])`` of the ``atom_site`` loop the runtime writes."""

    lines = path.read_text(encoding="utf-8").splitlines()
    fields: list[str] = []
    rows: list[list[str]] = []
    index = 0
    while index < len(lines):
        if lines[index] == "loop_" and index + 1 < len(lines) and lines[index + 1].startswith("_atom_site."):
            index += 1
            while index < len(lines) and lines[index].startswith("_atom_site."):
                fields.append(lines[index][len("_atom_site."):])
                index += 1
            while index < len(lines) and lines[index] not in ("#", "loop_") and not lines[index].startswith("_"):
                values = lines[index].split()
                if len(values) != len(fields):
                    raise AssertionError(f"atom_site row has {len(values)} values for {len(fields)} fields")
                rows.append(values)
                index += 1
            break
        index += 1
    if not rows:
        raise AssertionError(f"{path} has no atom_site rows")
    column = {name: position for position, name in enumerate(fields)}
    for required in ("label_atom_id", "label_comp_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
                     "B_iso_or_equiv"):
        if required not in column:
            raise AssertionError(f"atom_site lacks {required}")
    ids = [(row[column["label_atom_id"]], row[column["label_comp_id"]], int(row[column["label_seq_id"]]))
           for row in rows]
    coords = np.asarray([[float(row[column[axis]]) for axis in ("Cartn_x", "Cartn_y", "Cartn_z")] for row in rows])
    bfactors = np.asarray([float(row[column["B_iso_or_equiv"]]) for row in rows])
    return ids, coords, bfactors


def kabsch_rmsd(moving: np.ndarray, fixed: np.ndarray) -> float:
    """RMSD after the optimal proper superposition (float64)."""

    moving = np.asarray(moving, dtype=np.float64)
    fixed = np.asarray(fixed, dtype=np.float64)
    if moving.shape != fixed.shape or moving.ndim != 2 or moving.shape[1] != 3 or len(moving) < 3:
        raise ValueError("kabsch_rmsd needs two [n >= 3, 3] point sets")
    a = moving - moving.mean(axis=0)
    b = fixed - fixed.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((a @ rotation - b) ** 2, axis=1))))


def rmsd(first: np.ndarray, second: np.ndarray) -> float:
    """RMSD without superposition (the frames agree when both folds draw the same noise)."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("rmsd needs two point sets of one shape")
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=-1))))


def lddt(predicted: np.ndarray, reference: np.ndarray, cutoff: float = 15.0) -> float:
    """Global lDDT of ``predicted`` against ``reference`` over the reference's pairs within ``cutoff``."""

    predicted = np.asarray(predicted, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if predicted.shape != reference.shape:
        raise ValueError("lddt needs two point sets of one shape")
    true_distance = np.linalg.norm(reference[:, None] - reference[None], axis=-1)
    predicted_distance = np.linalg.norm(predicted[:, None] - predicted[None], axis=-1)
    pairs = (true_distance < cutoff) & ~np.eye(len(reference), dtype=bool)
    if not pairs.any():
        raise ValueError("lddt found no pair within the cutoff")
    error = np.abs(true_distance - predicted_distance)[pairs]
    return float(np.mean([(error < threshold).mean() for threshold in (0.5, 1.0, 2.0, 4.0)]))


def finite(value: float) -> bool:
    return math.isfinite(value)
