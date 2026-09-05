# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute a caller-built OpenFold3 bundle through the native CLI."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atom_rows(cif: str) -> list[list[str]]:
    return [line.split() for line in cif.splitlines() if line.startswith(("ATOM ", "HETATM "))]


class OpenFold3StructureRunner:
    @property
    def strategy_name(self) -> str:
        return "structure_prediction"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        if stage.name != "native_ubiquitin_structure":
            raise RuntimeError(f"unsupported OpenFold3 E2E stage: {stage.name}")
        if not ctx.binary_path:
            raise RuntimeError("OpenFold3 E2E requires the native trtmc binary")
        bundle = Path(ctx.engine_dir) / case.bundle
        query = Path(str(case.inputs["query"]))
        if not bundle.is_file() or bundle.is_symlink() or not query.is_file() or query.is_symlink():
            raise RuntimeError("OpenFold3 E2E bundle and query must be regular files")
        if not ctx.artifacts_dir:
            raise RuntimeError("OpenFold3 E2E artifact directory is required")
        output_dir = Path(ctx.artifacts_dir) / case.name
        output_dir.mkdir(parents=True, exist_ok=True)
        bundle_digest = _sha256(bundle)
        receipts = []
        for index in range(2):
            structure = output_dir / f"prediction-{index}.cif"
            metadata = output_dir / f"prediction-{index}.json"
            completed = subprocess.run(
                [
                    ctx.binary_path,
                    "predict-structure",
                    str(bundle),
                    "--input",
                    str(query),
                    "--output",
                    str(structure),
                    "--output-json",
                    str(metadata),
                ],
                capture_output=True,
                text=True,
                timeout=int(case.inputs.get("runtime_timeout_s", 1800)),
            )
            if completed.returncode:
                raise RuntimeError(f"OpenFold3 native execution failed: {completed.stderr}")
            receipts.append((structure.read_text("utf-8"), json.loads(metadata.read_text("utf-8"))))
        cif, confidence = receipts[0]
        atoms = _atom_rows(cif)
        coordinates = [float(value) for row in atoms for value in row[10:13]]
        plddt = confidence.get("plddt", [])
        pae = confidence.get("pae", [])
        pde = confidence.get("pde", [])
        finite_confidence = all(
            math.isfinite(float(value)) for values in (plddt, pae, pde) for value in values
        )
        valid_confidence_ranges = (
            finite_confidence
            and all(0.0 <= float(value) <= 100.0 for value in plddt)
            and all(0.0 <= float(value) <= 32.0 for value in (*pae, *pde))
            and 0.0 <= float(confidence.get("average_plddt", -1.0)) <= 100.0
            and 0.0 <= float(confidence.get("gpde", -1.0)) <= 32.0
            and 0.0 <= float(confidence.get("ptm", -1.0)) <= 1.0
        )
        return StageOutput(
            stage_name=stage.name,
            data={
                "schema_version": 1,
                "bundle_sha256": bundle_digest,
                "structure_path": str(output_dir / "prediction-0.cif"),
                "runtime_invariants": {
                    "repeat_exact": receipts[0] == receipts[1],
                    "bundle_unchanged": _sha256(bundle) == bundle_digest,
                    "precision": confidence.get("precision"),
                    "token_count": confidence.get("token_count"),
                    "atom_count": confidence.get("atom_count"),
                    "atom_rows": len(atoms),
                    "plddt_count": len(plddt),
                    "pae_count": len(pae),
                    "pde_count": len(pde),
                    "finite_confidence": finite_confidence,
                    "valid_confidence_ranges": valid_confidence_ranges,
                    "finite_coordinates": bool(coordinates)
                    and all(math.isfinite(value) for value in coordinates),
                    "coordinate_extent": (
                        max(coordinates) - min(coordinates) if coordinates else 0.0
                    ),
                    "mmcif_auth_atom_id": "_atom_site.auth_atom_id" in cif,
                    "mmcif_label_entity_id": "_atom_site.label_entity_id" in cif,
                    "sample_rank": confidence.get("sample_rank"),
                    "ranking_not_applicable": confidence.get("sample_ranking_score") is None
                    and confidence.get("sample_ranking_score_applicable") is False,
                },
            },
        )


runner = OpenFold3StructureRunner()
