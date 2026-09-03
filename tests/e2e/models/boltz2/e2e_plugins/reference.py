# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned native-output snapshots for the qualified Boltz-2 profiles."""

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


_SNAPSHOTS = {
    "boltz2-protein-monomer": {
        "structure_sha256": "7ec8745ebc862233169e28feab901290e482b06bdb22664e212d36ac0f3acd26",
        "metadata_sha256": "5392752bbedd31ba73c26b609260cad4d9ceea96ce8aad12c1b3d2a9eae6a5a2",
        "atom_count": 899,
        "token_count": 117,
        "confidence_score": 0.59235680103302,
        "complex_plddt": 0.6044121980667114,
        "ptm": 0.5441350936889648,
    },
    "boltz2-protein-monomer-variant-reuse": {
        "structure_sha256": "4bbe4b74cdd22d23f274b281db790e47c91b7ef0775f2f80ccd20cc5a8157335",
        "metadata_sha256": "6166e734092d4bb80be6c2d6973d0adb09b6f5a7978ad7c340c2e670b3e6de25",
        "atom_count": 899,
        "token_count": 117,
        "confidence_score": 0.6266596913337708,
        "complex_plddt": 0.6349093914031982,
        "ptm": 0.593660831451416,
    },
}


class Boltz2SnapshotReference:
    @property
    def backend_name(self) -> str:
        return "boltz2_snapshot"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        del ctx
        snapshot = _SNAPSHOTS.get(case.name)
        if snapshot is None:
            raise RuntimeError(f"Boltz-2 has no qualified snapshot for {case.name!r}")
        return StageOutput(
            stage_name=stage.name,
            data=dict(snapshot),
            metadata={"source": "checked_in_snapshot"},
        )


reference = Boltz2SnapshotReference()
