# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the native Boltz-2 structure-prediction CLI contract."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import time

from tests.e2e_harness.contracts import E2ECase, RunContext, StageOutput, StageSpec


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_json_line(output: str) -> dict:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("Boltz-2 native runner did not emit a JSON summary")


def _inspect_mmcif(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    coordinates: list[float] = []
    b_factors: list[float] = []
    atom_count = 0
    for line in text.splitlines():
        if not line.startswith("ATOM "):
            continue
        fields = line.split()
        if len(fields) != 13:
            raise RuntimeError("Boltz-2 mmCIF atom row differs from its native contract")
        coordinates.extend(float(value) for value in fields[7:10])
        b_factors.append(float(fields[11]))
        atom_count += 1
    return {
        "mmcif_header_valid": text.startswith("data_boltz2\n#\nloop_\n"),
        "atom_count": atom_count,
        "coordinates_finite": bool(coordinates) and all(math.isfinite(v) for v in coordinates),
        "b_factors_finite": bool(b_factors) and all(math.isfinite(v) for v in b_factors),
    }


class Boltz2StructureRunner:
    @property
    def strategy_name(self) -> str:
        return "structure_prediction"

    def run_stage(self, case: E2ECase, stage: StageSpec, ctx: RunContext) -> StageOutput:
        request = Path(str(case.inputs.get("request", "")))
        if not request.is_file():
            raise RuntimeError(f"Boltz-2 request does not exist: {request}")
        output_dir = Path(ctx.artifacts_dir) / case.name
        output_dir.mkdir(parents=True, exist_ok=True)
        structure = output_dir / "prediction.cif"
        metadata = output_dir / "prediction.metadata.json"
        bundle = Path(ctx.engine_dir) / case.bundle
        command = [
            ctx.binary_path,
            "predict-structure",
            str(bundle),
            "--input",
            str(request),
            "--output",
            str(structure),
            "--output-json",
            str(metadata),
            "--seed",
            str(case.inputs.get("seed", 42)),
            "--backend-dir",
            str(Path(ctx.binary_path).resolve().parent),
        ]
        if ctx.model_plugin_dir:
            command.extend(["--model-plugin-dir", ctx.model_plugin_dir])
        environment = os.environ.copy()
        if ctx.ld_library_path:
            environment["LD_LIBRARY_PATH"] = ctx.ld_library_path
        started = time.monotonic()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=int(case.metadata.get("runtime_timeout_s", 1800)),
            env=environment,
        )
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            raise RuntimeError(
                "Boltz-2 native prediction failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        summary = _last_json_line(completed.stdout)
        family_metadata = json.loads(metadata.read_text(encoding="utf-8"))
        data = {
            **_inspect_mmcif(structure),
            "structure_path": str(structure),
            "metadata_path": str(metadata),
            "structure_sha256": _sha256(structure),
            "metadata_sha256": _sha256(metadata),
            "token_count": len(family_metadata.get("plddt", [])),
            "request_sha256": family_metadata.get("request_sha256", ""),
            "confidence_score": summary.get("confidence_score"),
            "complex_plddt": summary.get("complex_plddt"),
            "ptm": summary.get("ptm"),
        }
        return StageOutput(
            stage_name=stage.name,
            data=data,
            timing_s=elapsed,
            metadata={"command": command, "stderr": completed.stderr[-2000:]},
        )


runner = Boltz2StructureRunner()
