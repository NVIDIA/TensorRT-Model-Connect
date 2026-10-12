# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute one internal Performance Qualification Target."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from . import matrix


def run_case(
    spec: Mapping[str, Any],
    environment: matrix.Environment,
    output: Path,
    *,
    no_build: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Run one case through the same execution contract as a campaign matrix."""

    resolved = matrix.preflight([dict(spec)], environment, require_runtime=True)
    if len(resolved) != 1:
        raise matrix.PerfMatrixError("single-case qualification resolved an unexpected case count")
    output.mkdir(parents=True, exist_ok=True)
    entry = resolved[0]
    try:
        result = matrix.execute_entry(
            entry,
            environment,
            output,
            no_build=no_build,
            verbose=verbose,
            attempt=1,
        )
    except (OSError, matrix.PerfMatrixError, subprocess.SubprocessError) as error:
        result = {
            "id": str(entry.spec["id"]),
            "model": entry.model.name,
            "family": entry.model.family,
            "operation": entry.spec["operation"],
            "testcase": entry.case.testcase_name,
            "status": "white",
            "attempts": 1,
            "error": str(error),
        }
    receipt = {
        "schema_version": matrix.RESULT_SCHEMA,
        "status": (
            "completed" if result.get("status") in matrix.TERMINAL_COMPARISONS else "failed"
        ),
        "suite": str(spec["id"]),
        "environment": environment.name,
        "selected_entry_ids": [str(spec["id"])],
        "rows": [result],
    }
    (output / "results.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    matrix.write_report(output, receipt)
    return result
