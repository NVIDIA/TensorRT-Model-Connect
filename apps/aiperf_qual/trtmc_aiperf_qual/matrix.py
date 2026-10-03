# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The execution matrix (DESIGN.md Section 7): one row per ready profile, generated from the configuration.

A row is ``executable`` when the profile has a native path (a generic adapter or its family's own) and an
accuracy scheme (gold benchmarks, whole-output checks, or Perf-only by declaration); the formal run starts
only when every row is.
"""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import ConfigError, Environment
from .models import resolve_model

COLUMNS = ("profile", "task", "operation", "bundle", "build_exception", "native_path", "native_model", "native_revision",
           "requirements", "prepare", "trust_remote_code", "accuracy_source", "benchmarks", "checks", "timed_request",
           "reference_modes", "executable", "problem")


def row(environment: Environment, profile: str) -> dict[str, Any]:
    try:
        model = resolve_model(profile, environment)
    except ConfigError as error:
        return {"profile": profile, "executable": False, "problem": f"configuration: {error}"}
    reference, l1 = model["reference"], model["performance"]["l1"]
    native = ("unsupported" if reference["backend"] == "unsupported"
              else reference.get("adapter") or f"generic {model['operation']}")
    problems = []
    if reference["backend"] == "unsupported":
        problems.append("no native adapter")
    if model["accuracy_source"] == "missing":
        problems.append("no accuracy scheme")
    return {"profile": profile, "task": model["task"], "operation": model["operation"],
            "bundle": model["candidate"]["bundle"], "build_exception": bool(model["candidate"].get("build")),
            "native_path": native, "native_model": reference.get("model") or model["candidate"]["checkpoint"],
            "native_revision": reference.get("revision") or "", "requirements": reference.get("requirements") or "",
            "prepare": reference.get("prepare") or "", "trust_remote_code": reference["trust_remote_code"],
            "accuracy_source": model["accuracy_source"],
            "benchmarks": " ".join(item["suite"] for item in model["absolute"]),
            "checks": " ".join(check["check"] for check in model["supplementary"]),
            "timed_request": l1["suite"].get("suite"), "reference_modes": " ".join(l1["reference_modes"]),
            "executable": not problems, "problem": "; ".join(problems)}


def write_matrix(environment: Environment, profiles: Sequence[str], output: Path | None) -> int:
    """Write the CSV; exit 0 only when every profile is executable."""
    rows = [row(environment, profile) for profile in profiles]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    if output:
        output.write_text(buffer.getvalue())
    else:
        sys.stdout.write(buffer.getvalue())
    blocked = [item["profile"] for item in rows if not item["executable"]]
    print(f"{len(rows) - len(blocked)} of {len(rows)} profiles executable" + (f"; blocked: {', '.join(blocked)}" if blocked else ""),
          file=sys.stderr)
    return 0 if not blocked else 1
