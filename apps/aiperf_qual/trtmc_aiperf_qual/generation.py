# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a suite's outputs on TRTMC and on the native model, keeping the output files.

Checks that judge generated media as a whole (``tts_intelligibility``, ``clip_alignment``) read the
files each request wrote under ``scratch/<request id>``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import Environment
from .services import serving

# (request directory, the server's record of the request) per suite sample, in suite order.
Outputs = list[tuple[Path, dict[str, Any]]]


def generate(environment: Environment, model: dict[str, Any], backend: str, out: Path, suite: Any,
             python: str | None = None, precision: str | None = None) -> Outputs:
    from .runner import _observations

    kwargs = {"precision": precision, "python": python} if backend != "trtmc" else {}
    with serving(environment, model, backend, out, keep_artifacts=True, **kwargs) as service:
        _observations(environment, service, model, suite, out / "aiperf")
    # AIPerf sends the samples in order, one at a time: the last records are the suite's.
    records = [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines() if line.strip()]
    ordered = [record for record in records if record.get("route", "").startswith("/v1/tasks/")]
    if len(ordered) < len(suite.samples):
        raise RuntimeError(f"{backend} answered {len(ordered)} of {len(suite.samples)} requests")
    return [(out / "scratch" / str(record["request_id"]), record) for record in ordered[-len(suite.samples):]]


def generate_native(environment: Environment, model: dict[str, Any], suite: Any, python: str, out: Path,
                    label: str) -> tuple[Outputs, str]:
    """The native model's outputs and which reference produced them: the generic adapter, else the
    family's declared reference, at the Perf precisions in order."""
    from .runner import timing_precisions

    reference, errors = model["reference"], []
    for backend in dict.fromkeys([reference["backend"], reference.get("fallback") or reference["backend"]]):
        for precision in timing_precisions(reference):
            try:
                return (generate(environment, model, backend, out / f"{label}-native-{backend}-{precision}", suite,
                                 python, precision), f"{backend} {precision}")
            except Exception as error:  # noqa: BLE001 - try the next precision, then the fallback
                errors.append(f"{backend} {precision}: {type(error).__name__}: {str(error)[-200:]}")
    raise RuntimeError("; ".join(errors)[-1500:])
