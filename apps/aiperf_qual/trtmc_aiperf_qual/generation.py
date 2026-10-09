# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate a suite's outputs on TRTMC and on the native model, keeping the output files.

Checks that judge generated media as a whole (``tts_intelligibility``, ``geneval``, ``edit_similarity``,
``replay_parity``) read the files each request wrote under ``scratch/<request id>``.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping

from .aiperf_runner import AiperfRun
from .config import Environment
from .services import serving
from .suites import request_sha

# (request directory, the server's record of the request) per suite sample, in suite order.
Outputs = list[tuple[Path, dict[str, Any]]]


def is_video(request: Mapping[str, Any]) -> bool:
    return request.get("media_type") == "video" or int(request.get("num_frames") or 1) > 1


def generate(environment: Environment, model: dict[str, Any], backend: str, out: Path, suite: Any,
             python: str | None = None, precision: str | None = None, reuse: bool = False) -> Outputs:
    """``reuse``: return an earlier generation in ``out`` when it sent exactly these requests and its
    outputs are still there (rechecking finished results)."""
    from .runner import _observations

    if reuse and (earlier := _earlier(out, suite)) is not None:
        return earlier
    kwargs = {"precision": precision, "python": python} if backend != "trtmc" else {}
    with serving(environment, model, backend, out, keep_artifacts=True, **kwargs) as service:
        _observations(environment, service, model, suite, out / "aiperf")
    outputs = _answered(out, suite)
    if outputs is None:
        raise RuntimeError(f"{backend} did not answer all {len(suite.samples)} requests")
    return outputs


def _answered(out: Path, suite: Any) -> Outputs | None:
    # Warmup advances AIPerf's dataset cursor. Join profiling requests to server
    # artifacts, then put them in suite order; arrival order is not label order.
    path = out / "records.jsonl"
    records = [json.loads(line) for line in path.read_text().split("\n") if line.strip()] if path.is_file() else []
    records = [record for record in records if record.get("route", "").startswith("/v1/tasks/")]
    by_id = {record["request_id"]: record for record in records}
    raw = AiperfRun(out / "aiperf", 0, []).raw_records()
    if len(by_id) != len(records) or len(raw) != len(suite.samples):
        return None
    by_request = defaultdict(deque)
    seen = set()
    for row in raw:
        identifier = row["metadata"].get("x_request_id")
        if identifier not in by_id or identifier in seen or row.get("status") != 200 or row.get("error"):
            return None
        seen.add(identifier)
        by_request[request_sha(row["payload"]["request"])].append(by_id[identifier])
    ordered = []
    for sample in suite.samples:
        available = by_request[request_sha(sample["request"])]
        if not available:
            return None
        record = available.popleft()
        ordered.append((out / "scratch" / str(record["request_id"]), record))
    return ordered


def _earlier(out: Path, suite: Any) -> Outputs | None:
    inputs = out / "aiperf.inputs.jsonl"
    expected = "".join(json.dumps({"text": json.dumps({"request": sample["request"]})}) + "\n" for sample in suite.samples)
    if not inputs.is_file() or inputs.read_text() != expected:
        return None
    outputs = _answered(out, suite)
    return outputs if outputs is not None and all(workdir.is_dir() for workdir, _ in outputs) else None


def generate_native(environment: Environment, model: dict[str, Any], suite: Any, python: str, out: Path,
                    label: str, skip: tuple[str, str] | None = None, reuse: bool = False) -> tuple[Outputs, str, str]:
    """The native model's outputs and the (backend, precision) that produced them: the reference adapter
    at the Perf precisions in order (``skip`` excluded)."""
    from .runner import timing_precisions

    errors = []
    for precision in timing_precisions(model["reference"]):
        if ("reference", precision) == skip:
            continue
        try:
            return (generate(environment, model, "reference", out / f"{label}-native-reference-{precision}", suite,
                             python, precision, reuse), "reference", precision)
        except Exception as error:  # noqa: BLE001 - try the next precision
            errors.append(f"{precision}: {type(error).__name__}: {str(error)[-200:]}")
    raise RuntimeError("; ".join(errors)[-1500:] or "no other native precision")


def media_source(workdir: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Where a request's generated frames are: its request directory, plus the frame files the
    observation lists and their frame indices (a server that keeps only sampled frames)."""
    observation = record.get("observation") or {}
    files = [str(path) for path in (observation.get("frame_artifacts") or observation.get("image_artifacts") or [])
             if isinstance(path, str)]
    source: dict[str, Any] = {"dir": str(workdir)}
    if files:
        source.update(files=files, indices=observation.get("artifact_indices"))
    return source
