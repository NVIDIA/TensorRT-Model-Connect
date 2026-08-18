# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort structured timing helpers for bundle builds."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Iterator


_OUTPUT_PATH_KEY = "_output_path"


def new_build_timing(output_path: str | Path | None = None) -> dict:
    timing = {
        "schema_version": 1,
        "phases": {},
    }
    if output_path is not None:
        timing[_OUTPUT_PATH_KEY] = str(output_path)
    return timing


def add_build_timing(timing: dict | None, key: str, seconds: float) -> None:
    if timing is None:
        return
    phases = timing.setdefault("phases", {})
    phases[key] = float(phases.get(key, 0.0)) + float(seconds)


def build_timing_phase(timing: dict | None, key: str) -> float:
    """Return one accumulated phase duration."""
    if timing is None:
        return 0.0
    phases = timing.get("phases", {})
    try:
        return float(phases.get(key, 0.0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def untracked_phase_time(
    elapsed: float,
    before: float,
    timing: dict | None,
    key: str,
) -> float:
    """Return elapsed time not already recorded in one nested timing phase."""
    tracked = max(0.0, build_timing_phase(timing, key) - before)
    return max(0.0, elapsed - tracked)


def compile_time_excluding_weight_load(
    components_elapsed: float,
    weights_before_components: float,
    timing: dict | None,
) -> float:
    """Exclude nested component weight loading from component wall time."""
    component_weights = max(
        0.0,
        build_timing_phase(timing, "weights_loading_s")
        - weights_before_components,
    )
    return max(0.0, components_elapsed - component_weights)


def untracked_compile_time(
    measured_compile_elapsed: float,
    compile_before_components: float,
    timing: dict | None,
) -> float:
    """Return compile wall time not already reported by component builders."""
    return untracked_phase_time(
        measured_compile_elapsed,
        compile_before_components,
        timing,
        "trt_compile_s",
    )


def write_build_timing(
    timing: dict | None,
    output_path: str | Path | None = None,
) -> None:
    if timing is None:
        return
    path = str(output_path or timing.get(_OUTPUT_PATH_KEY, "")).strip()
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in timing.items() if k != _OUTPUT_PATH_KEY}
        p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        # Timing should never make a build fail.
        pass


@contextmanager
def timed_build_phase(timing: dict | None, key: str) -> Iterator[None]:
    t0 = time.monotonic()
    try:
        yield
    finally:
        add_build_timing(timing, key, time.monotonic() - t0)
        write_build_timing(timing)


@contextmanager
def timed_weight_loading(timing: dict | None, component: str) -> Iterator[None]:
    key = f"weights_loading_{component}_s"
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        add_build_timing(timing, "weights_loading_s", elapsed)
        add_build_timing(timing, key, elapsed)
        write_build_timing(timing)


@contextmanager
def timed_trt_compile(timing: dict | None, component: str) -> Iterator[None]:
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - t0
        add_trt_compile_timing(timing, component, elapsed)


def add_trt_compile_timing(
    timing: dict | None,
    component: str,
    seconds: float,
) -> None:
    key = f"trt_compile_{component}_s"
    add_build_timing(timing, "trt_compile_s", seconds)
    add_build_timing(timing, key, seconds)
    write_build_timing(timing)
