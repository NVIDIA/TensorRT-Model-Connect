# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family reference scripts served over HTTP.

Models whose native reference cannot be loaded by the generic Hugging Face adapters (upstream
checkouts, NeMo, Ultralytics, custom pipelines) declare a reference in their benchmark
qualification case: a family ``reference.py`` or a shared task adapter. This backend runs that
declared reference for every request, through the same descriptor and command construction as
benchmark qualification, so the framework itself holds no family code.

Each request starts one reference process (model load excluded from timing); ``model_call_ms``
is the reference's own latency p50 over its warmup and iterations, and the observation is its
output summary.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .base import BackendError, BackendUnavailable, Invocation

MODES = {"eager": ("hf-eager", "pytorch-eager", "torch-eager"), "compile": ("torch-compile",)}


def _import_qualification(repository: Path):
    for root in (repository, repository / "core/builder", repository / "apps/benchmark"):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from qualification_tests.benchmark_qualification import catalog, runtime
    from qualification_tests.benchmark_qualification.performance import matrix, qualification

    return catalog, runtime, matrix, qualification


def qualification_case(repository: Path, profile: str, kind: str = "performance") -> Any:
    catalog, _, _, _ = _import_qualification(repository)
    cases = [case for case in catalog.discover(repository) if case.model == profile and case.kind == kind]
    if not cases:
        raise BackendUnavailable(f"{profile} has no benchmark qualification {kind} case")
    return cases[0]


class ScriptReferenceBackend:
    def __init__(self, *, repository: Path, profile: str, mode: str, scratch: Path, runtime_root: Path,
                 warmup: int, iterations: int, precision: str | None = None, timeout_s: int = 7200) -> None:
        catalog, runtime, matrix, qualification = _import_qualification(repository)
        self._matrix, self._runtime, self._qualification = matrix, runtime, qualification
        self._case = qualification_case(repository, profile)
        definition = catalog.load_benchmark(repository, self._case)
        scratch.mkdir(parents=True, exist_ok=True)
        self._scratch = scratch
        # The server already runs in the family's reference environment (see ``reference-env``).
        self._context = runtime.RuntimeContext(
            repository=repository, artifacts=scratch, data_root=scratch, environment_root=scratch,
            bundle_cache=scratch, bundle_roots=(), runtime_root=runtime_root,
            trtmc_bench=repository / "apps/benchmark/trtmc-bench", worker=None, datasets={},
            reference_pythons={profile: Path(sys.executable)}, no_build=True, verbose=False)
        configured = dict(self._case.values["reference"])
        if precision:
            configured["precision"] = precision
        self._baseline = {**qualification._resolve_reference_assets(self._case, self._context, configured),
                          **dict(definition["reference_timing"])}
        declared = [str(self._baseline.get("mode", "torch-compile"))]
        if isinstance(self._baseline.get("fallback"), str):
            declared.append(self._baseline["fallback"])
        self._modes = [value for value in declared if value in MODES.get(mode, ())]
        if not self._modes:
            raise BackendUnavailable(f"{profile} reference declares modes {declared}, none is {mode!r}")
        self._environment = matrix.Environment(
            name="trtmc-perf-serve-script", trtmc_bench=self._context.trtmc_bench,
            worker=runtime_root / "trtmc_benchmark_worker",
            hf_runner=repository / "qualification_tests/benchmark_qualification/performance/references/hf_transformers.py",
            task_runner=repository / "qualification_tests/benchmark_qualification/performance/references/generic_reference.py",
            results_root=scratch, scratch_root=scratch, bundle_cache=scratch, bundle_roots=(),
            runtime_root=runtime_root, bundle_retention="retain",
            local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1", timeout_seconds=timeout_s, references={},
            reference_python=Path(sys.executable))
        self._measurement = {"warmup": warmup, "iterations": iterations}
        self.operation = str(self._case.values["operation"])
        self._profile = profile

    def describe(self) -> Mapping[str, Any]:
        baseline = self._baseline
        return {"backend": "script", "operation": self.operation, "case": self._case.id,
                "model": self._case.candidate.get("checkpoint"), "revision": self._case.candidate.get("revision"),
                "precision": baseline.get("precision"), "modes": self._modes,
                "adapter": baseline.get("adapter") or baseline.get("script") or baseline.get("runner"),
                "measurement": dict(self._measurement), "timing_scope": baseline.get("timing_scope"),
                "input_preparation_included": baseline.get("input_preparation_included"),
                "numerics": {"deterministic": False}}

    def _entry(self, request: Mapping[str, Any], work: Path) -> Any:
        descriptor = self._runtime.write_model_descriptor(self._case, work, request, context=self._context)
        spec = {"id": f"aiperf.{self._profile}", "family": self._case.family, "operation": self.operation,
                "model": self._case.model, "manifest": str(descriptor),
                "workload": {"testcase": self._case.name, "request": dict(request)},
                "measurement": dict(self._measurement), "baseline": dict(self._baseline)}
        return self._matrix.resolve_entries([spec], self._environment)[0]

    def invoke(self, request: Mapping[str, Any], artifact_base: Path) -> Invocation:
        matrix = self._matrix
        work = Path(tempfile.mkdtemp(prefix="script-", dir=self._scratch))
        try:
            entry = self._entry(request, work)
        except (matrix.PerfMatrixError, self._runtime.QualificationError, ValueError, KeyError) as error:
            raise BackendError(f"cannot prepare the reference request: {error}") from error
        errors = []
        for mode in self._modes:
            output = work / f"reference-{re.sub(r'[^a-z-]', '', mode)}.json"
            command = matrix._baseline_command(entry, self._environment, output, mode=mode)
            completed = matrix.run_command(command, timeout=self._environment.timeout_seconds,
                                           stdout_path=work / f"{mode}.stdout.log",
                                           stderr_path=work / f"{mode}.stderr.log", verbose=False,
                                           env=matrix._command_environment())
            result = json.loads(output.read_text()) if output.is_file() else {}
            if completed["exit_code"] != 0 or result.get("status") != "completed":
                tail = (work / f"{mode}.stderr.log").read_text(errors="replace")[-800:]
                errors.append(f"{mode}: exit {completed['exit_code']} {result.get('error', '')} {tail}")
                continue
            observation = {**result.get("output_summary", {}), "reference_mode": mode}
            _link_media(observation, output, artifact_base)
            return Invocation(observation=observation, model_call_ms=matrix._p50(result),
                              extra={"samples_ms": result.get("samples_ms", []), "reference_mode": mode})
        raise BackendError("; ".join(errors) or "reference produced no result")

    def close(self) -> None:
        pass


def _link_media(observation: dict[str, Any], output: Path, artifact_base: Path) -> None:
    """Digest reference media; expose reference audio under the request's artifact prefix.

    Family references materialize only the first, middle, and last frame of a video, which are
    the frames the digest samples; the digest keeps the declared frame count.
    """
    media = sorted(output.with_suffix(".media").glob("*.png")) if output.with_suffix(".media").is_dir() else []
    if media:
        import numpy as np
        from PIL import Image

        from ..digests import image_digest

        frames = [np.asarray(Image.open(path).convert("RGB")) for path in media]
        digest = image_digest(frames)
        declared = int(observation.get("media_count") or observation.get("num_frames") or len(frames))
        if len(frames) == 3 and declared > 3:
            digest.update(frames=declared, sampled_frames=sorted({0, declared // 2, declared - 1}))
        observation["media_digest"] = digest
    audio = observation.get("audio_artifact")
    sources = []
    if isinstance(audio, str) and Path(audio).is_file():
        sources.append((Path(audio), ".wav"))
    artifact_base.parent.mkdir(parents=True, exist_ok=True)
    for source, suffix in sources:
        target = artifact_base.parent / f"{artifact_base.name}{suffix}"
        if not target.exists():
            os.symlink(source, target)
