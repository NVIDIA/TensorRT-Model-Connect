# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy by the family's own benchmark qualification cases, with TRTMC requests sent by AIPerf.

A family accuracy case (``families/<family>/tests/benchmark/*.yaml``) owns the dataset, selection,
native reference, metric, and gate. Its evaluation runs unchanged
(``qualification_tests.benchmark_qualification.accuracy.run_accuracy``); only the TRTMC side is
replaced: the candidate requests go through AIPerf to a trtmc-perf-serve session (persistent, or one
worker per request for the isolated re-check) instead of a one-shot trtmc-bench run.
"""

from __future__ import annotations

import json
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .bundles import bundle_path
from .config import Environment
from .models import PRECISIONS
from .services import serving
from .suites import Suite, request_sha

MAX_FAILURES = 10


def _qualification(repository: Path):
    from .models import _import_repository

    _import_repository(repository)
    from qualification_tests.benchmark_qualification import accuracy, catalog, runtime

    return accuracy, catalog, runtime


def cases(repository: Path, profile: str) -> list[Any]:
    """The family's accuracy cases of a catalog profile (empty when the family declares none)."""
    _, catalog, _ = _qualification(repository)
    return [case for case in catalog.discover(repository) if case.model == profile and case.kind == "accuracy"]


def _context(environment: Environment, repository: Path, out: Path) -> Any:
    _, _, runtime = _qualification(repository)
    return runtime.RuntimeContext(
        repository=repository, artifacts=out, data_root=environment.path("data_root"),
        environment_root=environment.path("reference_env_root"), bundle_cache=environment.path("bundle_root"),
        bundle_roots=(), runtime_root=environment.path("runtime_root"),
        trtmc_bench=repository / "apps/benchmark/trtmc-bench", worker=environment.path("worker"), datasets={},
        reference_pythons={}, no_build=True, verbose=False)


@contextmanager
def _patched(module: Any, name: str, value: Any) -> Iterator[None]:
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def _candidate_via_aiperf(environment: Environment, model: Mapping[str, Any], service: Mapping[str, Any]):
    """Replacement of accuracy._candidate_outputs: the same requests, sent by AIPerf to the session."""
    from trtmc_perf_serving.files import inline_files

    from .runner import _observations

    def candidate_outputs(case: Any, context: Any, output: Path, operation: str,
                          requests: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Path]:
        if operation != model["operation"]:
            raise RuntimeError(f"{case.id}: the TRTMC session serves {model['operation']!r}, not {operation!r}")
        samples = []
        for index, item in enumerate(requests):
            request = inline_files(dict(item["request"]))  # clients send files inline, never server paths
            samples.append({"sample_id": str(item.get("sample_id", index)), "task": case.name,
                            "request": request, "request_sha": request_sha(request)})
        suite = Suite(case.name, f"family-{case.id}", samples, {"suite": case.name, "family_case": case.id})
        output.mkdir(parents=True, exist_ok=True)
        observations = _observations(environment, service, model, suite, output / "candidate-aiperf")
        # Output artifacts are named relative to the request's working directory on the server
        # (kept with --keep-artifacts); AIPerf sent the samples in order, one at a time.
        records = [json.loads(line) for line in Path(service["records"]).read_text().splitlines() if line.strip()]
        request_ids = [record["request_id"] for record in records
                       if str(record.get("route", "")).startswith("/v1/tasks/")][-len(samples):]
        scratch = Path(service["records"]).parent / "scratch"
        outputs = [_resolve_artifacts(dict(observations[sample["request_sha"]]), scratch / request_id)
                   for sample, request_id in zip(samples, request_ids, strict=True)]
        (output / "candidate-outputs.json").write_text(json.dumps(outputs, indent=1, default=str))
        return outputs, bundle_path(environment, model)

    return candidate_outputs


def _resolve_artifacts(observation: dict[str, Any], workdir: Path) -> dict[str, Any]:
    """``*_artifact`` / ``*_artifacts`` paths as trtmc-bench returns them: absolute, inside the request's
    working directory."""
    root = workdir.resolve()

    def resolve(value: str) -> str:
        path = Path(value) if Path(value).is_absolute() else root / value
        path = path.resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"unsafe artifact path {value!r}")
        return str(path)

    for name, value in tuple(observation.items()):
        if name.endswith("_artifact") and isinstance(value, str):
            observation[name] = resolve(value)
        elif name.endswith("_artifacts") and isinstance(value, list):
            observation[name] = [resolve(item) for item in value if isinstance(item, str)]
    return observation


def _row_evidence(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """(explanation, candidate, reference) of one family sample row."""
    candidate = {k[len("candidate_"):]: v for k, v in row.items() if k.startswith("candidate_")}
    reference = {k[len("reference_"):]: v for k, v in row.items() if k.startswith("reference_")}
    detail = {k: v for k, v in row.items()
              if k not in ("sample_id", "passed") and not k.startswith(("candidate_", "reference_"))}
    return (json.dumps(detail, default=str)[:300], json.dumps(candidate, default=str)[:300],
            json.dumps(reference, default=str)[:300])


def counts(result: Mapping[str, Any], inputs: int | None = None) -> dict[str, Any]:
    """Sample counts, status, gate, and failing samples of a family result. ``passed`` is None when the
    family grades aggregate metrics only (for example mAP or corpus WER). ``inputs``: how many inputs a
    case without per-sample rows executed (one request); its media and frame counts stay metrics."""
    rows = [row for row in result.get("samples", []) if isinstance(row, Mapping)]
    metrics = dict(result.get("metrics", {}))
    total = len(rows) or int(metrics.get("samples", 0)) or int(inputs or 0)
    graded = [row for row in rows if "passed" in row]
    if graded:
        passed = sum(bool(row["passed"]) for row in graded)
    else:
        passed = int(metrics["passed_samples"]) if "passed_samples" in metrics else None
    failures, failed_samples = [], [str(row.get("sample_id")) for row in graded if not row["passed"]]
    for row in graded:
        if not row["passed"] and len(failures) < MAX_FAILURES:
            explanation, actual, expected = _row_evidence(row)
            failures.append({"sample_id": row.get("sample_id"), "explanation": explanation,
                             "actual": actual, "expected": expected})
    return {"status": "pass" if result.get("status") == "passed" else "fail", "samples": total,
            "expected_samples": total, "passed": passed,
            "pass_rate": passed / total if total and passed is not None else None,
            "gate": dict(result.get("gate", {})), "metrics": metrics, "failures": failures,
            "failed_samples": failed_samples}


def executed_inputs(case: Any) -> int | None:
    """Inputs a case without a dataset sends: its single request."""
    values = getattr(case, "values", None) or {}
    return 1 if values.get("request") is not None and not values.get("dataset") else None


def sampled_request(request: Mapping[str, Any]) -> bool:
    """The request samples (TRTMC does not replay PyTorch's random stream); top_k 1 is greedy."""
    stochastic = bool(request.get("do_sample")) or float(request.get("temperature") or 0.0) > 0.0
    return stochastic and int(request.get("top_k") or 0) != 1


def sampled(case: Any) -> bool:
    return sampled_request((getattr(case, "values", None) or {}).get("request") or {})


def item(case: Any, result: Mapping[str, Any], evidence: Path) -> dict[str, Any]:
    """A family result as one accuracy entry of the report (the family's own status and gate)."""
    inputs = executed_inputs(case)
    reference = (getattr(case, "values", None) or {}).get("reference") or {}
    return {"suite": case.name, "source": "family", "benchmark": case.benchmark, "required_passes": None,
            **counts(result, inputs), **({"executed_inputs": inputs} if inputs else {}),
            "sampled": sampled(case), "reference_precision": reference.get("precision"), "evidence": str(evidence)}


def refresh(entry: Mapping[str, Any]) -> dict[str, Any]:
    """A recorded family entry recounted from its saved result (``rejudge``); unchanged without one."""
    evidence = Path(str(entry.get("evidence") or ""))
    if entry.get("source") != "family" or evidence.suffix != ".json" or not evidence.is_file():
        return dict(entry)
    refreshed = {**entry, **counts(json.loads(evidence.read_text()), entry.get("executed_inputs"))}
    if refreshed["status"] == "fail" and (entry.get("sampled") or entry.get("precision_sensitive")):
        refreshed["status"] = "inconclusive"  # settled when it ran (sampling, or the native control)
    isolated = Path(str(evidence).replace("/family-persistent/", "/family-isolated/"))
    if entry.get("isolated_check") and isolated != evidence and isolated.is_file():
        recount = counts(json.loads(isolated.read_text()))
        refreshed["isolated_check"] = {key: recount[key] for key in ("status", "passed", "pass_rate")}
    return refreshed


def run(environment: Environment, model: dict[str, Any], out: Path, reference_python: str, *,
        isolated: bool = False, only: set[str] | None = None,
        control: tuple[str, str] | None = None) -> list[dict[str, Any]]:
    """Run the family's accuracy cases of ``model``; one report entry per case (errors included).
    ``control`` (backend, precision): the native model at that precision stands in for TRTMC, so the
    case measures how far the native model itself is from the family reference."""
    repository = environment.path("repo")
    accuracy, _, _ = _qualification(repository)
    selected = [case for case in cases(repository, model["catalog_profile"]) if only is None or case.name in only]
    if not selected:
        return []
    session = f"control-{control[0]}-{control[1]}" if control else "isolated" if isolated else "persistent"
    root = out / f"family-{session}"
    context = _context(environment, repository, root)
    entries = []
    server = (serving(environment, model, control[0], out / f"native-family-{session}", precision=control[1],
                      python=reference_python, keep_artifacts=True) if control else
              serving(environment, model, "trtmc", out / f"candidate-family-{session}", isolate_requests=isolated,
                      keep_artifacts=True))
    with server as service, \
            _patched(accuracy, "_candidate_outputs", _candidate_via_aiperf(environment, model, service)), \
            _reference_python(context, reference_python):
        for case in selected:
            evidence = context.case_artifacts(case) / "result.json"
            try:
                entries.append(item(case, accuracy.run_accuracy(case, context), evidence))
            except Exception as error:  # noqa: BLE001 - one failing case does not stop the others
                evidence.parent.mkdir(parents=True, exist_ok=True)
                (evidence.parent / "error.log").write_text(traceback.format_exc())
                entries.append({"suite": case.name, "source": "family", "benchmark": case.benchmark,
                                "status": "error", "samples": 0, "expected_samples": 0, "passed": 0,
                                "required_passes": None, "gate": dict(case.values.get("gate", {})),
                                "failures": [], "error": f"{type(error).__name__}: {error}"[:600],
                                "evidence": str(evidence.parent / "error.log")})
    return entries


@contextmanager
def _reference_python(context: Any, python: str) -> Iterator[None]:
    """Family references run in the family environment (the serving interpreter when none is declared)."""
    _, _, runtime = _qualification(context.repository)
    original = runtime.reference_python

    def resolved(case: Any, ctx: Any) -> Path:
        return Path(python) if python else original(case, ctx)

    with _patched(runtime, "reference_python", resolved):
        from qualification_tests.benchmark_qualification import accuracy

        with _patched(accuracy, "reference_python", resolved) if hasattr(accuracy, "reference_python") else \
                _noop():
            yield


@contextmanager
def _noop() -> Iterator[None]:
    yield


def attribute(environment: Environment, model: dict[str, Any], out: Path, reference_python: str,
              entries: list[dict[str, Any]]) -> None:
    """Settle failing family cases: a sampling case is ``inconclusive``;
    when the family reference runs at another precision than the candidate, the native model at the
    candidate precision repeats the case, and a failure it shares on the same samples (or, for
    aggregate-only metrics, at all) is ``inconclusive`` and marked ``precision_sensitive``."""
    candidate_precision = model["candidate"].get("precision")
    for entry in entries:
        if entry.get("status") != "fail":
            continue
        if entry.get("sampled"):
            entry["status"] = "inconclusive"
            continue
        if entry.get("reference_precision") in (None, candidate_precision) or candidate_precision not in PRECISIONS:
            continue
        reference, errors = model["reference"], []
        for backend in dict.fromkeys([reference["backend"], reference.get("fallback") or reference["backend"]]):
            try:
                control = run(environment, model, out, reference_python, only={entry["suite"]},
                              control=(backend, candidate_precision))[0]
            except Exception as error:  # noqa: BLE001 - try the fallback; no control is reported
                errors.append(f"{backend}: {type(error).__name__}: {error}"[:200])
                continue
            if control.get("status") == "error":
                errors.append(f"{backend}: {control.get('error', '')}"[:200])
                continue
            entry["native_control"] = {key: control.get(key) for key in ("status", "passed", "samples", "metrics")}
            entry["native_control"].update(backend=backend, precision=candidate_precision)
            shared = set(entry.get("failed_samples") or []) <= set(control.get("failed_samples") or []) \
                if entry.get("failed_samples") else True
            if control.get("status") == "fail" and shared:
                entry.update(status="inconclusive", precision_sensitive=True)
            break
        else:
            entry["native_control"] = {"status": "error", "error": "; ".join(errors)[:600]}
