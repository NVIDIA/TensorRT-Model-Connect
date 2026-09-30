# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify one model against its native reference.

Phases (each failure is recorded and the report is still written):

1. Reference goldens: the native model at the golden precision with deterministic numerics,
   cached per suite, reference identity, and platform (including the reference environment).
2. Noise floor: the native model at the candidate precision graded against the goldens; the Acc
   gate is relaxed to what the reference itself reaches at that precision.
3. Reference perf: the native model at the candidate precision, eager and torch.compile.
4. Candidate: Acc on a persistent TRTMC session (failing suites are re-checked on an isolated
   session to separate state carried across requests from numerical problems), then L1 perf.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

from . import judge
from .aiperf_runner import AiperfRun, run_aiperf
from .config import Environment
from .goldens import GoldenStore, golden_key, platform_id
from .report import write_report
from .services import gpu_exclusive, platform_fingerprint, reference_python, serving
from .suites import Suite, build_suite, limit_suite, request_sha

from trtmc_aiperf_plugins.accuracy import COMPARATORS

TASK_ENDPOINT = ["--endpoint-type", "trtmc_task"]
# A script reference repeats warmup + iterations inside one process per request.
SCRIPT_REFERENCE_RUN = {"warmup": 0, "requests": 1, "runs": 1}
SCRIPT_ITERATIONS = 10
# A script reference loads the model once per request: its Acc suites use the first samples only.
SCRIPT_MAX_SAMPLES = 10


def _script_measurement(measurement: Mapping[str, Any]) -> dict[str, int]:
    """Iterations inside one script reference process: the Task's timed request count, bounded to
    3..SCRIPT_ITERATIONS (slow generative Tasks measure few requests)."""
    timed = int(measurement["requests"]) * int(measurement.get("runs", 1))
    return {"warmup": int(measurement.get("warmup", 0)), "iterations": max(3, min(SCRIPT_ITERATIONS, timed))}


def probe(service: Mapping[str, Any], operation: str, request: Mapping[str, Any]) -> None:
    """One request before a batch run, so a reference that cannot serve fails in one call."""
    import urllib.error
    import urllib.request

    body = json.dumps({"request": request}).encode()
    call = urllib.request.Request(f"{service['url']}/v1/tasks/{operation}", data=body,
                                  headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(call, timeout=3600).read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"probe rejected: {error.read().decode(errors='replace')[-600:]}") from error


def _task_url(service: Mapping[str, Any], operation: str) -> list[str]:
    return ["--url", f"{service['url']}/v1/tasks/{operation}"]


def reference_code_digest(repository: Path) -> str:
    """Digest of the generic reference adapters and media digests: changing how references produce
    observations must invalidate their goldens."""
    root = repository / "apps/perf_serving/trtmc_perf_serving"
    files = sorted((root / "backends/reference").glob("*.py")) + [root / "digests.py", root / "backends/script.py"]
    return hashlib.sha256(b"".join(path.read_bytes() for path in files if path.is_file())).hexdigest()[:16]


def _reference_identity(model: Mapping[str, Any], precision: str, deterministic: bool,
                        info: Mapping[str, Any] | None = None) -> dict:
    reference = model["reference"]
    identity = {"profile": model["catalog_profile"], "backend": reference["backend"], "precision": precision,
                "deterministic": deterministic, "reference_model": reference.get("model"),
                "reference_code": reference.get("code_digest"),
                "plugins": importlib.metadata.version("trtmc-aiperf-plugins")}
    if reference.get("options"):  # adapter options change reference outputs
        identity["options"] = reference["options"]
    if info:
        identity.update(adapter=info.get("adapter"), model=info.get("model"), revision=info.get("revision"))
    return identity


def _observations(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any],
                  suite: Suite, out: Path) -> dict[str, Any]:
    """One observation per suite sample from a server (sequential, one request each)."""
    run = run_aiperf(environment, out, [*TASK_ENDPOINT, *_task_url(service, model["operation"]), "--concurrency", "1",
                                        "--input-file", str(suite.write_inputs(out.parent / f"{out.name}.inputs.jsonl")),
                                        "--custom-dataset-type", "single_turn", "--dataset-sampling-strategy",
                                        "sequential", "--request-count", str(len(suite.samples))])
    observations, errors = {}, []
    for record in run.raw_records():
        if record.get("status") == 200:
            observations[request_sha(record["payload"]["request"])] = json.loads(
                record["responses"][-1]["text"])["trtmc_observation"]
        elif record.get("error"):
            errors.append(str(record.get("error"))[:300])
    missing = [s["sample_id"] for s in suite.samples if s["request_sha"] not in observations]
    if missing:
        raise RuntimeError(f"{suite.name}: exit {run.exit_code}, missing {missing[:3]}, errors {errors[:2]}")
    # A non-zero AIPerf exit with every observation present (for example a dropped optional
    # telemetry service) does not invalidate the outputs.
    return observations


def _perf_run(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any], suite: Suite,
              measurement: Mapping[str, Any], out: Path, aggregation: str) -> tuple[AiperfRun, dict[str, Any]]:
    arguments = [*TASK_ENDPOINT, *_task_url(service, model["operation"]), "--concurrency", "1",
                 "--input-file", str(suite.write_inputs(out.parent / f"{out.name}.inputs.jsonl")),
                 "--custom-dataset-type", "single_turn", "--dataset-sampling-strategy", "sequential",
                 "--request-count", str(measurement["requests"])]
    if int(measurement.get("warmup", 0)) > 0:
        arguments += ["--warmup-request-count", str(measurement["warmup"])]
    # AIPerf 0.13.0's --num-profile-runs breaks endpoints with tokenizes_input: false (trtmc_task);
    # the repetitions run here and use the same t-distribution confidence interval.
    runs = []
    for index in range(1, int(measurement.get("runs", 1)) + 1):
        runs.append(run_aiperf(environment, out / f"run_{index:02d}", arguments))
        if (runs[-1].summary.get(judge.METRIC) or {}).get("p50") is None:
            break  # nothing succeeded; further runs cannot either
    stats = judge.across_runs([(run.summary.get(judge.METRIC) or {}).get("p50") for run in runs], aggregation)
    stats["client_latency_p50_ms"] = judge.median_client_latency(runs[-1].raw_records())
    stats["aiperf_exit"] = max(run.exit_code for run in runs)
    return runs[0], stats


def _grade(grader: str, params: Mapping[str, Any], observed: Mapping[str, Any], goldens: Mapping[str, Any],
           suite: Suite) -> dict[str, Any]:
    compare = COMPARATORS[grader]
    passed, failures = 0, []
    for sample in suite.samples:
        try:
            ok, reason, _, _ = compare(observed[sample["request_sha"]], goldens[sample["request_sha"]], **params)
        except (KeyError, TypeError, ValueError) as error:
            ok, reason = False, f"not comparable: {error}"
        passed += bool(ok)
        if not ok:
            failures.append({"sample_id": sample["sample_id"], "reason": str(reason)[:200]})
    return {"passed": passed, "total": len(suite.samples), "failures": failures[:5]}


def sampled_request(request: Mapping[str, Any]) -> bool:
    return float(request.get("temperature", 0.0) or 0.0) > 0.0 and int(request.get("top_k", 0) or 0) != 1


def output_check(l1: Mapping[str, Any], candidate: Any, references: Mapping[str, Any], mode: str,
                 sampled: bool = False) -> tuple[bool, str]:
    """Perf output sanity check against the mode's reference output; a compiled reference whose own
    numerics diverge is not held against the candidate when the eager reference output agrees."""
    compare = COMPARATORS[l1["output_grader"]]

    def check(reference: Any) -> tuple[bool, str]:
        try:
            params = dict(l1.get("output_grader_params", {}))
            if sampled and l1["output_grader"] == "parity_token_exact":
                params["sampled"] = True
            match, reason, _, _ = compare(candidate, reference, **params)
            return match, reason
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            return False, f"not comparable: {error}"

    match, reason = check(references.get(mode))
    if not match and mode != "eager" and references.get("eager") is not None:
        eager_match, _ = check(references["eager"])
        if eager_match:
            return True, f"matches the eager reference ({mode} reference output differs: {reason})"
    return match, reason


def unavailable_mode(mode: str, reason: str) -> dict[str, Any]:
    return {"reference_mode": mode, "light": "n/a", "candidate": {}, "reference": {},
            "reasons": [f"native reference unavailable: {reason[:300]}"], "notes": []}


class _Phases:
    """Runs phases, recording failures so one failing phase does not hide the others."""

    def __init__(self, out: Path) -> None:
        self.errors: dict[str, str] = {}
        self.out = out

    def run(self, name: str, call):
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - reported per phase
            self.errors[name] = f"{type(error).__name__}: {error}"[:1500]
            with open(self.out / "phase-errors.log", "a") as log:
                log.write(f"== {name}\n{traceback.format_exc()}\n")
            return None


def _reference_goldens(environment: Environment, model: dict[str, Any], suites: Mapping[str, Suite],
                       store: GoldenStore, platform: str, fingerprint: Mapping[str, Any], python: str,
                       out: Path) -> tuple[dict, dict, dict]:
    reference = model["reference"]
    goldens, status, noise = {}, {}, {}
    backend = reference["backend"]
    deterministic = backend == "reference"
    keys = {name: golden_key(suite, _reference_identity(model, reference["precision"], deterministic),
                             fingerprint) for name, suite in suites.items()}
    for name, key in keys.items():
        cached = store.load(suites[name], platform, key)
        if cached is not None:
            goldens[name], status[name] = cached, {"status": "cached", "platform": platform, "key": key}
    missing = [name for name in keys if name not in goldens]
    if missing:
        with serving(environment, model, backend, out / f"reference-golden-{backend}", precision=reference["precision"],
                     deterministic=deterministic, python=python) as service:
            for name in missing:
                goldens[name] = _observations(environment, service, model, suites[name], out / f"golden-{name}")
                location = store.save(suites[name], platform, keys[name], goldens[name], {
                    "suite": suites[name].manifest,
                    "reference": _reference_identity(model, reference["precision"], deterministic, service["info"]),
                    "platform_id": platform, "platform": fingerprint, "host": service["info"].get("host"),
                    "numerics": service["info"].get("numerics"), "created": time.time()})
                status[name] = {"status": "generated", "platform": platform, "key": keys[name], "path": str(location)}
    if model.get("noise_floor") and backend == "script":
        # A script reference reloads the model for every request; its noise floor would double the
        # slowest part of the run, so script-backed models are gated without it.
        reference["noise_error"] = "not measured for script references (one model load per request)"
    elif model.get("noise_floor"):
        try:
            noise = _noise_floor(environment, model, suites, goldens, store, platform, fingerprint, python, out)
        except Exception as error:  # noqa: BLE001 - the native model may not run at this precision
            reference["noise_error"] = f"{type(error).__name__}: {error}"[:600]
    return goldens, status, noise


def _noise_floor(environment: Environment, model: dict[str, Any], suites: Mapping[str, Suite],
                 goldens: Mapping[str, Any], store: GoldenStore, platform: str, fingerprint: Mapping[str, Any],
                 python: str, out: Path) -> dict[str, Any]:
    reference, backend, noise = model["reference"], model["reference"]["backend"], {}
    precision = reference["perf_precision"]
    noise_keys = {name: golden_key(suite, _reference_identity(model, precision, False), fingerprint)
                  for name, suite in suites.items()}
    observed = {name: store.load(suites[name], platform, key) for name, key in noise_keys.items()}
    todo = [name for name, value in observed.items() if value is None]
    if todo:
        with serving(environment, model, backend, out / f"reference-noise-{backend}", precision=precision,
                     python=python) as service:
            probe(service, model["operation"], suites[todo[0]].samples[0]["request"])
            for name in todo:
                observed[name] = _observations(environment, service, model, suites[name], out / f"noise-{name}")
                store.save(suites[name], platform, noise_keys[name], observed[name], {
                    "suite": suites[name].manifest, "purpose": "noise-floor",
                    "reference": _reference_identity(model, precision, False, service["info"]),
                    "platform_id": platform, "platform": fingerprint, "created": time.time()})
    for item in model["accuracy"]:
        name = item["suite"]["suite"]
        noise[name] = {"precision": precision, **_grade(item["grader"], item.get("grader_params", {}),
                                                       observed[name], goldens[name], suites[name])}
    return noise


def _run_accuracy(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any],
                  suites: Mapping[str, Suite], goldens: Mapping[str, Any], noise: Mapping[str, Any], out: Path,
                  session: str, only: set[str] | None = None) -> list[dict[str, Any]]:
    results = []
    for item in model["accuracy"]:
        suite = suites[item["suite"]["suite"]]
        if only is not None and suite.name not in only:
            continue
        suite_file = suite.write_with_goldens(out / f"{session}-{suite.name}.parity.jsonl", goldens[suite.name],
                                              item.get("grader_params"))
        run = run_aiperf(environment, out / f"acc-{session}-{suite.name}",
                         [*TASK_ENDPOINT, *_task_url(service, model["operation"]), "--concurrency", "1",
                          "--accuracy-benchmark", "trtmc_parity", "--accuracy-grader", item["grader"],
                          "--request-count", str(len(suite.samples))],
                         env={"TRTMC_PARITY_SUITE": str(suite_file)})
        labels = {str(goldens[suite.name][sample["request_sha"]].get("text", "")).strip(): sample["label"]
                  for sample in suite.samples if "label" in sample}
        verdict = judge.judge_accuracy(run.accuracy_records(), item.get("gate", {}), len(suite.samples), labels,
                                       noise=noise.get(suite.name), sampled=bool(item.get("sampled")))
        results.append({"suite": suite.name, "grader": item["grader"], "session": session,
                        "aiperf_exit": run.exit_code, **verdict})
    return results


def timing_precisions(reference: Mapping[str, Any]) -> list[str]:
    """Precisions to time the native model at, in order: a declared ``timing_precision`` (when the
    native model does not run correctly at the candidate precision), else the candidate precision
    with the golden precision as the fallback."""
    if reference.get("timing_precision"):
        return [reference["timing_precision"]]
    return list(dict.fromkeys([reference["perf_precision"], reference["precision"]]))


def _reference_perf(environment: Environment, model: Mapping[str, Any], l1: Mapping[str, Any], suite: Suite,
                    python: str, phases: _Phases, out: Path) -> dict[str, tuple[AiperfRun, dict, dict]]:
    reference = model["reference"]
    script = reference["backend"] == "script"
    results: dict[str, tuple[AiperfRun, dict, dict]] = {}
    precisions = timing_precisions(reference)
    for mode in l1["reference_modes"]:
        def measure(mode: str = mode) -> None:
            errors = []
            for precision in precisions:
                try:
                    with serving(environment, model, reference["backend"], out / f"reference-{mode}-{precision}",
                                 mode=mode, precision=precision, python=python,
                                 script_measurement=_script_measurement(l1["measurement"])) as service:
                        if not script:  # a script reference would run its full measurement for the probe
                            probe(service, model["operation"], suite.samples[0]["request"])
                        run, stats = _perf_run(environment, service, model, suite,
                                               SCRIPT_REFERENCE_RUN if script else l1["measurement"],
                                               out / f"perf-reference-{mode}-{precision}",
                                               "best" if script else l1["aggregation"].get(mode, "mean"))
                    if stats.get("p50_ms") is None:
                        raise RuntimeError(f"no successful {precision} requests")
                    stats["precision"] = precision
                    if errors:
                        stats["precision_fallback"] = errors[-1][:300]
                    results[mode] = (run, stats, service["info"])
                    return
                except Exception as error:  # noqa: BLE001 - try the next precision
                    errors.append(f"{precision}: {type(error).__name__}: {error}")
            raise RuntimeError("; ".join(errors)[:1500])
        phases.run(f"reference_perf_{mode}", measure)
    return results


def _candidate(environment: Environment, model: Mapping[str, Any], suites: Mapping[str, Suite],
               goldens: Mapping[str, Any], noise: Mapping[str, Any], l1: Mapping[str, Any] | None,
               perf_suite: Suite | None, reference_perf: Mapping[str, tuple], accuracy: list, performance: list,
               out: Path) -> None:
    with serving(environment, model, "trtmc", out / "candidate") as service:
        if goldens:
            accuracy.extend(_run_accuracy(environment, service, model, suites, goldens, noise, out, "persistent"))
        if not l1:
            return
        run, stats = _perf_run(environment, service, model, perf_suite, l1["measurement"], out / "perf-candidate",
                               l1["aggregation"].get("candidate", "mean"))
        candidate_output = judge.first_observation(run.raw_records())
        outputs = {mode: judge.first_observation(entry[0].raw_records()) for mode, entry in reference_perf.items()}
        for mode, (reference_run, reference_stats, info) in reference_perf.items():
            match, reason = output_check(l1, candidate_output, outputs, mode,
                                         sampled_request(perf_suite.samples[0]["request"]))
            verdict = judge.judge_performance(stats, reference_stats, margin_percent=float(l1["margin_percent"]),
                                              max_ci_percent=float(l1["max_ci_percent"]),
                                              outputs_match=match, output_reason=reason)
            if reference_stats.get("precision_fallback"):
                verdict["notes"].append(f"reference measured at {reference_stats['precision']} "
                                        f"({reference_stats['precision_fallback'][:160]})")
            performance.append({"reference_mode": mode, "candidate_timing_scope": service["info"].get("timing_scope"),
                                "reference_timing_scope": info.get("timing_scope"),
                                "reference_backend": info.get("backend"), **verdict})


def qualify(model: dict[str, Any], environment: Environment, out: Path) -> dict[str, Any]:
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    phases = _Phases(out)
    store = GoldenStore(Path(environment["golden_store"]["root"]), environment["golden_store"].get("read_url"))
    suites = {item["suite"]["suite"]: build_suite(item["suite"], environment) for item in model["accuracy"]}
    l1 = model["performance"].get("l1")
    perf_suite = build_suite(l1["suite"], environment) if l1 else None
    (out / "suites").mkdir(exist_ok=True)
    for suite in [*suites.values(), *([perf_suite] if perf_suite else [])]:
        (out / "suites" / f"{suite.name}.manifest.json").write_text(json.dumps(suite.manifest, indent=2))
    (out / "model.json").write_text(json.dumps(model, indent=2, default=str))

    reference = model["reference"]
    reference["code_digest"] = reference_code_digest(environment.path("repo"))
    python = reference_python(environment, model)
    # Goldens are keyed by the reference platform: GPU architecture plus the reference environment's
    # framework versions, so a changed family environment regenerates them.
    fingerprint = platform_fingerprint(environment, python)["fingerprint"]
    platform = platform_id(fingerprint)

    if reference["backend"] == "script":
        suites = {name: limit_suite(suite, SCRIPT_MAX_SAMPLES) for name, suite in suites.items()}
    references = phases.run("reference_goldens", lambda: _reference_goldens(
        environment, model, suites, store, platform, fingerprint, python, out))
    if references is None and reference.get("declared_precision"):
        # For example fp32 not fitting: use the reference precision the family declares.
        first_error = phases.errors.pop("reference_goldens")
        reference.update(precision=reference.pop("declared_precision"), precision_fallback_from=first_error[:500])
        model["noise_floor"] = model.get("noise_floor") and reference["precision"] != reference["perf_precision"]
        references = phases.run("reference_goldens", lambda: _reference_goldens(
            environment, model, suites, store, platform, fingerprint, python, out))
    if references is None and reference.get("fallback"):
        first_error = phases.errors.pop("reference_goldens")
        reference.update(backend=reference["fallback"], fallback=None, fallback_from=first_error[:500])
        if reference["backend"] == "script":
            suites = {name: limit_suite(suite, SCRIPT_MAX_SAMPLES) for name, suite in suites.items()}
        references = phases.run("reference_goldens", lambda: _reference_goldens(
            environment, model, suites, store, platform, fingerprint, python, out))
    goldens, golden_status, noise = references or ({}, {}, {})
    accuracy: list[dict[str, Any]] = []
    performance_l1: list[dict[str, Any]] = []
    with gpu_exclusive(environment):
        reference_perf = _reference_perf(environment, model, l1, perf_suite, python, phases, out) if l1 else {}
        phases.run("candidate", lambda: _candidate(environment, model, suites, goldens, noise, l1, perf_suite,
                                                   reference_perf, accuracy, performance_l1, out))
    if performance_l1:  # the candidate was measured: record modes whose native reference could not run
        measured = {item["reference_mode"] for item in performance_l1}
        performance_l1 += [unavailable_mode(mode, phases.errors.get(f"reference_perf_{mode}", "not measured"))
                           for mode in l1["reference_modes"] if mode not in measured]
    failing = {item["suite"] for item in accuracy if item["status"] == "fail"}
    if failing:
        def isolated() -> None:
            with serving(environment, model, "trtmc", out / "candidate-isolated", isolate_requests=True) as service:
                for item in _run_accuracy(environment, service, model, suites, goldens, noise, out, "isolated",
                                          only=failing):
                    for persistent in accuracy:
                        if persistent["suite"] == item["suite"]:
                            persistent["isolated_check"] = {key: item[key] for key in ("status", "passed", "pass_rate")}
        phases.run("isolated_check", isolated)
    for item in accuracy:
        item["golden"] = golden_status.get(item["suite"])

    result = {"model": model["model"], "operation": model["operation"], "task": model.get("task"),
              "family": model.get("family"), "started": started, "platform": {"id": platform, **fingerprint},
              "reference": {key: reference.get(key) for key in ("backend", "precision", "perf_precision",
                                                                 "timing_precision", "fallback_from", "noise_error",
                                                                 "precision_fallback_from")},
              "reference_python": python, "duration_s": time.time() - started, "accuracy": accuracy,
              "noise_floor": noise, "performance_l1": performance_l1, "errors": phases.errors,
              "provenance": {"aiperf": importlib.metadata.version("aiperf"),
                             "plugins": importlib.metadata.version("trtmc-aiperf-plugins"),
                             "suites": {name: suite.manifest for name, suite in suites.items()},
                             "perf_suite": perf_suite.manifest if perf_suite else None}}
    result["verdict"] = judge.verdict(result, expected_suites=len(model["accuracy"]),
                                      expected_modes=len(l1["reference_modes"]) if l1 else 0)
    write_report(out, result)
    return result
