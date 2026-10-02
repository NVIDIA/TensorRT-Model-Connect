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
from typing import Any, Mapping, Sequence

from . import absolute, alignment, edits, family, geneval, intelligibility, judge, sweep
from .aiperf_runner import AiperfRun, run_aiperf
from .config import Environment
from .family import sampled_request
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


# A benchmark bundle longer than this that rejects requests is rebuilt at this length once.
FALLBACK_SEQUENCE_LENGTH = 2048


def _probe_candidate(environment: Environment, model: Mapping[str, Any], suite: Suite | None, out: Path) -> None:
    with serving(environment, dict(model), "trtmc", out) as service:
        probe(service, model["operation"], suite.samples[0]["request"] if suite else {})


def serviceable_candidate(environment: Environment, model: dict[str, Any], plans: dict[str, Any], suite: Suite | None,
                          python: str, out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """TRTMC must serve one request before the native model spends hours on the benchmarks. A longer
    benchmark bundle (``-qual``) that rejects every request (some families fail TensorRT enqueue at
    4096 tokens) is rebuilt once at FALLBACK_SEQUENCE_LENGTH; its problems are re-planned to fit."""
    import copy

    from . import bundles

    try:
        _probe_candidate(environment, model, suite, out / "absolute-probe")
        return model, plans
    except Exception as error:  # noqa: BLE001 - maybe a shorter bundle serves
        first = f"{type(error).__name__}: {str(error)[-300:]}"
    build = model["candidate"].get("build") or {}
    length = int(model["candidate"].get("max_sequence_length") or 0)
    if length <= FALLBACK_SEQUENCE_LENGTH or not build.get("max_sequence_length"):
        raise RuntimeError(first)
    shorter = copy.deepcopy(model)
    name = f"{model['catalog_profile']}-qual-{FALLBACK_SEQUENCE_LENGTH}"
    shorter["candidate"].update(build={**build, "name": name, "max_sequence_length": FALLBACK_SEQUENCE_LENGTH},
                                max_sequence_length=FALLBACK_SEQUENCE_LENGTH,
                                bundle=f"{name}/{model['candidate']['bundle'].split('/', 1)[1]}")
    built = bundles.ensure_bundle(environment, shorter, python, out / f"retry-{FALLBACK_SEQUENCE_LENGTH}")
    if built["status"] == "failed":
        raise RuntimeError(f"{first}; the {FALLBACK_SEQUENCE_LENGTH}-token rebuild failed: {built.get('reason', '')[:300]}")
    _probe_candidate(environment, shorter, suite, out / f"absolute-probe-{FALLBACK_SEQUENCE_LENGTH}")
    shorter["candidate"]["sequence_fallback"] = (f"the {length}-token bundle rejected requests ({first[:200]}); "
                                                 f"benchmarks and Perf on a {FALLBACK_SEQUENCE_LENGTH}-token bundle")
    return shorter, {item["suite"]: absolute.plan(environment, shorter, item) for item in shorter["absolute"]}


def _task_url(service: Mapping[str, Any], operation: str) -> list[str]:
    return ["--url", f"{service['url']}/v1/tasks/{operation}"]


def family_code_digest(repository: Path, family: str | None) -> str | None:
    """Digest of the family's qualification reference code: changing it must invalidate goldens."""
    root = repository / "families" / str(family) / "tests/benchmark"
    files = sorted(root.glob("*.py")) if family and root.is_dir() else []
    return hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()[:16] if files else None


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
                "reference_code": reference.get("code_digest"), "family_code": reference.get("family_code"),
                "checkpoint": model["candidate"].get("checkpoint"), "revision": model["candidate"].get("revision"),
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
    runs, busy = [], []
    for index in range(1, int(measurement.get("runs", 1)) + 1):
        busy.append(gpu_busy_percent())
        runs.append(run_aiperf(environment, out / f"run_{index:02d}", arguments))
        if (runs[-1].summary.get(judge.METRIC) or {}).get("p50") is None:
            break  # nothing succeeded; further runs cannot either
    stats = judge.across_runs([(run.summary.get(judge.METRIC) or {}).get("p50") for run in runs], aggregation)
    stats["client_latency_p50_ms"] = judge.median_client_latency(runs[-1].raw_records())
    stats["aiperf_exit"] = max(run.exit_code for run in runs)
    expected_runs, problems = int(measurement.get("runs", 1)), []
    if len(runs) < expected_runs:
        problems.append(f"{len(runs)} of {expected_runs} runs completed")
    for run in runs:
        problem = run_completeness(run, int(measurement["requests"]))
        if problem:
            problems.append(f"{getattr(getattr(run, 'directory', None), 'name', 'run')}: {problem}")
    if problems:
        stats["incomplete"] = "; ".join(problems)[:600]
    elif stats["aiperf_exit"]:  # every request succeeded: a late telemetry/export failure only
        stats["exit_note"] = f"AIPerf exited {stats['aiperf_exit']} after all requests succeeded"
    measured = [value for value in busy if value is not None]
    if measured:
        stats["gpu_busy_percent"] = max(measured)
    return runs[0], stats


def run_completeness(run: Any, expected: int) -> str | None:
    """Why a timed run is not a complete measurement (a request missing, failed, or cancelled), or None."""
    records = run.raw_records()
    failed = [record for record in records if record.get("status") != 200 or record.get("error")
              or (record.get("metadata") or {}).get("was_cancelled")]
    if len(records) == expected and not failed:
        return None
    statuses = sorted({str(record.get("status")) for record in failed})
    return (f"{len(records) - len(failed)} of {expected} requests succeeded"
            + (f" (failed with status {', '.join(statuses)})" if failed else ""))


# Checks that judge whole outputs per Task (``supplementary``); each returns report entries.
SUPPLEMENTARY_CHECKS = {"tts_intelligibility": intelligibility.run, "clip_alignment": alignment.run,
                        "replay_parity": alignment.run_replay, "geneval": geneval.run,
                        "edit_similarity": edits.run}
# The report entries each check writes (rejudge leaves them; recheck replaces them).
SUPPLEMENTARY_SUITES = {"tts_intelligibility": ("tts-intelligibility",),
                        "clip_alignment": ("clip-alignment", "replay-parity"), "replay_parity": ("replay-parity",),
                        "geneval": ("geneval", "vbench-objects"), "edit_similarity": ("edit-similarity",)}


def applies(check: Mapping[str, Any], model: Mapping[str, Any]) -> bool:
    """A supplementary check limited to ``only_families`` skips other families (e.g. GenEval: images)."""
    return not check.get("only_families") or model.get("family") in check["only_families"]


def expected_suites(model: Mapping[str, Any]) -> dict[str, str]:
    """Every Accuracy result the configuration requires, mapped to the phase that produces it."""
    expected = {item["suite"]["suite"]: "candidate" for item in model.get("accuracy", [])}
    expected.update({name: "family_accuracy" for name in model.get("family_accuracy", [])})
    expected.update({item["suite"]: "candidate" for item in model.get("absolute", [])})
    for check in model.get("supplementary", []):
        if not applies(check, model):
            continue
        replay = model.get("family") in check.get("latent_replay_families", ())
        names = {"tts_intelligibility": ["tts-intelligibility"],
                 "clip_alignment": ["clip-alignment"] + (["replay-parity"] if replay else []),
                 "replay_parity": ["replay-parity"] if replay else [],
                 "geneval": [check.get("entry", "geneval")],
                 "edit_similarity": [check.get("entry", "edit-similarity")]}.get(check.get("check"), [])
        expected.update({name: check["check"] for name in names})
    return expected


def missing_results(model: Mapping[str, Any], accuracy: Sequence[Mapping[str, Any]],
                    errors: Mapping[str, str]) -> list[dict[str, Any]]:
    """Error entries for required results no phase produced (a phase that failed before writing them)."""
    produced = {item.get("suite") for item in accuracy}
    return [{"suite": name, "source": "missing", "status": "error", "samples": 0, "passed": None,
             "required_passes": None, "error": f"not produced ({phase}): {errors.get(phase, 'no result written')}"[:800]}
            for name, phase in expected_suites(model).items() if name not in produced]


def supplementary(environment: Environment, model: dict[str, Any], check: Mapping[str, Any], python: str,
                  out: Path) -> list[dict[str, Any]]:
    result = SUPPLEMENTARY_CHECKS[check["check"]](environment, model, check, python, out)
    return result if isinstance(result, list) else [result]


def gpu_busy_percent(samples: int = 5, interval_s: float = 0.2, settle_s: float = 1.0) -> float | None:
    """Other processes' GPU load just before a timed run, while our servers are idle.

    nvidia-smi averages utilization over its last sample period, so a reading right after our own
    request (seconds long for diffusion) still shows that request. The lowest of a few readings taken
    after a short settle is what persists without us.
    """
    import subprocess

    time.sleep(settle_s)
    readings = []
    for index in range(samples):
        try:
            completed = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=30)
            values = [float(line) for line in completed.stdout.split() if line.strip().replace(".", "").isdigit()]
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return None
        if values:
            readings.append(max(values))
        if index + 1 < samples:
            time.sleep(interval_s)
    return min(readings) if readings else None


def _grade(grader: str, params: Mapping[str, Any], observed: Mapping[str, Any], goldens: Mapping[str, Any],
           suite: Suite) -> dict[str, Any]:
    compare = COMPARATORS[grader]
    passed, failures, failed = 0, [], []
    for index, sample in enumerate(suite.samples):
        try:
            ok, reason, _, _ = compare(observed[sample["request_sha"]], goldens[sample["request_sha"]], **params)
        except (KeyError, TypeError, ValueError) as error:
            ok, reason = False, f"not comparable: {error}"
        passed += bool(ok)
        if not ok:
            failed.append(index)
            failures.append({"sample_id": sample["sample_id"], "reason": str(reason)[:200]})
    return {"passed": passed, "total": len(suite.samples), "failed_indices": failed, "failures": failures[:5]}


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
    native model does not run correctly at the candidate precision), else the candidate precision,
    then the family reference's declared precision and the golden precision as fallbacks."""
    if reference.get("timing_precision"):
        return [reference["timing_precision"]]
    # The family reference may accept only its declared precision (for example bf16).
    return list(dict.fromkeys(value for value in (reference["perf_precision"], reference.get("declared_precision"),
                                                  reference["precision"]) if value))


def _reference_perf(environment: Environment, model: Mapping[str, Any], l1: Mapping[str, Any], suite: Suite,
                    python: str, phases: _Phases, out: Path) -> dict[str, tuple[AiperfRun, dict, dict]]:
    """Time the native model per reference mode; the generic adapter falls back to the family's
    declared reference (``script``) when it cannot run the model at any precision."""
    reference = model["reference"]
    results: dict[str, tuple[AiperfRun, dict, dict]] = {}
    precisions = timing_precisions(reference)
    for mode in l1["reference_modes"]:
        def measure(mode: str = mode) -> None:
            errors = []
            for backend in dict.fromkeys([reference["backend"], reference.get("fallback") or reference["backend"]]):
                for precision in precisions:
                    try:
                        results[mode] = _time_reference(environment, model, l1, suite, python, backend, mode,
                                                        precision, out)
                        stats = results[mode][1]
                        if errors and backend == reference["backend"]:
                            stats["precision_fallback"] = errors[-1][:300]
                        if backend != reference["backend"]:
                            stats["backend_fallback"] = "; ".join(errors)[:300]
                        return
                    except Exception as error:  # noqa: BLE001 - try the next precision, then the fallback
                        errors.append(f"{backend} {precision}: {type(error).__name__}: {error}")
            raise RuntimeError("; ".join(errors)[:1500])
        phases.run(f"reference_perf_{mode}", measure)
    return results


def _time_reference(environment: Environment, model: Mapping[str, Any], l1: Mapping[str, Any], suite: Suite,
                    python: str, backend: str, mode: str, precision: str,
                    out: Path) -> tuple[AiperfRun, dict[str, Any], dict[str, Any]]:
    tag = f"{mode}-{precision}" if backend == model["reference"]["backend"] else f"{mode}-{backend}-{precision}"
    with serving(environment, model, backend, out / f"reference-{tag}", mode=mode, precision=precision,
                 python=python, script_measurement=_script_measurement(l1["measurement"])) as service:
        # A script reference in its own process measures itself per request; a persistent one
        # (harness session) is timed per request like the adapters.
        per_process = backend == "script" and service["info"].get("timing") != "persistent"
        if not per_process:
            probe(service, model["operation"], suite.samples[0]["request"])
        run, stats = _perf_run(environment, service, model, suite,
                               SCRIPT_REFERENCE_RUN if per_process else l1["measurement"],
                               out / f"perf-reference-{tag}",
                               "best" if per_process else l1["aggregation"].get(mode, "mean"))
        if per_process:
            stats["timing"] = "reference process (its own warmup and iterations, not AIPerf)"
    if stats.get("p50_ms") is None:
        raise RuntimeError(f"no successful {precision} requests")
    stats["precision"] = precision
    return run, stats, service["info"]


def _candidate(environment: Environment, model: Mapping[str, Any], suites: Mapping[str, Suite],
               goldens: Mapping[str, Any], noise: Mapping[str, Any], l1: Mapping[str, Any] | None,
               perf_suite: Suite | None, reference_perf: Mapping[str, tuple], accuracy: list, performance: list,
               out: Path, absolute_runs: Mapping[str, Any] | None = None) -> None:
    with serving(environment, model, "trtmc", out / "candidate") as service:
        if absolute_runs:
            accuracy.extend(absolute.candidate_entries(environment, service, model, out, **absolute_runs))
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
                                              outputs_match=match, output_reason=reason,
                                              not_equivalent=l1.get("not_equivalent"))
            if reference_stats.get("timing"):
                verdict["notes"].append(f"native timed by the {reference_stats['timing']}")
            if reference_stats.get("backend_fallback"):
                verdict["notes"].append("native: the family's declared reference (the generic adapter failed: "
                                        f"{reference_stats['backend_fallback'][:160]})")
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
    reference["family_code"] = family_code_digest(environment.path("repo"), model.get("family"))
    python = reference_python(environment, model)
    # Goldens are keyed by the reference platform: GPU architecture plus the reference environment's
    # framework versions, so a changed family environment regenerates them.
    fingerprint = platform_fingerprint(environment, python)["fingerprint"]
    platform = platform_id(fingerprint)

    if reference["backend"] == "script":
        suites = {name: limit_suite(suite, SCRIPT_MAX_SAMPLES) for name, suite in suites.items()}
    references = phases.run("reference_goldens", lambda: _reference_goldens(
        environment, model, suites, store, platform, fingerprint, python, out)) if suites else ({}, {}, {})
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
    # Absolute accuracy: the problems both sides answer (selected and length-checked outside the GPU lock).
    plans = phases.run("absolute_plan", lambda: {item["suite"]: absolute.plan(environment, model, item)
                                                 for item in model["absolute"]}) if model.get("absolute") else None
    if model.get("absolute") and plans is None:
        accuracy.extend(absolute.error_entry(item, 0, f"problem selection: {phases.errors.get('absolute_plan')}")
                        for item in model["absolute"])
    if model.get("family_accuracy"):
        def family_phase() -> None:
            entries = family.run(environment, model, out, python)
            family.attribute(environment, model, out, python, entries)
            accuracy.extend(entries)
        phases.run("family_accuracy", family_phase)
    for check in model.get("supplementary", []):
        if check.get("check") in SUPPLEMENTARY_CHECKS and applies(check, model):
            def run_check(check: Mapping[str, Any] = check) -> None:
                accuracy.extend(supplementary(environment, model, check, python, out))
            phases.run(check["check"], run_check)
    if plans:  # outside the GPU lock: a rebuild takes it
        served = phases.run("absolute_probe", lambda: serviceable_candidate(environment, model, plans, perf_suite,
                                                                            python, out))
        if served:
            model, plans = served
            (out / "model.json").write_text(json.dumps(model, indent=2, default=str))
    with gpu_exclusive(environment):
        reference_perf = _reference_perf(environment, model, l1, perf_suite, python, phases, out) if l1 else {}
        absolute_runs = None
        if plans:
            if "absolute_probe" in phases.errors:
                accuracy.extend(absolute.error_entry(item, len(plans[item["suite"]]),
                                                     f"TRTMC cannot serve the model: {phases.errors['absolute_probe'][:500]}")
                                for item in model["absolute"])
            else:
                native = phases.run("absolute_native", lambda: absolute.run_native(
                    environment, model, python, plans, out, perf_suite.samples[0]["request"] if perf_suite else None))
                absolute_runs = {"plans": plans, "native": native or {},
                                 "native_error": phases.errors.get("absolute_native")}
        phases.run("candidate", lambda: _candidate(environment, model, suites, goldens, noise, l1, perf_suite,
                                                   reference_perf, accuracy, performance_l1, out, absolute_runs))
        l2 = model["performance"].get("l2")
        performance_l2: dict[str, Any] = {}
        # The sweeps start the generic adapter: only where L1 could time it (not a script fallback).
        generic = any(item.get("reference_backend") == "reference" for item in performance_l1)
        if l2 and l2.get("kind") == "media" and generic:
            phases.run("perf_l2", lambda: performance_l2.update(sweep.run_media(
                environment, model, l2, out, python, timing_precisions(reference)[0])))
        elif l2 and generic:
            def serving_sweep() -> None:
                with serving(environment, model, "trtmc", out / "l2-candidate-server") as candidate, \
                        serving(environment, model, "reference", out / "l2-reference-server", mode="eager",
                                precision=timing_precisions(reference)[0], python=python) as native:
                    performance_l2.update(sweep.run(environment, model, l2, {"candidate": candidate,
                                                                            "reference": native}, out))
            phases.run("perf_l2", serving_sweep)
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
    failing_family = {item["suite"] for item in accuracy if item.get("source") == "family" and item["status"] == "fail"}
    if failing_family:
        def isolated_family() -> None:
            for item in family.run(environment, model, out, python, isolated=True, only=failing_family):
                for persistent in accuracy:
                    if persistent["suite"] == item["suite"] and persistent.get("source") == "family":
                        persistent["isolated_check"] = {key: item.get(key) for key in ("status", "passed", "pass_rate")}
        phases.run("isolated_family_check", isolated_family)
    for item in accuracy:
        if item.get("source") != "family":
            item["golden"] = golden_status.get(item["suite"])

    result = {"model": model["model"], "operation": model["operation"], "task": model.get("task"),
              "repro": f"trtmc-aiperf-qual run --profile {model['catalog_profile']} --environment "
                       f"{environment.values.get('environment_file', '<environment.yaml>')} --out {out}",
              "family": model.get("family"), "started": started, "platform": {"id": platform, **fingerprint},
              **({"coverage": model["coverage"]} if model.get("coverage") else {}),
              **({"candidate_note": model["candidate"]["sequence_fallback"]}
                 if model["candidate"].get("sequence_fallback") else {}),
              "reference": {key: reference.get(key) for key in ("backend", "precision", "perf_precision",
                                                                 "timing_precision", "fallback_from", "noise_error",
                                                                 "precision_fallback_from")},
              "reference_python": python, "duration_s": time.time() - started, "accuracy": accuracy,
              "noise_floor": noise, "performance_l1": performance_l1, "performance_l2": performance_l2,
              "errors": phases.errors,
              "provenance": {"aiperf": importlib.metadata.version("aiperf"),
                             "plugins": importlib.metadata.version("trtmc-aiperf-plugins"),
                             "suites": {name: suite.manifest for name, suite in suites.items()},
                             "perf_suite": perf_suite.manifest if perf_suite else None}}
    result["accuracy"] += missing_results(model, result["accuracy"], phases.errors)
    result["verdict"] = judge.verdict(result, expected_suites=list(expected_suites(model)),
                                      expected_modes=len(l1["reference_modes"]) if l1 else 0)
    write_report(out, result)
    return result
