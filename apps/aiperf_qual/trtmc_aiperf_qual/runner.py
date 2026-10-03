# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify one model against its native reference.

Phases (each failure is recorded and the report is still written):

1. Accuracy plan: the gold-labelled benchmark problems both sides answer (``absolute``); the Task's
   whole-output checks (``supplementary``) run as they come.
2. Probe: TRTMC serves one request before the native model spends time on the benchmarks.
3. Reference perf: the native model at the candidate precision, eager (and torch.compile where listed).
4. Native answers to the benchmarks, then the candidate: TRTMC's answers, then L1 perf.
"""

from __future__ import annotations

import importlib.metadata
import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import absolute, edits, geneval, intelligibility, judge, replay_parity, sweep, world_model
from .aiperf_runner import AiperfRun, run_aiperf
from .config import Environment
from .report import write_report
from .services import gpu_exclusive, platform_fingerprint, platform_id, reference_python, serving
from .suites import Suite, build_suite, request_sha, single_request_suite, unstated_defaults

from trtmc_aiperf_plugins.accuracy import COMPARATORS

TASK_ENDPOINT = ["--endpoint-type", "trtmc_task"]


def sampled_request(request: Mapping[str, Any]) -> bool:
    """The request samples (TRTMC does not replay PyTorch's random stream); top_k 1 is greedy."""
    stochastic = bool(request.get("do_sample")) or float(request.get("temperature") or 0.0) > 0.0
    return stochastic and int(request.get("top_k") or 0) != 1


TEXT_OPERATIONS = ("generate", "translate")
GREEDY = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "do_sample": False}
# The near-capacity request: a public passage filling the bundle up to these many generated tokens,
# at most NEAR_CAPACITY_MAX prompt tokens (native eager prefill memory).
NEAR_CAPACITY_NEW_TOKENS = 32
NEAR_CAPACITY_MAX = 16384
LENGTH_REJECTIONS = ("exceed", "capacity", "exhaust")  # words of TRTMC's prompt-length rejections
# TRTMC tokenizes the passage itself (it may count a few more tokens than the Hugging Face tokenizer) and a
# bundle's prefill profile may be shorter than its sequence length: a prompt TRTMC rejects for length is
# shortened to the longest it accepts (a binary search over the passage length).


def timed_request(model: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """The request Perf times: a sampling text request as its greedy variant, so both sides generate the
    same tokens (DESIGN.md 4.6); every other request as is."""
    if model["operation"] in TEXT_OPERATIONS and sampled_request(request):
        return {**request, **GREEDY}
    return dict(request)


def _passage(environment: Environment, tokenizer: Any, count: int) -> str:
    """WikiText-103 test text (pinned) cut to ``count`` tokens of the model's tokenizer."""
    from trtmc_aiperf_plugins.benchmarks import WIKITEXT_REVISION
    from trtmc_aiperf_plugins.benchmarks import load_dataset

    rows = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", revision=WIKITEXT_REVISION,
                        cache_dir=str(environment["hf_datasets_cache"]))
    text, ids = "", []
    for row in rows:
        text += row["text"]
        if len(text) > 8 * count:  # at least one token per eight characters
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= count:
                break
    return tokenizer.decode(ids[:count], skip_special_tokens=True)


def rendered_tokens(tokenizer: Any, request: Mapping[str, Any]) -> int:
    """Prompt tokens of a text request as rendered for the model: its chat template when the request asks
    for one, else the tokenizer with its special tokens."""
    prompt = str(request["prompt"])
    if request.get("use_chat_template"):
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                                 tokenize=True, return_dict=True,
                                                 enable_thinking=bool(request.get("enable_thinking", False)))
        return len(rendered["input_ids"])
    return len(tokenizer(prompt)["input_ids"])


def near_capacity_request(environment: Environment, model: Mapping[str, Any], request: Mapping[str, Any],
                          budget: int, service: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
    """The request with a passage whose rendered prompt has ``budget`` tokens (or the most below it that the
    tokenizer's round trip allows) and NEAR_CAPACITY_NEW_TOKENS to generate, greedy, or the longest passage the
    TRTMC ``service`` accepts below that; with the rendered prompt length."""
    from transformers import AutoTokenizer

    name, revision = absolute.tokenizer_source(model)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision,
                                              trust_remote_code=bool(model["reference"].get("trust_remote_code")))
    count = budget
    for _ in range(5):  # the template's tokens, then the decode / re-encode round trip, taken off the passage
        # ``request`` is the timed request, greedy already (``timed_request``): its own sampling fields stay, as a
        # family may pin its greedy contract (Qwen3-Omni: temperature 1 with top_k 1).
        long = {**request, "prompt": _passage(environment, tokenizer, count), "max_new_tokens": NEAR_CAPACITY_NEW_TOKENS}
        excess = rendered_tokens(tokenizer, long) - budget
        if excess <= 0:
            break
        count -= excess
    else:
        raise RuntimeError(f"cannot fit a passage into {budget} rendered prompt tokens")

    def accepted(request: Mapping[str, Any]) -> bool:
        try:
            probe(service, model["operation"], request)
            return True
        except RuntimeError as error:  # a length rejection ("exceed(s) the ... capacity / prefill profile",
            message = str(error).lower()  # "exhausted its KV cache")
            if "backend_rejected_request" not in message or not any(word in message for word in LENGTH_REJECTIONS):
                raise
            return False

    if not accepted(long):
        low, high = 0, count  # the longest accepted passage is in [low, high)
        while high - low > 1:
            middle = (low + high) // 2
            candidate = {**long, "prompt": _passage(environment, tokenizer, middle)}
            if accepted(candidate):
                low, long = middle, candidate
            else:
                high = middle
        if low == 0:
            raise RuntimeError("TRTMC rejects every near-capacity prompt for length")
    return long, rendered_tokens(tokenizer, long)


def near_capacity_applies(model: Mapping[str, Any], request: Mapping[str, Any]) -> bool:
    """The text generation Task's text-only requests (not vision-language models, whose image tokens share the
    bundle length) on a bundle long enough for a passage."""
    length = int(model["candidate"].get("max_sequence_length") or 0)
    return (model.get("task") == "text_generation" and not any(key.startswith("image") for key in request)
            and length > 2 * NEAR_CAPACITY_NEW_TOKENS)


def perf_suites(environment: Environment, model: Mapping[str, Any], suite: Suite,
                service: Mapping[str, Any] | None = None) -> list[Suite]:
    """The timed requests: the catalog request (greedy where it samples text) and, where
    ``near_capacity_applies``, the near-capacity request sized against the TRTMC ``service``."""
    request = timed_request(model, suite.samples[0]["request"])
    suites = [single_request_suite(suite.name, request, suite.manifest)]
    if near_capacity_applies(model, request):
        if service is None:
            raise RuntimeError("sizing the near-capacity request needs the TRTMC server")
        budget = min(int(model["candidate"]["max_sequence_length"]) - NEAR_CAPACITY_NEW_TOKENS, NEAR_CAPACITY_MAX)
        long, tokens = near_capacity_request(environment, model, request, budget, service)
        suites.append(single_request_suite(f"{suite.name}-near-capacity", long,
                                           {"near_capacity_tokens": tokens, "near_capacity_budget": budget}))
    return suites


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


def candidate_probe(environment: Environment, model: Mapping[str, Any], suite: Suite | None, out: Path,
                    phases: "_Phases", *, serviceability: bool, l1: bool) -> list[Suite]:
    """One TRTMC server before any native work: it must serve the first request (when Acc runs) and sizes the
    near-capacity request; returns the timed requests (none without L1 or when they cannot be built)."""
    with serving(environment, dict(model), "trtmc", out / "absolute-probe") as service:
        if serviceability:
            phases.run("absolute_probe", lambda: probe(service, model["operation"],
                                                        suite.samples[0]["request"] if suite else {}))
        if not l1 or suite is None:
            return []
        return phases.run("perf_requests", lambda: perf_suites(environment, model, suite, service)) or []


def _task_url(service: Mapping[str, Any], operation: str) -> list[str]:
    return ["--url", f"{service['url']}/v1/tasks/{operation}"]


def _observations(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any],
                  suite: Suite, out: Path) -> dict[str, Any]:
    """One observation per suite sample from a server (sequential, one request each)."""
    run = run_aiperf(environment, out, [*TASK_ENDPOINT, *_task_url(service, model["operation"]), "--concurrency", "1",
                                        "--input-file", str(suite.write_inputs(out.parent / f"{out.name}.inputs.jsonl")),
                                        "--custom-dataset-type", "single_turn", "--dataset-sampling-strategy",
                                        "sequential", "--request-count", str(len(suite.samples))],
                     timeout_s=absolute.run_timeout(environment, model))
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


def _responses(run: AiperfRun) -> list[tuple[float | None, dict[str, Any] | None]]:
    """(server model-call ms or None, observation or None) of every successful timed request; a success
    without a readable body has neither."""
    found = []
    for record in run.raw_records():
        if record.get("status") != 200:
            continue
        try:
            body = json.loads(record["responses"][-1]["text"])
        except (KeyError, IndexError, TypeError, ValueError):
            found.append((None, None))
            continue
        timing = (body.get("trtmc_timing") or {}).get("model_call_ms")
        found.append((None if timing is None else float(timing), body.get("trtmc_observation")))
    return found


def _perf_run(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any], suite: Suite,
              measurement: Mapping[str, Any], out: Path, aggregation: str) -> tuple[AiperfRun, dict[str, Any]]:
    """Repeated timed runs of the suite's request. Each run's statistic is the median server model-call
    time; every response must carry its time, and every response's work signature is kept."""
    arguments = [*TASK_ENDPOINT, *_task_url(service, model["operation"]), "--concurrency", "1",
                 "--input-file", str(suite.write_inputs(out.parent / f"{out.name}.inputs.jsonl")),
                 "--custom-dataset-type", "single_turn", "--dataset-sampling-strategy", "sequential",
                 "--request-count", str(measurement["requests"])]
    if int(measurement.get("warmup", 0)) > 0:
        arguments += ["--warmup-request-count", str(measurement["warmup"])]
    # AIPerf 0.13.0's --num-profile-runs breaks endpoints with tokenizes_input: false (trtmc_task);
    # the repetitions run here.
    runs, busy, per_run, work, untimed = [], [], [], [], 0
    for index in range(1, int(measurement.get("runs", 1)) + 1):
        busy.append(gpu_busy_percent())
        runs.append(run_aiperf(environment, out / f"run_{index:02d}", arguments,
                               timeout_s=absolute.run_timeout(environment, model)))
        responses = _responses(runs[-1])
        times = [ms for ms, _ in responses if ms is not None]
        untimed += len(responses) - len(times)
        if not times:
            per_run.append(None)
            break  # nothing succeeded; further runs cannot either
        per_run.append(statistics.median(times))
        work += [judge.work_signature(model["operation"], obs) for _, obs in responses]
    stats = judge.across_runs(per_run, aggregation)
    stats["work"] = list(dict.fromkeys(signature for signature in work if signature is not None))  # distinct
    stats["work_missing"] = sum(signature is None for signature in work)
    stats["client_latency_p50_ms"] = judge.median_client_latency(runs[-1].raw_records())
    stats["aiperf_exit"] = max(run.exit_code for run in runs)
    expected_runs, problems = int(measurement.get("runs", 1)), []
    if len(runs) < expected_runs:
        problems.append(f"{len(runs)} of {expected_runs} runs completed")
    if untimed:
        problems.append(f"{untimed} successful responses carry no model-call time")
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
SUPPLEMENTARY_CHECKS = {"tts_intelligibility": intelligibility.run, "replay_parity": replay_parity.run,
                        "geneval": geneval.run, "edit_similarity": edits.run, "world_model_parity": world_model.run}
# The report entries each check writes (rejudge leaves them; recheck replaces them).
SUPPLEMENTARY_SUITES = {"tts_intelligibility": ("tts-intelligibility", "tts-validity"), "replay_parity": ("replay-parity",),
                        "geneval": ("geneval", "vbench-objects", "vbench-objects-video-validity"),
                        "edit_similarity": ("edit-similarity-clip-i", "edit-similarity-dino", "edit-similarity-changed"),
                        "world_model_parity": ("world-model-parity",)}


def applies(check: Mapping[str, Any], model: Mapping[str, Any]) -> bool:
    """A supplementary check limited to ``only_families`` skips other families (e.g. GenEval: images)."""
    return not check.get("only_families") or model.get("family") in check["only_families"]


def _check_entries(check: Mapping[str, Any], model: Mapping[str, Any]) -> list[str]:
    """The report entries a supplementary check writes for this model."""
    if not applies(check, model):
        return []
    replay = model.get("family") in check.get("latent_replay_families", ())
    return {"tts_intelligibility": ["tts-intelligibility", "tts-validity"],
            "replay_parity": ["replay-parity"] if replay else [],
            "geneval": [check.get("entry", "geneval")] + ([f"{check.get('entry', 'geneval')}-video-validity"]
                                                          if check.get("video") else []),
            "edit_similarity": [f"{check.get('entry', 'edit-similarity')}-{part}"
                                for part in ("clip-i", "dino", "changed")],
            "world_model_parity": ["world-model-parity"]}.get(check.get("check"), [])


def informational_suites(model: Mapping[str, Any]) -> set[str]:
    """Entries reported but not judged: an ``informational`` check's."""
    names: set[str] = set()
    for check in model.get("supplementary", []):
        if check.get("informational"):
            names.update(_check_entries(check, model))
    return names


def mark_informational(model: Mapping[str, Any], accuracy: Sequence[dict[str, Any]]) -> None:
    names = informational_suites(model)
    for item in accuracy:
        if item.get("suite") in names:
            item["informational"] = True


def expected_suites(model: Mapping[str, Any]) -> dict[str, str]:
    """Every judged Accuracy result the configuration requires, mapped to the phase that produces it."""
    expected = {item["suite"]: "candidate" for item in model.get("absolute", [])}
    if model.get("accuracy_source") == "missing":  # a Task without an accuracy scheme: always an error
        expected["accuracy-scheme"] = "accuracy_scheme"
    if model.get("accuracy_source") == "none":  # Perf only: conversion parity is the Acc evidence
        expected[CONVERSION_PARITY] = "candidate"
    for check in model.get("supplementary", []):
        expected.update({name: check["check"] for name in _check_entries(check, model)})
    informational = informational_suites(model)
    return {name: phase for name, phase in expected.items() if name not in informational}


CONVERSION_PARITY = "conversion-parity"


def conversion_parity(performance: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A Perf-only model's Acc evidence (DESIGN.md 4.7): on every timed request TRTMC's output passes the
    Task's output check against the native eager output (text: the first 8 greedy token ids, or the whole
    text, are equal); a mismatch is a ``fail``. Empty when no eager comparison ran (``missing_results``
    then reports it)."""
    checks = [(item.get("request"), item.get("output_check") or {}) for item in performance
              if item.get("reference_mode") == "eager" and "match" in (item.get("output_check") or {})]
    if not checks:
        return []
    failures = [{"sample_id": str(request), "explanation": str(check.get("reason"))[:300]}
                for request, check in checks if not check["match"]]
    count = len(checks)
    return [{"suite": CONVERSION_PARITY, "source": "parity", "benchmark": "TRTMC vs native eager output on the timed requests",
             "samples": count, "expected_samples": count, "passed": count - len(failures), "required_passes": count,
             "status": "fail" if failures else "pass", "failures": failures,
             "reasons": [f"{len(failures)} of {count} timed requests differ from the native output"] if failures else []}]


def missing_results(model: Mapping[str, Any], accuracy: Sequence[Mapping[str, Any]],
                    errors: Mapping[str, str]) -> list[dict[str, Any]]:
    """Error entries for required results no phase produced (a phase that failed before writing them)."""
    produced = {item.get("suite") for item in accuracy}
    reasons = {"accuracy_scheme": f"no accuracy benchmark is configured for Task {model.get('task')!r}"}
    return [{"suite": name, "source": "missing", "status": "error", "samples": 0, "passed": None,
             "required_passes": None,
             "error": reasons.get(phase) or f"not produced ({phase}): {errors.get(phase, 'no result written')}"[:800]}
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


def unavailable_mode(mode: str, reason: str, request: str | None = None) -> dict[str, Any]:
    return {"reference_mode": mode, **({"request": request} if request else {}), "light": "n/a", "candidate": {},
            "reference": {}, "reasons": [f"native reference unavailable: {reason[:300]}"], "notes": []}


class _Phases:
    """Runs phases, recording failures so one failing phase does not hide the others. A phase with
    ``retries`` runs again after a failure (DESIGN.md Section 9: one retry of a phase that fails before
    producing its result); ``reset`` undoes a failed attempt's partial results first."""

    def __init__(self, out: Path) -> None:
        self.errors: dict[str, str] = {}
        self.out = out

    def run(self, name: str, call, retries: int = 0, reset=None):
        for attempt in range(retries + 1):
            try:
                return call()
            except Exception as error:  # noqa: BLE001 - reported per phase
                self.errors[name] = f"{type(error).__name__}: {error}"[:1500]
                with open(self.out / "phase-errors.log", "a") as log:
                    log.write(f"== {name} (attempt {attempt + 1})\n{traceback.format_exc()}\n")
                if reset is not None:
                    reset()
        return None


def timing_precisions(reference: Mapping[str, Any]) -> list[str]:
    """Precisions to time the native model at, in order: a declared ``timing_precision`` (when the
    native model does not run correctly at the candidate precision), else the candidate precision,
    then the reference precision (fp32) as the fallback."""
    if reference.get("timing_precision"):
        return [reference["timing_precision"]]
    return list(dict.fromkeys(value for value in (reference["perf_precision"], reference["precision"]) if value))


def _reference_perf(environment: Environment, model: Mapping[str, Any], l1: Mapping[str, Any], suites: Sequence[Suite],
                    python: str, phases: _Phases, out: Path) -> dict[str, dict[str, tuple[AiperfRun, dict, dict]]]:
    """Time the native model per reference mode and timed request, at the first precision it runs at."""
    results: dict[str, dict[str, tuple[AiperfRun, dict, dict]]] = {}
    for mode in l1["reference_modes"]:
        def measure(mode: str = mode) -> None:
            errors = []
            for precision in timing_precisions(model["reference"]):
                try:
                    results[mode] = _time_reference(environment, model, l1, suites, python, mode, precision, out)
                    if errors:
                        for _, stats, _ in results[mode].values():
                            stats["precision_fallback"] = errors[-1][:300]
                    return
                except Exception as error:  # noqa: BLE001 - try the next precision
                    errors.append(f"{precision}: {type(error).__name__}: {error}")
            raise RuntimeError("; ".join(errors)[:1500])
        phases.run(f"reference_perf_{mode}", measure, retries=retries(environment))
        if mode in results:
            phases.errors.pop(f"reference_perf_{mode}", None)  # a retry measured it
    return results


def _time_reference(environment: Environment, model: Mapping[str, Any], l1: Mapping[str, Any], suites: Sequence[Suite],
                    python: str, mode: str, precision: str,
                    out: Path) -> dict[str, tuple[AiperfRun, dict[str, Any], dict[str, Any]]]:
    tag = f"{mode}-{precision}"
    timed = {}
    with serving(environment, model, "reference", out / f"reference-{tag}", mode=mode, precision=precision,
                 python=python) as service:
        for suite in suites:
            probe(service, model["operation"], suite.samples[0]["request"])
            run, stats = _perf_run(environment, service, model, suite, l1["measurement"],
                                   out / f"perf-reference-{tag}-{suite.name}", l1["aggregation"].get(mode, "mean"))
            if stats.get("p50_ms") is None:
                raise RuntimeError(f"no successful {precision} requests ({suite.name})")
            stats["precision"] = precision
            timed[suite.name] = (run, stats, service["info"])
    return timed


def _candidate(environment: Environment, model: Mapping[str, Any], l1: Mapping[str, Any] | None,
               suites: Sequence[Suite], reference_perf: Mapping[str, Mapping[str, tuple]], accuracy: list,
               performance: list, out: Path, absolute_runs: Mapping[str, Any] | None = None) -> None:
    with serving(environment, model, "trtmc", out / "candidate", keep_artifacts=absolute.keeps_artifacts(model)) as service:
        if absolute_runs:
            accuracy.extend(absolute.candidate_entries(environment, service, model, out, **absolute_runs))
        if not l1:
            return
        for suite in suites:
            sampled = sampled_request(suite.samples[0]["request"])
            run, stats = _perf_run(environment, service, model, suite, l1["measurement"],
                                   out / f"perf-candidate-{suite.name}", l1["aggregation"].get("candidate", "mean"))
            candidate_output = judge.first_observation(run.raw_records())
            timed = {mode: by_suite[suite.name] for mode, by_suite in reference_perf.items() if suite.name in by_suite}
            outputs = {mode: judge.first_observation(entry[0].raw_records()) for mode, entry in timed.items()}
            for mode, (reference_run, reference_stats, info) in timed.items():
                match, reason = output_check(l1, candidate_output, outputs, mode, sampled)
                verdict = judge.judge_performance(stats, reference_stats, margin_percent=float(l1["margin_percent"]),
                                                  max_ci_percent=float(l1["max_ci_percent"]),
                                                  outputs_match=match, output_reason=reason,
                                                  not_equivalent=l1.get("not_equivalent"),
                                                  candidate_precision=model["reference"].get("perf_precision"))
                if (model["reference"].get("options") or {}).get("cpu_offload"):
                    verdict["notes"].append("the native pipeline offloads its weights to host memory (larger than the GPU)")
                performance.append({"reference_mode": mode, "request": suite.name,
                                    "candidate_timing_scope": service["info"].get("timing_scope"),
                                    "reference_timing_scope": info.get("timing_scope"),
                                    "reference_backend": info.get("backend"), **verdict})


SMOKE_MEASUREMENT = {"warmup": 0, "requests": 1, "runs": 1}
PHASE_RETRIES = 1  # a GPU phase that fails before producing its result runs once more (DESIGN.md Section 9)


def retries(environment: Environment) -> int:
    """Retries of a failed GPU phase: none in smoke mode (it is there to find failures fast)."""
    return 0 if environment.values.get("smoke") else PHASE_RETRIES


def smoke_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """The model as smoke mode runs it: L1 with one request against native eager, no opt-in sweeps."""
    l1 = model["performance"].get("l1")
    performance = {"l1": {**l1, "measurement": dict(SMOKE_MEASUREMENT), "reference_modes": ["eager"]}} if l1 else {}
    return {**model, "performance": performance}


def smoke_verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """smoke-pass when every phase ran and produced its results (whatever they say), else smoke-fail
    with the failing phases."""
    failing = [f"phase {name}" for name in result.get("errors") or {}]
    failing += [f"{item['suite']}: {item.get('error') or '; '.join(item.get('reasons', []))}"[:300]
                for item in result.get("accuracy", []) if item.get("status") == "error"]
    failing += [f"perf {item['reference_mode']}: {'; '.join(item.get('reasons', []))}"[:300]
                for item in result.get("performance_l1", []) if item.get("light") in ("error", "n/a")]
    if result.get("reference", {}).get("backend") == "unsupported":
        failing.append("no native adapter")
    return {**result.get("verdict", {}), "category": "smoke-fail" if failing else "smoke-pass", "failing": failing}


def qualify(model: dict[str, Any], environment: Environment, out: Path) -> dict[str, Any]:
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    phases = _Phases(out)
    smoke = bool(environment.values.get("smoke"))
    if smoke:
        model = smoke_model(model)
    l1 = model["performance"].get("l1")
    perf_suite = build_suite(l1["suite"], environment) if l1 else None
    (out / "suites").mkdir(exist_ok=True)
    if perf_suite:
        (out / "suites" / f"{perf_suite.name}.manifest.json").write_text(json.dumps(perf_suite.manifest, indent=2))
    (out / "model.json").write_text(json.dumps(model, indent=2, default=str))

    reference = model["reference"]
    if reference["backend"] == "unsupported":  # no native path: nothing to compare against
        result = {"model": model["model"], "operation": model["operation"], "task": model.get("task"),
                  "family": model.get("family"), "started": started, "reference": {"backend": "unsupported"},
                  "accuracy": [], "performance_l1": [], "performance_l2": {}, "duration_s": time.time() - started,
                  "errors": {"native": reference.get("not_covered") or
                             f"no native adapter serves {model['operation']!r} for Task {model.get('task')!r}"},
                  "provenance": {"aiperf": importlib.metadata.version("aiperf"),
                                 "plugins": importlib.metadata.version("trtmc-aiperf-plugins"), "perf_suite": None}}
        result["verdict"] = judge.verdict(result, expected_suites=[], expected_modes=0)
        result["mode"] = "smoke" if smoke else "formal"
        if smoke:
            result["verdict"] = smoke_verdict(result)
        write_report(out, result)
        return result
    python = reference_python(environment, model)
    fingerprint = platform_fingerprint(environment, python)["fingerprint"]
    from . import gold_metrics

    gold_metrics.protect([out.parent, *(environment.values.get(key) for key in (
        "bundle_root", "hf_hub_cache", "hf_datasets_cache", "data_root", "reference_env_root", "runtime_root"))])
    accuracy: list[dict[str, Any]] = []
    performance_l1: list[dict[str, Any]] = []
    unstated = unstated_defaults(perf_suite.samples[0]["request"]) if perf_suite else []
    if unstated:  # each side would apply its own default: different work
        phases.errors["perf_request"] = (f"the timed request leaves {', '.join(unstated)} to each side's default; "
                                         "state them in config/models performance.l1")
        l1 = None
    # Absolute accuracy: the problems both sides answer (selected and length-checked outside the GPU lock).
    plans = phases.run("absolute_plan", lambda: {item["suite"]: absolute.plan(environment, model, item)
                                                 for item in model["absolute"]}) if model.get("absolute") else None
    if model.get("absolute") and plans is None:
        accuracy.extend(absolute.error_entry(item, 0, f"problem selection: {phases.errors.get('absolute_plan')}")
                        for item in model["absolute"])
    for check in model.get("supplementary", []):
        if check.get("check") in SUPPLEMENTARY_CHECKS and applies(check, model):
            def run_check(check: Mapping[str, Any] = check) -> None:
                accuracy.extend(supplementary(environment, model, check, python, out))
            phases.run(check["check"], run_check)
    probe_server = bool(plans) or bool(l1 and perf_suite and near_capacity_applies(
        model, timed_request(model, perf_suite.samples[0]["request"])))
    if probe_server:
        timed_suites = phases.run("candidate_probe", lambda: candidate_probe(
            environment, model, perf_suite, out, phases, serviceability=bool(plans), l1=bool(l1))) or []
        if "candidate_probe" in phases.errors and plans:  # the server did not start: TRTMC cannot serve
            phases.errors.setdefault("absolute_probe", phases.errors["candidate_probe"])
    else:
        timed_suites = (phases.run("perf_requests", lambda: perf_suites(environment, model, perf_suite)) or []) if l1 else []
    if l1 and not timed_suites:
        l1 = None  # the timed requests could not be built: the phase error says why
    with gpu_exclusive(environment):
        reference_perf = _reference_perf(environment, model, l1, timed_suites, python, phases, out) if l1 else {}
        absolute_runs = None
        if plans:
            if "absolute_probe" in phases.errors:
                accuracy.extend(absolute.error_entry(item, len(plans[item["suite"]]),
                                                     f"TRTMC cannot serve the model: {phases.errors['absolute_probe'][:500]}")
                                for item in model["absolute"])
            else:
                native = phases.run("absolute_native", lambda: absolute.run_native(
                    environment, model, python, plans, out, perf_suite.samples[0]["request"] if perf_suite else None),
                    retries=retries(environment))
                if native:
                    phases.errors.pop("absolute_native", None)
                absolute_runs = {"plans": plans, "native": native or {},
                                 "native_error": phases.errors.get("absolute_native")}
        marks = (len(accuracy), len(performance_l1))

        def undo() -> None:  # a failed attempt's partial entries
            del accuracy[marks[0]:], performance_l1[marks[1]:]

        if phases.run("candidate", lambda: _candidate(environment, model, l1, timed_suites, reference_perf, accuracy,
                                                      performance_l1, out, absolute_runs) or True,
                      retries=retries(environment), reset=undo):
            phases.errors.pop("candidate", None)
        l2 = model["performance"].get("l2")
        performance_l2: dict[str, Any] = {}
        # The opt-in sweeps start the native adapter: only where L1 could time it.
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
        measured = {(item["reference_mode"], item.get("request")) for item in performance_l1}
        performance_l1 += [unavailable_mode(mode, phases.errors.get(f"reference_perf_{mode}", "not measured"), suite.name)
                           for mode in l1["reference_modes"] for suite in timed_suites
                           if (mode, suite.name) not in measured]
    if model.get("accuracy_source") == "none":
        accuracy.extend(conversion_parity(performance_l1))
    mark_informational(model, accuracy)

    result = {"model": model["model"], "operation": model["operation"], "task": model.get("task"),
              "repro": f"trtmc-aiperf-qual run --profile {model['catalog_profile']} --environment "
                       f"{environment.values.get('environment_file', '<environment.yaml>')} --out {out}",
              "family": model.get("family"), "started": started,
              "platform": {"id": platform_id(fingerprint), **fingerprint},
              "accuracy_source": model.get("accuracy_source"),
              **({"accuracy_note": model["accuracy_note"]} if model.get("accuracy_note") else {}),
              **({"coverage": model["coverage"]} if model.get("coverage") else {}),
              "reference": {key: reference.get(key) for key in ("backend", "precision", "perf_precision",
                                                                 "timing_precision")},
              "reference_python": python, "duration_s": time.time() - started, "accuracy": accuracy,
              "performance_l1": performance_l1, "performance_l2": performance_l2,
              "errors": phases.errors,
              "provenance": {"aiperf": importlib.metadata.version("aiperf"),
                             "plugins": importlib.metadata.version("trtmc-aiperf-plugins"),
                             "perf_suite": perf_suite.manifest if perf_suite else None,
                             "timed_requests": [suite.manifest for suite in timed_suites]}}
    result["accuracy"] += missing_results(model, result["accuracy"], phases.errors)
    result["verdict"] = judge.verdict(result, expected_suites=list(expected_suites(model)),
                                      expected_modes=len(timed_suites) if l1 else 0)
    result["mode"] = "smoke" if smoke else "formal"
    if smoke:
        result["verdict"] = smoke_verdict(result)
    write_report(out, result)
    return result
