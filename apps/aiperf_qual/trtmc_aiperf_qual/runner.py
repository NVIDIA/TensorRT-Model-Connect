# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify one model against its native reference.

The required quality datasets run once per side. Their responses feed both
quality scorers and task-call timings. Perf-only models retain their configured
request. Failures remain explicit results rather than additional fallback work.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import absolute, aiperf_metrics, compat, edits, execution, geneval, intelligibility, judge, replay_parity, sweep, world_model
from .aiperf_runner import AiperfRun, run_aiperf
from .config import Environment
from .report import write_report
from .services import gpu_exclusive, gpu_identity, platform_fingerprint, platform_id, reference_python, serving
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
# TRTMC tokenizes the passage itself (it may count a few more tokens than the Hugging Face tokenizer) and a
# bundle's prefill profile may be shorter than its sequence length: a prompt TRTMC rejects for length is
# shortened to the longest it accepts (a binary search over the passage length).


def timed_request(model: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """The request Perf times: a sampling text request as its greedy variant, so both sides generate the
    same tokens; every other request as is."""
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
            if "backend_rejected_request" not in message or not any(word in message for word in absolute.CAPACITY_WORDS):
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


MAX_RUN_REQUESTS = 2000  # a run of the shortest requests: enough for a stable median, bounded in time


def requests_per_run(measurement: Mapping[str, Any], settled: int, settle_s: float) -> int:
    """The configured requests per run, or as many as the settle's pace (``settled`` requests in ``settle_s``)
    fits in ``min_run_s``, up to MAX_RUN_REQUESTS."""
    requests = int(measurement["requests"])
    if settled and settle_s > 0 and measurement.get("min_run_s"):
        fitting = math.ceil(float(measurement["min_run_s"]) * settled / settle_s)
        requests = max(requests, min(MAX_RUN_REQUESTS, fitting))
    return requests


def _audio_seconds(observation: Mapping[str, Any] | None) -> float | None:
    seconds = ((observation or {}).get("audio_digest") or {}).get("seconds")
    return float(seconds) if seconds else None


def settle(service: Mapping[str, Any], operation: str, request: Mapping[str, Any], seconds: float,
           timeout_s: float) -> int:
    """The timed request sent back to back for ``seconds`` before a side's first timed run (a
    fresh server times short requests slower for its first seconds), each within the run deadline ``timeout_s``;
    returns how many were sent."""
    deadline, sent = time.monotonic() + seconds, 0
    while time.monotonic() < deadline:
        absolute._probe(service, operation, request, timeout_s)  # cancellable; a rejected or late request raises
        sent += 1
    return sent


def candidate_probe(environment: Environment, model: Mapping[str, Any], suite: Suite | None, out: Path,
                    phases: "_Phases", *, serviceability: bool, policy: bool) -> list[Suite]:
    """One TRTMC server before any native work: it must serve the first request (when Acc runs) and sizes the
    near-capacity request; returns the timed requests (none without performance or when they cannot be built)."""
    with serving(environment, dict(model), "trtmc", out / "absolute-probe") as service:
        if serviceability:
            phases.run("absolute_probe", lambda: probe(service, model["operation"],
                                                        suite.samples[0]["request"] if suite else {}))
        if not policy or suite is None:
            return []
        return phases.run("perf_requests", lambda: perf_suites(environment, model, suite, service)) or []


def _task_url(service: Mapping[str, Any], operation: str) -> list[str]:
    return ["--url", f"{service['url']}/v1/tasks/{operation}"]


def _observations(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any],
                  suite: Suite, out: Path) -> dict[str, Any]:
    units = [sample.get("cluster", sample.get("series", index)) for index, sample in enumerate(suite.samples)]
    with execution.workload(suite.name, "both", units=units):
        return _collect_outputs(environment, service, model, suite, out)


def _collect_outputs(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any],
                  suite: Suite, out: Path) -> dict[str, Any]:
    """One observation per suite sample from a server (sequential, one request each)."""
    for sample in suite.samples:
        if missing := unstated_defaults(sample["request"]):
            raise ValueError(f"{suite.name}: state {', '.join(missing)} explicitly before benchmarking")
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
    with execution.workload(suite.name, "performance"):
        return _measure_workload(environment, service, model, suite, measurement, out, aggregation)


def _measure_workload(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any], suite: Suite,
              measurement: Mapping[str, Any], out: Path, aggregation: str) -> tuple[AiperfRun, dict[str, Any]]:
    """Repeated timed runs of the suite's request. Each run's statistic is the median server model-call
    time (with ``per_audio_second``, per second of generated audio: a sampling speech model's outputs differ in
    length); every response must carry its time, and every response's work signature is kept. A request shorter
    than the settle's pace gets enough requests per run to last ``min_run_s``, so a run's median is not noise."""
    settle_s = float(measurement.get("settle_s", 0))
    settled = (settle(service, model["operation"], suite.samples[0]["request"], settle_s,
                      absolute.run_timeout(environment, model)) if settle_s > 0 else 0)
    requests = requests_per_run(measurement, settled, settle_s)
    per_audio_second = bool(measurement.get("per_audio_second"))
    arguments = [*TASK_ENDPOINT, *_task_url(service, model["operation"]), "--concurrency", "1",
                 "--input-file", str(suite.write_inputs(out.parent / f"{out.name}.inputs.jsonl")),
                 "--custom-dataset-type", "single_turn", "--dataset-sampling-strategy", "sequential",
                 "--request-count", str(requests)]
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
        if per_audio_second:
            responses = [(None if ms is None or not _audio_seconds(obs) else ms / _audio_seconds(obs), obs)
                         for ms, obs in responses]
        times = [ms for ms, _ in responses if ms is not None]
        untimed += len(responses) - len(times)
        if not times:
            per_run.append(None)
            break  # nothing succeeded; further runs cannot either
        per_run.append(statistics.median(times))
        # Per audio second, the work is the request itself (its generated length is normalized out).
        work += [() if per_audio_second else judge.work_signature(model["operation"], obs) for _, obs in responses]
    stats = judge.across_runs(per_run, aggregation)
    stats["work"] = list(dict.fromkeys(signature for signature in work if signature is not None))  # distinct
    stats["work_missing"] = sum(signature is None for signature in work)
    if model.get("service_metrics"):
        stats["client_latency_p50_ms"] = judge.median_client_latency(runs[-1].raw_records())
    stats["aiperf_exit"] = max(run.exit_code for run in runs)
    if settled:
        stats["settle_requests"] = settled
    stats["requests_per_run"] = requests
    if per_audio_second:
        stats["unit"] = "ms per audio second"
    expected_runs, problems = int(measurement.get("runs", 1)), []
    if len(runs) < expected_runs:
        problems.append(f"{len(runs)} of {expected_runs} runs completed")
    if untimed:
        problems.append(f"{untimed} successful responses carry no model-call time")
    for run in runs:
        problem = run_completeness(run, requests)
        if problem:
            problems.append(f"{getattr(getattr(run, 'directory', None), 'name', 'run')}: {problem}")
    if problems:
        stats["incomplete"] = "; ".join(problems)[:600]
    elif stats["aiperf_exit"]:  # every request succeeded: a late telemetry/export failure only
        stats["exit_note"] = f"AIPerf exited {stats['aiperf_exit']} after all requests succeeded"
    measured = [value for value in busy if value is not None]
    if measured:
        stats["gpu_busy_percent"] = max(measured)
    if len(measured) < len(busy):  # no reading is no evidence of an idle GPU
        stats["gpu_unmeasured_runs"] = len(busy) - len(measured)
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
    """A Perf-only model's Acc evidence: on every timed request TRTMC's output passes the
    Task's output check against the native eager output (text: the first 8 greedy token ids, or the whole
    text, are equal); a mismatch is a ``fail``. Empty when no eager comparison ran (``missing_results``
    then reports it)."""
    checks = [(item.get("request"), item.get("output_check") or {}) for item in performance
              if item.get("gate", True) and item.get("reference_mode") == "eager" and "match" in (item.get("output_check") or {})]
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


def gpu_busy_percent(samples: int = 30, interval_s: float = 0.5, settle_s: float = 1.0) -> float | None:
    """Other processes' GPU load just before a timed run, while our servers are idle.

    nvidia-smi averages utilization over its last sample period, so a reading right after our own
    request (seconds long for diffusion) still shows that request. The lowest of a few readings taken
    after a short settle is what persists without us.
    """
    import os
    import subprocess

    from . import cancel

    time.sleep(settle_s)
    readings = []
    selector = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    for index in range(samples):
        cancel.check()
        try:
            completed = subprocess.run(["nvidia-smi", *(["-i", selector] if selector else []),
                                       "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5, check=True)
            values = [float(line) for line in completed.stdout.split() if line.strip().replace(".", "").isdigit()]
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        if values:
            readings.append(max(values))
            if readings[-1] < judge.GPU_BUSY_PERCENT:
                return readings[-1]
        if index + 1 < samples:
            time.sleep(interval_s)
    return min(readings) if readings else None


def output_check(policy: Mapping[str, Any], candidate: Any, references: Mapping[str, Any], mode: str,
                 sampled: bool = False) -> tuple[bool, str]:
    """Perf output sanity check against the mode's reference output; a compiled reference whose own
    numerics diverge is not held against the candidate when the eager reference output agrees."""
    compare = COMPARATORS[policy["output_grader"]]

    def check(reference: Any) -> tuple[bool, str]:
        try:
            params = dict(policy.get("output_grader_params", {}))
            if sampled and policy["output_grader"] == "parity_token_exact":
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
    ``retries`` runs again after a failure (one retry of a phase that fails before
    producing its result); ``reset`` undoes a failed attempt's partial results first."""

    def __init__(self, out: Path) -> None:
        self.errors: dict[str, str] = {}
        self.out = out

    def run(self, name: str, call, retries: int = 0, reset=None):
        for attempt in range(retries + 1):
            evidence_start = execution.checkpoint()
            try:
                return call()
            except Exception as error:  # noqa: BLE001 - reported per phase
                execution.supersede(evidence_start)
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


def _reference_perf(environment: Environment, model: Mapping[str, Any], policy: Mapping[str, Any], suites: Sequence[Suite],
                    python: str, phases: _Phases, out: Path) -> dict[str, dict[str, tuple[AiperfRun, dict, dict]]]:
    """Time the native model per reference mode and timed request, at the first precision it runs at."""
    results: dict[str, dict[str, tuple[AiperfRun, dict, dict]]] = {}
    for mode in policy["reference_modes"]:
        def measure(mode: str = mode) -> None:
            errors = []
            for precision in timing_precisions(model["reference"]):
                try:
                    results[mode] = _time_reference(environment, model, policy, suites, python, mode, precision, out)
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


def _time_reference(environment: Environment, model: Mapping[str, Any], policy: Mapping[str, Any], suites: Sequence[Suite],
                    python: str, mode: str, precision: str,
                    out: Path) -> dict[str, tuple[AiperfRun, dict[str, Any], dict[str, Any]]]:
    tag = f"{mode}-{precision}"
    timed = {}
    with serving(environment, model, "reference", out / f"reference-{tag}", mode=mode, precision=precision,
                 python=python) as service:
        for suite in suites:
            probe(service, model["operation"], suite.samples[0]["request"])
            run, stats = _perf_run(environment, service, model, suite, policy["measurement"],
                                   out / f"perf-reference-{tag}-{suite.name}", policy["aggregation"].get(mode, "mean"))
            if stats.get("p50_ms") is None:
                raise RuntimeError(f"no successful {precision} requests ({suite.name})")
            stats["precision"] = precision
            timed[suite.name] = (run, stats, service["info"])
    return timed


def _candidate(environment: Environment, model: Mapping[str, Any], policy: Mapping[str, Any] | None,
               suites: Sequence[Suite], reference_perf: Mapping[str, Mapping[str, tuple]], accuracy: list,
               performance: list, out: Path, absolute_runs: Mapping[str, Any] | None = None) -> None:
    """Run candidate workloads on one server; both consumers use its response records."""
    if absolute_runs and absolute_runs.get("answered"):
        raise ValueError("overlapped answers cannot enter the isolated qualification executor")
    keep = absolute.keeps_artifacts(model)
    with serving(environment, model, "trtmc", out / "candidate", keep_artifacts=keep) as service:
        if absolute_runs:
            accuracy.extend(absolute.candidate_entries(environment, service, model, out, **absolute_runs))
        if not policy:
            return
        for suite in suites:
            sampled = sampled_request(suite.samples[0]["request"])
            run, stats = _perf_run(environment, service, model, suite, policy["measurement"],
                                   out / f"perf-candidate-{suite.name}", policy["aggregation"].get("candidate", "mean"))
            candidate_output = judge.first_observation(run.raw_records())
            timed = {mode: by_suite[suite.name] for mode, by_suite in reference_perf.items() if suite.name in by_suite}
            outputs = {mode: judge.first_observation(entry[0].raw_records()) for mode, entry in timed.items()}
            for mode, (reference_run, reference_stats, info) in timed.items():
                match, reason = output_check(policy, candidate_output, outputs, mode, sampled)
                verdict = judge.judge_performance(stats, reference_stats, margin_percent=float(policy["margin_percent"]),
                                                  max_ci_percent=float(policy["max_ci_percent"]),
                                                  guard_percent=float(policy.get("guard_percent", 0)),
                                                  outputs_match=match, output_reason=reason,
                                                  not_equivalent=policy.get("not_equivalent"),
                                                  candidate_precision=model["reference"].get("perf_precision"))
                if (model["reference"].get("options") or {}).get("cpu_offload"):
                    verdict["notes"].append("the native pipeline offloads its weights to host memory (larger than the GPU)")
                performance.append({"reference_mode": mode, "request": suite.name,
                                    "candidate_timing_scope": service["info"].get("timing_scope"),
                                    "reference_timing_scope": info.get("timing_scope"),
                                    "reference_backend": info.get("backend"), **verdict})




def order_check(environment: Environment, model: Mapping[str, Any], out: Path) -> dict[str, Any]:
    """The order check: the model's performance requests timed twice in both orders (native then TRTMC, TRTMC then native),
    against native eager at its timing precision. Each side's order effect is its p50 when timed second relative
    to its p50 when timed first (native after TRTMC vs native first; TRTMC after native vs TRTMC first); a
    request's speedup effect compounds both sides' ratios, each in its worse direction, and ``largest`` is the
    largest of them, compared with the Perf guard. The
    effect is resolved only when all four measurements are valid (``judge.measurement_problems``) and each
    order's two sides did the same work; otherwise, or when a measurement or the check itself fails, the
    check is ``unresolved``, never within the limit. The limit is the Perf guard (``guard_percent``): an effect
    above it means the guard is too narrow for this host. A previous ``order.json`` is removed first."""
    order = out / "order.json"
    order.unlink(missing_ok=True)
    result: dict[str, Any] = {"model": model["model"], "status": "unresolved", "order_effect": None, "largest": None,
                              "above_limit": None, "problems": []}
    try:
        result |= _order_timings(environment, model, out)
    except Exception as error:  # noqa: BLE001 - recorded as unresolved
        result["problems"] = [f"order check failed: {type(error).__name__}: {str(error)[-400:]}"]
    order.write_text(json.dumps(result, indent=2, default=str))
    return result


def _order_timings(environment: Environment, model: Mapping[str, Any], out: Path) -> dict[str, Any]:
    policy = model["performance"]
    max_ci = float(policy["max_ci_percent"])
    python = reference_python(environment, model)
    perf_suite = build_suite(policy["suite"], environment)
    with serving(environment, dict(model), "trtmc", out / "order-probe") as service:
        suites = perf_suites(environment, model, perf_suite, service)
    precision = timing_precisions(model["reference"])[0]

    def native(tag: str) -> dict[str, dict[str, Any]]:
        timed = _time_reference(environment, model, policy, suites, python, "eager", precision, out / tag)
        return {name: stats for name, (_, stats, _) in timed.items()}

    def candidate(tag: str) -> dict[str, dict[str, Any]]:
        with serving(environment, dict(model), "trtmc", out / tag) as service:
            return {suite.name: _perf_run(environment, service, model, suite, policy["measurement"],
                                          out / f"{tag}-{suite.name}", "mean")[1] for suite in suites}

    def measured(timer: Any, tag: str) -> dict[str, dict[str, Any]]:
        """One measurement's stats per request; a failed measurement is kept as each request's failure."""
        try:
            return timer(tag)
        except Exception as error:  # noqa: BLE001 - an invalid measurement, not a crash
            failure = f"measurement failed: {type(error).__name__}: {str(error)[-300:]}"
            return {suite.name: {"p50_ms": None, "incomplete": failure} for suite in suites}

    with gpu_exclusive(environment):
        timings = {"native_first": measured(native, "order-native-first"),
                   "trtmc_second": measured(candidate, "order-trtmc-second")}
        timings |= {"trtmc_first": measured(candidate, "order-trtmc-first"),
                    "native_second": measured(native, "order-native-second")}
    problems = [f"{name} {suite.name}: {problem}" for name, by_suite in timings.items() for suite in suites
                for problem in judge.measurement_problems(by_suite.get(suite.name) or {}, max_ci)]
    for first, second in (("native_first", "trtmc_second"), ("trtmc_first", "native_second")):
        problems += [f"{first}/{second} {suite.name}: {reason}" for suite in suites
                     if (reason := judge.work_check(timings[first].get(suite.name) or {},
                                                    timings[second].get(suite.name) or {}))]
    p50 = {name: {suite: stats.get("p50_ms") for suite, stats in by_suite.items()} for name, by_suite in timings.items()}
    result = {"precision": precision, "measurement": policy["measurement"], "p50_ms": p50, "measurements": timings,
              "problems": problems}
    if not problems:
        effects = {side: {suite.name: p50[f"{side}_second"][suite.name] / p50[f"{side}_first"][suite.name] - 1
                          for suite in suites} for side in ("native", "trtmc")}
        # What the effects can do to a speedup: each side's ratio in its worse direction, the two compounded.
        speedup = {suite.name: math.prod(max(1 + effects[side][suite.name], 1 / (1 + effects[side][suite.name]))
                                         for side in effects) - 1 for suite in suites}
        largest = max(speedup.values())
        limit = float(policy.get("guard_percent", 0)) / 100
        result |= {"order_effect": effects, "speedup_effect": speedup, "largest": largest, "limit": limit,
                   "above_limit": largest > limit, "status": "above-limit" if largest > limit else "within-limit"}
    return result


def _loaded(out: Path) -> list[str]:
    """The libraries TRTMC's servers mapped in this run (each server directory's record), all of them."""
    from .services import LOADED_LIBRARIES

    return sorted({path for record in out.glob(f"*/{LOADED_LIBRARIES}") for path in json.loads(record.read_text())})


def _bundle_identity(out: Path) -> dict[str, Any] | None:
    """The build record's identity of the bundle that was qualified (build.json, written by the campaign)."""
    path = out / "build.json"
    if not path.is_file():
        return None
    build = json.loads(path.read_text())
    return {key: build.get(key) for key in ("status", "bundle", "bundle_bytes", "bundle_sha256", "receipt_sha256")}


SMOKE_MEASUREMENT = {"warmup": 0, "requests": 1, "runs": 1}
PHASE_RETRIES = 1  # a GPU phase that fails before producing its result runs once more


def retries(environment: Environment) -> int:
    """Retries of a failed GPU phase: none in smoke mode (it is there to find failures fast)."""
    return 0 if environment.values.get("smoke") else PHASE_RETRIES


def smoke_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """The model as smoke mode runs it: performance with one request against native eager, no opt-in sweeps."""
    policy = model.get("performance")
    performance = {**policy, "measurement": dict(SMOKE_MEASUREMENT), "reference_modes": ["eager"]} if policy else {}
    return {**model, "performance": performance}


def smoke_verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """smoke-pass when every phase ran and produced its results (whatever they say), else smoke-fail
    with the failing phases."""
    failing = [f"phase {name}" for name in result.get("errors") or {}]
    failing += [f"{item['suite']}: {item.get('error') or '; '.join(item.get('reasons', []))}"[:300]
                for item in result.get("accuracy", []) if item.get("status") == "error"]
    failing += [f"perf {item['reference_mode']}: {'; '.join(item.get('reasons', []))}"[:300]
                for item in result.get("performance", []) if item.get("gate", True) and item.get("light") in ("error", "n/a")]
    # A missing required performance or quality result is also an error in the model-level verdict.
    verdict = result.get("verdict") or {}
    failing += [f"verdict {part}: error" for part in ("acc", "perf") if verdict.get(part) == "error" and not failing]
    if result.get("reference", {}).get("backend") == "unsupported":
        failing.append("no native adapter")
    return {**result.get("verdict", {}), "category": "smoke-fail" if failing else "smoke-pass", "failing": failing}


def quality_only(model: Mapping[str, Any]) -> bool:
    """Required quality workloads supply both scores and timings; no extra performance requests."""
    return bool(model.get("absolute") or any(
        not check.get("informational") and check.get("check") in SUPPLEMENTARY_CHECKS and applies(check, model)
        for check in model.get("supplementary", [])))


def qualify(model: dict[str, Any], environment: Environment, out: Path) -> dict[str, Any]:
    """Run quality workloads once per side and reuse their outputs for performance."""
    model = compat.configuration(model)
    environment = Environment({**environment.values, "native_replicas": 1, "candidate_replicas": 1,
                               "acc_mps": False, "acc_overlap": False})
    out.mkdir(parents=True, exist_ok=True)
    (out / "execution.jsonl").write_text("")
    evidence = execution.Session(
        out, model.get("performance", {}).get("measurement", {}), gpu_busy_percent,
        lambda observation: judge.work_signature(model["operation"], observation),
        lambda mine, theirs: judge.work_check({"work": [mine] if mine is not None else []},
                                             {"work": [theirs] if theirs is not None else []}) is None,
        smoke=bool(environment.values.get("smoke")),
        request_problems=lambda payload: unstated_defaults(payload.get("request") or {}))
    with execution.session(evidence):
        result = _qualify(model, environment, out)
    result["schema_version"] = execution.SCHEMA
    result["performance"].extend(evidence.natural_performance(result["accuracy"]))
    if quality_only(model):
        result["performance_source"] = "quality"
        result["performance_expected"] = [item["suite"] for item in model.get("absolute", [])]
        result["verdict"] = judge.verdict(result, expected_suites=list(expected_suites(model)), expected_modes=0)
        if environment.values.get("smoke"):
            result["verdict"] = smoke_verdict(result)
    result["aiperf_metrics"] = aiperf_metrics.entries(evidence.batches)
    for item in result["accuracy"]:
        item.pop("workload_perf", None)
    result["execution"] = {"records": str(out / "execution.jsonl"), "timing_contract": execution.TIMING_CONTRACT,
                           "device_layout": "single-serial", "replicas": 1, "concurrency": 1,
                           "quality_outputs_reused": True, "workloads": [
                               {"name": batch["workload"], "role": batch["role"]} for batch in evidence.batches]}
    write_report(out, result)
    return result


def _qualify(model: dict[str, Any], environment: Environment, out: Path) -> dict[str, Any]:
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    phases = _Phases(out)
    smoke = bool(environment.values.get("smoke"))
    if smoke:
        model = smoke_model(model)
    dataset_only = quality_only(model)
    policy = None if dataset_only else model.get("performance")
    perf_suite = build_suite(policy["suite"], environment) if policy else None
    (out / "suites").mkdir(exist_ok=True)
    if perf_suite:
        (out / "suites" / f"{perf_suite.name}.manifest.json").write_text(json.dumps(perf_suite.manifest, indent=2))
    (out / "model.json").write_text(json.dumps(model, indent=2, default=str))

    reference = model["reference"]
    if reference["backend"] == "unsupported":  # no native path: nothing to compare against
        result = {"model": model["model"], "operation": model["operation"], "task": model.get("task"),
                  "family": model.get("family"), "started": started, "reference": {"backend": "unsupported"},
                  "accuracy": [], "performance": [], "service_metrics": {}, "duration_s": time.time() - started,
                  "errors": {"native": reference.get("not_covered") or
                             f"no native adapter serves {model['operation']!r} for Task {model.get('task')!r}"},
                  "provenance": {"aiperf": importlib.metadata.version("aiperf"),
                                 "plugins": importlib.metadata.version("trtmc-aiperf-plugins"), "perf_suite": None}}
        result["verdict"] = judge.verdict(result, expected_suites=[], expected_modes=0)
        result["mode"] = "smoke" if smoke else "formal"
        if smoke:
            result["verdict"] = smoke_verdict(result)
        return result
    python = reference_python(environment, model)
    fingerprint = platform_fingerprint(environment, python)["fingerprint"]
    from . import gold_metrics

    gold_metrics.protect([out.parent, *(environment.values.get(key) for key in (
        "bundle_root", "hf_hub_cache", "hf_datasets_cache", "data_root", "reference_env_root", "runtime_root"))])
    accuracy: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    unstated = unstated_defaults(perf_suite.samples[0]["request"]) if perf_suite else []
    if unstated:  # each side would apply its own default: different work
        phases.errors["perf_request"] = (f"the timed request leaves {', '.join(unstated)} to each side's default; "
                                         "state them in config/models performance")
        policy = None
    # Absolute accuracy: the problems both sides answer (selected and length-checked outside the GPU lock).
    plans = phases.run("absolute_plan", lambda: {item["suite"]: absolute.plan(environment, model, item)
                                                 for item in model["absolute"]}) if model.get("absolute") else None
    if model.get("absolute") and plans is None:
        accuracy.extend(absolute.error_entry(item, 0, f"problem selection: {phases.errors.get('absolute_plan')}")
                        for item in model["absolute"])
    for check in model.get("supplementary", []):
        if not check.get("informational") and check.get("check") in SUPPLEMENTARY_CHECKS and applies(check, model):
            def run_check(check: Mapping[str, Any] = check) -> None:
                with gpu_exclusive(environment):
                    accuracy.extend(supplementary(environment, model, check, python, out))
            phases.run(check["check"], run_check)
    timed_suites = (phases.run("perf_requests", lambda: [single_request_suite(
        perf_suite.name, timed_request(model, perf_suite.samples[0]["request"]), perf_suite.manifest)]) or []) if policy else []
    if policy and not timed_suites:
        policy = None  # the timed requests could not be built: the phase error says why
    with gpu_exclusive(environment):
        reference_perf = _reference_perf(environment, model, policy, timed_suites, python, phases, out) if policy else {}
        absolute_runs = None
        if plans:
            if "absolute_probe" in phases.errors:
                accuracy.extend(absolute.error_entry(item, len(plans[item["suite"]]),
                                                     f"TRTMC cannot serve the model: {phases.errors['absolute_probe'][:500]}")
                                for item in model["absolute"])
            else:
                probe = perf_suite.samples[0]["request"] if perf_suite else None
                native = phases.run("absolute_native", lambda: absolute.run_native_alone(
                    environment, model, python, plans, out, probe), retries=retries(environment))
                if native:
                    phases.errors.pop("absolute_native", None)
                absolute_runs = {"plans": plans, "native": native or {},
                                 "native_error": phases.errors.get("absolute_native")}
        marks = (len(accuracy), len(performance))

        def undo() -> None:  # a failed attempt's partial entries
            del accuracy[marks[0]:], performance[marks[1]:]

        if (not dataset_only or absolute_runs) and phases.run("candidate", lambda: _candidate(
                environment, model, policy, timed_suites, reference_perf, accuracy, performance, out, absolute_runs) or True,
                      retries=retries(environment), reset=undo):
            phases.errors.pop("candidate", None)
        service_config = model.get("service_metrics")
        service_metrics: dict[str, Any] = {}
        # The opt-in sweeps start the native adapter: only where performance could time it.
        generic = reference.get("backend") == "reference" if dataset_only else any(
            item.get("reference_backend") == "reference" for item in performance)
        if service_config and generic:
            with execution.workload("service-metrics", "service"):
                phases.run("service_metrics", lambda: service_metrics.update(sweep.run_serial(
                    environment, model, service_config, out, python, timing_precisions(reference)[0])))
    if performance:  # the candidate was measured: record modes whose native reference could not run
        measured = {(item["reference_mode"], item.get("request")) for item in performance}
        performance += [unavailable_mode(mode, phases.errors.get(f"reference_perf_{mode}", "not measured"), suite.name)
                           for mode in policy["reference_modes"] for suite in timed_suites
                           if (mode, suite.name) not in measured]
    if model.get("accuracy_source") == "none":
        accuracy.extend(conversion_parity(performance))
    mark_informational(model, accuracy)

    result = {"model": model["model"], "operation": model["operation"], "task": model.get("task"),
              "repro": f"trtmc-aiperf-qual run --profile {model['catalog_profile']} --environment "
                       f"{environment.values.get('environment_file', '<environment.yaml>')} --out {out}",
              "family": model.get("family"), "started": started,
              "platform": {"id": platform_id(fingerprint), **fingerprint},
              "host": {**gpu_identity(environment), "trtmc_libraries": _loaded(out)},
              "bundle": _bundle_identity(out),
              "accuracy_source": model.get("accuracy_source"),
              "performance_source": "quality" if dataset_only else "fixed",
              **({"accuracy_note": model["accuracy_note"]} if model.get("accuracy_note") else {}),
              **({"coverage": model["coverage"]} if model.get("coverage") else {}),
              "reference": {key: reference.get(key) for key in ("backend", "precision", "perf_precision",
                                                                 "timing_precision")},
              "reference_python": python, "duration_s": time.time() - started, "accuracy": accuracy,
              "performance": performance, "service_metrics": service_metrics,
              "errors": phases.errors,
              "provenance": {"aiperf": importlib.metadata.version("aiperf"),
                             "plugins": importlib.metadata.version("trtmc-aiperf-plugins"),
                             "perf_suite": perf_suite.manifest if perf_suite else None,
                             "timed_requests": [suite.manifest for suite in timed_suites]}}
    result["accuracy"] += missing_results(model, result["accuracy"], phases.errors)
    result["verdict"] = judge.verdict(result, expected_suites=list(expected_suites(model)),
                                      expected_modes=len(timed_suites) if policy else 0)
    result["mode"] = "smoke" if smoke else "formal"
    if smoke:
        result["verdict"] = smoke_verdict(result)
    return result
