# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Absolute accuracy: both sides scored against gold answers, then compared.

TRTMC and the native model (the generic adapter, eager, at the candidate precision) answer the same
problems, and each side is scored against the gold answers:

- ``plugin`` entries run an AIPerf accuracy benchmark (trtmc_aiperf_plugins.benchmarks: MMLU, GSM8K,
  MATH-500, LAMBADA at pinned revisions, greedy decoding) graded by AIPerf;
- ``metric`` entries send a suite with gold labels (images, audio, sentence pairs) through AIPerf's
  trtmc_task endpoint and score the outputs here (gold_metrics).

An entry passes when the two scores differ by at most ``max_delta_points`` (or ``max_relative`` of
the native score): "close enough" is a size, not a test. The paired McNemar test (right/wrong
metrics) or bootstrap interval (corpus metrics: WER, Spearman, MSE) is reported as a note. A sampled model answers once per seed on each side and is held to the
difference of the mean accuracies.
"""

from __future__ import annotations

import functools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import gold_metrics
from .aiperf_runner import run_aiperf
from .config import Environment
from .judge import light
from .services import serving_replicas
from .suites import build_suite, request_sha

ALPHA = 0.05
WORKLOAD_MARGIN_PERCENT = 5.0
# A request answers within seconds; whole-benchmark runs of large native models take hours.
RUN_TIMEOUT_S = 12 * 3600


@functools.lru_cache(maxsize=None)
def _has_tokenizer(name: str, revision: str | None, trust_remote_code: bool) -> bool:
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=trust_remote_code)
    except Exception:  # noqa: BLE001 - for example an adapter-only checkpoint (LoRA) without tokenizer files
        return False
    return True


def tokenizer_source(model: Mapping[str, Any]) -> tuple[str, str | None]:
    """The checkpoint whose tokenizer counts and formats the prompts: the candidate's, unless it carries
    none (an adapter on a base model), then the native reference model's (that base model)."""
    candidate = (str(model["candidate"]["checkpoint"]), model["candidate"].get("revision") or None)
    reference = model["reference"]
    if reference.get("model") and not _has_tokenizer(*candidate, bool(reference.get("trust_remote_code"))):
        return str(reference["model"]), reference.get("revision") or None
    return candidate


def selection_environment(environment: Environment, model: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, str]:
    """TRTMC_ACCURACY_* settings of the plugin's problem selection (identical for both sides)."""
    environ = {"HF_DATASETS_CACHE": str(environment["hf_datasets_cache"]),
               "TRTMC_ACCURACY_TOKEN_LIMIT": str(model["candidate"]["max_sequence_length"]),
               # A chat template wraps the prompt; plain completions add at most a BOS token.
               "TRTMC_ACCURACY_TEMPLATE_MARGIN": "128" if item.get("endpoint") == "chat" else "8",
               "TRTMC_ACCURACY_TOKENIZER": tokenizer_source(model)[0]}
    if tokenizer_source(model)[1]:
        environ["TRTMC_ACCURACY_TOKENIZER_REVISION"] = str(tokenizer_source(model)[1])
    if model["reference"].get("trust_remote_code"):
        environ["TRTMC_ACCURACY_TRUST_REMOTE_CODE"] = "1"
    for key, name in (("per_task", "TRTMC_ACCURACY_PER_TASK"), ("limit", "TRTMC_ACCURACY_LIMIT"),
                      ("max_new_tokens", "TRTMC_ACCURACY_MAX_NEW_TOKENS")):
        if item.get(key):
            environ[name] = str(item[key])
    return environ


def plan(environment: Environment, model: Mapping[str, Any], item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The problems a run sends, in request order: their task and gold answer (and the request)."""
    if item.get("metric"):
        suite = build_suite(item["suite_definition"], environment)
        samples = suite.samples
        if item.get("truncate_tokens"):  # the same text on both sides, whatever each side's own cut
            samples = _head_truncated(model, samples, int(item["truncate_tokens"]))
        return [{"task": sample.get("task", suite.name), "gold": sample.get("label"), "sample_id": sample["sample_id"],
                 "request": sample["request"], "request_sha": sample["request_sha"]} for sample in samples]
    from trtmc_aiperf_plugins.benchmarks import problems

    chosen = problems(item["plugin"], item.get("tasks"), int(item.get("n_shots", 0)),
                      selection_environment(environment, model, item))
    return [{"task": problem.task, "gold": problem.ground_truth} for problem in chosen]


def _head_truncated(model: Mapping[str, Any], samples: Sequence[Mapping[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Samples whose ``prompt`` keeps only its first ``limit`` tokens (the model's tokenizer)."""
    from transformers import AutoTokenizer

    name, revision = tokenizer_source(model)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision,
                                              trust_remote_code=bool(model["reference"].get("trust_remote_code")))
    result = []
    for sample in samples:
        request = dict(sample["request"])
        if isinstance(request.get("prompt"), str):
            ids = tokenizer(request["prompt"], add_special_tokens=False)["input_ids"]
            if len(ids) > limit:
                request["prompt"] = tokenizer.decode(ids[:limit], skip_special_tokens=True)
        result.append({**sample, "request": request, "request_sha": request_sha(request)})
    return result


def _targets(service: Mapping[str, Any], path: str = "") -> list[str]:
    """AIPerf's --url of every copy of the server, one request in flight per copy (round robin)."""
    urls = list(service.get("urls") or [service["url"]])
    return [flag for url in urls for flag in ("--url", f"{url}{path}")] + ["--concurrency", str(len(urls))]


def _arguments(model: Mapping[str, Any], item: Mapping[str, Any], service: Mapping[str, Any], count: int,
               seed: int | None) -> list[str]:
    arguments = ["--endpoint-type", item["endpoint"], *_targets(service),
                 "--tokenizer", tokenizer_source(model)[0],
                 "--accuracy-benchmark", item["plugin"], "--accuracy-n-shots", str(int(item.get("n_shots", 0))),
                 "--request-count", str(count)]
    if tokenizer_source(model)[1]:
        arguments += ["--tokenizer-revision", str(tokenizer_source(model)[1])]
    if model["reference"].get("trust_remote_code"):
        arguments.append("--tokenizer-trust-remote-code")
    if item.get("tasks"):
        arguments += ["--accuracy-tasks", *item["tasks"]]
    # Greedy unless the model's catalog request samples; a seed then makes each repetition distinct.
    if seed is None:
        arguments += ["--extra-inputs", "temperature:0"]
    else:
        arguments += ["--extra-inputs", f"seed:{seed}"]
    return arguments


def _suite_side(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any],
                item: Mapping[str, Any], problems: Sequence[Mapping[str, Any]], out: Path) -> dict[str, Any]:
    """A gold suite through AIPerf's trtmc_task endpoint: the observation and timing of every problem."""
    out.parent.mkdir(parents=True, exist_ok=True)
    inputs = out.parent / f"{out.name}.inputs.jsonl"
    with open(inputs, "w") as handle:
        for problem in problems:
            handle.write(json.dumps({"text": json.dumps({"request": problem["request"]})}) + "\n")
    run = run_aiperf(environment, out, ["--endpoint-type", "trtmc_task",
                                        *_targets(service, f"/v1/tasks/{model['operation']}"), "--input-file", str(inputs), "--custom-dataset-type", "single_turn",
                                        "--dataset-sampling-strategy", "sequential",
                                        "--request-count", str(len(problems))], timeout_s=RUN_TIMEOUT_S)
    by_request, timing_by_request = {}, {}
    for record in run.raw_records():
        if record.get("status") != 200 or not record.get("responses"):
            continue
        body = json.loads(record["responses"][-1]["text"])
        key = request_sha(record["payload"]["request"])
        by_request[key] = body.get("trtmc_observation") or {}
        timing_by_request[key] = {"model_call_ms": float((body.get("trtmc_timing") or {}).get("model_call_ms", 0)),
                                  "completion_tokens": by_request[key].get("output_tokens")}
    observations = {index: by_request[problem["request_sha"]] for index, problem in enumerate(problems)
                    if problem["request_sha"] in by_request}
    found = {index: timing_by_request[problem["request_sha"]] for index, problem in enumerate(problems)
             if problem["request_sha"] in timing_by_request}
    return {"observations": {"greedy": observations}, "exit": {"greedy": run.exit_code}, "timings": {"greedy": found}}


def run_side(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any],
             item: Mapping[str, Any], problems: Sequence[Mapping[str, Any]], out: Path) -> dict[str, Any]:
    """Graded records per repetition ({seed or "greedy": {problem index: record}}) and the AIPerf exits."""
    if item.get("metric"):
        return _suite_side(environment, service, model, item, problems, out / item["suite"])
    count = len(problems)
    runs: dict[str, Any] = {"records": {}, "exit": {}, "timings": {}}
    for seed in item.get("seeds") or [None]:
        name = "greedy" if seed is None else f"seed{seed}"
        run = run_aiperf(environment, out / f"{item['suite']}-{name}", _arguments(model, item, service, count, seed),
                         env=selection_environment(environment, model, item), timeout_s=RUN_TIMEOUT_S)
        raw = run.raw_records()
        # AIPerf grades a failed request as an empty (wrong) answer: it is a missing answer instead.
        failed = {int(record["metadata"]["session_num"]) for record in raw if unanswered(record)}
        runs["records"][name] = {int(record["session_num"]): record for record in run.accuracy_records()
                                 if int(record["session_num"]) not in failed}
        runs["exit"][name] = run.exit_code
        runs["timings"][name] = timings(raw)
        if failed:
            runs.setdefault("failed", {})[name] = failed_reason(raw, len(failed))
    return runs


def unanswered(record: Mapping[str, Any]) -> bool:
    """A request without an answer (an HTTP or transport failure). An HTTP 200 with empty content (AIPerf's
    InvalidInferenceResultError) is an answer: AIPerf grades it, as a wrong one."""
    if record.get("status") != 200:
        return True
    error = record.get("error")
    return bool(error) and not (isinstance(error, Mapping) and error.get("type") == "InvalidInferenceResultError")


def failed_reason(raw_records: Sequence[Mapping[str, Any]], count: int) -> str:
    """How many requests failed, and the first error (e.g. the backend rejecting every request)."""
    first = next((record.get("error") for record in raw_records if unanswered(record)), None)
    message = (first or {}).get("message") if isinstance(first, Mapping) else first
    return f"{count} requests failed: {str(message)[:300]}"


def timings(raw_records: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, float]]:
    """Per problem: the server's model-call time and token counts (from the OpenAI response body)."""
    found = {}
    for record in raw_records:
        for response in record.get("responses") or []:
            try:
                body = json.loads(response.get("text") or "")
            except (TypeError, ValueError):
                continue
            timing, usage = body.get("trtmc_timing") or {}, body.get("usage") or {}
            if timing.get("model_call_ms") is not None:
                found[int(record["metadata"]["session_num"])] = {
                    "model_call_ms": float(timing["model_call_ms"]), "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens")}
    return found


def workload_perf(candidate: Mapping[int, Mapping[str, Any]], native: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Model-call time on the benchmark's own requests, over the problems both sides answered with the
    same number of tokens (informational: the Perf L1 gate times the family's request)."""
    pairs = [(candidate[index], native[index]) for index in sorted(set(candidate) & set(native))
             if candidate[index].get("completion_tokens") == native[index].get("completion_tokens")]
    if not pairs:
        return {"pairs": 0}
    mine = statistics.median(item["model_call_ms"] for item, _ in pairs)
    theirs = statistics.median(item["model_call_ms"] for _, item in pairs)
    # The TRTMC backend does not count prompt tokens (0): the native side's count.
    prompts = [theirs_item.get("prompt_tokens") or item.get("prompt_tokens") for item, theirs_item in pairs]
    prompts = [value for value in prompts if value]
    return {"pairs": len(pairs), "trtmc_p50_ms": round(mine, 2), "native_p50_ms": round(theirs, 2),
            "speedup": round(theirs / mine, 3) if mine else None,
            "light": light(mine, theirs, WORKLOAD_MARGIN_PERCENT) if mine and theirs else "white",
            "prompt_tokens_p50": statistics.median(prompts) if prompts else None}


def mcnemar_worse_p(trtmc_only: int, native_only: int) -> float:
    """One-sided exact McNemar p-value that TRTMC is worse: P(X >= native_only), X ~ Bin(n, 1/2)."""
    n = trtmc_only + native_only
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, k) for k in range(native_only, n + 1)) / 2 ** n)


def _correct(record: Mapping[str, Any] | None) -> bool | None:
    return None if record is None else bool(record.get("passed"))


def _paired(problems: Sequence[Mapping[str, Any]], candidate: Mapping[int, Any],
            native: Mapping[int, Any]) -> dict[str, Any]:
    counts = Counter()
    per_task: dict[str, Counter] = defaultdict(Counter)
    examples = []
    for index, problem in enumerate(problems):
        mine, theirs = _correct(candidate.get(index)), _correct(native.get(index))
        if mine is None or theirs is None:
            counts["missing_trtmc" if mine is None else "missing_native"] += 1
            continue
        key = {(True, True): "both_correct", (True, False): "trtmc_only", (False, True): "native_only",
               (False, False): "both_wrong"}[(mine, theirs)]
        counts[key] += 1
        per_task[problem["task"]][key] += 1
        counts["trtmc_unparsed"] += bool(candidate[index].get("unparsed"))
        counts["native_unparsed"] += bool(native[index].get("unparsed"))
        if key == "native_only" and len(examples) < 5:
            examples.append({"sample_id": f"{problem['task']}/{index}",
                             "explanation": f"native correct, TRTMC wrong (gold {str(problem['gold'])[:80]!r})",
                             "actual": str(candidate[index].get("actual"))[:200],
                             "expected": str(native[index].get("actual"))[:200]})
    return {"counts": dict(counts), "per_task": per_task, "examples": examples}


def status(metrics: Mapping[str, Any], gate: Mapping[str, Any], *, expected: int,
           paired: int) -> tuple[str, list[str]]:
    """pass / fail / error of a scored entry under ``gate`` (also used when re-judging a report): only
    the size of the difference decides ("close enough"); a statistically significant shift within
    the gate is reported as a note."""
    if paired < expected:
        return "error", [f"{expected - paired} of {expected} problems lack a graded answer on one side"
                         + (f" ({metrics['failed']})" if metrics.get("failed") else "")]
    if "native_accuracy" in metrics and not metrics["native_accuracy"]:
        # Not one right answer from the native model: the benchmark's prompt or answer format does not
        # fit the model (for example a reasoning model thinking aloud), so it says nothing about TRTMC.
        return "not-comparable", ["the native model answers no problem correctly: the benchmark's prompt "
                                  "or answer format does not fit this model"]
    limit = float(gate.get("max_delta_points", 1.0))
    if gate.get("max_relative") and metrics.get("native_score"):  # e.g. WER: 0.2 points or 3% of native
        limit = max(limit, float(gate["max_relative"]) * abs(float(metrics["native_score"])))
    if abs(metrics["delta_points"]) > limit:
        return "fail", [f"differs by {metrics['delta_points']:+.3f} points (gate {limit:.3g})"]
    return "pass", []


def notes(metrics: Mapping[str, Any]) -> list[str]:
    if metrics.get("mcnemar_p_worse") is not None and metrics["mcnemar_p_worse"] < ALPHA:
        return [f"TRTMC significantly worse within the gate (McNemar p={metrics['mcnemar_p_worse']:.3g})"]
    if metrics.get("significantly_worse"):
        return [f"TRTMC significantly worse within the gate (bootstrap 95% interval {metrics.get('ci95')})"]
    return []


def _graded(item: Mapping[str, Any], problems: Sequence[Mapping[str, Any]], side: Mapping[str, Any]) -> dict[str, Any]:
    """A gold suite's observations graded right/wrong by a binary metric, as AIPerf records look."""
    grade = gold_metrics.BINARY[item["metric"]]
    records = {}
    for name, observations in side["observations"].items():
        records[name] = {}
        for index, observation in observations.items():
            correct, answer = grade(problems[index]["gold"], observation, problems[index].get("task"))
            records[name][index] = {"passed": correct, "unparsed": not answer, "actual": answer}
    return {**side, "records": records}


def judge_corpus(item: Mapping[str, Any], problems: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any],
                 native: Mapping[str, Any]) -> dict[str, Any]:
    """The entry of a corpus metric (WER, Spearman): both sides' statistic and the bootstrap interval."""
    expected = len(problems)
    mine, theirs = candidate["observations"]["greedy"], native["observations"]["greedy"]
    paired = len(set(mine) & set(theirs))
    entry: dict[str, Any] = {"suite": item["suite"], "source": "absolute", "benchmark": item["metric"],
                             "endpoint": "trtmc_task", "expected_samples": expected, "samples": paired, "passed": None,
                             "gate": dict(item["gate"]),
                             "aiperf_exit": {"trtmc": candidate["exit"], "native": native["exit"]}}
    if paired < expected:
        entry["status"], entry["reasons"] = "error", [f"{expected - paired} of {expected} problems lack an output on one side"]
        return entry
    comparison = gold_metrics.compare_corpus(item["metric"], problems, mine, theirs, item.get("metric_params"))
    entry["metrics"] = {"trtmc_score": comparison["trtmc"], "native_score": comparison["native"],
                        "delta_points": comparison["delta_points"], "ci95": comparison["ci95"],
                        "significantly_worse": comparison["significantly_worse"],
                        "higher_is_better": comparison["higher_is_better"], "units": comparison["units"]}
    entry["workload_perf"] = workload_perf(candidate["timings"]["greedy"], native["timings"]["greedy"])
    entry["status"], entry["reasons"] = status(entry["metrics"], entry["gate"], expected=expected, paired=paired)
    entry["notes"] = notes(entry["metrics"])
    return entry


def judge(item: Mapping[str, Any], problems: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any],
          native: Mapping[str, Any]) -> dict[str, Any]:
    """The entry of one benchmark from both sides' graded records."""
    if item.get("metric") in gold_metrics.CORPUS:
        return judge_corpus(item, problems, candidate, native)
    if item.get("metric"):
        candidate, native = _graded(item, problems, candidate), _graded(item, problems, native)
    expected = len(problems)
    entry: dict[str, Any] = {"suite": item["suite"], "source": "absolute",
                             "benchmark": item.get("plugin") or item["metric"],
                             "endpoint": item.get("endpoint", "trtmc_task"), "expected_samples": expected, "passed": None,
                             "gate": dict(item["gate"]), "aiperf_exit": {"trtmc": candidate["exit"], "native": native["exit"]}}
    repetitions = list(candidate["records"])
    paired = [_paired(problems, candidate["records"][name], native["records"].get(name, {})) for name in repetitions]
    accuracies = []
    for pair in paired:
        counts = pair["counts"]
        scored = sum(counts.get(key, 0) for key in ("both_correct", "trtmc_only", "native_only", "both_wrong"))
        accuracies.append(((counts.get("both_correct", 0) + counts.get("trtmc_only", 0)) / scored if scored else 0.0,
                           (counts.get("both_correct", 0) + counts.get("native_only", 0)) / scored if scored else 0.0,
                           scored))
    scored = min(count for _, _, count in accuracies)
    mine = sum(value for value, _, _ in accuracies) / len(accuracies)
    theirs = sum(value for _, value, _ in accuracies) / len(accuracies)
    metrics: dict[str, Any] = {"trtmc_accuracy": round(100 * mine, 2), "native_accuracy": round(100 * theirs, 2),
                               "delta_points": round(100 * (mine - theirs), 2)}
    if len(repetitions) == 1:
        counts = paired[0]["counts"]
        metrics["mcnemar_p_worse"] = round(mcnemar_worse_p(counts.get("trtmc_only", 0), counts.get("native_only", 0)), 5)
        metrics["answer_agreement"] = round((counts.get("both_correct", 0) + counts.get("both_wrong", 0)) / scored, 4) if scored else None
        entry["counts"] = counts
        tasks = paired[0]["per_task"]
        deltas = {task: (c["trtmc_only"] - c["native_only"]) / max(1, sum(c.values())) for task, c in tasks.items()}
        entry["per_task_delta_points"] = {task: round(100 * deltas[task], 1)
                                          for task in sorted(deltas, key=lambda t: abs(deltas[t]), reverse=True)[:5]
                                          if deltas[task]}
        entry["failures"] = paired[0]["examples"]
    else:
        metrics["per_seed"] = [{"seed": name, "trtmc_accuracy": round(100 * a, 2), "native_accuracy": round(100 * b, 2)}
                               for name, (a, b, _) in zip(repetitions, accuracies)]
    entry["samples"] = scored
    failures = [*(candidate.get("failed") or {}).values(), *(native.get("failed") or {}).values()]
    if failures:
        metrics["failed"] = "; ".join(failures)[:600]
    entry["metrics"] = metrics
    first = repetitions[0]
    entry["workload_perf"] = workload_perf(candidate.get("timings", {}).get(first, {}),
                                           native.get("timings", {}).get(first, {}))
    entry["status"], entry["reasons"] = status(metrics, entry["gate"], expected=expected, paired=scored)
    entry["notes"] = notes(metrics)
    return entry


def error_entry(item: Mapping[str, Any], expected: int, error: str) -> dict[str, Any]:
    return {"suite": item["suite"], "source": "absolute", "benchmark": item.get("plugin") or item.get("metric"),
            "status": "error",
            "samples": 0, "expected_samples": expected, "passed": None, "gate": dict(item["gate"]),
            "error": error[:800]}


def _probe(service: Mapping[str, Any], operation: str, request: Mapping[str, Any]) -> None:
    import urllib.error
    import urllib.request

    call = urllib.request.Request(f"{service['url']}/v1/tasks/{operation}", data=json.dumps({"request": request}).encode(),
                                  headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(call, timeout=3600).read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"probe rejected: {error.read().decode(errors='replace')[-400:]}") from error


def run_native(environment: Environment, model: Mapping[str, Any], python: str, plans: Mapping[str, Sequence],
               out: Path, probe_request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Every benchmark on the native model: the generic adapter, eager, at the candidate precision; the
    family's declared reference (and its precisions) when the adapter cannot serve the model. The adapter
    runs as up to ``native_replicas`` (environment) copies that fit on the GPU, each answering one problem
    at a time: the answers do not change, its model-call times are then not comparable."""
    from .runner import timing_precisions

    reference = model["reference"]
    errors = []
    for backend in dict.fromkeys([reference["backend"], reference.get("fallback") or reference["backend"]]):
        for precision in timing_precisions(reference):
            tag = "" if (backend, precision) == (reference["backend"], reference["perf_precision"]) else f"-{backend}-{precision}"
            try:
                count = int(environment.values.get("native_replicas") or 1) if backend == "reference" else 1
                with serving_replicas(environment, dict(model), backend, out / f"absolute-native-server{tag}",
                                      count=count, mode="eager", precision=precision, python=python) as service:
                    if probe_request is not None:
                        _probe(service, model["operation"], probe_request)
                    runs = {item["suite"]: run_side(environment, service, model, item, plans[item["suite"]],
                                                    out / f"absolute-native{tag}") for item in model["absolute"]}
                return {"backend": backend, "precision": precision, "runs": runs, "replicas": service["replicas"],
                        **({"fallback_from": "; ".join(errors)[:600]} if errors else {})}
            except Exception as error:  # noqa: BLE001 - the next precision, then the family reference
                errors.append(f"{backend} {precision}: {type(error).__name__}: {str(error)[-300:]}")
    raise RuntimeError("; ".join(errors)[:1500])


def run_candidate(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any],
                  plans: Mapping[str, Sequence], out: Path) -> dict[str, Any]:
    return {item["suite"]: run_side(environment, service, model, item, plans[item["suite"]], out / "absolute-trtmc")
            for item in model["absolute"]}


def entries(model: Mapping[str, Any], plans: Mapping[str, Sequence], candidate: Mapping[str, Any],
            native: Mapping[str, Any], native_error: str | None) -> list[dict[str, Any]]:
    results = []
    runs = native.get("runs") or {}
    for item in model["absolute"]:
        problems = plans[item["suite"]]
        if native_error or item["suite"] not in runs:
            results.append(error_entry(item, len(problems), f"native side: {native_error or 'not run'}"))
        else:
            entry = judge(item, problems, candidate[item["suite"]], runs[item["suite"]])
            entry["native"] = {"backend": native.get("backend"), "precision": native.get("precision"), "mode": "eager",
                               "replicas": native.get("replicas", 1),
                               **({"fallback_from": native["fallback_from"]} if native.get("fallback_from") else {})}
            if native.get("replicas", 1) > 1 and entry.get("workload_perf", {}).get("pairs"):
                entry["workload_perf"] = {**entry["workload_perf"], "light": "white",
                                          "note": f"native ran as {native['replicas']} concurrent copies: "
                                                  "its model-call times are not comparable"}
            if model["candidate"].get("sequence_fallback"):
                entry["notes"] = [*entry.get("notes", []), model["candidate"]["sequence_fallback"]]
            results.append(entry)
    return results


def candidate_entries(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any], out: Path, *,
                      plans: Mapping[str, Sequence], native: Mapping[str, Any],
                      native_error: str | None) -> list[dict[str, Any]]:
    """TRTMC's answers on the candidate server, judged against the native ones (errors become entries,
    so the Perf measurement on the same server still runs)."""
    try:
        candidate = run_candidate(environment, service, model, plans, out)
    except Exception as error:  # noqa: BLE001 - the entries carry the failure
        return [error_entry(item, len(plans[item["suite"]]), f"TRTMC side: {type(error).__name__}: {error}")
                for item in model["absolute"]]
    return entries(model, plans, candidate, native, native_error)
