# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Absolute accuracy: both sides scored against gold answers, then compared.

TRTMC and the native model (the generic adapter, eager, at the candidate precision) answer the same
problems, and each side is scored against the gold answers:

- ``plugin`` entries run an AIPerf accuracy benchmark (trtmc_aiperf_plugins.benchmarks: MMLU, GSM8K,
  MATH-500, LAMBADA at pinned revisions, greedy decoding) graded by AIPerf;
- ``metric`` entries send a suite with gold labels (images, audio, sentence pairs) through AIPerf's
  trtmc_task endpoint and score the outputs here (gold_metrics).

Each entry is a paired non-inferiority decision (``noninferiority``, DESIGN.md Section 4): pass when
TRTMC's regression against the native model is shown to be below the benchmark's margin, fail when it
is shown to exceed it, inconclusive otherwise; ``not-comparable`` when the native score is below the
benchmark's suitability floor ``min_native``. A sampled model answers once per seed on each side and
is judged on the per-problem seed means.
"""

from __future__ import annotations

import functools
import json
import re
import statistics
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import cancel, gold_metrics, noninferiority
from .aiperf_runner import run_aiperf
from .config import Environment
from .judge import light
from .services import serving_replicas
from .suites import SELECTION_SEED, build_suite, request_sha

WORKLOAD_MARGIN_PERCENT = 5.0
# A request answers within seconds; whole-benchmark runs of large native models take hours.
RUN_TIMEOUT_S = 12 * 3600
# DESIGN.md Section 9: a deadline is three times the ledger's time, at least ten minutes.
DEADLINE_FACTOR, MIN_DEADLINE_S = 3, 600


@functools.lru_cache(maxsize=None)
def _ledger(path: str) -> dict[str, float]:
    return {name: float(seconds) for name, seconds in json.loads(Path(path).read_text()).items()}


def run_timeout(environment: Environment, model: Mapping[str, Any]) -> float:
    """An AIPerf run's deadline: the profile's in the environment's ``deadlines``, else three times its time in the
    environment's ``ledger`` (a ledger.json; relative to the repository), at least ten minutes, else
    RUN_TIMEOUT_S. Every run of a profile gets its whole-profile deadline, so no single run is cut short."""
    profile = model.get("catalog_profile")
    deadline = (environment.values.get("deadlines") or {}).get(profile)
    if deadline:
        return float(deadline)
    ledger = environment.values.get("ledger")
    if ledger:
        path = Path(ledger) if Path(ledger).is_absolute() else environment.path("repo") / ledger
        seconds = _ledger(str(path)).get(str(profile))
        if seconds is not None:
            return max(MIN_DEADLINE_S, DEADLINE_FACTOR * seconds)
    return RUN_TIMEOUT_S


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


COMMIT = re.compile(r"[0-9a-f]{40}")


@functools.lru_cache(maxsize=None)
def pinned_revision(name: str, revision: str | None) -> str | None:
    """The commit a tokenizer revision resolves to now, so every selection (the plan and both sides' AIPerf runs)
    loads exactly it: the hub's answer, else (offline, gated) the commit of the cached snapshot the revision
    resolves to; a commit or a local directory as given; the revision as given when neither resolves it (the
    plugin then does not cache that selection)."""
    if Path(name).is_dir() or (revision and COMMIT.fullmatch(revision)):
        return revision
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(name, revision=revision or None).sha or revision
    except Exception:  # noqa: BLE001 - offline, gated, or unknown: the cached snapshot's commit, else as given
        pass
    try:
        from huggingface_hub import try_to_load_from_cache

        found = try_to_load_from_cache(name, "tokenizer_config.json", revision=revision or None)
    except Exception:  # noqa: BLE001
        return revision
    cached = Path(found).parent.name if isinstance(found, str) and Path(found).parent.parent.name == "snapshots" else ""
    return cached if COMMIT.fullmatch(cached) else revision


def selection_environment(environment: Environment, model: Mapping[str, Any], item: Mapping[str, Any]) -> dict[str, str]:
    """TRTMC_ACCURACY_* settings of the plugin's problem selection (identical for both sides)."""
    environ = {"HF_DATASETS_CACHE": str(environment["hf_datasets_cache"]),
               # One selection per settings: the plan's, read back by both sides' AIPerf runs.
               "TRTMC_ACCURACY_CACHE": str(Path(environment["hf_datasets_cache"]) / "trtmc-accuracy-selections"),
               "TRTMC_ACCURACY_TOKEN_LIMIT": str(model["candidate"]["max_sequence_length"]),
               # The chat route counts the prompt as rendered (else a 128-token allowance when the tokenizer has
               # no template); plain completions add at most a BOS token.
               "TRTMC_ACCURACY_TEMPLATE_MARGIN": "128" if item.get("endpoint") == "chat" else "8",
               "TRTMC_ACCURACY_SEED": str(SELECTION_SEED),
               "TRTMC_ACCURACY_TOKENIZER": tokenizer_source(model)[0]}
    revision = pinned_revision(*tokenizer_source(model))
    if revision:
        environ["TRTMC_ACCURACY_TOKENIZER_REVISION"] = str(revision)
    if model["reference"].get("trust_remote_code"):
        environ["TRTMC_ACCURACY_TRUST_REMOTE_CODE"] = "1"
    if item.get("endpoint") == "chat":
        environ["TRTMC_ACCURACY_CHAT"] = "1"
    for key, name in (("per_task", "TRTMC_ACCURACY_PER_TASK"), ("limit", "TRTMC_ACCURACY_LIMIT"),
                      ("max_new_tokens", "TRTMC_ACCURACY_MAX_NEW_TOKENS")):
        if item.get(key):
            environ[name] = str(item[key])
    if environment.values.get("smoke"):  # one problem
        environ.pop("TRTMC_ACCURACY_PER_TASK", None)
        environ["TRTMC_ACCURACY_LIMIT"] = "1"
    return environ


def plan(environment: Environment, model: Mapping[str, Any], item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The problems a run sends, in request order: their task and gold answer (and the request)."""
    if item.get("metric"):
        definition = item["suite_definition"]
        if definition["source"].get("kind") == "family_inputs":  # the family's script, in its environment
            from .services import reference_python

            definition = {**definition, "source": {**definition["source"],
                                                   "python": reference_python(environment, dict(model))}}
        suite = build_suite(definition, environment)
        samples = suite.samples
        if item.get("truncate_tokens"):  # the same text on both sides, whatever each side's own cut
            samples = _head_truncated(model, samples, int(item["truncate_tokens"]))
        if item.get("fit_prompt"):  # text prompts on a generation bundle: the shipped length decides
            samples = _fitted(model, samples, int(item.get("min_new_tokens", 16)))
        return [{"task": sample.get("task", suite.name), "gold": sample.get("label"), "sample_id": sample["sample_id"],
                 "request": sample["request"], "request_sha": sample["request_sha"],
                 **{key: sample[key] for key in ("cluster", "series") if key in sample}} for sample in samples]
    from trtmc_aiperf_plugins.benchmarks import problems

    chosen = problems(item["plugin"], item.get("tasks"), int(item.get("n_shots", 0)),
                      selection_environment(environment, model, item))
    return [{"task": problem.task, "gold": problem.ground_truth} for problem in chosen]


def _fitted(model: Mapping[str, Any], samples: Sequence[Mapping[str, Any]], min_new: int) -> list[dict[str, Any]]:
    """Samples whose prompt leaves at least ``min_new`` tokens to generate on the bundle, each generating at
    most what is left (the same requests on both sides); the others are dropped (DESIGN.md Section 2)."""
    from transformers import AutoTokenizer

    length = int(model["candidate"].get("max_sequence_length") or 0)
    if not length:
        return list(samples)
    name, revision = tokenizer_source(model)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision,
                                              trust_remote_code=bool(model["reference"].get("trust_remote_code")))
    kept = []
    for sample in samples:
        request = dict(sample["request"])
        budget = length - len(tokenizer(str(request.get("prompt", "")), add_special_tokens=True)["input_ids"]) - 8
        if budget < min_new:
            continue
        request["max_new_tokens"] = min(int(request.get("max_new_tokens", budget)), budget)
        kept.append({**sample, "request": request, "request_sha": request_sha(request)})
    return kept


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
    # The serving base request is the catalog request: greedy as it is unless it samples (top_k 1 or temperature 0;
    # a family may pin its greedy contract, Qwen3-Omni: temperature 1 with top_k 1); a sampling model's runs are
    # repeated with distinct seeds.
    if seed is not None:
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
                                        "--request-count", str(len(problems))], timeout_s=run_timeout(environment, model))
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
    for seed in item.get("seeds") or [None]:  # smoke too: the seed-mean scorer of a sampled model runs
        name = "greedy" if seed is None else f"seed{seed}"
        run = run_aiperf(environment, out / f"{item['suite']}-{name}", _arguments(model, item, service, count, seed),
                         env=selection_environment(environment, model, item), timeout_s=run_timeout(environment, model))
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
    same number of tokens (informational: the Perf L1 gate times the catalog request)."""
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


def status(entry: Mapping[str, Any]) -> tuple[str, list[str]]:
    """pass / fail / inconclusive / not-comparable / error of a scored entry (also when re-judging a
    report): every expected problem answered on both sides, a native score above the suitability
    floor, then the entry's non-inferiority outcome under its gate."""
    metrics, gate = entry.get("metrics") or {}, entry.get("gate") or {}
    expected, paired = int(entry.get("expected_samples") or 0), int(entry.get("samples") or 0)
    if paired < expected:
        return "error", [f"{expected - paired} of {expected} problems lack a graded answer on one side"
                         + (f" ({metrics['failed']})" if metrics.get("failed") else "")]
    native = metrics.get("native_score")
    if native is not None and gate.get("min_native") is not None and native < float(gate["min_native"]):
        # Too few right answers from the native model (at or near chance): the benchmark does not
        # discriminate for this model, so it says nothing about TRTMC.
        return "not-comparable", [f"the native model scores {native:g} (< {gate['min_native']:g}): the benchmark "
                                  "does not fit this model"]
    test = metrics.get("test") or {}
    if test.get("outcome") is None:
        return "error", ["no non-inferiority outcome"]
    reasons = [] if test["outcome"] == "pass" else [
        f"regression {test.get('regression_points', 0.0):+.3f} points against margin "
        f"{noninferiority.margin(gate, native):g}: {test['outcome']}"]
    return test["outcome"], reasons


def binary_test(entry: Mapping[str, Any]) -> dict[str, Any]:
    """The non-inferiority outcome of a right/wrong entry from its recorded counts or per-problem
    regressions (so a re-judge under a new gate needs no rerun)."""
    gate, metrics = entry.get("gate") or {}, entry.get("metrics") or {}
    delta = noninferiority.margin(gate, metrics.get("native_score"))
    if metrics.get("per_problem_regression") is not None:
        return noninferiority.paired_means(metrics["per_problem_regression"], delta)
    counts = entry.get("counts") or {}
    return noninferiority.binary(counts.get("native_only", 0), counts.get("trtmc_only", 0), int(entry.get("samples") or 0),
                                 delta)


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
    """The entry of a corpus metric (WER, chrF, Spearman, mAP): both sides' statistic and the bootstrap
    outcome."""
    expected = len(problems)
    mine = gold_metrics.answered(item["metric"], problems, candidate["observations"]["greedy"])
    theirs = gold_metrics.answered(item["metric"], problems, native["observations"]["greedy"])
    paired = len(set(mine) & set(theirs))
    entry: dict[str, Any] = {"suite": item["suite"], "source": "absolute", "benchmark": item["metric"],
                             "endpoint": "trtmc_task", "expected_samples": expected, "samples": paired, "passed": None,
                             "gate": dict(item["gate"]),
                             "aiperf_exit": {"trtmc": candidate["exit"], "native": native["exit"]}}
    if paired < expected:
        entry["status"], entry["reasons"] = "error", [f"{expected - paired} of {expected} problems lack a usable output "
                                                      "on one side"]
        return entry
    comparison = gold_metrics.compare_corpus(item["metric"], problems, mine, theirs, item["gate"], item.get("metric_params"))
    test = {key: comparison[key] for key in ("outcome", "regression_points", "margin_points", "excess_interval90",
                                             "clusters", "resamples", "block", "reason") if key in comparison}
    entry["metrics"] = {"trtmc_score": comparison["trtmc"], "native_score": comparison["native"],
                        "higher_is_better": comparison["higher_is_better"], "units": comparison["units"], "test": test}
    entry["workload_perf"] = workload_perf(candidate["timings"]["greedy"], native["timings"]["greedy"])
    entry["status"], entry["reasons"] = status(entry)
    return entry


def judge_parity(item: Mapping[str, Any], problems: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any],
                 native: Mapping[str, Any]) -> dict[str, Any]:
    """A conversion-parity entry: every problem's TRTMC output within the gate's tolerance of the native
    output (no gold answer; a deterministic per-sample check, so no statistical test)."""
    compare = gold_metrics.PARITY[item["metric"]]
    mine, theirs = candidate["observations"]["greedy"], native["observations"]["greedy"]
    expected = len(problems)
    paired = sorted(set(mine) & set(theirs))
    entry: dict[str, Any] = {"suite": item["suite"], "source": "absolute", "benchmark": item["metric"],
                             "endpoint": "trtmc_task", "expected_samples": expected, "samples": len(paired),
                             "gate": dict(item["gate"]), "aiperf_exit": {"trtmc": candidate["exit"], "native": native["exit"]}}
    if len(paired) < expected:
        return {**entry, "passed": None, "status": "error",
                "reasons": [f"{expected - len(paired)} of {expected} problems lack an output on one side"]}
    results = [(index, *compare(mine[index], theirs[index], item["gate"])) for index in paired]
    unreadable = [reason for _, ok, reason in results if ok is None]
    if unreadable:  # an output missing or unreadable is missing evidence, never a failure
        return {**entry, "passed": None, "status": "error",
                "reasons": [f"{len(unreadable)} of {expected} outputs lack readable evidence ({unreadable[0][:200]})"]}
    failures = [{"sample_id": problems[index].get("sample_id", str(index)), "explanation": reason}
                for index, ok, reason in results if not ok]
    entry.update(passed=expected - len(failures), required_passes=expected, failures=failures[:5],
                 status="pass" if not failures else "fail",
                 reasons=[f"{len(failures)} of {expected} outputs outside the tolerance"] if failures else [])
    return entry


def _as_observations(side: Mapping[str, Any]) -> dict[str, Any]:
    """An AIPerf benchmark side (graded records) as text observations: each answer's normalized sentence."""
    from trtmc_aiperf_plugins.benchmarks import _sentence

    records = side["records"].get("greedy", {})
    return {**side, "observations": {"greedy": {index: {"text": _sentence(record.get("model_output") or "")}
                                                for index, record in records.items()}}}


def judge(item: Mapping[str, Any], problems: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any],
          native: Mapping[str, Any]) -> dict[str, Any]:
    """The entry of one benchmark from both sides' graded records."""
    if item.get("corpus_metric"):  # an AIPerf benchmark scored as a corpus (BART: chrF++ of its sentences)
        from trtmc_aiperf_plugins.benchmarks import _sentence

        corpus = {**item, "metric": item["corpus_metric"]}
        golds = [{**problem, "gold": _sentence(str(problem["gold"]))} for problem in problems]
        entry = judge_corpus(corpus, golds, _as_observations(candidate), _as_observations(native))
        return {**entry, "benchmark": f"{item.get('plugin')} ({item['corpus_metric']})", "endpoint": item.get("endpoint")}
    if item.get("metric") in gold_metrics.PARITY:
        return judge_parity(item, problems, candidate, native)
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
    answered = [index for index in range(expected)
                if all(index in candidate["records"][name] and index in native["records"].get(name, {})
                       for name in repetitions)]
    scored = len(answered)
    mine = sum(_correct(candidate["records"][name][index]) for name in repetitions for index in answered)
    theirs = sum(_correct(native["records"][name][index]) for name in repetitions for index in answered)
    total = scored * len(repetitions)
    metrics: dict[str, Any] = {"trtmc_score": 100.0 * mine / total if total else 0.0,
                               "native_score": 100.0 * theirs / total if total else 0.0}
    metrics["regression_points"] = metrics["native_score"] - metrics["trtmc_score"]
    if len(repetitions) == 1:
        counts = paired[0]["counts"]
        metrics["answer_agreement"] = (counts.get("both_correct", 0) + counts.get("both_wrong", 0)) / scored if scored else None
        entry["counts"] = counts
        tasks = paired[0]["per_task"]
        deltas = {task: (c["trtmc_only"] - c["native_only"]) / max(1, sum(c.values())) for task, c in tasks.items()}
        entry["per_task_delta_points"] = {task: round(100 * deltas[task], 1)
                                          for task in sorted(deltas, key=lambda t: abs(deltas[t]), reverse=True)[:5]
                                          if deltas[task]}
        entry["failures"] = paired[0]["examples"]
    else:  # seeds stay inside their problem: one regression per problem, averaged over the seeds
        metrics["per_problem_regression"] = [
            100.0 * sum(_correct(native["records"][name][index]) - _correct(candidate["records"][name][index])
                        for name in repetitions) / len(repetitions) for index in answered]
    entry["samples"] = scored
    failures = [*(candidate.get("failed") or {}).values(), *(native.get("failed") or {}).values()]
    if failures:
        metrics["failed"] = "; ".join(failures)[:600]
    entry["metrics"] = metrics
    metrics["test"] = binary_test(entry)
    first = repetitions[0]
    entry["workload_perf"] = workload_perf(candidate.get("timings", {}).get(first, {}),
                                           native.get("timings", {}).get(first, {}))
    entry["status"], entry["reasons"] = status(entry)
    return entry


def error_entry(item: Mapping[str, Any], expected: int, error: str) -> dict[str, Any]:
    return {"suite": item["suite"], "source": "absolute", "benchmark": item.get("plugin") or item.get("metric"),
            "status": "error",
            "samples": 0, "expected_samples": expected, "passed": None, "gate": dict(item["gate"]),
            "error": error[:800]}


def _probe(service: Mapping[str, Any], operation: str, request: Mapping[str, Any], timeout_s: float = 3600) -> None:
    import urllib.error
    import urllib.request

    call = urllib.request.Request(f"{service['url']}/v1/tasks/{operation}", data=json.dumps({"request": request}).encode(),
                                  headers={"Content-Type": "application/json"})

    def send() -> None:  # the whole exchange, an error body included, inside the cancellable wait
        try:
            urllib.request.urlopen(call, timeout=timeout_s).read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"probe rejected: {error.read().decode(errors='replace')[-400:]}") from error

    cancel.wait_for(send)


def keeps_artifacts(model: Mapping[str, Any]) -> bool:
    """A benchmark reads the output files the servers write (their scratch is then kept)."""
    return any(item.get("metric") in gold_metrics.ARTIFACT_METRICS for item in model.get("absolute", []))


def run_native(environment: Environment, model: Mapping[str, Any], python: str, plans: Mapping[str, Sequence],
               out: Path, probe_request: Mapping[str, Any] | None = None, *, mps_env: Mapping[str, str] | None = None,
               precisions: Sequence[str] | None = None, started: Callable[[Mapping[str, Any]], None] | None = None,
               go: threading.Event | None = None, finished: threading.Event | None = None,
               release: threading.Event | None = None) -> dict[str, Any]:
    """Every benchmark on the native model: the reference adapter, eager, at the first precision of
    ``timing_precisions`` it serves. The adapter runs as up to ``native_replicas`` (environment) copies
    that fit on the GPU, each answering one problem at a time: the answers do not change, its model-call
    times are then not comparable."""
    from .runner import timing_precisions

    errors = []
    for precision in precisions or timing_precisions(model["reference"]):
        try:
            count = 1 if environment.values.get("smoke") else int(environment.values.get("native_replicas") or 1)
            with serving_replicas(environment, dict(model), "reference", out / f"absolute-native-server-{precision}",
                                  count=count, mode="eager", precision=precision, python=python,
                                  keep_artifacts=keeps_artifacts(model), mps_env=mps_env) as service:
                try:
                    if started is not None:  # the copies hold their memory, idle: the other side may size its own
                        started(service)
                    while go is not None and not go.wait(1):  # the other side's copies are starting
                        cancel.check()
                    if probe_request is not None:
                        _probe(service, model["operation"], probe_request)
                    runs = {item["suite"]: run_side(environment, service, model, item, plans[item["suite"]],
                                                    out / f"absolute-native-{precision}")
                            for item in model["absolute"]}
                finally:  # sharing an MPS daemon, no copy leaves while the other side's copies still answer
                    if finished is not None:
                        finished.set()
                    while release is not None and not release.wait(1):
                        cancel.check()
            return {"backend": "reference", "precision": precision, "runs": runs, "replicas": service["replicas"],
                    "mps": bool(service.get("mps")),
                    **({"fallback_from": "; ".join(errors)[:600]} if errors else {})}
        except Exception as error:  # noqa: BLE001 - the next precision
            errors.append(f"{precision}: {type(error).__name__}: {str(error)[-300:]}")
    raise RuntimeError("; ".join(errors)[:1500])


def incomplete(model: Mapping[str, Any], plans: Mapping[str, Sequence], runs: Mapping[str, Any]) -> str | None:
    """Why one side's Acc runs did not answer every problem (failed or missing requests, a suite not run), or
    None. Wrong answers are answers: only what never came back counts."""
    for item in model["absolute"]:
        side = runs.get(item["suite"])
        if side is None:
            return f"{item['suite']}: not run"
        if side.get("failed"):
            return f"{item['suite']}: {next(iter(side['failed'].values()))}"
        answered = side.get("observations") if "observations" in side else side.get("records")
        for name, found in (answered or {}).items():
            if len(found) < len(plans[item["suite"]]):
                return f"{item['suite']} ({name}): {len(found)} of {len(plans[item['suite']])} answered"
        if not answered:
            return f"{item['suite']}: no answers"
    return None


def overlapped_acc(environment: Environment, model: Mapping[str, Any], python: str, plans: Mapping[str, Sequence],
                   out: Path, probe_request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Both sides' Acc answers at once (environment ``acc_overlap``), under one MPS daemon: the native copies start
    at the first native precision, sized alone, and wait idle; TRTMC's copies then start, sized against what is
    left with the native copies' growth held back; then both sides answer concurrently. Each copy answers one
    request at a time, so the answers are each side's own. Returns the native side or its error and TRTMC's runs
    with its copies or its error (an incomplete side is an error): the caller lets that side answer alone. No copy
    of either side stops before both sides have answered (a client leaving the shared MPS server stalled the other
    side's copies on GB300). An interrupt cancels the native side's work before the daemon stops."""
    from .runner import timing_precisions
    from .services import REPLICA_GROWTH, mps

    result: dict[str, Any] = {}
    ready, go, native_copies = threading.Event(), threading.Event(), {}
    native_finished, release = threading.Event(), threading.Event()

    def up(service: Mapping[str, Any]) -> None:
        native_copies.update(replicas=int(service.get("replicas") or 1), footprint=service.get("footprint_mib") or 0)
        ready.set()

    with mps(environment, out / "acc-mps") as shared:
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            # The native side's servers and runs in a directory of the overlap's own: a fallback run keeps them.
            native = pool.submit(run_native, environment, model, python, plans, out / "acc-overlap", probe_request,
                                 mps_env=shared, precisions=timing_precisions(model["reference"])[:1], started=up, go=go,
                                 finished=native_finished, release=release)

            def native_answered() -> None:  # a client leaving the shared MPS server can stall the others' work
                while not (native_finished.wait(1) or native.done() or cancel.EVENT.is_set()):
                    pass

            while not ready.wait(1) and not native.done():
                pass
            reserve = int((REPLICA_GROWTH - 1) * native_copies.get("footprint", 0) * native_copies.get("replicas", 0))
            try:
                with serving_replicas(environment, dict(model), "trtmc", out / "acc-overlap" / "candidate-acc",
                                      count=int(environment.values.get("candidate_replicas") or 1),
                                      keep_artifacts=keeps_artifacts(model), mps_env=shared,
                                      reserve_mib=reserve) as service:
                    go.set()
                    try:
                        candidate = run_candidate(environment, service, model, plans, out / "acc-overlap")
                    except BaseException as error:
                        if not isinstance(error, Exception):  # an interrupt: the native side stops too
                            cancel.EVENT.set()
                        raise
                    finally:  # TRTMC's copies stay until the native side has answered too, then both sides stop
                        try:
                            native_answered()
                        except BaseException:  # an interrupt during this wait: both sides stop, briefly
                            cancel.EVENT.set()
                            raise
                        finally:
                            release.set()
                    gap = incomplete(model, plans, candidate)
                    if gap:
                        result["candidate_error"] = f"incomplete: {gap}"
                    else:
                        result["candidate"] = candidate
                        result["copies"] = {"candidate_replicas": int(service.get("replicas") or 1),
                                            "candidate_mps": bool(service.get("mps"))}
            except Exception as error:  # noqa: BLE001 - TRTMC then answers on its own, after the native side
                result["candidate_error"] = f"{type(error).__name__}: {error}"[:1500]
            go.set()  # the native side answers even when TRTMC could not start
            release.set()
            try:
                native_result = native.result()
                gap = incomplete(model, plans, native_result["runs"])
                if gap:
                    result["native_error"] = f"incomplete: {gap}"
                else:
                    result["native"] = native_result
            except Exception as error:  # noqa: BLE001 - the native side then runs on its own
                result["native_error"] = f"{type(error).__name__}: {error}"[:1500]
        except BaseException:
            cancel.EVENT.set()  # the native side's AIPerf runs and server starts stop now
            raise
        finally:
            pool.shutdown(wait=True)  # its servers stopped before the daemon does
            cancel.EVENT.clear()
    return result


def run_candidate(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any],
                  plans: Mapping[str, Sequence], out: Path) -> dict[str, Any]:
    return {item["suite"]: run_side(environment, service, model, item, plans[item["suite"]], out / "absolute-trtmc")
            for item in model["absolute"]}


def entries(model: Mapping[str, Any], plans: Mapping[str, Sequence], candidate: Mapping[str, Any],
            native: Mapping[str, Any], native_error: str | None, candidate_replicas: int = 1,
            candidate_mps: bool = False, concurrent_sides: bool = False) -> list[dict[str, Any]]:
    results = []
    runs = native.get("runs") or {}
    for item in model["absolute"]:
        problems = plans[item["suite"]]
        if native_error or item["suite"] not in runs:
            results.append({**error_entry(item, len(problems), f"native side: {native_error or 'not run'}"),
                            "candidate_replicas": candidate_replicas, "candidate_mps": candidate_mps})
        else:
            entry = judge(item, problems, candidate[item["suite"]], runs[item["suite"]])
            entry["native"] = {"backend": native.get("backend"), "precision": native.get("precision"), "mode": "eager",
                               "replicas": native.get("replicas", 1), "mps": bool(native.get("mps")),
                               **({"fallback_from": native["fallback_from"]} if native.get("fallback_from") else {})}
            concurrent = [f"{side} ran as {copies} concurrent copies" for side, copies in
                          (("native", native.get("replicas", 1)), ("TRTMC", candidate_replicas)) if copies > 1]
            concurrent += ["both sides answered at once"] if concurrent_sides else []
            if concurrent and entry.get("workload_perf", {}).get("pairs"):
                entry["workload_perf"] = {**entry["workload_perf"], "light": "white",
                                          "note": f"{'; '.join(concurrent)}: model-call times are not comparable"}
            entry["candidate_replicas"], entry["candidate_mps"] = candidate_replicas, candidate_mps
            if concurrent_sides:
                entry["sides_concurrent"] = True
            results.append(entry)
    return results


def candidate_entries(environment: Environment, service: Mapping[str, Any], model: Mapping[str, Any], out: Path, *,
                      plans: Mapping[str, Sequence], native: Mapping[str, Any],
                      native_error: str | None) -> list[dict[str, Any]]:
    """TRTMC's answers on the candidate server (or its copies: ``service["replicas"]``), judged against the
    native ones (errors become entries, so the Perf measurement still runs)."""
    copies = {"candidate_replicas": int(service.get("replicas") or 1), "candidate_mps": bool(service.get("mps"))}
    try:
        candidate = run_candidate(environment, service, model, plans, out)
    except Exception as error:  # noqa: BLE001 - the entries carry the failure
        return [{**error_entry(item, len(plans[item["suite"]]), f"TRTMC side: {type(error).__name__}: {error}"),
                 **copies} for item in model["absolute"]]
    return entries(model, plans, candidate, native, native_error, **copies)
