# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify many models: build -> qualify -> retention per model, then one merged summary.

``run_one`` is the per-model flow (``run``); ``run_all`` runs a batch sequentially (``run-all``),
grouping profiles that share a checkpoint so a deleted checkpoint is never downloaded twice, and
downloading the next profile's checkpoint while the current one runs. A profile whose output
directory holds a final result is skipped unless ``rerun`` (the old directory is then kept aside as
``<profile>.<timestamp>``). ``summary`` merges result roots, for example one per host.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import json
import os
import re
import socket
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import bundles, retention
from .bundles import prefetch
from .config import Environment
from .models import checkpoints
from .report import acc_value, plain
from .services import reference_python

KEPT_ASIDE = re.compile(r"\.\d{10}$")  # <profile>.<unix time> of a previous run
RUN_KEY = "run-key.txt"
PACKAGE = Path(__file__).resolve().parent.parent  # apps/aiperf_qual: the harness code and configuration


def harness_digest() -> str:
    """sha256 of the harness sources and configuration (the code that produced a result)."""
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.rglob("*")):
        if path.suffix in (".py", ".yaml") and "__pycache__" not in path.parts and "tests" not in path.parts:
            digest.update(str(path.relative_to(PACKAGE)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


SOURCE_SUFFIXES = (".py", ".yaml", ".json", ".txt")


@functools.lru_cache(maxsize=None)
def tree_digest(root: str) -> str:
    """sha256 of the source files (``SOURCE_SUFFIXES``) below ``root``, symlinks followed; "" when absent."""
    if not Path(root).is_dir():
        return ""
    paths = []
    for directory, names, files in os.walk(root, followlinks=True):
        names[:] = [name for name in names if name != "__pycache__"]
        paths += [Path(directory) / name for name in files if name.endswith(SOURCE_SUFFIXES)]
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


@functools.lru_cache(maxsize=None)
def runtime_digest(worker: str, runtime_root: str) -> str:
    """sha256 over the contents of what a run loads from the TRTMC build: the worker binary and the shared libraries
    under the runtime root (by relative path). Build logs and timestamps are left out, so two hosts that built the
    same sources agree."""
    digest = hashlib.sha256()
    if Path(worker).is_file():
        digest.update(b"worker\0" + hashlib.sha256(Path(worker).read_bytes()).digest())
    root = Path(runtime_root)
    libraries = sorted({path for pattern in ("*.so", "*.so.*") for path in root.rglob(pattern) if path.is_file()}) \
        if root.is_dir() else []
    for path in libraries:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def code_digests(environment: Environment, model: Mapping[str, Any]) -> dict[str, str]:
    """The code a model's run executes besides the harness: the serving package, TRTMC (core, trtmc-bench,
    the worker and runtime), and the model's family (its TRTMC implementation and native adapter files)."""
    repo = environment.path("repo") if environment.values.get("repo") else PACKAGE.parent.parent
    return {"serving": tree_digest(str(PACKAGE.parent / "perf_serving" / "trtmc_perf_serving")),
            "core": tree_digest(str(repo / "core")),
            "benchmark": tree_digest(str(repo / "apps" / "benchmark" / "trtmc_benchmark")),
            "family": tree_digest(str(repo / "families" / str(model.get("family") or ""))) if model.get("family") else "",
            "runtime": runtime_digest(str(environment.values.get("worker") or ""),
                                      str(environment.values.get("runtime_root") or ""))}


@functools.lru_cache(maxsize=None)
def dependencies_digest(python: str) -> str:
    """sha256 of an interpreter's ``pip freeze`` (its resolved dependencies), "" when it cannot tell."""
    import subprocess

    try:
        frozen = subprocess.run([python, "-m", "pip", "freeze", "--all"], capture_output=True, text=True,
                                timeout=300, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    # A package installed from a local path names that path (a temporary directory for the AIPerf plugins); its
    # code is in the harness and code digests, so only its name and the local origin count here.
    frozen = re.sub(r"^(\S+) @ file://\S+$", r"\1 @ local", frozen, flags=re.MULTILINE)
    return hashlib.sha256(frozen.encode()).hexdigest()


def run_key(environment: Environment, model: Mapping[str, Any]) -> str:
    """Identity of a run: the resolved model configuration, the harness, the code it executes
    (``code_digests``), the mode (smoke / formal), and the resolved dependencies of the serving, AIPerf, and
    reference interpreters. A finished result counts only under the same key."""
    mode = "smoke" if environment.values.get("smoke") else "formal"
    pythons = {str(environment.values.get("serve_python") or ""), sys.executable}
    if (model.get("reference") or {}).get("requirements") and environment.values.get("reference_env_root"):
        try:
            pythons.add(reference_python(environment, dict(model)))
        except Exception:  # noqa: BLE001 - the run itself reports the environment failure
            pythons.add("reference-environment-unavailable")
    dependencies = {python: dependencies_digest(python) for python in sorted(pythons) if python and Path(python).exists()}
    text = json.dumps({"model": model, "harness": harness_digest(), "code": code_digests(environment, model), "mode": mode,
                       "dependencies": dependencies}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def qualify(model: dict[str, Any], environment: Environment, out: Path) -> dict[str, Any]:
    """runner.qualify, imported on use: ``summary`` runs where AIPerf is not installed."""
    from .runner import qualify as run

    return run(model, environment, out)


def _error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:300]


def set_aside(out: Path) -> Path | None:
    """Keep a previous run's directory as ``<name>.<unix time>`` so its results cannot stand in for
    the new run's (a failed rebuild must not leave an older pass visible)."""
    if not out.exists() or not any(out.iterdir()):
        return None
    kept = out.with_name(f"{out.name}.{int(time.time())}")
    while kept.exists():  # two runs in the same second: wait for the next (KEPT_ASIDE expects seconds)
        time.sleep(1)
        kept = out.with_name(f"{out.name}.{int(time.time())}")
    out.rename(kept)
    return kept


def warm_selections(environment: Environment, model: Mapping[str, Any]) -> None:
    """The plugin benchmarks' Acc selections (dataset loading and length filtering), computed into the shared
    selection cache while the bundle builds; the run's own plan reads them back. Best effort: a failure here
    leaves the plan to compute (and report) it."""
    from . import absolute

    for item in model.get("absolute") or []:
        if not item.get("metric") and item.get("plugin"):
            try:
                absolute.plan(environment, model, item)
            except Exception:  # noqa: BLE001 - the run's plan recomputes and reports it
                pass


def run_one(environment: Environment, model: dict[str, Any], out: Path) -> dict[str, Any]:
    """Build the bundle when missing, qualify, and apply the bundle retention policy. A previous run
    in ``out`` is set aside first."""
    started = time.time()
    set_aside(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / RUN_KEY).write_text(run_key(environment, model) + "\n")
    bundle_policy, _ = retention.policies(environment)
    record: dict[str, Any] = {"profile": model["model"], "task": model.get("task")}
    with ThreadPoolExecutor(max_workers=1) as warm:
        warm.submit(warm_selections, environment, model)  # CPU work while the GPU builds
        try:
            python = reference_python(environment, model)  # the family's requirements, as CI builds
            prefetch(environment, model)  # outside the GPU lock the build takes
            build = bundles.ensure_bundle(environment, model, out, python)
        except Exception as error:  # noqa: BLE001 - recorded; a batch goes on with the next model
            build = {"status": "failed", "reason": _error(error)}
    (out / "build.json").write_text(json.dumps({**record, **build}, indent=2))
    if build["status"] == "failed":
        record.update(category="build-failed", reason=build.get("reason", ""))
    else:
        try:
            verdict = qualify(model, environment, out)["verdict"]
            record.update(verdict)
        except Exception as error:  # noqa: BLE001
            record.update(category="error", reason=_error(error))
            (out / "error.json").write_text(json.dumps({**record, "traceback": traceback.format_exc()}, indent=2))
        if retention.should_delete_bundle(bundle_policy, record["category"], built=build.get("status") == "built"):
            record["bundle_deleted"] = retention.delete_bundle(environment, model)
    record["seconds"] = round(time.time() - started)
    return record


def order(models: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Profiles grouped by checkpoint (groups by their first profile name)."""
    return [model for group in _groups(models) for model in group]


def _groups(models: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for model in models:
        groups[model["candidate"].get("checkpoint") or model["model"]].append(model)
    return sorted((sorted(group, key=lambda m: m["model"]) for group in groups.values()),
                  key=lambda group: group[0]["model"])


def shard(models: Sequence[Mapping[str, Any]], index: int, count: int) -> list[Mapping[str, Any]]:
    """Checkpoint groups dealt round-robin: shard ``index`` of ``count`` (a group stays on one host)."""
    return [model for position, group in enumerate(_groups(models)) if position % count == index for model in group]


def _finished(out: Path, key: str | None = None) -> str | None:
    """Category of a final result in ``out`` (a verdict or a failed build) produced under ``key``, else None."""
    if key is not None and (not (out / RUN_KEY).is_file() or (out / RUN_KEY).read_text().strip() != key):
        return None
    report, build = out / "report.json", out / "build.json"
    if report.is_file():
        return json.loads(report.read_text()).get("verdict", {}).get("category")
    if build.is_file() and json.loads(build.read_text()).get("status") == "failed":
        return "build-failed"
    return None


def run_all(environment: Environment, models: Sequence[dict[str, Any]], out_root: Path, *,
            rerun: bool = False, prefetch_next: bool = True, keep_order: bool = False) -> list[dict[str, Any]]:
    """Each profile in turn (grouped by checkpoint, or in the given order with ``keep_order``: an assignment's);
    campaign.jsonl records the host, start time, and outcome of each."""
    _, hf_policy = retention.policies(environment)
    ordered = list(models) if keep_order else order(models)
    remaining = collections.Counter(repo for model in ordered for repo in checkpoints(model))
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=1) as downloads, open(out_root / "campaign.jsonl", "a") as log:
        for index, model in enumerate(ordered):
            profile, out = model["model"], out_root / model["model"]
            if prefetch_next and index + 1 < len(ordered):
                downloads.submit(prefetch, environment, ordered[index + 1])
            started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            finished = _finished(out, run_key(environment, model))
            if finished and not rerun:
                record = {"profile": profile, "status": "skipped", "category": finished}
            else:
                set_aside(out)
                record = run_one(environment, model, out)
            for repo in sorted(checkpoints(model)):
                remaining[repo] -= 1
                if remaining[repo] == 0 and hf_policy == "delete_unused":
                    record.setdefault("checkpoints_deleted", []).append(
                        retention.delete_checkpoint(Path(environment["hf_hub_cache"]), repo))
            record = {**record, "host": socket.gethostname(), "position": index + 1, "started_at": started_at}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            records.append(record)
    return records


CATEGORIES = ("error", "config-error", "build-failed", "not-run", "acc-issue", "not-covered", "acc-inconclusive",
              "not-comparable", "perf-issue", "perf-inconclusive", "pass", "excluded", "smoke-fail", "smoke-pass")
EXCLUSIONS = "excluded.json"
PLAN = "plan.json"
HARNESS_FAILURES = ("error", "build-failed")


def write_plan(out_root: Path, selected: Sequence[str], config_errors: Sequence[Mapping[str, Any]],
               extra: Mapping[str, Any] | None = None) -> None:
    """Record every profile a batch must report, so a missing result shows as ``not-run`` (with ``extra``: the
    host, the assignment's digest, and the campaign inputs of a multi-host run)."""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / PLAN).write_text(json.dumps({"selected": list(selected), "config_errors": list(config_errors),
                                             **dict(extra or {})}, indent=2) + "\n")


def exit_code(records: Sequence[Mapping[str, Any]], config_errors: Sequence[Mapping[str, Any]],
              smoke: bool = False) -> int:
    """2 when profiles could not be configured, 1 when a run failed in the harness (error, build; in smoke
    mode any model short of smoke-pass), 0 otherwise (qualification outcomes such as acc-issue are
    results, not failures)."""
    if config_errors:
        return 2
    if smoke:
        return 0 if records and all(record.get("category") == "smoke-pass" for record in records) else 1
    return 1 if any(record.get("category") in HARNESS_FAILURES for record in records) else 0


def write_exclusions(out_root: Path, excluded: Sequence[Mapping[str, Any]]) -> None:
    """Record the profiles the machine's model list leaves out, so the summary shows them."""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / EXCLUSIONS).write_text(json.dumps(list(excluded), indent=2) + "\n")


def _row(directory: Path) -> dict[str, Any] | None:
    """The result in a profile directory; ``time`` is when that run started (re-judging rewrites reports,
    so a report's own start time, else the file time)."""
    for name in ("report.json", "build.json", "error.json"):
        path = directory / name
        if not path.is_file():
            continue
        value = json.loads(path.read_text())
        if name == "report.json":
            return {"task": value.get("task"), "category": value["verdict"]["category"],
                    "directory": str(directory), "repro": value.get("repro"), "l2": value.get("performance_l2"),
                    "time": float(value.get("started") or path.stat().st_mtime),
                    "accuracy": value.get("accuracy", []), "perf": value.get("performance_l1", []),
                    "backend": (value.get("reference") or {}).get("backend", ""),
                    "notes": "; ".join([*([f"coverage: {value['coverage']}"] if value.get("coverage") else []),
                                        *(f"{key}: {text[:100]}" for key, text in value.get("errors", {}).items())])}
        if name == "build.json" and value.get("status") != "failed":
            continue
        return {"task": value.get("task"), "category": "build-failed" if name == "build.json" else "error",
                "directory": str(directory), "repro": value.get("repro"),
                "time": path.stat().st_mtime,
                "accuracy": [], "perf": [], "backend": "", "notes": value.get("reason", "")}
    return None


def _accuracy_text(items: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(acc_value(item) for item in items if not item.get("informational"))


def ms(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.0f} ms" if value >= 100 else f"{value:.1f} ms" if value >= 1 else f"{value:.3f} ms"


def _perf_text(profile: str, items: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(f"{request_label(profile, item)}: TRTMC {ms((item.get('candidate') or {}).get('p50_ms'))} · "
                     f"native {ms((item.get('reference') or {}).get('p50_ms'))}" for item in items)


# The owner's four results, worst first: White, no valid comparison (no verdict: an error or a failed build; or the
# comparison does not apply: the native model below a benchmark's floor, a Task without an Acc check, timings that
# cannot be compared); Red, Acc or Perf worse than the native model beyond its margin; Yellow, Perf about equal
# (counts as a pass) or an Acc difference not shown either way; Green, a pass.
SIGNALS = ("white", "red", "yellow", "green")
SIGNAL_NAMES = {"white": "White", "red": "Red", "yellow": "Yellow", "green": "Green"}
NO_VERDICT = {"error", "config-error", "build-failed", "not-run", "excluded", "smoke-fail"}
NOT_COMPARED = {"not-comparable", "not-covered"}


def reported_perf(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The timed requests the results report: the catalog's own (the near-capacity request stays in the evidence,
    owner's choice for now)."""
    return [item for item in row.get("perf", []) if "near-capacity" not in str(item.get("request") or "")]


def request_label(profile: str, item: Mapping[str, Any]) -> str:
    """A timed request named without its model (``catalog``, ``catalog-en``)."""
    name = str(item.get("request") or item.get("reference_mode") or "")
    return name[len(profile) + 1:] if name.startswith(profile + "-") else name


def _judged(row: Mapping[str, Any], status: str) -> list[Mapping[str, Any]]:
    return [item for item in row.get("accuracy", []) if item.get("status") == status and not item.get("informational")]


def signal(row: Mapping[str, Any]) -> str:
    """The row's result among ``SIGNALS``."""
    category = row["category"]
    if category in NO_VERDICT:
        return "white"
    perf = reported_perf(row)
    lights = [item.get("light") for item in perf]
    if category == "acc-issue" or _judged(row, "fail") or "red" in lights:
        return "red"
    if category in NOT_COMPARED or "white" in lights or (category == "perf-inconclusive" and not perf):
        return "white"
    if category == "acc-inconclusive" or "yellow" in lights:
        return "yellow"
    return "green"


def signal_reason(profile: str, row: Mapping[str, Any]) -> str:
    """Why the row is not Green, in one plain sentence per benchmark or request."""
    result, perf = signal(row), reported_perf(row)
    if result == "green":
        return plain(row.get("notes", ""))[:300]
    if row["category"] in NO_VERDICT:
        errors = [str(item.get("error") or "; ".join(item.get("reasons", []))) for item in _judged(row, "error")]
        return plain(next((text for text in [row.get("notes", ""), *errors] if text), row["category"]))[:300]
    parts: list[str] = []
    if result == "red":
        for item in _judged(row, "fail"):
            why = ("TRTMC worse than native beyond the margin" if "trtmc_score" in (item.get("metrics") or {})
                   else "; ".join(item.get("reasons", [])) or "failed")
            parts.append(f"{item['suite']}: {why}" + "".join(f" ({plain(note)})" for note in item.get("notes", [])))
        parts += [f"{request_label(profile, item)}: TRTMC slower than native" for item in perf if item.get("light") == "red"]
    elif result == "white":
        parts += [f"{item['suite']}: {plain('; '.join(item.get('reasons', [])))}"
                  for status in ("not-comparable", "not-covered") for item in _judged(row, status)]
        parts += [f"{request_label(profile, item)}: timings not comparable ({plain((item.get('reasons') or [''])[0])})"
                  for item in perf if item.get("light") == "white"]
    else:
        parts += [f"{item['suite']}: difference not shown either way" for item in _judged(row, "inconclusive")]
        parts += [f"{request_label(profile, item)}: TRTMC about equal to native" for item in perf
                  if item.get("light") == "yellow"]
    notes = plain(row.get("notes", ""))  # coverage, a regression against a baseline
    return "; ".join(part for part in [*parts, notes] if part)[:400] or row["category"]


def run_context(roots: Sequence[Path]) -> str:
    """Which run the roots hold: each root's host, the assignment and campaign inputs it ran under (``plan.json``),
    and when its first and last results started."""
    import datetime
    import hashlib

    def day(stamp: float) -> str:
        return datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    parts = []
    for root in roots:
        plan = json.loads((root / PLAN).read_text()) if (root / PLAN).is_file() else {}
        times = [row["time"] for row in (_row(path) for path in root.iterdir() if path.is_dir()) if row]
        inputs = plan.get("inputs")
        digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:12] if inputs else ""
        parts.append(f"{root.name}: host {plan.get('host') or '-'}"
                     + (f", assignment {str(plan['assignment'])[:12]}" if plan.get("assignment") else "")
                     + (f", campaign inputs {digest}" if digest else "")
                     + (f", {day(min(times))} to {day(max(times))}" if times else ""))
    return " · ".join(parts)


def collect(roots: Sequence[Path]) -> tuple[dict[str, dict[str, Any]], collections.Counter, dict[str, int]]:
    """The latest result of every planned or reported profile under the given roots; profiles a
    root's model list excluded are listed unless another root holds a result for them."""
    rows = {}
    for root in roots:
        path = root / EXCLUSIONS
        for item in json.loads(path.read_text()) if path.is_file() else []:
            rows[item["profile"]] = {"task": item.get("task"), "category": "excluded", "accuracy": [], "perf": [],
                                     "backend": "", "notes": item.get("reason", ""), "root": root.name}
        plan = json.loads((root / PLAN).read_text()) if (root / PLAN).is_file() else {}
        for name in plan.get("selected", []):
            rows.setdefault(name, {"task": None, "category": "not-run", "accuracy": [], "perf": [], "backend": "",
                                   "notes": "planned, no result", "root": root.name})
        for item in plan.get("config_errors", []):
            rows[item["profile"]] = {"task": None, "category": "config-error", "accuracy": [], "perf": [],
                                     "backend": "", "notes": item.get("reason", ""), "root": root.name}
    for root in roots:
        for directory in sorted(path for path in root.iterdir() if path.is_dir() and not KEPT_ASIDE.search(path.name)):
            row = _row(directory)
            if row and row["time"] >= rows.get(directory.name, {}).get("time", float("-inf")):
                rows[directory.name] = {**row, "root": root.name}  # the latest run of a profile wins
    counts = collections.Counter(row["category"] for row in rows.values())
    ordered = [*CATEGORIES, *sorted(set(counts) - set(CATEGORIES))]
    return rows, counts, {category: position for position, category in enumerate(ordered)}


REGRESSION_MARGIN_PERCENT = 5.0


def annotate_regressions(rows: Mapping[str, dict[str, Any]], baseline: Mapping[str, Mapping[str, Any]],
                         margin_percent: float = REGRESSION_MARGIN_PERCENT) -> None:
    """Compare TRTMC p50 per profile and native mode with a baseline run (for example the previous
    release); slower by more than the margin is noted as a regression (the category is unchanged)."""
    for profile, row in rows.items():
        key = lambda item: f"{item.get('reference_mode')}/{item.get('request') or ''}".rstrip("/")  # noqa: E731
        previous = {key(item): item for item in (baseline.get(profile) or {}).get("perf", [])}
        for item in row.get("perf", []):
            before = (previous.get(key(item)) or {}).get("candidate", {}).get("p50_ms")
            now = item.get("candidate", {}).get("p50_ms")
            if before and now:
                change = (now / before - 1) * 100
                row.setdefault("baseline", {})[key(item)] = change
                if change > margin_percent:
                    row["notes"] = (f"regression: TRTMC {key(item)} p50 +{change:.1f}% vs baseline; "
                                    + row.get("notes", "")).strip("; ")


def summary(roots: Sequence[Path], baseline: Sequence[Path] = ()) -> tuple[str, collections.Counter]:
    """Markdown summary of ``collect`` (optionally compared with baseline roots)."""
    rows, counts, rank = collect(roots)
    if baseline:
        annotate_regressions(rows, collect(baseline)[0])
    results = collections.Counter(signal(row) for row in rows.values())
    by_task: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows.values():
        by_task[row["task"] or "-"][signal(row)] += 1
    names = [SIGNAL_NAMES[result] for result in SIGNALS]
    lines = ["# TRTMC vs native qualification", "", f"{len(rows)} models from {', '.join(r.name for r in roots)}: "
             f"{results['green'] + results['yellow']} pass (Green + Yellow).", "",
             "Green: pass. Yellow: Perf about equal to native (counts as a pass) or an Acc difference not shown either "
             "way. Red: Acc or Perf worse than native beyond its margin. White: no valid comparison (an error or a "
             "failed build, or the comparison does not apply).", "",
             "| result | models |", "|---|---|", *(f"| {SIGNAL_NAMES[r]} | {results[r]} |" for r in SIGNALS),
             "", "## By Task", "", "| Task | " + " | ".join(names) + " |", "|---|" + "---|" * len(names),
             *(f"| {task} | " + " | ".join(str(by_task[task].get(r) or "") for r in SIGNALS) + " |"
               for task in sorted(by_task)),
             "", "## Per model", "", "| result | model | Task | host | Acc (TRTMC · native) | Perf p50 (TRTMC · native) | "
             "reason |", "|---|---|---|---|---|---|---|"]
    for profile in sorted(rows, key=lambda p: (SIGNALS.index(signal(rows[p])), rows[p]["task"] or "", p)):
        row = rows[profile]
        reason = signal_reason(profile, row).replace("|", "/").replace("\n", " ")
        lines.append(f"| {SIGNAL_NAMES[signal(row)]} | {profile} | {row['task'] or '-'} | {row['root']} | "
                     f"{_accuracy_text(row['accuracy'])} | {_perf_text(profile, reported_perf(row))} | {reason} |")
    return "\n".join(lines) + "\n", counts


REMOTE_ROOT = re.compile(r"^(?:(?P<name>[\w.-]+)=)?(?P<host>[\w.@-]+):(?P<path>/.*)$")
RESULT_FILES = ("report.json", "build.json", "error.json", EXCLUSIONS, PLAN)
EVIDENCE_FILES = ("report.md", "phase-errors.log", "build.log", "error.log", "server.log", "result.json")
MAX_EVIDENCE_BYTES = "5M"


def fetch_roots(specs: Sequence[str], ssh: str, into: Path, *, evidence: bool = False) -> list[Path]:
    """Result roots for ``summary``: local paths as given; ``[NAME=][USER@]HOST:/PATH`` fetched over ssh
    into ``into/NAME`` (default the host): result files, plus logs with ``evidence``."""
    import io
    import shlex
    import subprocess
    import tarfile

    from .config import ConfigError

    roots = []
    for spec in specs:
        match = REMOTE_ROOT.match(spec)
        if not match or Path(spec).exists():
            roots.append(Path(spec))
            continue
        target = into / (match["name"] or match["host"].rsplit("@", 1)[-1])
        target.mkdir(parents=True, exist_ok=True)
        names = " -o ".join(f"-name {shlex.quote(name)}" for name in RESULT_FILES)
        if evidence:  # result files up to <profile>/, logs below it
            logs = " -o ".join(f"-name {shlex.quote(name)}" for name in EVIDENCE_FILES)
            selection = (f"-maxdepth 7 -type f \\( \\( ! -path './*/*/*' \\( {names} \\) \\) -o "
                         f"\\( -size -{MAX_EVIDENCE_BYTES} \\( {logs} \\) \\) \\)")
        else:
            selection = f"-maxdepth 2 -type f \\( {names} \\)"
        command = (f"cd {shlex.quote(match['path'])} && find . {selection} -print0 "
                   "| tar --null -T - -cf -")
        fetched = subprocess.run([*shlex.split(ssh), match["host"], command], capture_output=True, timeout=1800)
        if fetched.returncode:
            raise ConfigError(f"cannot fetch {spec}: {fetched.stderr.decode(errors='replace').strip()[-300:]}")
        with tarfile.open(fileobj=io.BytesIO(fetched.stdout)) as archive:
            archive.extractall(target, filter="data")  # no paths outside the target
        roots.append(target)
    return roots


def parse_shard(value: str) -> tuple[int, int]:
    index, _, count = value.partition("/")
    if not (index.isdigit() and count.isdigit() and 0 <= int(index) < int(count)):
        print(f"trtmc-aiperf-qual: --shard must be INDEX/COUNT with 0 <= INDEX < COUNT, got {value!r}", file=sys.stderr)
        raise SystemExit(2)
    return int(index), int(count)
