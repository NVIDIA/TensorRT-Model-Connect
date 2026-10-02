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
import json
import re
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
from .report import counted
from .services import reference_python

KEPT_ASIDE = re.compile(r"\.\d{10}$")  # <profile>.<unix time> of a previous run


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


def run_one(environment: Environment, model: dict[str, Any], out: Path) -> dict[str, Any]:
    """Build the bundle when missing, qualify, and apply the bundle retention policy. A previous run
    in ``out`` is set aside first."""
    started = time.time()
    set_aside(out)
    out.mkdir(parents=True, exist_ok=True)
    bundle_policy, _ = retention.policies(environment)
    record: dict[str, Any] = {"profile": model["model"], "task": model.get("task")}
    try:
        python = reference_python(environment, model)  # the family environment builds and serves references
        prefetch(environment, model)  # outside the GPU lock the build takes
        build = bundles.ensure_bundle(environment, model, python, out)
        from .runner import FALLBACK_SEQUENCE_LENGTH, shorter_candidate  # on use: summary runs without AIPerf

        shorter = (shorter_candidate(model, f"the {model['candidate'].get('max_sequence_length')}-token "
                                            f"benchmark bundle did not build ({str(build.get('reason'))[:200]})")
                   if build["status"] == "failed" else None)
        if shorter is not None:  # a family that cannot build the benchmark length may build a shorter one
            retry = bundles.ensure_bundle(environment, shorter, python, out / f"retry-{FALLBACK_SEQUENCE_LENGTH}")
            if retry["status"] != "failed":
                model, build = shorter, {**retry, "first_build": build}
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


def _finished(out: Path) -> str | None:
    """Category of a final result in ``out`` (a verdict or a failed build), else None."""
    report, build = out / "report.json", out / "build.json"
    if report.is_file():
        return json.loads(report.read_text()).get("verdict", {}).get("category")
    if build.is_file() and json.loads(build.read_text()).get("status") == "failed":
        return "build-failed"
    return None


def run_all(environment: Environment, models: Sequence[dict[str, Any]], out_root: Path, *,
            rerun: bool = False, prefetch_next: bool = True) -> list[dict[str, Any]]:
    _, hf_policy = retention.policies(environment)
    ordered = order(models)
    remaining = collections.Counter(repo for model in ordered for repo in checkpoints(model))
    out_root.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=1) as downloads, open(out_root / "campaign.jsonl", "a") as log:
        for index, model in enumerate(ordered):
            profile, out = model["model"], out_root / model["model"]
            if prefetch_next and index + 1 < len(ordered):
                downloads.submit(prefetch, environment, ordered[index + 1])
            finished = _finished(out)
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
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            records.append(record)
    return records


CATEGORIES = ("error", "config-error", "build-failed", "not-run", "acc-issue", "acc-session-state", "perf-issue",
              "acc-inconclusive", "perf-inconclusive", "not-comparable", "pass", "excluded")
EXCLUSIONS = "excluded.json"
PLAN = "plan.json"
HARNESS_FAILURES = ("error", "build-failed")


def write_plan(out_root: Path, selected: Sequence[str], config_errors: Sequence[Mapping[str, Any]]) -> None:
    """Record every profile a batch must report, so a missing result shows as ``not-run``."""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / PLAN).write_text(json.dumps({"selected": list(selected), "config_errors": list(config_errors)},
                                            indent=2) + "\n")


def exit_code(records: Sequence[Mapping[str, Any]], config_errors: Sequence[Mapping[str, Any]]) -> int:
    """2 when profiles could not be configured, 1 when a run failed in the harness (error, build),
    0 otherwise (qualification outcomes such as acc-issue are results, not failures)."""
    if config_errors:
        return 2
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


def _isolated(check: Mapping[str, Any] | None) -> Any:
    """The isolated re-check's pass count, else its status (aggregate-only family results)."""
    if not check:
        return None
    return check["passed"] if check.get("passed") is not None else check.get("status")


def _accuracy_text(items: Sequence[Mapping[str, Any]]) -> str:
    def one(item: Mapping[str, Any]) -> str:
        extra = "".join(f", {label} {value}" for label, value in (
            ("isolated", _isolated(item.get("isolated_check"))),
            ("informational", "yes" if item.get("informational") else None)) if value is not None)
        need = (f"need {item['required_passes']}" if item.get("required_passes") is not None
                else f"family gate {json.dumps(item.get('gate', {}))}")
        status = f"{item['status']} " if item.get("status") else ""
        return f"{item['suite']} {status}{counted(item)} ({need}{extra})"
    return "; ".join(one(item) for item in items)


def _perf_text(items: Sequence[Mapping[str, Any]]) -> str:
    return ", ".join(f"{item['reference_mode']} {item['light']}"
                     + (f" {item['speedup']:.2f}x" if item.get("speedup") else "") for item in items)


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
        previous = {item.get("reference_mode"): item for item in (baseline.get(profile) or {}).get("perf", [])}
        for item in row.get("perf", []):
            before = (previous.get(item.get("reference_mode")) or {}).get("candidate", {}).get("p50_ms")
            now = item.get("candidate", {}).get("p50_ms")
            if before and now:
                change = (now / before - 1) * 100
                row.setdefault("baseline", {})[item["reference_mode"]] = change
                if change > margin_percent:
                    row["notes"] = (f"regression: TRTMC {item['reference_mode']} p50 +{change:.1f}% vs baseline; "
                                    + row.get("notes", "")).strip("; ")


def summary(roots: Sequence[Path], baseline: Sequence[Path] = ()) -> tuple[str, collections.Counter]:
    """Markdown summary of ``collect`` (optionally compared with baseline roots)."""
    rows, counts, rank = collect(roots)
    if baseline:
        annotate_regressions(rows, collect(baseline)[0])
    shown = sorted(counts, key=rank.get)
    by_task: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows.values():
        by_task[row["task"] or "-"][row["category"]] += 1
    lines = ["# TRTMC vs native qualification", "", f"{len(rows)} models from {', '.join(r.name for r in roots)}.",
             "", "| category | models |", "|---|---|", *(f"| {c} | {counts[c]} |" for c in shown),
             "", "## By Task", "", "| Task | " + " | ".join(shown) + " |", "|---|" + "---|" * len(shown),
             *(f"| {task} | " + " | ".join(str(by_task[task].get(c) or "") for c in shown) + " |"
               for task in sorted(by_task)),
             "", "## Per model", "", "| model | Task | root | category | Acc | Perf L1 (speedup vs native) | "
             "reference | notes |", "|---|---|---|---|---|---|---|---|"]
    for profile in sorted(rows, key=lambda p: (rank[rows[p]["category"]], rows[p]["task"] or "", p)):
        row = rows[profile]
        notes = row["notes"].replace("|", "/").replace("\n", " ")
        lines.append(f"| {profile} | {row['task'] or '-'} | {row['root']} | {row['category']} | "
                     f"{_accuracy_text(row['accuracy'])} | {_perf_text(row['perf'])} | {row['backend']} | {notes} |")
    return "\n".join(lines) + "\n", counts


REMOTE_ROOT = re.compile(r"^(?:(?P<name>[\w.-]+)=)?(?P<host>[\w.@-]+):(?P<path>/.*)$")
RESULT_FILES = ("report.json", "build.json", "error.json", EXCLUSIONS, PLAN)
EVIDENCE_FILES = ("report.md", "phase-errors.log", "build.log", "error.log", "server.log", "result.json")
MAX_EVIDENCE_BYTES = "5M"


def fetch_roots(specs: Sequence[str], ssh: str, into: Path, *, evidence: bool = False) -> list[Path]:
    """Result roots for ``summary``: local paths as given; ``[NAME=][USER@]HOST:/PATH`` fetched over ssh
    into ``into/NAME`` (default the host): result files, plus logs and family results with ``evidence``."""
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
        if evidence:  # result files up to <profile>/, logs and family results below it
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
