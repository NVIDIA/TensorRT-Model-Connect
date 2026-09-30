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
from .services import reference_python

KEPT_ASIDE = re.compile(r"\.\d{10}$")  # <profile>.<unix time> of a previous run


def qualify(model: dict[str, Any], environment: Environment, out: Path) -> dict[str, Any]:
    """runner.qualify, imported on use: ``summary`` runs where AIPerf is not installed."""
    from .runner import qualify as run

    return run(model, environment, out)


def _error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:300]


def run_one(environment: Environment, model: dict[str, Any], out: Path) -> dict[str, Any]:
    """Build the bundle when missing, qualify, and apply the bundle retention policy."""
    started = time.time()
    out.mkdir(parents=True, exist_ok=True)
    bundle_policy, _ = retention.policies(environment)
    record: dict[str, Any] = {"profile": model["model"], "task": model.get("task")}
    try:
        python = reference_python(environment, model)  # the family environment builds and serves references
        prefetch(environment, model)  # outside the GPU lock the build takes
        build = bundles.ensure_bundle(environment, model, python, out)
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
        if retention.should_delete_bundle(bundle_policy, record["category"]):
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
                if out.exists():
                    out.rename(out.with_name(f"{profile}.{int(time.time())}"))
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


CATEGORIES = ("pass", "acc-issue", "acc-session-state", "acc-inconclusive", "perf-issue", "perf-inconclusive",
              "not-comparable", "error", "build-failed", "excluded")
EXCLUSIONS = "excluded.json"


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
                    "time": float(value.get("started") or path.stat().st_mtime),
                    "accuracy": value.get("accuracy", []), "perf": value.get("performance_l1", []),
                    "backend": (value.get("reference") or {}).get("backend", ""),
                    "notes": "; ".join(f"{key}: {text[:100]}" for key, text in value.get("errors", {}).items())}
        if name == "build.json" and value.get("status") != "failed":
            continue
        return {"task": value.get("task"), "category": "build-failed" if name == "build.json" else "error",
                "time": path.stat().st_mtime,
                "accuracy": [], "perf": [], "backend": "", "notes": value.get("reason", "")}
    return None


def _accuracy_text(items: Sequence[Mapping[str, Any]]) -> str:
    def one(item: Mapping[str, Any]) -> str:
        extra = "".join(f", {label} {value}" for label, value in (
            ("ref@prec", (item.get("noise_floor") or {}).get("passed")),
            ("isolated", (item.get("isolated_check") or {}).get("passed"))) if value is not None)
        return f"{item['suite']} {item['passed']}/{item['samples']} (need {item['required_passes']}{extra})"
    return "; ".join(one(item) for item in items)


def _perf_text(items: Sequence[Mapping[str, Any]]) -> str:
    return ", ".join(f"{item['reference_mode']} {item['light']}"
                     + (f" {item['speedup']:.2f}x" if item.get("speedup") else "") for item in items)


def summary(roots: Sequence[Path]) -> tuple[str, collections.Counter]:
    """Markdown summary of the latest result of every profile under the given roots; profiles a root's
    model list excluded are listed unless another root holds a result for them."""
    rows = {}
    for root in roots:
        path = root / EXCLUSIONS
        for item in json.loads(path.read_text()) if path.is_file() else []:
            rows[item["profile"]] = {"task": item.get("task"), "category": "excluded", "accuracy": [], "perf": [],
                                     "backend": "", "notes": item.get("reason", ""), "root": root.name}
    for root in roots:
        for directory in sorted(path for path in root.iterdir() if path.is_dir() and not KEPT_ASIDE.search(path.name)):
            row = _row(directory)
            if row and row["time"] >= rows.get(directory.name, {}).get("time", float("-inf")):
                rows[directory.name] = {**row, "root": root.name}  # the latest run of a profile wins
    counts = collections.Counter(row["category"] for row in rows.values())
    ordered = [*CATEGORIES, *sorted(set(counts) - set(CATEGORIES))]
    rank = {category: position for position, category in enumerate(ordered)}
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
        lines.append(f"| {profile} | {row['task'] or ''} | {row['root']} | {row['category']} | "
                     f"{_accuracy_text(row['accuracy'])} | {_perf_text(row['perf'])} | {row['backend']} | {notes} |")
    return "\n".join(lines) + "\n", counts


REMOTE_ROOT = re.compile(r"^(?:(?P<name>[\w.-]+)=)?(?P<host>[\w.@-]+):(?P<path>/.*)$")
RESULT_FILES = ("report.json", "build.json", "error.json", EXCLUSIONS)


def fetch_roots(specs: Sequence[str], ssh: str, into: Path) -> list[Path]:
    """Result roots for ``summary``: local paths as given; ``[NAME=][USER@]HOST:/PATH`` fetched over ssh
    into ``into/NAME`` (default the host), result files only."""
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
        command = (f"cd {shlex.quote(match['path'])} && find . -maxdepth 2 -type f \\( {names} \\) -print0 "
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
