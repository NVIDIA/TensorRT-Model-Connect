# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The formal run on several GPU hosts (DESIGN.md Section 9): a frozen profile -> host assignment from the
ledger, each host's run list, and the checks that the hosts' result roots merge into one matrix."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .campaign import KEPT_ASIDE, PLAN, _groups, code_digests, dependencies_digest, harness_digest
from .config import ConfigError, Environment

RULE = ("checkpoint groups (profiles sharing a checkpoint stay together) by ledger seconds, longest first, each to "
        "the host with less predicted time; ties by group name, then host order; within a host, the assigned order")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(assignment: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical(dict(assignment)).encode()).hexdigest()


def assign(models: Sequence[Mapping[str, Any]], ledger: Mapping[str, float], hosts: Sequence[str]) -> dict[str, Any]:
    """The frozen assignment of ``models`` to ``hosts`` (RULE): every profile must have a ledger time."""
    if len(set(hosts)) != len(hosts) or not hosts:
        raise ConfigError(f"hosts must be distinct and given: {list(hosts)}")
    missing = sorted(model["model"] for model in models if model["model"] not in ledger)
    if missing:
        raise ConfigError(f"no ledger time for {', '.join(missing)}")
    groups = sorted(_groups(models), key=lambda group: (-sum(float(ledger[m["model"]]) for m in group), group[0]["model"]))
    lists: dict[str, list[str]] = {host: [] for host in hosts}
    load = {host: 0.0 for host in hosts}
    for group in groups:
        host = min(hosts, key=lambda name: (load[name], hosts.index(name)))
        lists[host] += [model["model"] for model in group]
        load[host] += sum(float(ledger[model["model"]]) for model in group)
    return {"rule": RULE, "hosts": lists, "predicted_s": {host: round(load[host]) for host in hosts},
            "ledger": {name: float(ledger[name]) for name in sorted(model["model"] for model in models)}}


def host_models(assignment: Mapping[str, Any], host: str,
                models: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The host's profiles in the assigned order; the assignment must cover exactly the selected profiles."""
    lists = assignment["hosts"]
    if host not in lists:
        raise ConfigError(f"host {host!r} is not in the assignment ({', '.join(lists)})")
    assigned = [name for names in lists.values() for name in names]
    selected = {model["model"]: model for model in models}
    duplicated = sorted({name for name in assigned if assigned.count(name) > 1})
    if duplicated or set(assigned) != set(selected):
        raise ConfigError(f"the assignment does not partition the selected profiles: duplicated {duplicated}, "
                          f"unassigned {sorted(set(selected) - set(assigned))}, "
                          f"unknown {sorted(set(assigned) - set(selected))}")
    return [selected[name] for name in lists[host]]


def campaign_inputs(environment: Environment) -> dict[str, Any]:
    """What every host must share: the harness, the serving and TRTMC code and runtime, and the resolved
    dependencies of the harness and serving interpreters (families' own code is in each profile's run key)."""
    code = code_digests(environment, {})
    code.pop("family", None)
    serve = str(environment.values.get("serve_python") or "")
    return {"harness": harness_digest(), "code": code,
            "dependencies": {"harness": dependencies_digest(sys.executable),
                             "serving": dependencies_digest(serve) if serve and Path(serve).exists() else ""}}


def check_resume(out_root: Path, plan: Mapping[str, Any]) -> None:
    """A root that already holds results continues only under the same assignment, host, and campaign inputs:
    otherwise its results would be relabelled (the run keys do not include them)."""
    path = out_root / PLAN
    if not path.is_file():
        return
    existing = json.loads(path.read_text())
    if all(existing.get(key) == plan.get(key) for key in ("assignment", "host", "inputs")):
        return
    held = [directory.name for directory in out_root.iterdir()
            if directory.is_dir() and not KEPT_ASIDE.search(directory.name) and _final(directory)]
    if held:
        raise ConfigError(f"{out_root} holds results run under another assignment, host, or campaign inputs "
                          f"({', '.join(sorted(held)[:5])}...): use a fresh --out-root or --rerun")


def set_aside_results(out_root: Path) -> list[str]:
    """``run-all --rerun`` under an assignment: every previous result in the root is set aside before the new plan
    is written, so an interrupted rerun leaves no old result to be read under the new plan."""
    from .campaign import set_aside

    held = [directory for directory in sorted(out_root.iterdir()) if directory.is_dir()
            and not KEPT_ASIDE.search(directory.name) and _final(directory)] if out_root.is_dir() else []
    for directory in held:
        set_aside(directory)
    return [directory.name for directory in held]


def _final(directory: Path) -> str | None:
    """The kind of final result in a profile directory (what ``summary`` reads): the report's mode,
    "build-failed", "error", or None."""
    if (directory / "report.json").is_file():
        return json.loads((directory / "report.json").read_text()).get("mode")
    build = directory / "build.json"
    if build.is_file() and json.loads(build.read_text()).get("status") == "failed":
        return "build-failed"
    return "error" if (directory / "error.json").is_file() else None


def merge_check(assignment: Mapping[str, Any], roots: Sequence[Path], mode: str = "formal") -> list[str]:
    """Why the result roots do not merge into the assignment's matrix ([] when they do): each host's root ran
    under this assignment with the same campaign inputs, holds only its own profiles, and every profile has
    exactly one result of ``mode`` (a formal report, or a smoke one when checking smoke roots; a failed build; a
    harness error) across the roots."""
    problems: list[str] = []
    expected = digest(assignment)
    owner: dict[str, str] = {}
    inputs: dict[str, str] = {}
    seen_hosts: list[str] = []
    for root in roots:
        plan = json.loads((root / PLAN).read_text()) if (root / PLAN).is_file() else {}
        host = plan.get("host")
        if plan.get("assignment") != expected or host not in assignment["hosts"]:
            problems.append(f"{root}: not run under this assignment ({plan.get('assignment')}, host {host})")
            continue
        seen_hosts.append(host)
        inputs[str(root)] = canonical(plan.get("inputs"))
        assigned = set(assignment["hosts"][host])
        for directory in sorted(path for path in root.iterdir() if path.is_dir() and not KEPT_ASIDE.search(path.name)):
            kind = _final(directory)
            if kind is None:
                continue
            if directory.name not in assigned:
                problems.append(f"{root}: {directory.name} is assigned to another host")
            elif kind not in (mode, "build-failed", "error"):
                problems.append(f"{root}: {directory.name} is a {kind} result")
            elif directory.name in owner:
                problems.append(f"{directory.name}: results in {owner[directory.name]} and {root}")
            else:
                owner[directory.name] = str(root)
    for host in assignment["hosts"]:
        if seen_hosts.count(host) != 1:
            problems.append(f"host {host}: {seen_hosts.count(host)} result roots")
    if len(set(inputs.values())) > 1:
        problems.append("the roots ran with different campaign inputs (harness, code, runtime, or dependencies)")
    for name in sorted(name for names in assignment["hosts"].values() for name in names):
        if name not in owner:
            problems.append(f"{name}: no {mode} result")
    return problems
