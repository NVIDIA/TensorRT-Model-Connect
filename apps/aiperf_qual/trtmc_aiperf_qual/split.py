# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The formal run on several GPU hosts: a frozen profile -> host assignment from the
ledger, each host's run list, and the checks that the hosts' result roots merge into one matrix.

A campaign can instead run as fragments, for a coordinator that hands profiles to hosts as they free up (a work
queue): each fragment (one attempt at some profiles on one host) runs ``run-all --campaign --fragment --host`` in a
root of its own, and the coordinator's manifest names the campaign's profiles and the fragments it accepted; the
roots merge when every profile has exactly one result among the accepted fragments."""

from __future__ import annotations

import hashlib
import json
import re
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


# The packages whose versions every host must share: TensorRT, and the native reference's torch and transformers.
PINNED_PACKAGES = ("tensorrt", "torch", "transformers")


def pinned_versions(python: str) -> dict[str, str]:
    """The interpreter's versions of PINNED_PACKAGES (from ``pip freeze``); {} when it cannot tell."""
    import subprocess

    try:
        frozen = subprocess.run([python, "-m", "pip", "freeze", "--all"], capture_output=True, text=True,
                                timeout=300, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    versions = {}
    for line in frozen.splitlines():
        name, separator, version = line.strip().partition("==")
        name = name.lower().replace("_", "-")
        if separator and name in PINNED_PACKAGES:
            versions[name] = version
    return versions


def campaign_inputs(environment: Environment) -> dict[str, Any]:
    """What every host must share: the harness, the serving and TRTMC sources, and the TensorRT, torch and
    transformers versions of the harness and serving interpreters (families' own code is in each profile's run
    key). ``informational`` records what may differ between hosts without stopping a merge: the built runtime
    (a rebuild of the same sources need not be byte-identical) and the interpreters' other resolved dependencies."""
    code = code_digests(environment, {})
    code.pop("family", None)
    runtime = code.pop("runtime", "")
    serve = str(environment.values.get("serve_python") or "")
    serving = serve if serve and Path(serve).exists() else ""
    return {"harness": harness_digest(), "code": code,
            "versions": {"harness": pinned_versions(sys.executable),
                         "serving": pinned_versions(serving) if serving else {}},
            "informational": {"runtime": runtime,
                              "dependencies": {"harness": dependencies_digest(sys.executable),
                                               "serving": dependencies_digest(serving) if serving else ""}}}


def required_inputs(inputs: Any) -> Any:
    """The part of a root's campaign inputs that must match across hosts. Inputs recorded before the split into
    required and informational parts are compared whole."""
    if isinstance(inputs, Mapping) and "informational" in inputs:
        return {key: value for key, value in inputs.items() if key != "informational"}
    return inputs


def informational_differences(roots: Sequence[Path]) -> list[str]:
    """Which informational campaign inputs differ across ``roots`` (reported, never a merge problem)."""
    seen: dict[str, set[str]] = {}
    for root in roots:
        plan = json.loads((root / PLAN).read_text()) if (root / PLAN).is_file() else {}
        informational = (plan.get("inputs") or {}).get("informational") if isinstance(plan.get("inputs"), Mapping) else None
        if not isinstance(informational, Mapping):
            continue
        for key, value in informational.items():
            values = value.items() if isinstance(value, Mapping) else [("", value)]
            for name, item in values:
                seen.setdefault(f"{key}/{name}".rstrip("/"), set()).add(canonical(item))
    return [f"the hosts' {name} differ (informational: TensorRT, torch, transformers and the sources match)"
            for name, values in sorted(seen.items()) if len(values) > 1]


FRAGMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def fragment_plan(campaign: str, fragment: str, host: str, environment: Environment) -> dict[str, Any]:
    """What a fragment's plan records besides its profiles: the campaign, the fragment, the host, and the campaign
    inputs every fragment of the campaign must share."""
    for name, value in (("--campaign", campaign), ("--fragment", fragment), ("--host", host)):
        if not FRAGMENT_ID.fullmatch(str(value or "")):
            raise ConfigError(f"{name} must be 1-128 letters, digits, '.', '_' or '-', not {value!r}")
    return {"campaign": campaign, "fragment": fragment, "host": host, "inputs": campaign_inputs(environment)}


def check_resume(out_root: Path, plan: Mapping[str, Any]) -> None:
    """A root that already holds results continues only under the same assignment (or campaign and fragment), host,
    and campaign inputs: otherwise its results would be relabelled (the run keys do not include them)."""
    path = out_root / PLAN
    if not path.is_file():
        return
    existing = json.loads(path.read_text())
    if all(existing.get(key) == plan.get(key) for key in ("assignment", "campaign", "fragment", "host", "inputs")):
        return
    held = [directory.name for directory in out_root.iterdir()
            if directory.is_dir() and not KEPT_ASIDE.search(directory.name) and _final(directory)]
    if held:
        raise ConfigError(f"{out_root} holds results run under another assignment or fragment, host, or campaign "
                          f"inputs ({', '.join(sorted(held)[:5])}...): use a fresh --out-root or --rerun")


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
        inputs[str(root)] = canonical(required_inputs(plan.get("inputs")))
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
        problems.append("the roots ran with different campaign inputs (harness, code, or the TensorRT, torch or "
                        "transformers version)")
    for name in sorted(name for names in assignment["hosts"].values() for name in names):
        if name not in owner:
            problems.append(f"{name}: no {mode} result")
    return problems


def _plan(root: Path) -> dict[str, Any]:
    return json.loads((root / PLAN).read_text()) if (root / PLAN).is_file() else {}


def _manifest_problems(manifest: Mapping[str, Any]) -> list[str]:
    campaign, expected, accepted = manifest.get("campaign"), manifest.get("expected"), manifest.get("accepted")
    problems = []
    if not isinstance(campaign, str) or not FRAGMENT_ID.fullmatch(campaign):
        problems.append(f"the manifest's campaign is not an identifier: {campaign!r}")
    for key, value in (("expected", expected), ("accepted", accepted)):
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            problems.append(f"the manifest's {key} is not a list of names")
        elif len(set(value)) != len(value):
            problems.append(f"the manifest's {key} lists a name twice")
    return problems


def accepted_roots(manifest: Mapping[str, Any], roots: Sequence[Path]) -> list[Path]:
    """The roots of the manifest's accepted fragments: a fragment it did not accept (a superseded attempt, or a
    late result of one) is left out of the merge."""
    accepted = set(manifest.get("accepted") or [])
    return [root for root in roots
            if _plan(root).get("campaign") == manifest.get("campaign") and _plan(root).get("fragment") in accepted]


def fragment_merge_check(manifest: Mapping[str, Any], roots: Sequence[Path], mode: str = "formal") -> list[str]:
    """Why the fragments' roots do not merge into the manifest's campaign ([] when they do). The manifest is
    ``{"campaign": ID, "expected": [profiles], "accepted": [fragments]}``. Each accepted fragment has exactly one root,
    run in this campaign with the same campaign inputs as the others; each holds only the profiles it was given;
    every expected profile has exactly one result among them (a result of ``mode``, a failed build, a harness error,
    or the configuration error the fragment recorded), and none holds a profile outside the campaign. Roots of other
    campaigns are problems; roots of fragments the manifest did not accept are left out."""
    problems = _manifest_problems(manifest)
    if problems:
        return problems
    expected, accepted = set(manifest["expected"]), manifest["accepted"]
    found: dict[str, list[Path]] = {}
    for root in roots:
        plan = _plan(root)
        if plan.get("campaign") != manifest["campaign"]:
            problems.append(f"{root}: not run in campaign {manifest['campaign']} ({plan.get('campaign')})")
        elif plan.get("fragment") in accepted:
            found.setdefault(plan["fragment"], []).append(root)
    owner: dict[str, str] = {}
    inputs: dict[str, str] = {}

    def claim(name: str, root: Path) -> None:
        if name not in expected:
            problems.append(f"{root}: {name} is not in the campaign")
        elif name in owner:
            problems.append(f"{name}: results in {owner[name]} and {root}")
        else:
            owner[name] = str(root)

    for fragment in accepted:
        if len(found.get(fragment, [])) != 1:
            problems.append(f"fragment {fragment}: {len(found.get(fragment, []))} result roots")
            continue
        root = found[fragment][0]
        plan = _plan(root)
        inputs[str(root)] = canonical(required_inputs(plan.get("inputs")))
        given = set(plan.get("selected") or [])
        for item in plan.get("config_errors") or []:
            claim(str(item.get("profile")), root)
        for directory in sorted(path for path in root.iterdir() if path.is_dir() and not KEPT_ASIDE.search(path.name)):
            kind = _final(directory)
            if kind is None:
                continue
            if directory.name not in given:
                problems.append(f"{root}: {directory.name} is not one of fragment {fragment}'s profiles")
            elif kind not in (mode, "build-failed", "error"):
                problems.append(f"{root}: {directory.name} is a {kind} result")
            else:
                claim(directory.name, root)
    if len(set(inputs.values())) > 1:
        problems.append("the fragments ran with different campaign inputs (harness, code, or the TensorRT, torch or "
                        "transformers version)")
    for name in sorted(expected - set(owner)):
        problems.append(f"{name}: no {mode} result")
    return problems
