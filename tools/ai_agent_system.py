#!/usr/bin/env python3
"""Shared GitHub queue helpers for the AI staging agent system."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_REMOTE = "github"
DEFAULT_TARGET = "ai-staging"
DEFAULT_PROMOTION_TARGET = "main"
DEFAULT_SOURCE_PREFIXES = ("ai-task-",)
ACTIVE_CHECK_STATUSES = {"created", "waiting_for_resource", "preparing", "pending", "running"}
AI_LABEL = "AI"
AI_GENERATED_LABEL = "ai-generated"

LABELS: dict[str, tuple[str, str]] = {
    AI_LABEL: ("#5319E7", "Human-facing tag for issues or pull requests produced by an AI agent."),
    "ai:task": ("#1F75CB", "Work item generated for AI implementation."),
    "ai:ready": ("#0E8A16", "Task is ready for an implementation agent."),
    "ai:claimed": ("#FBCA04", "Task has been claimed by an implementation agent."),
    "ai:implementing": ("#FBCA04", "Implementation is in progress."),
    AI_GENERATED_LABEL: ("#5319E7", "Issue or pull request was produced by an AI agent."),
    "ai:staging-pr": ("#0052CC", "AI-generated pull request targeting ai-staging."),
    "ai:sanity-pending": ("#BFDADC", "PR is waiting for sanity CI."),
    "ai:sanity-failed": ("#D93F0B", "PR failed minimal CI and needs rework."),
    "ai:sanity-green": ("#0E8A16", "PR sanity CI is green."),
    "ai:autopilot": ("#006B75", "PR is eligible for ai-staging autopilot."),
    "ai:staged": ("#0E8A16", "AI change has landed in ai-staging."),
    "ai:staging-failed": ("#D93F0B", "ai-staging full CI failed."),
    "ai:needs-rework": ("#D93F0B", "Task or PR needs another implementation pass."),
    "ai:dropped": ("#B60205", "AI task or PR was dropped as low-value or invalid."),
    "ai:needs-human": ("#D93F0B", "Human decision is required."),
    "ai:promotion": ("#5319E7", "ai-staging to main promotion PR."),
}

TASK_REQUIRED_HEADINGS = (
    "Scope",
    "Change",
    "Acceptance Criteria",
    "Verification",
    "Non-goals",
)


@dataclass(frozen=True)
class Config:
    project: str
    remote: str
    target: str
    promotion_target: str
    source_prefixes: tuple[str, ...]
    dry_run: bool


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+ " + shlex.join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=check, capture_output=capture, input=input_text, text=True)


def git(args: list[str], *, check: bool = True) -> str:
    result = run(["git", *args], check=check)
    return result.stdout.strip()


def infer_project_path(remote: str) -> str:
    remote_url = git(["remote", "get-url", remote])
    if "://" in remote_url:
        path = urllib.parse.urlparse(remote_url).path.lstrip("/")
    elif "@" in remote_url and ":" in remote_url:
        path = remote_url.split(":", 1)[1]
    else:
        path = remote_url
    if path.endswith(".git"):
        path = path[:-4]
    if not path or "/" not in path:
        raise SystemExit(f"Could not infer project path from remote URL: {remote_url!r}")
    return path


def encoded_project(project: str) -> str:
    """Compatibility wrapper for older tests; GitHub repo slugs stay raw."""

    return project.strip()


def infer_github_server_url(remote: str) -> str:
    remote_url = git(["remote", "get-url", remote])
    if "://" in remote_url:
        parsed = urllib.parse.urlparse(remote_url)
        if parsed.hostname:
            return f"https://{parsed.hostname}"
    if "@" in remote_url and ":" in remote_url:
        host = remote_url.split("@", 1)[1].split(":", 1)[0]
        if host:
            return f"https://{host}"
    raise SystemExit(f"Could not infer GitHub server URL from remote URL: {remote_url!r}")


def token_header() -> tuple[str, str] | None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return ("Authorization", f"Bearer {token}") if token else None


def api_base_url(cfg: Config) -> str:
    if os.environ.get("GH_API_URL"):
        return os.environ["GH_API_URL"].rstrip("/")
    server = infer_github_server_url(cfg.remote).rstrip("/")
    if server == "https://github.com":
        return "https://api.github.com"
    return server + "/api/v3"


def repo_path(cfg: Config, suffix: str = "") -> str:
    suffix = suffix.lstrip("/")
    return f"/repos/{cfg.project}/{suffix}" if suffix else f"/repos/{cfg.project}"


def api(
    cfg: Config,
    path: str,
    *,
    method: str | None = None,
    fields: dict[str, Any] | None = None,
) -> Any:
    write = (method or "").upper() in {"POST", "PUT", "PATCH", "DELETE"} or bool(fields)
    if cfg.dry_run and write:
        print(f"+ DRY RUN GitHub API {method or 'POST'} {path} {fields or {}}", file=sys.stderr)
        return None

    header = token_header()
    if header:
        data = json.dumps(fields).encode() if fields is not None else None
        request_method = method or ("POST" if data else "GET")
        url = api_base_url(cfg) + path
        print(f"+ {request_method} {url}", file=sys.stderr)
        request = urllib.request.Request(url, data=data, method=request_method)
        request.add_header(*header)
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request) as response:
                payload = response.read().decode()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise SystemExit(f"GitHub API failed: HTTP {exc.code} {exc.reason}: {body}") from exc
    else:
        cmd = ["gh", "api"]
        if method:
            cmd.extend(["-X", method])
        cmd.append(path)
        input_text = None
        if fields is not None:
            cmd.extend(["--input", "-"])
            input_text = json.dumps(fields)
        result = run(cmd, input_text=input_text)
        payload = result.stdout

    if not payload.strip():
        return None
    return json.loads(payload)


def paginated(cfg: Config, path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    sep = "&" if "?" in path else "?"
    page = 1
    while True:
        batch = api(cfg, f"{path}{sep}per_page=100&page={page}")
        if not isinstance(batch, list):
            raise SystemExit(f"Unexpected paginated response for {path}: {batch!r}")
        if not batch:
            break
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items


def csv_labels(labels: list[str]) -> str:
    return ",".join(labels)


def task_issue_labels(extra_labels: list[str]) -> list[str]:
    labels = [AI_LABEL, AI_GENERATED_LABEL, "ai:task", "ai:ready"]
    for label in extra_labels:
        if label not in labels:
            labels.append(label)
    return labels


def has_any_prefix(value: str, prefixes: tuple[str, ...]) -> bool:
    return value.startswith(prefixes)


def label_names(item: dict[str, Any]) -> list[str]:
    labels = item.get("labels") or []
    names: list[str] = []
    for label in labels:
        if isinstance(label, str):
            names.append(label)
        elif isinstance(label, dict) and label.get("name") is not None:
            names.append(str(label["name"]))
    return names


def item_number(item: dict[str, Any]) -> int:
    value = item.get("number") or item.get("id")
    if value is None:
        raise SystemExit(f"GitHub item has no number: {item!r}")
    return int(value)


def item_url(item: dict[str, Any]) -> str:
    return str(item.get("html_url") or "")


def issue_body(issue: dict[str, Any]) -> str:
    return str(issue.get("body") or issue.get("description") or "")


def repo_owner(cfg: Config) -> str:
    return cfg.project.split("/", 1)[0]


def pr_source_branch(pr: dict[str, Any]) -> str:
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    return str(head.get("ref") or "")


def pr_target_branch(pr: dict[str, Any]) -> str:
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    return str(base.get("ref") or "")


def pr_head_sha(pr: dict[str, Any]) -> str:
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    return str(head.get("sha") or pr.get("sha") or "")


def pr_merge_status(pr: dict[str, Any]) -> str:
    if pr.get("mergeable") is True:
        return "mergeable"
    if pr.get("mergeable") is False:
        return "conflict"
    return str(pr.get("mergeable_state") or "unknown")


def pr_ci_status(cfg: Config, pr: dict[str, Any]) -> str:
    sha = pr_head_sha(pr)
    if not sha:
        return "unknown"
    try:
        data = api(cfg, repo_path(cfg, f"commits/{sha}/check-runs"))
    except SystemExit as exc:
        print(f"warning: could not inspect checks for PR #{item_number(pr)}: {exc}", file=sys.stderr)
        return "unknown"
    check_runs = data.get("check_runs") if isinstance(data, dict) else None
    if not check_runs:
        return "none"

    statuses = {str(run_item.get("status") or "") for run_item in check_runs}
    conclusions = {str(run_item.get("conclusion") or "") for run_item in check_runs}
    if statuses & {"queued", "in_progress", "requested", "pending", "waiting"}:
        return "running"
    if conclusions & {"failure", "timed_out", "action_required"}:
        return "failed"
    if conclusions & {"cancelled", "canceled"}:
        return "canceled"
    if conclusions <= {"success", "skipped", "neutral", ""}:
        return "success"
    return "unknown"


def task_body(
    *,
    scope: str,
    change: str,
    acceptance: list[str],
    verification: list[str],
    non_goals: list[str],
    risk: str,
) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    return "\n".join(
        [
            "## Scope",
            scope.strip(),
            "",
            "## Change",
            change.strip(),
            "",
            "## Acceptance Criteria",
            bullets(acceptance),
            "",
            "## Verification",
            bullets(verification),
            "",
            "## Non-goals",
            bullets(non_goals),
            "",
            "## Risk",
            risk.strip(),
            "",
        ]
    )


def validate_task_description(description: str) -> list[str]:
    missing = []
    lowered = description.lower()
    for heading in TASK_REQUIRED_HEADINGS:
        if f"## {heading}".lower() not in lowered:
            missing.append(heading)
    return missing


def list_task_issues(cfg: Config, labels: list[str]) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"state": "open", "labels": csv_labels(labels)})
    return paginated(cfg, repo_path(cfg, f"issues?{query}"))


def list_ai_prs(cfg: Config) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "state": "open",
            "base": cfg.target,
            "sort": "created",
            "direction": "asc",
        }
    )
    prs = paginated(cfg, repo_path(cfg, f"pulls?{query}"))
    return [pr for pr in prs if has_any_prefix(pr_source_branch(pr), cfg.source_prefixes)]


def schedule_variables(schedule: dict[str, Any]) -> dict[str, str]:
    variables = schedule.get("variables") or []
    return {
        str(item.get("key")): str(item.get("value"))
        for item in variables
        if item.get("key") is not None
    }


def is_ai_promotion_schedule(schedule: dict[str, Any]) -> bool:
    variables = schedule_variables(schedule)
    description = str(schedule.get("description") or "").lower()
    return variables.get("AI_STAGING_PROMOTE") == "1" or (
        "ai" in description and "staging" in description and "promotion" in description
    )


def pr_details(cfg: Config, number: int) -> dict[str, Any]:
    data = api(cfg, repo_path(cfg, f"pulls/{number}"))
    if not isinstance(data, dict):
        raise SystemExit(f"Unexpected PR response for #{number}: {data!r}")
    return data


def issue_details(cfg: Config, number: int) -> dict[str, Any]:
    data = api(cfg, repo_path(cfg, f"issues/{number}"))
    if not isinstance(data, dict):
        raise SystemExit(f"Unexpected issue response for #{number}: {data!r}")
    return data


def approvals_left(cfg: Config, number: int) -> int | None:
    reviews = paginated(cfg, repo_path(cfg, f"pulls/{number}/reviews"))
    if not reviews:
        return None
    latest_by_user: dict[str, str] = {}
    for review in reviews:
        user = (review.get("user") or {}).get("login")
        if user:
            latest_by_user[str(user)] = str(review.get("state") or "")
    if any(state == "CHANGES_REQUESTED" for state in latest_by_user.values()):
        return 1
    return 0 if any(state == "APPROVED" for state in latest_by_user.values()) else None


def update_labels(
    cfg: Config,
    resource: str,
    number: int,
    *,
    add: set[str],
    remove: set[str],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    is_issue = resource == "issues"
    item = issue_details(cfg, number) if is_issue else pr_details(cfg, number)
    labels = set(label_names(item))
    labels.difference_update(remove)
    labels.update(add)
    updated = api(cfg, repo_path(cfg, f"issues/{number}/labels"), method="PUT", fields={"labels": sorted(labels)})
    if extra_fields:
        endpoint = f"issues/{number}" if is_issue else f"pulls/{number}"
        api(cfg, repo_path(cfg, endpoint), method="PATCH", fields=extra_fields)
    return updated if isinstance(updated, dict) else None


def create_note(cfg: Config, resource: str, number: int, body: str) -> None:
    del resource
    api(cfg, repo_path(cfg, f"issues/{number}/comments"), method="POST", fields={"body": body})


def related_prs_for_issue(cfg: Config, issue_number: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"q": f"repo:{cfg.project} is:pr #{issue_number}"})
    data = api(cfg, f"/search/issues?{query}")
    if not isinstance(data, dict):
        raise SystemExit(f"Unexpected related PR search response for #{issue_number}: {data!r}")
    return [item for item in data.get("items", []) if isinstance(item, dict)]


def closing_issue_numbers_for_pr(cfg: Config, pr_number: int) -> list[int]:
    pr = pr_details(cfg, pr_number)
    body = str(pr.get("body") or "")
    matches = re.findall(r"(?i)(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)", body)
    return [int(item) for item in dict.fromkeys(matches)]


def cmd_ensure_labels(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    existing = {label["name"] for label in paginated(cfg, repo_path(cfg, "labels"))}
    for name, (color, description) in LABELS.items():
        if name in existing:
            print(f"exists: {name}")
            continue
        api(
            cfg,
            repo_path(cfg, "labels"),
            method="POST",
            fields={"name": name, "color": color.lstrip("#"), "description": description},
        )
        print(f"created: {name}")
    return 0


def cmd_create_task(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    body = task_body(
        scope=args.scope,
        change=args.change,
        acceptance=args.acceptance,
        verification=args.verification,
        non_goals=args.non_goal,
        risk=args.risk,
    )
    labels = task_issue_labels(args.label)
    fields = {"title": args.title, "body": body, "labels": labels}
    if cfg.dry_run:
        print(json.dumps(fields, indent=2))
        return 0
    issue = api(cfg, repo_path(cfg, "issues"), method="POST", fields=fields)
    if not isinstance(issue, dict):
        raise SystemExit(f"Unexpected issue create response: {issue!r}")
    print(f"Created issue #{issue['number']}: {issue['html_url']}")
    return 0


def cmd_validate_task(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    if args.issue:
        issue = api(cfg, repo_path(cfg, f"issues/{args.issue}"))
        if not isinstance(issue, dict):
            raise SystemExit(f"Unexpected issue response: {issue!r}")
        description = issue_body(issue)
    elif args.file:
        with open(args.file, encoding="utf-8") as handle:
            description = handle.read()
    else:
        description = sys.stdin.read()

    missing = validate_task_description(description)
    if missing:
        print("Missing required task headings: " + ", ".join(missing))
        return 1
    print("Task description is valid.")
    return 0


def cmd_next_task(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    by_number: dict[int, dict[str, Any]] = {}
    for issue in [
        *list_task_issues(cfg, ["ai:task", "ai:ready"]),
        *list_task_issues(cfg, ["ai:task", "ai:needs-rework"]),
    ]:
        by_number[item_number(issue)] = issue
    tasks = list(by_number.values())
    candidates = [
        issue
        for issue in tasks
        if "ai:claimed" not in label_names(issue)
        and "ai:dropped" not in label_names(issue)
        and "ai:needs-human" not in label_names(issue)
    ]
    if not candidates:
        print("No ready unclaimed AI tasks.")
        return 0
    issue = sorted(candidates, key=lambda item: item.get("created_at", ""))[0]
    if args.json:
        print(json.dumps(issue, indent=2, sort_keys=True))
    else:
        print(f"#{item_number(issue)} {issue['title']}")
        print(item_url(issue))
    return 0


def cmd_claim_task(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    issue = api(cfg, repo_path(cfg, f"issues/{args.issue}"))
    if not isinstance(issue, dict):
        raise SystemExit(f"Unexpected issue response: {issue!r}")
    labels = set(label_names(issue))
    labels.discard("ai:ready")
    labels.discard("ai:needs-rework")
    labels.update({"ai:claimed", "ai:implementing"})
    updated = api(cfg, repo_path(cfg, f"issues/{args.issue}/labels"), method="PUT", fields={"labels": sorted(labels)})
    if cfg.dry_run:
        print(f"would claim issue #{args.issue}")
    elif isinstance(updated, dict):
        print(f"claimed issue #{args.issue}: {item_url(issue)}")
    return 0


def cmd_related_prs(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    details = [pr_details(cfg, item_number(item)) for item in related_prs_for_issue(cfg, args.issue)]
    if args.json:
        print(json.dumps(details, indent=2, sort_keys=True))
        return 0

    if not details:
        print(f"No related PRs for issue #{args.issue}.")
        return 0
    for pr in details:
        status = pr_ci_status(cfg, pr)
        print(
            f"PR #{item_number(pr)} "
            f"{pr.get('state', '-'):<8} "
            f"target={pr_target_branch(pr) or '-'} "
            f"source={pr_source_branch(pr) or '-'} "
            f"ci={status} "
            f"{item_url(pr)}"
        )
    return 0


def cmd_mark_rework(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    if not args.pr and not args.issue:
        raise SystemExit("mark-rework requires --pr, --issue, or both")

    issue_numbers = list(dict.fromkeys(args.issue or []))
    pr_url = None
    if args.pr:
        pr = pr_details(cfg, args.pr)
        pr_url = item_url(pr) or f"PR #{args.pr}"
        check_status = pr_ci_status(cfg, pr).lower()
        if args.skip_if_active_checks and check_status in ACTIVE_CHECK_STATUSES:
            print(f"skipped PR #{args.pr}: active head checks are {check_status}")
            return 0
        update_labels(
            cfg,
            "pulls",
            args.pr,
            add={"ai:needs-rework"},
            remove={"ai:sanity-pending", "ai:sanity-failed", "ai:sanity-green", "ai:autopilot"},
        )
        if not issue_numbers:
            issue_numbers = closing_issue_numbers_for_pr(cfg, args.pr)
        note = "Marked `ai:needs-rework` for another implementation pass."
        if args.reason:
            note += f"\n\nReason: {args.reason}"
        create_note(cfg, "pulls", args.pr, note)
        print(f"marked PR #{args.pr} as ai:needs-rework")

    if not issue_numbers:
        print("warning: no linked issue was found; pass --issue explicitly", file=sys.stderr)
        return 0

    for issue_number in issue_numbers:
        issue = issue_details(cfg, issue_number)
        extra_fields = {"state": "open"} if issue.get("state") != "open" else None
        update_labels(
            cfg,
            "issues",
            issue_number,
            add={AI_LABEL, AI_GENERATED_LABEL, "ai:task", "ai:needs-rework"},
            remove={"ai:ready", "ai:claimed", "ai:implementing", "ai:dropped", "ai:needs-human"},
            extra_fields=extra_fields,
        )
        note = "Marked `ai:needs-rework` for another implementation pass."
        if pr_url:
            note += f"\n\nRelated PR: {pr_url}"
        if args.reason:
            note += f"\n\nReason: {args.reason}"
        create_note(cfg, "issues", issue_number, note)
        print(f"marked issue #{issue_number} as ai:needs-rework")
    return 0


def classify_pr(cfg: Config, pr: dict[str, Any]) -> str:
    labels = set(label_names(pr))
    if "ai:staging-pr" not in labels:
        return "missing-ai-staging-label"
    check_status = pr_ci_status(cfg, pr)
    merge_status = pr_merge_status(pr)
    left = approvals_left(cfg, item_number(pr))
    if check_status == "success" and merge_status == "mergeable" and left in (None, 0):
        return "ready-for-autopilot"
    if check_status in {"failed", "canceled"}:
        return "needs-rework"
    if merge_status in {"conflict", "need_rebase"}:
        return "needs-rebase-or-conflict-resolution"
    if check_status in {"pending", "running", "created", "preparing"}:
        return "waiting-for-ci"
    return f"blocked:{merge_status}/{check_status}"


def cmd_dashboard(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    tasks = list_task_issues(cfg, ["ai:task"])
    prs = list_ai_prs(cfg)
    promotion_query = urllib.parse.urlencode({"state": "open", "base": cfg.promotion_target})
    promotion_prs = [
        pr
        for pr in paginated(cfg, repo_path(cfg, f"pulls?{promotion_query}"))
        if pr_source_branch(pr) == cfg.target
    ]

    print("AI task issues")
    print("--------------")
    if not tasks:
        print("none")
    for issue in tasks[: args.limit]:
        labels = ",".join(label_names(issue))
        print(f"#{item_number(issue):<4} {issue['state']:<7} {labels:<45} {issue['title'][:80]}")

    print("\nAI PRs targeting " + cfg.target)
    print("----------------" + "-" * len(cfg.target))
    if not prs:
        print("none")
    for item in prs[: args.limit]:
        pr = pr_details(cfg, item_number(item))
        check_status = pr_ci_status(cfg, pr)
        classification = classify_pr(cfg, pr)
        print(
            f"#{item_number(pr):<4} "
            f"{classification:<36} "
            f"{pr_merge_status(pr):<14} "
            f"{check_status:<9} "
            f"{pr_source_branch(pr)}"
        )

    print("\nPromotion PRs")
    print("-------------")
    if not promotion_prs:
        print("none")
    for item in promotion_prs[: args.limit]:
        pr = pr_details(cfg, item_number(item))
        print(f"#{item_number(pr):<4} {pr_ci_status(cfg, pr):<9} {pr['title']}")
    return 0


def remote_branch_exists(cfg: Config, branch: str) -> bool:
    result = run(["git", "ls-remote", "--heads", cfg.remote, branch], check=False)
    return result.returncode == 0 and bool(result.stdout.strip())


def cmd_preflight(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    failures: list[str] = []
    warnings: list[str] = []

    print("AI staging operator preflight")
    print("-----------------------------")

    status = git(["status", "--porcelain"])
    if status:
        warnings.append("local worktree is dirty; implementation cycles should start from a clean checkout")
    else:
        print("ok: local worktree is clean")

    for branch in dict.fromkeys(("main", cfg.target, cfg.promotion_target)):
        if remote_branch_exists(cfg, branch):
            print(f"ok: {cfg.remote}/{branch} exists")
        else:
            failures.append(f"missing remote branch: {cfg.remote}/{branch}")

    existing_labels = {label["name"] for label in paginated(cfg, repo_path(cfg, "labels"))}
    missing_labels = sorted(set(LABELS) - existing_labels)
    if missing_labels:
        failures.append("missing labels: " + ", ".join(missing_labels))
    else:
        print("ok: standard AI labels exist")

    try:
        permissions = api(cfg, repo_path(cfg, "actions/permissions"))
    except SystemExit as exc:
        warnings.append(f"could not inspect GitHub Actions permissions: {exc}")
    else:
        if isinstance(permissions, dict):
            print(f"ok: GitHub Actions enabled={permissions.get('enabled', 'unknown')}")

    try:
        runner_data = api(cfg, repo_path(cfg, "actions/runners?per_page=100"))
    except SystemExit as exc:
        warnings.append(f"could not inspect GitHub Actions runners: {exc}")
    else:
        runners = runner_data.get("runners") if isinstance(runner_data, dict) else []
        cpu_runners = [
            runner
            for runner in runners
            if any((label.get("name") == "cpu") for label in runner.get("labels", []))
        ]
        online_cpu = [runner for runner in cpu_runners if runner.get("status") == "online"]
        if online_cpu:
            names = ", ".join(str(runner.get("name") or runner.get("id")) for runner in online_cpu)
            print(f"ok: online cpu runner(s): {names}")
        elif runners:
            warnings.append("no online GitHub Actions runner with label 'cpu' was visible to the token")

    promotion_query = urllib.parse.urlencode({"state": "open", "base": cfg.promotion_target})
    promotion_prs = [
        pr
        for pr in paginated(cfg, repo_path(cfg, f"pulls?{promotion_query}"))
        if pr_source_branch(pr) == cfg.target
    ]
    if len(promotion_prs) > 1:
        warnings.append(f"multiple open promotion PRs exist: {', '.join('#' + str(item_number(pr)) for pr in promotion_prs)}")
    elif promotion_prs:
        pr = pr_details(cfg, item_number(promotion_prs[0]))
        print(f"ok: open promotion PR #{item_number(pr)} checks={pr_ci_status(cfg, pr)}")
    else:
        print("ok: no open promotion PR; the staging loop will create one when ai-staging has a tree diff")

    if warnings:
        print("\nWarnings")
        print("--------")
        for warning in warnings:
            print(f"- {warning}")
    if failures:
        print("\nFailures")
        print("--------")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nPreflight passed.")
    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", help="GitHub owner/repo path")
    parser.add_argument("--remote", default=os.environ.get("AI_STAGING_REMOTE", DEFAULT_REMOTE))
    parser.add_argument("--target", default=os.environ.get("AI_STAGING_BRANCH", DEFAULT_TARGET))
    parser.add_argument("--promotion-target", default=os.environ.get("AI_STAGING_PROMOTION_TARGET", DEFAULT_PROMOTION_TARGET))
    parser.add_argument("--source-prefix", action="append", help="AI source branch prefix; repeatable")
    parser.add_argument("--dry-run", action="store_true")


def config_from_args(args: argparse.Namespace) -> Config:
    project = args.project or os.environ.get("GITHUB_REPOSITORY") or infer_project_path(args.remote)
    return Config(
        project=encoded_project(project),
        remote=args.remote,
        target=args.target,
        promotion_target=args.promotion_target,
        source_prefixes=tuple(args.source_prefix or DEFAULT_SOURCE_PREFIXES),
        dry_run=args.dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ensure = subparsers.add_parser("ensure-labels", help="create standard AI system labels if missing")
    ensure.set_defaults(func=cmd_ensure_labels)

    create = subparsers.add_parser("create-task", help="create an atomic AI task issue")
    create.add_argument("--title", required=True)
    create.add_argument("--scope", required=True)
    create.add_argument("--change", required=True)
    create.add_argument("--acceptance", action="append", required=True)
    create.add_argument("--verification", action="append", required=True)
    create.add_argument("--non-goal", action="append", required=True)
    create.add_argument("--risk", default="Low. Narrow, locally verifiable, and rollback is obvious.")
    create.add_argument("--label", action="append", default=[])
    create.set_defaults(func=cmd_create_task)

    validate = subparsers.add_parser("validate-task", help="validate an AI task issue/body has required sections")
    validate.add_argument("--issue", type=int)
    validate.add_argument("--file")
    validate.set_defaults(func=cmd_validate_task)

    next_task = subparsers.add_parser("next-task", help="print the oldest ready, unclaimed AI task")
    next_task.add_argument("--json", action="store_true")
    next_task.set_defaults(func=cmd_next_task)

    claim = subparsers.add_parser("claim-task", help="mark a task issue as claimed/in progress")
    claim.add_argument("issue", type=int)
    claim.set_defaults(func=cmd_claim_task)

    related = subparsers.add_parser("related-prs", help="list pull requests related to an issue")
    related.add_argument("issue", type=int)
    related.add_argument("--json", action="store_true")
    related.set_defaults(func=cmd_related_prs)

    rework = subparsers.add_parser("mark-rework", help="mark an issue/PR for another implementation pass")
    rework.add_argument("--pr", type=int, help="pull request number to mark")
    rework.add_argument("--issue", type=int, action="append", default=[], help="issue number to mark; repeatable")
    rework.add_argument("--reason", default="")
    rework.add_argument("--skip-if-active-checks", action="store_true")
    rework.set_defaults(func=cmd_mark_rework)

    dashboard = subparsers.add_parser("dashboard", help="summarize AI tasks, ai-staging PRs, and promotion PRs")
    dashboard.add_argument("--limit", type=int, default=50)
    dashboard.set_defaults(func=cmd_dashboard)

    preflight = subparsers.add_parser("preflight", help="validate operator prerequisites for the AI staging loop")
    preflight.set_defaults(func=cmd_preflight)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
