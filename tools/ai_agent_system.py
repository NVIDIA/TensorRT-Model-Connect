#!/usr/bin/env python3
"""Shared GitLab queue helpers for the AI staging agent system."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_REMOTE = "origin"
DEFAULT_TARGET = "ai-staging"
DEFAULT_PROMOTION_TARGET = "master"
DEFAULT_SOURCE_PREFIXES = ("ai-task-",)
ACTIVE_PIPELINE_STATUSES = {"created", "waiting_for_resource", "preparing", "pending", "running"}
AI_LABEL = "AI"
AI_GENERATED_LABEL = "ai-generated"

LABELS: dict[str, tuple[str, str]] = {
    AI_LABEL: ("#5319E7", "Human-facing tag for issues or merge requests produced by an AI agent."),
    "ai:task": ("#1F75CB", "Work item generated for AI implementation."),
    "ai:ready": ("#0E8A16", "Task is ready for an implementation agent."),
    "ai:claimed": ("#FBCA04", "Task has been claimed by an implementation agent."),
    "ai:implementing": ("#FBCA04", "Implementation is in progress."),
    AI_GENERATED_LABEL: ("#5319E7", "Issue or merge request was produced by an AI agent."),
    "ai:staging-mr": ("#0052CC", "AI-generated merge request targeting ai-staging."),
    "ai:sanity-pending": ("#BFDADC", "MR is waiting for sanity CI."),
    "ai:sanity-failed": ("#D93F0B", "MR failed minimal CI and needs rework."),
    "ai:sanity-green": ("#0E8A16", "MR sanity CI is green."),
    "ai:autopilot": ("#006B75", "MR is eligible for ai-staging autopilot."),
    "ai:staged": ("#0E8A16", "AI change has landed in ai-staging."),
    "ai:staging-failed": ("#D93F0B", "ai-staging full CI failed."),
    "ai:needs-rework": ("#D93F0B", "Task or MR needs another implementation pass."),
    "ai:dropped": ("#B60205", "AI task or MR was dropped as low-value or invalid."),
    "ai:needs-human": ("#D93F0B", "Human decision is required."),
    "ai:promotion": ("#5319E7", "ai-staging to master promotion MR."),
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


def run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    print("+ " + shlex.join(cmd), file=sys.stderr)
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


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
    return project if project.isdecimal() else urllib.parse.quote(project, safe="")


def infer_gitlab_server_url(remote: str) -> str:
    remote_url = git(["remote", "get-url", remote])
    if "://" in remote_url:
        parsed = urllib.parse.urlparse(remote_url)
        if parsed.hostname:
            return f"https://{parsed.hostname}"
    if "@" in remote_url and ":" in remote_url:
        host = remote_url.split("@", 1)[1].split(":", 1)[0]
        if host:
            return f"https://{host}"
    raise SystemExit(f"Could not infer GitLab server URL from remote URL: {remote_url!r}")


def token_header() -> tuple[str, str] | None:
    token = os.environ.get("AI_STAGING_BOT_TOKEN") or os.environ.get("GITLAB_TOKEN")
    return ("PRIVATE-TOKEN", token) if token else None


def api_base_url(cfg: Config) -> str:
    if os.environ.get("CI_API_V4_URL"):
        return os.environ["CI_API_V4_URL"].rstrip("/")
    return infer_gitlab_server_url(cfg.remote).rstrip("/") + "/api/v4"


def api(
    cfg: Config,
    path: str,
    *,
    method: str | None = None,
    fields: dict[str, str] | None = None,
) -> Any:
    write = (method or "").upper() in {"POST", "PUT", "PATCH", "DELETE"} or bool(fields)
    if cfg.dry_run and write:
        print(f"+ DRY RUN GitLab API {method or 'POST'} {path} {fields or {}}", file=sys.stderr)
        return None

    header = token_header()
    if header:
        data = urllib.parse.urlencode(fields or {}).encode() if fields else None
        request_method = method or ("POST" if data else "GET")
        url = api_base_url(cfg) + path
        print(f"+ {request_method} {url}", file=sys.stderr)
        request = urllib.request.Request(url, data=data, method=request_method)
        request.add_header(*header)
        if data:
            request.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(request) as response:
                payload = response.read().decode()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise SystemExit(f"GitLab API failed: HTTP {exc.code} {exc.reason}: {body}") from exc
    else:
        cmd = ["glab", "api"]
        if method:
            cmd.extend(["-X", method])
        cmd.append(path)
        for key, value in (fields or {}).items():
            cmd.extend(["-f", f"{key}={value}"])
        result = run(cmd)
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
    query = urllib.parse.urlencode({"state": "opened", "labels": csv_labels(labels)})
    return paginated(cfg, f"/projects/{cfg.project}/issues?{query}")


def list_ai_mrs(cfg: Config) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "state": "opened",
            "target_branch": cfg.target,
            "order_by": "created_at",
            "sort": "asc",
        }
    )
    mrs = paginated(cfg, f"/projects/{cfg.project}/merge_requests?{query}")
    return [mr for mr in mrs if has_any_prefix(str(mr.get("source_branch") or ""), cfg.source_prefixes)]


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


def mr_details(cfg: Config, iid: int) -> dict[str, Any]:
    data = api(cfg, f"/projects/{cfg.project}/merge_requests/{iid}?include_rebase_in_progress=true")
    if not isinstance(data, dict):
        raise SystemExit(f"Unexpected MR response for !{iid}: {data!r}")
    return data


def issue_details(cfg: Config, iid: int) -> dict[str, Any]:
    data = api(cfg, f"/projects/{cfg.project}/issues/{iid}")
    if not isinstance(data, dict):
        raise SystemExit(f"Unexpected issue response for #{iid}: {data!r}")
    return data


def approvals_left(cfg: Config, iid: int) -> int | None:
    data = api(cfg, f"/projects/{cfg.project}/merge_requests/{iid}/approvals")
    if not isinstance(data, dict):
        return None
    value = data.get("approvals_left")
    return int(value) if value is not None else None


def update_labels(
    cfg: Config,
    resource: str,
    iid: int,
    *,
    add: set[str],
    remove: set[str],
    extra_fields: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    item = issue_details(cfg, iid) if resource == "issues" else mr_details(cfg, iid)
    labels = set(item.get("labels") or [])
    labels.difference_update(remove)
    labels.update(add)
    fields = {"labels": csv_labels(sorted(labels)), **(extra_fields or {})}
    updated = api(cfg, f"/projects/{cfg.project}/{resource}/{iid}", method="PUT", fields=fields)
    return updated if isinstance(updated, dict) else None


def create_note(cfg: Config, resource: str, iid: int, body: str) -> None:
    api(cfg, f"/projects/{cfg.project}/{resource}/{iid}/notes", method="POST", fields={"body": body})


def related_mrs_for_issue(cfg: Config, issue_iid: int) -> list[dict[str, Any]]:
    data = api(cfg, f"/projects/{cfg.project}/issues/{issue_iid}/related_merge_requests")
    if not isinstance(data, list):
        raise SystemExit(f"Unexpected related MR response for #{issue_iid}: {data!r}")
    return data


def closing_issue_iids_for_mr(cfg: Config, mr_iid: int) -> list[int]:
    try:
        data = api(cfg, f"/projects/{cfg.project}/merge_requests/{mr_iid}/closes_issues")
    except SystemExit as exc:
        print(f"warning: could not inspect closing issues for !{mr_iid}: {exc}", file=sys.stderr)
        return []
    if not isinstance(data, list):
        return []
    return [int(issue["iid"]) for issue in data if issue.get("iid") is not None]


def cmd_ensure_labels(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    existing = {label["name"] for label in paginated(cfg, f"/projects/{cfg.project}/labels")}
    for name, (color, description) in LABELS.items():
        if name in existing:
            print(f"exists: {name}")
            continue
        api(
            cfg,
            f"/projects/{cfg.project}/labels",
            method="POST",
            fields={"name": name, "color": color, "description": description},
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
    fields = {"title": args.title, "description": body, "labels": csv_labels(labels)}
    if cfg.dry_run:
        print(json.dumps(fields, indent=2))
        return 0
    issue = api(cfg, f"/projects/{cfg.project}/issues", method="POST", fields=fields)
    if not isinstance(issue, dict):
        raise SystemExit(f"Unexpected issue create response: {issue!r}")
    print(f"Created issue #{issue['iid']}: {issue['web_url']}")
    return 0


def cmd_validate_task(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    if args.issue:
        issue = api(cfg, f"/projects/{cfg.project}/issues/{args.issue}")
        if not isinstance(issue, dict):
            raise SystemExit(f"Unexpected issue response: {issue!r}")
        description = str(issue.get("description") or "")
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
    by_iid: dict[int, dict[str, Any]] = {}
    for issue in [
        *list_task_issues(cfg, ["ai:task", "ai:ready"]),
        *list_task_issues(cfg, ["ai:task", "ai:needs-rework"]),
    ]:
        by_iid[int(issue["iid"])] = issue
    tasks = list(by_iid.values())
    candidates = [
        issue
        for issue in tasks
        if "ai:claimed" not in issue.get("labels", [])
        and "ai:dropped" not in issue.get("labels", [])
        and "ai:needs-human" not in issue.get("labels", [])
    ]
    if not candidates:
        print("No ready unclaimed AI tasks.")
        return 0
    issue = sorted(candidates, key=lambda item: item.get("created_at", ""))[0]
    if args.json:
        print(json.dumps(issue, indent=2, sort_keys=True))
    else:
        print(f"#{issue['iid']} {issue['title']}")
        print(issue["web_url"])
    return 0


def cmd_claim_task(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    issue = api(cfg, f"/projects/{cfg.project}/issues/{args.issue}")
    if not isinstance(issue, dict):
        raise SystemExit(f"Unexpected issue response: {issue!r}")
    labels = set(issue.get("labels") or [])
    labels.discard("ai:ready")
    labels.discard("ai:needs-rework")
    labels.update({"ai:claimed", "ai:implementing"})
    updated = api(
        cfg,
        f"/projects/{cfg.project}/issues/{args.issue}",
        method="PUT",
        fields={"labels": csv_labels(sorted(labels))},
    )
    if cfg.dry_run:
        print(f"would claim issue #{args.issue}")
    elif isinstance(updated, dict):
        print(f"claimed issue #{updated['iid']}: {updated['web_url']}")
    return 0


def cmd_related_mrs(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    details = [mr_details(cfg, int(item["iid"])) for item in related_mrs_for_issue(cfg, args.issue)]
    if args.json:
        print(json.dumps(details, indent=2, sort_keys=True))
        return 0

    if not details:
        print(f"No related MRs for issue #{args.issue}.")
        return 0
    for mr in details:
        pipeline = mr.get("head_pipeline") or {}
        print(
            f"!{mr['iid']} "
            f"{mr.get('state', '-'):<8} "
            f"target={mr.get('target_branch', '-')} "
            f"source={mr.get('source_branch', '-')} "
            f"pipeline={pipeline.get('status') or '-'} "
            f"{mr.get('web_url')}"
        )
    return 0


def cmd_mark_rework(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    if not args.mr and not args.issue:
        raise SystemExit("mark-rework requires --mr, --issue, or both")

    issue_iids = list(dict.fromkeys(args.issue or []))
    mr_url = None
    if args.mr:
        mr = mr_details(cfg, args.mr)
        mr_url = str(mr.get("web_url") or f"!{args.mr}")
        pipeline_status = ((mr.get("head_pipeline") or {}).get("status") or "none").lower()
        if args.skip_if_active_pipeline and pipeline_status in ACTIVE_PIPELINE_STATUSES:
            print(f"skipped !{args.mr}: active head pipeline is {pipeline_status}")
            return 0
        update_labels(
            cfg,
            "merge_requests",
            args.mr,
            add={"ai:needs-rework"},
            remove={"ai:sanity-pending", "ai:sanity-failed", "ai:sanity-green", "ai:autopilot"},
        )
        if not issue_iids:
            issue_iids = closing_issue_iids_for_mr(cfg, args.mr)
        note = "Marked `ai:needs-rework` for another implementation pass."
        if args.reason:
            note += f"\n\nReason: {args.reason}"
        create_note(cfg, "merge_requests", args.mr, note)
        print(f"marked MR !{args.mr} as ai:needs-rework")

    if not issue_iids:
        print("warning: no linked issue was found; pass --issue explicitly", file=sys.stderr)
        return 0

    for issue_iid in issue_iids:
        issue = issue_details(cfg, issue_iid)
        extra_fields = {"state_event": "reopen"} if issue.get("state") != "opened" else None
        update_labels(
            cfg,
            "issues",
            issue_iid,
            add={AI_LABEL, AI_GENERATED_LABEL, "ai:task", "ai:needs-rework"},
            remove={"ai:ready", "ai:claimed", "ai:implementing", "ai:dropped", "ai:needs-human"},
            extra_fields=extra_fields,
        )
        note = "Marked `ai:needs-rework` for another implementation pass."
        if mr_url:
            note += f"\n\nRelated MR: {mr_url}"
        if args.reason:
            note += f"\n\nReason: {args.reason}"
        create_note(cfg, "issues", issue_iid, note)
        print(f"marked issue #{issue_iid} as ai:needs-rework")
    return 0


def classify_mr(cfg: Config, mr: dict[str, Any]) -> str:
    labels = set(mr.get("labels") or [])
    if "ai:staging-mr" not in labels:
        return "missing-ai-staging-label"
    pipeline = mr.get("head_pipeline") or {}
    pipeline_status = pipeline.get("status") or "none"
    merge_status = mr.get("detailed_merge_status") or "unknown"
    left = approvals_left(cfg, int(mr["iid"]))
    if pipeline_status == "success" and merge_status == "mergeable" and left in (None, 0):
        return "ready-for-autopilot"
    if pipeline_status in {"failed", "canceled"}:
        return "needs-rework"
    if merge_status in {"conflict", "need_rebase"}:
        return "needs-rebase-or-conflict-resolution"
    if pipeline_status in {"pending", "running", "created", "preparing"}:
        return "waiting-for-ci"
    return f"blocked:{merge_status}/{pipeline_status}"


def cmd_dashboard(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    tasks = list_task_issues(cfg, ["ai:task"])
    mrs = list_ai_mrs(cfg)
    promotion_query = urllib.parse.urlencode(
        {
            "state": "opened",
            "source_branch": cfg.target,
            "target_branch": cfg.promotion_target,
        }
    )
    promotion_mrs = paginated(cfg, f"/projects/{cfg.project}/merge_requests?{promotion_query}")

    print("AI task issues")
    print("--------------")
    if not tasks:
        print("none")
    for issue in tasks[: args.limit]:
        labels = ",".join(issue.get("labels", []))
        print(f"#{issue['iid']:<4} {issue['state']:<7} {labels:<45} {issue['title'][:80]}")

    print("\nAI MRs targeting " + cfg.target)
    print("----------------" + "-" * len(cfg.target))
    if not mrs:
        print("none")
    for item in mrs[: args.limit]:
        mr = mr_details(cfg, int(item["iid"]))
        pipeline = mr.get("head_pipeline") or {}
        classification = classify_mr(cfg, mr)
        print(
            f"!{mr['iid']:<4} "
            f"{classification:<36} "
            f"{str(mr.get('detailed_merge_status') or '-'):<14} "
            f"{str(pipeline.get('status') or '-'):<9} "
            f"{mr.get('source_branch')}"
        )

    print("\nPromotion MRs")
    print("-------------")
    if not promotion_mrs:
        print("none")
    for item in promotion_mrs[: args.limit]:
        mr = mr_details(cfg, int(item["iid"]))
        pipeline = mr.get("head_pipeline") or {}
        print(f"!{mr['iid']:<4} {str(pipeline.get('status') or '-'):<9} {mr['title']}")
    return 0


def remote_branch_exists(cfg: Config, branch: str) -> bool:
    result = run(["git", "ls-remote", "--heads", cfg.remote, branch], check=False)
    return result.returncode == 0 and bool(result.stdout.strip())


def collect_ai_promotion_schedules(cfg: Config) -> list[dict[str, Any]]:
    schedules = paginated(cfg, f"/projects/{cfg.project}/pipeline_schedules")
    details: list[dict[str, Any]] = []
    for schedule in schedules:
        schedule_id = schedule.get("id")
        if schedule_id is None:
            continue
        detail = api(cfg, f"/projects/{cfg.project}/pipeline_schedules/{schedule_id}")
        if not isinstance(detail, dict):
            continue
        if is_ai_promotion_schedule(detail):
            details.append(detail)
    return details


def project_variables(cfg: Config) -> list[dict[str, Any]] | None:
    try:
        variables = paginated(cfg, f"/projects/{cfg.project}/variables")
    except SystemExit as exc:
        print(f"warning: could not inspect project variables: {exc}", file=sys.stderr)
        return None
    return variables


def runner_details(cfg: Config, runner_id: int) -> dict[str, Any] | None:
    try:
        data = api(cfg, f"/runners/{runner_id}")
    except SystemExit as exc:
        print(f"warning: could not inspect runner {runner_id}: {exc}", file=sys.stderr)
        return None
    return data if isinstance(data, dict) else None


def latest_pipeline_failure_summary(cfg: Config, pipeline: dict[str, Any]) -> list[str]:
    pipeline_id = pipeline.get("id")
    if not pipeline_id or pipeline.get("status") != "failed":
        return []
    jobs = api(cfg, f"/projects/{cfg.project}/pipelines/{pipeline_id}/jobs?per_page=100")
    if not isinstance(jobs, list):
        return []
    failures = []
    for job in jobs:
        if job.get("status") != "failed":
            continue
        reason = job.get("failure_reason") or "unknown"
        name = job.get("name") or job.get("id")
        runner = (job.get("runner") or {}).get("description") or "no runner"
        failures.append(f"{name}: {reason} on {runner}")
    return failures


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

    for branch in dict.fromkeys(("master", cfg.target, cfg.promotion_target)):
        if remote_branch_exists(cfg, branch):
            print(f"ok: {cfg.remote}/{branch} exists")
        else:
            failures.append(f"missing remote branch: {cfg.remote}/{branch}")

    existing_labels = {label["name"] for label in paginated(cfg, f"/projects/{cfg.project}/labels")}
    missing_labels = sorted(set(LABELS) - existing_labels)
    if missing_labels:
        failures.append("missing labels: " + ", ".join(missing_labels))
    else:
        print("ok: standard AI labels exist")

    runners = paginated(cfg, f"/projects/{cfg.project}/runners")
    cpu_runners = []
    for runner in runners:
        detail = runner_details(cfg, int(runner["id"]))
        if not detail:
            continue
        if "cpu" in (detail.get("tag_list") or []):
            cpu_runners.append(detail)
    online_cpu = [runner for runner in cpu_runners if runner.get("status") == "online" and not runner.get("paused")]
    if online_cpu:
        descriptions = ", ".join(str(runner.get("description") or runner.get("id")) for runner in online_cpu)
        print(f"ok: online cpu runner(s): {descriptions}")
    else:
        failures.append("no online unpaused project runner with tag 'cpu'")

    schedules = collect_ai_promotion_schedules(cfg)
    active_schedules = [schedule for schedule in schedules if schedule.get("active")]
    if not active_schedules:
        failures.append("no active AI staging promotion schedule with AI_STAGING_PROMOTE=1")
    else:
        for schedule in active_schedules:
            variables = schedule_variables(schedule)
            ref = str(schedule.get("ref") or "")
            cron = schedule.get("cron")
            next_run = schedule.get("next_run_at")
            print(f"ok: promotion schedule {schedule.get('id')} ref={ref} cron={cron} next={next_run}")
            if not ref.endswith(f"/{cfg.promotion_target}") and ref != cfg.promotion_target:
                warnings.append(f"promotion schedule {schedule.get('id')} runs on {ref}, expected {cfg.promotion_target}")
            if variables.get("AI_STAGING_PROMOTE") != "1":
                failures.append(f"promotion schedule {schedule.get('id')} is missing AI_STAGING_PROMOTE=1")
            for failure in latest_pipeline_failure_summary(cfg, schedule.get("last_pipeline") or {}):
                warnings.append(f"last promotion schedule pipeline failed: {failure}")

    variables = project_variables(cfg)
    if variables is not None:
        variable_by_key = {str(item.get("key")): item for item in variables}
        token = variable_by_key.get("AI_STAGING_BOT_TOKEN")
        if not token:
            failures.append("missing project variable AI_STAGING_BOT_TOKEN")
        else:
            masked = bool(token.get("masked"))
            protected = bool(token.get("protected"))
            print(f"ok: AI_STAGING_BOT_TOKEN exists masked={masked} protected={protected}")
            if not masked:
                warnings.append("AI_STAGING_BOT_TOKEN is not masked")
            if not protected:
                warnings.append("AI_STAGING_BOT_TOKEN is not protected")

    promotion_query = urllib.parse.urlencode(
        {
            "state": "opened",
            "source_branch": cfg.target,
            "target_branch": cfg.promotion_target,
        }
    )
    promotion_mrs = paginated(cfg, f"/projects/{cfg.project}/merge_requests?{promotion_query}")
    if len(promotion_mrs) > 1:
        warnings.append(f"multiple open promotion MRs exist: {', '.join('!' + str(mr['iid']) for mr in promotion_mrs)}")
    elif promotion_mrs:
        mr = mr_details(cfg, int(promotion_mrs[0]["iid"]))
        pipeline = (mr.get("head_pipeline") or {}).get("status") or "none"
        print(f"ok: open promotion MR !{mr['iid']} pipeline={pipeline}")
    else:
        print("ok: no open promotion MR; schedule will create one when ai-staging has a tree diff")

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
    parser.add_argument("--project", help="GitLab project path or numeric id")
    parser.add_argument("--remote", default=os.environ.get("AI_STAGING_REMOTE", DEFAULT_REMOTE))
    parser.add_argument("--target", default=os.environ.get("AI_STAGING_BRANCH", DEFAULT_TARGET))
    parser.add_argument("--promotion-target", default=os.environ.get("AI_STAGING_PROMOTION_TARGET", DEFAULT_PROMOTION_TARGET))
    parser.add_argument("--source-prefix", action="append", help="AI source branch prefix; repeatable")
    parser.add_argument("--dry-run", action="store_true")


def config_from_args(args: argparse.Namespace) -> Config:
    project = args.project or os.environ.get("CI_PROJECT_PATH") or infer_project_path(args.remote)
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

    related = subparsers.add_parser("related-mrs", help="list merge requests related to an issue")
    related.add_argument("issue", type=int)
    related.add_argument("--json", action="store_true")
    related.set_defaults(func=cmd_related_mrs)

    rework = subparsers.add_parser("mark-rework", help="mark an issue/MR for another implementation pass")
    rework.add_argument("--mr", type=int, help="merge request IID to mark")
    rework.add_argument("--issue", type=int, action="append", default=[], help="issue IID to mark; repeatable")
    rework.add_argument("--reason", default="")
    rework.add_argument("--skip-if-active-pipeline", action="store_true")
    rework.set_defaults(func=cmd_mark_rework)

    dashboard = subparsers.add_parser("dashboard", help="summarize AI tasks, ai-staging MRs, and promotion MRs")
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
