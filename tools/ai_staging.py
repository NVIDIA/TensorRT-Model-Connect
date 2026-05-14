#!/usr/bin/env python3
"""Maintain the AI staging branch and route AI-generated pull requests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


DEFAULT_BRANCH = "ai-staging"
DEFAULT_REMOTE = "github"
DEFAULT_SOURCE_PREFIXES = ("ai-task-",)
DEFAULT_PROMOTION_PREFIX = "ai-staging-promotion"
AI_STAGING_PR_LABELS = "ai-generated,ai:staging-pr"
AI_PROMOTION_PR_LABELS = "ai:promotion"
ACTIVE_CHECK_STATUSES = {"created", "waiting_for_resource", "preparing", "pending", "running"}
FAILED_CHECK_STATUSES = {"failed", "canceled"}


@dataclass(frozen=True)
class Config:
    remote: str
    branch: str
    project: str
    source_prefixes: tuple[str, ...]
    dry_run: bool
    require_api_token: bool


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    dry_run: bool = False,
    cwd: str | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    location = f"(cd {shlex.quote(cwd)} && " if cwd else ""
    suffix = ")" if cwd else ""
    print("+ " + location + shlex.join(cmd) + suffix, file=sys.stderr)
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    result = subprocess.run(cmd, check=False, capture_output=capture, input=input_text, text=True, cwd=cwd)
    if check and result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        message = f"Command failed with exit code {result.returncode}: {shlex.join(cmd)}"
        if details:
            message += "\n" + details
        raise SystemExit(message)
    return result


def git(args: list[str], *, check: bool = True, dry_run: bool = False, cwd: str | None = None) -> str:
    result = run(["git", *args], check=check, capture=True, dry_run=dry_run, cwd=cwd)
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
        raise SystemExit(
            f"Could not infer GitHub project path from remote URL: {remote_url!r}. "
            "Pass --project explicitly."
        )
    return path


def encoded_project_path(project: str) -> str:
    return project.strip()


def config_from_args(args: argparse.Namespace) -> Config:
    project = args.project or os.environ.get("GITHUB_REPOSITORY") or infer_project_path(args.remote)
    prefixes = tuple(args.source_prefix or DEFAULT_SOURCE_PREFIXES)
    return Config(
        remote=args.remote,
        branch=args.branch,
        project=encoded_project_path(project),
        source_prefixes=prefixes,
        dry_run=args.dry_run,
        require_api_token=getattr(args, "require_api_token", False),
    )


def fetch_branch(cfg: Config, branch: str) -> None:
    git(
        [
            "fetch",
            cfg.remote,
            f"+refs/heads/{branch}:refs/remotes/{cfg.remote}/{branch}",
        ]
    )


def remote_ref(cfg: Config, branch: str) -> str:
    return f"{cfg.remote}/{branch}"


def remote_branch_exists(cfg: Config, branch: str) -> bool:
    output = git(["ls-remote", "--heads", cfg.remote, branch])
    return bool(output)


def local_branch_exists(branch: str) -> bool:
    result = run(["git", "rev-parse", "--verify", f"refs/heads/{branch}"], check=False)
    return result.returncode == 0


def assert_clean_worktree() -> None:
    status = git(["status", "--porcelain"])
    if status:
        raise SystemExit(
            "Refusing to operate with local changes present. Commit, stash, or use a clean worktree.\n"
            + status
        )


def ensure_branch(cfg: Config, *, source_branch: str = "main") -> bool:
    fetch_branch(cfg, source_branch)
    if remote_branch_exists(cfg, cfg.branch):
        print(f"{cfg.remote} branch exists: {cfg.branch}")
        fetch_branch(cfg, cfg.branch)
        return False

    source = f"refs/remotes/{cfg.remote}/{source_branch}:refs/heads/{cfg.branch}"
    run(["git", "push", cfg.remote, source], dry_run=cfg.dry_run)
    print(f"created {cfg.remote}/{cfg.branch} from {cfg.remote}/{source_branch}")
    return True


def sync_branch(cfg: Config, *, push: bool) -> None:
    assert_clean_worktree()
    ensure_branch(cfg)

    if cfg.dry_run:
        print(f"would sync {cfg.remote}/{cfg.branch} with {cfg.remote}/main")
        if push:
            print(f"would push synced HEAD to {cfg.remote}/{cfg.branch}")
        return

    worktree_path = tempfile.mkdtemp(prefix=f"{cfg.branch}-sync-")
    keep_worktree = False
    added_worktree = run(
        ["git", "worktree", "add", "--detach", worktree_path, f"{cfg.remote}/{cfg.branch}"],
        check=False,
    )
    if added_worktree.returncode != 0:
        shutil.rmtree(worktree_path, ignore_errors=True)
        details = (added_worktree.stderr or added_worktree.stdout or "").strip()
        raise SystemExit(f"Could not create temporary sync worktree at {worktree_path}.\n{details}")

    try:
        contains_main = run(
            ["git", "merge-base", "--is-ancestor", f"{cfg.remote}/main", "HEAD"],
            check=False,
            cwd=worktree_path,
        )
        if contains_main.returncode == 0:
            print(f"{cfg.branch} already contains {cfg.remote}/main")
        else:
            git(["merge", "--no-edit", f"{cfg.remote}/main"], cwd=worktree_path)

        if push:
            run(["git", "push", cfg.remote, f"HEAD:refs/heads/{cfg.branch}"], cwd=worktree_path)
    except BaseException:
        keep_worktree = True
        print(f"Preserving failed sync worktree for inspection: {worktree_path}", file=sys.stderr)
        raise
    finally:
        if not keep_worktree:
            run(["git", "worktree", "remove", "--force", worktree_path], check=False)
            shutil.rmtree(worktree_path, ignore_errors=True)


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
    raise SystemExit(
        f"Could not infer GitHub server URL from remote URL: {remote_url!r}. "
        "Set GH_API_URL."
    )


def github_api_base_url(cfg: Config) -> str:
    if os.environ.get("GH_API_URL"):
        return os.environ["GH_API_URL"].rstrip("/")
    server = infer_github_server_url(cfg.remote).rstrip("/")
    if server == "https://github.com":
        return "https://api.github.com"
    return server + "/api/v3"


def github_token_header() -> tuple[str, str] | None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return "Authorization", f"Bearer {token}"
    return None


def missing_token_message() -> str:
    return (
        "GITHUB_TOKEN is required for GitHub REST API access. "
        "Create a GitHub Actions secret or export GH_TOKEN/GITHUB_TOKEN with repository access."
    )


def repo_path(cfg: Config, suffix: str = "") -> str:
    suffix = suffix.lstrip("/")
    return f"/repos/{cfg.project}/{suffix}" if suffix else f"/repos/{cfg.project}"


def gh_api_json(path: str, *, method: str | None = None, fields: dict[str, Any] | None = None) -> Any:
    cmd = ["gh", "api"]
    if method:
        cmd.extend(["-X", method])
    cmd.append(path)
    input_text = None
    if fields is not None:
        cmd.extend(["--input", "-"])
        input_text = json.dumps(fields)
    result = run(cmd, input_text=input_text)
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def http_api_json(
    cfg: Config,
    path: str,
    *,
    method: str | None = None,
    fields: dict[str, Any] | None = None,
) -> Any:
    token_header = github_token_header()
    if not token_header:
        raise SystemExit(missing_token_message())

    data = json.dumps(fields).encode() if fields is not None else None
    request_method = method or ("POST" if data else "GET")
    url = github_api_base_url(cfg) + path
    print(f"+ {request_method} {url}", file=sys.stderr)
    request = urllib.request.Request(url, data=data, method=request_method)
    request.add_header(*token_header)
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

    if not payload.strip():
        return None
    return json.loads(payload)


def github_api_json(
    cfg: Config,
    path: str,
    *,
    method: str | None = None,
    fields: dict[str, Any] | None = None,
) -> Any:
    is_write = (method or "").upper() in {"POST", "PUT", "PATCH", "DELETE"} or bool(fields)
    if cfg.dry_run and is_write:
        print(f"+ DRY RUN GitHub API {method or 'POST'} {path} {fields or {}}", file=sys.stderr)
        return None
    if github_token_header():
        return http_api_json(cfg, path, method=method, fields=fields)
    if cfg.require_api_token:
        raise SystemExit(missing_token_message())
    return gh_api_json(path, method=method, fields=fields)


def github_paginated(cfg: Config, path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    separator = "&" if "?" in path else "?"
    page = 1
    while True:
        batch = github_api_json(cfg, f"{path}{separator}per_page=100&page={page}")
        if not isinstance(batch, list):
            raise SystemExit(f"Unexpected paginated GitHub API response for {path}: {batch!r}")
        if not batch:
            break
        items.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return items


def pr_number(pr: dict[str, Any]) -> int:
    value = pr.get("number") or pr.get("id")
    if value is None:
        raise SystemExit(f"GitHub pull request has no number: {pr!r}")
    return int(value)


def pr_url(pr: dict[str, Any]) -> str:
    return str(pr.get("html_url") or "")


def pr_head_ref(pr: dict[str, Any]) -> str:
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    return str(head.get("ref") or "")


def pr_base_ref(pr: dict[str, Any]) -> str:
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    return str(base.get("ref") or "")


def pr_head_sha(pr: dict[str, Any]) -> str:
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    return str(head.get("sha") or "")


def label_names(item: dict[str, Any]) -> list[str]:
    labels = item.get("labels") or []
    names: list[str] = []
    for label in labels:
        if isinstance(label, str):
            names.append(label)
        elif isinstance(label, dict) and label.get("name") is not None:
            names.append(str(label["name"]))
    return names


def csv_label_names(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def add_pr_labels(cfg: Config, pr: dict[str, Any], labels: str | None) -> None:
    requested = csv_label_names(labels)
    if not requested:
        return
    merged = sorted(set(label_names(pr)).union(requested))
    github_api_json(cfg, repo_path(cfg, f"issues/{pr_number(pr)}/labels"), method="PUT", fields={"labels": merged})


def pr_ci_status(cfg: Config, pr: dict[str, Any]) -> str:
    sha = pr_head_sha(pr)
    if not sha:
        return "unknown"
    try:
        data = github_api_json(cfg, repo_path(cfg, f"commits/{sha}/check-runs"))
    except SystemExit as exc:
        print(f"warning: could not inspect checks for PR #{pr_number(pr)}: {exc}", file=sys.stderr)
        return "unknown"
    check_runs = data.get("check_runs") if isinstance(data, dict) else None
    if not check_runs:
        return "none"
    statuses = {str(item.get("status") or "") for item in check_runs}
    conclusions = {str(item.get("conclusion") or "") for item in check_runs}
    if statuses & {"queued", "in_progress", "requested", "pending", "waiting"}:
        return "running"
    if conclusions & {"failure", "timed_out", "action_required"}:
        return "failed"
    if conclusions & {"cancelled", "canceled"}:
        return "canceled"
    if conclusions <= {"success", "skipped", "neutral", ""}:
        return "success"
    return "unknown"


def failed_check_runs_for_pr(cfg: Config, pr: dict[str, Any]) -> list[dict[str, Any]]:
    sha = pr_head_sha(pr)
    if not sha:
        return []
    data = github_api_json(cfg, repo_path(cfg, f"commits/{sha}/check-runs"))
    check_runs = data.get("check_runs") if isinstance(data, dict) else []
    return [
        item
        for item in check_runs
        if item.get("conclusion") in {"failure", "timed_out", "action_required", "cancelled", "canceled"}
    ]


def is_draft(pr: dict[str, Any]) -> bool:
    return bool(pr.get("draft") or pr.get("work_in_progress"))


def list_open_prs(cfg: Config) -> list[dict[str, Any]]:
    prs: list[dict[str, Any]] = []
    page = 1
    while True:
        path = (
            repo_path(cfg, "pulls")
            + f"?state=open&per_page=100&page={page}&sort=created&direction=asc"
        )
        batch = github_api_json(cfg, path)
        if not isinstance(batch, list):
            raise SystemExit(f"Unexpected GitHub API response for page {page}: {batch!r}")
        if not batch:
            break
        prs.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return prs


def matching_ai_prs(cfg: Config, *, skip_drafts: bool) -> list[dict[str, Any]]:
    matched = []
    for pr in list_open_prs(cfg):
        source_branch = pr_head_ref(pr)
        if not source_branch.startswith(cfg.source_prefixes):
            continue
        if skip_drafts and is_draft(pr):
            continue
        matched.append(pr)
    return matched


def print_pr_table(prs: list[dict[str, Any]], cfg: Config) -> None:
    if not prs:
        print("No matching pull requests.")
        return
    print("PR    target        source                                      checks    title")
    print("----  ------------  ------------------------------------------  --------  -----")
    for pr in prs:
        status = pr_ci_status(cfg, pr)
        title = str(pr.get("title") or "").replace("\n", " ")
        print(
            f"#{pr_number(pr):<3}  "
            f"{pr_base_ref(pr) or '-':<12}  "
            f"{pr_head_ref(pr) or '-':<42}  "
            f"{status:<8}  "
            f"{title[:90]}"
        )


def cmd_ensure(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    ensure_branch(cfg)
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    sync_branch(cfg, push=args.push)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    prs = matching_ai_prs(cfg, skip_drafts=args.skip_drafts)
    if args.json:
        print(json.dumps(prs, indent=2, sort_keys=True))
    else:
        print_pr_table(prs, cfg)
    return 0


def cmd_retarget(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    prs = matching_ai_prs(cfg, skip_drafts=args.skip_drafts)
    changed = 0
    for pr in prs:
        number = pr_number(pr)
        source = pr_head_ref(pr)
        current_target = pr_base_ref(pr)
        if not args.all_targets and current_target != args.from_target:
            continue
        if current_target == cfg.branch:
            continue

        print(f"#{number}: {source} {current_target} -> {cfg.branch}")
        changed += 1
        if cfg.dry_run:
            continue
        updated = github_api_json(
            cfg,
            repo_path(cfg, f"pulls/{number}"),
            method="PATCH",
            fields={"base": cfg.branch},
        )
        if isinstance(updated, dict):
            add_pr_labels(cfg, updated, AI_STAGING_PR_LABELS)

    print(f"retargeted {changed} pull request(s)" if not cfg.dry_run else f"would retarget {changed} pull request(s)")
    return 0


def cmd_full_cycle(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    sync_branch(cfg, push=args.push)
    if args.retarget:
        return cmd_retarget(args)
    print("retarget step skipped; pass --retarget to update matching pull requests")
    return 0


def branch_has_tree_diff(cfg: Config, target_branch: str, *, source_branch: str | None = None) -> bool:
    source_branch = source_branch or cfg.branch
    fetch_branch(cfg, target_branch)
    fetch_branch(cfg, source_branch)
    result = run(
        ["git", "diff", "--quiet", remote_ref(cfg, target_branch), remote_ref(cfg, source_branch)],
        check=False,
    )
    if result.returncode == 0:
        return False
    if result.returncode == 1:
        return True
    raise SystemExit(f"git diff failed with exit code {result.returncode}")


def branch_contains_target(cfg: Config, target_branch: str, *, source_branch: str | None = None) -> bool:
    source_branch = source_branch or cfg.branch
    result = run(
        ["git", "merge-base", "--is-ancestor", remote_ref(cfg, target_branch), remote_ref(cfg, source_branch)],
        check=False,
    )
    return result.returncode == 0


def git_lines(args: list[str]) -> list[str]:
    output = git(args)
    return output.splitlines() if output else []


def limited_lines(lines: list[str], *, limit: int) -> list[str]:
    if len(lines) <= limit:
        return lines
    omitted = len(lines) - limit
    return [*lines[:limit], f"... ({omitted} more line(s) omitted)"]


def status_counts(name_status_lines: list[str]) -> dict[str, int]:
    counts = {"added": 0, "modified": 0, "deleted": 0, "renamed": 0, "other": 0}
    for line in name_status_lines:
        status = line.split("\t", 1)[0]
        if status == "A":
            counts["added"] += 1
        elif status == "M":
            counts["modified"] += 1
        elif status == "D":
            counts["deleted"] += 1
        elif status.startswith("R"):
            counts["renamed"] += 1
        else:
            counts["other"] += 1
    return counts


def promotion_change_summary(cfg: Config, target_branch: str, *, source_branch: str | None = None) -> dict[str, Any]:
    source_branch = source_branch or cfg.branch
    base = remote_ref(cfg, target_branch)
    head = remote_ref(cfg, source_branch)
    rev_range = f"{base}..{head}"
    symmetric_range = f"{base}...{head}"

    target_sha = git(["rev-parse", "--short=12", base])
    source_sha = git(["rev-parse", "--short=12", head])
    commits = git_lines(
        [
            "log",
            "--cherry-pick",
            "--right-only",
            "--no-merges",
            "--pretty=format:%h %s",
            symmetric_range,
        ]
    )
    name_status = git_lines(["diff", "--name-status", "--find-renames", rev_range])
    diffstat = git_lines(["diff", "--stat", "--find-renames", rev_range])
    counts = status_counts(name_status)

    return {
        "target_sha": target_sha,
        "source_sha": source_sha,
        "commits": commits,
        "name_status": name_status,
        "diffstat": diffstat,
        "counts": counts,
    }


def find_promotion_pr(cfg: Config, target_branch: str, *, source_branch: str | None = None) -> dict[str, Any] | None:
    source_branch = source_branch or cfg.branch
    query = urllib.parse.urlencode(
        {
            "state": "open",
            "base": target_branch,
            "per_page": "100",
        }
    )
    response = github_api_json(cfg, repo_path(cfg, f"pulls?{query}"))
    if not isinstance(response, list):
        raise SystemExit(f"Unexpected GitHub API response while finding promotion PR: {response!r}")
    for pr in response:
        if pr_head_ref(pr) == source_branch:
            return pr
    return None


def promotion_description(
    cfg: Config,
    target_branch: str,
    *,
    is_up_to_date: bool,
    source_branch: str | None = None,
    summary_source_branch: str | None = None,
    rotation: dict[str, str] | None = None,
) -> str:
    source_branch = source_branch or cfg.branch
    summary = promotion_change_summary(cfg, target_branch, source_branch=summary_source_branch or source_branch)
    counts = summary["counts"]
    commits = limited_lines(summary["commits"], limit=40)
    changed_paths = limited_lines(summary["name_status"], limit=80)
    diffstat = limited_lines(summary["diffstat"], limit=80)
    status = (
        f"`{remote_ref(cfg, source_branch)}` contains `{remote_ref(cfg, target_branch)}`."
        if is_up_to_date
        else f"WARNING: `{remote_ref(cfg, source_branch)}` does not contain `{remote_ref(cfg, target_branch)}`. "
        "Review carefully before merging."
    )
    commit_block = "\n".join(f"- `{line.split(' ', 1)[0]}` {line.split(' ', 1)[1] if ' ' in line else ''}" for line in commits)
    if not commit_block:
        commit_block = "- No unique non-merge commit subjects found; review the tree diff below."
    changed_paths_block = "\n".join(changed_paths) if changed_paths else "No changed paths."
    diffstat_block = "\n".join(diffstat) if diffstat else "No diffstat."

    rotation_lines = []
    if rotation:
        rotation_lines = [
            "",
            "Rotation state:",
            "",
            f"- Snapshot branch: `{remote_ref(cfg, source_branch)}`",
            f"- Snapshot was cut from `{remote_ref(cfg, cfg.branch)}` @ `{rotation['staging_sha']}`",
            f"- `{remote_ref(cfg, cfg.branch)}` was reset to `{remote_ref(cfg, target_branch)}` @ `{rotation['target_sha']}` for future AI PRs",
        ]

    return "\n".join(
        [
            f"Promotion PR from `{source_branch}` to `{target_branch}` for human review.",
            "",
            f"This PR does not auto-merge. It promotes the current AI staging tree into `{target_branch}` after the full PR checks are green.",
            "",
            "Branch state:",
            "",
            f"- Source: `{remote_ref(cfg, source_branch)}` @ `{summary['source_sha']}`",
            f"- Target: `{remote_ref(cfg, target_branch)}` @ `{summary['target_sha']}`",
            f"- Up to date with target: {'yes' if is_up_to_date else 'no'}",
            *rotation_lines,
            "",
            status,
            "",
            "What is being promoted:",
            "",
            commit_block,
            "",
            "Net file change summary:",
            "",
            f"- Added: {counts['added']}",
            f"- Modified: {counts['modified']}",
            f"- Deleted: {counts['deleted']}",
            f"- Renamed: {counts['renamed']}",
            f"- Other: {counts['other']}",
            "",
            "Changed paths:",
            "",
            "```text",
            changed_paths_block,
            "```",
            "",
            "Diffstat:",
            "",
            "```text",
            diffstat_block,
            "```",
            "",
            "Review checklist:",
            "",
            "- Full PR checks are green.",
            "- Diff contains only expected AI staging changes.",
            f"- `{source_branch}` is up to date with `{target_branch}` before merge.",
            "- Individual AI-generated PRs included in this promotion have task scope, verification, and risk notes in their descriptions.",
            "",
            "Reviewer note: the tree diff and changed paths above are authoritative. Commit subjects are included as a readable summary, but may include staging-history commits when earlier promotions were squashed.",
        ]
    )


def timestamped_snapshot_branch(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix.rstrip('-')}-{timestamp}"


def push_snapshot_branch(cfg: Config, snapshot_branch: str, *, source_branch: str) -> None:
    source_ref = f"refs/remotes/{cfg.remote}/{source_branch}:refs/heads/{snapshot_branch}"
    run(["git", "push", cfg.remote, source_ref], dry_run=cfg.dry_run)
    action = "would create" if cfg.dry_run else "created"
    print(f"{action} snapshot branch {cfg.remote}/{snapshot_branch} from {cfg.remote}/{source_branch}")


def reset_remote_branch_to_target(
    cfg: Config,
    *,
    branch: str,
    target_branch: str,
    expected_old_sha: str,
) -> None:
    target_ref = f"refs/remotes/{cfg.remote}/{target_branch}:refs/heads/{branch}"
    lease = f"--force-with-lease=refs/heads/{branch}:{expected_old_sha}"
    run(["git", "push", lease, cfg.remote, target_ref], dry_run=cfg.dry_run)
    action = "would reset" if cfg.dry_run else "reset"
    print(f"{action} {cfg.remote}/{branch} to {cfg.remote}/{target_branch}")


def create_or_update_promotion_pr(
    cfg: Config,
    *,
    source_branch: str,
    target_branch: str,
    title: str,
    description: str,
    labels: str | None = AI_PROMOTION_PR_LABELS,
) -> dict[str, Any] | None:
    if cfg.dry_run:
        print(f"would create or update promotion PR from {source_branch} to {target_branch}: {title}")
        return None

    existing = find_promotion_pr(cfg, target_branch, source_branch=source_branch)
    if existing:
        fields = {
            "title": title,
            "body": description,
        }
        updated = github_api_json(
            cfg,
            repo_path(cfg, f"pulls/{pr_number(existing)}"),
            method="PATCH",
            fields=fields,
        )
        result = updated if isinstance(updated, dict) else existing
        add_pr_labels(cfg, result, labels)
        print(f"Updated promotion PR: #{pr_number(result)} {pr_url(result)}")
        return result

    fields = {
        "head": source_branch,
        "base": target_branch,
        "title": title,
        "body": description,
        "maintainer_can_modify": True,
    }
    pr = github_api_json(cfg, repo_path(cfg, "pulls"), method="POST", fields=fields)
    if not isinstance(pr, dict):
        raise SystemExit(f"Unexpected GitHub API response while creating promotion PR: {pr!r}")
    add_pr_labels(cfg, pr, labels)
    print(f"Created promotion PR: #{pr_number(pr)} {pr_url(pr)}")
    return pr


def cmd_promote(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    target_branch = args.target_branch
    if not branch_has_tree_diff(cfg, target_branch):
        print(f"No tree diff between {cfg.remote}/{target_branch} and {cfg.remote}/{cfg.branch}; no PR needed.")
        return 0

    is_up_to_date = branch_contains_target(cfg, target_branch)
    description = promotion_description(cfg, target_branch, is_up_to_date=is_up_to_date)
    create_or_update_promotion_pr(
        cfg,
        source_branch=cfg.branch,
        target_branch=target_branch,
        title=f"chore: promote {cfg.branch} to {target_branch}",
        description=description,
    )
    return 0


def cmd_rotate_promotion(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    target_branch = args.target_branch
    assert_clean_worktree()
    ensure_branch(cfg, source_branch=target_branch)
    has_tree_diff = branch_has_tree_diff(cfg, target_branch)

    staging_ref = remote_ref(cfg, cfg.branch)
    target_ref = remote_ref(cfg, target_branch)
    staging_sha = git(["rev-parse", staging_ref])
    target_sha = git(["rev-parse", target_ref])

    if not has_tree_diff:
        if staging_sha != target_sha:
            reset_remote_branch_to_target(
                cfg,
                branch=cfg.branch,
                target_branch=target_branch,
                expected_old_sha=staging_sha,
            )
        else:
            print(f"{staging_ref} already matches {target_ref}; no promotion PR needed.")
        return 0

    snapshot_branch = args.snapshot_branch or timestamped_snapshot_branch(args.snapshot_prefix)
    if remote_branch_exists(cfg, snapshot_branch):
        raise SystemExit(f"Snapshot branch already exists: {cfg.remote}/{snapshot_branch}")

    push_snapshot_branch(cfg, snapshot_branch, source_branch=cfg.branch)
    summary_source_branch = cfg.branch if cfg.dry_run else snapshot_branch
    if not cfg.dry_run:
        fetch_branch(cfg, snapshot_branch)

    reset_remote_branch_to_target(
        cfg,
        branch=cfg.branch,
        target_branch=target_branch,
        expected_old_sha=staging_sha,
    )

    is_up_to_date = branch_contains_target(cfg, target_branch, source_branch=summary_source_branch)
    description = promotion_description(
        cfg,
        target_branch,
        source_branch=snapshot_branch,
        summary_source_branch=summary_source_branch,
        is_up_to_date=is_up_to_date,
        rotation={"staging_sha": staging_sha[:12], "target_sha": target_sha[:12]},
    )
    create_or_update_promotion_pr(
        cfg,
        source_branch=snapshot_branch,
        target_branch=target_branch,
        title=f"chore: promote {snapshot_branch} to {target_branch}",
        description=description,
    )
    return 0


def normalized_snapshot_prefix(prefix: str) -> str:
    return f"{prefix.rstrip('-')}-"


def list_open_promotion_prs(cfg: Config, *, target_branch: str, snapshot_prefix: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "state": "open",
            "base": target_branch,
            "sort": "created",
            "direction": "asc",
        }
    )
    prs = github_paginated(cfg, repo_path(cfg, f"pulls?{query}"))
    return [
        pr
        for pr in prs
        if pr_head_ref(pr).startswith(normalized_snapshot_prefix(snapshot_prefix))
    ]


def format_failed_checks(checks: list[dict[str, Any]], *, limit: int = 10) -> str:
    if not checks:
        return "no failed checks found"
    lines = []
    for check in checks[:limit]:
        lines.append(
            f"{check.get('name', '-')} "
            f"conclusion={check.get('conclusion', '-')} "
            f"url={check.get('html_url', '-')}"
        )
    if len(checks) > limit:
        lines.append(f"... {len(checks) - limit} more failed check(s)")
    return "\n".join(lines)


def safe_tmp_prefix(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value)


def clean_rebase_promotion_branch(cfg: Config, *, source_branch: str, target_branch: str) -> bool:
    fetch_branch(cfg, target_branch)
    fetch_branch(cfg, source_branch)
    old_sha = git(["rev-parse", remote_ref(cfg, source_branch)])
    worktree_path = tempfile.mkdtemp(prefix=f"{safe_tmp_prefix(source_branch)}-rebase-")
    keep_worktree = False
    added = run(
        ["git", "worktree", "add", "--detach", worktree_path, remote_ref(cfg, source_branch)],
        check=False,
    )
    if added.returncode != 0:
        shutil.rmtree(worktree_path, ignore_errors=True)
        details = (added.stderr or added.stdout or "").strip()
        raise SystemExit(f"Could not create temporary promotion rebase worktree at {worktree_path}.\n{details}")

    try:
        rebased = run(["git", "rebase", remote_ref(cfg, target_branch)], check=False, cwd=worktree_path)
        if rebased.returncode != 0:
            run(["git", "rebase", "--abort"], check=False, cwd=worktree_path)
            print(f"! promotion rebase conflict: {cfg.remote}/{source_branch} onto {cfg.remote}/{target_branch}")
            return False

        lease = f"--force-with-lease=refs/heads/{source_branch}:{old_sha}"
        run(
            ["git", "push", lease, cfg.remote, f"HEAD:refs/heads/{source_branch}"],
            dry_run=cfg.dry_run,
            cwd=worktree_path,
        )
        action = "would rebase" if cfg.dry_run else "rebased"
        print(f"{action} {cfg.remote}/{source_branch} onto {cfg.remote}/{target_branch}")
        return True
    except BaseException:
        keep_worktree = True
        print(f"Preserving failed promotion rebase worktree for inspection: {worktree_path}", file=sys.stderr)
        raise
    finally:
        if not keep_worktree:
            run(["git", "worktree", "remove", "--force", worktree_path], check=False)
            shutil.rmtree(worktree_path, ignore_errors=True)


def cmd_babysit_promotion(args: argparse.Namespace) -> int:
    cfg = config_from_args(args)
    target_branch = args.target_branch
    assert_clean_worktree()
    fetch_branch(cfg, target_branch)
    promotion_prs = list_open_promotion_prs(
        cfg,
        target_branch=target_branch,
        snapshot_prefix=args.snapshot_prefix,
    )
    if not promotion_prs:
        print(f"No open promotion PRs targeting {target_branch} with prefix {normalized_snapshot_prefix(args.snapshot_prefix)}")
        return 0

    actions = 0
    for pr in promotion_prs:
        source_branch = pr_head_ref(pr)
        number = pr_number(pr)
        check_status = pr_ci_status(cfg, pr)

        if not remote_branch_exists(cfg, source_branch):
            print(f"#{number}: source branch missing: {cfg.remote}/{source_branch}")
            continue

        fetch_branch(cfg, source_branch)
        up_to_date = branch_contains_target(cfg, target_branch, source_branch=source_branch)
        print(
            f"#{number}: source={source_branch} checks={check_status} "
            f"up_to_date_with_{target_branch}={'yes' if up_to_date else 'no'} {pr_url(pr)}"
        )

        if not up_to_date and check_status not in ACTIVE_CHECK_STATUSES:
            if actions >= args.max_rebases:
                print(f"#{number}: skipped rebase; max rebases reached for this cycle")
                continue
            rebased = clean_rebase_promotion_branch(
                cfg,
                source_branch=source_branch,
                target_branch=target_branch,
            )
            actions += 1
            if rebased:
                fetch_branch(cfg, source_branch)
                description = promotion_description(
                    cfg,
                    target_branch,
                    source_branch=source_branch,
                    is_up_to_date=True,
                )
                create_or_update_promotion_pr(
                    cfg,
                    source_branch=source_branch,
                    target_branch=target_branch,
                    title=f"chore: promote {source_branch} to {target_branch}",
                    description=description,
                )
            else:
                print(f"#{number}: needs human rebase/conflict review")
            continue

        if check_status in ACTIVE_CHECK_STATUSES:
            print(f"#{number}: waiting for full PR checks")
        elif check_status == "success":
            print(f"#{number}: ready for human review")
        elif check_status in FAILED_CHECK_STATUSES:
            checks = failed_check_runs_for_pr(cfg, pr)
            print(
                f"#{number}: promotion checks {check_status}; "
                f"repair {source_branch} and push a new full run. Failed checks:\n{format_failed_checks(checks)}"
            )
        else:
            print(f"#{number}: promotion check status is {check_status}; no action")

    return 0


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote", default=os.environ.get("AI_STAGING_REMOTE", DEFAULT_REMOTE))
    parser.add_argument("--branch", default=os.environ.get("AI_STAGING_BRANCH", DEFAULT_BRANCH))
    parser.add_argument("--project", help="GitHub project path or numeric project id")
    parser.add_argument(
        "--source-prefix",
        action="append",
        help="AI source branch prefix to include. May be repeated. Default: ai-task-",
    )
    parser.add_argument("--dry-run", action="store_true", help="print writes without performing them")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ensure = subparsers.add_parser("ensure-branch", help="create the github AI staging branch if missing")
    ensure.set_defaults(func=cmd_ensure)

    sync = subparsers.add_parser("sync-branch", help="merge github/main into the AI staging branch")
    sync.add_argument("--push", action="store_true", help="push the synced branch to github")
    sync.set_defaults(func=cmd_sync)

    list_cmd = subparsers.add_parser("list", help="list open AI-generated pull requests")
    list_cmd.add_argument("--json", action="store_true", help="emit raw JSON")
    list_cmd.add_argument("--skip-drafts", action="store_true", help="exclude draft pull requests")
    list_cmd.set_defaults(func=cmd_list)

    retarget = subparsers.add_parser("retarget", help="retarget matching AI pull requests to the staging branch")
    retarget.add_argument("--from-target", default="main", help="only retarget PRs currently targeting this branch")
    retarget.add_argument("--all-targets", action="store_true", help="retarget regardless of current target branch")
    retarget.add_argument("--skip-drafts", action="store_true", help="exclude draft pull requests")
    retarget.set_defaults(func=cmd_retarget)

    full_cycle = subparsers.add_parser("full-cycle", help="sync branch and optionally retarget matching PRs")
    full_cycle.add_argument("--push", action="store_true", help="push the synced branch to github")
    full_cycle.add_argument("--retarget", action="store_true", help="retarget matching PRs after syncing")
    full_cycle.add_argument("--from-target", default="main", help="only retarget PRs currently targeting this branch")
    full_cycle.add_argument("--all-targets", action="store_true", help="retarget regardless of current target branch")
    full_cycle.add_argument("--skip-drafts", action="store_true", help="exclude draft pull requests")
    full_cycle.set_defaults(func=cmd_full_cycle)

    promote = subparsers.add_parser("promote", help="file an ai-staging to main promotion PR for human review")
    promote.add_argument(
        "--require-api-token",
        action="store_true",
        help="require GITHUB_TOKEN and never fall back to local gh authentication",
    )
    promote.add_argument("--target-branch", default=os.environ.get("AI_STAGING_PROMOTION_TARGET", "main"))
    promote.set_defaults(func=cmd_promote)

    rotate = subparsers.add_parser(
        "rotate-promotion",
        help="snapshot ai-staging to a timestamped branch, reset ai-staging to main, and file a promotion PR",
    )
    rotate.add_argument(
        "--require-api-token",
        action="store_true",
        help="require GITHUB_TOKEN and never fall back to local gh authentication",
    )
    rotate.add_argument("--target-branch", default=os.environ.get("AI_STAGING_PROMOTION_TARGET", "main"))
    rotate.add_argument("--snapshot-prefix", default=DEFAULT_PROMOTION_PREFIX)
    rotate.add_argument("--snapshot-branch", help="explicit snapshot branch name; default is <prefix>-YYYYmmdd-HHMMSS UTC")
    rotate.set_defaults(func=cmd_rotate_promotion)

    babysit = subparsers.add_parser(
        "babysit-promotion",
        help="watch open promotion PRs, clean-rebase outdated snapshot branches, and report full-CI status",
    )
    babysit.add_argument(
        "--require-api-token",
        action="store_true",
        help="require GITHUB_TOKEN and never fall back to local gh authentication",
    )
    babysit.add_argument("--target-branch", default=os.environ.get("AI_STAGING_PROMOTION_TARGET", "main"))
    babysit.add_argument("--snapshot-prefix", default=DEFAULT_PROMOTION_PREFIX)
    babysit.add_argument(
        "--max-rebases",
        type=int,
        default=1,
        help="maximum clean promotion-branch rebases to push in one cycle",
    )
    babysit.set_defaults(func=cmd_babysit_promotion)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
