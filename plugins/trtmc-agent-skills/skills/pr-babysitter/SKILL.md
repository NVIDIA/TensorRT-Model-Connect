---
name: pr-babysitter
description: >-
  Use when monitoring GitHub pull request CI, diagnosing failed checks, rebasing
  branches onto github/main, applying narrowly scoped fixes, and updating PRs
  until their latest checks are green or a human blocker is identified.
---

# PR Babysitter

## Purpose

Monitor open non-draft GitHub PRs, classify their CI state, rebase when they are
behind `main`, diagnose failed checks, make minimal fixes on the PR source
branch, and push updated commits. Work sequentially and report every PR state.

## Ground Rules

- Use the `github` remote and target `main`.
- Never push to `main`.
- Never change unrelated PR branches.
- Do not rewrite a PR's intent during rebase. Preserve the feature and adapt it
  to current `main`.
- Do not skip CI unless the user explicitly asks.
- Never request, enable, or queue GitHub auto-merge. Do not use
  `gh pr merge --auto` or any equivalent API flag.
- When merge authority is granted, merge only after the latest CI for the
  current PR head has completed successfully and the PR is mergeable.
- If a fix needs unavailable hardware, product judgment, or broad scope, stop
  and report the blocker.

## Lock Guard

Use a short-lived lock for long monitoring cycles:

```bash
LOCK_FILE="/tmp/trtmc_pr_babysitter.lock"
if [ -f "$LOCK_FILE" ]; then
  lock_age=$(( $(date +%s) - $(stat -c %Y "$LOCK_FILE" 2>/dev/null || echo 0) ))
  if [ "$lock_age" -lt 3600 ]; then
    echo "Previous PR babysitter cycle still running."
    exit 0
  fi
fi
touch "$LOCK_FILE"
```

Remove it before exiting:

```bash
rm -f "$LOCK_FILE"
```

## Survey PRs

Fetch current state dynamically. Do not hardcode PR numbers.

```bash
git fetch github main --prune
gh pr list \
  --repo NVIDIA/TensorRT-Model-Connect \
  --state open \
  --json number,title,headRefName,baseRefName,isDraft,mergeStateStatus,reviewDecision,statusCheckRollup,headRepositoryOwner
```

Skip draft PRs and PRs not targeting `main` unless the user asked for them.

For a specific PR:

```bash
gh pr view <number> \
  --repo NVIDIA/TensorRT-Model-Connect \
  --json number,title,headRefName,baseRefName,isDraft,mergeStateStatus,reviewDecision,statusCheckRollup,commits,files,url

gh pr checks <number> --repo NVIDIA/TensorRT-Model-Connect
```

## Classify

| State | Action |
|-------|--------|
| Checks pending/running | Wait; do not push noise; never request auto-merge |
| Checks green and branch current | Merge if merge authority was granted; otherwise report OK |
| Checks green but branch behind main | Rebase and push |
| Checks failed and branch behind main | Rebase first, then diagnose/fix |
| Checks failed and branch current | Diagnose/fix |
| Merge conflicts | Rebase locally if mechanical; otherwise report blocker |
| Missing required review | Report waiting on review |

## Merge After CI

Only merge when the user has explicitly authorized merging. Never treat
auto-merge as a way to wait for CI. GitHub auto-merge has previously accepted a
merge request while a check was still queued, so it is forbidden for this skill.

Before merging, verify all of the following against the latest PR head:

- `gh pr checks <number> --repo NVIDIA/TensorRT-Model-Connect --watch --interval 30`
  exits successfully.
- `gh pr view <number> --repo NVIDIA/TensorRT-Model-Connect --json headRefOid,mergeable,mergeStateStatus,statusCheckRollup`
  shows the same head SHA whose checks passed.
- `mergeable` is `MERGEABLE` and `mergeStateStatus` is clean enough for the
  repository ruleset, such as `CLEAN`.
- No required check is pending, queued, running, skipped unexpectedly, or
  failing.

If any check is pending or queued, wait and poll. If any check fails, diagnose
and fix it. Do not merge.

Use an explicit merge command only after those checks pass. Do not include
`--auto`:

```bash
gh pr merge <number> \
  --repo NVIDIA/TensorRT-Model-Connect \
  --squash \
  --delete-branch \
  --subject "<reviewed squash title>" \
  --body "<reviewed squash body>"
```

After merging, verify the PR state is `MERGED` and report the merge commit. If a
linked issue was expected to close, verify the issue state separately.

## Rebase

```bash
git fetch github main <branch>
git switch <branch>
git pull --ff-only github <branch>
git rebase github/main
```

For mechanical conflicts, read both sides, resolve, `git add`, and continue.
For semantic conflicts, stop and report exactly which files and decisions need a
human.

Push with lease:

```bash
git push github HEAD:<branch> --force-with-lease
```

## Diagnose Failed Checks

List recent runs:

```bash
gh run list --repo NVIDIA/TensorRT-Model-Connect --branch <branch> --limit 10
```

Inspect failed logs:

```bash
gh run view <run-id> --repo NVIDIA/TensorRT-Model-Connect --log-failed
```

Download artifacts when a job points to structured outputs:

```bash
mkdir -p .ci_artifacts/pr<number>
gh run download <run-id> --repo NVIDIA/TensorRT-Model-Connect --dir .ci_artifacts/pr<number>
```

When E2E artifacts exist, inspect `result.json` fields such as `status`,
`failure_type`, stage statuses, stage messages, metric values, and thresholds.

Classify failures:

| Diagnosis | Action |
|-----------|--------|
| Build or test error caused by PR code | Fix the PR code |
| Failure caused by rebase drift | Adapt the PR to current `main` |
| E2E failure also present on current `main` | Report as pre-existing; do not mask it |
| Infrastructure or unavailable hardware | Report with run/job links |
| Threshold-only disagreement near baseline | Compare against `main` before changing thresholds |

## Fix

- Read the PR diff first with `git diff github/main...HEAD`.
- Make the smallest code or test change that addresses the failure.
- Run the most relevant local verification available.
- Commit with a clear message that does not mention prohibited tool names.
- Push back to the same PR branch.
- Add a PR comment only when it communicates root cause, validation, or a
  blocker not already visible from commits.

## Report

Always include a complete cycle summary:

```text
PR     Action      Branch                         Details
#123   OK          feature-a                      Checks green
#124   REBASED     feature-b                      Rebased onto github/main and pushed
#125   FIXED       feature-c                      Fixed compile error; pytest ... passed
#126   BLOCKED     feature-d                      Semantic conflict in src/runtime/...
```

Include commands run, important check/run links, files changed, and residual
risk. If the cycle made changes, say which PR branch was pushed.
