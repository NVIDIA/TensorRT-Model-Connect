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
- Treat GitHub branch protection and rulesets as insufficient evidence of CI
  safety. The repository may allow a squash or rebase merge while checks are
  still queued, pending, running, or absent; that is still forbidden here.
- Use `$write-git-messages` for fix commits, PR comments, and squash or rebase
  text.
- If a fix needs unavailable hardware, product judgment, or broad scope, stop
  and report the blocker.

## Current CI Contract

Read `.github/workflows/internal-ci-bridge.yml` from current `github/main`.
Source keeps only Internal CI Bridge and Pages workflows. Premerge, including
legal compliance, and nightly execution stay in private Internal CI.

Before triggering premerge, capture the current PR metadata head and inspect its
existing protected status:

```bash
REPOSITORY=NVIDIA/TensorRT-Model-Connect
PR_NUMBER=<number>
pull=$(gh api "repos/$REPOSITORY/pulls/$PR_NUMBER")
PR_HEAD_SHA=$(jq -er '.head.sha' <<<"$pull")
gh api \
  "repos/$REPOSITORY/commits/$PR_HEAD_SHA/status" \
  --jq '[.statuses[] |
    select(.context == "TRTMC Internal CI / Automated premerge gate")][0]'

gh pr edit "$PR_NUMBER" \
  --repo "$REPOSITORY" \
  --add-label run-internal-ci
```

If a label event was created for an older SHA, let the bridge report the
superseded trigger, wait for Community CPU on the current PR head, and retry.
When authorization fails and leaves `run-internal-ci` attached, remove it before
adding it again; adding an existing label does not emit another label event.

Only an actor whose repository permission is `maintain` or `admin`
may add the one-shot trigger.

Never use the legacy `run-ci` label. The bridge consumes `run-internal-ci`,
verifies the open PR targets `main`, and rechecks the event SHA, PR metadata
SHA, and successful Community CPU run for that head before dispatching only
`pr_number` and `head_sha`.

Internal CI resolves and tests the exact pull-request merge whose first parent
is the current `main` revision and whose second parent is the authorized PR
head. It publishes the sanitized result on that exact head SHA.

The Source-visible premerge result is only the sanitized
`TRTMC Internal CI / Automated premerge gate` status on that exact head: `PENDING`, then `PASS` or
`FAIL`, with a target URL on the pull request's checks page. A successful
bridge dispatch is not a successful premerge result.

Raw logs, artifacts, internal packages, runner details, and nightly execution
stay private. Never copy them into Source Actions, Source artifacts, status
target URLs, or PR comments. A source-head guard failure is the exception to
the no-comment norm: the bridge publishes only the public event/PR/branch SHA
diagnostic, recovery instruction, and Source workflow link. It updates one
marked comment instead of creating repeated comments.

- If the current exact head is already `PENDING` or `PASS`, do not add the
  trigger again.
- If the head changes, treat the old result as stale and trigger the new head
  once.
- Retry a failed unchanged head only with explicit rerun authorization.
- Never trigger premerge after the PR has merged. A merge does not rerun the
  same premerge suite; Pages and scheduled Internal nightly follow independent
  paths.

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
auto-merge as a way to wait for CI. A `gh pr merge --auto` invocation has
previously merged immediately while a check was still queued because the
repository ruleset did not require status checks. Auto-merge is forbidden for
this skill. This is an agent-side hard gate: if GitHub would allow the merge
before CI is green, do not merge anyway.

Before merging, verify all of the following against the latest PR head:

- `gh pr checks <number> --repo NVIDIA/TensorRT-Model-Connect --watch --interval 30`
  exits successfully.
- `gh pr view <number> --repo NVIDIA/TensorRT-Model-Connect --json headRefOid,mergeable,mergeStateStatus,statusCheckRollup`
  shows the same head SHA whose checks passed.
- `mergeable` is `MERGEABLE` and `mergeStateStatus` is clean enough for the
  repository ruleset, such as `CLEAN`.
- Every expected CI check in `statusCheckRollup` has completed with conclusion
  `SUCCESS`. Treat `QUEUED`, `PENDING`, `IN_PROGRESS`, `WAITING`,
  `REQUESTED`, empty conclusion, missing check rollup, skipped unexpectedly,
  or failing as a merge blocker.
- The same completed successful check set belongs to the current `headRefOid`.
  If a new commit lands after checks pass, restart the wait.
- The current head has a successful `TRTMC Internal CI / Automated premerge gate` commit status.
  A green bridge run, a result on an older head, or a ruleset without required
  status checks does not satisfy this gate.

If `gh pr checks` exits nonzero, including exit code 8 for pending checks, do
not merge. If any check is pending or queued, wait and poll. If any check fails,
diagnose and fix it. If no CI checks are reported for the head SHA, report a
human blocker instead of merging.

Do not infer safety from `mergeable=MERGEABLE`, an allowed merge button, a
ruleset that lacks required status checks, or successful local tests. Those are
not substitutes for completed successful GitHub CI.

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

Do not overwrite a dirty user checkout. For a resolved same-repository source
branch, prefer a temporary worktree:

```bash
git fetch github main <branch>
git worktree add <temporary-path> -b <temporary-local-branch> github/<branch>
```

Inside it, rebase onto the freshly fetched base:

```bash
git rebase github/main
```

For mechanical conflicts, read both sides, resolve, `git add`, and continue.
For semantic conflicts, stop and report exactly which files and decisions need a
human. Run focused validation and verify that the rebased diff still matches
the PR intent.

Before a history rewrite, record the exact remote branch SHA. Push only the
resolved source branch with an explicit lease:

```bash
git push github HEAD:<branch> \
  --force-with-lease=<branch>:<previous-remote-sha>
```

For an additive fix, use a normal push. Re-read `headRefOid` after every push
and restart CI evaluation.

## Diagnose Failed Checks

Use Source Actions logs only for the retained Source workflows: Internal CI
Bridge and Pages. List recent Source bridge runs:

```bash
gh run list \
  --repo NVIDIA/TensorRT-Model-Connect \
  --workflow internal-ci-bridge.yml \
  --limit 20
```

Inspect failed retained-Source logs:

```bash
gh run view <run-id> --repo NVIDIA/TensorRT-Model-Connect --log-failed
```

For a failed `TRTMC Internal CI / Automated premerge gate` status, inspect Internal CI only when
authorized. If private evidence is unavailable, report the exact head,
sanitized status, and a human blocker; do not guess or disclose private URLs.

Download Internal artifacts only when authorized and when a private job points
to structured outputs:

```bash
mkdir -p .ci_artifacts/pr<number>
gh run download <run-id> \
  --repo <internal-owner>/<internal-ci-repository> \
  --dir .ci_artifacts/pr<number>
```

Keep downloaded Internal evidence local and private. Never attach it to the
Source PR or a Source workflow. When E2E artifacts exist, inspect `result.json`
fields such as `status`, `failure_type`, stage statuses, stage messages, metric
values, and thresholds.

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
- Use `$write-git-messages` for a clear commit message that complies with the
  repository ruleset.
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

For each PR include its URL, base, source repository/branch, exact head SHA,
current status/check URLs and states, action, validation, mergeability/review
state, blockers, residual risk, and whether a branch was pushed or merged. An
unchanged pending state is normal during monitoring.

<!-- Collaborative review anchor: batch 2. -->
