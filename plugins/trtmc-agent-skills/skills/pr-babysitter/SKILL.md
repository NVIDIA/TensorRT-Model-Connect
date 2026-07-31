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
- If a fix needs unavailable hardware, product judgment, or broad scope, stop
  and report the blocker.

## Current CI Contract

Read `.github/workflows/internal-ci-bridge.yml` from current `github/main`.
Source keeps only Internal CI Bridge, Legal Compliance, and Pages workflows.
Premerge and nightly execution stay in private Internal CI.

Before triggering premerge, compare the PR metadata head with the independently
resolved source branch head. This catches a GitHub PR tracking ref that stopped
advancing after a push:

```bash
REPOSITORY=NVIDIA/TensorRT-Model-Connect
PR_NUMBER=<number>
pull=$(gh api "repos/$REPOSITORY/pulls/$PR_NUMBER")
PR_HEAD_SHA=$(jq -er '.head.sha' <<<"$pull")
HEAD_REPOSITORY=$(jq -r '.head.repo.full_name // empty' <<<"$pull")
HEAD_REF=$(jq -r '.head.ref // empty' <<<"$pull")
test -n "$HEAD_REPOSITORY"
test -n "$HEAD_REF"
HEAD_REF_URI=$(jq -rn --arg value "$HEAD_REF" '$value | @uri')
BRANCH_HEAD_SHA=$(gh api \
  "repos/$HEAD_REPOSITORY/branches/$HEAD_REF_URI" \
  --jq .commit.sha)
test "$PR_HEAD_SHA" = "$BRANCH_HEAD_SHA"

gh api \
  "repos/$REPOSITORY/commits/$PR_HEAD_SHA/status" \
  --jq '[.statuses[] |
    select(.context == "trtmc/premerge/required")][0]'

gh pr edit "$PR_NUMBER" \
  --repo "$REPOSITORY" \
  --add-label run-internal-ci
```

Apply this check to both same-repository and accessible fork PRs. If the source
repository is absent, the source branch cannot be read, or the two SHAs still
differ after six 10-second retries, do not add the label:

- If the source branch SHA changes during the retries, an author is still
  pushing. Wait for it to settle and retry.
- If the source branch remains stable while the PR metadata SHA remains behind,
  treat the PR tracking ref as stale. Follow the recovery procedure in
  `tools/ci/README.md`, verify equality, and only then add the label.
- If a label event was already created for an older SHA, allow the bridge to
  consume it and report `TRIGGER_SUPERSEDED`; add a new label only after the
  current PR and branch SHAs agree.

Only an actor whose repository permission is `write`, `maintain`, or `admin`
may add the one-shot trigger.

Never use the legacy `run-ci` label. The bridge consumes `run-internal-ci`,
verifies the open PR targets `main`, and rechecks the event SHA, PR metadata
SHA, and actual source branch SHA before dispatching only `pr_number` and
`head_sha`.

Internal CI runs legal compliance and premerge tests against that exact head.
It may use the merge base only to select impacted tests; do not describe the
merge base or a synthetic merge commit as the tested revision.

The Source-visible premerge result is only the sanitized
`trtmc/premerge/required` status on that exact head: `PENDING`, then `PASS` or
`FAIL`, with a target URL under the Source commit's `/tests` tree. A successful
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
- The current head has a successful `trtmc/premerge/required` commit status.
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

Use Source Actions logs only for the retained Source workflows: Internal CI
Bridge, Legal Compliance, and Pages. List recent Source bridge runs:

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

For a failed `trtmc/premerge/required` status, inspect Internal CI only when
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
fields such as `status`,
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
