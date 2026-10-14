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

Read `.github/workflows/community-ci.yml` and `.github/community-ci.md` from
current `github/main`. PRs targeting `main` automatically run Stable Community
CI: CPU checks, GPU plan admission, selected family execution and confirmed
resource cleanup. `Stable Community GPU CI` is the single automated premerge gate.
Internal premerge is not required; private Nightly remains independent.

Capture the current PR head and inspect the latest aggregate status:

```bash
REPOSITORY=NVIDIA/TensorRT-Model-Connect
PR_NUMBER=<number>
pull=$(gh api "repos/$REPOSITORY/pulls/$PR_NUMBER")
PR_HEAD_SHA=$(jq -er '.head.sha' <<<"$pull")
gh api \
  "repos/$REPOSITORY/commits/$PR_HEAD_SHA/status" \
  --jq '[.statuses[] | select(.context == "Stable Community GPU CI")][0]'
```

A CPU-only result, dispatch success, optional Dev pass or stale-head result is
not a Stable pass. Legacy `Stable Community CI` successes may be CPU-only and
cannot satisfy the new context. Existing PRs without it require a fresh trusted
Stable execution; do not copy statuses or bulk-dispatch unrelated PRs.
The aggregate includes plan admission, any required GPU
coverage and confirmed cleanup; a valid no-GPU plan is reported explicitly.
Preparation errors and failures after the family entry point remain separate
classifications. Neither permits resource release to be skipped.

- A successful current-head aggregate makes a PR CI-ready for NVIDIA employee
  maintainer review. It is not human approval or authorization to merge.
- Do not infer employment from a GitHub username, email, organization association
  or repository permission. Use the project's human maintainer process.
- If the current exact head is already pending or successful, do not start a
  duplicate run. Opening or pushing the PR has already triggered Stable.
- If the head changes, treat old results as stale and observe the new automatic
  run. Retry a failed unchanged head only with rerun authorization.
- Do not apply the retired Internal CI label or use private premerge as an extra
  ordinary merge gate. Optional Dev qualification cannot overwrite Stable.
- Never start premerge after the PR has merged. Pages and private Nightly follow
  independent paths.

Raw private logs, artifacts, packages, runner details and URLs remain private.
Never copy them into public Actions, artifacts, statuses or PR comments.

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
| Stable green and branch current | Report CI-ready for NVIDIA maintainer review; merge only after human acceptance and explicit merge authority |
| Checks green but branch behind main | Rebase and push |
| Checks failed and branch behind main | Rebase first, then diagnose/fix |
| Checks failed and branch current | Diagnose/fix |
| Merge conflicts | Rebase locally if mechanical; otherwise report blocker |
| NVIDIA maintainer review or acceptance missing | Report waiting on human review |

## Merge After CI

Only merge when the user has explicitly authorized merging. Never treat
auto-merge as a way to wait for CI. A `gh pr merge --auto` invocation has
previously merged immediately while a check was still queued because the
repository ruleset did not require status checks. Auto-merge is forbidden for
this skill. This is an agent-side hard gate: if GitHub would allow the merge
before CI is green, do not merge anyway.

Before merging, verify all of the following against the latest PR head:

- `gh pr view <number> --repo NVIDIA/TensorRT-Model-Connect --json headRefOid,mergeable,mergeStateStatus,reviewDecision,statusCheckRollup`
  shows the same head SHA whose `Stable Community GPU CI` status succeeded.
- The latest `Stable Community GPU CI` status on that head is `success`, with the
  execution's final result and required cleanup complete. A missing, pending,
  failed or errored aggregate blocks merging. A successful observer or CPU stage
  cannot replace it.
- An NVIDIA employee maintainer has reviewed and accepted the change. CI success
  alone and the absence of a GitHub review rule do not satisfy this human step.
- `mergeable` is `MERGEABLE`, the repository rules permit merging, and there are
  no unresolved blocking review findings.
- The live required-status rules agree with the documented Stable gate. If the
  former Internal gate remains required, report the incomplete settings migration
  rather than changing settings or silently bypassing it.
- A new commit after validation or review requires rechecking current-head
  evidence and the review state.

`gh pr checks` is useful for diagnostics, but optional Dev, private CI and other
nonrequired checks are not additional premerge gates. Inspect their failures when
relevant without treating every visible check as required. Do not infer safety
from `mergeable=MERGEABLE`, an enabled merge button or local tests alone.

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

List Community CI runs and inspect the exact execution linked by the current
head's Stable status:

```bash
gh run list \
  --repo NVIDIA/TensorRT-Model-Connect \
  --workflow community-ci.yml \
  --limit 20
gh run view <run-id> --repo NVIDIA/TensorRT-Model-Connect --log-failed
```

Use the job summary and structured family results to separate plan admission,
preparation, family execution and cleanup failures. A failed family entry point
does not by itself establish that the PR introduced a model bug. Preserve the
original failure while checking release evidence; do not retry model failures
by replacing VMs or weaken assertions to make the run green.

Private Nightly evidence is optional and may be inspected only when authorized.
Keep it local and private. Never attach private logs, artifacts, runner details,
package coordinates or URLs to the public PR or workflow. If unavailable, report
the evidence gap without guessing. It is not an extra ordinary premerge gate.

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
