---
name: submit-github-pr
description: >-
  Use when publishing an existing TensorRT-Model-Connect change as a GitHub
  pull request. Verifies authenticated repository access, branch and diff
  scope, validation evidence, commit identity, reviewer-facing text, exact
  pushed head, and the created draft PR without merging it.
---

# Submit GitHub PR

## Boundaries

- Active repository: `NVIDIA/TensorRT-Model-Connect`.
- Use the local `github` remote and target GitHub `main`.
- Never push directly to `main`.
- Publish only the change the user placed in scope.
- Open a draft PR by default; omit `--draft` only when the user explicitly asks
  for a ready-for-review PR and the evidence supports that state.
- Do not add labels, reviewers, milestones, projects, or auto-merge unless the
  user explicitly requests them.
- Do not merge from this skill. Use `$pr-babysitter` for monitoring and merge.
- Use `$write-git-messages` for commit, PR, squash, and rebase text.

## 1. Authenticate And Verify The Remote

```bash
gh auth status
gh repo view NVIDIA/TensorRT-Model-Connect \
  --json nameWithOwner,url,defaultBranchRef
git remote get-url github
git fetch github main
```

The remote URL must resolve to
`https://github.com/NVIDIA/TensorRT-Model-Connect.git` or its authorized SSH
equivalent. If `gh` cannot read the private repository, stop for safe
reauthentication. Do not inspect or replay stored credentials.

## 2. Confirm Branch And Exact Scope

```bash
BRANCH=$(git branch --show-current)
test -n "$BRANCH"
test "$BRANCH" != main
BASE_SHA=$(git rev-parse github/main)
HEAD_SHA=$(git rev-parse HEAD)
git status --short --branch
git merge-base --is-ancestor "$BASE_SHA" "$HEAD_SHA"
git diff --stat "$BASE_SHA"...HEAD
git diff --name-status "$BASE_SHA"...HEAD
git log --oneline "$BASE_SHA"..HEAD
```

Review focused diffs, staged changes, and untracked files. Do not publish:

- unrelated user changes;
- generated artifacts or secrets not intended for review;
- an empty branch;
- a branch based on the wrong repository or target;
- commit messages containing `Claude`.

If the branch is behind or diverged from current `github/main`, rebase only when
that history rewrite is within the user's request. Revalidate after any rebase.

## 3. Validate The Publishable Head

Run `git diff --check` plus checks appropriate to the changed surface. Record
the exact command and pass/fail/skip/not-run result. Do not present:

- a dry run as execution;
- compilation as model parity;
- static docs checks as target-hardware proof;
- a check on a prior SHA as evidence for the current head.

Confirm the worktree contains no unsaved in-scope change before committing or
pushing. If a new commit is needed, use `$write-git-messages` and a GitHub-safe
author email; do not change global Git identity implicitly.

Create or update an architectural record only when an existing repository
policy, template, maintainer request, or user instruction requires it. This
skill does not invent ADR policy or files merely from the size/type of a diff.

## 4. Draft Reviewer Text

Give `$write-git-messages`:

- exact base and head SHA/ref;
- changed files and behavioral purpose;
- exit criteria and non-goals;
- validation commands with outcomes;
- unrun CI, GPU, model, performance, or qualification gates;
- dependency and merge order;
- compatibility, security, migration, rollback, and review-order notes.

The PR title should match the expected squash/rebase title and follow repository
convention. Keep the body focused on one purpose.

## 5. Push The Named Branch

```bash
git push -u github "$BRANCH"
PUSHED_SHA=$(git rev-parse "github/$BRANCH")
test "$PUSHED_SHA" = "$(git rev-parse HEAD)"
```

Do not force-push unless the user authorized the history rewrite and the exact
remote branch was resolved first. When rewriting an existing branch, prefer an
explicit lease for that branch and its previously observed SHA.

## 6. Create And Verify The PR

Pass the reviewed body on stdin to avoid leaving an unnecessary local file:

```bash
gh pr create \
  --repo NVIDIA/TensorRT-Model-Connect \
  --base main \
  --head "$BRANCH" \
  --draft \
  --title "<type(scope): concise summary>" \
  --body-file -
```

Verify the immutable facts:

```bash
gh pr view <number-or-url> \
  --repo NVIDIA/TensorRT-Model-Connect \
  --json number,url,isDraft,baseRefName,headRefName,headRefOid,title,state
```

The base must be `main`, head name must match, and `headRefOid` must equal
`PUSHED_SHA`. Then inspect initial checks:

```bash
gh pr checks <number-or-url> \
  --repo NVIDIA/TensorRT-Model-Connect
```

Queued, missing, skipped, or failed checks are not green.

## 7. Observe Exact-Head Stable CI

Opening or pushing a PR targeting `main` automatically starts Stable Community
CI. It runs CPU checks before GPU admission and selected family execution, and
waits for confirmed resource cleanup before its final result. Do not start
Internal premerge as part of the ordinary contribution flow. When explicitly
authorized by a maintainer or administrator, the retained bridge can be requested
with the one-shot `run-internal-ci` label or manual dispatch for optional
diagnostics, including large PRs. Its `Optional Internal CI` result is not a
merge gate and does not waive a Stable plan rejection or the small-PR policy.

Resolve the current PR API head and inspect its aggregate status:

```bash
REPOSITORY=NVIDIA/TensorRT-Model-Connect
PR_NUMBER=<number>
pull=$(gh api "repos/$REPOSITORY/pulls/$PR_NUMBER")
PR_HEAD_SHA=$(jq -er '.head.sha' <<<"$pull")
test "$PR_HEAD_SHA" = "$PUSHED_SHA"

gh api \
  "repos/$REPOSITORY/commits/$PR_HEAD_SHA/status" \
  --jq '[.statuses[] | select(.context == "Stable Community GPU CI")][0]'
```

A missing or pending status is not a pass. A historical `Stable Community CI`
success may be CPU-only and cannot satisfy the new GPU context. Do not start a duplicate execution
when one is already pending. A CPU-only result, dispatch success, optional Dev
result or older head cannot satisfy the aggregate. Retry a failed unchanged head
only when authorized.

A successful `Stable Community GPU CI` status makes the PR CI-ready for review by an
NVIDIA employee maintainer. It does not approve, merge or request a review.
Private Nightly remains independent. Keep private logs, artifacts, packages,
runner details and URLs out of the public PR.

## 8. Report And Hand Off

Report the PR URL and number, draft state, pushed head SHA, validation evidence,
initial exact-head CI state, and any dependency or unrun gate. Hand monitoring
and any merge work to `$pr-babysitter`; this skill never merges.

<!-- Collaborative review anchor: batch 2. -->
