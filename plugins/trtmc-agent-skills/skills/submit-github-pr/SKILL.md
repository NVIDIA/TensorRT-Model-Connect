---
name: submit-github-pr
description: >-
  Use when pushing a branch to GitHub and opening a pull request for
  TensorRT-Model-Connect. Enforces the repo's GitHub remote/main flow, prepares
  reviewer-ready PR text, and adds ADRs for architectural changes when needed.
---

# Submit GitHub PR

## Ground Rules

- Treat `https://github.com/NVIDIA/TensorRT-Model-Connect.git` and the local
  `github` remote as the active repository.
- Target `main`; never push directly to `main`.
- Push feature branches to `github`.
- Use `gh` for GitHub PR operations.
- Use `$write-git-messages` to draft the PR title/body and any commit or squash
  message.
- This skill opens PRs only. Do not merge from this skill. If the user asks to
  babysit or merge the PR, switch to `$pr-babysitter` and follow its CI gate.

## Quick Flow

```bash
git fetch github main
git status --short --branch
git diff --check
git diff --stat github/main...HEAD
git push -u github HEAD
```

Create the PR:

```bash
PR_HEAD_BRANCH=$(git branch --show-current)
PR_TITLE="docs: describe the change"
PR_BODY_FILE=/tmp/trtmc-pr-body.md
test -n "$PR_HEAD_BRANCH"
test -s "$PR_BODY_FILE"
gh pr create \
  --repo NVIDIA/TensorRT-Model-Connect \
  --base main \
  --head "$PR_HEAD_BRANCH" \
  --title "$PR_TITLE" \
  --body-file "$PR_BODY_FILE"
```

If `gh` is not authenticated but a GitHub token is available, prefer a temporary
environment variable sourced from the user's secret manager, and do not print
the token:

```bash
GH_TOKEN="${GITHUB_TOKEN:?Set GITHUB_TOKEN from your secret manager first}" \
  gh pr view --repo NVIDIA/TensorRT-Model-Connect
```

## ADR Check

Before creating the PR, decide whether the diff warrants an Architecture
Decision Record. Analyze the diff against `github/main`:

```bash
git diff --name-only github/main...HEAD
```

Create an ADR when the diff introduces or substantially changes any of these:

| Signal | Detection |
|--------|-----------|
| New runtime strategy | New plugin/strategy registration or runtime pipeline path |
| New family plugin | New family module under `python/tensorrt_model_connect/families/` |
| New pipeline class | New `.cpp` or `.h` under an owning model runtime directory such as `src/runtime/models/qwen/`, or under a shared runtime domain in `src/runtime/domains/` |
| Config schema change | New persisted config field or parser behavior |
| New E2E task strategy | New harness runner or comparator family |
| New comparator/reference | New comparator/reference mechanism used by tests |
| Architectural refactor | Broad runtime source moves, registry redesign, or cross-module contract change |

Do not create ADRs for routine bug fixes, tests without architectural impact,
dependency bumps, docs-only changes, or new manifests for existing families.

## ADR Creation

Determine the next ADR number:

```bash
LAST_NUM=$(ls website/docs/context/adr/[0-9]*.md 2>/dev/null | sort -V | tail -1 | grep -oE '[0-9]{4}' || true)
NEXT_NUM=$(printf "%04d" $((10#${LAST_NUM:-0} + 1)))
```

Choose a concrete kebab-case slug and confirm the resulting path:

```bash
ADR_SLUG=concise-decision-title
ADR_PATH="website/docs/context/adr/${NEXT_NUM}-${ADR_SLUG}.md"
printf '%s\n' "$ADR_PATH"
```

Create the file at `$ADR_PATH` with this template:

```markdown
---
number: <NNNN>
title: <concise decision title>
status: Proposed
date: <YYYY-MM-DD>
source_commits: [<sha1>, <sha2>]
---

## Context

<Why the decision was needed.>

## Decision

<What the code now does and the main files/patterns involved.>

## Considered Alternatives

<Alternatives from the discussion, or "Not captured; review and add if known.">

## Consequences

<Capabilities, constraints, maintenance cost, compatibility, and rollback notes.>
```

Update `website/docs/context/adr/README.md` with the new row. Commit the ADR in
the same branch before creating the PR.

## PR Title And Body

Use `$write-git-messages` to draft the PR title and body. This skill owns the
reviewer-facing message style; do not maintain a second PR body template here.

Provide these repo-specific facts to `$write-git-messages`:

- Base branch and head branch.
- Whether an ADR was created or intentionally skipped.
- Exact validation commands and pass/fail/not-run results.
- GitHub-specific notes such as CI state, labels, merge constraints, review
  order, or source-branch deletion policy.
- Any risk, rollback, compatibility, security, or dependency concerns found
  while preparing the PR.

After drafting, write the body to `$PR_BODY_FILE` and pass it with
`gh pr create --body-file "$PR_BODY_FILE"`.

## After Creation

```bash
gh pr view --repo NVIDIA/TensorRT-Model-Connect --web
PR_NUMBER=$(gh pr view --repo NVIDIA/TensorRT-Model-Connect --json number --jq .number)
test -n "$PR_NUMBER"
gh pr checks --repo NVIDIA/TensorRT-Model-Connect "$PR_NUMBER"
```

## Start Exact-Head Premerge

Creating or pushing the PR does not start premerge. After verifying that
`headRefOid` equals the pushed SHA, and when CI execution is authorized, have an
actor with `write`, `maintain`, or `admin` permission add
`run-internal-ci`.

Check the current head's `trtmc/premerge/required` status first. Do not add the
label when that head is already pending or has passed. Reapply it once only
after an intentional head change, and never after the PR has merged. Hand
monitoring and merge gating to `$pr-babysitter`.

Do not merge from this skill, even if the task includes merge authority. Use
`$pr-babysitter` for monitoring and merging. It has the required hard CI gate
and explicitly forbids `gh pr merge --auto` and any merge while checks are
queued, pending, running, missing, skipped unexpectedly, or failing.
