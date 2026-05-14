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
gh pr create \
  --repo NVIDIA/TensorRT-Model-Connect \
  --base main \
  --head <branch-name> \
  --title "<type(scope): concise summary>" \
  --body-file <pr-body.md>
```

If `gh` is not authenticated but a GitHub token file is available, prefer a
temporary environment variable and do not print the token:

```bash
GH_TOKEN=$(tr -d '\r\n' < /workspace/users/yizhuoz/github.txt) gh pr view --repo NVIDIA/TensorRT-Model-Connect
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
| New family plugin | New family module under `tensorrt_model_connect/tensorrt_model_connect/families/` |
| New pipeline class | New `.cpp` or `.h` under `src/runtime/pipelines/` or equivalent runtime path |
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

Create `website/docs/context/adr/${NEXT_NUM}-<slug>.md`:

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

After drafting, write the body to a temporary file and pass it with
`gh pr create --body-file <pr-body.md>`.

## After Creation

```bash
gh pr view --repo NVIDIA/TensorRT-Model-Connect --web
gh pr checks --repo NVIDIA/TensorRT-Model-Connect <pr-number>
```

Wait for GitHub CI before merging. Merge only when asked or when the task
explicitly includes merge authority; follow repository rules for squash or
rebase merge.
