---
name: submit-github-bug-issue
description: >-
  Use when converting QA findings, black-box failures, red-team reports,
  regression evidence, or local bug notes into GitHub Issues for
  NVIDIA/TensorRT-Model-Connect. Standardizes checking issue templates,
  checking labels, de-duplicating existing issues, drafting a bug report,
  creating the issue on GitHub, applying the bug label, and verifying the
  created issue.
---

# Submit GitHub Bug Issue

## Purpose

File actionable QA findings as GitHub issues in the active repository:
`NVIDIA/TensorRT-Model-Connect`.

Use this skill for bug issue submission only. Use `$submit-github-pr` for pull
requests and `$write-git-messages` for commit, PR, squash, or rebase messages.

## Ground Rules

- Treat GitHub as the active repository:
  `https://github.com/NVIDIA/TensorRT-Model-Connect.git`.
- Use the local `github` remote when Git operations are needed.
- Do not push branches or change repository files unless the user explicitly
  asks for code or skill edits.
- Do not create duplicate issues. Search first and reuse an existing issue when
  it already captures the same root cause.
- Keep claims evidence-based. Do not invent repro results, labels, issue
  numbers, hardware, CI behavior, severity, or suspected code locations.
- Prefer English issue text for GitHub unless the user explicitly requests
  another language.
- Never read, print, transform, or replay tokens/passwords from Git credential
  helpers. Use an already authenticated `gh` session.

## Workflow

### 1. Authenticate And Resolve The Repository

```bash
gh auth status
gh repo view NVIDIA/TensorRT-Model-Connect \
  --json nameWithOwner,url,defaultBranchRef
git remote get-url github
```

If `gh` is unavailable, unauthenticated, or cannot read the repository, stop
and ask the user to authenticate outside the sandbox. Do not fall back to
extracting credentials or unauthenticated REST calls for this private
repository.

### 2. Gather Evidence

Read the source artifact before drafting. For QA reports, extract:

- finding ID and severity
- concise symptom
- exact repro command or test case
- observed exit code, output, logs, or artifact path
- expected behavior
- affected model, strategy, command, or component
- environment details
- evidence count or case IDs
- suspected source path only when the evidence supports it

If the finding is too vague to create an actionable issue, ask for the missing
critical detail instead of filing a weak report.

For a CI-generated finding, anchor the report to the exact source commit and
the public, sanitized `trtmc/premerge/required` status. Include a minimal public
reproduction whenever possible. Internal CI logs, artifacts, package details,
runner details, and internal URLs are private evidence: do not copy them into a
Source issue. When authorized internal evidence is needed to establish the root
cause, summarize only the public, actionable conclusion and preserve the
Source-versus-Internal publication boundary.

### 3. Check Issue Templates

Check local templates:

```bash
find .github -maxdepth 3 -type f \
  \( -path ".github/ISSUE_TEMPLATE/*" -o -name "ISSUE_TEMPLATE.md" \) \
  -print
```

When a local template exists, follow it. When no local template exists, use the
inline fallback bug format in the draft step below.

### 4. Check Labels

Verify that the requested label exists before creating the issue:

```bash
gh label list --repo NVIDIA/TensorRT-Model-Connect --limit 100
```

For bug reports, use `bug` when it exists. If it does not exist, create the
issue without inventing a replacement label and report that the label was
missing.

### 5. Search For Duplicates

Search open issues using concrete terms from the finding: CLI flags, model
names, failing command, error string, source path, and root-cause wording.

```bash
gh issue list \
  --repo NVIDIA/TensorRT-Model-Connect \
  --state open \
  --search "<symptom terms>"
```

If a strong duplicate exists, do not file a new issue. Report the existing issue
URL and explain the match. If the match is weak, file the new issue and mention
related issues in the body only when relevant.

### 6. Draft The Issue

Use a concise title that starts with `Bug:` and names the failing behavior, not
only the component or severity.

Default bug format when the repository has no template:

````markdown
## Summary

<One short paragraph describing the bug and why it is wrong.>

## Environment

- Binary/version: <value if known>
- TensorRT/CUDA/platform/GPU: <values if known>
- Source run: <QA run, CI run, machine, or date if known>

## Affected Scope

<Commands, strategies, models, components, or cases affected. Use bullets or a
small table when useful.>

## Steps To Reproduce

```bash
<minimal command sequence>
```

## Actual Behavior

<Observed output, exit code, logs, generated artifacts, or failure mode.>

## Expected Behavior

<The contract the product should satisfy.>

## Impact

<Why this matters for users, CI, correctness, reliability, or debuggability.>

## Suspected Location

<Source path or subsystem, only when evidence supports it.>
````

Omit sections that are truly inapplicable, but keep `Summary`, `Steps To
Reproduce`, `Actual Behavior`, `Expected Behavior`, and `Impact` for normal QA
bugs.

### 7. Create The Issue

Create only after the user has authorized issue creation. Avoid writing a
temporary body file unless the user asked for a local artifact; `gh` accepts
stdin:

```bash
gh issue create \
  --repo NVIDIA/TensorRT-Model-Connect \
  --title "Bug: <concise symptom>" \
  --body-file - \
  --label bug
```

Omit `--label bug` when the label check proved it is absent. Do not create a
label unless the user explicitly requested label administration.

### 8. Verify And Report

After creation, verify the issue state, URL, title, and labels:

```bash
gh issue view <number> \
  --repo NVIDIA/TensorRT-Model-Connect \
  --json number,title,state,labels,url
```

Report back with:

- issue URL and number
- label status, especially whether `bug` was applied
- template status
- duplicate-search result
- exact evidence or repro that was not independently verified
- any limitation, such as missing label or incomplete environment details

## Multiple Findings

File one issue per root cause. Do not bundle unrelated failures into a single
issue just because they came from the same QA report. If several cases share the
same root cause, include them as affected scope/evidence in one issue.
