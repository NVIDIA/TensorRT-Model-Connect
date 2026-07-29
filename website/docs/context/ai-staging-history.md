---
title: Retired AI Staging Workflow
---

:::info Historical design

The `ai-staging` lane is not an active merge or CI path. This page preserves
why the tooling exists and what replaced it. Use
[Contributing](../extend/contributing.md) for the current workflow.

:::

## Why it was designed

The staging design attempted to coordinate many AI-authored changes without
letting automation write directly to `main`:

```text
scoped AI pull requests
  -> ai-staging integration branch
  -> minimal staging checks
  -> combined validation
  -> promotion pull request to main
  -> human review
```

Its goals were to serialize integration, expose conflicts before promotion,
retain a human-reviewed final pull request, and attach lifecycle labels to
queued, staged, failed, and promoted work.

## What remains in the repository

Two operator tools still implement parts of that design:

- `tools/ai_agent_system.py` contains issue-label, queue, dashboard, and
  preflight helpers.
- `tools/ai_staging.py` contains staging-branch synchronization, rotation, and
  promotion helpers.

Their parsers can be inspected without changing GitHub state:

```bash
python3 tools/ai_agent_system.py --help
python3 tools/ai_staging.py --help
```

The tools can also perform GitHub mutations. Their presence is implementation
history, not authorization to create labels, update issues, move branches, or
open pull requests.

## Why the lane is inactive

The current workflow set contains no GitHub Actions workflow that validates
pull requests targeting `ai-staging` or pushes to that branch. Historical
check names such as `ai-staging-build`, `ai-staging-lint-check`, and
`ai-staging-sanity` are not current protected gates. A branch or label created
by the retained tools would therefore not recreate the intended safety
contract.

The design also added an integration layer between a change and the protected
default branch. The current repository instead validates the exact GitHub
pull-request merge snapshot against `main`.

## Current replacement

The supported lifecycle is:

```text
short-lived branch from current GitHub main
  -> focused local validation
  -> pull request targeting main
  -> authorized one-shot run-internal-ci label
  -> exact-head private premerge
  -> sanitized trtmc/premerge/required status
  -> human review and repository-ruleset merge
```

`.github/workflows/internal-ci-bridge.yml` consumes the label, authorizes the
actor, captures the current PR head SHA without executing PR code, and
dispatches private premerge for that immutable revision. The public repository
receives only the sanitized `trtmc/premerge/required` status. Private
repository details, runner information, logs, artifacts, and URLs are not part
of the Source repository contract.

Do not treat the retired staging tools as a fallback merge path. Reactivating
that design would require a new maintainer decision, current workflows,
protected checks, permissions review, failure/recovery semantics, and
end-to-end validation before any branch could serve as an integration gate.
