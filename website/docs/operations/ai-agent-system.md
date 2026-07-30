# AI Agent Operations

AI-authored changes follow the same repository rules as human-authored
changes. `main` is the only current premerge CI target, and no agent may weaken
tests, bypass review, or push directly to `main`.

## Current lifecycle

```text
scoped issue or request
  -> short-lived branch from the current GitHub main
  -> implementation and local verification
  -> pull request targeting main
  -> one-shot run-internal-ci label
  -> private exact-head CI
  -> sanitized trtmc/premerge/required status
  -> human review and repository-ruleset merge
```

The source of truth is:

- `AGENTS.md` for repository and branch policy
- `.github/workflows/internal-ci-bridge.yml` for trusted exact-head dispatch
- `plugins/trtmc-agent-skills/skills/pr-babysitter/SKILL.md` for trigger and merge-gate behavior
- `tools/test_impact.py` for affected-model selection

Private Internal CI owns premerge, isolated model proof, and scheduled broad
qualification. Do not copy its raw logs, artifacts, runner details, package
coordinates, or private URLs into the Source PR. Merging a passing PR does not
trigger the same premerge suite again.

## Queue helper

`tools/ai_agent_system.py` retains issue-label and queue-management helpers.
Inspect its exact interface before use:

```bash
DOC_REMOTE="github"
if ! git remote get-url "$DOC_REMOTE" >/dev/null 2>&1; then
  DOC_REMOTE="origin"
fi

python3 tools/ai_agent_system.py --help
python3 tools/ai_agent_system.py --remote "$DOC_REMOTE" --dry-run dashboard
python3 tools/ai_agent_system.py --remote "$DOC_REMOTE" --dry-run preflight
```

The tool defaults to a remote named `github`; the fallback supports canonical
clones where that same repository is named `origin`. Verify that the resolved
URL is `NVIDIA/TensorRT-Model-Connect` as shown in the
[local workflow playbook](ai-local-pipeline.md) before any write. Commands
without `--dry-run` may create labels or mutate issues, so they require
explicit operator intent. The current `preflight` returns nonzero because the
remote `ai-staging` branch and its label set are absent; that result confirms
the lane is inactive rather than indicating a supported setup.

## Inactive staging design

The labels and branch-management code for an `ai-staging` integration lane
remain in `tools/ai_agent_system.py` and `tools/ai_staging.py`. They are
retained implementation, not evidence that the lane is operational. The
current Actions workflows contain no protected minimal-CI gate for
`ai-staging`; see [AI Staging Branch](ai-staging.md).

## Evidence required from an agent

Every pull request should state:

- the exact scope and non-goals
- the tested commit
- commands that were actually executed
- model, hardware, and artifact details for GPU/model claims
- remaining risks or unverified paths

Compilation, CI success, parity, performance, and release qualification are
separate evidence tiers. A lower tier must not be presented as proof of a
higher one.
