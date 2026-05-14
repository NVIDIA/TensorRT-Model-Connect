# AI Agent System

This system turns low-risk AI reliability work into an autonomous integration
flow while keeping `main` human-controlled.

## Goal

Human review should happen at one aggregate boundary:

```text
task issue -> AI implementation PR -> sanity checks -> ai-staging -> snapshot branch -> promotion PR -> full checks -> main
```

`main` is ground truth. `ai-staging` is an integration branch for AI-generated
changes. Individual AI PRs are disposable and must stay easy to rework or close.

## Agents

| Lane | Owns | Must not do |
|---|---|---|
| Task discovery | Create atomic, verifiable GitHub issues labeled `ai:task` and `ai:ready` | Implement code |
| Implementation | Claim one issue, create a per-issue worktree, use repo-local skills where applicable, open one PR targeting `ai-staging`, and repair generated PRs marked `ai:needs-rework` | Merge |
| Staging merge | Merge green AI PRs into `ai-staging` one at a time | Diagnose unrelated failed checks |
| Staging rotation | Snapshot `ai-staging`, reset `ai-staging` to `main`, and open the human-review promotion PR | Push or merge to `main` |
| Promotion repair | Keep promotion PR full checks green by modifying only timestamped promotion source branches | Merge, approve, push `main`, or modify `ai-staging` |

The lanes can be run by Codex or any other agent CLI. The operator docs use
Codex as the default example and describe how to substitute another CLI.

## Shared State

GitHub issues and pull requests are the queue. Do not use a repository file as
the queue.

Standard labels are managed by:

```bash
python3 tools/ai_agent_system.py --project NVIDIA/TensorRT-Model-Connect ensure-labels
```

Core labels:

```text
ai:task
ai:ready
ai:claimed
ai:implementing
ai-generated
ai:staging-pr
ai:sanity-pending
ai:sanity-failed
ai:sanity-green
ai:autopilot
ai:staged
ai:staging-failed
ai:needs-rework
ai:dropped
ai:needs-human
ai:promotion
```

## State Machine

```text
ai:task + ai:ready
  -> implementation agent claims issue
  -> ai-task-* branch and PR targeting ai-staging with ai-generated + ai:staging-pr labels
  -> PR head checks pass, or implementation rework repairs the existing PR
  -> staging merge lane lands PR into ai-staging
  -> staging rotation snapshots ai-staging to ai-staging-promotion-<timestamp>
  -> staging rotation resets ai-staging to main for the next batch
  -> promotion PR from snapshot branch to main
  -> promotion repair lane keeps full checks green
  -> human review and merge
```

Failure path:

```text
unclear task -> ai:needs-human
invalid or stale task -> ai:dropped
PR sanity check failure with no active fix run -> ai:needs-rework
merge/rebase conflict needing implementation and no active fix run -> ai:needs-rework
promotion PR full-check failure -> promotion repair fixes the promotion source branch, or reports human blocker
```

## Invariants

- `main` is never pushed by agents.
- Normal PRs targeting `main` keep existing CI behavior.
- AI implementation PRs target `ai-staging`.
- AI implementation PRs carry `ai-generated` and `ai:staging-pr` labels for filtering.
- AI implementation PRs run only sanity checks.
- Promotion PRs from timestamped staging snapshot branches to `main` run full checks.
- Promotion repair may push only `ai-staging-promotion-*` source branches.
- GitHub native auto-merge is not used for AI PRs unless the repository rules require it.

## Operator Commands

See [Local AI Workflow Playbook](ai-local-pipeline.md) for the human-run
startup sequence. It opens persistent agent CLI windows in tmux and starts loop
prompts for discovery, implementation, merge, staging, and promotion lanes.

Preflight:

```bash
python3 tools/ai_agent_system.py --project NVIDIA/TensorRT-Model-Connect --target ai-staging preflight
```

Dashboard:

```bash
python3 tools/ai_agent_system.py --project NVIDIA/TensorRT-Model-Connect --target ai-staging dashboard
```

Mark a PR and its linked issue for implementation rework:

```bash
python3 tools/ai_agent_system.py --project NVIDIA/TensorRT-Model-Connect mark-rework \
  --pr 123 \
  --skip-if-active-checks \
  --reason "rebase conflict against ai-staging"
```

Create a dry-run task:

```bash
python3 tools/ai_agent_system.py --project NVIDIA/TensorRT-Model-Connect --dry-run create-task \
  --title "tests: cover AI task contract validation" \
  --scope "tests/tools/test_ai_agent_system.py and tools/ai_agent_system.py only" \
  --change "Add coverage that validate-task rejects issue bodies missing required sections so implementation issues stay actionable." \
  --acceptance "A task body missing Verification or Acceptance Criteria fails validation and names the missing heading." \
  --verification "python3 -m pytest tests/tools/test_ai_agent_system.py -q" \
  --non-goal "Do not change GitHub API behavior or task labels."
```

Promotion PR:

```bash
python3 tools/ai_staging.py --project NVIDIA/TensorRT-Model-Connect --branch ai-staging rotate-promotion --target-branch main
```

Promotion PRs are generated from the actual `github/main..github/<snapshot>`
tree diff. Their descriptions include branch SHAs, the ai-staging reset SHA,
staged commit subjects, net file changes, changed paths, diffstat, and a review
checklist. Implementation agents must write complete individual PR descriptions
with the task link, scope, concrete changes, verification, risk, rollback, and
non-goals so the aggregate promotion remains reviewable.

Promotion repair:

```bash
python3 tools/ai_staging.py --project NVIDIA/TensorRT-Model-Connect --branch ai-staging babysit-promotion --target-branch main --max-rebases 1
```

The promotion repair lane may clean-rebase and modify timestamped promotion
source branches until the full PR checks are green. It never merges or approves
the promotion PR.
