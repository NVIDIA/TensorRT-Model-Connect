# AI Agent System

This system turns low-risk AI reliability work into an autonomous integration flow while keeping `master` human-controlled.

## Goal

Human review should happen at one aggregate boundary:

```text
task issue -> AI implementation MR -> sanity CI -> ai-staging -> snapshot branch -> promotion MR -> full CI -> master
```

`master` is ground truth. `ai-staging` is an integration branch for AI-generated changes. Individual AI MRs are disposable.

## Agents

| Agent | Skill | Owns | Must not do |
|---|---|---|---|
| Task discovery | `ai-task-discovery` | Create atomic, verifiable task issues | Implement code |
| Implementation | `/implement` + `ai-task-implementer` | Claim one issue, create a per-issue worktree, delegate implementation to a subagent, open one MR targeting `ai-staging`, and repair failed generated MRs marked `ai:needs-rework` | Merge |
| Staging autopilot | `gitlab-ai-staging-autopilot` | Merge green AI MRs into `ai-staging` one at a time | Diagnose failed CI |
| Staging babysitter | `ai-staging-babysitter` | Snapshot `ai-staging`, reset `ai-staging` to `master`, and open the human-review promotion MR | Push or merge to `master` |
| Promotion babysitter | `ai-promotion-babysitter` | Keep promotion MR full CI green by modifying only timestamped promotion source branches | Merge, approve, push `master`, or modify `ai-staging` |

## Shared State

GitLab issues and merge requests are the queue. Do not use a repository file as the queue.

Standard labels are managed by:

```bash
python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect ensure-labels
```

Core labels:

```text
ai:task
ai:ready
ai:claimed
ai:implementing
ai-generated
ai:staging-mr
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
  -> ai-task-* branch and MR targeting ai-staging with ai-generated + ai:staging-mr labels
  -> MR head pipeline passes, or implementation rework repairs the existing MR
  -> autopilot rebases and merges MR into ai-staging
  -> staging loop snapshots ai-staging to ai-staging-promotion-<timestamp>
  -> staging loop resets ai-staging to master for the next batch
  -> promotion MR from snapshot branch to master
  -> promotion babysitter keeps full CI green
  -> human review and merge
```

Failure path:

```text
unclear task -> ai:needs-human
invalid or stale task -> ai:dropped
MR sanity CI failure with no active fix pipeline -> ai:needs-rework
merge/rebase conflict needing implementation and no active fix pipeline -> ai:needs-rework
promotion MR full CI failure -> promotion babysitter fixes the promotion source branch, or reports human blocker
```

## Invariants

- `master` is never pushed by agents.
- Normal MRs targeting `master` keep existing CI behavior.
- AI implementation MRs target `ai-staging`.
- AI implementation MRs carry `ai-generated` and `ai:staging-mr` labels for filtering.
- AI implementation MRs run only sanity CI.
- Promotion MRs from timestamped staging snapshot branches to `master` run full CI.
- Promotion babysitting may push only `ai-staging-promotion-*` source branches.
- GitLab native auto-merge is not used for AI MRs unless the project globally requires successful pipelines.

## Operator Commands

See [Local AI Pipeline Playbook](ai-local-pipeline.md) for the human-run startup sequence. It opens
five persistent Claude Code windows in tmux and starts `/loop` with the short
commands `/discovery`, `/implement`, `/merge`, `/staging`, and `/promotion`.

Preflight:

```bash
python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect --target ai-staging preflight
```

Dashboard:

```bash
python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect --target ai-staging dashboard
```

Mark an MR and its linked issue for implementation rework:

```bash
python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect mark-rework \
  --mr 123 \
  --skip-if-active-pipeline \
  --reason "rebase conflict against ai-staging"
```

Create a dry-run task:

```bash
python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect --dry-run create-task \
  --title "tests: cover AI task contract validation" \
  --scope "tests/tools/test_ai_agent_system.py and tools/ai_agent_system.py only" \
  --change "Add coverage that validate-task rejects issue bodies missing required sections so implementation issues stay actionable." \
  --acceptance "A task body missing Verification or Acceptance Criteria fails validation and names the missing heading." \
  --verification "python3 -m pytest tests/tools/test_ai_agent_system.py -q" \
  --non-goal "Do not change GitLab API behavior or task labels."
```

Run one autopilot merge action:

```bash
python3 skills/gitlab-ai-staging-autopilot/scripts/ai_staging_autopilot.py \
  --project yifeif/tensorrt-model-connect \
  --target ai-staging \
  --source-prefix ai-task- \
  --once
```

Promotion MR:

```bash
python3 tools/ai_staging.py --project yifeif/tensorrt-model-connect --branch ai-staging rotate-promotion --target-branch master
```

Promotion MRs are generated from the actual `origin/master..origin/<snapshot>` tree diff. Their descriptions include branch SHAs, the ai-staging reset SHA, staged commit subjects, net file changes, changed paths, diffstat, and a review checklist. Implementation agents must write complete individual MR descriptions with the task link, scope, concrete changes, verification, risk, rollback, and non-goals so the aggregate promotion remains reviewable.

Promotion babysitting:

```bash
python3 tools/ai_staging.py --project yifeif/tensorrt-model-connect --branch ai-staging babysit-promotion --target-branch master --max-rebases 1
```

The promotion babysitter may clean-rebase and modify timestamped promotion source branches until the full MR pipeline is green. It never merges or approves the promotion MR.
