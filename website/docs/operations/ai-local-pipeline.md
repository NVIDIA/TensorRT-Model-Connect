# Local AI Pipeline Playbook

This is the human-operated startup path for the AI staging pipeline. The goal is
simple: open five persistent Claude Code windows in one tmux session, then start
one `/loop` command in each window.

GitLab remains the durable queue for issues, MRs, labels, pipelines, and branch
state. The local `.ai-pipeline/` directory is only used for implementation and
promotion-repair worktrees.

## Start

From the repo root:

```bash
git fetch origin master ai-staging
export REPO_ROOT="$PWD"
python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect --target ai-staging preflight
```

Open five Claude Code windows in tmux:

```bash
tmux new-session -d -s ai-pipeline -n discovery \
  "cd '$REPO_ROOT' && claude --permission-mode auto --worktree ai-discovery"

tmux new-window -t ai-pipeline -n implement \
  "cd '$REPO_ROOT' && claude --permission-mode auto --worktree ai-implement"

tmux new-window -t ai-pipeline -n merge \
  "cd '$REPO_ROOT' && claude --permission-mode auto --worktree ai-merge"

tmux new-window -t ai-pipeline -n staging \
  "cd '$REPO_ROOT' && claude --permission-mode auto --worktree ai-staging-worker"

tmux new-window -t ai-pipeline -n promotion \
  "cd '$REPO_ROOT' && claude --permission-mode auto --worktree ai-promotion"

tmux attach -t ai-pipeline
```

Do not use `--dangerously-skip-permissions`.

## Start The Loops

In the `discovery` Claude window:

```text
/loop 30m /discovery
```

In the `implement` Claude window:

```text
/loop 10m /implement
```

In the `merge` Claude window:

```text
/loop 5m /merge
```

In the `staging` Claude window:

```text
/loop 240m /staging
```

In the `promotion` Claude window:

```text
/loop 20m /promotion
```

That is the whole steady-state startup.

## What Each Window Does

```text
discovery
  creates small ai:ready GitLab issues

implement
  claims one ai:ready or ai:needs-rework issue from GitLab
  sends failed/canceled generated MRs back to ai:needs-rework
  creates an isolated per-issue worktree
  starts a subagent to implement and validate the task
  submits one ai-task-* MR targeting ai-staging, or repairs the existing MR for rework

merge
  merges green approved AI MRs into ai-staging
  sends rebase conflicts back to ai:needs-rework

staging
  snapshots ai-staging to a timestamped promotion branch
  resets ai-staging to master for future AI MRs
  opens a human-review MR from the snapshot branch to master

promotion
  watches ai-staging-promotion-* MRs targeting master
  keeps promotion source branches rebased onto master
  fixes failed full-CI issues on the promotion source branch
  leaves final review and merge to the human
```

## Monitor

From any shell in the repo:

```bash
python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect --target ai-staging dashboard
```

## Stop

Stop individual Claude sessions with `/quit`, or stop the full tmux session:

```bash
tmux kill-session -t ai-pipeline
```
