# Local AI Workflow Playbook

This is the human-operated startup path for the AI staging workflow. The goal is
to open persistent agent CLI windows in one tmux session, then start one loop
prompt in each window.

GitHub remains the durable queue for issues, PRs, labels, checks, and branch
state. The local `.ai-workflow/` directory is only used for implementation and
promotion-repair worktrees.

## Agent CLI

Use any agent CLI that can run in the repository checkout and accept a prompt.
Codex is the default example because it understands the repo-local
`AGENTS.md` and installed skills:

```bash
export TRTMC_AGENT_BIN="${TRTMC_AGENT_BIN:-codex}"
export TRTMC_AGENT_ARGS="${TRTMC_AGENT_ARGS:-exec -s danger-full-access -a never -C {workspace} {prompt}}"
```

For a different CLI, override both variables before opening the tmux session.
For example:

```bash
export TRTMC_AGENT_BIN=claude
export TRTMC_AGENT_ARGS='--print -p {prompt}'
```

The placeholders are:

- `{workspace}`: the repository checkout for that window.
- `{prompt}`: the loop command or task prompt to run.

## Start

From the repo root:

```bash
git fetch github main ai-staging
export REPO_ROOT="$PWD"
python3 tools/ai_agent_system.py --project NVIDIA/TensorRT-Model-Connect --target ai-staging preflight
```

Open five agent windows in tmux. The commands below show Codex explicitly; if
you use another CLI, replace the command after `&&` with your agent invocation.

```bash
tmux new-session -d -s ai-workflow -n discovery \
  "cd '$REPO_ROOT' && codex exec -s danger-full-access -a never -C '$REPO_ROOT' '/loop 30m /discovery'"

tmux new-window -t ai-workflow -n implement \
  "cd '$REPO_ROOT' && codex exec -s danger-full-access -a never -C '$REPO_ROOT' '/loop 10m /implement'"

tmux new-window -t ai-workflow -n merge \
  "cd '$REPO_ROOT' && codex exec -s danger-full-access -a never -C '$REPO_ROOT' '/loop 5m /merge'"

tmux new-window -t ai-workflow -n staging \
  "cd '$REPO_ROOT' && codex exec -s danger-full-access -a never -C '$REPO_ROOT' '/loop 240m /staging'"

tmux new-window -t ai-workflow -n promotion \
  "cd '$REPO_ROOT' && codex exec -s danger-full-access -a never -C '$REPO_ROOT' '/loop 20m /promotion'"

tmux attach -t ai-workflow
```

Do not skip repository permission checks unless the repository instructions
explicitly allow it for the environment you are using.

## What Each Window Does

```text
discovery
  creates small ai:ready GitHub issues

implement
  claims one ai:ready or ai:needs-rework issue from GitHub
  sends failed/canceled generated PRs back to ai:needs-rework
  creates an isolated per-issue worktree
  starts an implementation agent
  submits one ai-task-* PR targeting ai-staging, or repairs the existing PR for rework

merge
  merges green approved AI PRs into ai-staging
  sends rebase conflicts back to ai:needs-rework

staging
  snapshots ai-staging to a timestamped promotion branch
  resets ai-staging to main for future AI PRs
  opens a human-review PR from the snapshot branch to main

promotion
  watches ai-staging-promotion-* PRs targeting main
  keeps promotion source branches rebased onto main
  fixes failed full-check issues on the promotion source branch
  leaves final review and merge to the human
```

## Monitor

From any shell in the repo:

```bash
python3 tools/ai_agent_system.py --project NVIDIA/TensorRT-Model-Connect --target ai-staging dashboard
```

## Stop

Stop individual agent sessions with the CLI's normal exit command, or stop the
full tmux session:

```bash
tmux kill-session -t ai-workflow
```
