# AI Staging Branch

`github/ai-staging` is the integration branch for AI-generated pull requests.
AI-generated work should target this branch first, not `github/main`.

The branch has three invariants:

- `github/ai-staging` exists at all times.
- `github/ai-staging` is reset to current `github/main` at each staging rotation.
- AI-generated source branches, currently `ai-task-*`, open PRs against `ai-staging`.

Human promotion to `main` stays explicit. The staging rotation snapshots the
current `ai-staging` tree to a timestamped `ai-staging-promotion-*` branch,
resets `ai-staging` to `main` for the next batch, and opens a normal PR from
the snapshot branch to `main`.

## CI Policy

GitHub Actions checks for PRs targeting `ai-staging`, and direct pushes to
`ai-staging`, run only the low-cost CPU-tagged gate:

- `ai-staging-build`
- `ai-staging-lint-check`
- `ai-staging-sanity`

The sanity path hides GPUs with `CUDA_VISIBLE_DEVICES=""`, builds on the `cpu`
runner, runs the CLI help check, runs CTest with CUDA/TensorRT execution tests
excluded, and runs pytest with `gpu`, `trt`, and `e2e` tests deselected.
CPU-tagged jobs set the local development image pull policy to
`if-not-present` so a runner with `trtmc-dev-gb300:latest` already loaded does
not fail while trying to pull that local image from a registry.
Expensive impact analysis, coverage, graph, and E2E jobs are skipped for
`ai-staging`. Full CI still runs for normal PRs, scheduled GitHub Actions
workflows, manual workflow dispatches, and the final promotion PR to `main`.

New AI agent branches should be created from `github/ai-staging`. That keeps the
source branch workflow configuration aligned with the target branch. Existing
AI branches created from old `main` may need to be rebased or merged onto
`github/ai-staging` once before their PR checks reflect this policy.

GitHub evaluates pull-request workflow configuration from the source branch
SHA. The AI staging CI rules therefore must also exist on `main`, and existing
`ai-task-*` branches must be rebased or otherwise updated after the CI change
lands. Retargeting an old branch to `ai-staging` is not enough by itself.

## Operating Cycle

Use the multi-worker [Local AI Workflow Playbook](ai-local-pipeline.md).
That mode opens persistent agent CLI windows in tmux and starts the discovery,
implementation, merge, staging, and promotion loops. GitHub issues, PRs,
labels, checks, and branches remain the durable source of truth.

Preflight the GitHub setup with:

```bash
python3 tools/ai_agent_system.py --project NVIDIA/TensorRT-Model-Connect --target ai-staging preflight
```

Manual branch setup still uses the lower-level command below.

Run this from a clean checkout:

```bash
python3 tools/ai_staging.py full-cycle --push --retarget
```

That setup command:

1. Creates `github/ai-staging` from `github/main` if it is missing.
2. Fetches `github/main` and `github/ai-staging`.
3. Creates a detached temporary worktree from `github/ai-staging`.
4. Merges `github/main` into that temporary worktree.
5. Pushes the updated branch when `--push` is present.
6. Retargets open `ai-task-*` PRs from `main` to `ai-staging` when
   `--retarget` is present.

For a read-only preview:

```bash
python3 tools/ai_staging.py --dry-run full-cycle --push --retarget
```

To inspect the queue without changing anything:

```bash
python3 tools/ai_staging.py list
```

To retarget only after reviewing the list:

```bash
python3 tools/ai_staging.py retarget
```

## Promotion Rotation

The local staging loop files promotion PRs. It does not merge automatically.

Run:

```bash
python3 tools/ai_staging.py --project NVIDIA/TensorRT-Model-Connect --branch ai-staging rotate-promotion --target-branch main
```

The command is idempotent:

- If `github/ai-staging` has no tree diff from `github/main`, it does not file
  a promotion PR.
- If there is a tree diff, it creates `github/ai-staging-promotion-<UTC timestamp>`
  from current `github/ai-staging`.
- It resets `github/ai-staging` to current `github/main` with
  `--force-with-lease`.
- It opens a human-review PR from the timestamped snapshot branch to `main`.

The promotion PR description is generated from the actual
`github/main..github/<snapshot>` tree diff and includes source and target SHAs,
staged commit subjects, net file changes, changed paths, diffstat, and a review
checklist.

## Promotion Repair

Promotion PRs run the normal full CI. The promotion repair lane may modify only
the timestamped promotion source branch to make that full CI green.

Run:

```bash
python3 tools/ai_staging.py --project NVIDIA/TensorRT-Model-Connect --branch ai-staging babysit-promotion --target-branch main --max-rebases 1
```

The command lists open promotion PRs, clean-rebases outdated promotion source
branches onto `github/main`, refreshes the generated PR description after a
clean rebase, and prints failed full-check runs. The promotion agent uses that
status to repair exactly one failed promotion source branch per cycle and push
a new full run.

## Conflict Handling

If a generated PR conflicts with current `ai-staging`, the merge lane marks it
`ai:needs-rework` and leaves repair to the implementation lane.

If a promotion source branch conflicts with current `main`, the promotion
repair lane resolves only obvious mechanical conflicts. Non-trivial conflicts
are reported for human review. Promotion fixes are pushed to the timestamped
promotion source branch, never to `main`.
