# AI Staging Branch

`origin/ai-staging` is the integration branch for AI-generated merge requests.
AI-generated work should target this branch first, not `origin/master`.

The branch has three invariants:

- `origin/ai-staging` exists at all times.
- `origin/ai-staging` is reset to current `origin/master` at each staging rotation.
- AI-generated source branches, currently `ai-task-*`, open MRs against `ai-staging`.

Human promotion to `master` stays explicit. The staging rotation snapshots the
current `ai-staging` tree to a timestamped `ai-staging-promotion-*` branch,
resets `ai-staging` to `master` for the next batch, and opens a normal MR from
the snapshot branch to `master`.

## CI Policy

Pipelines for MRs targeting `ai-staging`, and direct pushes to `ai-staging`, run
only the low-cost CPU-tagged gate:

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
`ai-staging`. Full CI still runs for normal MRs, scheduled pipelines, web
pipelines, and the final promotion MR to `master`.

New AI agent branches should be created from `origin/ai-staging`. That keeps the
source branch CI configuration aligned with the target branch. Existing AI
branches created from old `master` may need to be rebased or merged onto
`origin/ai-staging` once before their MR pipeline reflects this policy.

GitLab evaluates merge-request pipeline configuration from the source branch
SHA. The AI staging CI rules therefore must also exist on `master`, and existing
`ai-task-*` branches must be rebased or otherwise updated after the CI change
lands. Retargeting an old branch to `ai-staging` is not enough by itself.

## Operating Cycle

Use the multi-worker [Local AI Pipeline Playbook](ai-local-pipeline.md).
That mode opens five persistent Claude Code
windows in tmux and starts `/loop` with `/discovery`, `/implement`, `/merge`,
`/staging`, and `/promotion`. GitLab issues, MRs, labels, pipelines, and
branches remain the durable source of truth.

Preflight the GitLab setup with:

```bash
python3 tools/ai_agent_system.py --project yifeif/tensorrt-model-connect --target ai-staging preflight
```

Manual branch setup still uses the lower-level command below.

Run this from a clean checkout:

```bash
python3 tools/ai_staging.py full-cycle --push --retarget
```

That legacy setup command:

1. Creates `origin/ai-staging` from `origin/master` if it is missing.
2. Fetches `origin/master` and `origin/ai-staging`.
3. Creates a detached temporary worktree from `origin/ai-staging`.
4. Merges `origin/master` into that temporary worktree.
5. Pushes the updated branch when `--push` is present.
6. Retargets open `ai-task-*` MRs from `master` to `ai-staging` when
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

The local staging loop files promotion MRs. It does not merge automatically.

Run:

```bash
python3 tools/ai_staging.py --project yifeif/tensorrt-model-connect --branch ai-staging rotate-promotion --target-branch master
```

The command is idempotent:

- If `origin/ai-staging` has no tree diff from `origin/master`, it does not file
  a promotion MR.
- If there is a tree diff, it creates `origin/ai-staging-promotion-<UTC timestamp>`
  from current `origin/ai-staging`.
- It resets `origin/ai-staging` to current `origin/master` with
  `--force-with-lease`.
- It opens a human-review MR from the timestamped snapshot branch to `master`.

The promotion MR description is generated from the actual
`origin/master..origin/<snapshot>` tree diff and includes source and target
SHAs, staged commit subjects, net file changes, changed paths, diffstat, and a
review checklist.

## Promotion Babysitting

Promotion MRs run the normal full CI. The promotion babysitter may modify only
the timestamped promotion source branch to make that full CI green.

Run:

```bash
python3 tools/ai_staging.py --project yifeif/tensorrt-model-connect --branch ai-staging babysit-promotion --target-branch master --max-rebases 1
```

The command lists open promotion MRs, clean-rebases outdated promotion source
branches onto `origin/master`, refreshes the generated MR description after a
clean rebase, and prints failed full-CI jobs. The `/promotion` Claude command
uses that status to repair exactly one failed promotion source branch per cycle
and push a new full pipeline.

## Conflict Handling

If a generated MR conflicts with current `ai-staging`, the merge lane marks it
`ai:needs-rework` and leaves repair to the implementation lane.

If a promotion source branch conflicts with current `master`, the promotion
babysitter resolves only obvious mechanical conflicts. Non-trivial conflicts are
reported for human review. Promotion fixes are pushed to the timestamped
promotion source branch, never to `master`.
