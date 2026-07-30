# AI Staging Branch

:::warning Historical workflow

The repository still contains `tools/ai_staging.py` and
`tools/ai_agent_system.py`, but the current GitHub Actions configuration does
not run CI for pull requests targeting `ai-staging` or pushes to that branch.
The former `ai-staging-build`, `ai-staging-lint-check`, and
`ai-staging-sanity` checks do not exist. Do not use this workflow as a merge
gate unless maintainers add and protect an `ai-staging` CI workflow again.

:::

## Current supported PR flow

The supported premerge path uses `.github/workflows/internal-ci-bridge.yml`:

1. Open a pull request targeting `main`.
2. Have an actor with `write`, `maintain`, or `admin` permission apply the
   one-shot `run-internal-ci` label when the current head is ready.
3. The Source bridge consumes that label and dispatches the exact PR head to
   private Internal CI.
4. Require `trtmc/premerge/required` to pass on that same head.
5. Merge only after human review and the repository rules permit it.

The complete logs and artifacts remain private. A passing PR is not retested by
the same premerge suite after merge.

## Retained staging utility

The following commands only inspect parser behavior or print proposed writes;
they do not make `ai-staging` a supported CI target:

```bash
DOC_REMOTE="github"
if ! git remote get-url "$DOC_REMOTE" >/dev/null 2>&1; then
  DOC_REMOTE="origin"
fi

python3 tools/ai_staging.py --help
python3 tools/ai_agent_system.py --help
python3 tools/ai_staging.py --remote "$DOC_REMOTE" --dry-run list
```

Both tools default to a remote named `github`; the fallback supports canonical
clones where that same repository is named `origin`. Confirm both the name and
canonical URL before any state-changing operation:

```bash
git remote -v
git remote get-url github 2>/dev/null || git remote get-url origin
```

The staging utility can still create, synchronize, retarget, rotate, and
promote branches, but those are GitHub mutations and require explicit operator
authorization. They are not documented here as the project's active merge
procedure.
