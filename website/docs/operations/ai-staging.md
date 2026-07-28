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

The active premerge workflow is `.github/workflows/trtmc-ci.yml`:

1. Open a pull request targeting `main` from a branch in
   `NVIDIA/TensorRT-Model-Connect`.
2. Apply the one-shot `run-ci` label when the revision is ready to test.
3. GitHub consumes that label and validates the pinned PR merge snapshot.
4. Review the `Premerge CI` result and its exact tested base/head evidence.
5. Merge only after the repository ruleset and human review requirements pass.

Fork pull requests cannot use the one-shot label because the workflow needs a
repository write token to consume it.

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
