# Local AI Workflow Playbook

This playbook covers local preparation for the current GitHub PR flow. It does
not activate the retired `ai-staging` CI design.

## 1. Confirm the checkout

```bash
git status --short
git remote -v

DOC_REMOTE="github"
if ! git remote get-url "$DOC_REMOTE" >/dev/null 2>&1; then
  DOC_REMOTE="origin"
fi
DOC_REMOTE_URL=$(git remote get-url "$DOC_REMOTE")
case "$DOC_REMOTE_URL" in
  https://github.com/NVIDIA/TensorRT-Model-Connect|\
  https://github.com/NVIDIA/TensorRT-Model-Connect.git|\
  git@github.com:NVIDIA/TensorRT-Model-Connect.git|\
  ssh://git@github.com/NVIDIA/TensorRT-Model-Connect.git) ;;
  *)
    echo "Refusing non-canonical remote: $DOC_REMOTE_URL" >&2
    exit 1
    ;;
esac

git fetch "$DOC_REMOTE" main
git rev-parse HEAD
git rev-parse "$DOC_REMOTE/main"
```

Preserve unrelated local work. Prefer the repository-standard `github` remote;
the fallback supports ordinary clones where the same canonical repository is
named `origin`. Start new work from `$DOC_REMOTE/main` on a short-lived branch;
do not push directly to `main`.

## 2. Inspect available operator tooling

```bash
python3 tools/ai_agent_system.py --help
python3 tools/ai_staging.py --help
python3 scripts/autopilot/autorun.py --help
```

The first two tools default to a remote named `github`. Pass the resolved
`$DOC_REMOTE` so the same commands also work in a canonical clone whose remote
is named `origin`.

## 3. Run read-only preflight

The queue helper's `preflight` and `dashboard` operations read GitHub state.
`--dry-run` prevents supported write paths from mutating it. In the current
repository, `preflight` is expected to return nonzero and report the missing
`ai-staging` branch/labels; run it only to inspect the inactive design:

```bash
python3 tools/ai_agent_system.py \
  --remote "$DOC_REMOTE" \
  --project NVIDIA/TensorRT-Model-Connect \
  --dry-run \
  preflight

python3 tools/ai_agent_system.py \
  --remote "$DOC_REMOTE" \
  --project NVIDIA/TensorRT-Model-Connect \
  --dry-run \
  dashboard
```

These commands require GitHub credentials and network access even in dry-run
mode because they inspect repository state.

## 4. Implement and verify

Follow the owning model's three descriptors and run the smallest meaningful
tests before broader gates:

```bash
PYTHONPATH=python:. python3 tools/test_impact.py \
  --base "$DOC_REMOTE/main" \
  --head HEAD

PYTHONPATH=python:. python3 -m pytest \
  tests/tools/test_model_plugin_encapsulation_static.py -q
```

Use the affected model's `tests/e2e/models/<family>/` command only when the
required model, TensorRT runtime, GPU, binary, and plugin directory are
available.

## 5. Publish for review

Open a pull request targeting `main`. The repository's supported premerge gate
starts only when an authorized maintainer applies `run-ci`; the workflow
consumes that label and tests an immutable PR merge snapshot.

The retained `ai-staging` utilities can mutate branches and pull requests, but
they are not part of the current supported flow because no Actions workflow
validates that target.
