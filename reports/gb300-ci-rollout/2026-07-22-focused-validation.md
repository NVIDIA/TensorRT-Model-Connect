# GB300 CI Trust-Hardening Focused Validation

Validation completed at `2026-07-22T05:49:14Z` in the isolated worktree on
branch `ci/gb300-28-slot-pool`. The branch was based on GitHub main
`2b08f200c5602a369ac592bdeb3960bf4e5e5ce2`; the tested Python and workflow
changes were still an uncommitted exact implementation patch on top of
`07f86c7eda9df1793d314d6d9e524dbfc3a49800` when these commands ran. This file
is repository-validation evidence only. It is not hardware, cache-readiness,
28-slot, Nightly, or merge acceptance.

## Results

- Consolidated rollout-focused pytest selection: `533 passed in 137.15s`.
- Source-quality pipeline: cyclomatic-complexity gate passed, changed-file lint
  passed, architecture contracts passed, and `156 passed in 31.54s`.
- Targeted Ruff for every changed Python file: passed.
- `git diff --check`: passed.
- actionlint v1.7.12 for all four changed workflows: passed after ignoring only
  its expected unknown-custom-label diagnostic for
  `trtmc-cache-anchor`; no YAML or expression diagnostic was ignored.

## Commands

```bash
PYTHONPATH=python:. python3 -m pytest \
  tests/tools/test_cache_lock.py \
  tests/tools/test_cache_warm_receipt.py \
  tests/tools/test_capacity_canary.py \
  tests/tools/test_ci_container_secrets.py \
  tests/tools/test_discover_cache_anchors.py \
  tests/tools/test_generate_model_proof_report.py \
  tests/tools/test_github_actions_ci.py \
  tests/tools/test_model_proof_runner.py \
  tests/tools/test_model_proof_security.py \
  tests/tools/test_warm_hf_cache_static.py -q

CI_BASE_REF=github/main python3 -m tools.ci pipeline source-quality

python3 -m ruff check --config ruff.toml \
  tools/ci/cache_warm_receipt.py tools/ci/capacity_canary.py \
  tools/ci/gpu_lease.py tools/ci/model_proof.py \
  tools/ci/model_proof_inner.py tools/ci/model_proof_security.py \
  scripts/generate_model_proof_report.py \
  tests/tools/test_cache_warm_receipt.py tests/tools/test_capacity_canary.py \
  tests/tools/test_generate_model_proof_report.py \
  tests/tools/test_github_actions_ci.py \
  tests/tools/test_model_proof_runner.py \
  tests/tools/test_model_proof_security.py

git diff --check

/tmp/trtmc-actionlint-v1.7.12 \
  -ignore 'label "trtmc-cache-anchor" is unknown' \
  .github/workflows/nightly.yml .github/workflows/trtmc-ci.yml \
  .github/workflows/model-proof.yml \
  .github/workflows/model-proof-capacity-canary.yml
```

The last required repository proof is GitHub CI on the exact committed and
pushed PR head. The hardware and Nightly acceptance gates remain intentionally
pending until the external prerequisites and controlled post-merge window are
ready.
