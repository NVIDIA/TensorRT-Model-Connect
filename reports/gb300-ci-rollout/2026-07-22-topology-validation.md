# GB300 Declarative-Topology Focused Validation

Validation completed on 2026-07-22 in the isolated
`ci/gb300-28-slot-pool` worktree after rebasing onto GitHub main
`c91cf50740e8fd2b346d83749485ccceefe89ff3`. This is repository validation of
the accepted declarative-topology design. It is not hardware, cache-readiness,
28-slot, Nightly, or merge acceptance; those gates remain pending.

## Results

- Consolidated rollout-focused pytest selection: `594 passed in 156.07s`.
- Source-quality pipeline: cyclomatic-complexity gate passed, changed-file lint
  passed, architecture contracts passed, and `157 passed in 32.44s`.
- Post-review workflow/cache selection after removing the manual topology input
  and making the shell lock open non-truncating: `323 passed in 3.00s`.
- Independent workflow/receipt/capacity selection: `271 passed`.
- actionlint v1.7.12 passed for all workflows with no ignored diagnostic.
- `git diff --check github/main` passed, including the then-uncommitted review
  delta.
- The cache-warm matrix contained exactly one node-label-only row for
  compute01 and one for compute02.
- A dependency audit found no `needs.*` reference outside a job's declared
  dependencies. Normal model matrices remain uncapped.
- The capacity workflow has no manual topology override and therefore always
  binds canary evidence to the protected-main topology file.

## Commands

```bash
PYTHONPATH=python:. python3 -m pytest -q \
  tests/tools/test_cache_lock.py \
  tests/tools/test_cache_warm_receipt.py \
  tests/tools/test_capacity_canary.py \
  tests/tools/test_ci_container_secrets.py \
  tests/tools/test_generate_model_proof_report.py \
  tests/tools/test_github_actions_ci.py \
  tests/tools/test_model_proof_runner.py \
  tests/tools/test_model_proof_security.py \
  tests/tools/test_warm_hf_cache_static.py \
  tests/tools/test_trtmc_bench.py

CI_BASE_REF=github/main python3 -m tools.ci pipeline source-quality

/tmp/trtmc-actionlint-v1.7.12 .github/workflows/*.yml

git diff --check github/main
```

The next repository proof is GitHub CI on the exact committed and pushed PR
head. Full host acceptance remains intentionally gated on repaired cache
ownership and the controlled drain/admission window.
