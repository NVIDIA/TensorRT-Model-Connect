# GB300 Declarative-Topology Focused Validation

Validation completed on 2026-07-22 in the isolated
`ci/gb300-28-slot-pool` worktree after rebasing onto GitHub main
`c91cf50740e8fd2b346d83749485ccceefe89ff3`. This is repository validation of
the accepted declarative-topology design, followed by exact-head Pre-Merge and
local-only cache-readiness evidence. It is not 28-slot, Nightly, or merge
acceptance; those gates remain pending.

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

## Subsequent exact-head and host-readiness evidence

- Pre-Merge run
  [29941192864](https://github.com/NVIDIA/TensorRT-Model-Connect/actions/runs/29941192864)
  passed on attempt 1 at `c200eed7574e351bfb3f0420c93afc3e6829eaab`.
  Legal, ownership/impact, source quality, source-only unit, five concurrent
  compute02 GPU proofs, combined report certification, and the final gate all
  succeeded.
- The exact `c200eed7` cache verifier and manifests passed on both nodes with
  networking disabled, no GPU device, a read-only Hugging Face cache mount,
  and no Hugging Face token. Each node reported 113 expected and present
  dependencies, zero missing, and zero downloaded.
- Both nodes emitted the same plan digest
  `61d144e2bff1306de9a1a7f9cba14a4ea128f61af46a648306ca63f8e7128d16`
  and resolved-cache digest
  `a9a81748787096c7fe827aae7d74265f9b7759e7de7c9a6377d74b1009fc0cfc`.
- The refreshed compute02 cache audit reported zero non-owner entries, zero
  unreadable files, zero unwritable files, and zero unwritable directories.
- The access-controlled evidence bundle is retained at
  `/workspace/users/yifeif/gb300-ci-rollout-evidence/20260722T174231Z-c200eed7-pre-merge`;
  its `SHA256SUMS` verifies every retained summary and log.

## Subsequent review and hardening validation

The post-`c200eed7` review delta added a protected, topology-derived
`rollback-capacity` mode and moved the one-shot Hugging Face token file outside
every ordinary container bind source. The latter closes a real-host alias in
which `RUNNER_TEMP` sits below `/workspace/users/yifeif`, a path mounted into
trusted containers.

- Rollout-focused suite: `606 passed in 154.05s`.
- Workflow/cache/security integration selection: `368 passed in 2.27s`.
- Source-quality pipeline: all gates passed, including `157 passed in 30.96s`.
- Capacity-only selection: `59 passed`.
- Secret/container plus workflow selection: `116 passed`.
- Ruff, Python bytecode compilation, actionlint v1.7.12, and both diff checks
  passed.
- The full canary contract still derives 16+12 slots. The rollback contract
  derives only compute02 GPUs 1 through 3 and rejects any requested capacity
  other than 12; no node or topology dispatch input exists.
- A realistic self-hosted `RUNNER_TEMP` regression and ordinary
  `-v`/`--volume`/`--mount` alias cases prove that token-file creation fails
  closed whenever another bind source would expose the secret.

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

Every later PR head must still pass normal exact-head GitHub CI. Full hardware
acceptance remains intentionally gated on the controlled drain/admission
window, merged-main warm receipts, exact 28-slot admission, canaries, test PRs,
and Nightly.
