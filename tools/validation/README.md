# Validation engine

`tools/validation` is the implementation boundary for TRTMC
reference-consistency validation. It is shared by the Dev/QA model-first CLI,
the comparison command, and nightly model-proof CI.

## Modules

- `catalog.py` loads workload definitions, projects E2E manifests, and resolves
  model/workload compatibility.
- `artifacts.py` validates the durable prediction artifact contract.
- `model_plugin_contract.py` selects fixed model-manifest testcase/stage
  contracts and serializes model-owned stage outputs across the independent
  reference and TRTMC processes.
- `engine.py` prepares aligned inputs, runs TRTMC, compares reference and TRTMC
  outputs, evaluates gates, and writes machine-readable summaries.
- `tools/trtmc_reference.py` owns independent reference execution and its
  setting-keyed shared cache.
- `tools/trtmc_compare.py` is the narrow subprocess entry point used by
  `trtmc-validate`.

`tools/trtmc_validate.py` remains the supported Dev/QA entry point:

```bash
python tools/trtmc_validate.py gpt2-125m
python tools/trtmc_validate.py --all --dry-run
```

Nightly CI calls `engine.py` directly for reviewed ETTh1 preparation and
evaluation. No validation path invokes a legacy task-eval CLI.

## Contracts

The engine must preserve these invariants:

- Reference and TRTMC consume the same prepared sample IDs and generation
  settings.
- Reference cache keys include the effective inference contract.
- Shared `.bundle` bundles are replaced in place instead of accumulating per
  run.
- Missing native reference runners fail closed.
- One model failure is recorded without corrupting later model runs unless the
  caller selected stop-on-failure.
- Comparison metrics and gates remain workload-owned and cannot be weakened to
  make a test pass.
- Reports expose execution, comparison, and final validation status separately,
  with bounded disagreement evidence and native reproduction commands.

The prepared manifest still uses the `task_eval` metadata key. That key is a
persisted artifact and reference-cache schema, not an executable dependency.
Changing it would invalidate shared reference caches and existing result
artifacts, so it is intentionally retained until an independently versioned
artifact-schema migration is approved.

## Adding a workload

A new workload is complete only when it has:

1. a deterministic dataset preparation contract in
   `tests/validation/workloads.yaml`;
2. an independent native reference runner;
3. a TRTMC runner using the same prepared inputs;
4. an output comparator and explicit gates;
5. native reproduction metadata for failed samples;
6. a model binding and sample limit in
   `tests/validation/model_workloads.yaml`.

Models without that complete contract stay visible as `not_compared`; E2E
execution is never reported as reference consistency.
