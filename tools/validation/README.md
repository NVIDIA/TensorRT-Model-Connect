# Validation Engine Migration

`trtmc-validate` is the Dev/QA entry point for reference-consistency validation.
`task_eval.py` currently contains much of its Implementation, but it is not the
desired Module boundary. The target is a validation engine whose Interfaces are
usable by the Dev/QA CLI and CI without invoking a legacy task-eval CLI.

This migration deliberately preserves datasets, sample selection, generation
settings, comparison gates, cache keys, result JSON, and report behavior. A
phase must not weaken a validation criterion to make its tests pass.

## Target dependency direction

```text
tools/trtmc_validate.py       tools/ci/validation.py
              \                 /
               tools/validation/engine.py
                 |       |       |
              catalog  runners  comparison
                 |       |       |
              datasets reference artifacts
```

The engine is the high-Leverage Module: its Interface accepts one model/workload
request and returns one structured result. Dataset-specific and model-specific
details remain behind narrow Interfaces. Compatibility Adapters may point from
`task_eval.py` into the new modules during migration; new modules must never
import `task_eval.py`.

## Invariants checked at every phase

- `trtmc_validate.py --all --dry-run` selects the same model/workload pairs and
  sample limits.
- Reference cache identity and reuse remain unchanged.
- HF and TRTMC consume the same prepared sample IDs and generation settings.
- Existing comparison metrics, gates, summary JSON, report inputs, and vanilla
  reproduction metadata remain unchanged.
- A missing native reference runner fails closed; it cannot silently fall back
  to the legacy task-eval Implementation.
- Existing validation and task-eval unit tests remain green, apart from failures
  reproduced before the phase.

## Migration phases

### 1. Extract catalog and artifact contracts

Move suite loading, manifest projection, selector matching, suite resolution,
and prediction artifact validation into `tools/validation/`. Switch
`trtmc_validate.py` and `trtmc_reference.py` to those Interfaces. Keep aliases
in `task_eval.py` as a temporary Compatibility Adapter.

Exit criteria:

- Neither validation entry point imports `task_eval`.
- All ready dataset-backed bindings resolve to a native reference runner.
- Catalog, reference, and legacy task-eval unit tests preserve their baseline.

Rollback: revert this phase; no catalog format or generated artifact changes are
made.

### 2. Extract comparison

Move response alignment, task-specific comparators, gate evaluation, and summary
writers into `tools/validation/comparison.py`. Change `trtmc_compare.py` to call
that Interface directly. Keep forwarding aliases in `task_eval.py` until all
callers migrate.

Exit criteria:

- `trtmc_compare.py` does not import or invoke `task_eval`.
- Golden comparison fixtures for text, embeddings, speech, image, video,
  segmentation, reranking, and time series produce unchanged summary JSON.
- Agreement/disagreement status and report lights are unchanged.

Rollback: point `trtmc_compare.py` back to the Compatibility Adapter.

### 3. Extract preparation and execution

Create:

- `datasets.py` for deterministic sample selection and prepared inputs.
- `trtmc_runner.py` for bundle execution and raw-output conversion.
- `engine.py` for the one-model validation lifecycle.

Move one dataset kind at a time. For each move, `task_eval.py` forwards its old
command to the new Interface. Preserve process isolation, shared-engine
replacement, reference cache reuse, failure recording, and reproduction
metadata.

Exit criteria:

- Every workload in `model_workloads.yaml` runs through `engine.py`.
- One model failure still supports both continue and stop policies.
- No validation subprocess command contains `task_eval.py`.
- A smoke matrix covers at least one fast text, encoder, speech, vision,
  diffusion, and time-series workload.

Rollback: revert only the dataset-kind migration being moved; mixed old/new
dataset kinds remain supported during this phase.

### 4. Migrate CI

Rename `tools/ci/task_eval.py` to `tools/ci/validation.py`. Keep networked
dataset preparation separate from the network-isolated model proof. Invoke the
same engine Interface used by Dev/QA and preserve reviewed ETTh1 policy,
prebuilt-bundle requirements, artifacts, and fail-closed summary validation.

Exit criteria:

- CI source and generated container commands do not reference `task_eval.py`.
- CI policy, dataset-integrity, network-isolation, and artifact tests pass.
- A nightly dry run selects the same ETTh1 models.

Rollback: restore the previous CI Adapter; validation engine modules remain
usable.

### 5. Rename configuration vocabulary

Move `tests/task_eval/validation_suites.yaml` to
`tests/validation/workloads.yaml`. Introduce `validation` as the canonical model
manifest key and temporarily read `task_eval` only when `validation` is absent.
Reject conflicting dual definitions. Rename remaining helper scripts, work
directories, logs, tests, and documentation without changing behavior.

Exit criteria:

- New code and manifests write only `validation` vocabulary.
- A repository audit finds legacy names only in the Compatibility Adapter and
  its removal tests.
- Catalog and dry-run snapshots remain unchanged.

Rollback: default loaders can temporarily point back to the old file path
because the schema is unchanged.

### 6. Remove the compatibility layer

Delete `tools/task_eval.py` only after all production, CI, test, impact-analysis,
and documentation consumers use the new Interfaces. Rename the remaining legacy
test module rather than discarding its coverage.

Exit criteria:

- A repository audit finds no executable reference to `task_eval.py`.
- Validation, CI, E2E impact analysis, report generation, and representative GPU
  smoke tests pass.
- The deletion changes no dataset, sample count, threshold, or model scope.

Rollback: restore the thin Compatibility Adapter; the migrated engine stays the
source of truth.

## Scope boundary

Adding aligned workloads for currently not-compared models is follow-up feature
work, not part of the structural migration. Each new workload still needs a
fixed dataset contract, an independent reference Implementation, a TRTMC runner,
a comparator, reproduction metadata, and an explicit gate. Keeping this separate
improves Locality: migration failures cannot be confused with new model accuracy
failures.
