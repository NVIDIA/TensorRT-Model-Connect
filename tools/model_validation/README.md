# Model Validation foundation

This package is the compatibility-first foundation for incrementally separating
Task Eval orchestration from task-specific behavior.

## Current phase

The current implementation:

1. Defines immutable, versioned `ValidationRequest` and `ValidationPlan`
   contracts.
2. Computes deterministic SHA-256 identities for the suite, each selected model,
   and the complete plan.
3. Writes `validation_plan.json` next to the existing Task Eval artifacts without
   changing `eval_summary.json`, legacy result fields, gates, or exit codes.
4. Provides a fail-closed task adapter registry for native suite migration.
5. Migrates ETTh1 workload identity and fidelity reduction to a native
   `TimeSeriesTaskAdapter`.
6. Runs explicit process-scoped HF/TRTFB Performance Evaluation with warmup,
   repetitions, raw observations, p50/p95, throughput, error rate, optional
   peak device memory, environment identity, comparison keys, and baseline
   gates. ETTh1 measurement outputs are verified against correctness-stage
   output digests before any baseline can be eligible for approval.

The existing `tools/task_eval.py` runtime still owns dataset preparation,
HF/TRTFB execution, scoring, and publication. Ordered sample identity is
therefore marked `deferred_to_legacy_runtime` in compatibility plans.

## Performance safety rule

Legacy compatibility plans reject Performance Evaluation. Existing `wall_ms` values
have different scopes across task families and are diagnostic only; they cannot
be used as a common baseline or performance gate.

A suite can request Performance Evaluation only after it has a native task
adapter and an explicit measurement profile defining warmup, measured
iterations, synchronization, process repetitions, metrics, and environment
compatibility.

ETTh1 currently uses `process_e2e` because its reference and TRT runners launch
a process for each sample. It must not be compared with a future
`warm_session` profile.

## Plan integrity

`validation_plan.json` contains:

- schema and plan kind;
- compatibility mode;
- suite contract identity;
- requested models, dataset override, limit, seed, and assessments;
- deferred or resolved workload identity;
- ordered model cases and model contract identities;
- a digest covering every serialized semantic field.

`ValidationPlan.from_dict()` recomputes the digest and rejects modified or
partially corrupted plans.

## Migration rule

Native task adapters are added one task kind at a time. Unregistered
kinds have no implicit fallback in the native registry. Legacy execution remains
available through the compatibility facade, but it cannot silently opt into
native Performance Evaluation.
