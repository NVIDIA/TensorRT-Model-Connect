# Traceability Status

Status: incomplete; this page does not claim ISO 26262 traceability closure.

The former hand-written matrix mixed architecture IDs, invented unit-test IDs,
and stale E2E IDs while marking every row `verified`. It was removed because
the repository did not contain matching evidence for most of those labels.

## Current machine-readable evidence

E2E trace IDs live in `testcases[*].trace_id` inside
`tests/e2e/models/*/manifests/*.json`. At this revision:

- 204 manifests contain 241 testcases.
- 215 testcases declare a trace ID.
- Those declarations contain 214 unique IDs.
- 26 testcases have no trace ID.
- `IT-E2E-QIMG-01` is declared twice.

Therefore repository-wide bidirectional traceability is not complete. These
numbers are an audit snapshot; the manifest files are authoritative.

## What may be called verified

A requirement/design/test relationship is verified only when:

1. the requirement or design identifier exists in a maintained source;
2. the test contains or is machine-linked to the exact same identifier;
3. the referenced test path exists and exercises the stated behavior;
4. the current test result is retained with the tested revision;
5. integration claims point to a literal manifest testcase trace ID; and
6. duplicate IDs and untraced testcases have been resolved.

Passing tests without an explicit link prove behavior, but do not prove the
formal trace relationship claimed by a matrix.

## Current audit commands

The repository currently has no dedicated traceability verifier. Use the
manifest and ownership checks as partial evidence:

```bash
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py
PYTHONPATH=python:. python3 -m pytest \
  tests/builder/test_manifest_validation.py \
  tests/tools/test_runtime_strategy_matrix_checker.py -q
```

The live checker validates the current runtime-strategy inventory and its
model-local runner/comparator mappings; its pytest target validates checker
behavior. These commands do not close the 26 missing IDs or the duplicate ID
above.

## Closure work

Before restoring a normative matrix:

1. define maintained architecture/requirement IDs;
2. add unique trace IDs to all in-scope testcases;
3. introduce a checker for uniqueness, referenced paths, and both directions;
4. add the checker to CI;
5. generate the published matrix from checked data; and
6. retain exact-head test evidence.

Until then, use model manifests as an E2E evidence index and describe
traceability as partial.
