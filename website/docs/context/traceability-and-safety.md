---
title: Traceability and Safety Status
---

:::warning No ISO 26262 certification claim

This repository does not contain enough controlled evidence to claim ISO
26262 certification, ISO 26262-6 compliance, or complete bidirectional
requirements-to-test traceability. This page records gaps; it is not a safety
case, legal advice, or functional-safety approval.

:::

## Snapshot

The measurements below were recomputed from GitHub `main` commit
`e6b798cdb145c38caf1ede8eda7f5ce83f894138` on 2026-07-31.
Manifest files remain authoritative; these numbers are not constants:

| Measure | Snapshot value |
| --- | ---: |
| JSON manifests below `python/tensorrt_model_connect/models/*/tests/manifests/` | 204 |
| Declared testcases | 241 |
| Testcases with a non-empty `trace_id` | 215 |
| Unique non-empty trace IDs | 214 |
| Testcases without a trace ID | 26 |
| Duplicate IDs | 1 |

`IT-E2E-QIMG-01` is declared by both
`python/tensorrt_model_connect/models/qwen_image/tests/manifests/qwen-image.json` and
`python/tensorrt_model_connect/models/qwen_image/tests/manifests/qwen-image-2512.json`.

The repository also contains `Trace:` and `Trace ID:` annotations in selected
Python and C++ tests. There is no dedicated tool that checks every
architecture, unit-design, unit-test, and integration-test relationship in
both directions.

## What a traceability claim requires

A requirement, design, and test relationship is verified only when:

1. the requirement or design identifier exists in a maintained source;
2. the implementation and test use the same identifier;
3. every referenced path exists and the test exercises the stated behavior;
4. identifiers are unique within their scope;
5. all in-scope requirements have tests and all in-scope tests have a
   maintained requirement or design source; and
6. the test result and artifact provenance identify the exact tested revision.

Passing tests prove observed behavior within their test boundary. They do not
by themselves prove the formal relationship that a bidirectional matrix would
claim.

## Current partial evidence

The repository can validate model ownership, change impact, and selected
manifest and runtime-strategy contracts:

```bash
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
PYTHONPATH=python:. python3 -m pytest \
  tests/builder/test_manifest_validation.py \
  tests/tools/test_runtime_strategy_matrix_checker.py -q
```

The repository-wide strategy-matrix diagnostic is not green on this snapshot:

```bash
PYTHONPATH=python:. python3 tools/check_runtime_strategy_matrix.py
```

At commit `e6b798cdb145c38caf1ede8eda7f5ce83f894138`, it exits nonzero because
`diffusion_sana_wm` is absent from the matrix and five speech/omni task entries
have no discoverable runner class. That is a codebase evidence gap, not a
documentation success signal.

These checks are useful evidence, but none closes the 26 missing IDs, the
duplicate ID, or the absent bidirectional architecture/design verifier.

## Safety-oriented practices versus certification

The project can use safety-oriented engineering practices without presenting
them as certification:

- reviewed requirements and design decisions;
- explicit model-owned and shared-code boundaries;
- deterministic and versioned build inputs;
- independent code review;
- unit, integration, parity, negative, and regression testing;
- exact-revision CI and artifact provenance;
- controlled acceptance thresholds; and
- documented limitations and deviations.

Current certification gaps include:

- no approved safety scope, safety plan, ASIL allocation, or safety case;
- incomplete and duplicate E2E trace IDs;
- no CI-enforced bidirectional architecture/design/test matrix;
- no controlled qualification argument for model and performance artifacts;
- no repository evidence that historical reviewer placeholders represent
  independent review;
- no tool-qualification, anomaly-management, or formal-assessment record
  appropriate to an ISO 26262 claim.

## Closure criteria

Before publishing a normative traceability matrix or compliance claim:

1. define and maintain the in-scope requirement and architecture IDs;
2. resolve duplicate IDs and add unique IDs to every in-scope testcase;
3. add a checker for uniqueness, referenced paths, and both trace directions;
4. run that checker in CI;
5. generate the published matrix from checked source data;
6. retain exact-revision test results and artifact provenance; and
7. complete the controlled safety planning, review, qualification, anomaly,
   and assessment work appropriate to the intended claim.

Until those criteria are met, use model manifests as a partial E2E evidence
index and describe repository-wide traceability as incomplete.

{/* Collaborative review anchor: batch 2. */}
