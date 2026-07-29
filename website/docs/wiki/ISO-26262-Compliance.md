# ISO 26262 Process Status

:::warning No certification claim

This repository does not contain sufficient evidence to claim ISO 26262
certification or complete ISO 26262-6 traceability. The current traceability
audit has missing and duplicate E2E trace IDs, and there is no CI-enforced
bidirectional architecture/design/test matrix.

:::

The project can use safety-oriented engineering practices without describing
itself as compliant:

- reviewed requirements and design decisions
- clear model/shared ownership
- deterministic, versioned build inputs
- independent code review
- unit, integration, parity, negative, and regression tests
- exact-head CI and retained artifact provenance
- controlled thresholds and documented deviations

## Current gaps

- The Wiki's former architecture and unit-design identifiers were not
  machine-linked to matching tests.
- Not every E2E testcase has a `trace_id`, and one current ID is duplicated.
- No repository tool verifies both directions of traceability.
- Qualification artifacts are not a controlled safety case.
- Reviewer/approval placeholders in historical pages are not evidence that
  independent review occurred.

See [Traceability Status](Traceability-Matrix.md) for the measured state.

## Before making a compliance claim

The project would need an approved scope and safety plan, controlled
requirements, qualified methods/tools where applicable, independence criteria,
configuration and change management, complete bidirectional traceability,
review records, anomaly management, reproducible verification evidence, and a
formal assessment appropriate to the claimed ASIL and product context.

This page is documentation of repository status, not legal or functional
safety advice.
