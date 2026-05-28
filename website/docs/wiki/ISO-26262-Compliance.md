# ISO 26262 Process Compliance

| Field | Value |
|-------|-------|
| Document ID | TRTMC-ISO-001 |
| Applicable standard | ISO 26262:2018 Part 6 (Software Development) |
| Revision | 1.0 |
| Date | 2026-03-12 |
| Author | Safety Architecture Team (TensorRT-Model-Connect Team) |
| Reviewer | Independent Review Required (TBD — assign before merge) |
| Review Status | Pending independent review |
| Status | Active |

---

## 1. Document Control

This document records the project's alignment with ISO 26262:2018 Part 6
process requirements. It is maintained alongside the codebase and updated
when process changes occur. Changes to this document require PR review by
at least one person who did not author the change.

---

## 2. Applicable Standard

**ISO 26262:2018 Part 6 -- Product development at the software level**

This standard defines requirements for software development in
safety-related automotive systems, covering architecture design, unit
design, unit verification, integration verification, and safety validation.

---

## 3. Current ASIL Classification

**QM (Quality Management)**

This project is not currently a safety-critical automotive system. We follow
ISO 26262 process as an engineering best practice for the following reasons:

- **Future qualification readiness**: The project may be deployed in
  safety-relevant contexts; establishing process compliance now reduces
  future qualification effort.
- **Engineering discipline**: ISO 26262 process requirements (traceability,
  independence, documentation integrity) improve code quality and
  maintainability regardless of safety classification.
- **Audit preparedness**: Following a recognized standard provides a
  structured framework for external review.

If the project is later classified at ASIL A-D, this document will be
updated with the specific ASIL level and any additional requirements
beyond QM.

---

## 4. Process Compliance Matrix

| ISO 26262-6 Clause | Requirement | Current Status | Evidence | Gap |
|---------------------|------------|----------------|----------|-----|
| **Section 7: Software architectural design** | Architecture shall be documented with sufficient detail to support verification. Safety-relevant interfaces and data flows shall be identified. | Implemented | `website/docs/wiki/Architecture-Overview.md`, `website/docs/wiki/Static-Design.md`, `website/docs/wiki/Dynamic-Design.md`, `website/docs/wiki/Pipeline-Deep-Dive.md` | None for QM |
| **Section 7.4.1**: Hierarchical structure | Software architecture shall exhibit a hierarchical structure with well-defined interfaces. | Implemented | Source layout: `src/bundle/`, `src/cabi/`, `src/runtime/`, `src/tokenizer/`, `src/utils/`, `include/trtmc/`. Public API in `include/trtmc/`. All runtime communication through defined interfaces (`ITokenizer`, `TrtModule`, pipeline contracts). | None for QM |
| **Section 7.4.3**: Restricted coupling | Coupling between components shall be restricted and documented. | Implemented | Pipeline factory (`src/runtime/registry/pipeline_factory.cpp`) delegates to manifest-registered plugins via `PipelineRegistry`. Each plugin in `src/runtime/plugins/` is self-contained. Pipelines depend only on `TrtModule` and `KvCache`/`RecurrentState` abstractions. Python build package (`python/tensorrt_model_connect/`) is decoupled from C++ runtime. | None for QM |
| **Section 8: Software unit design and implementation** | Unit design shall be documented. Coding guidelines shall be applied. | Implemented | Unit designs documented via UD-* trace IDs in `website/docs/wiki/Traceability-Matrix.md`. C++17 with `-Wall -Wextra -Wpedantic`. Cyclomatic complexity gate: `python tools/check_cyclomatic_complexity.py src --max-ccn 10`. | None for QM |
| **Section 8.4.3**: Coding guidelines | Static analysis and coding standards shall be enforced. | Partially implemented | Compiler warnings as errors (`-Wall -Wextra -Wpedantic`). Cyclomatic complexity gate enforced in CI. | No dedicated static analysis tool (e.g., clang-tidy) integrated in CI. |
| **Section 9: Software unit verification** | Each software unit shall be verified against its design. Test methods: requirements-based testing, interface testing, structural coverage. | Implemented | 92 C++ test files in `tests/cpp/`, 98 Python test files in `tests/builder/`, 62 Python test files in `tests/tools/`. All linked to ARCH-* contracts via `website/docs/wiki/Traceability-Matrix.md`. | Coverage metrics tracked but not yet gated in CI at a specific threshold. |
| **Section 9.4.1**: Requirements-based testing | Tests shall verify specified requirements. | Implemented | Mandatory test intent fields (Intent, Preconditions, Postconditions) documented in `website/docs/wiki/Traceability-Matrix.md`. Trace IDs link each test to its architecture contract. | Not all existing tests have been retroactively annotated with full intent fields. |
| **Section 9.4.2**: Interface testing | Unit interfaces shall be tested. | Implemented | C ABI interface tested in `tests/cpp/test_c_abi_entry.cpp`. Pipeline API tested in `tests/cpp/test_pipeline_api.cpp`. Python CLI tested in `tests/builder/test_cli.py`. | None for QM |
| **Section 9.4.4**: Structural coverage | Statement and branch coverage shall be measured. | Partially implemented | Coverage infrastructure in place (gcov/gcovr). Current line coverage: ~49.6%. Branch coverage: ~28.7%. | No minimum coverage threshold enforced in CI. Branch coverage below typical QM targets. |
| **Section 10: Software integration and verification** | Integration tests shall verify correct interaction between units. | Implemented | Unified E2E test suite: `tests/test_e2e.py` with 122 model manifests in `tests/e2e/models/`. Orchestrator in `tests/e2e_harness/orchestrator.py` coordinates full lifecycle (build, run, compare). | None for QM |
| **Section 10.4.1**: Integration test methods | Integration tests shall use requirements-based testing. | Implemented | Each E2E manifest maps to a `task_strategy` which selects runner, comparator, and reference backend. Thresholds defined per-strategy in `tests/e2e_harness/thresholds/`. Comparators enforce metric-based gating (cosine similarity, NED, mIoU, PSNR, etc.). | None for QM |
| **Section 11: Verification of software safety requirements** | Software safety requirements shall be validated against the technical safety concept. | Not applicable | QM classification -- no safety requirements defined. | Will require safety requirements specification if ASIL classification is assigned. |

---

## 5. Organizational Independence Requirements

For QM classification, the minimum independence requirement is:

> **Author is not the Reviewer** (1c independence level per ISO 26262-8 Table 1)

### Current Enforcement

| Requirement | Mechanism | Status |
|-------------|-----------|--------|
| Code author cannot approve their own PR | Git branch protection rules requiring at least one approving review from a non-author | Enforced |
| Architecture docs must be reviewed by someone other than the doc author | PR review process | Enforced |
| Tests should be designed by someone other than the implementation author | Team practice (not automated) | Partially enforced |
| Documentation accuracy must be independently verified | PR review checklist | Enforced |

### Escalation for Higher ASIL

If the project is classified above QM:

- **ASIL A-B**: Reviewer must be independent of the development team (1b level).
- **ASIL C-D**: Verification must be performed by an independent organization (1a level).

---

## 6. Documentation Integrity Controls

### Rules

1. **Architecture docs MUST describe only implemented code.** Documents must
   not describe planned, aspirational, or target-state architecture as if
   it were current. Every file path, class name, and interface referenced
   in architecture documents must exist in the repository.

2. **Target/planned architecture MUST be in separate documents** clearly
   labeled with "TARGET" or "PLANNED" in the title or header. The document
   `website/docs/wiki/Runtime-Target-Architecture.md` follows this convention.

3. **Every file path in docs MUST be verified to exist.** No phantom file
   references are acceptable. This applies to all documents in `website/docs/wiki/`,
   `AGENTS.md`, and any other documentation files.

### Enforcement

| Control | Implementation | Status |
|---------|---------------|--------|
| File reference verification tool | `tools/check_doc_file_references.py` | PLANNED -- not yet implemented |
| PR review checklist item | "All file paths in changed docs verified to exist" | Active |
| Traceability matrix file audit | All paths in `website/docs/wiki/Traceability-Matrix.md` verified 2026-03-12 | Complete |

### Corrective Actions Taken

The Traceability Matrix v1.0 contained 7 phantom file references to planned
but unimplemented components. These paths (such as builders, router, and
adapter directories that were part of the aspirational service-composed
architecture) were removed in revision 2.0 and replaced with references
to actual source files. See the v1.0→v2.0 diff in git history for details.

---

## 7. Traceability Requirements

### Bi-directional Traceability

The following trace chain must be maintained:

```
Architecture Contract (ARCH-*) <--> Unit Design (UD-*) <--> Unit/Integration Test (UT-*/IT-*)
```

- Every `ARCH-*` must have at least one `UD-*` link and at least one
  `UT-*` or `IT-*` link.
- Every `UT-*`/`IT-*` must trace back to exactly one primary `ARCH-*`.
- Every `UD-*` must reference real source files that exist in the repository.

### Matrix Location

The authoritative traceability matrix is maintained at:

`website/docs/wiki/Traceability-Matrix.md`

### Completeness Rules

Each row in the matrix must have a status:

| Status | Meaning |
|--------|---------|
| `draft` | Architecture contract defined but test evidence incomplete |
| `active` | Test evidence exists but not yet independently verified |
| `verified` | Unit + integration evidence available and independently confirmed |

A row may only transition to `verified` when:

1. All referenced source files exist.
2. All referenced test files exist and pass.
3. Test intent fields (Intent, Preconditions, Postconditions) are documented.
4. At least one person other than the test author has confirmed the trace.

---

## 8. Change Control

### Architecture Changes

When source code changes affect the software architecture:

- The corresponding architecture document(s) in `website/docs/wiki/` must be
  updated in the **same PR** as the code change.
- The Traceability Matrix must be updated if any `ARCH-*` or `UD-*`
  rows are affected.
- File path references in all documents must be verified against the
  post-change repository state.

### Test Changes

When tests are added, modified, or removed:

- The test must include or update trace IDs (`ARCH-*`, `UT-*`/`IT-*`).
- The test must document Intent, Preconditions, and Postconditions.
- The Traceability Matrix must be updated if trace links change.

### Documentation-Only Changes

Changes that affect only documentation files (no code or test changes)
are permitted only for:

- Corrections to inaccurate statements
- Clarifications that do not change the described behavior
- File path updates after source reorganization
- Status updates in the Traceability Matrix

---

## 9. Audit Readiness Checklist

This checklist should be reviewed before any external audit or formal
release gate.

- [ ] All architecture documents (`website/docs/wiki/`) describe only implemented code
- [ ] No phantom file references in any documentation
- [ ] Traceability Matrix (`website/docs/wiki/Traceability-Matrix.md`) has no orphan rows
  (every ARCH-* has UD-* + UT-*/IT-* evidence)
- [ ] All `UD-*` entries reference source files that exist in the repository
- [ ] All `UT-*`/`IT-*` entries reference test files that exist in the repository
- [ ] All tests pass in CI (Tier 1-4 regression suite)
- [ ] Organizational independence enforced for PR reviews (author != reviewer)
- [ ] Cyclomatic complexity gate passes (`--max-ccn 10`)
- [ ] Coverage metrics recorded (line, function, branch)
- [ ] No `draft` status rows remain in the Traceability Matrix for shipped features
- [ ] Target/planned architecture is in separate, clearly labeled documents

### Last Audit Readiness Review

| Item | Date | Result | Reviewer |
|------|------|--------|----------|
| File reference integrity (Traceability Matrix) | 2026-03-12 | Pass (all paths verified) | Automated verification + independent human review required (pending) |
| Phantom file correction | 2026-03-12 | 7 phantom references corrected | Automated verification + independent human review required (pending) |

---

## 10. Gap Summary and Remediation Plan

| Gap ID | Description | ISO 26262-6 Clause | Priority | Remediation |
|--------|-------------|---------------------|----------|-------------|
| GAP-001 | No dedicated static analysis tool in CI | Section 8.4.3 | Medium | Integrate clang-tidy or similar into CI pipeline |
| GAP-002 | No minimum coverage threshold enforced in CI | Section 9.4.4 | Medium | Define and enforce line/branch coverage gates |
| GAP-003 | Not all existing tests retroactively annotated with intent fields | Section 9.4.1 | Low | Progressive annotation of existing tests during maintenance |
| GAP-004 | File reference verification tool not yet implemented | Section 7 (doc integrity) | Medium | Implement `tools/check_doc_file_references.py` and add to CI |
| GAP-005 | Test designer independence not automated | Section 9 | Low | Acceptable at QM; document practice for higher ASIL |
