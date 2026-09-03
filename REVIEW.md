# Pull Request Review Contract

This file defines the repository-wide semantic review contract for automated
and human reviewers. `AGENTS.md` and the architecture and extension guides
under `website/docs/` remain authoritative. This contract turns those
decisions into a repeatable pull-request review.

## Review Objective

Review changed behavior, not only changed lines. Trace relevant callers,
consumers, configuration, tests, and data flow far enough to determine whether
the pull request preserves family ownership, shared-layer neutrality, and the
meaning of its validation evidence.

Automated review is a first-pass architecture review. It must reduce human
search effort by reporting evidence-backed concerns and explicit uncertainty;
it must not claim that an absence of findings proves correctness.

## Required Review Axes

### Standards

Check the pull request against the repository architecture and ownership rules:

1. A model family remains an independently implementable, testable, changeable,
   and revertible vertical slice.
2. One family does not import, include, link, load, read, inherit from, or reuse
   implementation or validation artifacts owned by another family.
3. Model-specific configuration, topology, tensor semantics, engine
   composition, runtime orchestration, preprocessing, postprocessing,
   reference behavior, and validation remain in the owning family.
4. Shared code contains only model-agnostic contracts and mechanics. A literal
   family or model name, family conditional, model tensor name, task-specific
   metric, dataset, threshold, probe, or runtime strategy in shared code is a
   reason to investigate ownership.
5. A family-oriented change does not require a central registry, switch,
   source list, strategy map, or edit to another family.
6. Applications, examples, and benchmarks depend on public build, load, task,
   and runtime APIs. Core, backend, and family implementations do not depend on
   application code.
7. Code similarity is never evidence for a cross-family abstraction.
   Duplication of model-specific code is intentional because it contains
   defects, merge conflicts, validation, and rollback within one family.
8. A new shared contract is justified by a concrete model-agnostic need and
   does not encode the behavior of its first consumer as a generic extension
   point.

Do not report pre-existing, unchanged architecture debt as a defect introduced
by the pull request. Report it only when changed code expands, depends on, or
makes that debt harder to remove.

### Spec

Check that the implementation and its evidence have the meaning claimed by the
pull request, linked issue, documentation, and tests:

1. Public APIs, CLI flags, bundle fields, metrics, gates, and configuration
   retain their documented semantics unless the change is explicit.
2. Shared behavior changes identify all consumers and provide proportionate
   compatibility or regression evidence. A passing test for one family is not
   evidence for all consumers of changed shared code.
3. Compared benchmark implementations measure equivalent regions. Treat
   preprocessing, transfer, synchronization, execution, reduction,
   postprocessing, validation, and serialization as distinct boundaries.
4. Workload accounting keeps execution units and task units distinct. Shards,
   batches, requests, queries, samples, tokens, and generated artifacts are not
   interchangeable denominators.
5. Aggregation and gate behavior preserve the intended level of evidence:
   per-sample, worst-sample, percentile, task aggregate, and reference-relative
   metrics are different contracts.
6. Thresholds, expected values, comparison oracles, and acceptance criteria are
   not weakened merely to obtain a passing result.
7. Tests exercise the changed behavior and would fail for the regression they
   claim to prevent.

## Evidence And Severity

Every finding must include:

- severity and whether it blocks the pull request;
- the violated invariant or specification;
- exact changed-file and line evidence;
- the dependency direction or before/after behavior;
- the affected family or shared consumers;
- the smallest correction that restores the contract.

Use these severities:

- **Blocking**: incorrect results, invalid performance claims, public contract
  breakage, direct cross-family dependency, or model-specific semantics added
  to a shared implementation.
- **High**: a likely ownership or behavioral defect with concrete evidence and
  material blast radius.
- **Medium**: non-blocking architecture debt introduced or expanded by this
  change, or incomplete evidence for a material shared change.
- **Low**: do not emit as an automated comment unless it prevents a specific
  future defect. Avoid style, naming, optional cleanup, and speculative
  refactoring comments.

Consolidate repeated manifestations of one root cause into one finding. Do not
repeat the same concern on every affected line.

## Required Outcome

The review must end with one of these outcomes:

- `PASS`: no standards or spec violation was found in the reviewed evidence.
- `BLOCK`: at least one evidence-backed blocking violation was found.
- `HUMAN REVIEW REQUIRED`: available evidence cannot resolve a material
  ownership, compatibility, measurement, or blast-radius question.

`PASS` means no violation was found; it is not proof that the pull request is
correct. Use `HUMAN REVIEW REQUIRED` instead of guessing. Human reviewers own
the final decision and any architecture exception.
