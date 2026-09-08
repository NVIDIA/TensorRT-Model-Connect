# Manual Review Probes

Use these probes after the fast diff-boundary pass. They capture recurring
semantic failures that static architecture tests and green CI can miss. Current
repository policy and code remain authoritative.

## Intent And Current Need

- Does the linked issue still reproduce on the PR's current base?
- Did #1093 or a later merge remove the original consumers or change the
  ownership boundary?
- Is a new abstraction used by a concrete current consumer in the same change?
- Is the proposed shared API needed by user behavior that existing public
  contracts cannot express, or is it only symmetry, future possibility, or
  line-count reduction?
- Does the PR description claim old paths, APIs, registries, or tests that no
  longer exist?

Do not equate “model-agnostic data format” with “every language needs another
parser.” Identify the canonical owner and an independent consumer first.

## Family Isolation Beyond Paths

- Search family production and tests for imports, includes, CMake links,
  `dlopen`, file reads, symlinks, fixtures, threshold files, and subprocess
  calls that enter a sibling family.
- Search shared code for literal family/checkpoint names, model tensor names,
  task-specific metrics, thresholds, datasets, probes, reference adapters,
  strategy maps, and family conditionals.
- Check whether adding or reverting the family requires editing a root source
  list, registry, switch, catalog exception, or another family.
- If the PR edits an architecture allowlist or guard so its new file can pass,
  review that as a shared-contract change. Adding a path to an expected set is
  not evidence that the path belongs in shared code.
- Treat a central allowlist or exclusion added only for one model as a coupling
  smell. Determine whether it is current approved application policy or model
  behavior that belongs in the family before reporting a violation.
- Verify package/import isolation: discovery must not import unselected family
  implementation or optional dependencies.

Passing path guards establishes structural conformance only. It does not prove
that configuration, runtime behavior, or validation semantics are isolated.

## Shared Contract Review

For every changed shared surface, list all known consumers and ask:

- Is the contract expressed in task/user terms rather than the first family's
  model terms?
- Does a default value accidentally encode one model's calibrated policy?
- Can specialization remain family-supplied without a family switch in core?
- Does the shared layer validate only safe generic mechanics while leaving
  section/output meaning with the family?
- Are public API, ABI, bundle, CLI, compatibility, and documentation effects
  declared and tested?
- Does the evidence cover structurally different consumers, not only the
  introducing family?

A new model-agnostic Task interface can be valid even when its first consumer
is one family. A new model helper, comparator, scheduler, or parser is not
justified merely because several implementations look similar.

## Runtime And Lifetime

- Trace ownership after `load_task()` and after the family factory returns.
- Look for references, pointers, spans, callbacks, or lambdas that outlive
  stack-local loader objects or `FamilyContext`.
- A deferred reader should own/copy the lightweight `BundleReader`; it must not
  retain a reference to temporary context.
- Confirm Task destruction precedes family/backend DSO unload and that streams,
  buffers, communicators, modules, samplers, and engines have clear owners.
- Ensure a family uses abstract `IBackend`/`IEngine`, not the concrete TRT
  backend or runtime loader implementation.
- Confirm distributed collectives load NCCL only when the family actually
  creates a communicator; rank-specific replicated plans alone do not justify
  it.
- Reject environment, current-directory, sibling-probe, or retry fallback that
  changes exact family/backend dispatch.

## Model Semantics And External Checkpoints

Inspect the pinned upstream revision rather than inferring behavior from the
family name.

- Match actual checkpoint weight keys and transformations exactly.
- Track tensor layout across reshape, transpose, flatten, concatenation, and
  binding boundaries.
- Check special classes/tokens such as no-object/background IDs, EOS variants,
  padding IDs, and ignored labels.
- Reproduce the model card's system/user messages, chat template options,
  preprocessing, generation settings, and task-specific control fields.
- Check resize aspect ratio, crop behavior, padding values, masks, coordinate
  spaces, and postprocessing scale restoration.
- Verify task defaults and support declarations describe the real capability,
  not only the base architecture class.
- Do not create a new family merely because a checkpoint has a specialized
  product use. The boundary follows mathematical topology and runtime
  orchestration; specialized checkpoint inputs and oracles may remain owned by
  an existing family when that family can validate them without shared policy.
- Check the exact checkpoint and asset licenses plus any naming, attribution,
  or redistribution conditions.

Native/reference agreement under the same malformed input is not proof of the
advertised model behavior.

## Validation And Oracle Strength

- Identify the bug or wrong output that each new test would reject.
- Confirm the changed production path is reached; mock fallback must not turn a
  build/load failure into a passing unit test.
- Confirm an E2E selector accounts for requested, executed, passed, failed, and
  skipped cases with one result per request.
- Check that an oracle observes every essential result field. Top-1 label alone
  may miss boxes/counts; mean and variance alone may accept unrelated images;
  one generated token may miss tokenizer, stopping, or cache defects.
- For numeric gates, inspect comparison direction, units, aggregation level,
  worst-case behavior, sample count, and negative controls.
- Verify reference and native sides use the same prompt framing, preprocessing,
  initial state/latents, precision intent, generation parameters, and output
  interpretation.
- Treat family-local E2E as family evidence. A shared runtime/build/CI change
  needs evidence proportional to every selected consumer class.
- Separate compilation, unit, runtime smoke, official-checkpoint inference,
  parity, target-hardware, performance, and release qualification.

## Dependency And Build Isolation

- A root requirement duplicated or contradicted by a family requirement creates
  two owners and can force cross-family coordination.
- A selected-family job should install only the shared base plus that family's
  extras; a green `--no-deps` install is not dependency compatibility proof.
- Check build files for central per-family sources, conditional names, or links
  to another family.
- Confirm family DSO naming and install rules are owner-local and that package
  validation discovers the DSO without importing all family implementations.
- Distinguish an immutable base-container pin from family-specific dependency
  policy.

## CI And Trust Boundaries

- Verify status/check conclusions against the current PR head SHA.
- Distinguish PR head tests from GitHub's exact merge-revision tests.
- Do not treat a dispatched workflow, successful bridge, or visible source job
  as the protected premerge result.
- Check that fork PR code cannot execute on a privileged self-hosted host before
  entering the intended isolated environment.
- Third-party actions should be full-SHA pinned and permissions least-privilege.
- Ensure changed test paths, JUnit locations, artifact names, and selectors
  still match their workflow consumers.
- Public contributor feedback must say which public command or phase failed
  without leaking private logs, hosts, artifacts, or URLs.

## Repository Hygiene

- New source/build file extensions must be classified by the legal-header
  checker rather than excluded merely to pass it.
- JavaScript and other languages must use the repository-approved SPDX comment
  form.
- New or copied media must appear in `ASSET_LICENSES.md` with correct origin and
  license when required.
- Tests must fail if a native addon, DSO, or executable cannot build or load;
  an automatic mock fallback hides the contribution's actual integration.
- PR categories, risk, paths, API claims, validation, and unrun gaps must agree
  with the diff and current repository layout.

## Avoid False Findings

- Shared-file modification alone is not coupling.
- Intentional family-local duplication is not a maintainability defect under
  this architecture.
- Pre-existing unchanged debt is not introduced by the PR.
- A family-only diff is not automatically behaviorally correct.
- Green static or CPU CI is not GPU, model-parity, or performance evidence.
- A passing round trip is not an independent compatibility proof.
- Missing evidence is uncertainty; describe exactly what remains unverified
  instead of asserting failure without evidence.
