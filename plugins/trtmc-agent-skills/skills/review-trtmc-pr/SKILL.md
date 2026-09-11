---
name: review-trtmc-pr
description: >-
  Review a TensorRT-Model-Connect GitHub PR or local contributor branch against
  the current post-#1093 model-family isolation architecture, repository rules,
  behavioral correctness, and exact-head validation evidence. Use for
  contributor self-review before marking a PR ready, or when deciding whether
  a community PR is compliant, decoupled, merge-ready, or needs maintainer
  feedback. The review is read-only unless the user separately asks to publish
  it or change the contribution.
---

# Review TRTMC PR

## Review Contract

- Review an explicit comparison: the exact PR base and head in pull-request
  mode, or the fetched canonical `main`, local `HEAD`, and disclosed working
  tree state in local self-review mode. Never substitute a remembered branch.
- Treat the canonical `NVIDIA/TensorRT-Model-Connect:main`, root `AGENTS.md`,
  `CONTRIBUTING.md`, and current architecture documents as authoritative. PR
  #1093 explains the cutover but is historical rationale, not a substitute for
  current code and policy.
- Review changed behavior and newly relied-on behavior. Do not charge the PR
  for unrelated, unchanged migration debt.
- Keep the PR and contributor branch unchanged. Do not approve, request
  changes, comment, label, trigger CI, push, or edit code unless the user
  explicitly asks for that separate action.
- Treat fork code as untrusted. Inspect it in a detached temporary worktree;
  never run it on a privileged or protected runner and never expose secrets.
- An absence of findings is not proof of correctness. State untested paths and
  unresolved evidence explicitly.

## Select The Review Mode

- Use **pull-request mode** when given a PR number or URL. Prefer this mode for
  the final contributor self-review because it includes the exact remote head,
  PR description, commits, linked issue, and current checks.
- Use **local self-review mode** when asked to review a branch before a PR
  exists. Include committed changes and disclose staged, unstaged, and
  untracked files. A dirty working tree cannot receive a ready-to-submit result
  because its uncommitted content has no immutable reviewed head.

For contributor self-review, remain read-only and tell the contributor whether
the self-review checkbox can honestly be selected. Do not mark the PR ready,
push, commit, edit files, or publish comments unless separately requested.

## Establish The Pull-Request Baseline

Record the repository, PR number, base SHA, head SHA, author, linked issue,
current review state, and check state. Refresh remote information instead of
using a prior session or stale local ref.

```bash
gh pr view <PR> --repo NVIDIA/TensorRT-Model-Connect \
  --json number,title,body,url,state,isDraft,author,baseRefName,headRefName,headRefOid,mergeable,mergeStateStatus,reviewDecision,changedFiles,files,commits,reviews,comments,statusCheckRollup
gh api repos/NVIDIA/TensorRT-Model-Connect/pulls/<PR> \
  --jq '{base_sha: .base.sha, head_sha: .head.sha, author_association: .author_association}'
gh pr diff <PR> --repo NVIDIA/TensorRT-Model-Connect
gh pr checks <PR> --repo NVIDIA/TensorRT-Model-Connect
```

Use the paginated pull-files API when a PR is large enough that a summarized
file list may be incomplete. Fetch the exact PR head and create a detached
temporary worktree when local searches or tests are needed. Preserve the
user's active worktree and remove only the temporary worktree created for the
review.

## Establish The Local Self-Review Baseline

Inspect `git remote -v` and select the fetch remote whose URL resolves to
`NVIDIA/TensorRT-Model-Connect`. External forks normally name it `upstream`;
maintainer clones may name it `github` or `origin`. Never use a contributor
fork's `main` as the canonical base merely because its remote is named
`origin`. Fetch canonical `main`, then record the base ref and SHA, merge base,
local head SHA, branch name, and working-tree state.

```bash
git remote -v
git fetch <canonical-remote> main
git branch --show-current
git rev-parse <canonical-remote>/main
git merge-base <canonical-remote>/main HEAD
git rev-parse HEAD
git status --short
git diff --stat <merge-base>
git diff --name-status <merge-base>
git diff <merge-base>
git ls-files --others --exclude-standard
git log --format=fuller <merge-base>..HEAD
```

`git diff <merge-base>` includes committed, staged, and tracked unstaged
content. Inspect every intended untracked file separately because Git does not
include it in that diff. If the canonical remote is ambiguous, `HEAD` is the
canonical main branch, or intended files cannot be distinguished from unrelated
local work, report the limitation instead of guessing.

Before reading implementation details, establish whether the linked issue's
problem still exists on current `main`. A long-lived PR may have been
superseded, narrowed, or made unnecessary by #1093 or a later merge. A proposed
abstraction with no current consumer does not inherit justification from an
obsolete issue.

## Load The Current Rules

Always read:

- `AGENTS.md`
- `CONTRIBUTING.md`
- `.github/pull_request_template.md`
- the linked issue or discussion and material maintainer comments, when one is
  available

Then read only the architecture guides relevant to the changed paths:

- family or model work: `website/docs/architecture/ai-native-horizontal-scaling.md`,
  `website/docs/architecture/validation-design.md`, and
  `website/docs/extend/add-model-family.md`;
- Python build or discovery: `website/docs/architecture/build-pipeline.md`;
- native build: `website/docs/architecture/build-system.md`;
- loader, DSO, Task, or backend work:
  `website/docs/architecture/runtime-lifecycle.md` and
  `website/docs/architecture/runtime-plugins.md`;
- cross-layer ownership: `website/docs/architecture/units-and-ownership.md`.

When reviewing the PR title, body, validation claims, or proposed contributor
reply, also read and apply `../write-git-messages/SKILL.md`.

## Fast Review Funnel

### 1. Reconcile Intent With Current Main

Summarize the problem, observable exit criteria, non-goals, affected public
contracts, and claimed evidence. Compare these with the linked issue and the
current base. Flag stale paths, claims about already-existing APIs, obsolete
registries, or a solution that no longer has a concrete consumer.

For a new checkpoint or model, inspect the authoritative model card, config,
tokenizer/preprocessor configuration, license, and exact pinned revision.
Confirm its real task and required input protocol; architectural similarity to
an existing family does not prove that the existing reference or validation
path exercises the checkpoint's product behavior.

### 2. Classify The Diff Boundary

Build a changed-path map before deep review:

| Changed area | First question |
| --- | --- |
| `families/<family>/**` | Can this family be built, tested, changed, and reverted without a sibling edit? |
| more than one family | Is there a real multi-owner requirement, or a cross-family dependency/bulk migration? |
| `core/**` or public headers | Is this a narrow model-agnostic contract with concrete consumers? |
| `apps/**` or `examples/**` | Does it consume only public APIs and keep model semantics in the family? |
| `tools/**`, CI, or shared validation | Does it schedule generic mechanics without owning family policy or weakening evidence? |
| root dependencies/build files | Does one family now force coordination or dependency changes for every family? |
| media, weights, fixtures, generated files | Are source, license, attribution, and repository policy satisfied? |

A normal family contribution should change only `families/<owner>/**`. A
shared Task contract may legitimately accompany the first family whose user
behavior cannot be represented by an existing Task API; a shared edit is a
review trigger, not an automatic violation.

### 3. Check Architecture Ownership

Verify these invariants from code, build files, tests, and data flow:

- The family owns checkpoint identity, tasks/default, config, weights,
  topology, TensorRT graph, bundle-section semantics, runtime orchestration,
  bindings, preprocessing, postprocessing, dependencies, fixtures, manifests,
  thresholds, references, and oracles.
- A family does not import, include, link, load, inherit from, symlink to, or
  read implementation or validation artifacts owned by a sibling family.
- Similar model-specific code remains family-local. Do not recommend a shared
  helper merely to remove duplication.
- `support.py` is dependency-free apart from the support contract, uses exact
  identity matching, and produces exactly one owner or a clear error. It does
  not use broad prefixes, scores, priority, repository-name guesses, or
  first-match fallback.
- `model.py` exposes one plain `build(request, writer)` function, explicitly
  handles or rejects all request dimensions, and writes the selected task and
  backend without guessing.
- Core contains only model-agnostic discovery/loading, bounded bundle
  mechanics, public Build/Task/Engine contracts, stable primitives, and the
  explicit BYOK boundary. It contains no family switch, model tensor name,
  tokenizer policy, topology, threshold, dataset, reference logic, or
  family-specific default disguised as a generic API.
- Core owns safe bundle framing and bounded reads; each family owns its section
  names, schemas, and meaning. A second parser or new shared abstraction needs
  a current independent consumer and a single clearly owned contract.
- Runtime loads exactly the family and backend named by the bundle. Family
  pipelines use abstract Task and Engine APIs, do not link the concrete backend
  or loader, and do not retain references to temporary factory context.
- Applications, examples, benchmarks, and BYOK consume public APIs. Core,
  backend, and families never depend on application code.
- Generic family semantics do not branch on GPU, SM, CUDA, driver, or TensorRT
  version. A real family-local TensorRT custom plugin may be valid; expanding
  ad hoc CUDA/cuBLAS helper execution outside TensorRT or TVM-FFI is not.
- Shared dependencies are genuinely common. Family-only packages and versions
  stay in the owner `requirements.txt`, and one family job does not merge
  incompatible environments from several families.

Read [references/manual-review-probes.md](references/manual-review-probes.md)
when any model, family, runtime, bundle, benchmark, validation, dependency, or
CI behavior changes. It covers semantic failures that path guards cannot
detect.

### 4. Trace Behavioral Correctness

Follow the changed path end to end rather than reviewing isolated functions:

```text
checkpoint -> support resolution -> family build -> bundle
           -> exact family DSO -> Task call -> family-owned oracle
```

Inspect relevant callers, consumers, configuration, tensor shapes/layouts,
weight names, special tokens/chat templates, resize/pad/mask behavior,
postprocessing, object lifetimes, failure propagation, and teardown. Compare
the native path with the official reference using identical semantic inputs.

Do not accept parity between two implementations that share the same wrong
assumption. A writer-reader round trip is useful consistency evidence but is
not an independent format validator. Likewise, a test that compares only a
convenient subset of outputs does not validate omitted task semantics.

### 5. Audit Validation Meaning

- Tests must exercise the changed behavior and fail for the regression they
  claim to prevent.
- A requested E2E case must produce exactly one executed, non-skipped result.
  Missing, duplicate, skipped, or unintended cases fail closed; a zero-test
  pytest exit is not a pass.
- Oracles must cover the task's essential outputs. Examples include boxes and
  detection counts for detection, generated content for text, reference-relative
  image/video evidence, and temporal behavior when the task promises it.
- Reference and native inputs, tokenizer framing, build options, precision,
  and aggregation levels must be aligned before drawing a parity conclusion.
- Thresholds, expected values, oracle strength, and acceptance criteria must
  not be weakened to obtain green CI. If a test appears wrong, require human
  review of the evidence.
- Benchmark sides must time equivalent regions and keep shards, batches,
  requests, queries, samples, tokens, and artifacts as distinct accounting
  units.
- A family-only test does not prove a changed shared contract for all
  consumers. Scale evidence to the actual impact classification.

### 6. Audit Contributor And Repository Compliance

Check that:

- the PR title is focused, imperative, Conventional Commit-style, and contains
  no banned terms;
- every commit introduced by the PR has a valid author-owned DCO sign-off;
- all pull-request template sections are complete, change categories match the
  actual diff, exactly one risk level is selected, and paths/claims describe
  the current architecture;
- validation lists exact commands, outcomes, tested head, checkpoint and
  dependency revisions, hardware/environment when relevant, and unrun paths;
- the contributor self-review checkbox is selected only after the contributor
  has actually reviewed the change;
- repository documentation, code comments, user-facing messages, and PR text
  are English except model data needed for multilingual validation;
- SPDX headers, third-party notices, asset entries, and redistribution rights
  are complete;
- the change remains focused and does not add compatibility paths, fallbacks,
  registries, speculative hooks, or unrelated cleanup.

## Run Proportionate Checks

In pull-request mode, run checks from the exact-head temporary worktree. In
local mode, run them from the reviewed checkout and report whether it was
clean. Start with the current lightweight commands from `CONTRIBUTING.md`:

```bash
git diff --check <base-sha>...<head-sha>
PYTHONPATH=core/builder:apps/benchmark:. python3 -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python3 tools/test_impact.py --validate
PYTHONPATH=core/builder:apps/benchmark:. python3 tools/test_impact.py \
  --base <base-sha> --head <head-sha>
PYTHONPATH=core/builder:apps/benchmark:. python3 -m pytest \
  tools/tests/test_architecture.py tools/tests/test_family_impact.py
```

For a dirty local self-review, use `git diff --check <merge-base>` to include
tracked working-tree changes, inspect untracked files explicitly, and state
that impact selection covers the committed head unless explicit files were
supplied. Do not represent uncommitted content as reviewed by the `HEAD` SHA.

Then run the smallest tests that directly exercise each changed contract and
affected family. Do not run downloaded contributor code with credentials or on
protected hardware. If dependencies, checkpoints, or target hardware are not
available, inspect the test logic and report the missing evidence instead of
claiming execution.

Interpret CI carefully:

- Public Community CI and protected premerge are different evidence tiers.
- A bridge dispatch is not a protected test pass.
- Only a passing `TRTMC Internal CI / Automated premerge gate` on the current
  head proves protected premerge for that head.
- A later push invalidates earlier head evidence. A base advance may also
  require a fresh exact-merge result.
- Green source or architecture checks do not prove TensorRT build, checkpoint
  inference, model parity, target-platform behavior, or performance.

## Report Findings First

For a maintainer review, return a concise review in this order:

1. `Verdict: PASS | BLOCK | HUMAN REVIEW REQUIRED`, `Merge readiness: READY |
   NOT READY`, the recommended GitHub action (`Approve`, `Request changes`,
   `Comment`, or `Wait`), and exact base/head.
2. Findings ordered by severity. For each finding include:
   - `Blocking`, `High`, or `Medium` and a short title;
   - violated architecture rule or claimed behavior;
   - exact changed-file line evidence and the relevant caller/consumer;
   - affected family or shared blast radius;
   - the smallest correction or evidence that would resolve it.
3. A short architecture summary covering family ownership, shared neutrality,
   one-way application dependencies, and validation ownership.
4. Checks/evidence verified, followed by untested paths and residual risk.
5. A contributor-facing reply only when requested. Draft it in friendly,
   specific English; distinguish repository evolution from contributor error.

For contributor self-review, use the same finding and evidence structure, but
replace merge authority with:

1. `Verdict: PASS | BLOCK | HUMAN REVIEW REQUIRED`, `Submission readiness:
   READY TO MARK READY | READY TO OPEN DRAFT | KEEP DRAFT | NOT READY`, exact
   base/head, and whether the worktree is clean.
2. The next contributor action. Do not recommend `Approve` or `Request
   changes`, because contributors do not review their own PR in that role.
3. Whether the contributor can honestly select the **Contributor Self-Review**
   checkbox. Keep the detailed verdict, findings, exact head, and evidence in
   the review response; the PR template requires only the confirmation.

Keep review verdict separate from merge readiness. A code/architecture `PASS`
may still be `NOT READY` while required exact-head CI, model proof, or human
approval is pending.

Use `BLOCK` for an evidence-backed incorrect result, validation weakening,
public-contract defect, direct cross-family dependency, model semantics in
shared implementation, or other merge blocker. Use `HUMAN REVIEW REQUIRED`
when a material contract, consumer need, compatibility choice, or architecture
exception cannot be resolved from available evidence. Use `PASS` only when no
violation was found, and preserve its limits.

Do not emit style-only Low findings. Consolidate repeated symptoms under one
root cause, distinguish facts from risks, and never invent a test result or
claim that CI covers an unexecuted path.
