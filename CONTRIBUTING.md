# Contributing to TensorRT-Model-Connect

Thank you for your interest in contributing. External contributors should do all
development in a personal fork and submit changes to
[`NVIDIA/TensorRT-Model-Connect`](https://github.com/NVIDIA/TensorRT-Model-Connect)
through a pull request. Do not work directly on the upstream `main` branch.

The [Contributor Quickstart](website/docs/extend/contributing.md) and
[architecture guide](website/docs/architecture/ai-native-horizontal-scaling.md)
provide project-specific design, testing, and ownership guidance.

Maintainers can optionally request an
[advisory documentation review](.github/documentation-check.md). It is manually
triggered, keeps reports private, and does not replace the existing merge gates.

## Development workflow

### 1. Fork and clone the repository

Use GitHub's **Fork** button to create
`https://github.com/YOUR-GITHUB-USERNAME/TensorRT-Model-Connect`, then clone
your fork and add the NVIDIA repository as `upstream`:

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect
git remote add upstream https://github.com/NVIDIA/TensorRT-Model-Connect.git
git fetch upstream
```

In this layout, `origin` is your writable fork and `upstream` is the canonical
repository. Keep these roles separate.

### 2. Create a focused branch from current upstream `main`

Create a short-lived topic branch for each contribution. Do not develop on your
fork's `main`, because it may be behind upstream:

```bash
git switch -c docs/improve-contributing upstream/main
```

Use a concise branch name that describes one purpose, such as
`fix/qwen-tokenizer` or `docs/model-validation`.

### 3. Find the narrowest owner

Before editing, identify the component that owns the behavior. A normal model
contribution changes only `families/<owner>/**`. It may duplicate sibling
implementation but must not import it. A shared-core change requires a concrete
model-independent contract that the current public API cannot express.

Start with the relevant guide:

- [Add a Model Family](website/docs/extend/add-model-family.md)
- [AI-Native Horizontal Scaling Architecture](website/docs/architecture/ai-native-horizontal-scaling.md)
- [Bring Your Own Kernel](examples/byok/README.md), only for an explicit BYOK contribution

Python family builders are plain functions and must not inherit. Do not add a
compatibility path, fallback, migration layer, family dependency hash, or
artifact digest. Digest pins are limited to CI container base images.

Open an issue before investing in a large, cross-cutting, or user-visible design
when the intended ownership or approach is not already clear.

### 4. Make a reviewable change and validate it locally

Keep each pull request focused on one problem. Add or update focused tests when
behavior changes, and update documentation when users or developers will observe
the change.

Install the lightweight commit-time quality hooks once in each clone:

```bash
python3 -m pip install --requirement requirements/community-ci.txt
pre-commit install --install-hooks
```

On Windows, use `py -3 -m pip` in place of `python3 -m pip`.

The hooks trim trailing whitespace, ensure one final newline, validate YAML,
check Ruff, and verify clang-format. Pre-commit manages the Ruff and
clang-format environments on Linux, macOS, and Windows. There is no pre-push
hook: builds and the broader source-only CPU suite run automatically on the
pull request after the branch is pushed.

Start with repository consistency checks:

```bash
PYTHONPATH=core/builder:apps/benchmark:. python3 -m tools.model_ci validate
PYTHONPATH=core/builder:apps/benchmark:. python3 tools/test_impact.py --validate
git diff --check
```

Then run the smallest meaningful tests for the code you changed. Model work also
needs the declared model, runtime, hardware, and comparison evidence. For
documentation changes, run:

```bash
npm --prefix website ci
npm --prefix website run test:model-support
npm --prefix website run build
```

Do not weaken a test, oracle, or acceptance threshold to make a change pass. If
you believe a test is incorrect, explain the evidence in the pull request and
ask a maintainer to review it.

### 5. Commit with a clear message and DCO sign-off

Repository history uses short, imperative, Conventional Commit-style subjects,
for example:

```text
docs(contributing): clarify external workflow
fix(qwen): preserve tokenizer contract
test(ci): cover fork pull requests
```

Sign off every commit as described in [Signing Your Work](#signing-your-work):

```bash
git commit --signoff -m "docs(contributing): clarify external workflow"
```

### 6. Sync your branch and push it to your fork

Before opening the pull request, incorporate the current upstream `main` and
rerun the affected checks:

```bash
git fetch upstream
git rebase upstream/main
git push --set-upstream origin docs/improve-contributing
```

If you rebase a branch that you already pushed, never use an unguarded force
push. Use `git push --force-with-lease`, and coordinate before rewriting a
branch that other people are using.

### 7. Open a draft pull request against upstream `main`

Open the pull request from your fork branch to
`NVIDIA/TensorRT-Model-Connect:main` and keep it in draft while completing the
self-review below. Complete every section of the pull-request template; use
`Not applicable: <reason>` instead of deleting a section. Include:

- **Background**: the problem, motivation, current behavior, and linked issue;
- **Exit Criteria**: the conditions that define completion, including important
  non-goals;
- **Implementation**: the approach, affected models and components, and any
  API, ABI, bundle, dependency, compatibility, migration, or rollout changes;
- **Validation**: exact commands and results, tested head and dependency/model
  revisions, environment and hardware, plus paths that were not run; and
- **Contributor Self-Review**: confirmation that you reviewed your change; and
- **Notes For Future Readers**: remaining risk, compatibility or rollout notes,
  third-party provenance, and useful follow-up context.

Once the pull request is marked ready, `PR Metadata / Required` checks that
these sections and the structured validation evidence are present, including
the contributor self-review confirmation. The trusted triage workflow derives
model and component labels from the actual diff and repository ownership
metadata; it uses the template only for declared risk and compatibility-change
labels. DCO sign-off is enforced by the repository's DCO check rather than a
self-attested template checkbox.

Compilation, unit tests, inference, model parity, target-hardware execution,
performance, and release qualification are separate evidence levels. Claim only
what the recorded validation proves.

### 8. Perform contributor self-review

Review the exact current draft pull-request head after implementation and local
validation. Codex users can run the repository-provided skill from the
repository root:

```text
$review-trtmc-pr review this draft PR as a contributor self-review. Do not
publish comments or change the branch. Report architecture, correctness, and
validation findings first.
```

For an earlier pass before a draft PR exists, run:

```text
$review-trtmc-pr self-review my current branch and working tree against
upstream/main. Do not modify files.
```

Use `/skills` or type `$` to confirm that the skill is available. If it does
not appear, restart Codex from the repository root.

Using Codex is recommended, not required. A manual review or another review
tool is also acceptable. When the review is complete, select the single
**Contributor Self-Review** checkbox in the pull-request description.

Resolve blocking and high-severity findings before marking the pull request
ready. If a finding requires an architecture or policy decision, leave the pull
request in draft, record the question, and ask a maintainer. Rerun the affected
validation and self-review after material changes. Self-review does not replace
Community CI or maintainer review.

### 9. Run contributor-visible Community CI

Opening a pull request targeting `main` or pushing a new commit automatically
starts **Stable Community GPU CI**. Its four parallel CPU stages cover source
quality, documentation, ownership and impact, and C++/Python unit tests. Only
after they all pass does CI check the GPU test plan and allocate an isolated VM
for GPU-impacting changes. Each selected family runs in a fresh container; the
final result waits for confirmed VM cleanup.

Keep each PR focused on one family where possible. The automatic GPU plan admits
at most five directly changed or added families; larger changes must be split
before a VM is allocated. Shared changes additionally run the five shared smoke
families. Documentation-only changes require no GPU allocation. CI does not
silently drop selected families to fit its execution budget.

The CPU jobs run on fresh GitHub-hosted runners. GPU CI uses a disposable cloud
VM and a shared dependency base, with family-owned dependency layers prepared
only for selected families. Public Dockerfiles and dependency inputs allow local
reproduction without private registry access. Contributor test containers receive
no repository, Hub, Brev or registry credentials.

`Stable Community GPU CI` on the current PR head is the single automated premerge
gate. Historical `Stable Community CI` results from the CPU-only pipeline do not
satisfy this new gate. Existing PRs need a fresh Stable execution that publishes
the new context. `Community CPU / Required` alone is not completion of that gate. Native
checks such as DCO and PR Metadata remain visible contributor feedback and do
not replace the aggregate. See the [Community CI guide](.github/community-ci.md)
for failure classification, test selection and environment preparation.

### 10. Hand off for NVIDIA maintainer review

When `Stable Community GPU CI` succeeds on the current PR head, automated validation
is complete and the PR is ready for review by an NVIDIA employee maintainer.
Internal premerge is optional, not a merge requirement. A maintainer or
administrator may explicitly request it with the one-shot `run-internal-ci`
label or the Internal CI Bridge's manual dispatch, for example to diagnose a
large PR. Its separate `Optional Internal CI` result does not override a Stable
plan rejection; large changes still need to be split under the automatic policy.
A green result does not approve the change or merge it automatically.

Use the Actions summary and linked logs when validation fails. Preparation
failures are reported separately from failures after the family's test entry
point. The latter identify a family execution problem, not necessarily proof
that the PR introduced a new bug. Resource cleanup remains mandatory in both
cases. A pending result during cleanup is not a passing result.

Private Nightly and separately authorized diagnostics remain independent.
Private runner details, logs, artifacts and package coordinates must not be
copied into public issues or PRs. Dev qualification is maintainer-operated and
does not substitute for the required Stable result.

### 11. Respond to review and keep evidence current

Address review feedback on the same topic branch and sign off every new commit.
A new head requires a fresh automatic Stable result. Keep the PR current with
upstream when requested and preserve meaningful test criteria when diagnosing
failures. After an NVIDIA employee maintainer reviews and accepts the change,
maintainers merge it according to the repository ruleset. CI readiness and human
approval are separate requirements; GitHub permissions alone do not identify
someone's employment.

## Established development practices

The repository's development history consistently favors these practices:

- keep commits and pull requests single-purpose and easy to review;
- use an imperative Conventional Commit-style subject with a useful scope;
- keep model-owned behavior and its tests close to the model owner;
- pair behavior changes with focused regression coverage;
- record exact commands, revisions, artifacts, hardware, and untested paths;
- preserve copyright, license, attribution, and third-party provenance; and
- treat CI success as evidence for the exact tested revision, not for later
  commits or unexecuted configurations.

## Licensing and attribution

Contributions accepted into this project are licensed under the Apache License
2.0 unless explicitly stated otherwise.

Preserve all existing copyright, license, and attribution notices. If a
contribution incorporates or is derived from third-party material, identify its
source, version, and license in the pull request and include any notices required
for redistribution. Do not submit material unless you have the right to do so
under a license compatible with this project.

The DCO sign-off requirement is enforced prospectively for commits introduced
by pull requests and protected-branch updates after its adoption. Existing Git
history is not rewritten.

## Signing Your Work

- We require all contributors to sign off their commits. This certifies that
  the contribution is your original work, that you have the right to submit it
  under the same license, or that it uses a compatible license.
  - Contributions containing commits without a `Signed-off-by` line are not
    accepted.
- To sign off a commit, use the `--signoff` (or `-s`) option:

  ```bash
  git commit --signoff -m "docs(contributing): clarify external workflow"
  ```

  This appends a line like this to the commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```
- The full text of the [Developer Certificate of
  Origin](https://developercertificate.org/) follows:

  ```
    Developer Certificate of Origin
    Version 1.1
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.
    Developer's Certificate of Origin 1.1
    By making a contribution to this project, I certify that:
    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or
    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or
    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified it.
    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```
