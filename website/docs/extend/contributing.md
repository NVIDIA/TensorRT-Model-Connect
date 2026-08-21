---
title: Contributor Quickstart
description: Prepare, validate, and submit a focused TensorRT-Model-Connect contribution.
---

Contributions use the same ownership and evidence rules described throughout
this site. Start with a small, reviewable change and prove the behavior at the
lowest meaningful layer before requesting broader CI.

The repository-root
[CONTRIBUTING.md](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/CONTRIBUTING.md)
is authoritative for the external contribution workflow, licensing, and
Developer Certificate of Origin requirements. This page adds project-specific
development and validation detail.

## 1. Fork the repository and prepare a clean branch

External development happens in a personal fork. Use GitHub's **Fork** button
on the canonical repository, then clone your fork, add the NVIDIA repository as
`upstream`, and start from its current `main`:

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect
git remote add upstream https://github.com/NVIDIA/TensorRT-Model-Connect.git
git fetch upstream
git switch -c docs/improve-getting-started upstream/main
```

In this layout, `origin` is your writable fork and `upstream` is the canonical
repository. Do not develop on either repository's `main`, and preserve
unrelated local work.

For source changes, use the development environment described in
[System Requirements](../getting-started/environment-and-repro.md).

Install the local quality hooks once in the clone:

```bash
python3 -m pip install --requirement requirements/community-ci.txt
pre-commit install --install-hooks
```

Commit-time hooks trim trailing whitespace, ensure one final newline, validate
YAML, check Ruff, and verify clang-format. Some commit-time hooks modify files;
review and stage those fixes before committing again. Pre-commit manages the
Ruff and clang-format environments on Linux, macOS, and Windows instead of
depending on host-installed binaries.

The local hook intentionally stays lightweight and does not build the CLI or
run the complete CPU suite. After pushing, comment `/run-ci` on the pull request
to run source quality, ownership analysis, and the selected source-only C++ and
Python units on GitHub-hosted public CPU runners. The protected suite retains
the filesystem-specific cache-reflink contract that public runners cannot
portably execute.

## 2. Find the owner before editing

Choose the narrowest current owner:

| Change | Start here |
| --- | --- |
| Native model support | [Add a Model Family](add-model-family.md) |
| Exact delegated runtime for an existing family | [Add an Optimized Runtime Implementation](add-optimized-runtime.md) |
| Native runtime behavior | [Add a Runtime Strategy](add-runtime-strategy.md) |
| User-facing configuration | [Add a Config Schema](add-config-schema.md) |
| Model validation | [Validate a Model Contribution](model-validation.md) |
| External family-owned kernel | [Bring Your Own Kernel](../tutorials/advanced/bring-your-own-kernel.md) |
| Public API or shared infrastructure | [Developer Guide](../developer-guide/overview.md), then the owning architecture or API reference |

Similar model behavior is not automatically shared infrastructure. Family
semantics stay with the owning family unless at least two real owners require
the same model-independent contract.

## 3. Preserve legal metadata

- Keep existing copyright, license, and attribution notices.
- Add the repository SPDX header to new source files.
- Identify third-party source, version, license, and required notices in the
  pull request.
- Sign every commit:

```bash
git commit --signoff
```

The sign-off certifies the contribution under the Developer Certificate of
Origin. Unsigned commits are not accepted.

## 4. Validate the smallest meaningful scope

Always start with repository consistency:

```bash
PYTHONPATH=python:. python3 tools/model_ci.py validate
PYTHONPATH=python:. python3 tools/test_impact.py --validate
git diff --check
```

Then run focused tests for the owner you changed. Model changes also need the
declared E2E contract and its required model, runtime, GPU, and comparison
artifacts. Documentation changes need strict references and a production site
build:

```bash
python3 tools/check_doc_file_references.py --strict website/docs
npm --prefix website ci
npm --prefix website run build
```

Do not weaken an acceptance threshold to make a change pass. If a test is
wrong, explain the evidence and request maintainer review.

## 5. Push to your fork and open the pull request

Sync with current upstream, push the short-lived branch to your fork, and open a
pull request targeting `NVIDIA/TensorRT-Model-Connect:main`:

```bash
git fetch upstream
git rebase upstream/main
git push --set-upstream origin docs/improve-getting-started
```

The pull request should record:

- exact scope and non-goals;
- exact base and tested head revisions;
- commands actually executed;
- model, artifact, hardware, and environment for GPU claims;
- remaining risks and paths not executed.

Compilation, source tests, model parity, target-hardware execution,
performance, and release qualification are different evidence tiers.

## 6. Read public CPU validation

The pull-request author starts contributor-visible, GitHub-hosted `Community
CPU` validation by adding this exact comment to the pull request:

```text
/run-ci
```

The trusted request workflow runs from `main`, captures the exact PR merge
revision, and dispatches source quality, ownership and impact, and source-only
C++ and Python units. A maintainer or admin may submit the same comment on the
author's behalf.

The test jobs have read-only repository permission and no access to private
runners, secrets, or GPUs. Their sanitized checks and detailed public logs are
published on the exact merge revision. Fix any failures and wait for
`Community CPU / Required` to pass. If you push another commit, comment
`/run-ci` again after the new head is ready.

## 7. Coordinate protected repository CI

The repository premerge workflow is one-shot and label-driven. Pushing a branch
or creating a pull request alone does not start the gate. After local checks
pass and the pull request is ready, add this comment:

```text
@yifeif-nv This PR is ready for CI. Please trigger CI for the current head.
```

After public CPU validation passes, the maintainer verifies the PR's
`headRefOid` and applies `run-internal-ci`. The trusted bridge consumes that
label, verifies the current exact-merge public CPU result, captures the
immutable PR head SHA, and dispatches private premerge validation for that exact
revision. The public result is contributor feedback, not an authorization
token; `run-internal-ci` remains the protected-resource security boundary.

Wait for the `trtmc/premerge/required` status on the same head SHA to complete
successfully. If the head changes intentionally, finish the update and local
validation before mentioning `@yifeif-nv` once to request a new run. Only a
maintainer with repository `maintain` or `admin` permission can authorize the
trigger. Private CI repository details, runner information, logs, artifacts,
and URLs are not public documentation.

{/* Collaborative review anchor: batch 2. */}
