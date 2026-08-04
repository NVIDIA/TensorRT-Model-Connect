---
title: Contributor Quickstart
description: Prepare, validate, and submit a focused TensorRT-Model-Connect contribution.
---

Contributions use the same ownership and evidence rules described throughout
this site. Start with a small, reviewable change and prove the behavior at the
lowest meaningful layer before requesting broader CI.

The repository-root
[CONTRIBUTING.md](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/CONTRIBUTING.md)
is authoritative for licensing and Developer Certificate of Origin
requirements. This page adds the project-specific development and pull-request
workflow.

## 1. Prepare a clean branch

Use the canonical GitHub repository and start from its current `main`:

```bash
git clone https://github.com/NVIDIA/TensorRT-Model-Connect.git
cd TensorRT-Model-Connect
git switch -c docs/improve-getting-started origin/main
```

If your established clone names the canonical remote `github`, substitute
`github/main`. Do not push directly to `main`, and preserve unrelated local
work.

For source changes, use the development environment described in
[Prerequisites and Environment](../getting-started/environment-and-repro.md).

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
| Public API or shared infrastructure | [Architecture & Design](../architecture/overview.md), then the owning API reference |

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

## 5. Open the pull request

Push the short-lived branch and open a pull request targeting `main`. The pull
request should record:

- exact scope and non-goals;
- exact base and tested head revisions;
- commands actually executed;
- model, artifact, hardware, and environment for GPU claims;
- remaining risks and paths not executed.

Compilation, source tests, model parity, target-hardware execution,
performance, and release qualification are different evidence tiers.

## 6. Coordinate repository CI

The repository premerge workflow is one-shot and label-driven. After verifying
the PR's `headRefOid`, an authorized collaborator applies `run-internal-ci`.
The trusted bridge consumes that label, captures the immutable PR head SHA, and
dispatches private premerge validation for that exact revision. Pushing a
branch or creating a PR alone does not start the gate.

Wait for the `trtmc/premerge/required` status on the same head SHA to complete
successfully. If the head changes intentionally, verify the new SHA before
requesting one new run. Fork contributors must coordinate the label with a
maintainer who has repository `maintain` or `admin` permission. Private CI
repository details,
runner information, logs, artifacts, and URLs are not public documentation.

{/* Collaborative review anchor. */}
