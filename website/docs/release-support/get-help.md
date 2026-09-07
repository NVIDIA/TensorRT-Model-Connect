---
title: Get Help and File an Issue
description: Choose the right support route and provide enough evidence for maintainers to reproduce and triage a request.
---

Use the route that matches what you need. Public issues are visible to
everyone, so remove credentials, private URLs, proprietary logs, and
license-restricted model artifacts before submitting anything.

:::danger Security vulnerabilities

Do not open a public issue for a suspected vulnerability. Follow the
[security policy](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/SECURITY.md)
to report it privately to NVIDIA PSIRT.

:::

## Choose an issue type

| Need | Route |
| --- | --- |
| Help using or understanding TensorRT-Model-Connect | [Ask a question](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new?template=question.yml) |
| Reproducible behavior that differs from the documented contract | [Report a bug](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new?template=bug_report.yml) |
| A new model, capability, or improvement | [Request a feature](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new?template=feature_request.yml) |
| Incorrect, unclear, or missing documentation | [Request a documentation change](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new?template=documentation_request.yml) |
| Unsure which route applies | [Open the issue chooser](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new/choose) |

Questions receive best-effort help from maintainers and the community. The
question form does not create a support SLA or establish that an unqualified
model, configuration, or target is supported.

## Before filing

1. Check the [supported-model inventory](../models-recipes/overview.md) for the
   exact checkpoint, profile, precision, runtime path, and evidence level.
2. Check [Known Issues](known-issues.md), this section's
   [Troubleshooting](troubleshooting.md), and
   [First-run Troubleshooting](../getting-started/troubleshooting.md).
3. Search [open and closed issues](https://github.com/NVIDIA/TensorRT-Model-Connect/issues?q=is%3Aissue)
   for the same model, error, or requested behavior. Add evidence to an
   existing issue instead of opening a duplicate.
4. Reduce failures to the smallest command, configuration, and input that still
   reproduces the behavior. Preserve the first error rather than retrying with
   unrelated flags.

## Information maintainers need

Include the evidence that applies to your request:

- the TensorRT-Model-Connect release, tag, or full commit SHA;
- installation method and exact container image or package version;
- operating system, GPU model, driver, CUDA, and TensorRT versions;
- exact Hugging Face model ID and revision, or a description of the local
  checkpoint without uploading restricted artifacts;
- exact commands and relevant configuration, with secrets redacted;
- expected behavior, observed behavior, and the first relevant error or log;
- a minimal reproducer and whether the problem also occurs with the documented
  model-owned manifest; and
- for performance reports, the timing boundary, warmup, sample count, workload,
  and output-quality gate.

A command being accepted by a parser, a family being registered, and a model
being qualified on an exact hardware/software tuple are different evidence
levels. State what you verified and what you did not run.

## Contributing a fix

If you plan to submit a change, read
[CONTRIBUTING.md](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/CONTRIBUTING.md)
and the [Developer Guide](../developer-guide/overview.md). Link the issue from
the pull request so reviewers can connect the proposed implementation to the
reported behavior and exit criteria.

{/* Collaborative review anchor: batch 2. */}
