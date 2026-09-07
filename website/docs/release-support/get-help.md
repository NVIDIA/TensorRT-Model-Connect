---
title: Get Help and File an Issue
description: Choose the right public route and provide reproducible, sanitized evidence.
---

Public issues are visible to everyone. Remove credentials, private URLs,
non-public logs, personal filesystem paths, and license-restricted model
artifacts before submitting anything.

:::danger Security vulnerabilities

Do not open a public issue for a suspected vulnerability. Follow the
[security policy](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/SECURITY.md)
to report it privately to NVIDIA PSIRT.

:::

## Choose an issue type

| Need | Route |
| --- | --- |
| Help using the project | [Ask a question](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new?template=question.yml) |
| Reproducible behavior that differs from the documented contract | [Report a bug](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new?template=bug_report.yml) |
| A model, capability, or improvement | [Request a feature](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new?template=feature_request.yml) |
| Incorrect or missing documentation | [Request a documentation change](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new?template=documentation_request.yml) |
| Unsure | [Open the issue chooser](https://github.com/NVIDIA/TensorRT-Model-Connect/issues/new/choose) |

Community help is best effort and does not qualify an untested checkpoint,
configuration, or target.

## Before filing

1. Check the [Supported Models](../models-recipes/overview.md) inventory for the
   exact family, checkpoint, task, precision, and topology.
2. Check [Known Issues](known-issues.md) and
   [Troubleshooting](troubleshooting.md).
3. Search [open and closed issues](https://github.com/NVIDIA/TensorRT-Model-Connect/issues?q=is%3Aissue)
   and add evidence to an existing report when appropriate.
4. Reduce the problem to the smallest exact command and input that still
   reproduces the first failure.

## Include useful evidence

- full TensorRT-Model-Connect commit SHA or release tag;
- installation method and package/container version;
- OS, host architecture, GPU, driver, CUDA, and TensorRT versions;
- exact model ID and revision, or a non-sensitive description of a prepared
  local checkpoint;
- owning family manifest and testcase when one applies;
- exact sanitized commands, expected behavior, observed behavior, and first
  relevant error;
- bundle header from `trtmc inspect`, without uploading restricted sections;
- whether source checks, build, runtime execution, reference comparison, and
  performance measurement were run; and
- for performance, the timing boundary, warmup, sample count, workload, and
  output-quality gate.

If you plan to contribute a fix, read
[CONTRIBUTING.md](https://github.com/NVIDIA/TensorRT-Model-Connect/blob/main/CONTRIBUTING.md)
and the [Developer Guide](../developer-guide/overview.md).
