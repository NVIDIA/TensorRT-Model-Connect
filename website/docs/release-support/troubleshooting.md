---
title: Troubleshooting
description: Route failures to model resolution, build, bundle, load, execution, or validation ownership.
---

| Boundary | Diagnostic | Owner to inspect |
| --- | --- | --- |
| Model resolution | Exact ID/revision, config, auth, local assets | Python family descriptor/plugin |
| TensorRT build | First parser/builder error, shapes, precision, workspace | Family builder and backend |
| Bundle | `trtmc inspect`, section list, checksum | Bundle writer/format |
| Runtime dispatch | Native strategy or optimized descriptor | Model DSO or provider adapter |
| Dependency load | Loader error, DSO search path, ABI cohort | Host environment/backend/provider |
| Task execution | Exact input, request config, first runtime error | Model pipeline/task implementation |
| Validation | Oracle, comparator, thresholds, skipped prerequisites | E2E manifest and harness |
| Performance | Timing boundary, warmup, variance, quality gate | Benchmark configuration |

Preserve the first error and the complete command. Retrying with several
unrelated flags can hide the original ownership boundary.

For a new installation, start with
[First-run Troubleshooting](../getting-started/troubleshooting.md). For a model
result, reproduce the exact model-owned manifest before opening a generalized
framework issue.
