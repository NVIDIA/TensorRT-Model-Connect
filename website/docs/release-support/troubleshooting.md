---
title: Troubleshooting
description: Route failures to model resolution, build, bundle, load, execution, or validation ownership.
---

| Boundary | Diagnostic | Owner to inspect |
| --- | --- | --- |
| Model resolution | Exact ID/revision, config, auth, local assets | Family `support.py` and builder |
| TensorRT build | First parser/builder error, shapes, precision, workspace | Family builder and backend |
| Bundle | `trtmc inspect`, section list, checksum | Bundle writer/format |
| Runtime loading | Required family task and runtime-root contents | Family DSO and shared loader |
| Dependency load | Loader error, exact runtime root, ABI cohort | Host environment, core, backend, and family DSOs |
| Task execution | Exact input, request config, first runtime error | Model pipeline/task implementation |
| Validation | Oracle, comparator, thresholds, skipped prerequisites | E2E manifest and harness |
| Performance | Timing boundary, warmup, variance, quality gate | Benchmark configuration |

Preserve the first error and the complete command. Retrying with several
unrelated flags can hide the original ownership boundary.

For a new installation, start with
[First-run Troubleshooting](../getting-started/troubleshooting.md). For a model
result, reproduce the exact model-owned manifest before opening a generalized
framework issue.

When the failure is reproducible, follow [Get Help and File an Issue](get-help.md)
to select the right issue form and include the exact environment, model
revision, commands, expected behavior, observed behavior, and first error.

{/* Collaborative review anchor: batch 2. */}
