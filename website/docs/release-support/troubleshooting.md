---
title: Troubleshooting
description: Route failures to checkpoint resolution, family build, bundle, loader, backend, Task execution, or validation ownership.
---

| Boundary | First diagnostic | Owner to inspect |
| --- | --- | --- |
| Checkpoint resolution | exact root metadata and zero/one/multiple family matches | `families/*/support.py` and build core resolver |
| Family build | first exception, task, precision, shapes, topology | selected `families/<family>/model.py` |
| Bundle | `trtmc inspect`, format/family/task/backend, bounded sections | `BundleWriter`, bundle reader, or family section producer |
| Runtime loading | explicit runtime root and exact backend/family DSO names | runtime loader and product installation |
| Backend | TensorRT version, engine deserialization, binding error | selected backend plus family engine contract |
| Task execution | exact input/config and first runtime error | selected family pipeline |
| Validation | selected testcase, oracle, thresholds, skipped prerequisites | `families/<family>/tests/` |
| Performance | timing scope, warmup, variance, output gate | `apps/benchmark/` configuration |

## Common boundaries

### No family claims the checkpoint

Inspect the root `config.json`, `model_index.json`, and relative snapshot
filenames. Resolution uses exact family-owned identity; it does not guess from a
repository name.

### More than one family claims the checkpoint

This is an ownership bug. Narrow the conflicting `support.py` declarations.
The resolver does not score matches or choose the first family.

### Runtime DSO is not found

`--runtime-root` must contain the matching core/runtime loader, requested
backend, and exact `libtrtmc_model_<family>.so` from the same build. The loader
does not search fallback directories.

### A bundle inspects but does not run

Inspection proves only bounded container parsing. Engine deserialization,
bindings, family section semantics, and Task execution happen later.

### A family E2E skips

Real E2E requires an explicit selector or `TRTMC_E2E=1`, the checkpoint,
family dependencies, compatible native targets, and required GPU topology. A
skip is not passing evidence.

Preserve the first error and complete command. When the failure is reproducible,
follow [Get Help](get-help.md) and sanitize all public evidence.
