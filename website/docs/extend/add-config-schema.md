---
title: Configuration Boundaries
---

The pre-#1093 schema registry and generic `--config` / `--set` surface were
removed. Current build inputs are explicit `BuildRequest` fields, runtime
inputs are typed Task request/config fields, and family-only build/runtime data
belongs to family-owned code and bundle sections.

## Choose the narrow owner

| Need | Current owner |
| --- | --- |
| Stable model-agnostic build input | `BuildRequest` and the Python build CLI under `core/builder/tensorrt_model_connect/`. |
| Stable user-visible runtime input | The relevant interface/config in `core/runtime/include/trtmc/task.h` and its CLI adapter in `apps/cli/`. |
| Backend-only load setting | The loader/Engine contract, only when independent of model semantics. |
| Family-specific build or runtime policy | `families/<family>/model.py`, family bundle sections, and `families/<family>/runtime/`. |

Do not create a shared schema, options bag, environment-variable convention,
or generic dictionary for a family-only knob. Prefer deriving family policy
from the exact checkpoint, task, and existing request fields.

## Adding a shared field

Before adding a field, demonstrate that it is stable and model-agnostic. Then:

1. add the typed field at the narrow public boundary;
2. validate its range before control transfers to a family or backend;
3. make every family either implement or explicitly reject non-default values;
4. add parser/API tests for invalid and boundary inputs;
5. add at least one end-to-end family proof that the value changes observable
   behavior as documented.

Do not silently ignore an unsupported value or retry another family/backend.

## Examples in the current API

- Build dimensions, precision, parallel sizes, family-owned quantization, and
  direct dynamic-KV selection are typed `BuildRequest` fields.
- Text sampling lives in `TextGenerationConfig` because it is part of the
  public text-generation task.
- Runtime-sized KV capacity is a narrow loader input consumed only by
  compatible families.
- TensorRT-RTX cache and CUDA graph settings are backend load options and are
  rejected for standard TensorRT bundles.

The restored
[config-registry status document](../context/config-registry-status.md) is a
historical record of the retired implementation, not current API guidance.
