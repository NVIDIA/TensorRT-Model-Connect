---
title: Family Runtime Selection
---

PR #1093 removed native runtime-strategy keys and their central registry. The
runtime now selects exactly one family DSO from the format-1 bundle header.

```json
{
  "format": 1,
  "family": "qwen",
  "task": "text_generation",
  "backend": "trt"
}
```

The loader opens `libtrtmc_model_qwen.so`, resolves its single
`trtmc_create_family` factory, and verifies that the returned Task matches
`text_generation`. Family-specific dispatch stays inside that DSO and is not a
second public selection axis.

| Concept | Current identity |
| --- | --- |
| Source/build/runtime owner | `family` |
| User-visible behavior | `task` and its abstract Task interface |
| Engine implementation | `backend` |
| Family-private pipeline variation | Family-owned bundle sections and code, not a registry key |

To add behavior, edit the owning `families/<family>/model.py`, `runtime/`, and
tests. To add another owner, add a complete family directory. The former
`runtime_strategy`, `task_strategy`, registrar macros, central DSO index, and
optimized-provider probe are not current extension points.
