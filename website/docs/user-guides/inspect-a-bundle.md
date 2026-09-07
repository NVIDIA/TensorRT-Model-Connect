---
title: Inspect a Bundle
description: Identify bundle ownership and required runtime DSOs before loading it.
---

```bash
trtmc inspect model.bundle
```

Inspection returns JSON containing:

| Field | Meaning |
| --- | --- |
| `format` | Shared container version; the current reader accepts format `1`. |
| `family` | Exact DSO owner, for example `qwen` -> `libtrtmc_model_qwen.so`. |
| `task` | Abstract Task interface the factory must return. |
| `backend` | Engine DSO identity: `trt` or `trt_rtx`. |
| `sections` | Family-owned names with validated offsets and lengths. |

The inspector does not decode family sections and has no `--list-engines`
mode. Plan names vary by family; inspect the selected family's `model.py` and
runtime factory when debugging their semantics.

To execute, place matching `libtrtmc_core.so`, `libtrtmc_runtime.so`,
`libtrtmc_backend_<backend>.so`, and `libtrtmc_model_<family>.so` in one
explicit runtime root. There is no registry, plugin directory, or fallback
search path.

For a guided lab, use
[Inspect Bundles](../tutorials/beginner/inspect-bundles.md).
