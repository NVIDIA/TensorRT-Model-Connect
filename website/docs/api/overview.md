---
title: Reference
description: Exact build, runtime, bundle, testing, and performance contracts.
---

Reference pages are for exact lookup. Begin with the
[Quick Start](../getting-started/quick-start.md) if you have not built a bundle,
or use the [User Guides](../user-guides/overview.md) for goal-oriented
procedures.

TensorRT-Model-Connect separates build tools from runtime entry points:

| API | Entry point | Best for |
| --- | --- | --- |
| Python build API | `python -m tensorrt_model_connect build` and `tensorrt_model_connect.build()` | Resolving a supported checkpoint and building a `.bundle`. |
| Native CLI | `trtmc inspect` and task commands such as `trtmc run` | Inspecting a bundle or invoking one abstract Task interface. |
| C Task SDK | `trtmc_get_api()` from `trtmc/trtmc.h` | Typed native calls through the public C boundary for migrated families. |
| Header-only C++ SDK | `#include <trtmc/trtmc.hpp>` and `trtmc::Model::load()` | User-compiled convenience wrappers and RAII over that same C boundary. |

The new Task SDK is experimental pending its first stable release. Runtime and
family implementations upgrade together; the older internal C++ loader API is
not a stable user ABI. See [C and C++ Task SDK](cpp-api.md).

The build and runtime entry points are intentionally separate. The Python
builder resolves exactly one `families/<family>/support.py`, imports only that
family's `model.py`, and writes a bundle. The native loader reads the bundle's
`family`, `task`, and `backend`, then loads exactly one family DSO and one
backend DSO from the selected runtime root.

```text
Hugging Face model ID or local snapshot
  -> python -m tensorrt_model_connect build
  -> model.bundle
  -> trtmc::Model::load() or trtmc TASK
  -> task-specific output
```

The installed `trtmc` Python entrypoint uses the Python builder only for `build`;
other commands replace that process with the packaged native executable. There
is no resident Python inference wrapper. The SDK defaults to its installed
library directory, with an explicit runtime-root override when needed; it does
not retry arbitrary backends or family implementations after a failed call.
