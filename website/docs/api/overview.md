---
title: Reference
description: Exact build, runtime, bundle, testing, and performance contracts.
---

Reference pages are for exact lookup. Begin with the
[Quick Start](../getting-started/quick-start.md) if you have not built a bundle,
or use the [User Guides](../user-guides/overview.md) for goal-oriented
procedures.

TensorRT-Model-Connect exposes three public entry layers:

| API | Entry point | Best for |
| --- | --- | --- |
| Python build API | `python -m tensorrt_model_connect build` and `tensorrt_model_connect.build()` | Resolving a supported checkpoint and building a `.bundle`. |
| Native CLI | `trtmc inspect` and task commands such as `trtmc run` | Inspecting a bundle or invoking one abstract Task interface. |
| C++ Task API | `#include <trtmc/runtime/family_loader.h>` and `trtmc::load_task()` | Native applications that need task-specific results. |

The build and runtime entry points are intentionally separate. The Python
builder resolves exactly one `families/<family>/support.py`, imports only that
family's `model.py`, and writes a bundle. The native loader reads the bundle's
`family`, `task`, and `backend`, then loads exactly one family DSO and one
backend DSO from one selected plugin root.

```text
Hugging Face model ID or local snapshot
  -> python -m tensorrt_model_connect build
  -> model.bundle
  -> trtmc::load_task() with an explicit root, or trtmc TASK with CLI discovery
  -> task-specific output
```

There is no Python runtime wrapper, central model registry, runtime-strategy
switch, sibling-family probe, or load-time fallback. CLI discovery selects one
root before the Runtime Loader performs an exact load.
