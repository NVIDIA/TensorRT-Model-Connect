---
title: API Manual
---

TensorRT-Model-Connect exposes three public API layers.

| API | Entry point | Best for |
| --- | --- | --- |
| Python builder | `tensorrt_model_connect.build()` and `trtmc build` | Building `.trtfb` bundles from HuggingFace IDs or local model directories. |
| C++ runtime | `#include <trtmc/pipeline.h>` and `trtmc::load()` | Native applications that want task-specific inference results. |
| C ABI | `trtmc_create_pipeline_ex()` | FFI callers and existing C-style integrations. |

The command-line interface is a thin adapter over these APIs:

- `trtmc build` is implemented by `src/cli/main.cpp` delegating to `python/tensorrt_model_connect/build_cli.py`.
- Runtime subcommands such as `trtmc run` are implemented under `src/cli/`.

The core contract is always the same:

```text
HuggingFace model or local model directory
  -> trtmc build
  -> model.trtfb
  -> trtmc::load() or trtmc run
  -> task-specific output
```
