---
title: Reference
description: Exact CLI, Python, C++, bundle, configuration, testing, and performance contracts.
---

Reference pages are for exact lookup, not progressive learning or task
instruction. Begin with [your first NLP inference](../getting-started/quick-start.md)
if you have not built a bundle, or use [User Guides](../user-guides/overview.md)
when you need a goal-oriented procedure.

TensorRT-Model-Connect exposes five public entry layers:

| API | Entry point | Best for |
| --- | --- | --- |
| Python builder | `tensorrt_model_connect.build()` and `trtmc build` | Building `.bundle` bundles from Hugging Face IDs or local model directories. |
| Python runtime wrapper | `tensorrt_model_connect.Pipeline` | Text and vision-language generation through the native `trtmc` executable from Python. |
| C++ runtime | `#include <trtmc/pipeline.h>` and `trtmc::load()` | Native applications that want task-specific inference results. |
| C-linkage subset | `trtmc_create_pipeline_ex()` and `trtmc_generate_batch()` | C++ shims and experimental FFI integration; the current header/handle is not yet a complete pure-C ownership API. |
| Local serving control plane | `trtmc serve` | Persistent native bundle workers behind local chat, audio, and Realtime transcription APIs. |

The command-line interface is a thin adapter over these APIs:

- `trtmc build` is implemented by `src/cli/main.cpp` delegating to `python/tensorrt_model_connect/build_cli.py`.
- Runtime subcommands such as `trtmc run` are implemented under `src/cli/`.
- `trtmc serve` delegates HTTP handling to the optional Python control plane;
  each registered bundle remains loaded in one or more native `_serve-worker`
  replicas.
- `tensorrt_model_connect.Pipeline` is a subprocess wrapper over `trtmc run`
  and `trtmc inspect`, not an in-process binding to `IPipeline`.

The core contract is always the same:

```text
Hugging Face model or local model directory
  -> trtmc build
  -> model.bundle
  -> trtmc::load() or trtmc run
  -> task-specific output
```

{/* Collaborative review anchor: batch 2. */}
