---
title: System Overview
description: The build, bundle, and runtime boundaries of TensorRT-Model-Connect.
---

import Diagram from '@site/src/components/Diagram';

TensorRT-Model-Connect turns a Hugging Face checkpoint or local model directory
into a deployable `.trtfb` bundle, then loads that bundle behind task-oriented
C++ APIs.

The most important boundary is the bundle:

- Python owns source-model diversity and artifact construction.
- `.trtfb` carries the contract from build time to run time.
- C++ owns bundle loading, task dispatch, and request execution.

Read the [Glossary](../getting-started/glossary.md) first if terms such as
checkpoint, engine, bundle, DSO, prefill, or KV cache are new to you.

## System block diagram

<Diagram
  src="/img/diagrams/trtmc-system-map.svg"
  alt="System map from a Hugging Face checkpoint through build routing and a native or optimized bundle to the C++ runtime and typed task result"
  caption="The bundle is the deployment boundary: native artifacts resolve installed model and backend DSOs, while optimized artifacts carry their exact implementation DSO."
/>

The diagram shows two artifact shapes, not two user-selected public APIs.
`trtmc build` and the Python `build()` function resolve the model family first.
A family-owned native default may claim the request immediately. Otherwise, an
exact model/revision/target/options profile may select an optimized adapter; no
qualified profile continues to the native builder.

## The two bundle paths

| Concern | Native bundle | Optimized-runtime bundle |
| --- | --- | --- |
| Build owner | Python `FamilyPlugin` and family-local TensorRT builders | Family-local implementation/profile and isolated adapter |
| Primary identity | `runtime_strategy` | `optimized_runtime.json` implementation and profile |
| Runtime implementation | Installed `libtrtmc_model_<owner>.so` | Exact embedded `libtrtmc_impl_*.so` |
| TensorRT execution boundary | Installed backend DSO implementing `IBackend` | Delegated implementation behind its private factory |
| Fallback behavior | Used when no optimized profile claims the request | Descriptor presence claims this path; load failures do not fall back to native |
| Evidence | Native E2E manifest and model-owned tests | Exact profile qualification plus adapter, bundle, and host evidence |

Both shapes still depend on compatible host facilities such as the NVIDIA
driver, CUDA, TensorRT, the dynamic loader, and system libraries. A bundle is a
deployment artifact, not a complete operating-system or GPU-runtime image.

## Design rules

### Model knowledge stays model-owned

Checkpoint mapping, graph semantics, runtime state, tokenization,
pre/postprocessing, and task behavior stay with the owning model family.
Shared code owns stable contracts and genuinely model-independent mechanics.

### Dispatch uses artifact identity

The runtime does not choose a pipeline by searching a Hugging Face model name.
A native bundle dispatches through its `runtime_strategy`; an optimized bundle
dispatches through its integrity-bound implementation/profile descriptor.

### Public APIs are task-oriented

Applications load a bundle and call methods such as `generate()`,
`transcribe()`, `generate_image()`, `embed()`, or `solve()`. Unsupported methods
fail explicitly for that concrete pipeline.

### TensorRT ABI details stay behind a boundary

Native pipelines use `IBackend` and `ITrtModule`; TensorRT headers and
ABI-sensitive calls live behind backend DSOs. Optimized implementations own
their delegated execution internally.

### Buildability is not qualification

Source, unit tests, model E2E evidence, exact-profile qualification, and
performance evidence prove different things. Do not infer model support or
parity from the existence of a family package alone.

## Where each concern is explained

| Question | Canonical page |
| --- | --- |
| Which source unit owns a behavior? | [Units and Ownership](units-and-ownership.md) |
| How does a checkpoint become a bundle? | [Build Pipeline](build-pipeline.md) |
| What is physically stored in `.trtfb`? | [Bundle Format](bundle-format.md) |
| How does a bundle become an `IPipeline` and serve requests? | [Runtime Lifecycle](runtime-lifecycle.md) |
| How are native targets, DSOs, and wheels assembled? | [Build System](build-system.md) |
| Which evidence layer proves which contract? | [Validation Design](validation-design.md) |

## Source-of-truth entry points

| Boundary | Primary implementation |
| --- | --- |
| Build CLI | `python/tensorrt_model_connect/build_cli.py` |
| Public Python build API | `python/tensorrt_model_connect/engine_builder.py` |
| Family discovery | `python/tensorrt_model_connect/families/__init__.py` |
| Optimized selection and packaging | `python/tensorrt_model_connect/runtime_provider/` |
| Bundle writer and reader | `python/tensorrt_model_connect/bundle_writer.py`, `src/bundle/` |
| Public task API | `include/trtmc/pipeline.h` |
| Pipeline creation | `src/runtime/registry/pipeline_factory.cpp` |
| Native plugin loading | `src/runtime/registry/pipeline_plugin_loader.cpp` |
| Optimized implementation loading | `src/runtime/providers/optimized_runtime_host.cpp` |
