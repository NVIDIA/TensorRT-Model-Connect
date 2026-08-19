---
title: Units and Ownership
description: Source-level responsibilities and the authoritative contracts between units.
---

import Diagram from '@site/src/components/Diagram';

Architecture explains the system flow. Unit ownership answers a different
question: **where does a behavior belong?**

The repository follows one rule throughout build, runtime, and tests:
model-specific knowledge stays under one model owner root. Shared units own
public contracts, bundle transport, loading, configuration mechanics, device
abstractions, and code that is genuinely model-independent.

## Ownership block diagram

<Diagram
  src="/img/diagrams/architecture/build-ownership.svg"
  alt="Build-time ownership from public entry points into one model folder containing its descriptor, Python build, native runtime, and tests"
  caption="Every model-specific build, runtime, and evidence surface stays in one owner folder; the bundle is the deployment handoff."
/>

<Diagram
  src="/img/diagrams/architecture/runtime-ownership.svg"
  alt="Runtime ownership from the trtmc CLI, Python Pipeline wrapper, C++ API, and limited C-linkage surface through shared loaders to an optimized implementation or native model and backend DSOs"
  caption="All public runtime surfaces enter shared factory and loader units; optimized artifacts own a private implementation, while native model and backend DSOs divide task behavior from engine execution."
/>

The backend arrow is an interface boundary. A model DSO does not select and
link a backend directly; `PipelineFactory` loads a compatible backend and
passes its `IBackend*` through `PipelineContext`.

## Public authorities

Use these files as the contract authority. Implementation pages and examples
must agree with them.

| Contract | Authority |
| --- | --- |
| Pipeline operations, request configs, and typed results | `include/trtmc/pipeline.h` |
| Bundle inspection types | `include/trtmc/bundle.h` |
| Pipeline factory and native pool construction | `include/trtmc/runtime/pipeline_factory.h` |
| Native plugin construction context | `include/trtmc/runtime/pipeline_plugin.h` |
| Native strategy registry | `include/trtmc/runtime/pipeline_registry.h` |
| Model DSO loading | `include/trtmc/runtime/pipeline_plugin_loader.h` |
| Backend and engine-execution abstraction | `include/trtmc/runtime/trt_backend.h`, `include/trtmc/runtime/trt_module.h` |
| Runtime configuration | `include/trtmc/config/` |
| Model descriptor and build entry | `python/tensorrt_model_connect/models/<owner>/MODEL.toml` and `model.py` |
| Python bundle serialization | `python/tensorrt_model_connect/bundle_writer.py` |

The declarations with C linkage at the bottom of `pipeline.h` are a limited
C-linkage C++ subset. They expose C++ types and do not provide a complete
opaque-handle ownership API, so they are not a stable pure-C ABI.

## Major unit groups

| Unit group | Main path | Owns |
| --- | --- | --- |
| Public C++ API | `include/trtmc/` | Task methods, typed results, load options, bundle inspection |
| CLI | `src/cli/` | Argument parsing, file/media adaptation, and calls into public APIs |
| Python package | `python/tensorrt_model_connect/` | Public build API, CLI bridge, model resolution, and bundle writing |
| Model owners | `python/tensorrt_model_connect/models/<owner>/` | Descriptor, checkpoint/config semantics, Python builder, native runtime, tests, manifests, assets, and optional tools |
| Bundle implementation | `src/bundle/` | Safe header parsing and section access |
| Runtime registry | `src/runtime/registry/` | Native strategy resolution, DSO loading, plugin lookup, and pipeline creation |
| Optimized host | `src/runtime/providers/` | Descriptor validation, artifact materialization, and private factory loading |
| Runtime core | `src/runtime/core/` | Device tensors, streams, pooling, distributed setup, and reusable execution mechanics |
| Runtime domains | `src/runtime/domains/` | Small assumption-free helpers shared by real model owners |
| Backend | `src/runtime/backend/` | TensorRT ABI-sensitive engine deserialization and execution |
| Shared tests and evidence | `tests/` | Cross-model builder, runtime, tooling, and harness contracts |
| Model evidence | `python/tensorrt_model_connect/models/<owner>/tests/` | Owner-local Python, C++, E2E, manifest, asset, and qualification contracts |

## One descriptor, one owner

One descriptor owns every model-specific surface:

| Descriptor | Responsibility |
| --- | --- |
| `python/tensorrt_model_connect/models/<owner>/MODEL.toml` | Builder discovery, runtime plugins and strategies, config schemas, focused C++ tests, E2E manifests, task defaults, and validation ownership |

The owner directory and descriptor `id` are identical. There is no runtime-ID
alias or second model descriptor. `runtime_strategy` selects one native
implementation owned by that folder; `task_strategy` selects the shared
runner/comparator contract and does not select a DSO.

An optimized implementation uses a different ownership shape:

```text
python/tensorrt_model_connect/models/<family>/<implementation>/IMPLEMENTATION.toml
python/tensorrt_model_connect/models/<family>/<implementation>/profiles/*.toml
python/tensorrt_model_connect/models/<family>/tests/<implementation>/test_adapter.py
python/tensorrt_model_connect/models/<family>/tests/<implementation>/test_runtime_contract.py
```

It produces an embedded implementation DSO and does not need a synthetic native
strategy or runtime model descriptor. Source-side tests prove selection,
packaging, and fail-closed runtime contracts. The profile's semantic-source
digest and separately retained external target evidence carry different
meanings; neither should be presented as the other.

## Boundaries that prevent accidental coupling

| Boundary | Reason |
| --- | --- |
| Build adapter versus runtime constructor | Build code understands source artifacts; runtime code understands deployment artifacts and requests. |
| `IPipelinePlugin` versus `IPipeline` | The plugin validates and constructs once; the pipeline owns request-time behavior. |
| Model DSO versus backend DSO | The model owns semantics; the backend owns TensorRT ABI and engine execution. |
| `runtime_strategy` versus `task_strategy` | One dispatches a native implementation; the other groups an E2E user contract. |
| Schema config versus one-off flags | Schemas provide types, defaults, validation, provenance, and shared CLI/API surfaces. |
| Unit evidence versus model qualification | Local tests isolate behavior; exact checkpoint/hardware evidence proves integration or performance. |

## Reading a change

When tracing an unfamiliar change, follow its owner before following similar
code elsewhere:

1. Find the single model descriptor that declares the owner.
2. Follow the exact strategy, implementation/profile, or task identity named by
   that descriptor.
3. Read the owner's `model.py`, runtime plugin, or optimized adapter before
   reading shared infrastructure.
4. Check which test or qualification artifact proves the stated behavior.
5. Escalate to a shared abstraction only when multiple independent owners need
   the same assumption-free contract.

For implementation recipes, continue to [Extend the Project](../extend/overview.md).

{/* Collaborative review anchor: batch 2. */}
